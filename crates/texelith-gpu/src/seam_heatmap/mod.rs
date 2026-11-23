//! Read-only seam inspection of actual linear material textures (ADR 0005).
//! No graph variant is required: callers inspect the selected output lease.
use crate::{
    GpuError, GpuExecutor, NodeUniform, Pass, PreparedPass,
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
};
use std::sync::{Arc, Mutex};
use texelith_core::Extent;

/// Absolute forward-gradient RMS in four-pixel bands and its finite ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeamMetrics {
    pub boundary_rms: f32,
    pub interior_rms: f32,
    pub ratio: f32,
}

/// Reusable heatmap/reduction bindings. All internal textures, uniforms, upload
/// staging and scalar readback are charged to the caller's shared pool.
/// Input and output payloads remain the caller's responsibility.
///
/// Heatmap values are unnormalized forward-gradient magnitudes. RGB distance is
/// RMS over linear RGB, excluding alpha. Singleton axes contribute no samples.
/// Each mask uses four pixels per side (shrinking symmetrically for small grids);
/// the interior mask is centered and has the same number of directional samples.
/// The ratio denominator is floored at 1e-8. Extreme scalar differences saturate
/// at 1e18 before squaring so finite imported heights retain finite diagnostics.
pub struct PreparedSeamHeatmap {
    passes: Vec<PreparedPass>,
    uniforms: [NodeUniform; 3],
    pool: Arc<Mutex<ResourcePool>>,
    allocations: Vec<AllocationId>,
    metrics: wgpu::Texture,
    readback: wgpu::Buffer,
    device: wgpu::Device,
    queue: Arc<wgpu::Queue>,
    submitted: Option<FenceId>,
}

fn extent(width: u32, height: u32) -> Extent {
    Extent { width, height }
}

impl PreparedSeamHeatmap {
    pub fn new(
        executor: &GpuExecutor,
        input: &wgpu::Texture,
        heatmap: &wgpu::Texture,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        let size = texture_extent(input)?;
        if texture_extent(heatmap)? != size
            || input == heatmap
            || !matches!(
                input.format(),
                wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba16Float
            )
            || heatmap.format() != wgpu::TextureFormat::R32Float
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !heatmap
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError(
                "invalid seam diagnostic textures or extents".into(),
            ));
        }
        let groups = [size.width.div_ceil(8), size.height.div_ceil(8), 1];
        let partial_extent = extent(groups[0] * 2, groups[1]);
        let reduction_groups = (groups[0] * groups[1]).div_ceil(4096);
        let middle_extent = extent(reduction_groups * 2, 1);
        let metric_extent = extent(3, 1);
        let texture = |extent| AllocationDesc::Texture {
            extent,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        };
        let plan = [
            texture(partial_extent),
            texture(middle_extent),
            texture(metric_extent),
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Staging { bytes: 192 },
            AllocationDesc::Readback {
                extent: metric_extent,
                bytes_per_pixel: 4,
            },
        ];
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(executor.device(), &plan)
            .map_err(|e| GpuError(e.to_string()))?;
        let partial = resources[0].1.texture();
        let middle = resources[1].1.texture();
        let metrics = resources[2].1.texture().clone();
        let readback = resources[7].1.buffer().clone();
        let common = common_layout(executor.device(), wgpu::TextureFormat::R32Float);
        let extra = executor
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("seam partial sums"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let make_pipeline = |source: &'static str, additional: bool| {
            let layouts = [Some(&common), Some(&extra)];
            let layout =
                executor
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("seam diagnostic"),
                        bind_group_layouts: &layouts[..if additional { 2 } else { 1 }],
                        immediate_size: 0,
                    });
            let module = executor
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("seam diagnostic"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            executor
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("seam diagnostic"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let gradient = make_pipeline(
            include_str!("../../shaders/seam_heatmap/gradient.wgsl"),
            true,
        );
        let reduction = make_pipeline(
            include_str!("../../shaders/seam_heatmap/reduce.wgsl"),
            false,
        );
        let mut uniform = NodeUniform::new(size, size);
        // Counts are directional, excluding singleton axes.
        let samples =
            2 * (4.min(size.width / 2) * size.height + 4.min(size.height / 2) * size.width);
        uniform.parameters[0] = if samples == 0 {
            0.0
        } else {
            1.0 / samples as f32
        };
        uniform.flags[0] = u32::from(input.format() == wgpu::TextureFormat::Rgba16Float);
        let mut reduce_uniform = NodeUniform::new(middle_extent, partial_extent);
        reduce_uniform.flags[0] = 0;
        let mut final_uniform = NodeUniform::new(metric_extent, middle_extent);
        final_uniform.flags[0] = 1;
        let uniforms = [uniform, reduce_uniform, final_uniform];
        let partial_view = partial.create_view(&Default::default());
        let additional = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("seam partials"),
                layout: &extra,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&partial_view),
                }],
            });
        let passes = [
            (&gradient, input, heatmap, groups, Some(additional)),
            (&reduction, partial, middle, [reduction_groups, 1, 1], None),
            (&reduction, middle, &metrics, [1, 1, 1], None),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (pipeline, input, output, groups, additional))| {
            let buffer = resources[3 + index].1.buffer();
            let input_view = input.create_view(&Default::default());
            let output_view = output.create_view(&Default::default());
            let common = executor
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("seam common ABI"),
                    layout: &common,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
            PreparedPass {
                pipeline: pipeline.clone(),
                common,
                buffer: buffer.clone(),
                weights: None,
                queue: executor.queue.clone(),
                operation: Pass::Downsample,
                evaluation: size,
                source: size,
                groups,
                additional,
            }
        })
        .collect();
        Ok(Self {
            passes,
            uniforms,
            pool,
            allocations: resources.iter().map(|(id, _)| *id).collect(),
            metrics,
            readback,
            device: executor.device().clone(),
            queue: executor.queue.clone(),
            submitted: None,
        })
    }

    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }

    /// Check the caller's generation token between passes. If false is returned,
    /// discard this encoder and do not publish its heatmap or metrics.
    pub fn encode_if_current(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mut current: impl FnMut() -> bool,
    ) -> bool {
        for pass in &self.passes {
            if !current() {
                return false;
            }
            pass.encode(encoder);
        }
        current()
    }

    /// Fence all reservations and queue immutable uniforms on first submission.
    /// Call immediately before submitting the encoded passes, then acknowledge
    /// completion before retiring this fence, even if the generation goes stale.
    /// Preparation and discarded encoders never leave pending queue uploads.
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        if self.submitted.is_none() {
            for (pass, uniform) in self.passes.iter().zip(&self.uniforms) {
                self.queue
                    .write_buffer(&pass.buffer, 0, bytemuck::bytes_of(uniform));
            }
        }
        self.submitted = Some(fence);
        Ok(())
    }

    pub fn complete(&mut self, fence: FenceId) {
        self.pool.lock().unwrap().complete(fence);
    }

    /// Blocking worker-only scalar readback. Includes a queue completion fence;
    /// never call on the UI thread. All per-pixel reduction remains on the GPU.
    pub fn read_metrics(&self) -> Result<SeamMetrics, GpuError> {
        if self.submitted.is_none() {
            return Err(GpuError("seam diagnostic has not been submitted".into()));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.metrics.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            self.metrics.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|e| GpuError(e.to_string()))?;
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|e| GpuError(e.to_string()))?
            .map_err(|e| GpuError(e.to_string()))?;
        let mapped = self
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError(e.to_string()))?;
        let values: [f32; 3] = std::array::from_fn(|i| {
            f32::from_le_bytes(mapped[i * 4..i * 4 + 4].try_into().unwrap())
        });
        drop(mapped);
        self.readback.unmap();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(GpuError("nonfinite seam metrics".into()));
        }
        Ok(SeamMetrics {
            boundary_rms: values[0],
            interior_rms: values[1],
            ratio: values[2],
        })
    }
}

impl Drop for PreparedSeamHeatmap {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            let _ = pool.release_transient(*id);
        }
    }
}
