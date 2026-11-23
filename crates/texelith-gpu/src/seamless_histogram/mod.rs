//! Integer source histograms and variance-normalized Gaussian luminance blending.
//! Source mapping is independent of the selected spatial base and strength.
use crate::{GpuError, NodeUniform, executor, workgroups};
use std::sync::atomic::{AtomicU64, Ordering};
use texelith_core::{Boundary, Extent, SeamlessBase};

pub const BINS: u32 = 1024;
pub const GAUSSIAN_ENTRIES: u64 = 8193;
pub const ALGORITHM_REVISION: u32 = 1;
const TABLE_BYTES: u64 = GAUSSIAN_ENTRIES * 4;
const CDF_BYTES: u64 = 2056 * 4;
static HITS: AtomicU64 = AtomicU64::new(0);
static BUILDS: AtomicU64 = AtomicU64::new(0);
/// Serial verification counters; never scheduling inputs.
pub fn diagnostics() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), BUILDS.load(Ordering::Relaxed))
}
pub(crate) fn cache_event(hit: bool) {
    if hit {
        HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILDS.fetch_add(1, Ordering::Relaxed);
    }
}
fn chunks(e: Extent) -> u32 {
    (e.width * e.height).div_ceil(4096)
}
/// Storage, uniforms and queue-write staging, excluding input/output textures.
pub fn scratch_bytes(e: Extent) -> u64 {
    u64::from(chunks(e)) * (1024 * 4 + 8) + CDF_BYTES + TABLE_BYTES * 2 + 128
}
/// Fixed Gaussian CDF, integrated in f64 once per preparation, with no image data.
pub fn gaussian_table() -> Vec<f32> {
    let mut values = vec![0.0f64; GAUSSIAN_ENTRIES as usize];
    let density = |i: usize| (-0.5 * (i as f64 / 512.0 - 8.0).powi(2)).exp();
    for i in 1..values.len() {
        values[i] = values[i - 1] + (density(i - 1) + density(i)) / 1024.0;
    }
    let total = values[values.len() - 1];
    values.into_iter().map(|v| (v / total) as f32).collect()
}

/// Caller must admit `scratch_bytes(source)` before creating these resources and
/// retain that reservation through every referencing submission fence.
pub struct HistogramResources {
    partial: wgpu::Buffer,
    moments: wgpu::Buffer,
    pub cdf: wgpu::Buffer,
    gaussian: wgpu::Buffer,
    uniform: wgpu::Buffer,
    source: Extent,
}
impl HistogramResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, source: Extent) -> Self {
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("histogram admitted storage"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let gaussian = buffer(TABLE_BYTES, storage | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&gaussian, 0, bytemuck::cast_slice(&gaussian_table()));
        Self {
            partial: buffer(u64::from(chunks(source)) * 4096, storage),
            moments: buffer(u64::from(chunks(source)) * 8, storage),
            cdf: buffer(CDF_BYTES, storage),
            gaussian,
            uniform: buffer(
                64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            source,
        }
    }
}
/// Pipelines are lazy worker state; the group-0 ABI remains unchanged.
pub struct HistogramKernels {
    scalar: Vec<wgpu::ComputePipeline>,
    color: Vec<wgpu::ComputePipeline>,
}
impl HistogramKernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let create = |format, storage| {
            let common = executor::common_layout(device, format);
            let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("histogram group-1 storage ABI"),
                entries: &[0, 1, 2, 3].map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 3,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("histogram"),
                bind_group_layouts: &[Some(&common), Some(&extra)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("histogram"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/seamless_histogram/histogram.wgsl")
                        .replace("OUTPUT_FORMAT", storage)
                        .into(),
                ),
            });
            ["histogram", "reduce", "cdf", "blend"]
                .map(|entry| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&layout),
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .to_vec()
        };
        Self {
            scalar: create(wgpu::TextureFormat::R32Float, "r32float"),
            color: create(wgpu::TextureFormat::Rgba16Float, "rgba16float"),
        }
    }
    #[allow(clippy::too_many_arguments)] pub fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        resources: &HistogramResources,
        base: &SeamlessBase,
        boundary: Boundary,
        strength: f32,
    ) -> Result<PreparedHistogram, GpuError> {
        base.validate()?;
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(GpuError("invalid histogram strength".into()));
        }
        let source = executor::texture_extent(input)?;
        let evaluation = executor::texture_extent(output)?;
        let color = output.format() == wgpu::TextureFormat::Rgba16Float;
        if input == output
            || source != resources.source
            || !(matches!(
                output.format(),
                wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba16Float
            ) && (input.format() == output.format()
                || (!color && input.format() == wgpu::TextureFormat::Rgba8Unorm)))
        {
            return Err(GpuError("invalid histogram source/output".into()));
        }
        let pipelines = if color { &self.color } else { &self.scalar };
        let mut uniform = NodeUniform::new(evaluation, source);
        uniform.flags[0] = u32::from(boundary == Boundary::Repeat);
        uniform.flags[3] = u32::from(color);
        uniform.parameters[1] = strength;
        match *base {
            SeamlessBase::OffsetBlend { overlap } => uniform.parameters[0] = overlap,
            SeamlessBase::Splat {
                seed,
                grid,
                overlap,
            } => {
                uniform.parameters[0] = overlap;
                uniform.flags[1] = seed;
                uniform.flags[2] = grid;
            }
        }
        queue.write_buffer(&resources.uniform, 0, bytemuck::bytes_of(&uniform));
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram common"),
            layout: &pipelines[0].get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&iv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ov),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.uniform.as_entire_binding(),
                },
            ],
        });
        let extra = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram storage"),
            layout: &pipelines[0].get_bind_group_layout(1),
            entries: &[
                &resources.partial,
                &resources.moments,
                &resources.cdf,
                &resources.gaussian,
            ]
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let [x, y, _] = workgroups(evaluation)?;
        Ok(PreparedHistogram {
            pipelines: pipelines.clone(),
            common,
            extra,
            groups: (x, y),
            chunks: chunks(source),
        })
    }
}
pub struct PreparedHistogram {
    pipelines: Vec<wgpu::ComputePipeline>,
    common: wgpu::BindGroup,
    extra: wgpu::BindGroup,
    groups: (u32, u32),
    chunks: u32,
}
impl PreparedHistogram {
    pub fn encode_pass(&self, encoder: &mut wgpu::CommandEncoder, index: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines[index]);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &self.extra, &[]);
        let (x, y) = match index {
            0 => (self.chunks, 1),
            1 => (16, 1),
            2 => (1, 1),
            _ => self.groups,
        };
        pass.dispatch_workgroups(x, y, 1);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, rebuild: bool) {
        for i in if rebuild { 0..4 } else { 3..4 } {
            self.encode_pass(encoder, i);
        }
    }
}
