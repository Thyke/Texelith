//! Export-only conversion of full-resolution typed maps to numeric file planes.
//! Caller-owned source/output textures must be admitted to the same ledger.
//! Encoding is allocation-free; the caller schedules bounded batches, checks job
//! cancellation between passes, and acknowledges actual queue completion fences.
use crate::{
    GpuError, GpuExecutor, NodeUniform, Pass, PreparedPass,
    executor::{common_layout, texture_extent},
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
    workgroups,
};
use std::sync::{Arc, Mutex};
use texelith_core::{
    Boundary, ChannelSource, Component, DisplacementSettings, ExportFormat, ExportOutputSemantics,
    ExportPlan, Extent, IntegerDither, NormalConvention, OutputPort, OutputRef, SizePolicy,
    Transfer, validate_export,
};

/// Full-resolution material output, with sampling and signed-normal metadata.
pub struct ExportPackingSource<'a> {
    pub reference: OutputRef,
    pub texture: &'a wgpu::Texture,
    pub boundary: Boundary,
    pub normal_convention: NormalConvention,
}

/// File semantics and padding geometry accompany the numeric Rgba32Float carrier.
#[derive(Clone, Debug)]
pub struct ExportPackingMetadata {
    pub extent: Extent,
    pub valid_content: [u32; 4],
    pub uv_scale: [f32; 2],
    pub padding: Boundary,
    pub semantics: ExportOutputSemantics,
    pub displacement: DisplacementSettings,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Channels {
    descriptors: [[u32; 4]; 4],
    constants: [f32; 4],
}

const _: () = {
    assert!(size_of::<Channels>() == 80);
    assert!(align_of::<Channels>() == 16);
};

/// Lazily compiled on the worker and reusable across export jobs.
#[derive(Default)]
pub struct ExportPackingKernel {
    packing: std::sync::OnceLock<wgpu::ComputePipeline>,
    horizontal: std::sync::OnceLock<wgpu::ComputePipeline>,
}

fn sampled(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn pipeline(device: &wgpu::Device, packing: bool) -> wgpu::ComputePipeline {
    let common = common_layout(device, wgpu::TextureFormat::Rgba32Float);
    let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("export channel ABI"),
        entries: &[
            sampled(0),
            sampled(1),
            sampled(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(80),
                },
                count: None,
            },
        ],
    });
    let groups = [Some(&common), Some(&extra)];
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("export packing"),
        bind_group_layouts: &groups[..if packing { 2 } else { 1 }],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("export packing"),
        source: wgpu::ShaderSource::Wgsl(if packing {
            concat!(
                include_str!("../../shaders/export_packing/common.wgsl"),
                "\n",
                include_str!("../../shaders/export_packing/packing.wgsl")
            )
            .into()
        } else {
            concat!(
                include_str!("../../shaders/export_packing/common.wgsl"),
                "\n",
                include_str!("../../shaders/export_packing/horizontal.wgsl")
            )
            .into()
        }),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("export packing"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Prepared, immutable export output. Scratch and queue-write staging remain
/// charged after drop until the last submitted fence completes.
pub struct PreparedExportPacking {
    passes: Vec<PreparedPass>,
    metadata: ExportPackingMetadata,
    pool: Arc<Mutex<ResourcePool>>,
    allocations: Vec<AllocationId>,
}

impl PreparedExportPacking {
    pub fn new(
        executor: &GpuExecutor,
        plan: &ExportPlan,
        output_index: usize,
        sources: &[ExportPackingSource<'_>],
        output: &wgpu::Texture,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        ExportPackingKernel::default().prepare(executor, plan, output_index, sources, output, pool)
    }

    pub fn metadata(&self) -> &ExportPackingMetadata {
        &self.metadata
    }
    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        Ok(())
    }
    /// Call only after waiting for the queue through this fence.
    pub fn complete(&mut self, fence: FenceId) {
        self.pool.lock().unwrap().complete(fence);
    }
}
impl Drop for PreparedExportPacking {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.allocations {
            let _ = pool.release_transient(*id);
        }
    }
}

impl ExportPackingKernel {
    pub fn prepare(
        &self,
        executor: &GpuExecutor,
        plan: &ExportPlan,
        output_index: usize,
        sources: &[ExportPackingSource<'_>],
        output: &wgpu::Texture,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<PreparedExportPacking, GpuError> {
        validate_export(plan)?;
        let selected = plan
            .outputs
            .get(output_index)
            .ok_or_else(|| GpuError("invalid export output index".into()))?;
        let extent = match plan.size {
            SizePolicy::Keep => plan.source_extent,
            SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
                Extent::new(width, height)?
            }
        };
        if texture_extent(output)? != extent
            || output.format() != wgpu::TextureFormat::Rgba32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError(
                "invalid export destination format, extent or usage".into(),
            ));
        }
        for (i, source) in sources.iter().enumerate() {
            let vector = matches!(
                source.reference.port,
                OutputPort::Albedo | OutputPort::Normal
            );
            if sources[..i]
                .iter()
                .any(|other| other.reference == source.reference)
                || source.texture == output
                || texture_extent(source.texture)? != plan.source_extent
                || source.texture.format()
                    != if vector {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        wgpu::TextureFormat::R32Float
                    }
                || !source
                    .texture
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            {
                return Err(GpuError(
                    "invalid, duplicate, aliased or proxy export source".into(),
                ));
            }
        }
        let mut channels = Channels {
            descriptors: [[0; 4]; 4],
            constants: [0.0, 0.0, 0.0, 1.0],
        };
        let mut indices = [None; 4];
        for (i, channel) in selected.channels.iter().enumerate() {
            let mut flags =
                u32::from(channel.invert) | (u32::from(channel.transfer == Transfer::Srgb) << 1);
            let (kind, component, boundary) = match channel.source {
                ChannelSource::Constant(value) => {
                    channels.constants[i] = value;
                    (0, 0, Boundary::Clamp)
                }
                ChannelSource::Map(reference, component) => {
                    let index = sources
                        .iter()
                        .position(|s| s.reference == reference)
                        .ok_or_else(|| GpuError("missing full-resolution export source".into()))?;
                    indices[i] = Some(index);
                    let source = &sources[index];
                    let kind = match reference.port {
                        OutputPort::Albedo => 2,
                        OutputPort::Normal => 3,
                        _ => 1,
                    };
                    if kind == 3 && source.normal_convention != plan.profile.normal_convention {
                        flags |= 4;
                    }
                    (
                        kind,
                        match component {
                            Component::R => 0,
                            Component::G => 1,
                            Component::B => 2,
                            Component::A => 3,
                        },
                        source.boundary,
                    )
                }
            };
            channels.descriptors[i] = [
                kind,
                component,
                u32::from(boundary == Boundary::Repeat),
                flags,
            ];
        }
        let resize_x = matches!(plan.size, SizePolicy::Resize { .. })
            && extent.width != plan.source_extent.width;
        let mut unique = Vec::new();
        if resize_x {
            for index in indices.into_iter().flatten() {
                if !unique.contains(&index) {
                    unique.push(index);
                }
            }
        }
        // Rgba32Float is intentionally task-local: the shared pool texture enum
        // has no such format yet. Reserve exact payload bytes atomically before
        // creating these textures; wgpu holds submitted bindings through fences.
        let intermediate = Extent::new(extent.width, plan.source_extent.height)?;
        let mut allocation_plan = vec![
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Buffer { bytes: 80 },
            AllocationDesc::Staging { bytes: 144 },
        ];
        for _ in &unique {
            allocation_plan.extend([
                AllocationDesc::Buffer { bytes: 64 },
                AllocationDesc::Staging { bytes: 64 },
                AllocationDesc::Staging {
                    bytes: u64::from(intermediate.width) * u64::from(intermediate.height) * 16,
                },
            ]);
        }
        // Constant-only output needs a readable dummy, never the output itself.
        let dummy = indices.iter().all(Option::is_none);
        if dummy {
            allocation_plan.push(AllocationDesc::Texture {
                extent: Extent::new(1, 1)?,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            });
        }
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(executor.device(), &allocation_plan)
            .map_err(|e| GpuError(e.to_string()))?;
        let mut prepared = PreparedExportPacking {
            passes: Vec::with_capacity(unique.len() + 1),
            pool,
            allocations: resources.iter().map(|(id, _)| *id).collect(),
            metadata: ExportPackingMetadata {
                extent,
                valid_content: [0, 0, extent.width, extent.height],
                uv_scale: [1.0; 2],
                padding: plan.padding,
                semantics: selected.semantics(),
                displacement: plan.displacement,
            },
        };
        if matches!(plan.size, SizePolicy::Pad { .. }) {
            prepared.metadata.valid_content =
                [0, 0, plan.source_extent.width, plan.source_extent.height];
            prepared.metadata.uv_scale = [
                plan.source_extent.width as f32 / extent.width as f32,
                plan.source_extent.height as f32 / extent.height as f32,
            ];
        }
        let mut intermediates = Vec::new();
        for (i, index) in unique.iter().enumerate() {
            let texture = executor.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("export horizontal f32 precision"),
                size: wgpu::Extent3d {
                    width: intermediate.width,
                    height: intermediate.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let mut uniform = NodeUniform::new(intermediate, plan.source_extent);
            uniform.flags[0] = u32::from(sources[*index].boundary == Boundary::Repeat);
            let pipeline = self
                .horizontal
                .get_or_init(|| pipeline(executor.device(), false));
            prepared.passes.push(make_pass(
                executor,
                pipeline,
                sources[*index].texture,
                &texture,
                uniform,
                resources[3 + i * 3].1.buffer(),
                None,
            )?);
            intermediates.push(texture);
        }
        let fallback = if dummy {
            resources.last().unwrap().1.texture()
        } else {
            sources[indices.iter().flatten().next().copied().unwrap()].texture
        };
        let inputs: [_; 4] = std::array::from_fn(|i| {
            indices[i].map_or(fallback, |index| {
                unique
                    .iter()
                    .position(|j| *j == index)
                    .map_or(sources[index].texture, |j| &intermediates[j])
            })
        });
        let pipeline = self
            .packing
            .get_or_init(|| pipeline(executor.device(), true));
        let cb = resources[1].1.buffer();
        executor
            .queue
            .write_buffer(cb, 0, bytemuck::bytes_of(&channels));
        let views = inputs.map(|t| t.create_view(&Default::default()));
        let extra = executor
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("export channel sources"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cb.as_entire_binding(),
                    },
                ],
            });
        let mut uniform = NodeUniform::new(extent, plan.source_extent);
        let q = selected.quantization.unwrap_or_default();
        uniform.parameters = [
            q.clamp_min,
            q.clamp_max,
            plan.displacement.midlevel,
            plan.displacement.scale,
        ];
        uniform.flags = [
            match plan.size {
                SizePolicy::Keep => 0,
                SizePolicy::Resize { .. } => 1,
                SizePolicy::Pad { .. } => 2,
            },
            if selected.format == ExportFormat::Exr32 {
                0
            } else {
                (1 << selected.format.bit_depth()) - 1
            },
            u32::from(q.dither == IntegerDither::Ordered4x4),
            selected.channels.len() as u32,
        ];
        uniform.reserved[0] = u32::from(plan.padding == Boundary::Repeat);
        prepared.passes.push(make_pass(
            executor,
            pipeline,
            inputs[0],
            output,
            uniform,
            resources[0].1.buffer(),
            Some(extra),
        )?);
        Ok(prepared)
    }
}

fn make_pass(
    executor: &GpuExecutor,
    pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    uniform: NodeUniform,
    buffer: &wgpu::Buffer,
    additional: Option<wgpu::BindGroup>,
) -> Result<PreparedPass, GpuError> {
    executor
        .queue
        .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
    let iv = input.create_view(&Default::default());
    let ov = output.create_view(&Default::default());
    let common = executor
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export common ABI"),
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
    let evaluation = Extent::new(uniform.dimensions[0], uniform.dimensions[1])?;
    Ok(PreparedPass {
        pipeline: pipeline.clone(),
        common,
        buffer: buffer.clone(),
        weights: None,
        queue: executor.queue.clone(),
        operation: Pass::Downsample,
        evaluation,
        source: Extent::new(uniform.dimensions[2], uniform.dimensions[3])?,
        groups: workgroups(evaluation)?,
        additional,
    })
}
