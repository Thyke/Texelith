use crate::{GpuError, Pass, workgroups};
use std::{num::NonZeroU64, sync::Arc};
use texelith_core::Extent;
use wgpu::util::DeviceExt;

const SHADERS: [&str; 6] = [
    include_str!("../shaders/source.wgsl"),
    include_str!("../shaders/downsample.wgsl"),
    include_str!("../shaders/levels.wgsl"),
    include_str!("../shaders/blur.wgsl"),
    include_str!("../shaders/normal.wgsl"),
    include_str!("../shaders/display.wgsl"),
];
const NAMES: [&str; 6] = [
    "source",
    "downsample",
    "levels",
    "blur",
    "normal",
    "display",
];

pub struct GpuExecutor {
    device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pipelines: Vec<wgpu::ComputePipeline>,
    typed_pipelines: [std::sync::OnceLock<wgpu::ComputePipeline>; 2],
    pyramid_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    pub(crate) height_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    // FAZ1_GPU_PIPELINES: T-048 begin
    pub(crate) normal_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    // FAZ1_GPU_PIPELINES: T-048 end
}

impl GpuExecutor {
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }
    /// Prepare a wrapped scalar primitive without allocation of any textures.
    /// The additional sampled texture uses group 1 binding 0 (unfilterable f32).
    pub fn prepare_pyramid(
        &self,
        operation: crate::PyramidPass,
        input: &wgpu::Texture,
        coarse: Option<&wgpu::Texture>,
        output: &wgpu::Texture,
        boundary: texelith_core::Boundary,
    ) -> Result<PreparedPass, GpuError> {
        self.prepare_pyramid_with_buffer(operation, input, coarse, output, boundary, None)
    }

    pub(crate) fn prepare_pyramid_with_buffer(
        &self,
        operation: crate::PyramidPass,
        input: &wgpu::Texture,
        coarse: Option<&wgpu::Texture>,
        output: &wgpu::Texture,
        boundary: texelith_core::Boundary,
        pooled_buffer: Option<&wgpu::Buffer>,
    ) -> Result<PreparedPass, GpuError> {
        use crate::{Axis, NodeUniform, PyramidPass as P};
        let fine = texture_extent(input)?;
        let evaluation = texture_extent(output)?;
        let half = |e: Extent| Extent {
            width: e.width.div_ceil(2),
            height: e.height.div_ceil(2),
        };
        let needs_coarse = matches!(operation, P::LaplacianBand | P::WeightedReconstruct { .. });
        let valid_read = |t: &wgpu::Texture| {
            t != output
                && t.format() == wgpu::TextureFormat::R32Float
                && t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        };
        if !valid_read(input)
            || output.format() != wgpu::TextureFormat::R32Float
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || needs_coarse != coarse.is_some()
            || match operation {
                P::AreaDownsample | P::GaussianAreaDownsample => evaluation != half(fine),
                P::BilinearUpsample => fine != half(evaluation),
                _ => fine != evaluation,
            }
        {
            return Err(GpuError("invalid pyramid textures or extents".into()));
        }
        if let Some(t) = coarse
            && (!valid_read(t) || texture_extent(t)? != half(fine))
        {
            return Err(GpuError("invalid pyramid coarse input".into()));
        }
        let mut uniform = NodeUniform::new(evaluation, fine);
        uniform.flags = [
            match operation {
                P::Gaussian(_) => 0,
                P::AreaDownsample => 1,
                P::GaussianAreaDownsample => 5,
                P::BilinearUpsample => 2,
                P::LaplacianBand => 3,
                P::WeightedReconstruct { .. } => 4,
            },
            u32::from(boundary == texelith_core::Boundary::Repeat),
            u32::from(matches!(operation, P::Gaussian(Axis::Y))),
            0,
        ];
        if let P::WeightedReconstruct {
            band_weight,
            coarse_weight,
        } = operation
        {
            if !band_weight.is_finite() || !coarse_weight.is_finite() {
                return Err(GpuError("nonfinite pyramid weight".into()));
            }
            uniform.parameters = [band_weight, coarse_weight, 0.0, 0.0];
        }
        let pipeline = self
            .pyramid_pipeline
            .get_or_init(|| {
                let common = common_layout(&self.device, wgpu::TextureFormat::R32Float);
                let extra =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("pyramid coarse ABI"),
                            entries: &[wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: false,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            }],
                        });
                let layout = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("wrapped pyramid"),
                        bind_group_layouts: &[Some(&common), Some(&extra)],
                        immediate_size: 0,
                    });
                let source = concat!(
                    include_str!("../shaders/wrapped.wgsl"),
                    "\n",
                    include_str!("../shaders/pyramid.wgsl")
                );
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("wrapped pyramid"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("wrapped pyramid"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .clone();
        let buffer = if let Some(buffer) = pooled_buffer {
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
            buffer.clone()
        } else {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pyramid NodeUniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let cv = coarse.unwrap_or(input).create_view(&Default::default());
        let common = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyramid common"),
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
        let additional = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyramid coarse"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&cv),
            }],
        });
        Ok(PreparedPass {
            pipeline,
            common,
            buffer,
            weights: None,
            queue: self.queue.clone(),
            operation: Pass::Downsample,
            evaluation,
            source: fine,
            groups: workgroups(evaluation)?,
            additional: Some(additional),
        })
    }

    /// Compile once on the worker with the presentation device/queue. No adapter,
    /// device, runtime, thread, or polling loop is created here. Await validation
    /// outside UI update. The device must enable at least 4096 texture dimensions.
    pub async fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<Self, GpuError> {
        let limits = device.limits();
        if limits.max_texture_dimension_2d < 4096
            || limits.max_compute_workgroup_size_x < 8
            || limits.max_compute_workgroup_size_y < 8
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroups_per_dimension < 512
        {
            return Err(GpuError(
                "shared device lacks required 4K compute limits".into(),
            ));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let common: Vec<_> = [
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .map(|format| common_layout(&device, format))
        .collect();
        let weights = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texelith blur weights"),
            entries: &[uniform_entry(0, 128)],
        });
        let mut pipelines = Vec::with_capacity(6);
        for (index, source) in SHADERS.into_iter().enumerate() {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(NAMES[index]),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let common = &common[match index {
                4 => 1,
                5 => 2,
                _ => 0,
            }];
            let groups = [Some(common), Some(&weights)];
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(NAMES[index]),
                bind_group_layouts: &groups[..if index == 3 { 2 } else { 1 }],
                immediate_size: 0,
            });
            pipelines.push(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(NAMES[index]),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        if let Some(error) = scope.pop().await {
            return Err(GpuError(error.to_string()));
        }
        Ok(Self {
            device,
            queue,
            pipelines,
            typed_pipelines: Default::default(),
            pyramid_pipeline: Default::default(),
            height_pipeline: Default::default(),
            // FAZ1_GPU_PIPELINE_INIT: T-048 begin
            normal_pipeline: Default::default(),
            // FAZ1_GPU_PIPELINE_INIT: T-048 end
        })
    }

    /// Prepare persistent bindings for caller-owned textures. Both must belong
    /// to this shared device. No texture is allocated or submitted here; bind
    /// groups retain their resources. Different allocations are mandatory.
    pub fn prepare(
        &self,
        operation: Pass,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        source: Extent,
    ) -> Result<PreparedPass, GpuError> {
        self.prepare_inner(operation, input, output, source, None)
    }

    /// Worker-owned, budgeted uniform allocations. Bindings can be retained and
    /// updated while the exact input/output allocations remain in the pool.
    pub(crate) fn prepare_pooled(
        &self,
        operation: Pass,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        source: Extent,
        buffers: (&wgpu::Buffer, Option<&wgpu::Buffer>),
    ) -> Result<PreparedPass, GpuError> {
        self.prepare_inner(operation, input, output, source, Some(buffers))
    }

    fn prepare_inner(
        &self,
        operation: Pass,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        source: Extent,
        buffers: Option<(&wgpu::Buffer, Option<&wgpu::Buffer>)>,
    ) -> Result<PreparedPass, GpuError> {
        if input == output {
            return Err(GpuError("compute input and output must not alias".into()));
        }
        let evaluation = texture_extent(output)?;
        let input_extent = texture_extent(input)?;
        let expected_input = if matches!(operation, Pass::Downsample) {
            source
        } else {
            evaluation
        };
        if input_extent != expected_input
            || input.format() != operation.input_format()
            || output.format() != operation.output_format()
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(GpuError(
                "compute texture extent, format or usage mismatch".into(),
            ));
        }
        let (uniform, weights) = operation.uniforms(evaluation, source)?;
        let pipeline = if operation.index() < 6 {
            self.pipelines[operation.index()].clone()
        } else {
            self.typed_pipelines[operation.index() - 6]
                .get_or_init(|| {
                    let common = common_layout(&self.device, wgpu::TextureFormat::Rgba16Float);
                    let layout =
                        self.device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some(operation.name()),
                                bind_group_layouts: &[Some(&common)],
                                immediate_size: 0,
                            });
                    let module = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(operation.name()),
                            source: wgpu::ShaderSource::Wgsl(
                                include_str!("typed_source.wgsl").into(),
                            ),
                        });
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(operation.name()),
                            layout: Some(&layout),
                            module: &module,
                            entry_point: Some(if matches!(operation, Pass::ColorSource) {
                                "color"
                            } else {
                                "normal"
                            }),
                            compilation_options: Default::default(),
                            cache: None,
                        })
                })
                .clone()
        };
        let buffer = if let Some((buffer, _)) = buffers {
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
            buffer.clone()
        } else {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Texelith node uniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        };
        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let common = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(operation.name()),
            layout: &pipeline.get_bind_group_layout(0),
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
        let weights = weights.map(|weights| {
            let buffer = if let Some((_, Some(buffer))) = buffers {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::bytes_of(&weights));
                buffer.clone()
            } else {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Texelith Gaussian weights"),
                        contents: bytemuck::bytes_of(&weights),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            };
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texelith Gaussian weights"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, group)
        });
        Ok(PreparedPass {
            pipeline,
            common,
            buffer,
            weights,
            queue: self.queue.clone(),
            operation,
            evaluation,
            source,
            groups: workgroups(evaluation)?,
            additional: None,
        })
    }
}

/// Reusable encoding state. Keep separate instances for each pass in a batch
/// (particularly blur X/Y); their uniforms must not overwrite each other.
#[derive(Clone)]
pub struct PreparedPass {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) common: wgpu::BindGroup,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) weights: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) operation: Pass,
    pub(crate) evaluation: Extent,
    pub(crate) source: Extent,
    pub(crate) groups: [u32; 3],
    pub(crate) additional: Option<wgpu::BindGroup>,
}

impl PreparedPass {
    /// Reuse buffers/bindings for a parameter edit. Call only after submitting
    /// any earlier encoding that uses this object: queue writes precede the next
    /// submission, not individual dispatches in an unsubmitted command buffer.
    /// T-017 also waits for retirement before reusing the output texture.
    pub fn update(&mut self, operation: Pass) -> Result<(), GpuError> {
        if self.additional.is_some() {
            return Err(GpuError("pyramid bindings are immutable".into()));
        }
        if operation.index() != self.operation.index()
            || operation.input_format() != self.operation.input_format()
        {
            return Err(GpuError(
                "parameter update cannot change pipeline or input format".into(),
            ));
        }
        let (uniform, weights) = operation.uniforms(self.evaluation, self.source)?;
        self.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&uniform));
        if let (Some((buffer, _)), Some(weights)) = (&self.weights, weights) {
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&weights));
        }
        self.operation = operation;
        Ok(())
    }

    /// Encodes into the caller's batch with no resource allocations or UI wait.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.operation.name()),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        if let Some((_, group)) = &self.weights {
            pass.set_bind_group(1, group, &[]);
        }
        if let Some(group) = &self.additional {
            pass.set_bind_group(1, group, &[]);
        }
        pass.dispatch_workgroups(self.groups[0], self.groups[1], self.groups[2]);
    }

    pub fn uniform_bytes(&self) -> u64 {
        64 + if self.weights.is_some() { 128 } else { 0 }
    }
}

pub(crate) fn texture_extent(texture: &wgpu::Texture) -> Result<Extent, GpuError> {
    if texture.dimension() != wgpu::TextureDimension::D2
        || texture.depth_or_array_layers() != 1
        || texture.mip_level_count() != 1
        || texture.sample_count() != 1
    {
        return Err(GpuError(
            "compute requires single-layer, single-mip, single-sample 2D textures".into(),
        ));
    }
    Ok(Extent::new(texture.width(), texture.height())?)
}

fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

pub(crate) fn common_layout(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texelith common ABI"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            uniform_entry(2, 64),
        ],
    })
}
