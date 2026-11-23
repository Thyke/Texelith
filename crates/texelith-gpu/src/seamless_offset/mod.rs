//! Periodic overlap-add for correlated linear color and scalar source layers.
//!
//! All layers use the same four continuous translations and cubic windows.
//! Structured photos can lose composition or show ghosting after blending.
use crate::{GpuError, NodeUniform, Pass, PreparedPass, executor, workgroups};
use std::sync::Arc;
use texelith_core::{Boundary, Extent};

pub const EPSILON: f32 = 1e-8;
pub const ALGORITHM_REVISION: u32 = 1;

/// One continuously translated source coordinate and its unnormalized window.
#[derive(Clone, Copy, Debug)]
pub struct WeightedSample {
    pub uv: [f32; 2],
    pub weight: f32,
}

/// Reusable spatial metadata; independent of source semantics and dimensions.
#[derive(Clone, Copy, Debug)]
pub struct OffsetTransform {
    overlap: f32,
}
impl OffsetTransform {
    pub fn new(overlap: f32) -> Result<Self, GpuError> {
        if !overlap.is_finite() || !(0.1..=0.5).contains(&overlap) {
            return Err(GpuError(
                "offset overlap must be finite and in [0.1, 0.5]".into(),
            ));
        }
        Ok(Self { overlap })
    }

    /// Stable order: half/half, zero/half, half/zero, zero/zero.
    /// Coordinates supplied by callers must be finite.
    pub fn samples(&self, uv: [f32; 2]) -> [WeightedSample; 4] {
        let window = |s: f32| {
            let t = (s.min(1.0 - s) / self.overlap).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        std::array::from_fn(|i| {
            let uv = [
                (uv[0] + if i & 1 == 0 { 0.5 } else { 0.0 }).rem_euclid(1.0),
                (uv[1] + if i & 2 == 0 { 0.5 } else { 0.0 }).rem_euclid(1.0),
            ];
            WeightedSample {
                uv,
                weight: window(uv[0]) * window(uv[1]),
            }
        })
    }

    pub fn overlap(&self) -> f32 {
        self.overlap
    }
}

/// Format-specific pipelines, created on the compute worker and reused.
pub struct OffsetKernels {
    scalar: wgpu::ComputePipeline,
    color: wgpu::ComputePipeline,
    queue: Arc<wgpu::Queue>,
}

impl OffsetKernels {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self {
        let pipeline = |format, storage| {
            let common = executor::common_layout(device, format);
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("offset overlap-add"),
                bind_group_layouts: &[Some(&common)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("offset overlap-add"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/seamless_offset/offset.wgsl")
                        .replace("OUTPUT_FORMAT", storage)
                        .into(),
                ),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("offset overlap-add"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            scalar: pipeline(wgpu::TextureFormat::R32Float, "r32float"),
            color: pipeline(wgpu::TextureFormat::Rgba16Float, "rgba16float"),
            queue,
        }
    }

    /// Prepare only when resource bindings change. Uniform storage is supplied
    /// by the caller's admission ledger, including its 64-byte write staging.
    /// Samples the continuous source field at output pixel centers, allowing
    /// correlated layers to share a transform at reduced evaluation extents.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        buffer: &wgpu::Buffer,
        transform: OffsetTransform,
        boundary: Boundary,
    ) -> Result<PreparedOffset, GpuError> {
        let source = executor::texture_extent(input)?;
        let evaluation = executor::texture_extent(output)?;
        if input == output
            || input.format() != output.format()
            || !matches!(
                input.format(),
                wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba16Float
            )
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || buffer.size() < 64
            || !buffer
                .usage()
                .contains(wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
        {
            return Err(GpuError("invalid offset textures or uniform buffer".into()));
        }
        let pipeline = if input.format() == wgpu::TextureFormat::R32Float {
            &self.scalar
        } else {
            &self.color
        };
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("offset overlap-add"),
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
        let prepared = PreparedOffset(PreparedPass {
            pipeline: pipeline.clone(),
            common,
            buffer: buffer.clone(),
            weights: None,
            queue: self.queue.clone(),
            // PreparedPass uses operation only as an encoding label. Typed
            // offset updates below never use its legacy Pass::update dispatch.
            operation: Pass::Downsample,
            evaluation,
            source,
            groups: workgroups(evaluation)?,
            additional: None,
        });
        prepared.update(transform, boundary);
        Ok(prepared)
    }
}

/// Allocation-free encode/update boundary for direct executors and the worker.
pub struct PreparedOffset(pub(crate) PreparedPass);
impl PreparedOffset {
    pub fn update(&self, transform: OffsetTransform, boundary: Boundary) {
        update(&self.0, transform, boundary);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode(encoder);
    }
}

pub(crate) fn update(pass: &PreparedPass, transform: OffsetTransform, boundary: Boundary) {
    let mut uniform = NodeUniform::new(pass.evaluation, pass.source);
    uniform.parameters[0] = transform.overlap;
    uniform.flags[0] = u32::from(boundary == Boundary::Repeat);
    pass.queue
        .write_buffer(&pass.buffer, 0, bytemuck::bytes_of(&uniform));
}

/// No image scratch: one uniform plus its queue-write payload.
pub fn scratch_bytes(_extent: Extent) -> u64 {
    128
}
