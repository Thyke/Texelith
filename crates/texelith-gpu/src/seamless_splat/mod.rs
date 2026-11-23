//! Seeded periodic lattice splats for correlated linear color and scalar source layers.
//!
//! All layers share wrapped cell assignments, quarter turns and compact windows.
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

/// Source-independent patch assignments reusable across correlated layers.
#[derive(Clone, Copy, Debug)]
pub struct SplatTransform {
    seed: u32,
    grid: u32,
    overlap: f32,
}
impl SplatTransform {
    pub fn new(seed: u32, grid: u32, overlap: f32) -> Result<Self, GpuError> {
        if !(2..=8).contains(&grid) || !overlap.is_finite() || !(0.1..=0.5).contains(&overlap) {
            return Err(GpuError(
                "splat grid must be in [2, 8], overlap finite in [0.1, 0.5]".into(),
            ));
        }
        Ok(Self {
            seed,
            grid,
            overlap,
        })
    }

    /// Nine torus replicas in stable row-major order; callers supply finite UVs.
    /// Weights are unnormalized and have compact C1 support in cell units.
    pub fn samples(&self, uv: [f32; 2]) -> [WeightedSample; 9] {
        let p = uv.map(|v| v.rem_euclid(1.0) * self.grid as f32);
        let cell = p.map(|v| v.floor() as i32);
        let window = |d: f32| {
            let t = (1.0 - d.abs() / (1.0 + self.overlap)).clamp(0.0, 1.0);
            let w = t * t * (3.0 - 2.0 * t);
            let w2 = w * w;
            w2 * w2
        };
        std::array::from_fn(|i| {
            let c = [cell[0] + i as i32 % 3 - 1, cell[1] + i as i32 / 3 - 1];
            let d = [p[0] - c[0] as f32 - 0.5, p[1] - c[1] as f32 - 0.5];
            let x = c[0].rem_euclid(self.grid as i32) as u32;
            let y = c[1].rem_euclid(self.grid as i32) as u32;
            let mut h = self.seed ^ x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7feb_352d);
            h ^= h >> 15;
            h = h.wrapping_mul(0x846c_a68b);
            h ^= h >> 16;
            let translation = [
                (h & 0x7fff) as f32 / 32768.0,
                ((h >> 15) & 0x7fff) as f32 / 32768.0,
            ];
            let rotated = match h >> 30 {
                0 => d,
                1 => [-d[1], d[0]],
                2 => [-d[0], -d[1]],
                _ => [d[1], -d[0]],
            };
            WeightedSample {
                uv: std::array::from_fn(|axis| {
                    (translation[axis] + rotated[axis] / self.grid as f32).rem_euclid(1.0)
                }),
                weight: window(d[0]) * window(d[1]),
            }
        })
    }
}

/// Format-specific pipelines, created on the compute worker and reused.
pub struct SplatKernels {
    scalar: wgpu::ComputePipeline,
    color: wgpu::ComputePipeline,
    queue: Arc<wgpu::Queue>,
}

impl SplatKernels {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self {
        let pipeline = |format, storage| {
            let common = executor::common_layout(device, format);
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("splat overlap-add"),
                bind_group_layouts: &[Some(&common)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("splat overlap-add"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/seamless_splat/splat.wgsl")
                        .replace("OUTPUT_FORMAT", storage)
                        .into(),
                ),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("splat overlap-add"),
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

    /// Prepare only when resource bindings change. Scalar sources use R32Float
    /// or raw linear RGBA8 (red is height); color uses linear Rgba16Float. Uniform storage is supplied
    /// by the caller's admission ledger, including its 64-byte write staging.
    /// Samples the continuous source field at output pixel centers, allowing
    /// correlated layers to share a transform at reduced evaluation extents.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        buffer: &wgpu::Buffer,
        transform: SplatTransform,
        boundary: Boundary,
    ) -> Result<PreparedSplat, GpuError> {
        let source = executor::texture_extent(input)?;
        let evaluation = executor::texture_extent(output)?;
        if input == output
            || !(matches!(
                output.format(),
                wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rgba16Float
            ) && (input.format() == output.format()
                || (input.format() == wgpu::TextureFormat::Rgba8Unorm
                    && output.format() == wgpu::TextureFormat::R32Float)))
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || buffer.size() < 64
            || !buffer
                .usage()
                .contains(wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
        {
            return Err(GpuError("invalid splat textures or uniform buffer".into()));
        }
        let pipeline = if output.format() == wgpu::TextureFormat::R32Float {
            &self.scalar
        } else {
            &self.color
        };
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat overlap-add"),
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
        let prepared = PreparedSplat(PreparedPass {
            pipeline: pipeline.clone(),
            common,
            buffer: buffer.clone(),
            weights: None,
            queue: self.queue.clone(),
            // PreparedPass uses operation only as an encoding label. Typed
            // splat updates below never use its legacy Pass::update dispatch.
            operation: if input.format() == wgpu::TextureFormat::Rgba8Unorm {
                Pass::Source(texelith_core::InputInterpretation::LinearHeight)
            } else {
                Pass::Downsample
            },
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
pub struct PreparedSplat(pub(crate) PreparedPass);
impl PreparedSplat {
    pub fn update(&self, transform: SplatTransform, boundary: Boundary) {
        update(&self.0, transform, boundary);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.0.encode(encoder);
    }
}

pub(crate) fn update(pass: &PreparedPass, transform: SplatTransform, boundary: Boundary) {
    let mut uniform = NodeUniform::new(pass.evaluation, pass.source);
    uniform.parameters[0] = transform.overlap;
    uniform.flags[1] = transform.seed;
    uniform.flags[2] = transform.grid;
    uniform.flags[3] = u32::from(matches!(pass.operation, Pass::Source(_)));
    uniform.flags[0] = u32::from(boundary == Boundary::Repeat);
    pass.queue
        .write_buffer(&pass.buffer, 0, bytemuck::bytes_of(&uniform));
}

/// No image scratch: one uniform plus its queue-write payload.
pub fn scratch_bytes(_extent: Extent) -> u64 {
    128
}
