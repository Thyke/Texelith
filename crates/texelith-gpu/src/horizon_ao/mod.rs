//! Height horizon AO. Preparation computes source-space sample geometry once;
//! encoding reuses bindings and never allocates, submits or waits.
use crate::{GpuError, NodeUniform, executor::common_layout, workgroups};
use bytemuck::{Pod, Zeroable};
use std::{num::NonZeroU64, sync::Arc};
use texelith_core::{Boundary, Extent, MaterialNodeParams};

#[derive(Clone, Copy, Debug)]
pub struct AoParameters {
    pub directions: u32,
    pub steps: u32,
    pub radius_source_px: f32,
    pub bias: f32,
    pub boundary: Boundary,
}
impl Default for AoParameters {
    fn default() -> Self {
        Self {
            directions: 8,
            steps: 8,
            radius_source_px: 32.0,
            bias: 0.01,
            boundary: Boundary::Repeat,
        }
    }
}
impl AoParameters {
    pub fn validate(self) -> Result<(), GpuError> {
        MaterialNodeParams::HorizonAo {
            directions: self.directions,
            steps: self.steps,
            radius_source_px: self.radius_source_px,
            bias: self.bias,
            boundary: self.boundary,
        }
        .validate()?;
        Ok(())
    }
    pub fn sample_bytes(self) -> Result<u64, GpuError> {
        self.validate()?;
        Ok(u64::from(self.directions) * u64::from(self.steps) * 32)
    }
    /// Uniform, sample geometry and their queue-write staging; no textures.
    pub fn scratch_bytes(self) -> Result<u64, GpuError> {
        Ok(2 * (64 + self.sample_bytes()?))
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Sample {
    // Integer pixel offset, inverse UV distance, padding.
    geometry: [f32; 4],
    weights: [f32; 4],
}
const _: () = {
    assert!(std::mem::size_of::<Sample>() == 32);
    assert!(std::mem::align_of::<Sample>() == 16);
};

/// All payload buffers are supplied by the caller's admission ledger.
pub struct AoResources<'a> {
    pub input: &'a wgpu::Texture,
    pub output: &'a wgpu::Texture,
    pub uniform: &'a wgpu::Buffer,
    pub samples: &'a wgpu::Buffer,
}

pub struct HorizonAoKernel {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::ComputePipeline,
    directions: u32,
    steps: u32,
}
impl HorizonAoKernel {
    /// Compile lazily on the worker; quality counts specialize the loop bounds.
    pub fn new(device: Arc<wgpu::Device>, directions: u32, steps: u32) -> Result<Self, GpuError> {
        AoParameters {
            directions,
            steps,
            ..Default::default()
        }
        .validate()?;
        let common = common_layout(&device, wgpu::TextureFormat::R32Float);
        let samples = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AO source-space samples"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(32 * u64::from(directions * steps)),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("height horizon AO"),
            bind_group_layouts: &[Some(&common), Some(&samples)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("height horizon AO"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "const SAMPLE_COUNT: u32 = {}u;\n{}",
                    directions * steps,
                    include_str!("../../shaders/horizon_ao/ao.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("height horizon AO"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("DIRECTIONS", f64::from(directions)),
                    ("STEPS", f64::from(steps)),
                ],
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self {
            device,
            pipeline,
            directions,
            steps,
        })
    }

    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        resources: AoResources<'_>,
        source: Extent,
        params: AoParameters,
    ) -> Result<PreparedHorizonAo, GpuError> {
        params.validate()?;
        source.validate()?;
        let AoResources {
            input,
            output,
            uniform,
            samples,
        } = resources;
        let extent = Extent::new(input.width(), input.height())?;
        if self.directions != params.directions
            || self.steps != params.steps
            || input == output
            || input.size() != output.size()
            || input.format() != wgpu::TextureFormat::R32Float
            || output.format() != wgpu::TextureFormat::R32Float
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !output
                .usage()
                .contains(wgpu::TextureUsages::STORAGE_BINDING)
            || uniform.size() < 64
            || samples.size() < params.sample_bytes()?
        {
            return Err(GpuError(
                "invalid AO resources or quality specialization".into(),
            ));
        }
        let mut geometry = Vec::with_capacity((params.directions * params.steps) as usize);
        for d in 0..params.directions {
            let angle = std::f64::consts::TAU * f64::from(d) / f64::from(params.directions);
            // Cardinal directions are exact by definition. Avoid tiny libm
            // residuals turning a single texel sample into four texture loads.
            let (cos, sin) = if (d * 4).is_multiple_of(params.directions) {
                match d * 4 / params.directions {
                    0 => (1.0, 0.0),
                    1 => (0.0, 1.0),
                    2 => (-1.0, 0.0),
                    _ => (0.0, -1.0),
                }
            } else {
                (angle.cos(), angle.sin())
            };
            for k in 1..=params.steps {
                let distance =
                    f64::from(params.radius_source_px) * f64::from(k) / f64::from(params.steps);
                let du = cos * distance / f64::from(source.width);
                let dv = sin * distance / f64::from(source.height);
                // Keep fractional offsets separate from absolute pixel positions:
                // adding small fractions to 4K coordinates loses useful precision.
                let x = du * f64::from(extent.width);
                let y = dv * f64::from(extent.height);
                let tx = x - x.floor();
                let ty = y - y.floor();
                geometry.push(Sample {
                    geometry: [
                        x.floor() as f32,
                        y.floor() as f32,
                        (1.0 / du.hypot(dv)) as f32,
                        0.0,
                    ],
                    weights: [
                        ((1.0 - tx) * (1.0 - ty)) as f32,
                        (tx * (1.0 - ty)) as f32,
                        ((1.0 - tx) * ty) as f32,
                        (tx * ty) as f32,
                    ],
                });
            }
        }
        let mut u = NodeUniform::new(extent, source);
        u.parameters[0] = params.bias;
        u.parameters[1] = (f64::from(params.radius_source_px) * f64::from(extent.width)
            / f64::from(source.width))
        .ceil() as f32
            + 1.0;
        u.parameters[2] = (f64::from(params.radius_source_px) * f64::from(extent.height)
            / f64::from(source.height))
        .ceil() as f32
            + 1.0;
        u.flags[0] = u32::from(params.boundary == Boundary::Repeat);
        queue.write_buffer(uniform, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(samples, 0, bytemuck::cast_slice(&geometry));
        let iv = input.create_view(&Default::default());
        let ov = output.create_view(&Default::default());
        let common = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AO common"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let extra = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AO samples"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: samples.as_entire_binding(),
            }],
        });
        Ok(PreparedHorizonAo {
            pipeline: self.pipeline.clone(),
            common,
            extra,
            groups: workgroups(extent)?,
        })
    }
}

pub struct PreparedHorizonAo {
    pipeline: wgpu::ComputePipeline,
    common: wgpu::BindGroup,
    extra: wgpu::BindGroup,
    groups: [u32; 3],
}
impl PreparedHorizonAo {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("height horizon AO"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.common, &[]);
        pass.set_bind_group(1, &self.extra, &[]);
        pass.dispatch_workgroups(self.groups[0], self.groups[1], self.groups[2]);
    }
}
