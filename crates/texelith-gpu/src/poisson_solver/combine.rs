use super::*;
use crate::{NodeUniform, executor::common_layout};
use wgpu::util::DeviceExt;

pub(super) struct Kernel {
    layout: wgpu::BindGroupLayout,
    extra: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
pub(super) struct Combine {
    pipeline: wgpu::ComputePipeline,
    binding: wgpu::BindGroup,
    extra: wgpu::BindGroup,
    extent: Extent,
}
impl Kernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = common_layout(device, wgpu::TextureFormat::R32Float);
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Poisson correction"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Poisson correction"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/poisson_solver/combine.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Poisson correction"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&layout), Some(&extra)],
                    immediate_size: 0,
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            extra,
            pipeline,
        }
    }
    pub fn prepare(
        &self,
        arena: &mut Arena<'_>,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        correction: &wgpu::Texture,
        op: u32,
        boundary: Boundary,
    ) -> Result<Combine, GpuError> {
        arena.reserve(64)?;
        let extent = crate::executor::texture_extent(output)?;
        let mut node = NodeUniform::new(extent, crate::executor::texture_extent(input)?);
        node.flags = [op, u32::from(boundary == Boundary::Repeat), 0, 0];
        let uniform = arena
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poisson correction parameters"),
                contents: bytemuck::bytes_of(&node),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let a = input.create_view(&Default::default());
        let b = output.create_view(&Default::default());
        let c = correction.create_view(&Default::default());
        let binding = arena.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let extra = arena.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.extra,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&c),
            }],
        });
        Ok(Combine {
            pipeline: self.pipeline.clone(),
            binding,
            extra,
            extent,
        })
    }
}
impl Combine {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Poisson correction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.binding, &[]);
        pass.set_bind_group(1, &self.extra, &[]);
        pass.dispatch_workgroups(
            self.extent.width.div_ceil(8),
            self.extent.height.div_ceil(8),
            1,
        );
    }
}
