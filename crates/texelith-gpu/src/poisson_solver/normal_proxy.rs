//! Fused RGB8 decoding and area restriction for imported normal proxies.
use crate::{Pass, PreparedPass, executor::common_layout};
use std::sync::Arc;
use texelith_core::{Extent, NormalConvention};

pub(crate) struct Kernel(wgpu::ComputePipeline);

impl Kernel {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = common_layout(device, wgpu::TextureFormat::Rgba16Float);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal source proxy"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/poisson_solver/normal_proxy.wgsl").into(),
            ),
        });
        Self(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("normal source proxy"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        )
    }

    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        queue: Arc<wgpu::Queue>,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        uniform: &wgpu::Buffer,
    ) -> PreparedPass {
        let evaluation = Extent {
            width: output.width(),
            height: output.height(),
        };
        let source = Extent {
            width: input.width(),
            height: input.height(),
        };
        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());
        let common = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal source proxy"),
            layout: &self.0.get_bind_group_layout(0),
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
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        PreparedPass {
            pipeline: self.0.clone(),
            common,
            buffer: uniform.clone(),
            weights: None,
            queue,
            operation: Pass::NormalSource(NormalConvention::OpenGl),
            evaluation,
            source,
            groups: [
                evaluation.width.div_ceil(8),
                evaluation.height.div_ceil(8),
                1,
            ],
            additional: None,
        }
    }
}
