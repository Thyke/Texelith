use super::*;
pub(super) struct LightingPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    background: wgpu::RenderPipeline,
}
impl LightingPipeline {
    pub(super) fn new(
        device: &wgpu::Device,
        material: &wgpu::BindGroupLayout,
        object: &wgpu::BindGroupLayout,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T-067 scene environment"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                texture_entry(2, wgpu::ShaderStages::FRAGMENT),
                texture_entry(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let source = super::material_shader(true);
        let pipeline = crate::pbr::pipeline_with_source(device, &layout, material, object, &source);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("environment background layout"),
            bind_group_layouts: &[Some(&layout), Some(material), Some(object)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("environment background"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let background = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("environment background"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("sky_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("sky_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            layout,
            pipeline,
            background,
        }
    }
}
impl ViewportRenderer {
    /// Environment views are admitted irradiance, specular atlas and BRDF LUT.
    pub fn prepare_pbr_ibl(
        &self,
        device: &wgpu::Device,
        bindings: ViewportBindings<'_>,
        mesh: (&wgpu::Buffer, &wgpu::Buffer, u32),
        environment: [&wgpu::TextureView; 3],
    ) -> PreparedPreview {
        let lighting = self.ibl.get_or_init(|| {
            LightingPipeline::new(device, &self.material_layout, &self.object_layout)
        });
        let scene = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("T-067 lit environment"),
            layout: &lighting.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bindings.scene_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(environment[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(environment[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(environment[2]),
                },
            ],
        });
        let mut prepared = self.prepare_pbr(device, bindings, mesh.0, mesh.1, mesh.2);
        prepared.render_pipeline = lighting.pipeline.clone();
        prepared.scene = scene;
        prepared.background = Some(lighting.background.clone());
        prepared
    }
}
