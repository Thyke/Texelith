pub(crate) fn pipeline(
    device: &wgpu::Device,
    scene: &wgpu::BindGroupLayout,
    material: &wgpu::BindGroupLayout,
    object: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    pipeline_with_source(
        device,
        scene,
        material,
        object,
        &crate::viewport_shared::material_shader(false),
    )
}
pub(crate) fn pipeline_with_source(
    device: &wgpu::Device,
    scene: &wgpu::BindGroupLayout,
    material: &wgpu::BindGroupLayout,
    object: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("T-065 PBR groups 0/1/2"),
        bind_group_layouts: &[Some(scene), Some(material), Some(object)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("T-065 direct GGX"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32x2,3=>Float32x4];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("T-065 direct PBR"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 48,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ATTRIBUTES,
            })],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
pub(crate) fn display_pipeline(
    device: &wgpu::Device,
    bindings: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("T-065 tone/transfer layout"),
        bind_group_layouts: &[Some(bindings)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("T-065 tone/exact sRGB"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pbr/display.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("T-065 HDR display"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
