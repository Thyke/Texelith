use super::support::*;
/// Probe the same direction/mip sampling code used by the real viewport.
pub fn check(g: &Gpu, specular: &wgpu::Texture, data: &[f32]) -> Result<()> {
    let output = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("IBL mip interpolation oracle"),
        size: wgpu::Extent3d {
            width: 4,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let layout = g
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let source = format!(
        "{}\n{}",
        include_str!("../../../shaders/ibl/cube.wgsl"),
        r#"
@group(0) @binding(0) var input_map:texture_2d<f32>;
@group(0) @binding(1) var output_map:texture_storage_2d<rgba16float,write>;
fn mip(d:vec3<f32>,level:u32)->vec3<f32> {return cube_sample(input_map,d,128u>>level,256u-(256u>>level));}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let lod=array(0.0,0.5,3.5,7.0)[id.x];let lo=u32(floor(lod));let hi=min(lo+1u,7u);
    textureStore(output_map,vec2<i32>(i32(id.x),0),vec4(mix(mip(vec3(1.0,0.0,0.0),lo),mip(vec3(1.0,0.0,0.0),hi),fract(lod)),1.0));
}"#
    );
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = g
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &specular.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            },
        ],
    });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    g.queue.submit([encoder.finish()]);
    g.finish()?;
    let bytes = g.read(&output, 8)?;
    let actual: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let center = |m: u32, c: usize| {
        let size = 128usize >> m;
        let row = 256usize - (256usize >> m);
        let a = (size - 1) / 2;
        let b = size / 2;
        [
            data[((row + a) * 768 + a) * 4 + c],
            data[((row + a) * 768 + b) * 4 + c],
            data[((row + b) * 768 + a) * 4 + c],
            data[((row + b) * 768 + b) * 4 + c],
        ]
        .iter()
        .sum::<f32>()
            / 4.0
    };
    for (i, lod) in [0.0f32, 0.5, 3.5, 7.0].into_iter().enumerate() {
        for c in 0..3 {
            let lo = lod.floor() as u32;
            let hi = (lo + 1).min(7);
            let expected = center(lo, c) * (1.0 - lod.fract()) + center(hi, c) * lod.fract();
            require(
                (actual[i * 4 + c] - expected).abs() < 0.002,
                "roughness mip interpolation snapped or sampled wrong row",
            )?;
        }
    }
    // +X's directional component loses contrast at maximum roughness.
    require(
        center(0, 0) > center(7, 0) + 0.1,
        "specular roughness did not blur directional lighting",
    )?;
    Ok(())
}
