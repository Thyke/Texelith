//! Render each channel through the production fragment shader. The reference
//! has constant maps evaluated analytically at the ray/constant-height hit, so
//! it neither marches a ray nor shares the implementation's sampling code.
use super::support::*;
use texelith_core::{Extent, PreviewMesh};
use texelith_gpu::viewport_shared::{ViewportBindings, ViewportRenderer};
use wgpu::util::DeviceExt;
fn value(channel: usize, uv: [f32; 2], varied: bool) -> [f32; 4] {
    let f = 0.75 * uv[0] + 0.25 * uv[1];
    match channel {
        0 => {
            if varied {
                [f, 0.3, 0.1, 1.]
            } else {
                [0.4, 0.3, 0.1, 1.]
            }
        }
        1 => {
            if varied {
                [uv[0] - 0.5, uv[1] - 0.5, 1., 1.]
            } else {
                [0., 0., 1., 1.]
            }
        }
        2 => [if varied { f } else { 0.45 }, 0., 0., 1.],
        3 => [if varied { f } else { 0.2 }, 0., 0., 1.],
        4 => [if varied { f } else { 1. }, 0., 0., 1.],
        _ => [0.8, 0., 0., 1.],
    }
}
fn texture(
    g: &Gpu,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("T-068 channel oracle outside app ledger"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
fn image(g: &Gpu, w: u32, h: u32, data: &[[f32; 4]]) -> wgpu::Texture {
    let t = texture(
        g,
        w,
        h,
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    g.queue.write_texture(
        t.as_image_copy(),
        bytemuck::cast_slice(data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 16),
            rows_per_image: Some(h),
        },
        t.size(),
    );
    t
}
pub fn check(g: &Gpu) -> Result<()> {
    let renderer = ViewportRenderer::new(&g.device);
    let mesh = texelith_gpu::pbr::primitive(&PreviewMesh::Plane).map_err(|e| e.to_string())?;
    let vertices: Vec<[f32; 12]> = mesh
        .vertices
        .iter()
        .map(|v| {
            [
                v.position[0],
                v.position[1],
                v.position[2],
                v.normal[0],
                v.normal[1],
                v.normal[2],
                v.uv[0],
                v.uv[1],
                v.tangent[0],
                v.tangent[1],
                v.tangent[2],
                v.tangent[3],
            ]
        })
        .collect();
    let buffer = |data: &[u8], usage| {
        g.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            })
    };
    let vb = buffer(bytemuck::cast_slice(&vertices), wgpu::BufferUsages::VERTEX);
    let ib = buffer(
        bytemuck::cast_slice(&mesh.indices),
        wgpu::BufferUsages::INDEX,
    );
    let scene: [[f32; 4]; 4] = [
        [0.7, 0.3, 3., 1.],
        [0.; 4],
        [0.7, -0.2, 0.6855655, 2.],
        [0.; 4],
    ];
    let scene = buffer(bytemuck::cast_slice(&scene), wgpu::BufferUsages::UNIFORM);
    let identity: [[f32; 4]; 4] = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    let object = buffer(bytemuck::cast_slice(&identity), wgpu::BufferUsages::UNIFORM);
    // Distinct mip radiance makes roughness observable under IBL too. A black
    // specular atlas would make its negative control depend on direct light.
    let mut specular = vec![[0.; 4]; 768 * 255];
    for mip in 0..8 {
        let row = 256 - (256 >> mip);
        let radiance = 4.0 * (1.0 - mip as f32 / 7.0);
        for y in row..row + (128 >> mip) {
            specular[y * 768..(y + 1) * 768].fill([radiance, radiance, radiance, 1.]);
        }
    }
    let env = [
        image(g, 192, 32, &vec![[1.; 4]; 192 * 32]),
        image(g, 768, 255, &specular),
        image(g, 128, 128, &vec![[1., 0., 0., 1.]; 128 * 128]),
    ];
    let env_views = env.each_ref().map(|t| t.create_view(&Default::default()));
    let extent = Extent::new(129, 129).map_err(|e| e.to_string())?;
    let hdr = texture(
        g,
        129,
        129,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth = texture(
        g,
        129,
        129,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );
    let display = texture(
        g,
        129,
        129,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let hdr = hdr.create_view(&Default::default());
    let depth = depth.create_view(&Default::default());
    let display_view = display.create_view(&Default::default());
    let hit = [
        0.5 + 0.7 * 0.3 * 0.7f32.tan(),
        0.5 - 0.7 * 0.3 * 0.3f32.tan() / 0.7f32.cos(),
    ];
    for ibl in [false, true] {
        for channel in 0..=5 {
            let mut colors = Vec::new();
            for (case, reference) in [false, true, false].into_iter().enumerate() {
                let side = if reference { 1 } else { 64 };
                let maps: Vec<_> = (0..6)
                    .map(|kind| {
                        let mut data = Vec::new();
                        for y in 0..side {
                            for x in 0..side {
                                let uv = if reference {
                                    hit
                                } else {
                                    [
                                        (x as f32 + 0.5) / side as f32,
                                        (y as f32 + 0.5) / side as f32,
                                    ]
                                };
                                data.push(value(kind, uv, channel == 5 || channel == kind));
                            }
                        }
                        image(g, side, side, &data)
                    })
                    .collect();
                let views: Vec<_> = maps
                    .iter()
                    .map(|m| m.create_view(&Default::default()))
                    .collect();
                let uniform: [[f32; 4]; 4] = [
                    [1., 1., 1., -1.],
                    [-1., 1., 0., 0.],
                    [if reference || case == 2 { 0. } else { 0.7 }, 0.5, 0., 0.],
                    [0.; 4],
                ];
                let material = buffer(bytemuck::cast_slice(&uniform), wgpu::BufferUsages::UNIFORM);
                let bindings = ViewportBindings {
                    hdr: &hdr,
                    depth: &depth,
                    display: &display_view,
                    material: std::array::from_fn(|i| &views[i]),
                    scene_uniform: &scene,
                    material_uniform: &material,
                    object_uniform: &object,
                    extent,
                };
                let prepared = if ibl {
                    renderer.prepare_pbr_ibl(
                        &g.device,
                        bindings,
                        (&vb, &ib, mesh.indices.len() as u32),
                        env_views.each_ref(),
                    )
                } else {
                    renderer.prepare_pbr(&g.device, bindings, &vb, &ib, mesh.indices.len() as u32)
                };
                let mut encoder = g.device.create_command_encoder(&Default::default());
                prepared.encode(&mut encoder);
                g.queue.submit([encoder.finish()]);
                let bytes = g.read_display(&display)?;
                let center = (64 * 129 + 64) * 4;
                colors.push(bytes[center..center + 4].to_vec());
            }
            require(
                colors[0]
                    .iter()
                    .zip(&colors[1])
                    .all(|(a, b)| a.abs_diff(*b) <= 1),
                &format!(
                    "channel {channel} uses inconsistent displaced UV (IBL={ibl}): {colors:?}"
                ),
            )?;
            if channel != 4 || ibl {
                require(
                    colors[2]
                        .iter()
                        .zip(&colors[1])
                        .any(|(a, b)| a.abs_diff(*b) > 1),
                    &format!("channel {channel} oracle cannot reject undisplaced UV (IBL={ibl})"),
                )?;
            }
        }
    }
    Ok(())
}
