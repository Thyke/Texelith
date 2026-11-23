use super::support::*;
use texelith_gpu::{ibl::*, pool::AllocationDesc};
pub fn prefilter(g: &Gpu, environment: &Environment, probe: bool) -> Result<Vec<Vec<f32>>> {
    let outputs = output_allocations().map(|desc| {
        let AllocationDesc::Texture {
            extent,
            format,
            usage,
        } = desc
        else {
            unreachable!()
        };
        g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-067 isolated numerical fixture"),
            size: wgpu::Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    });
    let views = outputs
        .each_ref()
        .map(|t| t.create_view(&Default::default()));
    let source = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-067 fixture"),
        size: environment.pixels().len() as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    g.queue
        .write_buffer(&source, 0, bytemuck::cast_slice(environment.pixels()));
    let kernel = IblKernel::new(&g.device);
    for (mode, face, mip, size) in passes() {
        let e = environment.extent();
        let data: [u32; 16] = [
            mode, face, mip, size, e.width, e.height, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        g.queue
            .write_buffer(&uniform, 0, bytemuck::cast_slice(&data));
        let mut encoder = g.device.create_command_encoder(&Default::default());
        kernel
            .prepare(
                &g.device,
                &source,
                &views[if mode == 0 { 2 } else { 0 }],
                &views[mode as usize],
                &uniform,
                size,
            )
            .encode(&mut encoder);
        g.queue.submit([encoder.finish()]);
        g.finish()?;
    }
    let data: Vec<Vec<f32>> = outputs
        .iter()
        .map(|texture| {
            Ok(g.read(texture, 8)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        })
        .collect::<Result<_>>()?;
    if probe {
        super::interpolation::check(g, &outputs[2], &data[2])?;
    }
    Ok(data)
}
pub fn constant_and_direction(g: &Gpu) -> Result<()> {
    use texelith_core::Extent;
    let constant = Environment::new(
        Extent {
            width: 7,
            height: 3,
        },
        &[[0.25, 1.0, 4.0]; 21],
    )
    .map_err(|e| e.to_string())?;
    let data = prefilter(g, &constant, false)?;
    let check = |p: &[f32]| -> Result<()> {
        for (a, b) in p.iter().zip([0.25, 1.0, 4.0, 1.0]) {
            require(
                a.is_finite() && (*a - b).abs() < 0.005,
                "constant environment energy/symmetry",
            )?;
        }
        Ok(())
    };
    for image in &data[..2] {
        for p in image.as_chunks::<4>().0.iter() {
            check(p)?;
        }
    }
    for mip in 0..8 {
        let size = 128 >> mip;
        let row = 256 - (256 >> mip);
        for y in row..row + size {
            for x in 0..6 * size {
                let i = (y * 768 + x) * 4;
                check(&data[2][i..i + 4])?;
            }
        }
    }
    require(data[3].iter().all(|v| v.is_finite()), "nonfinite BRDF LUT")?;
    // For a perfectly smooth conductor at normal incidence, A+B tends to one.
    let i = (127 * 4) as usize;
    require(
        (data[3][i] + data[3][i + 1] - 1.0).abs() < 0.01,
        "smooth normal BRDF energy",
    )?;
    let (w, h) = (257, 129);
    let pixels = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let phi = ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
                let theta = (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
                [
                    0.5 + 0.5 * theta.sin() * phi.cos(),
                    0.5 + 0.5 * theta.cos(),
                    0.5 + 0.5 * theta.sin() * phi.sin(),
                ]
            })
        })
        .collect::<Vec<_>>();
    let directional = Environment::new(
        Extent {
            width: w,
            height: h,
        },
        &pixels,
    )
    .map_err(|e| e.to_string())?;
    let first = prefilter(g, &directional, true)?;
    let second = prefilter(g, &directional, false)?;
    require(first == second, "prefilter is not deterministic")?;
    for (face, axis, expected) in [
        (0, 0, 1.0),
        (1, 0, 0.0),
        (2, 1, 1.0),
        (3, 1, 0.0),
        (4, 2, 1.0),
        (5, 2, 0.0),
    ] {
        let index = (128 * 1536 + face * 256 + 128) * 4 + axis;
        require(
            (first[0][index] - expected).abs() < 0.002,
            "equirectangular cube orientation",
        )?;
    }
    // Opposite sides of +X/-Z shared boundary must agree for smooth input.
    for y in [0, 64, 128, 255] {
        for c in 0..3 {
            let a = first[0][(y * 1536 + 255) * 4 + c];
            let b = first[0][(y * 1536 + 5 * 256) * 4 + c];
            require((a - b).abs() < 0.005, "cube edge discontinuity")?;
        }
    }
    // The l=1 cosine convolution is analytic: E/pi=.5+(2/3)*.5*n.
    let center = (16 * 192 + 16) * 4;
    require(
        (first[1][center] - (0.5 + 1.0 / 3.0)).abs() < 0.025,
        "diffuse analytic convolution",
    )?;
    Ok(())
}
