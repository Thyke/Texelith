//! Public leaf API proposal: MetallicKernel::{new, prepare}, prepared.encode.
//! Numeric R32Float masks deliberately bypass the PaintMask producer. The only
//! anticipated API is isolated here; there are no test-side GPU implementations.
use super::{
    reference::{self, Combine, Controls},
    support::{Gpu, with_gpu},
};
use texelith_core::Extent;
use texelith_gpu::metallic::MetallicKernel;

fn texture(gpu: &Gpu, e: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("T-054 numeric fixture"),
        size: wgpu::Extent3d {
            width: e.width,
            height: e.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn uniform(gpu: &Gpu) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-054 caller-owned NodeUniform"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn evaluate(
    gpu: &Gpu,
    kernel: &MetallicKernel,
    e: Extent,
    colors: &[[f32; 3]],
    mask: Option<&[f32]>,
    c: Controls,
) -> Vec<f32> {
    assert_eq!(colors.len(), (e.width * e.height) as usize);
    let input = texture(gpu, e, wgpu::TextureFormat::Rgba16Float);
    // All kernel fixtures are exactly representable positive half floats. Photo
    // transfer/half rounding is covered separately through the real worker.
    let encode = |v: f32| -> u16 {
        if v == 0.0 {
            return 0;
        }
        let bits = v.to_bits();
        assert!((2.0_f32.powi(-14)..=1.0).contains(&v) && bits & 0x1fff == 0);
        (((bits >> 23) - 112) << 10 | (bits >> 13 & 1023)) as u16
    };
    let data: Vec<u16> = colors
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            [
                encode(p[0]),
                encode(p[1]),
                encode(p[2]),
                if i % 2 == 0 { 0 } else { 0x3c00 },
            ]
        })
        .collect();
    gpu.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 8),
            rows_per_image: Some(e.height),
        },
        input.size(),
    );
    let mask_texture = mask.map(|values| {
        assert_eq!(values.len(), colors.len());
        let t = texture(gpu, e, wgpu::TextureFormat::R32Float);
        gpu.queue.write_texture(
            t.as_image_copy(),
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(e.width * 4),
                rows_per_image: Some(e.height),
            },
            t.size(),
        );
        t
    });
    let output = texture(gpu, e, wgpu::TextureFormat::R32Float);
    let buffer = uniform(gpu);
    let pass = kernel
        .prepare(
            &gpu.device,
            &input,
            &output,
            mask_texture.as_ref(),
            &buffer,
            &c.params(),
        )
        .expect("public metallic node must accept explicit numeric inputs");
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    pass.encode(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    gpu.read(&output)
}

#[test]
fn absent_mask_matches_explicit_constant_one_in_every_composition_mode() {
    with_gpu(|gpu| {
        let kernel = MetallicKernel::new(&gpu.device, gpu.queue.clone());
        let e = Extent::new(7, 1).unwrap();
        let colors = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 1.0].map(|r| [r, 0.0, 0.0]);
        for (mode, expected) in [
            (Combine::Replace, vec![1.0; 7]),
            (Combine::Add, vec![1.0; 7]),
            (Combine::Subtract, vec![0.0; 7]),
        ] {
            let c = Controls {
                picked: [0.0; 3],
                radius: 0.25,
                feather: 0.5,
                mode,
                ..Default::default()
            };
            let explicit = evaluate(gpu, &kernel, e, &colors, Some(&[1.0; 7]), c);
            reference::compare(&explicit, &expected);
            let absent = evaluate(gpu, &kernel, e, &colors, None, c);
            assert_eq!(
                absent, explicit,
                "absent mask must be ConstantOne: {mode:?}"
            );
        }
    });
}

#[test]
fn exact_center_radius_feather_endpoints_and_threshold_equalities_match_literals() {
    with_gpu(|gpu| {
        let kernel = MetallicKernel::new(&gpu.device, gpu.queue.clone());
        let e = Extent::new(7, 1).unwrap();
        let colors = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75].map(|r| [r, 0.0, 0.0]);
        let base = Controls {
            picked: [0.0; 3],
            radius: 0.25,
            feather: 0.5,
            mode: Combine::Add,
            ..Default::default()
        };
        for (c, expected) in [
            (base, vec![1., 1., 1., 1., 1., 0., 0.]),
            (
                Controls {
                    threshold: 0.5001,
                    ..base
                },
                vec![1., 1., 1., 1., 0., 0., 0.],
            ),
            (
                Controls {
                    threshold: 1.0,
                    ..base
                },
                vec![1., 1., 1., 0., 0., 0., 0.],
            ),
            (
                Controls {
                    threshold: 0.0,
                    ..base
                },
                vec![1.; 7],
            ),
            (
                Controls {
                    feather: 0.0,
                    ..base
                },
                vec![1., 1., 1., 0., 0., 0., 0.],
            ),
            (
                Controls {
                    radius: 0.0,
                    feather: 0.0,
                    ..base
                },
                vec![1., 0., 0., 0., 0., 0., 0.],
            ),
        ] {
            reference::compare(
                &evaluate(gpu, &kernel, e, &colors, Some(&[0.0; 7]), c),
                &expected,
            );
        }
    });
}

#[test]
fn numeric_mask_replace_add_subtract_are_linear_clamped_and_keep_fractional_precision() {
    with_gpu(|gpu| {
        let kernel = MetallicKernel::new(&gpu.device, gpu.queue.clone());
        let e = Extent::new(10, 1).unwrap();
        let colors: Vec<_> = (0..10)
            .map(|i| if i < 5 { [0.0; 3] } else { [1.0; 3] })
            .collect();
        let mask = [
            0.0, 0.1234567, 0.5, 0.9999, 1.0, 0.0, 0.1234567, 0.5, 0.9999, 1.0,
        ];
        for mode in [Combine::Replace, Combine::Add, Combine::Subtract] {
            let c = Controls {
                picked: [0.0; 3],
                feather: 0.0,
                mode,
                ..Default::default()
            };
            let expected: Vec<_> = colors
                .iter()
                .zip(mask)
                .map(|(p, m)| reference::metallic(p.map(f64::from), Some(m), c))
                .collect();
            let actual = evaluate(gpu, &kernel, e, &colors, Some(&mask), c);
            reference::compare(&actual, &expected);
            assert!(
                actual.iter().any(|&v| v > 0.0 && v < 1.0),
                "mask must not be thresholded"
            );
            // Missing optional masks compose exactly like explicit constant one.
            reference::compare(
                &evaluate(gpu, &kernel, e, &colors, None, c),
                &[if matches!(mode, Combine::Subtract) {
                    0.0
                } else {
                    1.0
                }; 10],
            );
        }
    });
}

#[test]
fn numeric_npot_ramps_impulses_singletons_and_rgb_metric_match_independent_oracle() {
    with_gpu(|gpu| {
        let kernel = MetallicKernel::new(&gpu.device, gpu.queue.clone());
        for (w, h) in [(1, 1), (1, 31), (31, 1), (17, 13), (257, 193)] {
            let e = Extent::new(w, h).unwrap();
            let colors: Vec<_> = (0..w * h)
                .map(|i| {
                    [
                        ((i % w) % 17) as f32 / 16.0,
                        ((i / w) % 17) as f32 / 16.0,
                        if i == 0 { 1.0 } else { 0.0 },
                    ]
                })
                .collect();
            let mask: Vec<_> = (0..w * h).map(|i| (i % 101) as f32 / 100.0).collect();
            for mode in [Combine::Replace, Combine::Add, Combine::Subtract] {
                let c = Controls {
                    picked: [0.25, 0.5, 0.0],
                    radius: 0.375,
                    feather: 0.25,
                    threshold: 0.75,
                    mode,
                };
                let expected: Vec<_> = colors
                    .iter()
                    .zip(&mask)
                    .map(|(p, &m)| reference::metallic(p.map(f64::from), Some(m), c))
                    .collect();
                reference::compare(
                    &evaluate(gpu, &kernel, e, &colors, Some(&mask), c),
                    &expected,
                );
            }
        }
        // Same luminance is not the same RGB distance; all three components count.
        let c = Controls {
            picked: [0.0; 3],
            radius: 0.5,
            feather: 0.0,
            mode: Combine::Add,
            ..Default::default()
        };
        reference::compare(
            &evaluate(
                gpu,
                &kernel,
                Extent::new(5, 1).unwrap(),
                &[
                    [0.5, 0., 0.],
                    [0., 0.5, 0.],
                    [0., 0., 0.5],
                    [0.375, 0.375, 0.],
                    [1.; 3],
                ],
                Some(&[0.0; 5]),
                c,
            ),
            &[1., 1., 1., 0., 0.],
        );
        reference::compare(
            &evaluate(
                gpu,
                &kernel,
                Extent::new(2, 1).unwrap(),
                &[[0.; 3], [1.; 3]],
                Some(&[0.0; 2]),
                Controls {
                    radius: 3.0_f32.sqrt(),
                    ..c
                },
            ),
            &[1., 1.],
        );
    });
}

#[test]
fn public_prepare_rejects_aliases_wrong_formats_extents_and_invalid_parameters() {
    with_gpu(|gpu| {
        let kernel = MetallicKernel::new(&gpu.device, gpu.queue.clone());
        let e = Extent::new(17, 13).unwrap();
        let color = texture(gpu, e, wgpu::TextureFormat::Rgba16Float);
        let output = texture(gpu, e, wgpu::TextureFormat::R32Float);
        let mask = texture(gpu, e, wgpu::TextureFormat::R32Float);
        let wrong_size = texture(
            gpu,
            Extent::new(13, 17).unwrap(),
            wgpu::TextureFormat::R32Float,
        );
        let buffer = uniform(gpu);
        let c = Controls::default();
        for (input, out, painted) in [
            (&output, &output, None),
            (&color, &color, None),
            (&color, &output, Some(&output)),
            (&color, &output, Some(&color)),
            (&color, &wrong_size, None),
            (&color, &output, Some(&wrong_size)),
        ] {
            assert!(
                kernel
                    .prepare(&gpu.device, input, out, painted, &buffer, &c.params())
                    .is_err()
            );
        }
        for c in [
            Controls {
                radius: f32::NAN,
                ..c
            },
            Controls { feather: -0.1, ..c },
            Controls {
                threshold: 1.1,
                ..c
            },
            Controls {
                picked: [f32::INFINITY; 3],
                ..c
            },
        ] {
            assert!(
                kernel
                    .prepare(
                        &gpu.device,
                        &color,
                        &output,
                        Some(&mask),
                        &buffer,
                        &c.params()
                    )
                    .is_err()
            );
        }
    });
}
