//! Proposed additive PreparedRoughness API: see the T-053 handoff for signatures.
//! Explicit R32 numeric masks avoid a dependency on any paint/curvature producer.
use super::{
    gpu::{Gpu, with_gpu},
    material::IDENTITY,
    oracle,
};
use std::sync::{Arc, Mutex};
use texelith_core::{Extent, MaterialNodeParams as P};
use texelith_gpu::{
    PreparedRoughness,
    pool::{FenceId, ResourcePool},
};

const GRAY: [(u16, f64); 9] = [
    (0x0000, 0.0),
    (0x3000, 0.125),
    (0x3400, 0.25),
    (0x3600, 0.375),
    (0x3800, 0.5),
    (0x3900, 0.625),
    (0x3a00, 0.75),
    (0x3b00, 0.875),
    (0x3c00, 1.0),
];

fn color(gpu: &Gpu, extent: Extent, samples: &[(u16, f64)]) -> wgpu::Texture {
    assert_eq!(samples.len(), (extent.width * extent.height) as usize);
    let pixels: Vec<u16> = samples
        .iter()
        .flat_map(|(bits, _)| [*bits, *bits, *bits, 0x3c00])
        .collect();
    gpu.upload(
        extent,
        wgpu::TextureFormat::Rgba16Float,
        bytemuck::cast_slice(&pixels),
    )
}

fn run(gpu: &Gpu, input: &wgpu::Texture, mask: Option<&wgpu::Texture>, params: &P) -> Vec<f32> {
    let extent = Extent::new(input.width(), input.height()).unwrap();
    let output = gpu.texture(extent, wgpu::TextureFormat::R32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(4 * 1024 * 1024)));
    let mut prepared =
        PreparedRoughness::new(&gpu.executor, input, mask, &output, params, pool.clone()).unwrap();
    assert!(
        !prepared.passes().is_empty(),
        "zero GPU passes is not an implementation"
    );
    assert!(
        pool.lock().unwrap().diagnostics().allocated_bytes > 0,
        "uniforms/knots/staging must be charged"
    );
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for pass in prepared.passes() {
        pass.encode(&mut encoder);
    }
    prepared.mark_submitted(FenceId(1)).unwrap();
    gpu.queue.submit([encoder.finish()]);
    gpu.wait();
    prepared.complete(FenceId(1));
    let actual = gpu.read_scalar(&output);
    drop(prepared);
    assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
    actual
}

#[test]
fn gpu_knots_interpolate_hold_endpoints_and_support_narrow_nonzero_segments() {
    with_gpu(|gpu| {
        let samples = [
            (0x0000, 0.0),
            (0x3400, 0.25),
            (0x3600, 0.375),
            (0x3800, 0.5),
            (0x3801, 0.500_488_3),
            (0x3900, 0.625),
            (0x3a00, 0.75),
            (0x3c00, 1.0),
        ];
        let input = color(gpu, Extent::new(8, 1).unwrap(), &samples);
        let curve = [[0.25, 0.75], [0.5, 0.25], [0.500_488_3, 0.875], [0.75, 0.5]];
        let params = P::Roughness {
            curve: curve.to_vec(),
            base: 0.125,
            invert: false,
        };
        let actual = run(gpu, &input, None, &params);
        // Closed-form anchors, including the narrow segment, independent of oracle search.
        oracle::compare(
            &actual,
            &[0.75, 0.75, 0.5, 0.25, 0.875, 0.687866927592955, 0.5, 0.5],
            2e-5,
        );
    });
}

#[test]
fn gpu_numeric_mask_zero_one_fraction_and_missing_one_blend_after_curve_then_invert() {
    with_gpu(|gpu| {
        let e = Extent::new(9, 1).unwrap();
        let input = color(gpu, e, &GRAY);
        let masks = [0.0f32, 1.0, 0.25, 0.75, 0.5, 0.0, 1.0, 0.25, 0.75];
        let mask = gpu.upload(
            e,
            wgpu::TextureFormat::R32Float,
            bytemuck::cast_slice(&masks),
        );
        let ones = gpu.upload(
            e,
            wgpu::TextureFormat::R32Float,
            bytemuck::cast_slice(&[1.0f32; 9]),
        );
        let curve = [[0.0, 0.125], [0.25, 0.875], [1.0, 0.25]];
        let luma: Vec<_> = GRAY.iter().map(|(_, v)| *v).collect();
        for invert in [false, true] {
            let params = P::Roughness {
                curve: curve.to_vec(),
                base: 0.75,
                invert,
            };
            let actual = run(gpu, &input, Some(&mask), &params);
            oracle::compare(
                &actual,
                &oracle::roughness(&luma, Some(&masks), &curve, 0.75, invert),
                2e-5,
            );
            assert!((actual[0] - if invert { 0.25 } else { 0.75 }).abs() < 2e-5);
            let missing = run(gpu, &input, None, &params);
            let explicit = run(gpu, &input, Some(&ones), &params);
            assert_eq!(
                missing.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                explicit.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    });
}

#[test]
fn gpu_smoothness_is_exact_complement_and_double_inversion_recovers_dyadic_values() {
    with_gpu(|gpu| {
        let input = color(gpu, Extent::new(9, 1).unwrap(), &GRAY);
        let roughness = run(
            gpu,
            &input,
            None,
            &P::Roughness {
                curve: IDENTITY.to_vec(),
                base: 0.5,
                invert: false,
            },
        );
        let smoothness = run(
            gpu,
            &input,
            None,
            &P::Roughness {
                curve: IDENTITY.to_vec(),
                base: 0.5,
                invert: true,
            },
        );
        for ((r, s), (_, value)) in roughness.iter().zip(&smoothness).zip(GRAY) {
            assert!((f64::from(*r) - value).abs() < 2e-5);
            assert_eq!(*s, 1.0 - *r, "smoothness must be the final subtraction");
            assert!((1.0 - *s - *r).abs() <= f32::EPSILON);
        }
        // A second production inversion is expressed by the equivalent curve on
        // the same linear luminance; no new out-of-scope scalar/color adapter.
        let twice = run(
            gpu,
            &input,
            None,
            &P::Roughness {
                curve: vec![[0.0, 1.0], [1.0, 0.0]],
                base: 0.5,
                invert: true,
            },
        );
        oracle::compare(&twice, &GRAY.map(|(_, v)| v), 2e-5);
    });
}

#[test]
fn gpu_npot_and_singleton_numeric_masks_are_finite_bounded_and_bitwise_repeatable() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 17), (17, 1), (257, 193)] {
            let e = Extent::new(w, h).unwrap();
            let samples: Vec<_> = (0..w * h)
                .map(|i| GRAY[(i as usize * 7) % GRAY.len()])
                .collect();
            let input = color(gpu, e, &samples);
            let masks: Vec<_> = (0..w * h).map(|i| (i % 5) as f32 * 0.25).collect();
            let mask = gpu.upload(
                e,
                wgpu::TextureFormat::R32Float,
                bytemuck::cast_slice(&masks),
            );
            let curve: Vec<_> = (0..16)
                .map(|i| [i as f32 / 15.0, if i % 2 == 0 { 0.0 } else { 1.0 }])
                .collect();
            let params = P::Roughness {
                curve: curve.clone(),
                base: 0.375,
                invert: true,
            };
            let a = run(gpu, &input, Some(&mask), &params);
            let b = run(gpu, &input, Some(&mask), &params);
            oracle::compare(
                &a,
                &oracle::roughness(
                    &samples.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                    Some(&masks),
                    &curve,
                    0.375,
                    true,
                ),
                2e-5,
            );
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    });
}

#[test]
fn prepared_pass_rejects_invalid_params_formats_extents_usages_alias_and_atomic_budget_failure() {
    with_gpu(|gpu| {
        let e = Extent::new(9, 1).unwrap();
        let input = color(gpu, e, &GRAY);
        let output = gpu.texture(e, wgpu::TextureFormat::R32Float);
        let mask = gpu.texture(e, wgpu::TextureFormat::R32Float);
        let wrong_extent = gpu.texture(Extent::new(1, 9).unwrap(), wgpu::TextureFormat::R32Float);
        let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
        let valid = P::Roughness {
            curve: IDENTITY.to_vec(),
            base: 0.5,
            invert: false,
        };
        for (read, mask, write) in [
            (&output, None, &output),
            (&input, None, &input),
            (&input, Some(&mask), &mask),
            (&input, Some(&wrong_extent), &output),
            (&input, Some(&input), &output),
            (&input, None, &wrong_extent),
        ] {
            assert!(
                PreparedRoughness::new(&gpu.executor, read, mask, write, &valid, pool.clone())
                    .is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        let restricted = |format, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("T-053 invalid usage"),
                size: input.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let no_read = restricted(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::COPY_DST,
        );
        let no_write = restricted(
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let mask_no_read = restricted(wgpu::TextureFormat::R32Float, wgpu::TextureUsages::COPY_DST);
        for (read, mask, write) in [
            (&no_read, None, &output),
            (&input, None, &no_write),
            (&input, Some(&mask_no_read), &output),
        ] {
            assert!(
                PreparedRoughness::new(&gpu.executor, read, mask, write, &valid, pool.clone())
                    .is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        let mut invalid = vec![
            P::Roughness {
                curve: IDENTITY.to_vec(),
                base: f32::NAN,
                invert: false,
            },
            P::Roughness {
                curve: IDENTITY.to_vec(),
                base: 1.1,
                invert: false,
            },
            P::Roughness {
                curve: IDENTITY.to_vec(),
                base: -0.1,
                invert: false,
            },
            P::Roughness {
                curve: vec![[0.0, 0.0]],
                base: 0.5,
                invert: false,
            },
            P::Roughness {
                curve: vec![[0.0, 0.0]; 17],
                base: 0.5,
                invert: false,
            },
            P::Roughness {
                curve: vec![[0.5, 0.0], [0.5, 1.0]],
                base: 0.5,
                invert: false,
            },
            P::Roughness {
                curve: vec![[1.0, 0.0], [0.0, 1.0]],
                base: 0.5,
                invert: false,
            },
        ];
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            invalid.push(P::Roughness {
                curve: vec![[0.0, 0.0], [value, 1.0]],
                base: 0.5,
                invert: false,
            });
            invalid.push(P::Roughness {
                curve: vec![[0.0, value], [1.0, 1.0]],
                base: 0.5,
                invert: false,
            });
        }
        for params in invalid {
            assert!(
                PreparedRoughness::new(&gpu.executor, &input, None, &output, &params, pool.clone())
                    .is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
        let tiny = Arc::new(Mutex::new(ResourcePool::new(1)));
        assert!(
            PreparedRoughness::new(&gpu.executor, &input, None, &output, &valid, tiny.clone())
                .is_err()
        );
        assert_eq!(tiny.lock().unwrap().diagnostics().allocated_bytes, 0);
        let mut active =
            PreparedRoughness::new(&gpu.executor, &input, None, &output, &valid, pool.clone())
                .unwrap();
        active.mark_submitted(FenceId(5)).unwrap();
        let charged = pool.lock().unwrap().diagnostics().allocated_bytes;
        assert!(charged > 0);
        drop(active);
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, charged);
        pool.lock().unwrap().complete(FenceId(4));
        assert_eq!(pool.lock().unwrap().diagnostics().retired_bytes, charged);
        pool.lock().unwrap().complete(FenceId(5));
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
    });
}

#[test]
fn prepared_rejects_mask_arrays_and_mip_chains_before_pool_admission() {
    with_gpu(|gpu| {
        let e = Extent::new(9, 1).unwrap();
        let input = color(gpu, e, &GRAY);
        let output = gpu.texture(e, wgpu::TextureFormat::R32Float);
        let params = P::Roughness {
            curve: IDENTITY.to_vec(),
            base: 0.5,
            invert: false,
        };
        let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024)));
        for (layers, mips) in [(2, 1), (1, 2)] {
            let mask = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("T-053 invalid mask shape"),
                size: wgpu::Extent3d {
                    width: e.width,
                    height: e.height,
                    depth_or_array_layers: layers,
                },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            assert!(
                PreparedRoughness::new(
                    &gpu.executor,
                    &input,
                    Some(&mask),
                    &output,
                    &params,
                    pool.clone(),
                )
                .is_err()
            );
            assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        }
    });
}
