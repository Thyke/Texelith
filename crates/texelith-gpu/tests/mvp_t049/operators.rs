use super::{
    gpu::{compare, extent, with_gpu},
    oracle::{self, Image},
};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, Extent, NormalConvention};
use texelith_gpu::{
    Axis, Pass, PoissonGrid, PoissonOperators, PoissonPass as P, pool::ResourcePool,
};

const TOL: f64 = 2e-5;
fn grid(w: usize, h: usize, boundary: Boundary) -> PoissonGrid {
    PoissonGrid::new(Extent::new(w as u32, h as u32).unwrap(), boundary).unwrap()
}
fn scalar() -> Extent {
    Extent::new(1, 1).unwrap()
}

#[test]
fn hierarchy_ceil_halves_to_singleton_and_rejects_untrusted_extents() {
    for (w, h) in [(257, 193), (1, 17), (17, 1), (1, 1), (4096, 4096)] {
        let mut g = grid(w, h, Boundary::Repeat);
        let (mut w, mut h) = (w, h);
        let mut count = 1;
        while w > 1 || h > 1 {
            g = g.coarsen().unwrap();
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            assert_eq!(g.extent(), Extent::new(w as u32, h as u32).unwrap());
            count += 1;
        }
        assert!(g.coarsen().is_err(), "no duplicate terminal levels");
        assert!(count <= 13);
    }
    for (w, h) in [(0, 1), (1, 0), (4097, 1), (u32::MAX, u32::MAX)] {
        assert!(
            PoissonGrid::new(
                Extent {
                    width: w,
                    height: h
                },
                Boundary::Repeat
            )
            .is_err()
        );
    }
}

#[test]
fn forward_and_transpose_match_incidence_oracle_on_npot_ramps_impulses_and_singletons() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 17), (17, 1), (7, 5), (257, 193)] {
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                let repeat = boundary == Boundary::Repeat;
                let g = grid(w, h, boundary);
                // Small amplitudes keep UV-scaled second derivatives within the
                // ADR absolute scalar tolerance even at 257x193.
                let mut impulse = Image::constant(w, h, 0.0);
                impulse.v[0] = 2.0f64.powi(-20);
                let ramp = Image::new(
                    w,
                    h,
                    (0..w * h)
                        .map(|i| (i % w + 2 * (i / w)) as f64 * 2.0f64.powi(-20))
                        .collect(),
                );
                for image in [Image::seeded(w, h, 49, 2.0f64.powi(-20)), ramp, impulse] {
                    let input = gpu.scalar(&image);
                    let u = gpu.run(&g, P::Difference(Axis::X), &[&input], extent(&image));
                    let v = gpu.run(&g, P::Difference(Axis::Y), &[&input], extent(&image));
                    compare(&gpu.read(&u), &image.difference(repeat, false), TOL);
                    compare(&gpu.read(&v), &image.difference(repeat, true), TOL);
                    let adjoint = gpu.run(&g, P::Transpose, &[&u, &v], extent(&image));
                    let a = gpu.run(&g, P::Laplacian, &[&input], extent(&image));
                    compare(&gpu.read(&adjoint), &image.apply(repeat), TOL);
                    compare(&gpu.read(&a), &image.apply(repeat), TOL);
                }
            }
        }
    });
}

#[test]
fn adjoint_dot_product_relative_error_and_positive_energy() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 19), (19, 1), (31, 17), (257, 193)] {
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                let g = grid(w, h, boundary);
                let x = Image::seeded(w, h, 49, 1.0);
                let u = Image::seeded(w, h, 51, 1.0);
                let v = Image::seeded(w, h, 53, 1.0);
                let (xt, ut, vt) = (gpu.scalar(&x), gpu.scalar(&u), gpu.scalar(&v));
                let dx = gpu.read(&gpu.run(&g, P::Difference(Axis::X), &[&xt], extent(&x)));
                let dy = gpu.read(&gpu.run(&g, P::Difference(Axis::Y), &[&xt], extent(&x)));
                let dt = gpu.read(&gpu.run(&g, P::Transpose, &[&ut, &vt], extent(&x)));
                let (lhs, rhs) = (dx.dot(&u) + dy.dot(&v), x.dot(&dt));
                let scale = lhs.abs().max(rhs.abs()).max(1e-12);
                assert!(
                    (lhs - rhs).abs() / scale <= TOL,
                    "adjoint {w}x{h} {boundary:?}: {lhs} != {rhs}"
                );
                let ax = gpu.read(&gpu.run(&g, P::Laplacian, &[&xt], extent(&x)));
                assert!(x.dot(&ax) >= -TOL);
                if w * h > 1 {
                    assert!(
                        x.dot(&ax) > 0.0,
                        "zero operator cannot satisfy the contract"
                    );
                }
            }
        }
    });
}

#[test]
fn neumann_ignores_outward_flux_including_poisoned_singleton_axes() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 17), (17, 1), (9, 7)] {
            let g = grid(w, h, Boundary::Clamp);
            let mut u = Image::constant(w, h, 0.0);
            let mut v = u.clone();
            for y in 0..h {
                u.v[y * w + w - 1] = 123.0;
            }
            for x in 0..w {
                v.v[(h - 1) * w + x] = -456.0;
            }
            let (ut, vt) = (gpu.scalar(&u), gpu.scalar(&v));
            let output = gpu.run(&g, P::Transpose, &[&ut, &vt], extent(&u));
            compare(&gpu.read(&output), &Image::constant(w, h, 0.0), 0.0);
        }
    });
}

#[test]
fn constant_nullspace_and_periodic_sinusoid_have_analytic_eigenvalues() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 17), (17, 1), (32, 16), (257, 193)] {
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                let g = grid(w, h, boundary);
                let x = Image::constant(w, h, -3.25);
                let input = gpu.scalar(&x);
                compare(
                    &gpu.read(&gpu.run(&g, P::Laplacian, &[&input], extent(&x))),
                    &Image::constant(w, h, 0.0),
                    TOL,
                );
            }
            let tau = std::f64::consts::TAU;
            let scale = 1.0 / (w * w + h * h) as f64;
            let x = Image::new(
                w,
                h,
                (0..w * h)
                    .map(|i| {
                        scale
                            * ((tau * (i % w) as f64 / w as f64).sin()
                                + (tau * (i / w) as f64 / h as f64).sin())
                    })
                    .collect(),
            );
            let expected = Image::new(
                w,
                h,
                (0..w * h)
                    .map(|i| {
                        scale
                            * (4.0
                                * (w * w) as f64
                                * (std::f64::consts::PI / w as f64).sin().powi(2)
                                * (tau * (i % w) as f64 / w as f64).sin()
                                + 4.0
                                    * (h * h) as f64
                                    * (std::f64::consts::PI / h as f64).sin().powi(2)
                                    * (tau * (i / w) as f64 / h as f64).sin())
                    })
                    .collect(),
            );
            let input = gpu.scalar(&x);
            compare(
                &gpu.read(&gpu.run(
                    &grid(w, h, Boundary::Repeat),
                    P::Laplacian,
                    &[&input],
                    extent(&x),
                )),
                &expected,
                TOL,
            );
        }
    });
}

#[test]
fn odd_restriction_and_bilinear_prolongation_match_pixel_center_oracles() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 17), (17, 1), (5, 3), (257, 193)] {
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                let g = grid(w, h, boundary);
                let fine = Image::seeded(w, h, 49, 1.0);
                let input = gpu.scalar(&fine);
                let expected = fine.restrict();
                let coarse = gpu.run(&g, P::Restrict, &[&input], extent(&expected));
                compare(&gpu.read(&coarse), &expected, TOL);
                let lifted = gpu.run(&g, P::Prolong, &[&coarse], extent(&fine));
                compare(
                    &gpu.read(&lifted),
                    &expected.prolong(w, h, boundary == Boundary::Repeat),
                    TOL,
                );
                let constant = gpu.scalar(&Image::constant(expected.w, expected.h, -2.25));
                compare(
                    &gpu.read(&gpu.run(&g, P::Prolong, &[&constant], extent(&fine))),
                    &Image::constant(w, h, -2.25),
                    TOL,
                );
            }
        }
    });
}

#[test]
fn coarse_operator_is_recursive_rap_on_257_by_193_and_preserves_nullspace() {
    with_gpu(|gpu| {
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            let mut g = grid(257, 193, boundary);
            let mut levels: Vec<(usize, usize)> = vec![(257, 193)];
            for _ in 0..3 {
                let (w, h) = *levels.last().unwrap();
                levels.push((w.div_ceil(2), h.div_ceil(2)));
                g = g.coarsen().unwrap();
                let (w, h) = *levels.last().unwrap();
                let x = Image::seeded(w, h, 49, 2.0f64.powi(-20));
                let expected = oracle::galerkin(&x, &levels, boundary == Boundary::Repeat);
                let input = gpu.scalar(&x);
                let actual = gpu.read(&gpu.run(&g, P::Laplacian, &[&input], extent(&x)));
                compare(&actual, &expected, TOL);
                assert!(expected.norm() > 1e-5, "ineffective Galerkin fixture");
                // A ceil-half re-discretization is observably different on odd grids.
                assert!(expected.sub(&x.apply(boundary == Boundary::Repeat)).norm() > 1e-4);
                let constant = gpu.scalar(&Image::constant(w, h, 2.0));
                compare(
                    &gpu.read(&gpu.run(&g, P::Laplacian, &[&constant], extent(&x))),
                    &Image::constant(w, h, 0.0),
                    TOL,
                );
            }
        }
    });
}

#[test]
fn gpu_reductions_mean_projection_residual_and_l2_are_not_rms_or_max_norm() {
    with_gpu(|gpu| {
        let g = grid(2, 1, Boundary::Repeat);
        let anchor = gpu.scalar(&Image::new(2, 1, vec![3.0, 4.0]));
        let norm = gpu.read(&gpu.run(&g, P::L2Norm, &[&anchor], scalar()));
        assert!((norm.v[0] - 5.0).abs() <= TOL);
        for (w, h) in [(1, 1), (1, 17), (17, 1), (7, 5), (257, 193)] {
            let g = grid(w, h, Boundary::Repeat);
            let constant = Image::constant(w, h, -2.25);
            let ct = gpu.scalar(&constant);
            let cm = gpu.run(&g, P::Mean, &[&ct], scalar());
            assert_eq!(gpu.read(&cm).v[0], -2.25);
            let zero = gpu.run(&g, P::ProjectMean, &[&ct, &cm], extent(&constant));
            compare(&gpu.read(&zero), &Image::constant(w, h, 0.0), 0.0);
            assert_eq!(
                gpu.read(&gpu.run(&g, P::L2Norm, &[&zero], scalar())).v[0],
                0.0
            );
            let x = Image::seeded(w, h, 49, 0.25);
            let input = gpu.scalar(&x);
            let mean = gpu.run(&g, P::Mean, &[&input], scalar());
            assert!((gpu.read(&mean).v[0] - x.mean()).abs() <= TOL);
            let projection = gpu.run(&g, P::ProjectMean, &[&input, &mean], extent(&x));
            compare(&gpu.read(&projection), &x.project(), TOL);
            assert!(gpu.read(&projection).mean().abs() <= TOL);
            let rhs = Image::seeded(w, h, 51, 2.0f64.powi(-20));
            let h0 = Image::seeded(w, h, 53, 2.0f64.powi(-20));
            let (bt, ht) = (gpu.scalar(&rhs), gpu.scalar(&h0));
            let expected = rhs.sub(&h0.apply(true));
            let residual = gpu.run(&g, P::Residual, &[&ht, &bt], extent(&x));
            compare(&gpu.read(&residual), &expected, TOL);
            let l2 = gpu.read(&gpu.run(&g, P::L2Norm, &[&residual], scalar())).v[0];
            assert!((l2 - expected.norm()).abs() <= TOL.max(expected.norm() * TOL));
        }
    });
}

#[test]
fn weighted_jacobi_uses_true_fine_and_galerkin_diagonals_and_zero_diagonal_is_finite() {
    with_gpu(|gpu| {
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            for levels in [
                vec![(1, 1)],
                vec![(1, 7)],
                vec![(7, 1)],
                vec![(7, 5)],
                vec![(7, 5), (4, 3)],
                vec![(7, 5), (4, 3), (2, 2)],
            ] {
                let mut g = grid(levels[0].0, levels[0].1, boundary);
                for _ in 1..levels.len() {
                    g = g.coarsen().unwrap();
                }
                let (w, h) = *levels.last().unwrap();
                let x = Image::seeded(w, h, 49, 0.25);
                let rhs = Image::seeded(w, h, 51, 0.25);
                let (xt, bt) = (gpu.scalar(&x), gpu.scalar(&rhs));
                let actual =
                    gpu.read(&gpu.run(&g, P::Jacobi { omega: 2.0 / 3.0 }, &[&xt, &bt], extent(&x)));
                compare(
                    &actual,
                    &oracle::jacobi(&x, &rhs, &levels, boundary == Boundary::Repeat, 2.0 / 3.0),
                    TOL,
                );
            }
        }
    });
}

#[test]
fn imported_normals_are_linear_unit_length_flip_y_once_and_report_clamped_pixels_once() {
    with_gpu(|gpu| {
        let codes = [
            [128, 128, 255, 0],
            [200, 90, 240, 255],
            [255, 255, 128, 7],
            [255, 0, 0, 77],
            [128, 128, 134, 23],
            [20, 240, 200, 111],
        ];
        let e = Extent::new(codes.len() as u32, 1).unwrap();
        let raw = gpu.texture(e, wgpu::TextureFormat::Rgba8Unorm);
        gpu.upload(&raw, &codes.into_iter().flatten().collect::<Vec<u8>>(), 4);
        let normal = gpu.texture(e, wgpu::TextureFormat::Rgba16Float);
        let prepared = gpu
            .executor
            .prepare(
                Pass::NormalSource(NormalConvention::OpenGl),
                &raw,
                &normal,
                e,
            )
            .unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        gpu.wait();
        let normals: Vec<[f64; 3]> = gpu
            .read_bytes(&normal)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| {
                std::array::from_fn(|c| {
                    super::abi::decode_half(u16::from_le_bytes([b[2 * c], b[2 * c + 1]]))
                })
            })
            .collect();
        for (n, code) in normals.iter().zip(codes) {
            let signed = [code[0], code[1], code[2]].map(|c| f64::from(c) / 255.0 * 2.0 - 1.0);
            let len = signed.iter().map(|v| v * v).sum::<f64>().sqrt();
            for c in 0..3 {
                assert!((n[c] - signed[c] / len).abs() <= 0.002);
            }
            assert!((n.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() <= 0.002);
        }
        let g = grid(codes.len(), 1, Boundary::Repeat);
        let mut outputs = Vec::new();
        for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
            let (u, v, count) =
                oracle::normal_gradients(&normals, convention == NormalConvention::DirectX);
            let du = gpu.run(
                &g,
                P::NormalGradient {
                    axis: Axis::X,
                    convention,
                },
                &[&normal],
                e,
            );
            let dv = gpu.run(
                &g,
                P::NormalGradient {
                    axis: Axis::Y,
                    convention,
                },
                &[&normal],
                e,
            );
            compare(&gpu.read(&du), &Image::new(codes.len(), 1, u), TOL);
            compare(&gpu.read(&dv), &Image::new(codes.len(), 1, v), TOL);
            let clamped = gpu
                .read(&gpu.run(&g, P::ClampCount, &[&normal], scalar()))
                .v[0];
            assert_eq!(
                count, 2,
                "fixture includes one double-axis clamp and one back-facing normal"
            );
            assert_eq!(clamped, count as f64);
            outputs.push((gpu.read(&du), gpu.read(&dv)));
        }
        compare(&outputs[0].0, &outputs[1].0, 0.0);
        for (gl, dx) in outputs[0].1.v.iter().zip(&outputs[1].1.v) {
            assert_eq!(*gl, -dx);
        }
    });
}

#[test]
fn exact_flat_and_grazing_signed_normals_have_zero_gradient_and_bounded_slopes() {
    with_gpu(|gpu| {
        let g = grid(4, 1, Boundary::Repeat);
        let e = Extent::new(4, 1).unwrap();
        // Exactly representable unit vectors: flat, +X, -X, back-facing Z.
        let halves: [[u16; 4]; 4] = [
            [0, 0, 0x3c00, 0x3c00],
            [0x3c00, 0, 0, 0x3c00],
            [0xbc00, 0, 0, 0x3c00],
            [0, 0, 0xbc00, 0x3c00],
        ];
        let normal = gpu.texture(e, wgpu::TextureFormat::Rgba16Float);
        let bytes: Vec<u8> = halves
            .into_iter()
            .flatten()
            .flat_map(u16::to_le_bytes)
            .collect();
        gpu.upload(&normal, &bytes, 8);
        let u = gpu.run(
            &g,
            P::NormalGradient {
                axis: Axis::X,
                convention: NormalConvention::OpenGl,
            },
            &[&normal],
            e,
        );
        let v = gpu.run(
            &g,
            P::NormalGradient {
                axis: Axis::Y,
                convention: NormalConvention::OpenGl,
            },
            &[&normal],
            e,
        );
        compare(
            &gpu.read(&u),
            &Image::new(4, 1, vec![0.0, -20.0, 20.0, 0.0]),
            TOL,
        );
        compare(&gpu.read(&v), &Image::constant(4, 1, 0.0), 0.0);
        assert_eq!(
            gpu.read(&gpu.run(&g, P::ClampCount, &[&normal], scalar()))
                .v[0],
            3.0
        );
    });
}

#[test]
fn prepare_rejects_aliases_formats_arity_extents_and_invalid_omega_before_dispatch() {
    with_gpu(|gpu| {
        let g = grid(7, 5, Boundary::Repeat);
        let input = gpu.scalar(&Image::constant(7, 5, 0.0));
        let out = gpu.texture(Extent::new(7, 5).unwrap(), wgpu::TextureFormat::R32Float);
        let wrong = gpu.texture(Extent::new(7, 5).unwrap(), wgpu::TextureFormat::Rgba16Float);
        let half = gpu.texture(Extent::new(3, 2).unwrap(), wgpu::TextureFormat::R32Float);
        for (op, inputs, target) in [
            (P::Laplacian, vec![&input], &input),
            (P::Laplacian, vec![&wrong], &out),
            (P::Laplacian, vec![&input], &wrong),
            (P::Laplacian, vec![], &out),
            (P::Transpose, vec![&input], &out),
            (P::Laplacian, vec![&input, &input], &out),
            (P::Restrict, vec![&input], &half),
            (P::Mean, vec![&input], &out),
        ] {
            assert!(gpu.operators.prepare(&g, op, &inputs, target).is_err());
        }
        for omega in [f32::NAN, f32::INFINITY, -0.1, 0.0, 1.01] {
            assert!(
                gpu.operators
                    .prepare(&g, P::Jacobi { omega }, &[&input, &input], &out)
                    .is_err()
            );
        }
    });
}

#[test]
fn preparation_is_non_submitting_deterministic_and_cancellable_without_scratch_leaks() {
    with_gpu(|gpu| {
        let g = grid(31, 17, Boundary::Repeat);
        let image = Image::seeded(31, 17, 49, 2.0f64.powi(-20));
        let input = gpu.scalar(&image);
        let out = gpu.scalar(&Image::constant(31, 17, -19.0));
        let before = gpu.read_bytes(&out);
        let pool = Arc::new(Mutex::new(ResourcePool::new(201_326_592)));
        let operators = PoissonOperators::new(gpu.device.clone(), pool.clone()).unwrap();
        let baseline = pool.lock().unwrap().diagnostics().allocated_bytes;
        let pass = operators
            .prepare(&g, P::Laplacian, &[&input], &out)
            .unwrap();
        gpu.wait();
        assert_eq!(gpu.read_bytes(&out), before, "prepare must not submit work");
        drop(pass); // Cancellation before encoding: no output mutation/publication.
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, baseline);
        let a = gpu.run(&g, P::Laplacian, &[&input], extent(&image));
        let b = gpu.run(&g, P::Laplacian, &[&input], extent(&image));
        assert_eq!(
            gpu.read_bytes(&a),
            gpu.read_bytes(&b),
            "same input/params must give identical bytes"
        );
        let low = Arc::new(Mutex::new(ResourcePool::new(0)));
        let rejected = PoissonOperators::new(gpu.device.clone(), low.clone())
            .and_then(|op| op.prepare(&g, P::Laplacian, &[&input], &out));
        let error = rejected
            .err()
            .expect("uniform/reduction scratch cannot bypass admission");
        assert!(
            error.to_string().to_ascii_lowercase().contains("budget"),
            "wrong rejection: {error}"
        );
        assert_eq!(low.lock().unwrap().diagnostics().allocated_bytes, 0);
        assert_eq!(gpu.read_bytes(&out), before);
    });
}
