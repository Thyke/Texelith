use super::{
    oracle::{self, Controls, ao, compare, image},
    support::{GIB, request, with_gpu},
};
use texelith_core::{Boundary, Extent};
use texelith_gpu::{Quality, TypedSourcePixels};

#[test]
fn flat_signed_height_is_exactly_one_at_default_and_maximum_quality() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let mut id = 0;
        for (w, h, value) in [(1, 1, -2.0), (1, 19, 0.25), (19, 1, 4.0), (17, 13, 0.5)] {
            let e = Extent::new(w, h).unwrap();
            let source = gpu.install(
                &worker,
                e,
                TypedSourcePixels::HeightF32(vec![value; (w * h) as usize].into()),
            );
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                for (directions, steps, radius) in [(8, 8, 32.0), (32, 32, 512.0)] {
                    id += 1;
                    let c = Controls {
                        directions,
                        steps,
                        radius,
                        boundary,
                        ..Default::default()
                    };
                    let result = gpu.evaluate(&worker, request(&source, id, c, Quality::Full));
                    assert!(
                        result.iter().all(|&v| v.to_bits() == 1.0_f32.to_bits()),
                        "flat AO must be exactly 1"
                    );
                }
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn gpu_matches_cpu_on_ramps_impulses_seeded_npot_and_singleton_axes() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let mut id = 0;
        for (w, h, kind) in [
            (17, 11, 0),
            (31, 19, 1),
            (257, 193, 2),
            (1, 23, 0),
            (23, 1, 2),
        ] {
            let e = Extent::new(w, h).unwrap();
            let pixels = match kind {
                0 => image(e, |x, y| (x as f32 * 0.7 + y as f32 * 0.3) / 1024.0 - 0.02),
                1 => image(e, |x, y| if x == 0 && y == 0 { 0.05 } else { 0.0 }),
                _ => oracle::seeded(e),
            };
            let source = gpu.install(
                &worker,
                e,
                TypedSourcePixels::HeightF32(pixels.clone().into()),
            );
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                for (directions, steps, radius, bias) in
                    [(8, 8, 32.0, 0.01), (7, 11, 5.5, 0.0), (4, 4, 512.0, 0.07)]
                {
                    id += 1;
                    let c = Controls {
                        directions,
                        steps,
                        radius,
                        bias,
                        boundary,
                    };
                    compare(
                        &gpu.evaluate(&worker, request(&source, id, c, Quality::Full)),
                        &ao(&pixels, e, e, c),
                    );
                }
                if kind == 1 {
                    id += 1;
                    let c = Controls {
                        directions: 32,
                        steps: 32,
                        radius: 512.0,
                        boundary,
                        ..Default::default()
                    };
                    compare(
                        &gpu.evaluate(&worker, request(&source, id, c, Quality::Full)),
                        &ao(&pixels, e, e, c),
                    );
                }
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn cavity_is_darker_than_exposed_plateau_without_range_or_transfer_errors() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(33, 33).unwrap();
        let pixels = image(e, |x, y| {
            if (x as i32 - 16).abs() <= 3 && (y as i32 - 16).abs() <= 3 {
                0.0
            } else {
                0.1
            }
        });
        let c = Controls {
            radius: 12.0,
            ..Default::default()
        };
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(pixels.clone().into()),
        );
        let result = gpu.evaluate(&worker, request(&source, 1, c, Quality::Full));
        compare(&result, &ao(&pixels, e, e, c));
        assert!(result[16 * 33 + 16] < 0.9);
        assert_eq!(result[0], 1.0);
        assert!(result[16 * 33 + 16] + 0.05 < result[0]);
        worker.join().unwrap();
    });
}

#[test]
fn repeat_ao_commutes_with_wrapped_translation_and_differs_from_clamp_at_edges() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(29, 17).unwrap();
        let original = image(e, |x, y| if x == 28 && y < 9 { 0.08 } else { 0.0 });
        let translated = image(e, |x, y| {
            original[(((y + 5) % 17) * 29 + (x + 11) % 29) as usize]
        });
        let mut id = 0;
        let mut results = Vec::new();
        for (pixels, boundary) in [
            (&original, Boundary::Repeat),
            (&translated, Boundary::Repeat),
            (&original, Boundary::Clamp),
        ] {
            id += 1;
            let c = Controls {
                radius: 6.5,
                boundary,
                ..Default::default()
            };
            let source = gpu.install(
                &worker,
                e,
                TypedSourcePixels::HeightF32(pixels.clone().into()),
            );
            let result = gpu.evaluate(&worker, request(&source, id, c, Quality::Full));
            compare(&result, &ao(pixels, e, e, c));
            results.push(result);
        }
        let shifted: Vec<_> = image(e, |x, y| {
            results[0][(((y + 5) % 17) * 29 + (x + 11) % 29) as usize]
        })
        .into_iter()
        .map(f64::from)
        .collect();
        compare(&results[1], &shifted);
        assert!(
            results[0][4 * 29] + 0.01 < results[2][4 * 29],
            "Repeat must see the obstacle across the left edge"
        );
        worker.join().unwrap();
    });
}

#[test]
fn repeat_periodic_fixture_has_continuous_wrapped_edges() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(32, 24).unwrap();
        // Periodic, reflection-symmetric samples around the wrapped pixel boundary.
        let pixels = image(e, |x, y| {
            let u = std::f64::consts::TAU * (f64::from(x) + 0.5) / 32.0;
            let v = std::f64::consts::TAU * (f64::from(y) + 0.5) / 24.0;
            (0.02 * (u.cos() + v.cos())) as f32
        });
        let c = Controls {
            radius: 7.0,
            ..Default::default()
        };
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(pixels.clone().into()),
        );
        let result = gpu.evaluate(&worker, request(&source, 1, c, Quality::Full));
        compare(&result, &ao(&pixels, e, e, c));
        for y in 0..24 {
            assert!((result[y * 32] - result[y * 32 + 31]).abs() <= 2e-5);
        }
        for x in 0..32 {
            assert!((result[x] - result[23 * 32 + x]).abs() <= 2e-5);
        }
        worker.join().unwrap();
    });
}

#[test]
fn direction_step_radius_and_bias_edits_change_computation_and_invalidate_cached_output() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(31, 23).unwrap();
        let pixels = oracle::seeded(e);
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(pixels.clone().into()),
        );
        let base = Controls {
            radius: 9.5,
            ..Default::default()
        };
        let baseline = gpu.evaluate(&worker, request(&source, 1, base, Quality::Full));
        compare(&baseline, &ao(&pixels, e, e, base));
        for (i, c) in [
            Controls {
                directions: 13,
                ..base
            },
            Controls { steps: 17, ..base },
            Controls {
                radius: 3.25,
                ..base
            },
            Controls { bias: 0.3, ..base },
        ]
        .into_iter()
        .enumerate()
        {
            let result = gpu.evaluate(&worker, request(&source, i as u64 + 2, c, Quality::Full));
            compare(&result, &ao(&pixels, e, e, c));
            assert!(
                result
                    .iter()
                    .zip(&baseline)
                    .any(|(a, b)| (a - b).abs() > 1e-4),
                "ineffective control fixture: {c:?}"
            );
            if c.bias > base.bias {
                assert!(result.iter().zip(&baseline).all(|(a, b)| *a + 2e-5 >= *b));
            }
        }
        let before = worker.diagnostics();
        let a = gpu.evaluate(&worker, request(&source, 6, base, Quality::Full));
        let b = gpu.evaluate(&worker, request(&source, 7, base, Quality::Full));
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&a),
            bytemuck::cast_slice::<f32, u8>(&b)
        );
        assert_eq!(a, baseline);
        assert!(worker.diagnostics().cache_hits > before.cache_hits);
        // A new worker forces a fresh dispatch, so determinism isn't only a cache hit.
        let fresh = gpu.worker(GIB);
        let source = gpu.install(&fresh, e, TypedSourcePixels::HeightF32(pixels.into()));
        let cold = gpu.evaluate(&fresh, request(&source, 1, base, Quality::Full));
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&a),
            bytemuck::cast_slice::<f32, u8>(&cold)
        );
        fresh.join().unwrap();
        worker.join().unwrap();
    });
}

#[test]
fn linear_u16_and_f32_height_sources_preserve_small_height_differences() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(19, 13).unwrap();
        let raw: Vec<u16> = (0..247).map(|i| 32768 + (i % 19) * 61).collect();
        let pixels: Vec<f32> = raw.iter().map(|&v| f32::from(v) / 65535.0).collect();
        let c = Controls {
            radius: 3.0,
            bias: 0.0,
            ..Default::default()
        };
        let expected = ao(&pixels, e, e, c);
        assert!(
            expected.iter().any(|&v| v < 1.0 - 4e-5),
            "fixture must detect height quantization"
        );
        for (i, input) in [
            TypedSourcePixels::HeightU16(raw.into()),
            TypedSourcePixels::HeightF32(pixels.into()),
        ]
        .into_iter()
        .enumerate()
        {
            let source = gpu.install(&worker, e, input);
            compare(
                &gpu.evaluate(&worker, request(&source, i as u64 + 1, c, Quality::Full)),
                &expected,
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn radius_is_in_original_source_pixels_at_proxy_resolution_and_uses_rectangular_uv_metric() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let source_extent = Extent::new(2048, 32).unwrap();
        // Each 2x2 source cell is constant: the existing 2x proxy resampler has
        // an exact oracle, independent of its filtering choice at this ratio.
        let pixels = image(source_extent, |x, _| {
            if (x / 16).is_multiple_of(2) {
                0.0
            } else {
                0.01
            }
        });
        let source = gpu.install(
            &worker,
            source_extent,
            TypedSourcePixels::HeightF32(pixels.into()),
        );
        let e = texelith_gpu::proxy_extent(source_extent).unwrap();
        let proxy_pixels = image(e, |x, _| if (x / 8).is_multiple_of(2) { 0.0 } else { 0.01 });
        for (id, radius) in [(1, 8.0), (2, 32.0)] {
            let c = Controls {
                radius,
                ..Default::default()
            };
            let expected = ao(&proxy_pixels, e, source_extent, c);
            let wrong = ao(&proxy_pixels, e, e, c);
            assert!(
                expected
                    .iter()
                    .zip(wrong)
                    .any(|(a, b)| (a - b).abs() > 1e-3),
                "fixture must distinguish source/proxy radius"
            );
            compare(
                &gpu.evaluate(&worker, request(&source, id, c, Quality::Proxy)),
                &expected,
            );
        }
        worker.join().unwrap();
    });
}
