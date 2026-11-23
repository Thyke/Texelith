use super::{gpu::with_gpu, reference as r, support::*};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind};
use texelith_gpu::{GpuConfig, GpuWorker, Quality, TypedSourcePixels};

#[test]
fn scalar_oracle_covers_both_bases_strengths_boundaries_npot_and_singleton_axes() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, "constant"),
            (1, 17, "ramp"),
            (17, 1, "seeded"),
            (7, 5, "ramp"),
            (17, 9, "impulse"),
            (257, 193, "seeded"),
        ] {
            let e = Extent::new(w, h).unwrap();
            let values = r::fixture(e, kind);
            let source = height(&worker, e, values.clone());
            let rgb: Vec<_> = values.iter().map(|v| [f64::from(*v); 3]).collect();
            for base in [
                BASES[0],
                BASES[1],
                Base::Offset(0.1),
                Base::Offset(0.5),
                Base::Splat {
                    seed: 0,
                    grid: 2,
                    overlap: 0.1,
                },
                Base::Splat {
                    seed: u32::MAX,
                    grid: 8,
                    overlap: 0.5,
                },
            ] {
                for boundary in [Boundary::Repeat, Boundary::Clamp] {
                    for strength in [0.0, 0.375, 1.0] {
                        serial += 1;
                        let c = Controls {
                            base,
                            boundary,
                            strength,
                        };
                        let actual = evaluate(
                            g,
                            &worker,
                            request(std::slice::from_ref(&source), serial, Some(c)),
                        );
                        let expected = r::render(
                            e,
                            e,
                            &rgb,
                            &base.core(),
                            boundary,
                            f64::from(strength),
                            false,
                        );
                        compare(
                            actual[&MapKind::Height].iter().map(|p| p[0]),
                            &expected.iter().map(|p| p[0]).collect::<Vec<_>>(),
                            2e-5,
                        );
                    }
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn strength_zero_equals_selected_base_and_midpoint_interpolates_reconstructed_luminance() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(64, 48).unwrap();
        let source = height(&worker, e, r::fixture(e, "seeded"));
        for (i, base) in BASES.into_iter().enumerate() {
            let mut serial = i as u64 * 10 + 1;
            let mut run = |strength| {
                serial += 1;
                evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            strength,
                            ..Default::default()
                        }),
                    ),
                )
            };
            let zero = run(0.0);
            let mid = run(0.5);
            let one = run(1.0);
            let plain = evaluate(
                g,
                &worker,
                request_method(
                    std::slice::from_ref(&source),
                    serial + 1,
                    Some((base.method(), Boundary::Repeat)),
                ),
            );
            compare(
                zero[&MapKind::Height].iter().map(|p| p[0]),
                &plain[&MapKind::Height]
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
                2e-5,
            );
            let expected: Vec<_> = zero[&MapKind::Height]
                .iter()
                .zip(&one[&MapKind::Height])
                .map(|(a, b)| (a[0] + b[0]) * 0.5)
                .collect();
            compare(mid[&MapKind::Height].iter().map(|p| p[0]), &expected, 2e-5);
            let change = zero[&MapKind::Height]
                .iter()
                .zip(&one[&MapKind::Height])
                .map(|(a, b)| (a[0] - b[0]).abs())
                .sum::<f64>()
                / (e.width * e.height) as f64;
            assert!(
                change > 0.005,
                "strength=1 must execute histogram reconstruction, change={change}"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn flat_low_variance_and_near_black_stay_finite_and_preserve_f32_height_precision() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(19, 7).unwrap();
        let mut serial = 0;
        for value in [0.0f32, 1e-12, 0.500001, 1.0, -3.25, 1.75] {
            let source = height(&worker, e, vec![value; (e.width * e.height) as usize]);
            for base in BASES {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            ..Default::default()
                        }),
                    ),
                );
                compare(
                    actual[&MapKind::Height].iter().map(|p| p[0]),
                    &vec![f64::from(value); (e.width * e.height) as usize],
                    3e-7,
                );
            }
        }
        // Variance is nonzero but below 1e-6: still use the exact base blend.
        for amplitude in [1e-8f32, 0.0005] {
            let values: Vec<_> = r::fixture(e, "seeded")
                .into_iter()
                .map(|v| 0.2 + amplitude * v)
                .collect();
            let source = height(&worker, e, values);
            for base in BASES {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            ..Default::default()
                        }),
                    ),
                );
                serial += 1;
                let expected = evaluate(
                    g,
                    &worker,
                    request_method(
                        std::slice::from_ref(&source),
                        serial,
                        Some((base.method(), Boundary::Repeat)),
                    ),
                );
                compare(
                    actual[&MapKind::Height].iter().map(|p| p[0]),
                    &expected[&MapKind::Height]
                        .iter()
                        .map(|p| p[0])
                        .collect::<Vec<_>>(),
                    3e-7,
                );
            }
        }
        // A larger, convenient fallback threshold must not swallow non-flat data.
        let input: Vec<_> = r::fixture(e, "seeded")
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i % 2 == 0 {
                    0.5f32 - 0.0011
                } else {
                    0.5f32 + 0.0011
                }
            })
            .collect();
        let rgb: Vec<_> = input.iter().map(|v| [f64::from(*v); 3]).collect();
        assert!(
            r::Histogram::new(&input.iter().copied().map(f64::from).collect::<Vec<_>>()).variance
                > 1e-6
        );
        let source = height(&worker, e, input);
        for base in BASES {
            serial += 1;
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    serial,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            let expected = r::render(e, e, &rgb, &base.core(), Boundary::Repeat, 1.0, false);
            compare(
                actual[&MapKind::Height].iter().map(|p| p[0]),
                &expected.iter().map(|p| p[0]).collect::<Vec<_>>(),
                2e-5,
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn linear_luminance_color_oracle_preserves_chroma_handles_black_and_clamps_sdr() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(31, 19).unwrap();
        let codes = [0, 10, 11, 64, 128, 192, 255];
        let bytes: Vec<_> = (0..e.width * e.height)
            .flat_map(|i| {
                [
                    codes[i as usize % 7],
                    codes[(i as usize * 3 + 1) % 7],
                    codes[(i as usize * 5 + 2) % 7],
                    (i % 256) as u8,
                ]
            })
            .collect();
        let rgb: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| std::array::from_fn(|c| r::half(r::linear(p[c]))))
            .collect();
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let mut serial = 0;
        for base in BASES {
            for strength in [0.0, 0.375, 1.0] {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            strength,
                            ..Default::default()
                        }),
                    ),
                );
                let expected = r::render(
                    e,
                    e,
                    &rgb,
                    &base.core(),
                    Boundary::Repeat,
                    f64::from(strength),
                    true,
                );
                for c in 0..3 {
                    compare(
                        actual[&MapKind::Albedo].iter().map(|p| p[c]),
                        &expected.iter().map(|p| p[c]).collect::<Vec<_>>(),
                        5e-4,
                    );
                }
                assert!(
                    actual[&MapKind::Albedo]
                        .iter()
                        .all(|p| p[3] == 1.0 && p[..3].iter().all(|v| (0.0..=1.0).contains(v)))
                );
            }
        }
        for code in [[0, 0, 0, 0], [1, 0, 2, 255], [128, 64, 255, 13]] {
            let source = install(
                &worker,
                e,
                TypedSourcePixels::Rgba8 {
                    pixels: code.repeat((e.width * e.height) as usize).into(),
                    interpretation: InputInterpretation::SrgbPhoto,
                },
            );
            for base in BASES {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            ..Default::default()
                        }),
                    ),
                );
                for p in &actual[&MapKind::Albedo] {
                    for c in 0..3 {
                        assert!((p[c] - r::half(r::linear(code[c]))).abs() <= 2e-5);
                    }
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn fixed_1024_square_noise_preserves_luminance_cdf_within_point_zero_five() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1024, 1024).unwrap();
        let values = r::fixture(e, "noise");
        let source = height(&worker, e, values.clone());
        let original: Vec<_> = values.into_iter().map(f64::from).collect();
        for (i, base) in BASES.into_iter().enumerate() {
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    i as u64 + 1,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            let result: Vec<_> = actual[&MapKind::Height].iter().map(|p| p[0]).collect();
            let distance = r::cdf_distance(&original, &result);
            assert!(distance <= 0.05, "{base:?} CDF max distance {distance}");
            assert!(result.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        // Repeat the luminance gate through the actual color conversion path;
        // success on an R32 height input alone does not validate linear albedo.
        let codes: Vec<u8> = r::fixture(e, "noise")
            .iter()
            .map(|v| (v * 255.0).round() as u8)
            .collect();
        let bytes: Vec<_> = codes
            .iter()
            .enumerate()
            .flat_map(|(i, v)| [*v, *v, *v, (i % 256) as u8])
            .collect();
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let decoded: [f64; 256] = std::array::from_fn(|i| r::half(r::linear(i as u8)));
        let original: Vec<_> = codes.iter().map(|v| decoded[*v as usize]).collect();
        for (i, base) in BASES.into_iter().enumerate() {
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    i as u64 + 3,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            let luma: Vec<_> = actual[&MapKind::Albedo]
                .iter()
                .map(|p| r::luminance([p[0], p[1], p[2]]))
                .collect();
            let distance = r::cdf_distance(&original, &luma);
            assert!(
                distance <= 0.05,
                "linear albedo {base:?}: CDF distance {distance}"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn cold_warm_determinism_and_content_semantic_and_parameter_lut_invalidation() {
    with_gpu(|g| {
        let e = Extent::new(37, 29).unwrap();
        let input = r::fixture(e, "seeded");
        let mut previous = None;
        for _ in 0..2 {
            let worker =
                GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
            let a = height(&worker, e, input.clone());
            let c = Controls::default();
            let first = evaluate(g, &worker, request(std::slice::from_ref(&a), 1, Some(c)));
            if let Some(previous) = &previous {
                assert_eq!(previous, &first);
            }
            previous = Some(first.clone());
            let diagnostics = worker.diagnostics();
            let warm = evaluate(g, &worker, request(std::slice::from_ref(&a), 2, Some(c)));
            assert_eq!(first, warm);
            assert_eq!(diagnostics.submitted, worker.diagnostics().submitted);
            assert!(worker.diagnostics().cache_hits > diagnostics.cache_hits);
            let changed: Vec<_> = input.iter().map(|v| 0.1 + 0.4 * v).collect();
            let b = height(&worker, e, changed.clone());
            let actual = evaluate(g, &worker, request(&[b], 3, Some(c)));
            let expected = r::render(
                e,
                e,
                &changed
                    .iter()
                    .map(|v| [f64::from(*v); 3])
                    .collect::<Vec<_>>(),
                &c.base.core(),
                c.boundary,
                1.0,
                false,
            );
            compare(
                actual[&MapKind::Height].iter().map(|p| p[0]),
                &expected.iter().map(|p| p[0]).collect::<Vec<_>>(),
                2e-5,
            );
            assert_ne!(
                first, actual,
                "changing only source content must invalidate source LUT"
            );
            let restored = evaluate(g, &worker, request(&[a], 4, Some(c)));
            assert_eq!(
                restored, first,
                "returning to original content must not reuse changed LUT"
            );
            // Identical uploaded bytes interpreted as color versus height have
            // different luminance, transfer function and source cache identities.
            let bytes: Vec<_> = (0..e.width * e.height)
                .flat_map(|i| [(i % 256) as u8, ((i * 17) % 256) as u8, 255, 11])
                .collect();
            for (i, interpretation) in [
                InputInterpretation::LinearHeight,
                InputInterpretation::SrgbPhoto,
            ]
            .into_iter()
            .enumerate()
            {
                let source = install(
                    &worker,
                    e,
                    TypedSourcePixels::Rgba8 {
                        pixels: bytes.clone().into(),
                        interpretation,
                    },
                );
                let color = interpretation == InputInterpretation::SrgbPhoto;
                let rgb: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| {
                        if color {
                            std::array::from_fn(|c| r::half(r::linear(p[c])))
                        } else {
                            [f64::from(p[0]) / 255.0; 3]
                        }
                    })
                    .collect();
                let actual = evaluate(g, &worker, request(&[source], 5 + i as u64, Some(c)));
                let expected = r::render(e, e, &rgb, &c.base.core(), c.boundary, 1.0, color);
                let kind = if color {
                    MapKind::Albedo
                } else {
                    MapKind::Height
                };
                for channel in 0..if color { 3 } else { 1 } {
                    compare(
                        actual[&kind].iter().map(|p| p[channel]),
                        &expected.iter().map(|p| p[channel]).collect::<Vec<_>>(),
                        if color { 5e-4 } else { 2e-5 },
                    );
                }
            }
            worker.join().unwrap();
        }
    });
}

#[test]
fn proxies_build_histogram_from_original_content_and_sample_continuous_source_coordinates() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let mut serial = 0;
        for (w, h) in [(1025, 193), (1, 1025), (1025, 1)] {
            let e = Extent::new(w, h).unwrap();
            let input = r::fixture(e, "seeded");
            let source = height(&worker, e, input.clone());
            let rgb: Vec<_> = input.iter().map(|v| [f64::from(*v); 3]).collect();
            for base in BASES {
                serial += 1;
                let mut request = request(
                    std::slice::from_ref(&source),
                    serial,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                );
                request.extent = texelith_gpu::proxy_extent(e).unwrap();
                request.quality = Quality::Proxy;
                let out = request.extent;
                let actual = evaluate(g, &worker, request);
                let expected = r::render(e, out, &rgb, &base.core(), Boundary::Repeat, 1.0, false);
                compare(
                    actual[&MapKind::Height].iter().map(|p| p[0]),
                    &expected.iter().map(|p| p[0]).collect::<Vec<_>>(),
                    2e-5,
                );
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn histogram_reduces_fixed_noise_wrap_rms_and_retains_half_the_interior_detail() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(257, 193).unwrap();
        let input = r::fixture(e, "discontinuous_noise");
        let source = height(&worker, e, input.clone());
        let (before, interior) =
            r::seam_rms(e, &input.into_iter().map(f64::from).collect::<Vec<_>>());
        for (i, base) in BASES.into_iter().enumerate() {
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    i as u64 + 1,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            let output: Vec<_> = actual[&MapKind::Height].iter().map(|p| p[0]).collect();
            let (after, detail) = r::seam_rms(e, &output);
            assert!(before > 0.01 && interior > 0.01);
            assert!(
                after <= before * 0.5,
                "{base:?}: boundary {before} -> {after}"
            );
            assert!(
                detail >= interior * 0.5,
                "{base:?}: interior {interior} -> {detail}"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn independent_base_weights_match_existing_apis_and_histogram_field_is_periodic() {
    let e = Extent::new(31, 19).unwrap();
    let values: Vec<_> = r::fixture(e, "seeded")
        .iter()
        .map(|v| [f64::from(*v); 3])
        .collect();
    let histogram = r::Histogram::new(&values.iter().map(|v| v[0]).collect::<Vec<_>>());
    for base in BASES {
        for uv in [
            [0.0, 0.0],
            [0.25, 0.5],
            [0.125, 0.375],
            [0.999, 0.001],
            [-0.25, 2.5],
        ] {
            let actual: Vec<_> = match base {
                Base::Offset(overlap) => {
                    texelith_gpu::seamless_offset::OffsetTransform::new(overlap)
                        .unwrap()
                        .samples(uv.map(|v| v as f32))
                        .iter()
                        .map(|t| (t.uv, t.weight))
                        .collect()
                }
                Base::Splat {
                    seed,
                    grid,
                    overlap,
                } => texelith_gpu::seamless_splat::SplatTransform::new(seed, grid, overlap)
                    .unwrap()
                    .samples(uv.map(|v| v as f32))
                    .iter()
                    .map(|t| (t.uv, t.weight))
                    .collect(),
            };
            let expected = r::taps(uv, &base.core());
            assert_eq!(actual.len(), expected.len());
            assert!(expected.iter().map(|t| t.weight).sum::<f64>() > 1e-8);
            for ((coord, weight), tap) in actual.iter().zip(expected) {
                assert!((f64::from(*weight) - tap.weight).abs() < 2e-5);
                for (a, b) in coord.iter().zip(tap.uv) {
                    assert!((f64::from(*a) - b).abs() < 2e-5);
                }
            }
            let at = |uv| {
                r::at(
                    e,
                    &values,
                    &histogram,
                    uv,
                    &base.core(),
                    Boundary::Repeat,
                    1.0,
                    false,
                )
            };
            for shift in [[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]] {
                assert!((at(uv)[0] - at([uv[0] + shift[0], uv[1] + shift[1]])[0]).abs() < 2e-5);
            }
        }
    }
}
