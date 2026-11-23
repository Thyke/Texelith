use super::{gpu::with_gpu, reference, support::*};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind};
use texelith_gpu::{GpuConfig, GpuWorker, TypedSourcePixels};

#[test]
fn scalar_matches_f64_oracle_on_constants_ramps_impulses_npot_and_singletons() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, "constant"),
            (1, 17, "ramp"),
            (17, 1, "seeded"),
            (7, 5, "constant"),
            (7, 5, "ramp"),
            (17, 9, "impulse"),
            (257, 193, "seeded"),
        ] {
            let e = Extent::new(w, h).unwrap();
            let values = reference::fixture(e, kind);
            let source = height(&worker, e, values.clone());
            let values: Vec<_> = values.into_iter().map(f64::from).collect();
            let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (seed, grid, overlap) in [(0, 2, 0.1f32), (58, 4, 0.25), (u32::MAX, 8, 0.5)] {
                for boundary in [Boundary::Repeat, Boundary::Clamp] {
                    serial += 1;
                    let c = Controls {
                        seed,
                        grid,
                        overlap,
                        boundary,
                    };
                    let actual = evaluate(
                        gpu,
                        &worker,
                        request(std::slice::from_ref(&source), serial, Some(c)),
                    );
                    let expected =
                        reference::render(e, e, &values, seed, grid, f64::from(overlap), boundary);
                    compare(
                        actual[&MapKind::Height].iter().map(|p| p[0]),
                        &expected,
                        2e-5,
                    );
                    assert!(
                        actual[&MapKind::Height]
                            .iter()
                            .all(|p| p[0] >= lo - 2e-5 && p[0] <= hi + 2e-5)
                    );
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn constant_height_keeps_signed_range_and_precision_below_u16_quantization() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(9, 7).unwrap();
        for (i, value) in [-3.25f32, 1.75, 0.500001].into_iter().enumerate() {
            let source = height(&worker, e, vec![value; 63]);
            let actual = evaluate(
                gpu,
                &worker,
                request(&[source], i as u64 + 1, Some(Controls::default())),
            );
            compare(
                actual[&MapKind::Height].iter().map(|p| p[0]),
                &vec![f64::from(value); 63],
                if (0.0..=1.0).contains(&value) {
                    3e-7
                } else {
                    2e-5
                },
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn color_uses_linear_rgb_alpha_one_and_same_seeded_transform_as_scalar() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(31, 19).unwrap();
        let codes = [0, 10, 11, 64, 128, 192, 255];
        let bytes: Vec<_> = (0..31 * 19)
            .flat_map(|i| {
                [
                    codes[i % 7],
                    codes[(i * 3 + 1) % 7],
                    codes[(i * 5 + 2) % 7],
                    (i % 256) as u8,
                ]
            })
            .collect();
        let channels: [Vec<f64>; 3] = std::array::from_fn(|c| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| reference::half(reference::linear(p[c])))
                .collect()
        });
        let color = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let scalar = height(&worker, e, channels[0].iter().map(|v| *v as f32).collect());
        for (i, (seed, grid, overlap)) in [(0, 2, 0.1), (58, 4, 0.25), (u32::MAX, 8, 0.5)]
            .into_iter()
            .enumerate()
        {
            let c = Controls {
                seed,
                grid,
                overlap,
                ..Default::default()
            };
            let actual = evaluate(
                gpu,
                &worker,
                request(&[color.clone(), scalar.clone()], i as u64 + 1, Some(c)),
            );
            for (channel, values) in channels.iter().enumerate() {
                let expected = reference::render(
                    e,
                    e,
                    values,
                    seed,
                    grid,
                    f64::from(overlap),
                    Boundary::Repeat,
                );
                compare(
                    actual[&MapKind::Albedo].iter().map(|p| p[channel]),
                    &expected,
                    5e-4,
                );
            }
            compare(
                actual[&MapKind::Albedo].iter().map(|p| p[0]),
                &actual[&MapKind::Height]
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
                5e-4,
            );
            assert!(actual[&MapKind::Albedo].iter().all(|p| p[3] == 1.0));
        }
        worker.join().unwrap();
    });
}

#[test]
fn same_seed_is_bit_identical_cold_and_cached_but_seed_grid_and_overlap_edits_change_output() {
    with_gpu(|gpu| {
        let e = Extent::new(33, 19).unwrap();
        let mut baseline = None;
        for _ in 0..2 {
            let worker =
                GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default())
                    .unwrap();
            let source = height(&worker, e, reference::fixture(e, "seeded"));
            let c = Controls::default();
            let a = evaluate(
                gpu,
                &worker,
                request(std::slice::from_ref(&source), 1, Some(c)),
            );
            let before = worker.diagnostics();
            let b = evaluate(
                gpu,
                &worker,
                request(std::slice::from_ref(&source), 2, Some(c)),
            );
            assert_eq!(a, b);
            assert_eq!(before.submitted, worker.diagnostics().submitted);
            assert!(worker.diagnostics().cache_hits > before.cache_hits);
            let bits: Vec<_> = a[&MapKind::Height]
                .iter()
                .map(|p| (p[0] as f32).to_bits())
                .collect();
            if let Some(previous) = &baseline {
                assert_eq!(&bits, previous);
            }
            baseline = Some(bits);
            for (i, changed) in [
                Controls { seed: 59, ..c },
                Controls { grid: 2, ..c },
                Controls { overlap: 0.5, ..c },
            ]
            .into_iter()
            .enumerate()
            {
                let changed = evaluate(
                    gpu,
                    &worker,
                    request(std::slice::from_ref(&source), i as u64 + 3, Some(changed)),
                );
                let rms = (a[&MapKind::Height]
                    .iter()
                    .zip(&changed[&MapKind::Height])
                    .map(|(a, b)| (a[0] - b[0]).powi(2))
                    .sum::<f64>()
                    / (e.width * e.height) as f64)
                    .sqrt();
                assert!(
                    rms > 1e-3,
                    "parameter edit did not visibly change fixed fixture: RMS={rms}"
                );
            }
            worker.join().unwrap();
        }
    });
}

#[test]
fn fixed_noise_seam_rms_halves_while_retaining_half_the_interior_rms() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        for (i, (w, h)) in [(256, 256), (257, 193)].into_iter().enumerate() {
            let e = Extent::new(w, h).unwrap();
            let input = reference::fixture(e, "discontinuous_noise");
            let source = height(&worker, e, input.clone());
            let actual = evaluate(
                gpu,
                &worker,
                request(&[source], i as u64 + 1, Some(Controls::default())),
            );
            let (before, interior_before) =
                reference::seam_rms(e, &input.into_iter().map(f64::from).collect::<Vec<_>>());
            let (after, interior_after) = reference::seam_rms(
                e,
                &actual[&MapKind::Height]
                    .iter()
                    .map(|p| p[0])
                    .collect::<Vec<_>>(),
            );
            assert!(before > 0.01 && interior_before > 0.01);
            assert!(after <= before * 0.5, "boundary RMS {before} -> {after}");
            assert!(
                interior_after >= interior_before * 0.5,
                "interior RMS {interior_before} -> {interior_after}"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn structured_fixture_changes_composition_and_disabling_splat_restores_source_exactly() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(64, 48).unwrap();
        let input = reference::fixture(e, "structured");
        let source = height(&worker, e, input.clone());
        let original = evaluate(
            gpu,
            &worker,
            request(std::slice::from_ref(&source), 1, None),
        );
        assert_eq!(
            original[&MapKind::Height]
                .iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            input.iter().copied().map(f64::from).collect::<Vec<_>>()
        );
        let blended = evaluate(
            gpu,
            &worker,
            request(std::slice::from_ref(&source), 2, Some(Controls::default())),
        );
        let mae = blended[&MapKind::Height]
            .iter()
            .zip(&input)
            .map(|(a, b)| (a[0] - f64::from(*b)).abs())
            .sum::<f64>()
            / input.len() as f64;
        assert!(
            mae > 0.03,
            "fixed structured fixture must expose composition loss"
        );
        let restored = evaluate(gpu, &worker, request(&[source], 3, None));
        assert_eq!(original, restored);
        worker.join().unwrap();
    });
}

#[test]
fn scalar_proxy_samples_the_original_source_at_continuous_output_pixel_centers() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        for (i, (w, h)) in [(1025, 193), (1, 1025), (1025, 1)].into_iter().enumerate() {
            let e = Extent::new(w, h).unwrap();
            let output = texelith_gpu::proxy_extent(e).unwrap();
            assert_ne!(e, output);
            let input = reference::fixture(e, "seeded");
            let source = height(&worker, e, input.clone());
            let c = Controls::default();
            let mut r = request(&[source], i as u64 + 1, Some(c));
            r.quality = texelith_gpu::Quality::Proxy;
            r.extent = output;
            let actual = evaluate(gpu, &worker, r);
            let expected = reference::render(
                e,
                output,
                &input.into_iter().map(f64::from).collect::<Vec<_>>(),
                c.seed,
                c.grid,
                f64::from(c.overlap),
                c.boundary,
            );
            compare(
                actual[&MapKind::Height].iter().map(|p| p[0]),
                &expected,
                2e-5,
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn constant_linear_albedo_is_preserved_on_npot_and_singleton_axes() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let expected = [128, 64, 255].map(|c| reference::half(reference::linear(c)));
        for (i, (w, h)) in [(1, 1), (1, 17), (17, 1), (257, 193)]
            .into_iter()
            .enumerate()
        {
            let e = Extent::new(w, h).unwrap();
            let source = install(
                &worker,
                e,
                TypedSourcePixels::Rgba8 {
                    pixels: [128, 64, 255, 13].repeat((w * h) as usize).into(),
                    interpretation: InputInterpretation::SrgbPhoto,
                },
            );
            let actual = evaluate(
                gpu,
                &worker,
                request(&[source], i as u64 + 1, Some(Controls::default())),
            );
            for pixel in &actual[&MapKind::Albedo] {
                for c in 0..3 {
                    assert!((pixel[c] - expected[c]).abs() <= 2e-5);
                }
                assert_eq!(pixel[3], 1.0);
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn linear_rgba8_height_proxy_samples_original_red_without_area_restriction() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1025, 193).unwrap();
        let bytes: Vec<_> = (0..e.width * e.height)
            .flat_map(|i| [(i.wrapping_mul(71) % 256) as u8, 255, 128, 0])
            .collect();
        let input: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f64::from(p[0]) / 255.0)
            .collect();
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::LinearHeight,
            },
        );
        let c = Controls::default();
        let mut r = request(&[source], 1, Some(c));
        r.quality = texelith_gpu::Quality::Proxy;
        r.extent = texelith_gpu::proxy_extent(e).unwrap();
        let output = r.extent;
        let actual = evaluate(gpu, &worker, r);
        compare(
            actual[&MapKind::Height].iter().map(|p| p[0]),
            &reference::render(
                e,
                output,
                &input,
                c.seed,
                c.grid,
                f64::from(c.overlap),
                c.boundary,
            ),
            2e-5,
        );
        worker.join().unwrap();
    });
}
