use super::{
    oracle::{self as cpu, Controls, compare, encode, image},
    support::{GIB, PORTS, Planes, WITH_NORMAL, request, with_gpu},
};
use texelith_core::{Boundary, Extent, MapKind, NormalConvention, OutputPort};

fn check(planes: &Planes, e: Extent, convention: NormalConvention, c: Controls) {
    let expected = cpu::curvature(planes.normal.as_ref().unwrap(), e, convention, c);
    for (port, reference) in PORTS.iter().zip(&expected) {
        compare(&planes.scalar[&port.map_kind()], reference);
    }
    for (&edge, &cavity) in planes.scalar[&MapKind::Edge]
        .iter()
        .zip(&planes.scalar[&MapKind::Cavity])
    {
        assert!(
            edge == 0.0 || cavity == 0.0,
            "convex and concave cannot overlap"
        );
    }
}

#[test]
fn constant_imported_normals_produce_exact_zero_including_singleton_axes() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let mut id = 0;
        for (w, h) in [(1, 1), (1, 19), (23, 1), (17, 13)] {
            let e = Extent::new(w, h).unwrap();
            // Includes a steep flat plane: curvature is independent of its tilt.
            for n in [[0.0, 0.0, 1.0], [0.8, -0.6, 0.00001]] {
                let rgb = encode(&vec![n; (w * h) as usize], NormalConvention::OpenGl);
                let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
                for boundary in [Boundary::Clamp, Boundary::Repeat] {
                    id += 1;
                    let c = Controls {
                        boundary,
                        ..Default::default()
                    };
                    let planes = gpu.evaluate(
                        &worker,
                        request(&source, id, c, NormalConvention::OpenGl, &WITH_NORMAL),
                    );
                    check(&planes, e, NormalConvention::OpenGl, c);
                    assert!(planes.scalar.values().flatten().all(|&v| v == 0.0));
                }
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn analytic_bump_is_positive_bowl_negative_and_outputs_remain_separate() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(33, 25).unwrap();
        let c = Controls {
            edge: 0.25,
            cavity: 0.6,
            boundary: Boundary::Clamp,
        };
        for (id, sign) in [(1, 1.0), (2, -1.0)] {
            let rgb = encode(&cpu::paraboloid(e, sign), NormalConvention::OpenGl);
            let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
            let planes = gpu.evaluate(
                &worker,
                request(&source, id, c, NormalConvention::OpenGl, &WITH_NORMAL),
            );
            check(&planes, e, NormalConvention::OpenGl, c);
            let center = (12 * 33 + 16) as usize;
            assert!(f64::from(planes.scalar[&MapKind::Curvature][center]) * sign > 0.5);
            let (active, inactive) = if sign > 0.0 {
                (MapKind::Edge, MapKind::Cavity)
            } else {
                (MapKind::Cavity, MapKind::Edge)
            };
            assert!(planes.scalar[&active][center] > 0.1);
            assert!(planes.scalar[&inactive].iter().all(|&v| v == 0.0));
        }
        worker.join().unwrap();
    });
}

#[test]
fn independent_oracle_covers_uv_axis_signs_impulse_npot_and_steep_normals() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let mut id = 0;
        for (w, h, fixture) in [
            (19, 11, 0),
            (11, 19, 1),
            (31, 17, 2),
            (257, 193, 3),
            (1, 23, 3),
            (23, 1, 3),
            (17, 13, 4),
        ] {
            let e = Extent::new(w, h).unwrap();
            let n = match fixture {
                0 => image(e, |x, _| {
                    [0.6 * (f64::from(x) + 0.5) / f64::from(w) - 0.3, 0.0, 1.0]
                }),
                1 => image(e, |_, y| {
                    [0.0, -0.6 * (f64::from(y) + 0.5) / f64::from(h) + 0.3, 1.0]
                }),
                2 => image(e, |x, y| {
                    if x == 0 && y == 0 {
                        [0.6, -0.7, 0.1]
                    } else {
                        [0.0, 0.0, 1.0]
                    }
                }),
                3 => cpu::seeded(e),
                _ => image(e, |x, y| {
                    [
                        if x % 2 == 0 { 1.0 } else { -1.0 },
                        if y % 2 == 0 { 1.0 } else { -1.0 },
                        0.00001,
                    ]
                }),
            };
            let rgb = encode(&n, NormalConvention::OpenGl);
            let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                id += 1;
                let c = Controls {
                    edge: 0.003,
                    cavity: 0.007,
                    boundary,
                };
                let planes = gpu.evaluate(
                    &worker,
                    request(&source, id, c, NormalConvention::OpenGl, &WITH_NORMAL),
                );
                for (actual, expected) in planes
                    .normal
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(cpu::decoded(&rgb))
                {
                    for channel in 0..3 {
                        assert!((actual[channel] - expected[channel]).abs() <= 2e-3);
                    }
                }
                check(&planes, e, NormalConvention::OpenGl, c);
                if fixture <= 1 {
                    let center = ((h / 2) * w + w / 2) as usize;
                    assert!(
                        planes.scalar[&MapKind::Curvature][center] > 0.3,
                        "both UV axes contribute positively for a bump"
                    );
                }
            }
        }
        assert!(worker.diagnostics().peak_bytes <= GIB);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn convention_is_canonicalized_once_and_metadata_changes_y_divergence() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(23, 17).unwrap();
        let n = image(e, |_, y| [0.0, 0.6 * f64::from(y) / 17.0 - 0.3, 1.0]);
        let c = Controls {
            boundary: Boundary::Clamp,
            ..Default::default()
        };
        let mut results = Vec::new();
        for (i, convention) in [NormalConvention::OpenGl, NormalConvention::DirectX]
            .into_iter()
            .enumerate()
        {
            let rgb = encode(&n, convention);
            let source = gpu.install(&worker, e, &rgb, convention);
            let planes = gpu.evaluate(
                &worker,
                request(&source, i as u64 + 1, c, convention, &WITH_NORMAL),
            );
            check(&planes, e, convention, c);
            results.push(planes.scalar);
        }
        for kind in [MapKind::Curvature, MapKind::Edge, MapKind::Cavity] {
            compare(
                &results[0][&kind],
                &results[1][&kind]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
            );
        }
        let rgb = encode(&n, NormalConvention::OpenGl);
        let source = gpu.install(&worker, e, &rgb, NormalConvention::DirectX);
        let wrong = gpu.evaluate(
            &worker,
            request(&source, 3, c, NormalConvention::DirectX, &WITH_NORMAL),
        );
        check(&wrong, e, NormalConvention::DirectX, c);
        for (&a, &b) in results[0][&MapKind::Curvature]
            .iter()
            .zip(&wrong.scalar[&MapKind::Curvature])
        {
            assert!((a + b).abs() <= 2e-5);
        }
        assert!(wrong.scalar[&MapKind::Curvature].iter().any(|&v| v > 0.3));
        worker.join().unwrap();
    });
}

#[test]
fn edge_and_cavity_scales_are_independent_and_do_not_remap_signed_curvature() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(29, 17).unwrap();
        let rgb = encode(&cpu::seeded(e), NormalConvention::OpenGl);
        let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
        let base = Controls {
            edge: 0.003,
            cavity: 0.007,
            ..Default::default()
        };
        let baseline = gpu.evaluate(
            &worker,
            request(&source, 1, base, NormalConvention::OpenGl, &WITH_NORMAL),
        );
        assert!(
            baseline.scalar[&MapKind::Curvature]
                .iter()
                .any(|&v| v < -1.0)
        );
        assert!(
            baseline.scalar[&MapKind::Curvature]
                .iter()
                .any(|&v| v > 1.0)
        );
        for (i, c) in [
            Controls { edge: 0.0, ..base },
            Controls {
                cavity: 0.0,
                ..base
            },
            Controls {
                edge: f32::MAX,
                cavity: f32::MAX,
                ..base
            },
        ]
        .into_iter()
        .enumerate()
        {
            let planes = gpu.evaluate(
                &worker,
                request(
                    &source,
                    i as u64 + 2,
                    c,
                    NormalConvention::OpenGl,
                    &WITH_NORMAL,
                ),
            );
            check(&planes, e, NormalConvention::OpenGl, c);
            assert_eq!(
                planes.scalar[&MapKind::Curvature],
                baseline.scalar[&MapKind::Curvature]
            );
            if c.edge == base.edge {
                assert_eq!(
                    planes.scalar[&MapKind::Edge],
                    baseline.scalar[&MapKind::Edge]
                );
            }
            if c.cavity == base.cavity {
                assert_eq!(
                    planes.scalar[&MapKind::Cavity],
                    baseline.scalar[&MapKind::Cavity]
                );
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn each_secondary_port_works_alone_after_combined_outputs_and_cache_reuse() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(31, 19).unwrap();
        let rgb = encode(&cpu::seeded(e), NormalConvention::OpenGl);
        let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
        let c = Controls {
            edge: 0.003,
            cavity: 0.007,
            ..Default::default()
        };
        // Start with a secondary-only request to expose primary-output assumptions.
        let first = gpu.evaluate(
            &worker,
            request(
                &source,
                1,
                c,
                NormalConvention::OpenGl,
                &[OutputPort::Cavity],
            ),
        );
        let all = gpu.evaluate(
            &worker,
            request(&source, 2, c, NormalConvention::OpenGl, &WITH_NORMAL),
        );
        check(&all, e, NormalConvention::OpenGl, c);
        assert_eq!(first.scalar[&MapKind::Cavity], all.scalar[&MapKind::Cavity]);
        for (i, port) in PORTS.into_iter().enumerate() {
            let one = gpu.evaluate(
                &worker,
                request(&source, i as u64 + 3, c, NormalConvention::OpenGl, &[port]),
            );
            assert_eq!(one.scalar[&port.map_kind()], all.scalar[&port.map_kind()]);
        }
        let before = worker.diagnostics().cache_hits;
        let warm = gpu.evaluate(
            &worker,
            request(&source, 6, c, NormalConvention::OpenGl, &PORTS),
        );
        assert!(worker.diagnostics().cache_hits > before);
        assert_eq!(warm.scalar, all.scalar);
        worker.join().unwrap();
        let fresh = gpu.worker(GIB);
        let source = gpu.install(&fresh, e, &rgb, NormalConvention::OpenGl);
        let cold = gpu.evaluate(
            &fresh,
            request(&source, 1, c, NormalConvention::OpenGl, &PORTS),
        );
        for kind in [MapKind::Curvature, MapKind::Edge, MapKind::Cavity] {
            assert_eq!(
                bytemuck::cast_slice::<f32, u8>(&cold.scalar[&kind]),
                bytemuck::cast_slice::<f32, u8>(&warm.scalar[&kind])
            );
        }
        fresh.join().unwrap();
    });
}

#[test]
fn repeat_is_translation_equivariant_and_boundary_edits_invalidate_cached_outputs() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(29, 17).unwrap();
        let n = cpu::seeded(e);
        let shifted = image(e, |x, y| n[(((y + 5) % 17) * 29 + (x + 11) % 29) as usize]);
        let mut results = Vec::new();
        for (i, normals) in [&n, &shifted].into_iter().enumerate() {
            let rgb = encode(normals, NormalConvention::OpenGl);
            let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                let c = Controls {
                    edge: 0.003,
                    cavity: 0.007,
                    boundary,
                };
                let planes = gpu.evaluate(
                    &worker,
                    request(
                        &source,
                        (i * 2 + results.len() + 1) as u64,
                        c,
                        NormalConvention::OpenGl,
                        &WITH_NORMAL,
                    ),
                );
                check(&planes, e, NormalConvention::OpenGl, c);
                results.push(planes.scalar);
            }
        }
        for kind in [MapKind::Curvature, MapKind::Edge, MapKind::Cavity] {
            let shifted_reference = image(e, |x, y| {
                f64::from(results[0][&kind][(((y + 5) % 17) * 29 + (x + 11) % 29) as usize])
            });
            compare(&results[2][&kind], &shifted_reference);
        }
        assert!(
            results[0][&MapKind::Curvature]
                .iter()
                .zip(&results[1][&MapKind::Curvature])
                .any(|(a, b)| (a - b).abs() > 0.1)
        );
        worker.join().unwrap();
    });
}

#[test]
fn periodic_fixture_has_small_wrapped_edge_seam_scores() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(32, 24).unwrap();
        let n = image(e, |x, y| {
            let nx = 0.12 * (std::f64::consts::TAU * (f64::from(x) + 0.5) / 32.0).sin();
            let ny = -0.08 * (std::f64::consts::TAU * (f64::from(y) + 0.5) / 24.0).sin();
            [nx, ny, (1.0 - nx * nx - ny * ny).sqrt()]
        });
        let rgb = encode(&n, NormalConvention::OpenGl);
        let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
        let c = Controls::default();
        let planes = gpu.evaluate(
            &worker,
            request(&source, 1, c, NormalConvention::OpenGl, &WITH_NORMAL),
        );
        check(&planes, e, NormalConvention::OpenGl, c);
        let expected = cpu::curvature(
            planes.normal.as_ref().unwrap(),
            e,
            NormalConvention::OpenGl,
            c,
        );
        for (port, reference) in PORTS.iter().zip(&expected) {
            let values = &planes.scalar[&port.map_kind()];
            let actual = seam_scores(values.iter().copied().map(f64::from), e);
            let baseline = seam_scores(reference.iter().copied(), e);
            // The imported Rgba16Float field is quantized before normalized UV
            // differentiation: its f64 oracle itself has a 9.1564257e-5 vertical
            // seam on DX12, so the old 2e-5 bound is incompatible with the pixel
            // oracle. For each axis, |GPU seam| <= |oracle seam| + 2*(2e-5)
            // by the triangle inequality. Retain a separate small-oracle-seam
            // check, and prove below that a 1e-3 injected seam is rejected.
            assert!(baseline.into_iter().all(|v| v <= 1e-4));
            assert!(
                within_seam_bound(actual, baseline),
                "wrapped seam: {actual:?}, oracle: {baseline:?}"
            );
        }
        worker.join().unwrap();
    });
}

fn seam_scores(values: impl Iterator<Item = f64>, e: Extent) -> [f64; 2] {
    let values: Vec<_> = values.collect();
    let w = e.width as usize;
    let h = e.height as usize;
    assert_eq!(values.len(), w * h);
    [
        (0..h)
            .map(|y| (values[y * w] - values[y * w + w - 1]).abs())
            .fold(0.0, f64::max),
        (0..w)
            .map(|x| (values[x] - values[(h - 1) * w + x]).abs())
            .fold(0.0, f64::max),
    ]
}

fn within_seam_bound(actual: [f64; 2], oracle: [f64; 2]) -> bool {
    actual
        .into_iter()
        .zip(oracle)
        .all(|(a, b)| a.is_finite() && a <= b + 4e-5)
}

#[test]
fn periodic_seam_bound_rejects_an_injected_edge_defect_on_either_axis() {
    let e = Extent::new(32, 24).unwrap();
    // Include the measured quantization floor rather than assuming an ideal zero.
    let baseline = [9.1564257e-5; 2];
    assert!(within_seam_bound(baseline.map(|v| v + 4e-5), baseline));
    for axis in 0..2 {
        let mut values = vec![0.0; (e.width * e.height) as usize];
        for y in 0..e.height {
            for x in 0..e.width {
                if (axis == 0 && x == 0) || (axis == 1 && y == 0) {
                    values[(y * e.width + x) as usize] = 1e-3;
                }
            }
        }
        assert!(!within_seam_bound(
            seam_scores(values.into_iter(), e),
            baseline
        ));
    }
}
