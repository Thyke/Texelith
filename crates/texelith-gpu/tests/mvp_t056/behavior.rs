use super::{reference, support::*};
use texelith_core::{
    Boundary, BrushSample, CommandRequest, EvaluationSnapshot, Extent, MaskMode, Session, StrokeId,
};
use texelith_gpu::Quality;

#[test]
fn empty_mask_is_linear_r32_zero_on_npot_and_singleton_axes_without_source_upload() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        for (i, (w, h)) in [(31, 17), (1, 19), (23, 1), (1, 1)].into_iter().enumerate() {
            let e = Extent::new(w, h).unwrap();
            let actual = gpu.evaluate(&worker, request(e, &[], i as u64 + 1, Quality::Full));
            assert_eq!(actual, vec![0.0; (w * h) as usize]);
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn ordered_add_subtract_replace_and_saturation_match_independent_scalar_oracle() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(31, 17).unwrap();
        let a = dot(1, [0.5; 2], 0.231, 0.6, MaskMode::Add);
        let b = dot(2, [0.5; 2], 0.231, 0.25, MaskMode::Subtract);
        let c = dot(3, [0.5; 2], 0.231, 1.0, MaskMode::Replace);
        let cases = [
            vec![a.clone(), b.clone()],
            vec![b.clone(), a.clone()],
            vec![a.clone(), a.clone().with_id(4)],
            vec![b.clone()],
            vec![a.clone(), b.clone(), c.clone()],
            vec![c, a, b],
        ];
        let mut results = Vec::new();
        for (i, strokes) in cases.iter().enumerate() {
            let actual = gpu.evaluate(&worker, request(e, strokes, i as u64 + 1, Quality::Full));
            reference::compare(&actual, &reference::raster(e, strokes, Boundary::Repeat));
            results.push(actual);
        }
        let center = (e.height / 2 * e.width + e.width / 2) as usize;
        for (actual, expected) in results.iter().zip([0.35, 0.6, 1.0, 0.0, 1.0, 0.75]) {
            assert!((actual[center] - expected).abs() < 2e-5);
        }
        assert_ne!(results[0], results[1], "stroke order affects pixels");
        worker.join().unwrap();
    });
}

trait WithId {
    fn with_id(self, id: u64) -> Self;
}
impl WithId for texelith_core::BrushStroke {
    fn with_id(mut self, id: u64) -> Self {
        self.id = StrokeId(id);
        self
    }
}

#[test]
fn opacity_and_pressure_endpoints_preserve_numeric_values_without_srgb_or_u8_quantization() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(17, 13).unwrap();
        for (i, (opacity, pressure)) in [
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5001, 1.0),
            (0.5, 1.0),
            (0.5, 0.25),
        ]
        .into_iter()
        .enumerate()
        {
            let mut stroke = dot(1, [0.5; 2], 0.241, opacity, MaskMode::Replace);
            stroke.points[0].pressure = pressure;
            let actual = gpu.evaluate(
                &worker,
                request(e, &[stroke.clone()], i as u64 + 1, Quality::Full),
            );
            reference::compare(&actual, &reference::raster(e, &[stroke], Boundary::Repeat));
            assert!((actual[(6 * 17 + 8) as usize] - opacity * pressure).abs() <= 2e-5);
        }
        worker.join().unwrap();
    });
}

#[test]
fn stationary_session_strokes_preserve_maximum_endpoint_pressure() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(17, 13).unwrap();
        let center = (e.height / 2 * e.width + e.width / 2) as usize;
        let mut id = 0;
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            for pressures in [[0.0, 1.0], [1.0, 0.0], [0.25, 0.75]] {
                let mut session = Session::from_snapshot(EvaluationSnapshot {
                    generation: 0,
                    graph: with_boundary(graph(e, &[]), boundary),
                });
                session
                    .dispatch(CommandRequest::BeginStroke {
                        node: NODE,
                        stroke_id: StrokeId(1),
                        brush: dot(1, [0.5; 2], 0.241, 1.0, MaskMode::Add).brush,
                        sample: BrushSample {
                            uv: [0.5; 2],
                            pressure: pressures[0],
                        },
                    })
                    .unwrap();
                session
                    .dispatch(CommandRequest::AppendStroke {
                        stroke_id: StrokeId(1),
                        samples: pressures[1..]
                            .iter()
                            .map(|&pressure| BrushSample {
                                uv: [0.5; 2],
                                pressure,
                            })
                            .collect(),
                    })
                    .unwrap();
                session
                    .dispatch(CommandRequest::CommitStroke {
                        stroke_id: StrokeId(1),
                    })
                    .unwrap();
                let snapshot = session.snapshot().unwrap();
                let texelith_core::MaterialNodeParams::PaintMask { strokes, .. } =
                    &snapshot.graph.material_node(NODE).unwrap().params
                else {
                    unreachable!()
                };
                assert_eq!(strokes[0].points.len(), pressures.len());
                let expected = reference::raster(e, strokes, boundary);
                id += 1;
                let actual =
                    gpu.evaluate(&worker, request_graph(snapshot.graph, id, Quality::Full));
                let peak = pressures.iter().copied().fold(0.0_f32, f32::max);
                assert_eq!(
                    actual[center], peak,
                    "stationary GPU pressure: {pressures:?}"
                );
                assert_eq!(
                    expected[center],
                    f64::from(peak),
                    "stationary oracle pressure"
                );
                reference::compare(&actual, &expected);
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn hardness_zero_is_soft_and_hardness_one_has_a_flat_interior_and_bounded_support() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(65, 65).unwrap();
        let hard = dot(1, [0.5; 2], 0.241, 0.75, MaskMode::Replace);
        let mut soft = hard.clone();
        soft.brush.hardness = 0.0;
        let h = gpu.evaluate(&worker, request(e, &[hard], 1, Quality::Full));
        let s = gpu.evaluate(&worker, request(e, &[soft], 2, Quality::Full));
        let row = 32 * 65;
        assert!((s[row + 32] - 0.75).abs() <= 2e-5);
        assert!(s[row + 40] > 0.0 && s[row + 40] < 0.75);
        assert!((h[row + 40] - 0.75).abs() <= 2e-5);
        for x in 32..64 {
            assert!(
                s[row + x] + 2e-5 >= s[row + x + 1],
                "soft radial falloff reversed"
            );
        }
        for values in [&s, &h] {
            assert!(
                values
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=0.75).contains(v))
            );
            assert_eq!(
                values[row + 49],
                0.0,
                "outside radius must remain untouched"
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn repeat_wraps_edge_and_corner_brushes_and_clamp_does_not() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(37, 29).unwrap();
        for (i, uv) in [[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]].into_iter().enumerate() {
            let strokes = [dot(1, uv, 0.181, 0.7, MaskMode::Add)];
            let repeat = gpu.evaluate(
                &worker,
                request(e, &strokes, i as u64 * 2 + 1, Quality::Full),
            );
            reference::compare(&repeat, &reference::raster(e, &strokes, Boundary::Repeat));
            let clamped = gpu.evaluate(
                &worker,
                request_graph(
                    with_boundary(graph(e, &strokes), Boundary::Clamp),
                    i as u64 * 2 + 2,
                    Quality::Full,
                ),
            );
            reference::compare(&clamped, &reference::raster(e, &strokes, Boundary::Clamp));
            assert_ne!(
                repeat, clamped,
                "explicit boundary must affect painted edges"
            );
            if uv[0] == 0.0 {
                for y in 0..e.height {
                    assert_eq!(
                        repeat[(y * e.width) as usize],
                        repeat[(y * e.width + e.width - 1) as usize]
                    );
                }
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn canonical_t055_segments_replay_without_dab_buildup_and_same_backend_is_bit_stable() {
    with_gpu(|gpu| {
        let e = Extent::new(127, 63).unwrap();
        let mut session = Session::from_snapshot(EvaluationSnapshot {
            generation: 0,
            graph: graph(e, &[]),
        });
        session
            .dispatch(CommandRequest::BeginStroke {
                node: NODE,
                stroke_id: StrokeId(1),
                brush: dot(1, [0.125, 0.5], 0.125, 0.25, MaskMode::Add).brush,
                sample: BrushSample {
                    uv: [0.125, 0.5],
                    pressure: 1.0,
                },
            })
            .unwrap();
        session
            .dispatch(CommandRequest::AppendStroke {
                stroke_id: StrokeId(1),
                samples: vec![BrushSample {
                    uv: [0.875, 0.5],
                    pressure: 1.0,
                }],
            })
            .unwrap();
        session
            .dispatch(CommandRequest::CommitStroke {
                stroke_id: StrokeId(1),
            })
            .unwrap();
        let snapshot = session.snapshot().unwrap();
        let texelith_core::MaterialNodeParams::PaintMask { strokes, .. } =
            &snapshot.graph.material_node(NODE).unwrap().params
        else {
            unreachable!()
        };
        assert!(
            strokes[0].points.len() > 16,
            "use the producer's canonical samples"
        );
        let expected = reference::raster(e, strokes, Boundary::Repeat);
        let worker = gpu.worker(GIB);
        let first = gpu.evaluate(
            &worker,
            request_graph(snapshot.graph.clone(), 1, Quality::Full),
        );
        reference::compare(&first, &expected);
        let submissions = worker.diagnostics().submitted;
        let hits = worker.diagnostics().cache_hits;
        let same = gpu.evaluate(
            &worker,
            request_graph(snapshot.graph.clone(), 2, Quality::Full),
        );
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            same.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(worker.diagnostics().submitted, submissions);
        assert!(worker.diagnostics().cache_hits > hits);
        worker.join().unwrap();
        let cold = gpu.worker(GIB);
        let replay = gpu.evaluate(&cold, request_graph(snapshot.graph, 3, Quality::Full));
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            replay.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "determinism must survive cache eviction"
        );
        cold.join().unwrap();
    });
}

#[test]
fn canonical_turns_and_interpolated_pressure_match_segment_geometry() {
    with_gpu(|gpu| {
        let e = Extent::new(79, 61).unwrap();
        let mut session = Session::from_snapshot(EvaluationSnapshot {
            generation: 0,
            graph: graph(e, &[]),
        });
        session
            .dispatch(CommandRequest::BeginStroke {
                node: NODE,
                stroke_id: StrokeId(1),
                brush: dot(1, [0.125, 0.25], 0.125, 0.5, MaskMode::Add).brush,
                sample: BrushSample {
                    uv: [0.125, 0.25],
                    pressure: 0.25,
                },
            })
            .unwrap();
        session
            .dispatch(CommandRequest::AppendStroke {
                stroke_id: StrokeId(1),
                samples: vec![
                    BrushSample {
                        uv: [0.75, 0.25],
                        pressure: 0.875,
                    },
                    BrushSample {
                        uv: [0.75, 0.75],
                        pressure: 0.5,
                    },
                ],
            })
            .unwrap();
        session
            .dispatch(CommandRequest::CommitStroke {
                stroke_id: StrokeId(1),
            })
            .unwrap();
        let snapshot = session.snapshot().unwrap();
        let texelith_core::MaterialNodeParams::PaintMask { strokes, .. } =
            &snapshot.graph.material_node(NODE).unwrap().params
        else {
            unreachable!()
        };
        let expected = reference::raster(e, strokes, Boundary::Repeat);
        let worker = gpu.worker(GIB);
        let actual = gpu.evaluate(&worker, request_graph(snapshot.graph, 1, Quality::Full));
        reference::compare(&actual, &expected);
        worker.join().unwrap();
    });
}

#[test]
fn proxy_and_full_rasterize_the_same_uv_geometry_at_actual_npot_pixel_centers() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let full = Extent::new(2051, 7).unwrap();
        let strokes = [
            dot(1, [0.07, 0.5], 0.181, 0.6, MaskMode::Replace),
            dot(2, [0.92, 0.5], 0.221, 0.2, MaskMode::Subtract),
        ];
        for (i, q) in [Quality::Proxy, Quality::Full].into_iter().enumerate() {
            let r = request(full, &strokes, i as u64 + 1, q);
            let extent = r.extent;
            let actual = gpu.evaluate(&worker, r);
            reference::compare(
                &actual,
                &reference::raster(extent, &strokes, Boundary::Repeat),
            );
            if q == Quality::Proxy {
                assert_eq!(extent, Extent::new(1024, 3).unwrap());
            }
        }
        worker.join().unwrap();
    });
}

#[test]
fn undo_redo_and_branching_restore_pixels_with_warm_and_cold_checkpoint_caches() {
    with_gpu(|gpu| {
        let e = Extent::new(63, 47).unwrap();
        let mut session = Session::from_snapshot(EvaluationSnapshot {
            generation: 0,
            graph: graph(e, &[]),
        });
        let worker = gpu.worker(GIB);
        let mut snapshots = vec![session.snapshot().unwrap().graph];
        for id in 1..=20 {
            let stroke = dot(
                id,
                [0.5; 2],
                0.181,
                if id % 2 == 0 { 0.1 } else { 0.2 },
                if id % 2 == 0 {
                    MaskMode::Subtract
                } else {
                    MaskMode::Add
                },
            );
            session
                .dispatch(CommandRequest::BeginStroke {
                    node: NODE,
                    stroke_id: stroke.id,
                    brush: stroke.brush,
                    sample: stroke.points[0],
                })
                .unwrap();
            session
                .dispatch(CommandRequest::CommitStroke {
                    stroke_id: stroke.id,
                })
                .unwrap();
            let g = session.snapshot().unwrap().graph;
            let texelith_core::MaterialNodeParams::PaintMask { strokes, .. } =
                &g.material_node(NODE).unwrap().params
            else {
                unreachable!()
            };
            let actual = gpu.evaluate(&worker, request_graph(g.clone(), id, Quality::Full));
            reference::compare(&actual, &reference::raster(e, strokes, Boundary::Repeat));
            snapshots.push(g);
        }
        for _ in 0..11 {
            session.dispatch(CommandRequest::Undo).unwrap();
        }
        let undone = session.snapshot().unwrap().graph;
        let actual = gpu.evaluate(&worker, request_graph(undone.clone(), 21, Quality::Full));
        let cold = gpu.worker(GIB);
        let replay = gpu.evaluate(
            &cold,
            request_graph(snapshots[9].clone(), 22, Quality::Full),
        );
        assert_eq!(
            actual, replay,
            "undo must replay the matching prefix after eviction"
        );
        cold.join().unwrap();
        session.dispatch(CommandRequest::Redo).unwrap();
        let redo = gpu.evaluate(
            &worker,
            request_graph(session.snapshot().unwrap().graph, 23, Quality::Full),
        );
        assert_ne!(redo, actual);
        // A different suffix must not reuse a checkpoint from the old branch.
        session.dispatch(CommandRequest::Undo).unwrap();
        let branch = dot(99, [0.2, 0.3], 0.12, 0.85, MaskMode::Replace);
        session
            .dispatch(CommandRequest::BeginStroke {
                node: NODE,
                stroke_id: branch.id,
                brush: branch.brush,
                sample: branch.points[0],
            })
            .unwrap();
        session
            .dispatch(CommandRequest::CommitStroke {
                stroke_id: branch.id,
            })
            .unwrap();
        assert_eq!(session.history().redo_entries, 0);
        let g = session.snapshot().unwrap().graph;
        let texelith_core::MaterialNodeParams::PaintMask { strokes, .. } =
            &g.material_node(NODE).unwrap().params
        else {
            unreachable!()
        };
        let branched = gpu.evaluate(&worker, request_graph(g.clone(), 24, Quality::Full));
        reference::compare(&branched, &reference::raster(e, strokes, Boundary::Repeat));
        assert_ne!(branched, redo);
        assert!(worker.diagnostics().peak_bytes <= GIB);
        worker.join().unwrap();
    });
}

#[test]
fn decreasing_pressure_and_cross_batch_coverage_preserve_the_first_dab_and_maximum() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(63, 47).unwrap();
        let mut stroke = dot(1, [0.4, 0.5], 0.181, 0.7, MaskMode::Replace);
        stroke.points = (0..600)
            .map(|i| BrushSample {
                uv: [0.4 + i as f32 * 0.00025, 0.5],
                pressure: 1.0 - i as f32 / 800.0,
            })
            .collect();
        let actual = gpu.evaluate(&worker, request(e, &[stroke.clone()], 1, Quality::Full));
        reference::compare(&actual, &reference::raster(e, &[stroke], Boundary::Repeat));
        worker.join().unwrap();
    });
}
