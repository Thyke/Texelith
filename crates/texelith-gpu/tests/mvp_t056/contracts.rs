use super::{reference, support::*};
use texelith_core::{Boundary, BrushSample, Extent, MapKind, MaskMode, MaterialNodeParams};
use texelith_gpu::scheduler::{Completion, Scheduler};
use texelith_gpu::{MaterialError, MaterialPlan, Quality, ResourceReservations, WorkerEvent};

fn plan(request: &texelith_gpu::MaterialRequest) -> MaterialPlan {
    MaterialPlan::new(request, GIB, ResourceReservations::default())
        .expect("PaintMask must have a registered budgeted adapter")
}

#[test]
fn mask_cache_identity_covers_brush_samples_order_quality_and_boundary_but_not_edit_ids() {
    let e = Extent::new(2049, 17).unwrap();
    let first = dot(1, [0.5; 2], 0.2, 0.5, MaskMode::Add);
    let second = dot(2, [0.2; 2], 0.1, 0.25, MaskMode::Subtract);
    let strokes = vec![first.clone(), second.clone()];
    let key = |strokes: &[texelith_core::BrushStroke], id, quality| {
        plan(&request(e, strokes, id, quality)).output_keys()[&MapKind::Mask]
    };
    let original = key(&strokes, 1, Quality::Full);
    assert_eq!(original, key(&strokes, 99, Quality::Full));
    let mut renamed = strokes.clone();
    renamed[0].id.0 = 123;
    assert_eq!(original, key(&renamed, 2, Quality::Full));
    assert_ne!(original, key(&strokes, 2, Quality::Proxy));
    assert_ne!(original, key(&[second, first], 2, Quality::Full));
    for field in 0..7 {
        let mut changed = strokes.clone();
        match field {
            0 => changed[0].brush.radius = 0.21,
            1 => changed[0].brush.hardness = 0.5,
            2 => changed[0].brush.opacity = 0.6,
            3 => changed[0].brush.mode = MaskMode::Replace,
            4 => changed[0].points[0].uv[0] = 0.51,
            5 => changed[0].points[0].pressure = 0.7,
            _ => changed[0].points.push(BrushSample {
                uv: [0.52, 0.5],
                pressure: 1.0,
            }),
        }
        assert_ne!(
            original,
            key(&changed, 3, Quality::Full),
            "unhashed stroke field {field}"
        );
    }
    let repeat = request_graph(
        with_boundary(graph(e, &strokes), Boundary::Repeat),
        4,
        Quality::Full,
    );
    let clamp = request_graph(
        with_boundary(graph(e, &strokes), Boundary::Clamp),
        5,
        Quality::Full,
    );
    assert_ne!(plan(&repeat).output_keys(), plan(&clamp).output_keys());
}

#[test]
fn explicit_repeat_and_clamp_selectors_round_trip_instead_of_being_silently_ignored() {
    for boundary in [Boundary::Repeat, Boundary::Clamp] {
        let g = with_boundary(graph(Extent::new(17, 13).unwrap(), &[]), boundary);
        assert!(g.is_material());
    }
}

#[test]
fn four_k_plan_reserves_output_scratch_uploads_and_shared_memory_at_exact_budget_boundary() {
    let e = Extent::new(4096, 4096).unwrap();
    let r = request(
        e,
        &[dot(1, [0.5; 2], 0.1, 0.5, MaskMode::Add)],
        1,
        Quality::Full,
    );
    let reservations = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 64 * 1024 * 1024,
    };
    let peak = MaterialPlan::new(&r, GIB, reservations)
        .expect("4K paint must fit 1 GiB")
        .peak_bytes();
    assert!(
        peak > reservations.viewport_bytes + 4096 * 4096 * 4,
        "include stroke staging/uniforms and scratch, not only output"
    );
    assert!(peak <= GIB);
    assert!(MaterialPlan::new(&r, peak, reservations).is_ok());
    assert!(matches!(MaterialPlan::new(&r, peak - 1, reservations),
        Err(MaterialError::BudgetExceeded { required_bytes, budget_bytes })
            if required_bytes == peak && budget_bytes == peak - 1));
}

#[test]
fn worker_budget_rejection_precedes_dispatch_and_recovery_has_no_partial_mask() {
    with_gpu(|gpu| {
        let budget = 1024 * 1024;
        let worker = gpu.worker(budget);
        let e = Extent::new(63, 47).unwrap();
        let strokes = [dot(1, [0.5; 2], 0.23, 0.5, MaskMode::Replace)];
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: budget,
                readback_bytes: 0,
            })
            .unwrap();
        let submitted = worker.diagnostics().submitted;
        worker
            .request_material(request(e, &strokes, 1, Quality::Full))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::BudgetExceeded { required_bytes, budget_bytes }
            if required_bytes > budget && budget_bytes == budget),
            "{event:?}"
        );
        assert_eq!(worker.diagnostics().submitted, submitted);
        assert!(worker.try_latest_material().is_none());
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let actual = gpu.evaluate(&worker, request(e, &strokes, 2, Quality::Full));
        reference::compare(&actual, &reference::raster(e, &strokes, Boundary::Repeat));
        assert!(worker.diagnostics().peak_bytes <= budget);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn superseded_paint_work_has_one_active_one_pending_and_cannot_publish() {
    let e = Extent::new(31, 17).unwrap();
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(e, &[], 1, Quality::Full))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    for id in 2..=64 {
        scheduler
            .request_material(request(
                e,
                &[dot(id, [0.5; 2], 0.2, id as f32 / 64.0, MaskMode::Add)],
                id,
                Quality::Full,
            ))
            .unwrap();
        assert!(!scheduler.should_continue(active.id));
        assert!(scheduler.start_next().is_none());
    }
    assert_eq!(scheduler.complete(active.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let latest = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(latest.id), Completion::Published);
    assert_eq!(scheduler.try_latest_material().unwrap().request_id, 64);
    let d = scheduler.diagnostics();
    assert_eq!(d.peak_in_flight, 1);
    assert_eq!(d.peak_pending, 1);
    assert_eq!(d.stale_publications, 0);
    assert!(d.cancellation_requests > 0);
}

#[test]
fn latest_paint_supersedes_large_batched_replay_without_partial_publication() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(511, 257).unwrap();
        let mut stroke = dot(1, [0.5; 2], 0.02, 0.001, MaskMode::Add);
        // A valid long path crossing enough segment batches to exercise the
        // worker cancellation boundary; no UI events or unbounded upload queue.
        stroke.points = (0..8192)
            .map(|i| BrushSample {
                uv: [0.5 + 0.15 * (i as f32 * 0.01).sin(), 0.5],
                pressure: 1.0,
            })
            .collect();
        stroke.validate().unwrap();
        let r = request(e, &[stroke], 1, Quality::Full);
        plan(&r);
        worker.request_material(r).unwrap();
        wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("{event:?}");
            }
            (worker.diagnostics().submitted > 0).then_some(())
        });
        for id in 2..=33 {
            worker
                .request_material(request(
                    e,
                    &[dot(
                        id,
                        [0.5; 2],
                        0.201,
                        if id == 33 { 0.7 } else { 0.2 },
                        MaskMode::Replace,
                    )],
                    id,
                    Quality::Full,
                ))
                .unwrap();
        }
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("{event:?}");
            }
            worker.try_latest_material().filter(|b| b.request_id == 33)
        });
        let actual = gpu.read(bundle.outputs[&MapKind::Mask].texture());
        let expected = reference::raster(
            e,
            &[dot(33, [0.5; 2], 0.201, 0.7, MaskMode::Replace)],
            Boundary::Repeat,
        );
        reference::compare(&actual, &expected);
        let d = worker.diagnostics();
        assert_eq!(d.peak_in_flight, 1);
        assert!(d.peak_pending <= 1);
        assert!(d.peak_bytes <= GIB);
        assert_eq!(d.stale_publications, 0);
        assert_eq!(d.validation_errors, 0);
        // Submission observation is not proof that a long request remained
        // active. Real cancelled-batch counts belong in the hardware report.
        drop(bundle);
        worker.join().unwrap();
    });
}

#[test]
fn digest_only_nonempty_legacy_masks_fail_explicitly_instead_of_fabricating_pixels() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(17, 13).unwrap();
        let mut dto = texelith_core::MaterialGraphDto::from(graph(e, &[]));
        dto.graph[0].params = MaterialNodeParams::PaintMask {
            boundary: texelith_core::Boundary::Repeat,
            extent: e,
            stroke_digest: [56; 32],
            strokes: vec![],
        };
        let r = request_graph(
            texelith_core::Graph::try_from(dto).unwrap(),
            1,
            Quality::Full,
        );
        let error = match worker.request_material(r) {
            Err(error) => error.to_string(),
            Ok(()) => {
                let event = wait(|| {
                    assert!(
                        worker.try_latest_material().is_none(),
                        "digest alone cannot reconstruct paint"
                    );
                    worker.try_event()
                });
                match event {
                    WorkerEvent::Error(error) => error.to_string(),
                    other => panic!("expected missing replay data error, received {other:?}"),
                }
            }
        };
        assert!(
            ["digest", "stroke", "replay", "vector"]
                .iter()
                .any(|word| error.to_lowercase().contains(word)),
            "missing vector data must produce a specific error: {error}"
        );
        worker.join().unwrap();
    });
}

#[test]
fn hard_brush_reference_has_hand_calculated_order_and_periodic_corner_values() {
    let e = Extent::new(5, 5).unwrap();
    let a = dot(1, [0.5; 2], 0.15, 0.6, MaskMode::Add);
    let b = dot(2, [0.5; 2], 0.15, 0.25, MaskMode::Subtract);
    let v = reference::raster(e, &[a, b], Boundary::Clamp);
    assert!((v[12] - 0.35).abs() < 1e-7);
    assert_eq!(v.iter().filter(|&&v| v != 0.0).count(), 1);
    let corner = [dot(1, [0.0; 2], 0.15, 1.0, MaskMode::Add)];
    let repeat = reference::raster(e, &corner, Boundary::Repeat);
    let clamp = reference::raster(e, &corner, Boundary::Clamp);
    for i in [0, 4, 20, 24] {
        assert_eq!(repeat[i], 1.0);
    }
    assert_eq!(clamp[0], 1.0);
    for i in [4, 20, 24] {
        assert_eq!(clamp[i], 0.0);
    }
}

#[test]
fn legacy_repeat_serialization_and_frozen_cache_bytes_remain_unchanged() {
    use texelith_core::{
        CacheRevisions, MaterialNode, MaterialNodeParams, OutputPort,
        canonical_material_node_bytes, material_node_cache_key,
    };
    let old = serde_json::json!({"PaintMask":{"extent":{"width":17,"height":13},"stroke_digest":vec![0;32]}});
    let params: MaterialNodeParams = serde_json::from_value(old.clone()).unwrap();
    assert!(matches!(
        params,
        MaterialNodeParams::PaintMask {
            boundary: Boundary::Repeat,
            ..
        }
    ));
    assert_eq!(serde_json::to_value(&params).unwrap(), old);
    let e = Extent::new(17, 13).unwrap();
    let node = MaterialNode {
        id: NODE,
        inputs: vec![],
        params,
    };
    // Frozen pre-boundary schema-2 encoding: header/revisions/tag/format,
    // extent/digest, zero inputs, evaluation extent and source extent.
    let mut frozen = b"texelith.node-cache\0".to_vec();
    for n in [2u32, 1, 1] {
        frozen.extend_from_slice(&n.to_le_bytes());
    }
    frozen.extend_from_slice(&[11, 9, 0]);
    for n in [17u32, 13] {
        frozen.extend_from_slice(&n.to_le_bytes());
    }
    frozen.extend_from_slice(&[0; 32]);
    for n in [0u32, 17, 13, 17, 13] {
        frozen.extend_from_slice(&n.to_le_bytes());
    }
    assert_eq!(
        canonical_material_node_bytes(&node, OutputPort::Mask, &[], e, e, CacheRevisions::CURRENT)
            .unwrap(),
        frozen
    );
    let key = material_node_cache_key(&node, OutputPort::Mask, &[], e, e, CacheRevisions::CURRENT)
        .unwrap();
    let explicit = graph(e, &[]).material_node(NODE).unwrap().clone();
    assert_eq!(
        material_node_cache_key(
            &explicit,
            OutputPort::Mask,
            &[],
            e,
            e,
            CacheRevisions::CURRENT
        )
        .unwrap(),
        key
    );
}

#[test]
fn clamp_boundary_survives_stroke_commit_undo_and_redo() {
    use texelith_core::{CommandRequest, EvaluationSnapshot, Session, StrokeId};
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 0,
        graph: with_boundary(graph(Extent::new(17, 13).unwrap(), &[]), Boundary::Clamp),
    });
    let stroke = dot(1, [0.5; 2], 0.1, 0.7, MaskMode::Add);
    session
        .dispatch(CommandRequest::BeginStroke {
            node: NODE,
            stroke_id: StrokeId(1),
            brush: stroke.brush,
            sample: stroke.points[0],
        })
        .unwrap();
    session
        .dispatch(CommandRequest::CommitStroke {
            stroke_id: StrokeId(1),
        })
        .unwrap();
    for command in [CommandRequest::Undo, CommandRequest::Redo] {
        assert!(matches!(
            session
                .snapshot()
                .unwrap()
                .graph
                .material_node(NODE)
                .unwrap()
                .params,
            MaterialNodeParams::PaintMask {
                boundary: Boundary::Clamp,
                ..
            }
        ));
        session.dispatch(command).unwrap();
    }
    assert!(matches!(
        session
            .snapshot()
            .unwrap()
            .graph
            .material_node(NODE)
            .unwrap()
            .params,
        MaterialNodeParams::PaintMask {
            boundary: Boundary::Clamp,
            ..
        }
    ));
}
