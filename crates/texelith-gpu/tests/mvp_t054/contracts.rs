use super::{
    harness_contract,
    reference::{Combine, Controls},
    support::{GIB, request, wait, with_gpu},
};
use texelith_core::{Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::scheduler::{Completion, Scheduler};
use texelith_gpu::{
    MaterialError, MaterialPlan, Quality, ResourceReservations, TypedSourcePixels, WorkerEvent,
};

fn source(e: Extent) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [54; 32],
        extent: e,
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

#[test]
fn public_parameters_accept_endpoints_and_reject_nonfinite_or_out_of_range_controls() {
    let base = Controls::default();
    for c in [
        base,
        Controls {
            picked: [0.; 3],
            radius: 0.,
            threshold: 0.,
            feather: 0.,
            ..base
        },
        Controls {
            picked: [1.; 3],
            radius: 3.0_f32.sqrt(),
            threshold: 1.,
            feather: f32::MAX,
            ..base
        },
    ] {
        assert!(
            c.params().validate().is_ok(),
            "rejected legal controls: {c:?}"
        );
    }
    for c in [
        Controls {
            picked: [-0.01, 0., 0.],
            ..base
        },
        Controls {
            picked: [0., 1.01, 0.],
            ..base
        },
        Controls {
            picked: [0., 0., f32::NAN],
            ..base
        },
        Controls {
            picked: [f32::INFINITY; 3],
            ..base
        },
        Controls {
            radius: -0.01,
            ..base
        },
        Controls {
            radius: 1.733,
            ..base
        },
        Controls {
            radius: f32::NAN,
            ..base
        },
        Controls {
            radius: f32::INFINITY,
            ..base
        },
        Controls {
            threshold: -0.01,
            ..base
        },
        Controls {
            threshold: 1.01,
            ..base
        },
        Controls {
            threshold: f32::NAN,
            ..base
        },
        Controls {
            threshold: f32::INFINITY,
            ..base
        },
        Controls {
            feather: -0.01,
            ..base
        },
        Controls {
            feather: f32::NAN,
            ..base
        },
        Controls {
            feather: f32::INFINITY,
            ..base
        },
    ] {
        assert!(
            c.params().validate().is_err(),
            "accepted invalid controls: {c:?}"
        );
    }
}

#[test]
fn public_plan_cache_key_covers_all_color_controls_mask_modes_source_and_quality() {
    let source = source(Extent::new(2048, 32).unwrap());
    let key = |source: &SourceDescriptor, id, c, q| {
        *MaterialPlan::new(
            &request(source, id, c, q),
            GIB,
            ResourceReservations::default(),
        )
        .expect("Metallic must be supported through the public planner")
        .output_keys()
        .get(&MapKind::Metallic)
        .unwrap()
    };
    let base = Controls::default();
    let first = key(&source, 1, base, Quality::Full);
    assert_eq!(
        first,
        key(&source, 99, base, Quality::Full),
        "generation is not content"
    );
    for c in [
        Controls {
            picked: [0.25, 0.5, 0.75],
            ..base
        },
        Controls {
            radius: 0.5,
            ..base
        },
        Controls {
            threshold: 0.75,
            ..base
        },
        Controls {
            feather: 0.0,
            ..base
        },
        Controls {
            mode: Combine::Add,
            ..base
        },
        Controls {
            mode: Combine::Subtract,
            ..base
        },
    ] {
        assert_ne!(
            first,
            key(&source, 2, c, Quality::Full),
            "missing control in cache key: {c:?}"
        );
    }
    assert_ne!(first, key(&source, 2, base, Quality::Proxy));
    let changed = SourceDescriptor {
        content_hash: [55; 32],
        ..source.clone()
    };
    assert_ne!(first, key(&changed, 2, base, Quality::Full));
}

#[test]
fn four_k_plan_accounts_for_color_scalar_output_and_shared_reservations_with_exact_budget_boundary()
{
    let r = request(
        &source(Extent::new(4096, 4096).unwrap()),
        1,
        Controls::default(),
        Quality::Full,
    );
    let reserved = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 128 * 1024 * 1024,
    };
    let peak = MaterialPlan::new(&r, GIB, reserved)
        .expect("4K metallic must fit shared 1 GiB ledger")
        .peak_bytes();
    // Color and metallic are simultaneously live while the node reads/writes.
    assert!(peak >= 4096 * 4096 * 12 + reserved.viewport_bytes);
    assert!(peak <= GIB);
    assert!(MaterialPlan::new(&r, peak, reserved).is_ok());
    assert!(
        matches!(MaterialPlan::new(&r, peak-1, reserved), Err(MaterialError::BudgetExceeded { required_bytes, budget_bytes }) if required_bytes == peak && budget_bytes == peak-1)
    );
}

#[test]
fn worker_rejects_over_budget_before_submission_then_recovers_after_reservation_release() {
    with_gpu(|gpu| {
        let budget = 512 * 1024;
        let worker = gpu.worker(budget);
        let e = Extent::new(129, 129).unwrap();
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: vec![0; 129 * 129 * 4].into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: budget,
                readback_bytes: 0,
            })
            .unwrap();
        let before = worker.diagnostics().submitted;
        let c = Controls {
            picked: [0.; 3],
            ..Default::default()
        };
        worker
            .request_material(request(&source, 1, c, Quality::Full))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::BudgetExceeded { required_bytes, budget_bytes } if required_bytes > budget && budget_bytes == budget),
            "{event:?}"
        );
        assert_eq!(worker.diagnostics().submitted, before);
        assert!(worker.try_latest_material().is_none());
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let result = gpu.evaluate(&worker, request(&source, 2, c, Quality::Full));
        assert!(result.iter().all(|&v| v == 1.0));
        assert!(worker.diagnostics().peak_bytes <= budget);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn submitted_metallic_generation_is_cancelled_at_scheduler_boundary_and_only_latest_can_publish() {
    let source = source(Extent::new(257, 193).unwrap());
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(&source, 1, Controls::default(), Quality::Full))
        .unwrap();
    let first = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(first.id));
    for id in 2..=64 {
        let c = Controls {
            threshold: id as f32 / 64.0,
            ..Default::default()
        };
        scheduler
            .request_material(request(&source, id, c, Quality::Full))
            .unwrap();
        assert!(!scheduler.should_continue(first.id));
        assert!(scheduler.start_next().is_none());
    }
    assert_eq!(scheduler.complete(first.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let last = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    assert_eq!(scheduler.try_latest_material().unwrap().request_id, 64);
    let stats = scheduler.diagnostics();
    assert_eq!(stats.peak_pending, 1);
    assert_eq!(stats.peak_in_flight, 1);
    assert_eq!(stats.stale_publications, 0);
    assert!(stats.cancellation_requests > 0);
}

#[test]
fn worker_supersession_eventually_publishes_latest_color_parameters_without_validation_errors() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(1024, 513).unwrap();
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: vec![0; (e.width * e.height * 4) as usize].into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        for id in 1..=64 {
            let c = Controls {
                picked: if id == 64 { [0.; 3] } else { [1.; 3] },
                feather: 0.0,
                ..Default::default()
            };
            worker
                .request_material(request(&source, id, c, Quality::Full))
                .expect("Metallic runtime support");
        }
        // An earlier generation may have completed before supersession. Drain it;
        // scheduler contract above checks rejection of a known cancelled token.
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("{event:?}");
            }
            worker
                .try_latest_material()
                .filter(|bundle| bundle.request_id == 64)
        });
        assert_eq!(bundle.generation, 64);
        let values = gpu.read(bundle.outputs[&MapKind::Metallic].texture());
        assert!(values.iter().all(|&v| v == 1.0));
        let stats = worker.diagnostics();
        assert_eq!(stats.stale_publications, 0);
        assert_eq!(stats.validation_errors, 0);
        assert_eq!(stats.peak_in_flight, 1);
        assert!(stats.peak_pending <= 1);
        assert!(stats.peak_bytes <= GIB);
        drop(bundle);
        worker.join().unwrap();
    });
}

#[test]
fn metallic_suite_is_registered_in_maps_and_all_and_empty_or_unknown_suites_fail() {
    for suite in ["metallic", "maps", "all"] {
        assert!(
            harness_contract::select_suite(suite)
                .expect("metallic suite must exist")
                .contains(&"metallic"),
            "missing metallic in {suite}"
        );
    }
    for suite in ["", " ", "metallics", "unknown"] {
        assert!(harness_contract::select_suite(suite).is_err());
    }
}

#[test]
fn full_node_timing_includes_every_pass_and_strict_sixteen_ms_gate_after_ten_warmups() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![5.0, 11.0]; 100]);
    let t = harness_contract::aggregate_timings(&rows, 10).unwrap();
    assert_eq!(t.measurements, 100);
    assert_eq!(t.p95_ms, 16.0);
    assert!(harness_contract::aggregate_timings(&rows[..10], 10).is_err());
    // Schema-only fixture; this is never written as hardware evidence.
    let valid = serde_json::json!({
        "passed": true, "suite": "metallic", "size": 4096,
        "adapter": {"name":"RTX 5070 Ti", "backend":"Dx12"},
        "seed":54, "manifest":{"picked_color":[0.5,0.5,0.5],"radius":0.25,"threshold":0.5,"feather":0.1,"mask_mode":"Replace"},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[{"name":"cpu_oracle", "passed":true}],
        "timing":{"kind":"completion_upper_bound", "scope":"full_node", "warmups":10,"measurements":100,"p95_ms":2.0,"submission_ms":0.1,"peak_bytes":536870912,"cancellation_errors":0},
        "maximum_quality":{"p95_ms":3.0}
    });
    assert!(harness_contract::validate_report(&valid).is_ok());
    for (path, value) in [
        ("/checks", serde_json::json!([])),
        ("/checks/0/passed", serde_json::json!(false)),
        ("/timing/p95_ms", serde_json::json!(t.p95_ms)),
        ("/timing/warmups", serde_json::json!(9)),
        ("/timing/measurements", serde_json::json!(99)),
        ("/timing/peak_bytes", serde_json::json!(0)),
        ("/timing/cancellation_errors", serde_json::json!(1)),
        ("/timing/kind", serde_json::json!("cpu_encode_only")),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            harness_contract::validate_report(&bad).is_err(),
            "accepted invalid report at {path}"
        );
    }
}
