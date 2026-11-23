use super::{
    harness_contract,
    oracle::Controls,
    support::{GIB, request, with_gpu},
};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::scheduler::{Completion, Scheduler};
use texelith_gpu::{MaterialError, MaterialPlan, Quality, ResourceReservations, TypedSourcePixels};

#[test]
fn real_worker_cancels_submitted_ao_and_recovers_without_stale_publications() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        // Generated in memory only. Enough work remains after submission to
        // observe cancellation, without relying on an arbitrary sleep.
        let e = Extent::new(1024, 513).unwrap();
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(super::oracle::seeded(e).into()),
        );
        let before = worker.diagnostics();
        let expensive = Controls {
            directions: 32,
            steps: 32,
            radius: 512.0,
            ..Default::default()
        };
        worker
            .request_material(request(&source, 1, expensive, Quality::Full))
            .expect("HorizonAo runtime support");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while worker.diagnostics().submitted == before.submitted {
            if let Some(event) = worker.try_event() {
                panic!("AO failed before submission: {event:?}");
            }
            assert!(std::time::Instant::now() < deadline, "AO never submitted");
            std::thread::yield_now();
        }
        let unoccluded = Controls {
            bias: 1e10,
            ..Default::default()
        };
        for id in 2..=32 {
            worker
                .request_material(request(&source, id, unoccluded, Quality::Full))
                .unwrap();
        }
        let bundle = super::support::wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("AO cancellation failed: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!((bundle.request_id, bundle.generation), (32, 32));
        let values = gpu.read(bundle.outputs[&MapKind::Ao].texture());
        assert!(values.iter().all(|&v| v == 1.0));
        let stats = worker.diagnostics();
        assert!(
            stats.cancellation_requests > before.cancellation_requests,
            "must cancel active GPU work, not merely replace pending requests"
        );
        assert!(stats.cancelled_pass_batches > before.cancelled_pass_batches);
        assert_eq!(stats.stale_publications, 0);
        assert_eq!(stats.validation_errors, 0);
        assert_eq!(stats.peak_in_flight, 1);
        assert_eq!(stats.peak_pending, 1);
        assert!(stats.peak_bytes <= GIB);
        drop(bundle);
        worker.join().unwrap();
    });
}

fn source() -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [51; 32],
        extent: Extent::new(257, 193).unwrap(),
        interpretation: InputInterpretation::LinearHeight,
    }
}

#[test]
fn controls_accept_adr_bounds_and_reject_nonfinite_or_out_of_range_values() {
    let base = Controls::default();
    for c in [
        base,
        Controls {
            directions: 4,
            steps: 4,
            radius: 1.0,
            bias: 0.0,
            ..base
        },
        Controls {
            directions: 32,
            steps: 32,
            radius: 512.0,
            ..base
        },
    ] {
        assert!(c.params().validate().is_ok());
    }
    for c in [
        Controls {
            directions: 3,
            ..base
        },
        Controls {
            directions: 33,
            ..base
        },
        Controls { steps: 3, ..base },
        Controls { steps: 33, ..base },
        Controls {
            radius: 0.99,
            ..base
        },
        Controls {
            radius: 512.01,
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
            bias: -0.01,
            ..base
        },
        Controls {
            bias: f32::NAN,
            ..base
        },
        Controls {
            bias: f32::INFINITY,
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
fn four_k_default_ao_plan_fits_shared_one_gib_ledger() {
    let source = SourceDescriptor {
        extent: Extent::new(4096, 4096).unwrap(),
        ..source()
    };
    let r = request(&source, 1, Controls::default(), Quality::Full);
    let reserve = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 128 * 1024 * 1024,
    };
    let plan =
        MaterialPlan::new(&r, GIB, reserve).expect("4K AO must fit with shared reservations");
    assert!(plan.peak_bytes() <= GIB);
    assert!(
        plan.peak_bytes() >= reserve.viewport_bytes + reserve.readback_bytes + 64 * 1024 * 1024
    );
}

#[test]
fn ao_report_requires_nonempty_checks_measured_samples_and_strict_default_p95() {
    // Schema fixture, never saved as hardware evidence. Max-quality timing can
    // exceed the default threshold, but must be recorded as a separate run.
    let report = serde_json::json!({
        "passed": true, "suite": "ao", "size": 4096,
        "adapter": {"name":"RTX 5070 Ti","backend":"Dx12"},
        "seed":51,"manifest":{"directions":8,"steps":8,"radius_source_px":32.0,"bias":0.01,"boundary":"repeat"},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[{"name":"cpu_oracle","passed":true}],
        "timing":{"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":100,
            "p95_ms":4.0,"submission_ms":0.1,"peak_bytes":536870912,"cancellation_errors":0},
        "maximum_quality":{"directions":32,"steps":32,"radius_source_px":512.0,"p95_ms":40.0}
    });
    assert!(harness_contract::validate_report(&report).is_ok());
    for (path, value) in [
        ("/checks", serde_json::json!([])),
        ("/checks/0/passed", serde_json::json!(false)),
        ("/timing/warmups", serde_json::json!(9)),
        ("/timing/measurements", serde_json::json!(99)),
        ("/timing/p95_ms", serde_json::json!(16.0)),
        ("/timing/p95_ms", serde_json::Value::Null),
        ("/timing/peak_bytes", serde_json::json!(0)),
        ("/timing/cancellation_errors", serde_json::json!(1)),
        ("/timing/kind", serde_json::json!("cpu_encode_only")),
    ] {
        let mut bad = report.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            harness_contract::validate_report(&bad).is_err(),
            "accepted invalid timing report at {path}"
        );
    }
}

#[test]
fn planner_accepts_ao_and_cache_identity_includes_every_control_and_quality() {
    let source = SourceDescriptor {
        extent: Extent::new(2048, 193).unwrap(),
        ..source()
    };
    let key = |id, c, q| {
        *MaterialPlan::new(
            &request(&source, id, c, q),
            GIB,
            ResourceReservations::default(),
        )
        .expect("HorizonAo planning must be supported")
        .output_keys()
        .get(&MapKind::Ao)
        .unwrap()
    };
    let base = Controls::default();
    let first = key(1, base, Quality::Full);
    assert_eq!(
        first,
        key(99, base, Quality::Full),
        "job/generation is not content"
    );
    for c in [
        Controls {
            directions: 9,
            ..base
        },
        Controls { steps: 9, ..base },
        Controls {
            radius: 33.0,
            ..base
        },
        Controls { bias: 0.02, ..base },
        Controls {
            boundary: Boundary::Clamp,
            ..base
        },
    ] {
        assert_ne!(first, key(2, c, Quality::Full));
    }
    assert_ne!(first, key(2, base, Quality::Proxy));
}

#[test]
fn planner_budget_includes_input_output_scratch_and_shared_reservations() {
    let r = request(&source(), 1, Controls::default(), Quality::Full);
    let reserved = ResourceReservations {
        viewport_bytes: 8192,
        readback_bytes: 4096,
    };
    let peak = MaterialPlan::new(&r, GIB, reserved)
        .expect("AO planner support")
        .peak_bytes();
    let two_maps = 2 * u64::from(r.extent.width) * u64::from(r.extent.height) * 4;
    assert!(peak >= two_maps + reserved.viewport_bytes);
    assert!(peak <= GIB);
    assert!(
        matches!(MaterialPlan::new(&r,peak-1,reserved),Err(MaterialError::BudgetExceeded {
        required_bytes,budget_bytes }) if required_bytes==peak && budget_bytes==peak-1)
    );
    assert!(MaterialPlan::new(&r, peak, reserved).is_ok());
}

#[test]
fn low_budget_worker_rejects_before_submission_then_recovers_with_a_small_ao_job() {
    with_gpu(|gpu| {
        let budget = 256 * 1024;
        let worker = gpu.worker(budget);
        let e = Extent::new(129, 129).unwrap();
        let source = gpu.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(vec![0.0; 129 * 129].into()),
        );
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: budget,
                readback_bytes: 0,
            })
            .unwrap();
        let submitted = worker.diagnostics().submitted;
        worker
            .request_material(request(&source, 1, Controls::default(), Quality::Full))
            .unwrap();
        let error = super::support::wait(|| worker.try_event());
        assert!(
            matches!(error, texelith_gpu::WorkerEvent::BudgetExceeded { required_bytes, budget_bytes }
                if required_bytes > budget && budget_bytes == budget),
            "expected a bounded budget rejection, got {error:?}"
        );
        assert_eq!(worker.diagnostics().submitted, submitted);
        assert!(worker.try_latest_material().is_none());
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let result = gpu.evaluate(
            &worker,
            request(&source, 2, Controls::default(), Quality::Full),
        );
        assert!(result.iter().all(|&v| v == 1.0));
        assert!(worker.diagnostics().peak_bytes <= budget);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn ao_cancellation_checks_between_passes_and_only_latest_generation_can_publish() {
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(&source(), 1, Controls::default(), Quality::Full))
        .unwrap();
    let first = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(first.id));
    for id in 2..=64 {
        let c = Controls {
            radius: id as f32,
            ..Default::default()
        };
        scheduler
            .request_material(request(&source(), id, c, Quality::Full))
            .unwrap();
        assert!(!scheduler.should_continue(first.id));
        assert!(scheduler.start_next().is_none());
        assert!(scheduler.try_latest_material().is_none());
    }
    assert_eq!(scheduler.complete(first.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let last = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    assert_eq!(scheduler.try_latest_material().unwrap().request_id, 64);
    assert_eq!(scheduler.diagnostics().peak_in_flight, 1);
    assert_eq!(scheduler.diagnostics().peak_pending, 1);
    assert_eq!(scheduler.diagnostics().stale_publications, 0);
    assert!(scheduler.diagnostics().cancellation_requests > 0);
}

#[test]
fn ao_suite_is_registered_in_maps_and_all_and_unknown_or_empty_suites_fail() {
    for name in ["ao", "maps", "all"] {
        assert!(
            harness_contract::select_suite(name)
                .unwrap()
                .contains(&"ao"),
            "missing AO in {name}"
        );
    }
    for name in ["", " ", "aos", "unknown"] {
        assert!(harness_contract::select_suite(name).is_err());
    }
}

#[test]
fn full_node_timing_sums_passes_and_excludes_only_the_ten_warmups() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![3.0, 13.0]; 100]);
    let timing = harness_contract::aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(
        timing.p95_ms, 16.0,
        "a 16 ms total fails the strict <16 ms default gate"
    );
    assert!(harness_contract::aggregate_timings(&rows[..10], 10).is_err());
}
