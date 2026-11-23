use super::{
    gpu::{decode, with_gpu},
    normal::{Controls, evaluate, install_f32, request, wait},
    reference as cpu,
};
use texelith_core::{Boundary, Extent, MapKind, NormalConvention, NormalFilter};
use texelith_gpu::{
    GpuConfig, GpuWorker, MaterialError, MaterialPlan, ResourceReservations, TypedSourcePixels,
    scheduler::{Completion, Scheduler},
};

const GIB: u64 = 1024 * 1024 * 1024;
fn source(e: Extent) -> texelith_core::SourceDescriptor {
    TypedSourcePixels::HeightF32(vec![0.0; (e.width * e.height) as usize].into())
        .descriptor(e)
        .unwrap()
}

#[test]
fn planner_admits_every_normal_mode_at_4k_and_rejects_peak_minus_one() {
    let source = source(Extent::new(4096, 4096).unwrap());
    let reservations = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 0,
    };
    for filter in [
        NormalFilter::Central,
        NormalFilter::Sobel,
        NormalFilter::Scharr,
    ] {
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                for weights in [vec![1.0], vec![0.25; 16]] {
                    let c = Controls {
                        filter,
                        boundary,
                        convention,
                        weights,
                        ..Default::default()
                    };
                    let r = request(&source, 1, &c);
                    let plan = MaterialPlan::new(&r, GIB, reservations)
                        .expect("normal mode must be budget planned");
                    assert!(plan.peak_bytes() > reservations.viewport_bytes);
                    assert!(plan.peak_bytes() <= GIB);
                    assert!(matches!(
                        MaterialPlan::new(&r, plan.peak_bytes() - 1, reservations),
                        Err(MaterialError::BudgetExceeded { .. })
                    ));
                }
            }
        }
    }
}

#[test]
fn output_cache_keys_include_all_controls_and_exclude_job_and_generation() {
    let source = source(Extent::new(33, 19).unwrap());
    let c = Controls::default();
    let key = |r| {
        *MaterialPlan::new(&r, GIB, ResourceReservations::default())
            .unwrap()
            .output_keys()
            .get(&MapKind::Normal)
            .unwrap()
    };
    let base = key(request(&source, 1, &c));
    let mut equivalent = request(&source, 9, &c);
    equivalent.job_id = texelith_gpu::JobId(99);
    assert_eq!(base, key(equivalent));
    for changed in [
        Controls {
            strength: 2.0,
            ..c.clone()
        },
        Controls {
            filter: NormalFilter::Scharr,
            ..c.clone()
        },
        Controls {
            filter: NormalFilter::Central,
            ..c.clone()
        },
        Controls {
            weights: vec![0.5, 0.5],
            ..c.clone()
        },
        Controls {
            boundary: Boundary::Clamp,
            ..c.clone()
        },
        Controls {
            convention: NormalConvention::DirectX,
            ..c.clone()
        },
    ] {
        assert_ne!(base, key(request(&source, 2, &changed)));
    }
}

#[test]
fn normal_scheduler_replaces_pending_work_and_discards_stale_generation() {
    let source = source(Extent::new(33, 19).unwrap());
    let c = Controls {
        weights: vec![0.25; 16],
        ..Default::default()
    };
    let mut scheduler = Scheduler::default();
    scheduler.request_material(request(&source, 1, &c)).unwrap();
    let first = scheduler.start_next().unwrap();
    for serial in 2..=64 {
        scheduler
            .request_material(request(&source, serial, &c))
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
    let d = scheduler.diagnostics();
    assert_eq!(d.peak_in_flight, 1);
    assert_eq!(d.peak_pending, 1);
    assert!(d.cancellation_requests > 0);
    assert_eq!(d.stale_publications, 0);
}

#[test]
fn real_low_budget_normal_request_rejects_without_partial_output_then_recovers() {
    with_gpu(|gpu| {
        let budget = 600 * 1024;
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: budget,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let e = Extent::new(257, 193).unwrap();
        let source = install_f32(&worker, e, cpu::seeded(e));
        let c = Controls {
            weights: vec![0.25; 16],
            ..Default::default()
        };
        let r = request(&source, 1, &c);
        assert!(
            matches!(
                MaterialPlan::new(&r, budget, ResourceReservations::default()),
                Err(MaterialError::BudgetExceeded { .. })
            ),
            "unsupported is not budget rejection"
        );
        worker.request_material(r).unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, texelith_gpu::WorkerEvent::BudgetExceeded {
            budget_bytes, required_bytes
        } if budget_bytes == budget && required_bytes > budget),
            "{event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        assert_eq!(worker.diagnostics().submitted, 0);
        let tiny = Extent::new(17, 13).unwrap();
        let data = cpu::ramp(tiny, 0.25, -0.125);
        let small = install_f32(&worker, tiny, data.clone());
        let c = Controls::default();
        let actual = decode(&evaluate(
            gpu,
            &worker,
            request(&small, 2, &c),
            c.convention,
        ));
        cpu::compare(
            &actual,
            &cpu::normals(tiny, &data, c.filter, c.boundary, c.strength, c.convention),
        );
        assert!(worker.diagnostics().peak_bytes <= budget);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn real_multi_scale_worker_observes_cancellation_between_submitted_passes() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1024, 1024).unwrap();
        let source = install_f32(&worker, e, cpu::seeded(e));
        let c = Controls {
            filter: NormalFilter::Scharr,
            weights: vec![0.25; 16],
            ..Default::default()
        };
        worker.request_material(request(&source, 1, &c)).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while worker.diagnostics().submitted < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "normal never reached submitted passes"
            );
            assert!(
                worker.try_latest_material().is_none(),
                "cold multi-scale normal completed before cancellation"
            );
            if let Some(event) = worker.try_event() {
                panic!("normal cancellation setup failed: {event:?}");
            }
            std::thread::yield_now();
        }
        let zero = Controls {
            weights: vec![0.0; 16],
            ..c
        };
        for serial in 2..=65 {
            worker
                .request_material(request(&source, serial, &zero))
                .unwrap();
        }
        let bundle = wait(|| {
            assert!(worker.try_event().is_none());
            worker.try_latest_material()
        });
        assert_eq!(bundle.request_id, 65);
        assert_eq!(bundle.generation, 65);
        assert_eq!(bundle.normal_convention, Some(zero.convention));
        let actual = decode(&gpu.read(bundle.outputs[&MapKind::Normal].texture()));
        cpu::compare(&actual, &vec![[0.0, 0.0, 1.0]; actual.len()]);
        let d = worker.diagnostics();
        assert!(d.cancellation_requests > 0);
        assert!(d.cancelled_pass_batches > 0);
        assert_eq!(d.stale_publications, 0);
        assert_eq!(d.peak_in_flight, 1);
        assert_eq!(d.peak_pending, 1);
        assert_eq!(d.validation_errors, 0);
        assert!(d.peak_bytes <= GIB);
        drop(bundle);
        worker.join().unwrap();
    });
}
