use super::{gpu::with_gpu, reference, support::*};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuWorker, MaterialError, MaterialPlan, ResourceReservations, WorkerEvent,
};

const GIB: u64 = 1_073_741_824;
fn descriptor(e: Extent, color: bool) -> SourceDescriptor {
    SourceDescriptor {
        extent: e,
        content_hash: [if color { 1 } else { 2 }; 32],
        interpretation: if color {
            InputInterpretation::SrgbPhoto
        } else {
            InputInterpretation::LinearHeight
        },
    }
}

#[test]
fn four_k_color_scalar_and_correlated_plans_reserve_viewport_and_reject_peak_minus_one_byte() {
    let e = Extent::new(4096, 4096).unwrap();
    for sources in [
        vec![descriptor(e, false)],
        vec![descriptor(e, true)],
        vec![descriptor(e, true), descriptor(e, false)],
    ] {
        for grid in [2, 4, 8] {
            let r = request(
                &sources,
                1,
                Some(Controls {
                    grid,
                    ..Default::default()
                }),
            );
            let reserved = ResourceReservations {
                viewport_bytes: 144 * 1024 * 1024,
                readback_bytes: 0,
            };
            let plan = MaterialPlan::new(&r, GIB, reserved).expect("splat plan must be supported");
            let output_bytes: u64 = sources
                .iter()
                .map(|s| {
                    u64::from(e.width)
                        * u64::from(e.height)
                        * if s.interpretation == InputInterpretation::SrgbPhoto {
                            8
                        } else {
                            4
                        }
                })
                .sum();
            assert!(plan.peak_bytes() >= output_bytes + reserved.viewport_bytes);
            assert!(plan.peak_bytes() <= GIB);
            assert!(
                matches!(MaterialPlan::new(&r,plan.peak_bytes()-1,reserved),Err(MaterialError::BudgetExceeded {required_bytes,budget_bytes}) if required_bytes==plan.peak_bytes() && budget_bytes==plan.peak_bytes()-1)
            );
            assert!(MaterialPlan::new(&r, plan.peak_bytes(), reserved).is_ok());
        }
    }
}

#[test]
fn cache_keys_include_seed_grid_overlap_boundary_content_and_semantics_but_not_generation() {
    let source = descriptor(Extent::new(33, 19).unwrap(), false);
    let key = |s: &SourceDescriptor, c: Controls, serial| {
        *MaterialPlan::new(
            &request(std::slice::from_ref(s), serial, Some(c)),
            GIB,
            Default::default(),
        )
        .unwrap()
        .output_keys()
        .values()
        .next()
        .unwrap()
    };
    let c = Controls::default();
    let base = key(&source, c, 1);
    assert_eq!(base, key(&source, c, 99));
    for changed in [
        Controls { seed: 59, ..c },
        Controls { grid: 8, ..c },
        Controls { overlap: 0.1, ..c },
        Controls {
            boundary: Boundary::Clamp,
            ..c
        },
    ] {
        assert_ne!(base, key(&source, changed, 2));
    }
    let mut changed = source.clone();
    changed.content_hash[0] ^= 1;
    assert_ne!(base, key(&changed, c, 3));
    changed = source.clone();
    changed.interpretation = InputInterpretation::SrgbPhoto;
    assert_ne!(base, key(&changed, c, 4));
}

#[test]
fn invalid_grid_overlap_and_nonfinite_parameters_are_rejected_without_gpu_work() {
    for grid in [0, 1, 9, u32::MAX] {
        assert!(
            Controls {
                grid,
                ..Default::default()
            }
            .method()
            .validate()
            .is_err()
        );
    }
    for overlap in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.099,
        0.501,
    ] {
        assert!(
            Controls {
                overlap,
                ..Default::default()
            }
            .method()
            .validate()
            .is_err()
        );
    }
    for grid in 2..=8 {
        for overlap in [0.1, 0.25, 0.5] {
            assert!(
                Controls {
                    seed: u32::MAX,
                    grid,
                    overlap,
                    ..Default::default()
                }
                .method()
                .validate()
                .is_ok()
            );
        }
    }
}

#[test]
fn latest_slot_cancels_seed_edits_without_publishing_mixed_correlated_outputs() {
    use texelith_gpu::scheduler::{Completion, Scheduler};
    let e = Extent::new(257, 193).unwrap();
    let sources = [descriptor(e, true), descriptor(e, false)];
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(&sources, 1, Some(Controls::default())))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(active.id));
    for id in 2..=64 {
        scheduler
            .request_material(request(
                &sources,
                id,
                Some(Controls {
                    seed: id as u32,
                    ..Default::default()
                }),
            ))
            .unwrap();
        assert!(!scheduler.should_continue(active.id));
        assert!(scheduler.start_next().is_none());
        assert!(scheduler.try_latest_material().is_none());
    }
    assert_eq!(scheduler.complete(active.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let last = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(last.id), Completion::Published);
    let result = scheduler.try_latest_material().unwrap();
    assert_eq!(
        (
            result.request_id,
            result.snapshot.generation,
            result.outputs.len()
        ),
        (64, 64, 2)
    );
    let d = scheduler.diagnostics();
    assert_eq!(d.peak_in_flight, 1);
    assert_eq!(d.peak_pending, 1);
    assert_eq!(d.stale_publications, 0);
    assert!(d.cancellation_requests > 0);
}

#[test]
fn low_budget_worker_rejects_atomically_and_recovers_when_reservation_is_released() {
    with_gpu(|gpu| {
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: 65536,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let e = Extent::new(16, 16).unwrap();
        let source = height(&worker, e, reference::fixture(e, "seeded"));
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: 65536,
                readback_bytes: 0,
            })
            .unwrap();
        worker
            .request_material(request(
                std::slice::from_ref(&source),
                1,
                Some(Controls::default()),
            ))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event,WorkerEvent::BudgetExceeded {budget_bytes:65536,required_bytes} if required_bytes>65536),
            "expected budget rejection, got {event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        wait(|| {
            (worker.diagnostics().in_flight == 0 && worker.diagnostics().pending == 0).then_some(())
        });
        worker
            .set_resource_reservations(Default::default())
            .unwrap();
        let actual = evaluate(
            gpu,
            &worker,
            request(&[source], 2, Some(Controls::default())),
        );
        assert_eq!(actual.len(), 1);
        assert!(worker.diagnostics().peak_bytes <= 65536);
        worker.join().unwrap();
    });
}

#[test]
fn real_worker_cancels_active_seed_and_only_publishes_latest_coherent_generation() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1024, 512).unwrap();
        let source = height(&worker, e, reference::fixture(e, "seeded"));
        worker
            .request_material(request(
                std::slice::from_ref(&source),
                1,
                Some(Controls::default()),
            ))
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            assert!(worker.try_event().is_none(), "cold splat request failed");
            if worker.diagnostics().in_flight == 1 {
                break;
            }
            assert!(
                worker.try_latest_material().is_none(),
                "request completed before cancellation was exercised"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "splat did not reach active work"
            );
            std::thread::yield_now();
        }
        let c = Controls {
            seed: 59,
            ..Default::default()
        };
        let actual = evaluate(gpu, &worker, request(&[source], 2, Some(c)));
        let input: Vec<_> = reference::fixture(e, "seeded")
            .into_iter()
            .map(f64::from)
            .collect();
        let expected = reference::render(
            e,
            e,
            &input,
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
        let d = worker.diagnostics();
        assert!(d.cancellation_requests > 0);
        assert_eq!(d.stale_publications, 0);
        assert_eq!(d.peak_in_flight, 1);
        assert_eq!(d.peak_pending, 1);
        assert!(d.peak_bytes <= GIB);
        worker.join().unwrap();
    });
}
