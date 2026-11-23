use super::{gpu::with_gpu, reference as r, support::*};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};
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
fn four_k_plans_admit_both_bases_and_correlated_outputs_and_charge_histogram_scratch() {
    let e = Extent::new(4096, 4096).unwrap();
    for sources in [
        vec![descriptor(e, false)],
        vec![descriptor(e, true)],
        vec![descriptor(e, true), descriptor(e, false)],
    ] {
        for base in BASES {
            let reservations = ResourceReservations {
                viewport_bytes: 144 * 1024 * 1024,
                readback_bytes: 0,
            };
            let request = request(
                &sources,
                1,
                Some(Controls {
                    base,
                    ..Default::default()
                }),
            );
            let plan = MaterialPlan::new(&request, GIB, reservations)
                .expect("histogram mode must be supported");
            let map_bytes: Vec<u64> = sources
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
                .collect();
            let output_bytes: u64 = map_bytes.iter().sum();
            // At least one 1024-u32 histogram must be resident beyond the source,
            // converted source, output and viewport payloads. This lower bound
            // does not prescribe a particular multi-pass scratch representation.
            // Correlated graphs can retire one converted input before processing
            // the next, so do not charge every source/output pair simultaneously.
            assert!(
                plan.peak_bytes()
                    >= reservations.viewport_bytes
                        + output_bytes * 2
                        + map_bytes.iter().min().unwrap()
                        + 4096
            );
            assert!(plan.peak_bytes() <= GIB);
            let peak = plan.peak_bytes();
            assert!(
                matches!(MaterialPlan::new(&request,peak-1,reservations),Err(MaterialError::BudgetExceeded {required_bytes,budget_bytes}) if required_bytes==peak && budget_bytes==peak-1)
            );
            assert!(MaterialPlan::new(&request, peak, reservations).is_ok());
        }
    }
}

#[test]
fn output_cache_identity_includes_source_semantics_base_strength_and_boundary() {
    let source = descriptor(Extent::new(33, 19).unwrap(), false);
    let key = |s: &SourceDescriptor, c, serial| {
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
    let original = key(&source, c, 1);
    assert_eq!(original, key(&source, c, 99));
    for changed in [
        Controls { strength: 0.0, ..c },
        Controls { strength: 0.5, ..c },
        Controls {
            base: Base::Offset(0.1),
            ..c
        },
        Controls {
            base: BASES[1],
            ..c
        },
        Controls {
            base: Base::Splat {
                seed: 60,
                grid: 4,
                overlap: 0.25,
            },
            ..c
        },
        Controls {
            base: Base::Splat {
                seed: 59,
                grid: 8,
                overlap: 0.5,
            },
            ..c
        },
        Controls {
            boundary: Boundary::Clamp,
            ..c
        },
    ] {
        assert_ne!(original, key(&source, changed, 2));
    }
    let mut changed = source.clone();
    changed.content_hash[0] ^= 1;
    assert_ne!(original, key(&changed, c, 3));
    changed = source.clone();
    changed.interpretation = InputInterpretation::SrgbPhoto;
    assert_ne!(original, key(&changed, c, 4));
    changed = source.clone();
    changed.extent = Extent::new(19, 33).unwrap();
    assert_ne!(original, key(&changed, c, 5));
    // Base parameters must still distinguish keys WITHIN histogram splat mode.
    let splat = Controls {
        base: BASES[1],
        ..c
    };
    let splat_key = key(&source, splat, 5);
    for base in [
        Base::Splat {
            seed: 60,
            grid: 4,
            overlap: 0.25,
        },
        Base::Splat {
            seed: 59,
            grid: 2,
            overlap: 0.25,
        },
        Base::Splat {
            seed: 59,
            grid: 4,
            overlap: 0.5,
        },
    ] {
        assert_ne!(splat_key, key(&source, Controls { base, ..splat }, 6));
    }
}

#[test]
fn invalid_strength_and_base_parameters_fail_before_dispatch() {
    for strength in [-0.001, 1.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            Controls {
                strength,
                ..Default::default()
            }
            .method()
            .validate()
            .is_err()
        );
    }
    for base in [
        Base::Offset(0.099),
        Base::Offset(0.501),
        Base::Offset(f32::NAN),
        Base::Splat {
            seed: 59,
            grid: 1,
            overlap: 0.25,
        },
        Base::Splat {
            seed: 59,
            grid: 9,
            overlap: 0.25,
        },
        Base::Splat {
            seed: 59,
            grid: 4,
            overlap: f32::INFINITY,
        },
    ] {
        assert!(
            Controls {
                base,
                ..Default::default()
            }
            .method()
            .validate()
            .is_err()
        );
    }
    for base in [
        Base::Offset(0.1),
        Base::Offset(0.5),
        Base::Splat {
            seed: u32::MAX,
            grid: 2,
            overlap: 0.1,
        },
        Base::Splat {
            seed: 0,
            grid: 8,
            overlap: 0.5,
        },
    ] {
        for strength in [0.0, 0.5, 1.0] {
            assert!(
                Controls {
                    base,
                    strength,
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
fn scheduler_coalesces_strength_edits_and_discards_stale_correlated_generation() {
    use texelith_gpu::scheduler::{Completion, Scheduler};
    let e = Extent::new(257, 193).unwrap();
    let sources = [descriptor(e, true), descriptor(e, false)];
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(&sources, 1, Some(Controls::default())))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(active.id));
    for serial in 2..=64 {
        scheduler
            .request_material(request(
                &sources,
                serial,
                Some(Controls {
                    strength: serial as f32 / 64.0,
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
    let latest = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(latest.id), Completion::Published);
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
    assert_eq!(
        (d.peak_in_flight, d.peak_pending, d.stale_publications),
        (1, 1, 0)
    );
    assert!(d.cancellation_requests > 0);
}

#[test]
fn low_budget_rejects_atomically_then_recovers_with_charged_histogram_allocations() {
    with_gpu(|g| {
        let worker = GpuWorker::spawn(
            g.device.clone(),
            g.queue.clone(),
            GpuConfig {
                budget_bytes: 262144,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let e = Extent::new(16, 16).unwrap();
        let source = height(&worker, e, r::fixture(e, "seeded"));
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: 262144,
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
            matches!(event,WorkerEvent::BudgetExceeded {budget_bytes:262144,required_bytes} if required_bytes>262144),
            "expected budget rejection: {event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        wait(|| {
            (worker.diagnostics().in_flight == 0 && worker.diagnostics().pending == 0).then_some(())
        });
        worker
            .set_resource_reservations(Default::default())
            .unwrap();
        let actual = evaluate(g, &worker, request(&[source], 2, Some(Controls::default())));
        assert_eq!(actual.len(), 1);
        assert!(worker.diagnostics().peak_bytes <= 262144);
        worker.join().unwrap();
    });
}

#[test]
fn real_worker_cancels_active_histogram_and_only_publishes_latest_result() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1024, 512).unwrap();
        let input = r::fixture(e, "seeded");
        let source = height(&worker, e, input.clone());
        worker
            .request_material(request(
                std::slice::from_ref(&source),
                1,
                Some(Controls::default()),
            ))
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while worker.diagnostics().in_flight != 1 {
            assert!(
                worker.try_event().is_none(),
                "cold histogram request failed"
            );
            assert!(
                worker.try_latest_material().is_none(),
                "work completed before cancellation was exercised"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "histogram never became active"
            );
            std::thread::yield_now();
        }
        let c = Controls {
            base: BASES[1],
            strength: 0.5,
            ..Default::default()
        };
        let actual = evaluate(g, &worker, request(&[source], 2, Some(c)));
        let rgb: Vec<_> = input.iter().map(|v| [f64::from(*v); 3]).collect();
        let expected = r::render(
            e,
            e,
            &rgb,
            &c.base.core(),
            c.boundary,
            f64::from(c.strength),
            false,
        );
        compare(
            actual[&texelith_core::MapKind::Height].iter().map(|p| p[0]),
            &expected.iter().map(|p| p[0]).collect::<Vec<_>>(),
            2e-5,
        );
        let d = worker.diagnostics();
        assert!(d.cancellation_requests > 0);
        assert_eq!(
            (
                d.peak_in_flight,
                d.peak_pending,
                d.stale_publications,
                d.validation_errors
            ),
            (1, 1, 0, 0)
        );
        assert!(d.peak_bytes <= GIB);
        worker.join().unwrap();
    });
}
