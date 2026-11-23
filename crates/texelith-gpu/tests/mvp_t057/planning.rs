use super::{
    behavior::{evaluate, height, request, wait},
    gpu::with_gpu,
    reference,
};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuWorker, MaterialError, MaterialPlan, ResourceReservations, WorkerEvent,
};

const GIB: u64 = 1_073_741_824;

fn descriptor(extent: Extent, color: bool) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [if color { 1 } else { 2 }; 32],
        extent,
        interpretation: if color {
            InputInterpretation::SrgbPhoto
        } else {
            InputInterpretation::LinearHeight
        },
    }
}

#[test]
fn full_4k_correlated_plan_reserves_viewport_and_rejects_one_byte_below_peak() {
    let extent = Extent::new(4096, 4096).unwrap();
    for sources in [
        vec![descriptor(extent, false)],
        vec![descriptor(extent, true)],
        vec![descriptor(extent, true), descriptor(extent, false)],
    ] {
        let r = request(&sources, 1, Some(0.25), Boundary::Repeat);
        let reservations = ResourceReservations {
            viewport_bytes: 144 * 1024 * 1024,
            readback_bytes: 0,
        };
        let plan = MaterialPlan::new(&r, GIB, reservations).unwrap();
        // Independent lower bound: all requested outputs plus the viewport share.
        let output_bytes: u64 = sources
            .iter()
            .map(|s| {
                u64::from(extent.width)
                    * u64::from(extent.height)
                    * if s.interpretation == InputInterpretation::SrgbPhoto {
                        8
                    } else {
                        4
                    }
            })
            .sum();
        assert!(plan.peak_bytes() >= output_bytes + reservations.viewport_bytes);
        assert!(plan.peak_bytes() <= GIB);
        assert!(
            matches!(MaterialPlan::new(&r,plan.peak_bytes()-1,reservations),
            Err(MaterialError::BudgetExceeded { required_bytes, budget_bytes })
            if required_bytes == plan.peak_bytes() && budget_bytes == plan.peak_bytes()-1)
        );
    }
}

#[test]
fn cache_keys_distinguish_overlap_boundary_input_content_and_semantic_and_ignore_generation() {
    let e = Extent::new(33, 19).unwrap();
    let source = descriptor(e, false);
    let key = |s: &SourceDescriptor, overlap, boundary, serial| {
        *MaterialPlan::new(
            &request(std::slice::from_ref(s), serial, Some(overlap), boundary),
            GIB,
            ResourceReservations::default(),
        )
        .unwrap()
        .output_keys()
        .values()
        .next()
        .unwrap()
    };
    let base = key(&source, 0.25, Boundary::Repeat, 1);
    assert_eq!(base, key(&source, 0.25, Boundary::Repeat, 2));
    assert_ne!(base, key(&source, 0.1, Boundary::Repeat, 3));
    assert_ne!(base, key(&source, 0.5, Boundary::Repeat, 4));
    assert_ne!(base, key(&source, 0.25, Boundary::Clamp, 5));
    let mut changed = source.clone();
    changed.content_hash[0] ^= 1;
    assert_ne!(base, key(&changed, 0.25, Boundary::Repeat, 6));
    changed.interpretation = InputInterpretation::SrgbPhoto;
    assert_ne!(base, key(&changed, 0.25, Boundary::Repeat, 7));
}

#[test]
fn offset_cache_identity_is_distinct_from_splat_and_histogram_methods() {
    use texelith_core::{
        CacheKey, CacheRevisions, MaterialGraphDto, MaterialNodeParams, OutputPort, SeamlessBase,
        SeamlessMethod, material_node_cache_key,
    };
    let source = descriptor(Extent::new(7, 5).unwrap(), false);
    let r = request(
        std::slice::from_ref(&source),
        1,
        Some(0.25),
        Boundary::Repeat,
    );
    let mut dto = MaterialGraphDto::from(r.snapshot.graph.clone());
    let key = |node: &texelith_core::MaterialNode| {
        material_node_cache_key(
            node,
            OutputPort::Height,
            &[CacheKey([57; 32])],
            source.extent,
            source.extent,
            CacheRevisions::CURRENT,
        )
        .unwrap()
    };
    let offset = key(&dto.graph[1]);
    for method in [
        SeamlessMethod::Splat {
            seed: 57,
            grid: 4,
            overlap: 0.25,
        },
        SeamlessMethod::Histogram {
            base: SeamlessBase::OffsetBlend { overlap: 0.25 },
            strength: 1.0,
        },
    ] {
        dto.graph[1].params = MaterialNodeParams::Seamless {
            method,
            boundary: Boundary::Repeat,
        };
        assert_ne!(offset, key(&dto.graph[1]));
    }
}

#[test]
fn offset_requests_cancel_in_latest_slot_without_stale_or_mixed_publication() {
    use texelith_gpu::scheduler::{Completion, Scheduler};
    let e = Extent::new(257, 193).unwrap();
    let sources = [descriptor(e, true), descriptor(e, false)];
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(&sources, 1, Some(0.25), Boundary::Repeat))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(active.id));
    for id in 2..=64 {
        scheduler
            .request_material(request(&sources, id, Some(0.5), Boundary::Repeat))
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
    assert_eq!(result.request_id, 64);
    assert_eq!(result.snapshot.generation, 64);
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(scheduler.diagnostics().peak_in_flight, 1);
    assert_eq!(scheduler.diagnostics().peak_pending, 1);
    assert_eq!(scheduler.diagnostics().stale_publications, 0);
    assert!(scheduler.diagnostics().cancellation_requests > 0);
}

#[test]
fn low_budget_worker_rejects_without_partial_outputs_and_recovers_after_reservation_release() {
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
        let input = reference::scalar_fixture(e, "seeded");
        let source = height(&worker, e, input.clone());
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
                Some(0.25),
                Boundary::Repeat,
            ))
            .unwrap();
        assert!(
            matches!(wait(|| worker.try_event()),WorkerEvent::BudgetExceeded { budget_bytes:65536,required_bytes } if required_bytes > 65536)
        );
        assert!(worker.try_latest_material().is_none());
        wait(|| {
            (worker.diagnostics().in_flight == 0 && worker.diagnostics().pending == 0).then_some(())
        });
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let result = evaluate(
            gpu,
            &worker,
            request(&[source], 2, Some(0.25), Boundary::Repeat),
        );
        let expected = reference::render(
            e,
            &input.into_iter().map(f64::from).collect::<Vec<_>>(),
            0.25,
            Boundary::Repeat,
        );
        super::behavior::compare(
            result[&MapKind::Height].iter().map(|p| p[0]),
            &expected,
            2e-5,
        );
        assert!(worker.diagnostics().peak_bytes <= 65536);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn real_worker_replaces_an_in_flight_request_and_publishes_the_latest_offset() {
    use std::time::{Duration, Instant};
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(1024, 512).unwrap();
        let input = reference::scalar_fixture(e, "seeded");
        let source = height(&worker, e, input.clone());
        worker
            .request_material(request(
                std::slice::from_ref(&source),
                1,
                Some(0.1),
                Boundary::Repeat,
            ))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            assert!(worker.try_event().is_none(), "cold offset request failed");
            if worker.diagnostics().in_flight == 1 {
                break;
            }
            assert!(
                worker.try_latest_material().is_none(),
                "request finished before cancellation was exercised"
            );
            assert!(
                Instant::now() < deadline,
                "offset never reached active work"
            );
            std::thread::yield_now();
        }
        let result = evaluate(
            gpu,
            &worker,
            request(&[source], 2, Some(0.5), Boundary::Repeat),
        );
        let expected = reference::render(
            e,
            &input.into_iter().map(f64::from).collect::<Vec<_>>(),
            0.5,
            Boundary::Repeat,
        );
        super::behavior::compare(
            result[&MapKind::Height].iter().map(|p| p[0]),
            &expected,
            2e-5,
        );
        let stats = worker.diagnostics();
        assert!(
            stats.cancellation_requests > 0,
            "real cancellation must be exercised"
        );
        assert_eq!(stats.stale_publications, 0);
        assert_eq!(stats.peak_in_flight, 1);
        assert_eq!(stats.peak_pending, 1);
        assert!(stats.peak_bytes <= GIB);
        worker.join().unwrap();
    });
}
