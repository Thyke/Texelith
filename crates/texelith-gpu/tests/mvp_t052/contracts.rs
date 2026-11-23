use super::{
    oracle::Controls,
    support::{self, GIB, PORTS, request, with_gpu},
};
use std::collections::BTreeSet;
use texelith_core::{
    Boundary, Extent, InputInterpretation, MapKind, NormalConvention, OutputPort, SourceDescriptor,
};
use texelith_gpu::{
    JobId, MaterialError, MaterialPlan, ResourceReservations, WorkerEvent,
    scheduler::{Completion, Scheduler},
};

fn source(e: Extent) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [52; 32],
        extent: e,
        interpretation: InputInterpretation::LinearNormal,
    }
}

#[test]
fn scales_accept_zero_and_finite_max_but_reject_negative_nan_and_infinity() {
    let base = Controls::default();
    for value in [0.0, 1.0, f32::MAX] {
        assert!(
            Controls {
                edge: value,
                cavity: value,
                ..base
            }
            .params()
            .validate()
            .is_ok()
        );
    }
    for value in [-0.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            Controls {
                edge: value,
                ..base
            }
            .params()
            .validate()
            .is_err()
        );
        assert!(
            Controls {
                cavity: value,
                ..base
            }
            .params()
            .validate()
            .is_err()
        );
    }
}

#[test]
fn output_keys_distinguish_ports_and_controls_but_not_job_or_generation() {
    let s = source(Extent::new(31, 19).unwrap());
    let c = Controls::default();
    let keys = |r| {
        MaterialPlan::new(&r, GIB, ResourceReservations::default())
            .expect("Curvature must be planned")
            .output_keys()
            .clone()
    };
    let baseline = keys(request(&s, 1, c, NormalConvention::OpenGl, &PORTS));
    assert_eq!(
        baseline
            .values()
            .map(|key| key.0)
            .collect::<BTreeSet<_>>()
            .len(),
        3,
        "named ports cannot alias one primary cache key"
    );
    let mut other_identity = request(&s, 99, c, NormalConvention::OpenGl, &PORTS);
    other_identity.job_id = JobId(1234);
    assert_eq!(keys(other_identity), baseline);
    for port in PORTS {
        let single = keys(request(&s, 2, c, NormalConvention::OpenGl, &[port]));
        assert_eq!(single[&port.map_kind()], baseline[&port.map_kind()]);
    }
    for changed in [
        Controls { edge: 0.0, ..c },
        Controls { cavity: 2.0, ..c },
        Controls {
            boundary: Boundary::Clamp,
            ..c
        },
    ] {
        let edited = keys(request(&s, 3, changed, NormalConvention::OpenGl, &PORTS));
        assert_ne!(edited, baseline);
    }
    let dx = keys(request(&s, 4, c, NormalConvention::DirectX, &PORTS));
    for kind in [MapKind::Curvature, MapKind::Edge, MapKind::Cavity] {
        assert_ne!(
            dx[&kind], baseline[&kind],
            "convention must invalidate {kind:?}"
        );
    }
}

#[test]
fn four_k_three_output_plan_charges_all_planes_and_shared_reservations() {
    let s = source(Extent::new(4096, 4096).unwrap());
    let reserves = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 128 * 1024 * 1024,
    };
    let r = request(&s, 1, Controls::default(), NormalConvention::OpenGl, &PORTS);
    let plan = MaterialPlan::new(&r, GIB, reserves)
        .expect("4K curvature must fit the shared 1 GiB ledger");
    let scalar = 4096_u64 * 4096 * 4;
    assert!(plan.peak_bytes() >= reserves.viewport_bytes + reserves.readback_bytes + scalar * 3);
    assert!(plan.peak_bytes() <= GIB);
    assert_eq!(plan.output_keys().len(), 3);
    let peak = plan.peak_bytes();
    assert!(
        matches!(MaterialPlan::new(&r, peak - 1, reserves), Err(MaterialError::BudgetExceeded { required_bytes, budget_bytes }) if required_bytes == peak && budget_bytes == peak - 1)
    );
    assert!(MaterialPlan::new(&r, peak, reserves).is_ok());
}

#[test]
fn mismatched_material_kind_and_empty_output_requests_are_rejected() {
    let s = source(Extent::new(17, 13).unwrap());
    let mut r = request(
        &s,
        1,
        Controls::default(),
        NormalConvention::OpenGl,
        &[OutputPort::Edge],
    );
    r.outputs = [(MapKind::Edge, support::output(OutputPort::Cavity))].into();
    assert!(matches!(
        MaterialPlan::new(&r, GIB, ResourceReservations::default()),
        Err(MaterialError::InvalidRequest(_))
    ));
    r.outputs.clear();
    assert!(matches!(
        MaterialPlan::new(&r, GIB, ResourceReservations::default()),
        Err(MaterialError::InvalidRequest(_))
    ));
}

#[test]
fn cancellation_between_passes_keeps_only_latest_coherent_output_set() {
    let s = source(Extent::new(257, 193).unwrap());
    let mut scheduler = Scheduler::default();
    scheduler
        .request_material(request(
            &s,
            1,
            Controls::default(),
            NormalConvention::OpenGl,
            &PORTS,
        ))
        .unwrap();
    let active = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(active.id));
    for id in 2..=64 {
        let ports: &[OutputPort] = if id % 2 == 0 {
            &PORTS
        } else {
            &[OutputPort::Cavity]
        };
        scheduler
            .request_material(request(
                &s,
                id,
                Controls {
                    edge: id as f32,
                    ..Default::default()
                },
                NormalConvention::OpenGl,
                ports,
            ))
            .unwrap();
        assert!(!scheduler.should_continue(active.id));
        assert!(scheduler.start_next().is_none());
        assert!(scheduler.try_latest_material().is_none());
    }
    assert_eq!(scheduler.complete(active.id), Completion::Discarded);
    assert!(scheduler.try_latest_material().is_none());
    let latest = scheduler.start_next().unwrap();
    assert!(scheduler.should_continue(latest.id));
    assert_eq!(scheduler.complete(latest.id), Completion::Published);
    let published = scheduler.try_latest_material().unwrap();
    assert_eq!(
        (
            published.job_id,
            published.request_id,
            published.snapshot.generation
        ),
        (JobId(52), 64, 64)
    );
    assert_eq!(published.outputs.len(), 3);
    for port in PORTS {
        assert_eq!(published.outputs[&port.map_kind()], support::output(port));
    }
    let d = scheduler.diagnostics();
    assert_eq!(d.peak_in_flight, 1);
    assert_eq!(d.peak_pending, 1);
    assert_eq!(d.stale_publications, 0);
    assert!(d.cancellation_requests > 0);
}

#[test]
fn low_budget_rejects_before_compute_and_worker_recovers_with_all_outputs() {
    with_gpu(|gpu| {
        let budget = 256 * 1024;
        let worker = gpu.worker(budget);
        let e = Extent::new(65, 49).unwrap();
        let rgb =
            super::oracle::encode(&super::oracle::paraboloid(e, 1.0), NormalConvention::OpenGl);
        let s = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: budget,
                readback_bytes: 0,
            })
            .unwrap();
        let submitted = worker.diagnostics().submitted;
        worker
            .request_material(request(
                &s,
                1,
                Controls::default(),
                NormalConvention::OpenGl,
                &PORTS,
            ))
            .unwrap();
        let event = support::wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::BudgetExceeded { required_bytes, budget_bytes } if required_bytes > budget && budget_bytes == budget),
            "{event:?}"
        );
        assert_eq!(worker.diagnostics().submitted, submitted);
        assert!(worker.try_latest_material().is_none());
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let planes = gpu.evaluate(
            &worker,
            request(&s, 2, Controls::default(), NormalConvention::OpenGl, &PORTS),
        );
        assert_eq!(planes.scalar.len(), 3);
        assert!(worker.diagnostics().peak_bytes <= budget);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn real_worker_replaces_in_flight_generation_without_mixing_output_ports() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let e = Extent::new(1024, 513).unwrap();
        let rgb =
            super::oracle::encode(&super::oracle::paraboloid(e, 1.0), NormalConvention::OpenGl);
        let s = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
        let before = worker.diagnostics();
        worker
            .request_material(request(
                &s,
                1,
                Controls::default(),
                NormalConvention::OpenGl,
                &PORTS,
            ))
            .unwrap();
        support::wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("curvature failed before cancellation: {event:?}");
            }
            (worker.diagnostics().submitted > before.submitted).then_some(())
        });
        // A zero edge/cavity generation is distinguishable from the positive
        // bump's old Edge output even when the signed plane has a cache hit.
        let latest = Controls {
            edge: 0.0,
            cavity: 0.0,
            ..Default::default()
        };
        for id in 2..32 {
            worker
                .request_material(request(&s, id, latest, NormalConvention::OpenGl, &PORTS))
                .unwrap();
        }
        let planes = gpu.evaluate(
            &worker,
            request(&s, 32, latest, NormalConvention::OpenGl, &PORTS),
        );
        assert!(planes.scalar[&MapKind::Edge].iter().all(|&v| v == 0.0));
        assert!(planes.scalar[&MapKind::Cavity].iter().all(|&v| v == 0.0));
        assert!(planes.scalar[&MapKind::Curvature].iter().any(|&v| v > 0.1));
        let d = worker.diagnostics();
        assert!(d.cancellation_requests > before.cancellation_requests);
        assert_eq!(d.stale_publications, 0);
        assert_eq!(d.peak_in_flight, 1);
        assert_eq!(d.peak_pending, 1);
        assert_eq!(d.validation_errors, 0);
        assert!(d.peak_bytes <= GIB);
        worker.join().unwrap();
    });
}
