use super::{
    oracle::{self as cpu, Controls},
    support::{self, GIB, Gpu, PORTS, Planes, WITH_NORMAL, request},
};
use serde_json::{Value, json};
use texelith_core::{Boundary, Extent, MapKind, NormalConvention};
use texelith_gpu::{ResourceReservations, WorkerDiagnostics, WorkerEvent};

fn compare(planes: &Planes, extent: Extent, convention: NormalConvention, controls: Controls) {
    let expected = cpu::curvature(
        planes.normal.as_ref().unwrap(),
        extent,
        convention,
        controls,
    );
    for (port, values) in PORTS.iter().zip(&expected) {
        cpu::compare(&planes.scalar[&port.map_kind()], values);
    }
    assert!(
        planes.scalar[&MapKind::Edge]
            .iter()
            .zip(&planes.scalar[&MapKind::Cavity])
            .all(|(&a, &b)| a == 0.0 || b == 0.0)
    );
}

pub(super) fn run(gpu: &Gpu) -> (Vec<Value>, WorkerDiagnostics) {
    let worker = gpu.worker(GIB);
    let mut serial = 0;
    for (w, h, fixture) in [
        (1, 1, 0),
        (17, 13, 0),
        (1, 23, 3),
        (23, 1, 3),
        (33, 25, 1),
        (33, 25, 2),
        (257, 193, 3),
        (17, 13, 4),
        (31, 19, 5),
        (19, 31, 6),
        (23, 17, 7),
    ] {
        let e = Extent::new(w, h).unwrap();
        let normals = match fixture {
            0 => vec![[0.8, -0.6, 0.00001]; (w * h) as usize],
            1 => cpu::paraboloid(e, 1.0),
            2 => cpu::paraboloid(e, -1.0),
            3 => cpu::seeded(e),
            4 => cpu::image(e, |x, y| {
                [
                    if x % 2 == 0 { 1.0 } else { -1.0 },
                    if y % 2 == 0 { 1.0 } else { -1.0 },
                    0.00001,
                ]
            }),
            5 => cpu::image(e, |x, y| {
                if x == 0 && y == 0 {
                    [0.6, -0.7, 0.1]
                } else {
                    [0.0, 0.0, 1.0]
                }
            }),
            6 => cpu::image(e, |x, y| {
                [
                    0.3 * f64::from(x) / f64::from(w),
                    -0.2 * f64::from(y) / f64::from(h),
                    1.0,
                ]
            }),
            // Y encodes to 255/0, whose signed decoded values are exactly +/-1.
            // Other RGB8 Y complements can normalize/round asymmetrically into
            // half floats during source import, before curvature runs. Use an
            // exact imported counterpart for the output-equivalence assertion;
            // every other fixture still checks both conventions against its
            // own post-import f64 oracle above, with the unchanged 2e-5 bound.
            _ => cpu::image(e, |x, y| [0.002 * f64::from(x), 1.0, 0.002 * f64::from(y)]),
        };
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            let controls = Controls {
                edge: 0.003,
                cavity: 0.007,
                boundary,
            };
            let mut conventions = Vec::new();
            for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                let rgb = cpu::encode(&normals, convention);
                let source = gpu.install(&worker, e, &rgb, convention);
                serial += 1;
                let a = gpu.evaluate(
                    &worker,
                    request(&source, serial, controls, convention, &WITH_NORMAL),
                );
                compare(&a, e, convention, controls);
                if fixture == 0 {
                    assert!(a.scalar.values().flatten().all(|&v| v == 0.0));
                }
                if boundary == Boundary::Clamp && (fixture == 1 || fixture == 2) {
                    let center = ((h / 2) * w + w / 2) as usize;
                    let sign = if fixture == 1 { 1.0 } else { -1.0 };
                    assert!(f64::from(a.scalar[&MapKind::Curvature][center]) * sign > 0.5);
                    let inactive = if fixture == 1 {
                        MapKind::Cavity
                    } else {
                        MapKind::Edge
                    };
                    assert!(a.scalar[&inactive].iter().all(|&v| v == 0.0));
                }
                serial += 1;
                let b = gpu.evaluate(
                    &worker,
                    request(&source, serial, controls, convention, &PORTS),
                );
                assert_eq!(a.scalar, b.scalar);
                for port in PORTS {
                    serial += 1;
                    let one = gpu.evaluate(
                        &worker,
                        request(&source, serial, controls, convention, &[port]),
                    );
                    assert_eq!(one.scalar[&port.map_kind()], a.scalar[&port.map_kind()]);
                }
                for scale in [0.0, f32::MAX] {
                    let changed = Controls {
                        edge: scale,
                        cavity: scale,
                        ..controls
                    };
                    serial += 1;
                    let b = gpu.evaluate(
                        &worker,
                        request(&source, serial, changed, convention, &WITH_NORMAL),
                    );
                    compare(&b, e, convention, changed);
                    assert_eq!(a.scalar[&MapKind::Curvature], b.scalar[&MapKind::Curvature]);
                }
                if fixture == 7 {
                    let canonical = a
                        .normal
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|n| {
                            let mut n = *n;
                            if convention == NormalConvention::DirectX {
                                n[1] = -n[1];
                            }
                            n
                        })
                        .collect::<Vec<_>>();
                    conventions.push((a.scalar, canonical));
                }
            }
            if fixture == 7 {
                assert_eq!(conventions[0].1, conventions[1].1);
                for port in PORTS {
                    cpu::compare(
                        &conventions[0].0[&port.map_kind()],
                        &conventions[1].0[&port.map_kind()]
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
    }
    assert_eq!(worker.diagnostics().validation_errors, 0);
    worker.join().unwrap();

    let worker = gpu.worker(256 * 1024);
    let e = Extent::new(65, 49).unwrap();
    let rgb = cpu::encode(&cpu::paraboloid(e, 1.0), NormalConvention::OpenGl);
    let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 256 * 1024,
            readback_bytes: 0,
        })
        .unwrap();
    let before = worker.diagnostics().submitted;
    worker
        .request_material(request(
            &source,
            1,
            Controls::default(),
            NormalConvention::OpenGl,
            &PORTS,
        ))
        .unwrap();
    assert!(
        matches!(support::wait(||worker.try_event()),WorkerEvent::BudgetExceeded {required_bytes,budget_bytes} if required_bytes>budget_bytes)
    );
    assert_eq!(before, worker.diagnostics().submitted);
    assert!(worker.try_latest_material().is_none());
    worker
        .set_resource_reservations(ResourceReservations::default())
        .unwrap();
    let recovered = gpu.evaluate(
        &worker,
        request(
            &source,
            2,
            Controls::default(),
            NormalConvention::OpenGl,
            &PORTS,
        ),
    );
    assert_eq!(recovered.scalar.len(), 3);
    assert!(worker.diagnostics().peak_bytes <= 256 * 1024);
    worker.join().unwrap();

    let worker = gpu.worker(GIB);
    let e = Extent::new(1024, 513).unwrap();
    let rgb = cpu::encode(&cpu::paraboloid(e, 1.0), NormalConvention::OpenGl);
    let source = gpu.install(&worker, e, &rgb, NormalConvention::OpenGl);
    let before = worker.diagnostics().submitted;
    worker
        .request_material(request(
            &source,
            1,
            Controls::default(),
            NormalConvention::OpenGl,
            &PORTS,
        ))
        .unwrap();
    support::wait(|| (worker.diagnostics().submitted > before).then_some(()));
    let controls = Controls {
        edge: 0.0,
        cavity: 0.0,
        ..Default::default()
    };
    for id in 2..32 {
        worker
            .request_material(request(
                &source,
                id,
                controls,
                NormalConvention::OpenGl,
                &PORTS,
            ))
            .unwrap();
    }
    let latest = gpu.evaluate(
        &worker,
        request(&source, 32, controls, NormalConvention::OpenGl, &PORTS),
    );
    assert!(latest.scalar[&MapKind::Edge].iter().all(|&v| v == 0.0));
    assert!(latest.scalar[&MapKind::Cavity].iter().all(|&v| v == 0.0));
    assert!(latest.scalar[&MapKind::Curvature].iter().any(|&v| v > 0.1));
    let stats = worker.diagnostics();
    assert!(stats.cancellation_requests > 0);
    assert_eq!(stats.stale_publications, 0);
    assert_eq!(stats.peak_in_flight, 1);
    assert_eq!(stats.peak_pending, 1);
    assert_eq!(stats.validation_errors, 0);
    assert!(stats.peak_bytes <= GIB);
    worker.join().unwrap();
    // Each name above is backed by the assertions, including independent CPU
    // comparisons on post-import half normals rather than unquantized originals.
    (
        [
            "flat_zero",
            "bump_positive",
            "bowl_negative",
            "output_separation",
            "convention_equivalence",
            "steep_finite",
            "cpu_oracle",
            "npot_257x193",
            "singleton_axes",
            "repeat_clamp",
            "determinism",
            "cancellation",
            "low_budget",
        ]
        .iter()
        .map(|name| json!({"name":name,"passed":true}))
        .collect(),
        stats,
    )
}
