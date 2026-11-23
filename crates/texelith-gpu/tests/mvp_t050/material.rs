//! Public material/worker APIs only. No dependency on dispatch registry internals.
use super::{
    gpu::{Gpu, with_gpu},
    oracle::{self as cpu, Grid},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputInterpretation, InputPort,
    MapKind, MaterialGraphDto, MaterialNode, MaterialNodeParams as P, NodeId, NormalConvention,
    OutputPort, OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialError, MaterialPlan, MaterialRequest, Quality,
    ResourceReservations, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

const GIB: u64 = 1024 * 1024 * 1024;
#[derive(Clone)]
struct Controls {
    boundary: Boundary,
    cycles: u32,
    pre: u32,
    post: u32,
    coarse: u32,
    relative: f32,
    absolute: f32,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            boundary: Boundary::Repeat,
            cycles: 8,
            pre: 3,
            post: 3,
            coarse: 32,
            relative: 1e-3,
            absolute: 1e-6,
        }
    }
}
fn request(
    source: &SourceDescriptor,
    serial: u64,
    convention: NormalConvention,
    c: &Controls,
) -> MaterialRequest {
    let normal = OutputRef {
        node: NodeId(1),
        port: OutputPort::Normal,
    };
    let height = OutputRef {
        node: NodeId(2),
        port: OutputPort::Height,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: normal.node,
                inputs: vec![],
                params: P::NormalSource {
                    source: source.clone(),
                    convention,
                },
            },
            MaterialNode {
                id: height.node,
                inputs: vec![InputEdge {
                    port: InputPort::Normal,
                    source: normal,
                }],
                params: P::HeightFromNormal {
                    boundary: c.boundary,
                    v_cycles: c.cycles,
                    pre_smooth: c.pre,
                    post_smooth: c.post,
                    coarsest_iterations: c.coarse,
                    relative_tolerance: c.relative,
                    absolute_tolerance: c.absolute,
                },
            },
        ],
        outputs: [(MapKind::Height, height)].into(),
        selected_output: height,
    };
    MaterialRequest {
        job_id: JobId(50),
        request_id: serial,
        extent: source.extent,
        quality: Quality::Full,
        outputs: dto.outputs.clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: serial,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}
fn descriptor(e: Extent) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [50; 32],
        extent: e,
        interpretation: InputInterpretation::LinearNormal,
    }
}
fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(v) = next() {
            return v;
        }
        assert!(Instant::now() < deadline, "T-050 worker timeout");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn with_worker(run: impl FnOnce(&Gpu, &GpuWorker)) {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(gpu, &worker)));
        let joined = worker.join();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        joined.unwrap();
        assert_eq!(worker.diagnostics().validation_errors, 0);
    });
}
fn install(
    worker: &GpuWorker,
    g: Grid,
    bytes: Vec<u8>,
    convention: NormalConvention,
) -> SourceDescriptor {
    let pixels = TypedSourcePixels::NormalRgb8 {
        pixels: bytes.into(),
        convention,
    };
    let descriptor = pixels
        .descriptor(Extent::new(g.w as u32, g.h as u32).unwrap())
        .unwrap();
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    let event = wait(|| worker.try_event());
    assert!(
        matches!(event,WorkerEvent::SourceReady {content_hash} if content_hash==descriptor.content_hash),
        "{event:?}"
    );
    descriptor
}
fn encode(n: &[[f64; 3]]) -> Vec<u8> {
    n.iter()
        .flat_map(|p| p.map(|v| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8))
        .collect()
}
fn evaluate(gpu: &Gpu, worker: &GpuWorker, r: MaterialRequest) -> Vec<f64> {
    let serial = r.request_id;
    let extent = r.extent;
    worker.request_material(r).unwrap();
    let bundle = wait(|| {
        let event = worker.try_event();
        assert!(
            event.is_none(),
            "HeightFromNormal worker returned an error: {event:?}"
        );
        worker.try_latest_material()
    });
    assert_eq!(bundle.job_id, JobId(50));
    assert_eq!(bundle.request_id, serial);
    assert_eq!(bundle.generation, serial);
    assert_eq!(bundle.extent, extent);
    assert_eq!(bundle.outputs.len(), 1);
    let output = bundle.outputs[&MapKind::Height].texture();
    assert_eq!(output.format(), wgpu::TextureFormat::R32Float);
    let actual = gpu.read(output);
    assert!(cpu::mean(&actual).abs() < 2e-5);
    assert!(actual.iter().all(|v| v.is_finite()));
    actual
}

#[test]
fn public_material_plan_supports_4k_height_from_normal_and_enforces_exact_peak_budget() {
    let source = descriptor(Extent::new(4096, 4096).unwrap());
    let reserved = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 128 * 1024 * 1024,
    };
    for boundary in [Boundary::Repeat, Boundary::Clamp] {
        for cycles in [8, 16] {
            let c = Controls {
                boundary,
                cycles,
                ..Default::default()
            };
            let r = request(&source, 1, NormalConvention::OpenGl, &c);
            let plan =
                MaterialPlan::new(&r, GIB, reserved).expect("HeightFromNormal must be planned");
            // At least source normal, output height, and viewport reservations are live.
            assert!(plan.peak_bytes() >= 4096u64 * 4096 * 12 + reserved.viewport_bytes);
            assert!(plan.peak_bytes() <= GIB);
            assert!(MaterialPlan::new(&r, plan.peak_bytes(), reserved).is_ok());
            assert!(matches!(MaterialPlan::new(&r,plan.peak_bytes()-1,reserved),
            Err(MaterialError::BudgetExceeded {required_bytes,budget_bytes})
                if required_bytes==plan.peak_bytes() && budget_bytes==plan.peak_bytes()-1));
        }
    }
}
#[test]
fn interactive_4k_source_publishes_only_the_requested_1k_height_proxy() {
    with_worker(|gpu, worker| {
        let g = Grid {
            w: 4096,
            h: 4096,
            repeat: true,
        };
        let source = install(
            worker,
            g,
            [128, 128, 255].repeat(g.len()),
            NormalConvention::OpenGl,
        );
        let mut r = request(&source, 1, NormalConvention::OpenGl, &Controls::default());
        r.quality = Quality::Proxy;
        r.extent = Extent::new(1024, 1024).unwrap();
        let plan = MaterialPlan::new(&r, GIB, ResourceReservations::default()).unwrap();
        // Admission also includes full-resolution source conversion/upload.
        assert!(plan.peak_bytes() <= GIB);
        let actual = evaluate(gpu, worker, r);
        assert_eq!(actual.len(), 1024 * 1024);
        assert!(actual.iter().all(|v| *v == 0.0));
    });
}
#[test]
fn cache_identity_contains_all_solver_controls_and_normal_convention() {
    let source = descriptor(Extent::new(17, 13).unwrap());
    let c = Controls::default();
    let key = |r: &MaterialRequest| {
        *MaterialPlan::new(r, GIB, ResourceReservations::default())
            .unwrap()
            .output_keys()
            .get(&MapKind::Height)
            .unwrap()
    };
    let base = key(&request(&source, 1, NormalConvention::OpenGl, &c));
    let mut equivalent = request(&source, 99, NormalConvention::OpenGl, &c);
    equivalent.job_id = JobId(999);
    assert_eq!(base, key(&equivalent));
    for c in [
        Controls {
            boundary: Boundary::Clamp,
            ..c.clone()
        },
        Controls {
            cycles: 16,
            ..c.clone()
        },
        Controls {
            pre: 4,
            ..c.clone()
        },
        Controls {
            post: 4,
            ..c.clone()
        },
        Controls {
            coarse: 64,
            ..c.clone()
        },
        Controls {
            relative: 2e-3,
            ..c.clone()
        },
        Controls {
            absolute: 2e-6,
            ..c.clone()
        },
    ] {
        assert_ne!(
            base,
            key(&request(&source, 1, NormalConvention::OpenGl, &c))
        );
    }
    assert_ne!(
        base,
        key(&request(&source, 1, NormalConvention::DirectX, &c))
    );
}

#[test]
fn normal_proxy_matches_area_oracle_for_npot_singletons_and_both_conventions() {
    with_worker(|gpu, worker| {
        let mut serial = 0;
        for (w, h) in [(2053, 31), (2053, 1), (1, 2053)] {
            let g = Grid { w, h, repeat: true };
            let extent =
                texelith_gpu::proxy_extent(Extent::new(w as u32, h as u32).unwrap()).unwrap();
            let proxy = Grid {
                w: extent.width as usize,
                h: extent.height as usize,
                repeat: true,
            };
            let original = super::proxy_oracle::analytic_rgb(g);
            let mut previous: Option<Vec<[f64; 3]>> = None;
            for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                let mut bytes = original.clone();
                if convention == NormalConvention::DirectX {
                    for p in bytes.as_chunks_mut::<3>().0 {
                        p[1] = 255 - p[1];
                    }
                }
                let expected = super::proxy_oracle::restrict(&bytes, g, proxy);
                let source = install(worker, g, bytes, convention);
                serial += 1;
                let mut r = request(&source, serial, convention, &Controls::default());
                r.outputs = [(
                    MapKind::Normal,
                    OutputRef {
                        node: NodeId(1),
                        port: OutputPort::Normal,
                    },
                )]
                .into();
                r.extent = extent;
                r.quality = Quality::Proxy;
                let plan = MaterialPlan::new(&r, GIB, ResourceReservations::default()).unwrap();
                worker.request_material(r).unwrap();
                let bundle = wait(|| {
                    assert!(worker.try_event().is_none());
                    worker.try_latest_material()
                });
                assert_eq!(bundle.extent, extent);
                assert_eq!(bundle.normal_convention, Some(convention));
                let actual = gpu.read_normal(bundle.outputs[&MapKind::Normal].texture());
                assert_eq!(actual.len(), proxy.len());
                for (a, e) in actual.iter().zip(&expected) {
                    assert!(
                        a.iter().zip(e).all(|(a, e)| (a - e).abs() < 5e-4),
                        "{a:?} != {e:?}"
                    );
                    assert!((a.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 2e-3);
                }
                if let Some(gl) = &previous {
                    for (gl, dx) in gl.iter().zip(&actual) {
                        assert!(
                            (gl[0] - dx[0]).abs() < 2e-6
                                && (gl[1] + dx[1]).abs() < 2e-6
                                && (gl[2] - dx[2]).abs() < 2e-6
                        );
                    }
                }
                previous = Some(actual);
                assert!(plan.peak_bytes() <= GIB);
                assert!(worker.diagnostics().peak_bytes <= GIB);
            }
        }
    });
}
#[test]
fn normal_import_branch_produces_signed_linear_height_and_flips_dx_y_once() {
    with_worker(|gpu, worker| {
        let g = Grid {
            w: 33,
            h: 25,
            repeat: true,
        };
        let expected = cpu::analytic(g);
        let gl = encode(&cpu::normals(&g.gradient(&expected), false));
        let mut dx = gl.clone();
        for p in dx.as_chunks_mut::<3>().0 {
            p[1] = 255 - p[1];
        }
        let source = install(worker, g, gl, NormalConvention::OpenGl);
        let a = evaluate(
            gpu,
            worker,
            request(&source, 1, NormalConvention::OpenGl, &Controls::default()),
        );
        assert!(cpu::rms(&a, &expected) < 1e-3);
        assert!(
            a.iter().any(|v| *v < -0.01) && a.iter().any(|v| *v > 0.01),
            "implicit remap or clamp"
        );
        let source = install(worker, g, dx, NormalConvention::DirectX);
        let b = evaluate(
            gpu,
            worker,
            request(&source, 2, NormalConvention::DirectX, &Controls::default()),
        );
        assert!(
            cpu::rms(&a, &b) < 2e-5,
            "normal convention applied zero or two times"
        );
        assert_eq!(worker.diagnostics().stale_publications, 0);
        assert!(worker.diagnostics().peak_bytes <= GIB);
    });
}
#[test]
fn exhausted_cycle_budget_is_an_error_without_publication_and_next_request_recovers() {
    with_worker(|gpu, worker| {
        let g = Grid {
            w: 65,
            h: 47,
            repeat: true,
        };
        let bytes = encode(&cpu::normals(&g.gradient(&cpu::seeded(g.len())), false));
        let source = install(worker, g, bytes, NormalConvention::OpenGl);
        let c = Controls {
            cycles: 1,
            relative: 1e-12,
            absolute: 1e-12,
            ..Default::default()
        };
        worker
            .request_material(request(&source, 1, NormalConvention::OpenGl, &c))
            .unwrap();
        let event = wait(|| {
            assert!(
                worker.try_latest_material().is_none(),
                "nonconverged height silently published"
            );
            worker.try_event()
        });
        match event {
            WorkerEvent::Error(error) => assert!(
                error.to_string().to_lowercase().contains("residual"),
                "nonconvergence error must include residual: {error}"
            ),
            other => panic!("expected explicit solver failure, got {other:?}"),
        }
        let flat = install(
            worker,
            g,
            [128, 128, 255].repeat(g.len()),
            NormalConvention::OpenGl,
        );
        let result = evaluate(
            gpu,
            worker,
            request(&flat, 2, NormalConvention::OpenGl, &Controls::default()),
        );
        assert!(result.iter().all(|v| *v == 0.0));
    });
}
#[test]
fn worker_budget_rejection_occurs_before_solver_submission_and_recovers() {
    with_worker(|gpu, worker| {
        let g = Grid {
            w: 17,
            h: 13,
            repeat: true,
        };
        let source = install(
            worker,
            g,
            [128, 128, 255].repeat(g.len()),
            NormalConvention::OpenGl,
        );
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: GIB,
                readback_bytes: 0,
            })
            .unwrap();
        let submitted = worker.diagnostics().submitted;
        worker
            .request_material(request(
                &source,
                1,
                NormalConvention::OpenGl,
                &Controls::default(),
            ))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event,WorkerEvent::BudgetExceeded {required_bytes,budget_bytes}
            if required_bytes>GIB && budget_bytes==GIB),
            "{event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        assert_eq!(worker.diagnostics().submitted, submitted);
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let result = evaluate(
            gpu,
            worker,
            request(&source, 2, NormalConvention::OpenGl, &Controls::default()),
        );
        assert!(result.iter().all(|v| *v == 0.0));
    });
}
#[test]
fn cancellation_during_real_cycles_discards_stale_height_and_publishes_latest_only() {
    with_worker(|gpu, worker| {
        let g = Grid {
            w: 1024,
            h: 1024,
            repeat: true,
        };
        let bytes = encode(&cpu::normals(&g.gradient(&cpu::seeded(g.len())), false));
        let source = install(worker, g, bytes, NormalConvention::OpenGl);
        let tiny = Grid {
            w: 17,
            h: 13,
            repeat: true,
        };
        let flat = install(
            worker,
            tiny,
            [128, 128, 255].repeat(tiny.len()),
            NormalConvention::OpenGl,
        );
        let costly = Controls {
            cycles: 16,
            relative: 1e-12,
            absolute: 1e-12,
            ..Default::default()
        };
        let baseline = worker.diagnostics().submitted;
        worker
            .request_material(request(&source, 1, NormalConvention::OpenGl, &costly))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while worker.diagnostics().submitted <= baseline + 1 {
            assert!(
                Instant::now() < deadline,
                "solver never reached in-flight passes"
            );
            assert!(
                worker.try_latest_material().is_none(),
                "solver finished before cancellation"
            );
            assert!(
                worker.try_event().is_none(),
                "solver failed before cancellation"
            );
            std::thread::yield_now();
        }
        // A different output extent/source prevents a warm cache hit for the replacement.
        for id in 2..=65 {
            worker
                .request_material(request(
                    &flat,
                    id,
                    NormalConvention::OpenGl,
                    &Controls::default(),
                ))
                .unwrap();
        }
        let result = wait(|| {
            assert!(
                worker.try_event().is_none(),
                "cancellation became a worker failure"
            );
            worker.try_latest_material()
        });
        assert_eq!(result.request_id, 65);
        assert_eq!(result.generation, 65);
        assert!(
            gpu.read(result.outputs[&MapKind::Height].texture())
                .iter()
                .all(|v| *v == 0.0)
        );
        let d = worker.diagnostics();
        assert!(d.cancellation_requests > 0 && d.cancelled_pass_batches > 0);
        assert_eq!(d.peak_in_flight, 1);
        assert_eq!(d.peak_pending, 1);
        assert_eq!(d.stale_publications, 0);
        assert!(d.peak_bytes <= GIB);
    });
}
