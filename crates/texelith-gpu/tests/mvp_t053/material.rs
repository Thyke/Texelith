use super::{
    gpu::{Gpu, with_gpu},
    oracle,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputEdge, InputInterpretation, InputPort, MapKind,
    MaterialGraphDto, MaterialNode, MaterialNodeParams as P, NodeId, OutputPort, OutputRef,
    SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuWorker, JobId, MaterialError, MaterialPlan, MaterialRequest,
    Quality, ResourceReservations, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
    scheduler::{Completion, Scheduler},
};

const GIB: u64 = 1_073_741_824;
pub const IDENTITY: [[f32; 2]; 2] = [[0.0, 0.0], [1.0, 1.0]];

pub fn request(
    source: &SourceDescriptor,
    serial: u64,
    curve: &[[f32; 2]],
    base: f32,
    invert: bool,
) -> MaterialRequest {
    let color = OutputRef {
        node: NodeId(1),
        port: OutputPort::Albedo,
    };
    let roughness = OutputRef {
        node: NodeId(2),
        port: OutputPort::Roughness,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: color.node,
                inputs: vec![],
                params: P::ColorSource(source.clone()),
            },
            MaterialNode {
                id: roughness.node,
                inputs: vec![InputEdge {
                    port: InputPort::Color,
                    source: color,
                }],
                params: P::Roughness {
                    curve: curve.to_vec(),
                    base,
                    invert,
                },
            },
        ],
        outputs: [(MapKind::Roughness, roughness)].into(),
        selected_output: roughness,
    };
    MaterialRequest {
        job_id: JobId(53),
        request_id: serial,
        outputs: dto.outputs.clone(),
        extent: source.extent,
        quality: Quality::Full,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: serial,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}

fn descriptor(w: u32, h: u32) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [53; 32],
        extent: Extent::new(w, h).unwrap(),
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-053 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn install(worker: &GpuWorker, extent: Extent, rgba: Vec<u8>) -> SourceDescriptor {
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: rgba.into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let descriptor = pixels.descriptor(extent).unwrap();
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    let event = wait(|| worker.try_event());
    assert!(
        matches!(event, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
        "{event:?}"
    );
    descriptor
}

fn evaluate(gpu: &Gpu, worker: &GpuWorker, request: MaterialRequest) -> Vec<f32> {
    let serial = request.request_id;
    let extent = request.extent;
    worker.request_material(request).unwrap();
    let bundle: CompletedMaterial = wait(|| {
        if let Some(event) = worker.try_event() {
            panic!("roughness evaluation failed: {event:?}");
        }
        worker.try_latest_material()
    });
    assert_eq!(
        (bundle.job_id, bundle.request_id, bundle.generation),
        (JobId(53), serial, serial)
    );
    assert_eq!(bundle.extent, extent);
    assert_eq!(bundle.quality, Quality::Full);
    assert_eq!(bundle.outputs.len(), 1);
    assert_eq!(bundle.normal_convention, None);
    let values = gpu.read_scalar(bundle.outputs[&MapKind::Roughness].texture());
    assert_eq!(values.len(), (extent.width * extent.height) as usize);
    values
}

#[test]
fn worker_default_uses_linear_rec709_ignores_alpha_and_never_srgb_encodes_output() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [10, 10, 10],
            [11, 11, 11],
            [128, 128, 128],
            [255, 255, 255],
            [0, 0, 0],
        ];
        let bytes: Vec<u8> = colors
            .into_iter()
            .flat_map(|[r, g, b]| [[r, g, b, 0], [r, g, b, 255]])
            .flatten()
            .collect();
        let source = install(&worker, Extent::new(16, 1).unwrap(), bytes.clone());
        let actual = evaluate(gpu, &worker, request(&source, 1, &IDENTITY, 0.5, false));
        oracle::compare(&actual, &oracle::luminance(&bytes), 2e-5);
        for pair in actual.as_chunks::<2>().0 {
            assert_eq!(pair[0].to_bits(), pair[1].to_bits());
        }
        assert!((actual[10] - 128.0 / 255.0).abs() > 0.1);
        worker.join().unwrap();
    });
}

#[test]
fn worker_curve_remap_covers_constant_ramp_impulse_npot_singletons_and_all_sixteen_knots() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let curve: Vec<_> = (0..16)
            .map(|i| [i as f32 / 15.0, if i % 2 == 0 { 0.125 } else { 0.875 }])
            .collect();
        let mut serial = 0;
        for (w, h, kind) in [
            (1, 1, 0),
            (1, 17, 1),
            (17, 1, 1),
            (9, 7, 0),
            (31, 17, 2),
            (257, 193, 3),
        ] {
            let extent = Extent::new(w, h).unwrap();
            let mut state = 53u32;
            let bytes: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let value = match kind {
                        0 => 128,
                        1 => (i * 255 / (w * h - 1).max(1)) as u8,
                        2 => {
                            if i == w * h - 1 {
                                255
                            } else {
                                0
                            }
                        }
                        _ => (state >> 24) as u8,
                    };
                    [value, value, value, (state >> 16) as u8]
                })
                .collect();
            let luma = oracle::luminance(&bytes);
            let source = install(&worker, extent, bytes);
            for invert in [false, true] {
                serial += 1;
                let expected = oracle::roughness(&luma, None, &curve, 0.25, invert);
                let actual = evaluate(gpu, &worker, request(&source, serial, &curve, 0.25, invert));
                oracle::compare(&actual, &expected, 2e-5);
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn worker_parameter_edits_invalidate_cache_and_restored_content_is_bitwise_identical() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let bytes: Vec<u8> = (0..33 * 19)
            .flat_map(|i| [(i % 256) as u8, 128, 64, 255])
            .collect();
        let source = install(&worker, Extent::new(33, 19).unwrap(), bytes.clone());
        let original = evaluate(gpu, &worker, request(&source, 1, &IDENTITY, 0.5, false));
        let before = worker.diagnostics();
        let repeated = evaluate(gpu, &worker, request(&source, 2, &IDENTITY, 0.5, false));
        assert_eq!(
            original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeated.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let after = worker.diagnostics();
        assert!(after.cache_hits > before.cache_hits);
        assert_eq!(after.submitted, before.submitted);
        let inverted = evaluate(gpu, &worker, request(&source, 3, &IDENTITY, 0.5, true));
        for (r, s) in original.iter().zip(&inverted) {
            assert_eq!(*s, 1.0 - *r);
        }
        let curve = [[0.0, 0.2], [1.0, 0.8]];
        let edited = evaluate(gpu, &worker, request(&source, 4, &curve, 0.5, false));
        oracle::compare(
            &edited,
            &oracle::roughness(&oracle::luminance(&bytes), None, &curve, 0.5, false),
            2e-5,
        );
        assert!(
            edited
                .iter()
                .zip(&original)
                .any(|(a, b)| (a - b).abs() > 0.01)
        );
        let restored = evaluate(gpu, &worker, request(&source, 5, &IDENTITY, 0.5, false));
        assert_eq!(
            original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            restored.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn planner_supports_4k_roughness_and_rejects_peak_minus_one_with_viewport_reserved() {
    let r = request(&descriptor(4096, 4096), 1, &IDENTITY, 0.5, false);
    let reservations = ResourceReservations {
        viewport_bytes: 144 * 1024 * 1024,
        readback_bytes: 0,
    };
    let plan = MaterialPlan::new(&r, GIB, reservations).expect("roughness must be supported");
    assert!(plan.peak_bytes() > reservations.viewport_bytes);
    assert!(plan.peak_bytes() <= GIB);
    assert!(matches!(
        MaterialPlan::new(&r, plan.peak_bytes() - 1, reservations),
        Err(MaterialError::BudgetExceeded { .. })
    ));
    let key = plan.output_keys()[&MapKind::Roughness];
    for changed in [
        request(&descriptor(4096, 4096), 2, &IDENTITY, 0.25, false),
        request(&descriptor(4096, 4096), 3, &IDENTITY, 0.5, true),
        request(
            &descriptor(4096, 4096),
            4,
            &[[0.0, 1.0], [1.0, 0.0]],
            0.5,
            false,
        ),
    ] {
        assert_ne!(
            key,
            MaterialPlan::new(&changed, GIB, reservations)
                .unwrap()
                .output_keys()[&MapKind::Roughness]
        );
    }
    let same = request(&descriptor(4096, 4096), 999, &IDENTITY, 0.5, false);
    assert_eq!(
        key,
        MaterialPlan::new(&same, GIB, reservations)
            .unwrap()
            .output_keys()[&MapKind::Roughness]
    );
}

#[test]
fn roughness_scheduler_cancels_active_generation_and_never_publishes_partial_results() {
    let mut scheduler = Scheduler::default();
    let source = descriptor(31, 17);
    scheduler
        .request_material(request(&source, 1, &IDENTITY, 0.5, false))
        .unwrap();
    let old = scheduler.start_next().unwrap();
    scheduler
        .request_material(request(&source, 2, &IDENTITY, 0.5, true))
        .unwrap();
    assert!(!scheduler.should_continue(old.id));
    assert!(scheduler.start_next().is_none());
    assert!(scheduler.try_latest_material().is_none());
    assert_eq!(scheduler.complete(old.id), Completion::Discarded);
    let current = scheduler.start_next().unwrap();
    assert_eq!(scheduler.complete(current.id), Completion::Published);
    let result = scheduler.try_latest_material().unwrap();
    assert_eq!(result.request_id, 2);
    assert_eq!(result.snapshot.generation, 2);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(scheduler.diagnostics().peak_in_flight, 1);
    assert_eq!(scheduler.diagnostics().peak_pending, 1);
}

#[test]
fn worker_budget_failure_has_no_partial_output_and_recovers_after_reservations_drop() {
    with_gpu(|gpu| {
        let worker = GpuWorker::spawn(
            gpu.device.clone(),
            gpu.queue.clone(),
            GpuConfig {
                budget_bytes: 64 * 1024,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        let bytes = vec![128; 16 * 16 * 4];
        let source = install(&worker, Extent::new(16, 16).unwrap(), bytes.clone());
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: 60 * 1024,
                readback_bytes: 0,
            })
            .unwrap();
        worker
            .request_material(request(&source, 1, &IDENTITY, 0.5, false))
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::BudgetExceeded {budget_bytes: 65536, required_bytes} if required_bytes > 65536),
            "{event:?}"
        );
        assert!(worker.try_latest_material().is_none());
        wait(|| (worker.diagnostics().in_flight == 0).then_some(()));
        worker
            .set_resource_reservations(ResourceReservations::default())
            .unwrap();
        let actual = evaluate(gpu, &worker, request(&source, 2, &IDENTITY, 0.5, false));
        oracle::compare(&actual, &oracle::luminance(&bytes), 2e-5);
        assert!(worker.diagnostics().peak_bytes <= 64 * 1024);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        worker.join().unwrap();
    });
}

#[test]
fn worker_roughness_proxy_has_declared_extent_and_area_restricts_linear_color() {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let full = Extent::new(2051, 7).unwrap();
        let bytes: Vec<u8> = (0..full.width * full.height)
            .flat_map(|i| {
                let value = ((i * 53 + i / full.width * 17) % 256) as u8;
                [value, value, value, 255]
            })
            .collect();
        let luma = oracle::luminance(&bytes);
        let source = install(&worker, full, bytes);
        let mut r = request(&source, 1, &IDENTITY, 0.5, false);
        r.quality = Quality::Proxy;
        r.extent = texelith_gpu::proxy_extent(full).unwrap();
        let proxy = r.extent;
        worker.request_material(r).unwrap();
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("proxy failed: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(bundle.extent, proxy);
        assert_eq!(bundle.quality, Quality::Proxy);
        let texture = bundle.outputs[&MapKind::Roughness].texture();
        assert_eq!(
            (texture.width(), texture.height()),
            (proxy.width, proxy.height)
        );
        let mut expected = Vec::new();
        let sx = f64::from(full.width) / f64::from(proxy.width);
        let sy = f64::from(full.height) / f64::from(proxy.height);
        for y in 0..proxy.height {
            for x in 0..proxy.width {
                let (x0, x1) = (f64::from(x) * sx, f64::from(x + 1) * sx);
                let (y0, y1) = (f64::from(y) * sy, f64::from(y + 1) * sy);
                let mut sum = 0.0;
                for iy in y0.floor() as u32..y1.ceil() as u32 {
                    for ix in x0.floor() as u32..x1.ceil() as u32 {
                        let wx = (x1.min(f64::from(ix + 1)) - x0.max(f64::from(ix))).max(0.0);
                        let wy = (y1.min(f64::from(iy + 1)) - y0.max(f64::from(iy))).max(0.0);
                        sum += luma[(iy * full.width + ix) as usize] * wx * wy;
                    }
                }
                expected.push(sum / (sx * sy));
            }
        }
        oracle::compare(&gpu.read_scalar(texture), &expected, 2e-5);
        assert_eq!(worker.diagnostics().validation_errors, 0);
        drop(bundle);
        worker.join().unwrap();
    });
}

fn assert_white_npot_proxy(curve: &[[f32; 2]]) {
    with_gpu(|gpu| {
        let worker =
            GpuWorker::spawn(gpu.device.clone(), gpu.queue.clone(), GpuConfig::default()).unwrap();
        let full = Extent::new(4096, 3001).unwrap();
        let source = install(
            &worker,
            full,
            vec![255; (full.width * full.height * 4) as usize],
        );
        let mut r = request(&source, 1, curve, 0.5, false);
        r.quality = Quality::Proxy;
        r.extent = texelith_gpu::proxy_extent(full).unwrap();
        assert_eq!(r.extent, Extent::new(1024, 750).unwrap());
        worker.request_material(r).unwrap();
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("white NPOT proxy failed: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(bundle.extent, Extent::new(1024, 750).unwrap());
        assert_eq!(bundle.quality, Quality::Proxy);
        let texture = bundle.outputs[&MapKind::Roughness].texture();
        assert_eq!((texture.width(), texture.height()), (1024, 750));
        let values = gpu.read_scalar(texture);
        assert_eq!(values.len(), 1024 * 750);
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, 1.0, "white NPOT proxy pixel {i}");
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        drop(bundle);
        worker.join().unwrap();
    });
}

#[test]
fn worker_white_npot_proxy_preserves_exact_one() {
    assert_white_npot_proxy(&IDENTITY);
}

#[test]
fn worker_white_npot_proxy_preserves_exact_one_through_steep_curve() {
    assert_white_npot_proxy(&[[0.0, 0.0], [0.9999, 0.0], [1.0, 1.0]]);
}
