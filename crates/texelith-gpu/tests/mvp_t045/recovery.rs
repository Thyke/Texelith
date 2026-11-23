//! Real-worker regressions: adapter absence is a failure, never a passing skip.
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, Node, NodeId, NodeParams,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialRequest, Quality, RenderRequest, ResourceReservations,
    SaveError, SourceUpload, TypedSourcePixels, WorkerEvent,
};

static GPU: Mutex<()> = Mutex::new(());

struct Worker(GpuWorker);
impl Drop for Worker {
    fn drop(&mut self) {
        self.0.join().unwrap();
    }
}

fn worker(budget_bytes: u64) -> Worker {
    let backends = if cfg!(windows) {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.enumerate_adapters(backends))
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .min_by_key(|a| a.get_info().device_type != wgpu::DeviceType::DiscreteGpu)
        .expect("worker regression tests require a native GPU");
    eprintln!("worker regression adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    Worker(
        GpuWorker::spawn(
            Arc::new(device),
            Arc::new(queue),
            GpuConfig {
                budget_bytes,
                max_cache_entries: 8,
            },
        )
        .unwrap(),
    )
}

fn wait<T>(mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "worker result timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn request(worker: &GpuWorker) -> RenderRequest {
    let extent = Extent::new(16, 16).unwrap();
    let pixels: Arc<[u8]> = vec![128; 16 * 16 * 4].into();
    let descriptor = TypedSourcePixels::Rgba8 {
        pixels: pixels.clone(),
        interpretation: InputInterpretation::LinearHeight,
    }
    .descriptor(extent)
    .unwrap();
    worker
        .install_source(SourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .unwrap();
    assert!(
        matches!(wait(|| worker.try_event()), WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash)
    );
    let graph = Graph::new(
        vec![
            Node {
                id: NodeId(1),
                input: None,
                params: NodeParams::Source(descriptor),
            },
            Node {
                id: NodeId(2),
                input: Some(NodeId(1)),
                params: NodeParams::NormalFromHeight { strength: 1.0 },
            },
        ],
        NodeId(2),
    )
    .unwrap();
    RenderRequest {
        request_id: 1,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 1,
            graph,
        }),
        output: NodeId(1),
        extent,
        quality: Quality::Full,
    }
}

#[test]
fn failed_save_is_consumable_and_retry_succeeds_without_cancelling_intent() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let worker = worker(7 * 1024);
    let worker = &worker.0;
    let height = request(worker);
    worker.request(height.clone()).unwrap();
    drop(wait(|| worker.try_latest()));
    let mut normal = height.clone();
    normal.output = NodeId(2);
    worker.request_save(normal).unwrap();
    assert!(
        matches!(wait(|| worker.try_event()), WorkerEvent::BudgetExceeded { budget_bytes: 7168, required_bytes } if required_bytes > 7168)
    );
    assert_eq!(worker.request_save(height.clone()), Err(SaveError::Busy));
    let failure = wait(|| worker.try_save());
    assert!(
        matches!(failure, Err(SaveError::Failed(ref message)) if message.contains("BudgetExceeded")),
        "{failure:?}"
    );
    assert!(worker.try_save().is_none());
    let retry = worker.request_save(height).unwrap();
    let saved = wait(|| worker.try_save()).unwrap();
    assert_eq!(saved.save_id, retry);
    assert_eq!(saved.output, NodeId(1));
    assert_eq!(saved.pixels, [128, 128].repeat(16 * 16));
    assert!(worker.try_event().is_none());
    assert!(worker.diagnostics().peak_bytes <= 7168);
    assert_eq!(worker.diagnostics().validation_errors, 0);
}

#[test]
fn completed_material_clears_released_reservation_diagnostics() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let worker = worker(64 * 1024);
    let worker = &worker.0;
    let legacy = request(worker);
    let graph = Graph::try_from(texelith_core::MaterialGraphDto::from(
        legacy.snapshot.graph.clone(),
    ))
    .unwrap();
    let mut material = MaterialRequest {
        job_id: JobId(1),
        request_id: 1,
        outputs: graph.material_outputs().unwrap().clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 1,
            graph,
        }),
        extent: legacy.extent,
        quality: Quality::Full,
    };
    for (viewport, readback) in [(8192, 4096), (0, 0), (4096, 8192), (0, 0), (8192, 4096)] {
        worker
            .set_resource_reservations(ResourceReservations {
                viewport_bytes: viewport,
                readback_bytes: readback,
            })
            .unwrap();
        worker.request_material(material.clone()).unwrap();
        drop(wait(|| worker.try_latest_material()));
        let stats = worker.diagnostics();
        assert_eq!(stats.viewport_reserved_bytes, viewport);
        assert_eq!(stats.readback_reserved_bytes, readback);
        assert!(worker.try_event().is_none());
        material.request_id += 1;
    }
    wait(|| (worker.diagnostics().retired_bytes == 0).then_some(()));
    assert_eq!(worker.diagnostics().validation_errors, 0);
    worker.join().unwrap();
    assert_eq!(worker.diagnostics().viewport_reserved_bytes, 0);
    assert_eq!(worker.diagnostics().readback_reserved_bytes, 0);
}
