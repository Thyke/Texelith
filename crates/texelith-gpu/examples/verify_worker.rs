//! Explicit hardware checks of worker cache hits, save bytes and lease lifetime.
//! This is not an eframe presentation or PNG encoder test.
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, MapKind, Node, NodeId, NodeParams,
    SourceDescriptor,
};
use texelith_gpu::{
    CompletedFrame, CompletedReadback, GpuConfig, GpuWorker, Quality, RenderRequest, SaveError,
    SourceUpload, WorkerEvent,
};

fn request(
    id: u64,
    generation: u64,
    source: &SourceDescriptor,
    strength: f32,
    output: u64,
    quality: Quality,
) -> RenderRequest {
    let nodes = [
        NodeParams::Source(source.clone()),
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Blur {
            sigma_source_px: 0.0,
        },
        NodeParams::NormalFromHeight { strength },
    ]
    .into_iter()
    .enumerate()
    .map(|(i, params)| Node {
        id: NodeId(i as u64 + 1),
        input: (i != 0).then_some(NodeId(i as u64)),
        params,
    })
    .collect();
    RenderRequest {
        request_id: id,
        snapshot: Arc::new(EvaluationSnapshot {
            generation,
            graph: Graph::new(nodes, NodeId(output)).unwrap(),
        }),
        output: NodeId(output),
        extent: if quality == Quality::Full {
            source.extent
        } else {
            texelith_gpu::proxy_extent(source.extent).unwrap()
        },
        quality,
    }
}
fn events(worker: &GpuWorker) -> Result<bool, String> {
    let mut ready = false;
    while let Some(event) = worker.try_event() {
        match event {
            WorkerEvent::SourceReady { .. } => ready = true,
            other => return Err(format!("unexpected worker event: {other:?}")),
        }
    }
    Ok(ready)
}
fn wait<T>(mut receive: impl FnMut() -> Result<Option<T>, String>) -> Result<T, String> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(value) = receive()? {
            return Ok(value);
        }
        if Instant::now() > deadline {
            return Err("worker verification timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn frame(worker: &GpuWorker) -> Result<CompletedFrame, String> {
    wait(|| {
        events(worker)?;
        Ok(worker.try_latest())
    })
}
fn saved(worker: &GpuWorker) -> Result<CompletedReadback, String> {
    wait(|| {
        events(worker)?;
        worker.try_save().transpose().map_err(|e| e.to_string())
    })
}
fn install(worker: &GpuWorker, extent: Extent, value: u8) -> Result<SourceDescriptor, String> {
    let mut pixels = vec![value; extent.width as usize * extent.height as usize * 4];
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    let descriptor =
        SourceDescriptor::from_rgba8(&pixels, extent, InputInterpretation::LinearHeight)
            .map_err(|e| e.to_string())?;
    worker
        .install_source(SourceUpload {
            descriptor: descriptor.clone(),
            pixels: pixels.into(),
        })
        .map_err(|e| e.to_string())?;
    wait(|| Ok(events(worker)?.then_some(())))?;
    Ok(descriptor)
}

fn execute() -> Result<serde_json::Value, String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    descriptor.flags |= wgpu::InstanceFlags::VALIDATION;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12))
        .into_iter()
        .find(|adapter| adapter.get_info().name.contains("RTX 5070 Ti"))
        .ok_or("RTX 5070 Ti unavailable")?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let worker =
        GpuWorker::spawn(device.clone(), queue, GpuConfig::default()).map_err(|e| e.to_string())?;
    let extent = Extent {
        width: 65,
        height: 3,
    };
    let source = install(&worker, extent, 128)?;
    worker
        .request(request(1, 1, &source, 1.0, 4, Quality::Full))
        .map_err(|e| e.to_string())?;
    let held = frame(&worker)?;
    let before = worker.diagnostics();
    worker
        .request(request(2, 2, &source, 9.0, 4, Quality::Full))
        .map_err(|e| e.to_string())?;
    let edited = frame(&worker)?;
    let after = worker.diagnostics();
    if after.cache_hits - before.cache_hits != 3 || after.cache_misses - before.cache_misses != 1 {
        return Err("strength edit did not reuse exactly three ancestors".into());
    }
    if held.lease.texture() == edited.lease.texture() {
        return Err("UI-held display allocation was overwritten".into());
    }
    drop(edited);
    worker
        .request(request(3, 3, &source, 1.0, 4, Quality::Full))
        .map_err(|e| e.to_string())?;
    drop(frame(&worker)?);
    let undo = worker.diagnostics();
    if undo.cache_hits - after.cache_hits != 4 || undo.cache_misses != after.cache_misses {
        return Err("undo failed to hit the older normal cache entry".into());
    }
    let save_id = worker
        .request_save(request(100, 3, &source, 1.0, 3, Quality::Full))
        .map_err(|e| e.to_string())?;
    if !matches!(
        worker.request_save(request(101, 3, &source, 1.0, 3, Quality::Full)),
        Err(SaveError::Busy)
    ) {
        return Err("second save was not Busy".into());
    }
    let height = saved(&worker)?;
    if height.save_id != save_id
        || height.kind != MapKind::Height
        || height.extent != extent
        || height.pixels != [128u8, 128].repeat(65 * 3)
    {
        return Err("NPOT 16-bit height bytes/identity mismatch".into());
    }
    worker
        .request_save(request(102, 3, &source, 1.0, 4, Quality::Full))
        .map_err(|e| e.to_string())?;
    let normal = saved(&worker)?;
    if normal.kind != MapKind::Normal || normal.pixels != [128u8, 128, 255].repeat(65 * 3) {
        return Err("NPOT RGB normal bytes mismatch".into());
    }
    worker
        .request_save(request(103, 3, &source, 1.0, 4, Quality::Full))
        .map_err(|e| e.to_string())?;
    worker
        .request(request(4, 4, &source, 2.0, 4, Quality::Proxy))
        .map_err(|e| e.to_string())?;
    wait(|| match worker.try_save() {
        Some(Err(SaveError::Cancelled)) => Ok(Some(())),
        None => Ok(None),
        other => Err(format!("expected cancelled save, got {other:?}")),
    })?;
    drop(frame(&worker)?);

    // A proxy is not a valid save source: the export must evaluate the original
    // extent and use the newly installed content, including its rightmost pixel.
    let larger = install(
        &worker,
        Extent {
            width: 2049,
            height: 3,
        },
        224,
    )?;
    worker
        .request(request(5, 5, &larger, 1.0, 3, Quality::Proxy))
        .map_err(|e| e.to_string())?;
    let proxy = frame(&worker)?;
    if proxy.extent.width != 1024 {
        return Err("wrong proxy extent".into());
    }
    drop(proxy);
    worker
        .request_save(request(104, 5, &larger, 1.0, 3, Quality::Full))
        .map_err(|e| e.to_string())?;
    let full = saved(&worker)?;
    if full.extent != larger.extent || full.pixels != [224u8, 224].repeat(2049 * 3) {
        return Err("save used a proxy or the replaced source".into());
    }
    let stats = worker.diagnostics();
    if stats.peak_in_flight != 1 || stats.peak_pending != 1 || stats.validation_errors != 0 {
        return Err("worker bounds/validation failed".into());
    }
    worker.join().map_err(|e| e.to_string())?;
    wait(|| {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        Ok(
            (worker.diagnostics().allocated_bytes == u64::from(extent.width * extent.height * 4))
                .then_some(()),
        )
    })?;
    // Only the held display lease survives shutdown. Dropping it retires its
    // queue use, and completion must reduce the ledger to zero.
    drop(held);
    wait(|| {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        Ok((worker.diagnostics().allocated_bytes == 0).then_some(()))
    })?;
    Ok(
        json!({ "passed": true, "adapter": info.name, "backend": format!("{:?}", info.backend), "driver": info.driver,
        "checks": ["upstream-cache-hits", "undo-cache-hit", "pinned-display-not-reused", "save-busy", "npot-height16-bytes", "npot-normal8-bytes", "save-cancelled-on-edit", "source-replacement", "full-save-from-proxy", "lease-survives-shutdown", "zero-after-last-lease-retirement"],
        "peak_bytes": stats.peak_bytes, "compute_submissions": stats.submitted, "validation_errors": stats.validation_errors,
        "png_encoding_tested": false, "eframe_lifecycle_tested": false }),
    )
}
fn main() {
    let result = execute();
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => json!({ "passed": false, "error": error }),
    };
    let path = std::path::Path::new("target/evidence/T-017/worker-readback.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    if let Err(error) = result {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
    println!("PASS: target/evidence/T-017/worker-readback.json");
}
