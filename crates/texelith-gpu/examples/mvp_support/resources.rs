//! Hardware checks use the production worker, not a parallel model of its planner.
use super::contract;
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, MapKind, MaterialGraphDto, Node,
    NodeId, NodeParams, SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuWorker, JobId, MaterialRequest, Quality, ResourceReservations,
    TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

const MIB: u64 = 1024 * 1024;
const BUDGET: u64 = 1024 * MIB;
type Result<T> = std::result::Result<T, String>;
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn wait<T>(mut check: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = check()? {
            return Ok(value);
        }
        if Instant::now() >= deadline {
            return Err("resource check timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn events(worker: &GpuWorker) -> Result<Option<[u8; 32]>> {
    match worker.try_event() {
        Some(WorkerEvent::SourceReady { content_hash }) => Ok(Some(content_hash)),
        Some(event) => Err(format!("unexpected worker event: {event:?}")),
        None => Ok(None),
    }
}
fn complete(worker: &GpuWorker) -> Result<CompletedMaterial> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        events(worker)?;
        if let Some(bundle) = worker.try_latest_material() {
            return Ok(bundle);
        }
        if Instant::now() >= deadline {
            return Err("material publication timeout".into());
        }
        std::thread::yield_now();
    }
}
fn install(
    worker: &GpuWorker,
    extent: Extent,
    pixels: TypedSourcePixels,
) -> Result<SourceDescriptor> {
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    let ready = wait(|| events(worker))?;
    require(
        ready == descriptor.content_hash,
        "wrong source acknowledgement",
    )?;
    Ok(descriptor)
}
fn request(
    source: &SourceDescriptor,
    job: u64,
    id: u64,
    levels: u64,
    strength: f32,
) -> MaterialRequest {
    let mut nodes = vec![Node {
        id: NodeId(1),
        input: None,
        params: NodeParams::Source(source.clone()),
    }];
    for i in 0..levels {
        nodes.push(Node {
            id: NodeId(i + 2),
            input: Some(NodeId(i + 1)),
            params: NodeParams::Levels {
                black: 0.0,
                white: 1.0,
                gamma: 1.0,
            },
        });
    }
    nodes.push(Node {
        id: NodeId(levels + 2),
        input: Some(NodeId(levels + 1)),
        params: NodeParams::NormalFromHeight { strength },
    });
    let graph = Graph::try_from(MaterialGraphDto::from(
        Graph::new(nodes, NodeId(levels + 2)).unwrap(),
    ))
    .unwrap();
    MaterialRequest {
        job_id: JobId(job),
        request_id: id,
        outputs: graph.material_outputs().unwrap().clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 1,
            graph,
        }),
        extent: source.extent,
        quality: Quality::Full,
    }
}
fn typed_request(source: &SourceDescriptor, job: u64) -> MaterialRequest {
    let mut request = request(source, job, 1, 0, 1.0);
    let mut dto = MaterialGraphDto::from(request.snapshot.graph.clone());
    dto.graph[0].params = texelith_core::MaterialNodeParams::HeightSource(source.clone());
    request.snapshot = Arc::new(EvaluationSnapshot {
        generation: 1,
        graph: Graph::try_from(dto).unwrap(),
    });
    request
}

fn source_only(
    source: &SourceDescriptor,
    params: texelith_core::MaterialNodeParams,
    port: texelith_core::OutputPort,
    job: u64,
) -> MaterialRequest {
    let output = texelith_core::OutputRef {
        node: NodeId(1),
        port,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![texelith_core::MaterialNode {
            id: NodeId(1),
            inputs: Vec::new(),
            params,
        }],
        outputs: [(port.map_kind(), output)].into(),
        selected_output: output,
    };
    MaterialRequest {
        job_id: JobId(job),
        request_id: 1,
        outputs: dto.outputs.clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: 1,
            graph: Graph::try_from(dto).unwrap(),
        }),
        extent: source.extent,
        quality: Quality::Full,
    }
}

// This buffer is covered by the worker's held readback reservation. All reads
// are serialized, and the buffer is destroyed before that phase is released.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Result<Vec<u8>> {
    let bpp = if texture.format() == wgpu::TextureFormat::Rgba16Float {
        8
    } else {
        4
    };
    let pitch = (texture.width() * bpp).div_ceil(256) * 256;
    let bytes = u64::from(pitch) * u64::from(texture.height());
    require(bytes <= 128 * MIB, "readback exceeds reserved share")?;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resource oracle reserved readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    rx.recv_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let packed = mapped
        .chunks_exact(pitch as usize)
        .flat_map(|row| row[..(texture.width() * bpp) as usize].iter().copied())
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(packed)
}
fn check_bundle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bundle: &CompletedMaterial,
    job: u64,
    value: f32,
) -> Result<()> {
    require(
        bundle.job_id == JobId(job) && bundle.generation == 1 && bundle.outputs.len() == 2,
        "incoherent material identity or output set",
    )?;
    let height = read(device, queue, bundle.outputs[&MapKind::Height].texture())?;
    require(
        height
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| (f32::from_le_bytes(*p) - value).abs() <= 2e-5),
        "source selection pixel mismatch",
    )?;
    let normals = read(device, queue, bundle.outputs[&MapKind::Normal].texture())?;
    require(
        super::harness::normals(&normals)
            .iter()
            .all(|n| n[0].abs() < 0.002 && n[1].abs() < 0.002 && (n[2] - 1.0).abs() < 0.002),
        "flat normal mismatch",
    )
}
fn reservations(worker: &GpuWorker, viewport: u64, readback: u64) -> Result<()> {
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: viewport,
            readback_bytes: readback,
        })
        .map_err(|e| e.to_string())
}

pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
) -> Result<Value> {
    let errors = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = errors.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        recorded.lock().unwrap().push(error.to_string())
    }));
    let worker = GpuWorker::spawn(device.clone(), queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let outcome = exercise(&worker, device.clone(), queue.clone(), extent, info, errors);
    worker.join().map_err(|e| e.to_string())?;
    let result = outcome?;
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    low_budget(device, queue)?;
    Ok(result)
}

fn low_budget(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<()> {
    let worker = GpuWorker::spawn(
        device,
        queue,
        GpuConfig {
            budget_bytes: 64 * 1024,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let outcome = (|| {
        let extent = Extent::new(16, 16).unwrap();
        let source = install(
            &worker,
            extent,
            TypedSourcePixels::Rgba8 {
                pixels: vec![128; 16 * 16 * 4].into(),
                interpretation: InputInterpretation::LinearHeight,
            },
        )?;
        reservations(&worker, 60 * 1024, 0)?;
        worker
            .request_material(request(&source, 1, 1, 3, 1.0))
            .map_err(|e| e.to_string())?;
        wait(|| match worker.try_event() {
            Some(WorkerEvent::BudgetExceeded {
                budget_bytes: 65536,
                ..
            }) => Ok(Some(())),
            Some(e) => Err(format!("low-budget worker did not reject: {e:?}")),
            None => Ok(None),
        })?;
        require(
            worker.try_latest_material().is_none(),
            "low-budget partial publication",
        )?;
        reservations(&worker, 0, 0)?;
        worker
            .request_material(request(&source, 1, 2, 3, 1.0))
            .map_err(|e| e.to_string())?;
        let recovered = complete(&worker)?;
        require(
            recovered.outputs.len() == 2 && worker.diagnostics().peak_bytes <= 65536,
            "low-budget recovery failed",
        )
    })();
    worker.join().map_err(|e| e.to_string())?;
    outcome
}
fn exercise(
    worker: &GpuWorker,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
    errors: Arc<std::sync::Mutex<Vec<String>>>,
) -> Result<Value> {
    let mut checks = Vec::new();
    let tiny = Extent::new(17, 13).unwrap();
    reservations(worker, 144 * MIB, 128 * MIB)?;
    let rgba = |value: u8, e: Extent| TypedSourcePixels::Rgba8 {
        pixels: (0..e.width * e.height)
            .flat_map(|_| [value, value, value, 255])
            .collect::<Vec<_>>()
            .into(),
        interpretation: InputInterpretation::LinearHeight,
    };
    let first = install(worker, tiny, rgba(51, tiny))?;
    let second = install(worker, tiny, rgba(204, tiny))?;
    for (source, job, value) in [(&second, 2, 0.8), (&first, 1, 0.2)] {
        worker
            .request_material(request(source, job, 1, 2, 1.0))
            .map_err(|e| e.to_string())?;
        let bundle = complete(worker)?;
        require(
            bundle.request_id == 1 && bundle.extent == tiny && bundle.quality == Quality::Full,
            "wrong request metadata",
        )?;
        check_bundle(&device, &queue, &bundle, job, value)?;
        drop(bundle);
    }
    // Mismatched caller identity must fail without poisoning the resident registry.
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: first.clone(),
            pixels: rgba(204, tiny),
        })
        .map_err(|e| e.to_string())?;
    wait(|| match worker.try_event() {
        Some(WorkerEvent::Error(error)) if error.0.contains("content hash") => Ok(Some(())),
        Some(event) => Err(format!("expected source rejection, got {event:?}")),
        None => Ok(None),
    })?;
    worker
        .request_material(request(&first, 1, 2, 2, 1.0))
        .map_err(|e| e.to_string())?;
    check_bundle(&device, &queue, &complete(worker)?, 1, 0.2)?;
    for (pixels, value, job) in [
        (
            TypedSourcePixels::HeightU16(vec![32768; (tiny.width * tiny.height) as usize].into()),
            32768.0 / 65535.0,
            3,
        ),
        (
            TypedSourcePixels::HeightF32(
                vec![0.1234567; (tiny.width * tiny.height) as usize].into(),
            ),
            0.1234567,
            4,
        ),
    ] {
        let source = install(worker, tiny, pixels)?;
        worker
            .request_material(typed_request(&source, job))
            .map_err(|e| e.to_string())?;
        check_bundle(&device, &queue, &complete(worker)?, job, value)?;
    }
    let color = install(
        worker,
        tiny,
        TypedSourcePixels::Rgba8 {
            pixels: (0..tiny.width * tiny.height)
                .flat_map(|_| [128, 64, 255, 0])
                .collect::<Vec<_>>()
                .into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )?;
    worker
        .request_material(source_only(
            &color,
            texelith_core::MaterialNodeParams::ColorSource(color.clone()),
            texelith_core::OutputPort::Albedo,
            5,
        ))
        .map_err(|e| e.to_string())?;
    let colored = complete(worker)?;
    let values = read(&device, &queue, colored.outputs[&MapKind::Albedo].texture())?;
    require(
        super::harness::normals(&values).iter().all(|p| {
            (p[0] - 0.2158605).abs() < 0.002
                && (p[1] - 0.0512695).abs() < 0.002
                && p[2] == 1.0
                && p[3] == 1.0
        }),
        "typed color transfer mismatch",
    )?;
    drop(colored);
    for (job, convention) in [
        (6, texelith_core::NormalConvention::OpenGl),
        (7, texelith_core::NormalConvention::DirectX),
    ] {
        let normal = install(
            worker,
            tiny,
            TypedSourcePixels::NormalRgb8 {
                pixels: (0..tiny.width * tiny.height)
                    .flat_map(|_| [128, 255, 255])
                    .collect::<Vec<_>>()
                    .into(),
                convention,
            },
        )?;
        worker
            .request_material(source_only(
                &normal,
                texelith_core::MaterialNodeParams::NormalSource {
                    source: normal.clone(),
                    convention,
                },
                texelith_core::OutputPort::Normal,
                job,
            ))
            .map_err(|e| e.to_string())?;
        let bundle = complete(worker)?;
        require(
            bundle.normal_convention == Some(convention),
            "lost imported normal convention",
        )?;
        let values = read(&device, &queue, bundle.outputs[&MapKind::Normal].texture())?;
        require(
            super::harness::normals(&values).iter().all(|p| {
                (p[0] - 0.00277296).abs() < 0.002
                    && (p[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.002
                    && (p[2] - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.002
            }),
            "normal import normalized or flipped incorrectly",
        )?;
    }
    checks.extend(
        ["typed_sources", "job_identity", "coherent_outputs"]
            .map(|name| json!({"name":name,"passed":true})),
    );

    let source = install(worker, extent, rgba(102, extent))?;
    // The real 48-node chain requires >3 GiB without live-range recycling.
    worker
        .request_material(request(&source, 10, 1, 48, 1.0))
        .map_err(|e| e.to_string())?;
    let long = complete(worker)?;
    check_bundle(&device, &queue, &long, 10, 0.4)?;
    require(
        worker.diagnostics().peak_bytes <= BUDGET,
        "4K chain exceeded hard budget",
    )?;
    checks.push(json!({"name":"liveness","passed":true,"levels":48,"peak_bytes":worker.diagnostics().peak_bytes}));

    // Submit a final consumer before dropping its leases. Queue callbacks own
    // backing handles until completion; a surviving clone remains accounted.
    let before = worker.diagnostics().allocated_bytes;
    let held = long.outputs[&MapKind::Normal].clone();
    let pitch = (extent.width * 8).div_ceil(256) * 256;
    let consumer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reserved last lease consumer"),
        size: u64::from(pitch) * u64::from(extent.height),
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        held.texture().as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &consumer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(extent.height),
            },
        },
        held.texture().size(),
    );
    queue.submit([encoder.finish()]);
    drop(long);
    require(
        worker.diagnostics().allocated_bytes >= before.saturating_sub(128 * MIB),
        "live clone lost accounting",
    )?;
    drop(held);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    drop(consumer);
    require(
        worker.diagnostics().retired_bytes == 0,
        "draw fence did not retire",
    )?;
    checks.push(json!({"name":"lease_fence","passed":true,"method":"real queue completion after last consumer"}));

    // Every sample changes the normal key. Unchanged height is shared; this is
    // an actual full-node computation, never a warm-result-only timing sample.
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for id in 2..=111 {
        let r = request(&source, 10, id, 48, id as f32 / 12.0);
        let before_submit = worker.diagnostics().queue_submit_ns;
        let start = Instant::now();
        worker.request_material(r).map_err(|e| e.to_string())?;
        let bundle = complete(worker)?;
        submissions
            .push((worker.diagnostics().queue_submit_ns - before_submit) as f64 / 1_000_000.0);
        require(
            bundle.request_id == id && bundle.outputs.len() == 2,
            "timed material publication mismatch",
        )?;
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
        drop(bundle);
    }
    let timing = contract::aggregate_timings(&rows, 10)?;
    let stats = worker.diagnostics();
    require(
        stats.cache_evictions > 0 && stats.cache_hits > 0,
        "cache eviction/reuse not exercised",
    )?;
    checks.push(json!({"name":"cache_eviction","passed":true,"evictions":stats.cache_evictions,"hits":stats.cache_hits}));

    // A cold chain is deliberately superseded after at least one real submission.
    let before_submitted = worker.diagnostics().submitted;
    let mut cold = request(&source, 20, 1, 120, 1.0);
    let mut dto = MaterialGraphDto::from(cold.snapshot.graph.clone());
    if let texelith_core::MaterialNodeParams::Levels { gamma, .. } = &mut dto.graph[1].params {
        *gamma = 1.1;
    }
    cold.snapshot = Arc::new(EvaluationSnapshot {
        generation: 1,
        graph: Graph::try_from(dto).unwrap(),
    });
    worker.request_material(cold).map_err(|e| e.to_string())?;
    wait(|| {
        events(worker)?;
        let stats = worker.diagnostics();
        Ok((stats.in_flight == 1 && stats.submitted > before_submitted).then_some(()))
    })?;
    worker
        .request_material(request(&source, 21, 1, 2, 1.0))
        .map_err(|e| e.to_string())?;
    let latest = complete(worker)?;
    require(latest.job_id == JobId(21), "cancelled material published")?;
    drop(latest);
    let stats = worker.diagnostics();
    require(
        stats.cancelled_pass_batches > 0 && stats.submitted - before_submitted < 120,
        "superseded passes were not stopped",
    )?;
    require(
        stats.peak_in_flight == 1 && stats.peak_pending == 1 && stats.stale_publications == 0,
        "unbounded or stale publication",
    )?;
    checks.push(json!({"name":"cancellation","passed":true,"cancelled_batches":stats.cancelled_pass_batches,"submissions_after_cancel_test":stats.submitted-before_submitted}));

    // Fail through the real worker admission path, then recover on that worker.
    reservations(worker, BUDGET, 128 * MIB)?;
    worker
        .request_material(request(&source, 21, 2, 2, 1.0))
        .map_err(|e| e.to_string())?;
    wait(|| match worker.try_event() {
        Some(WorkerEvent::BudgetExceeded {
            budget_bytes,
            required_bytes,
        }) => {
            require(
                budget_bytes == BUDGET && required_bytes > budget_bytes,
                "unbounded budget error",
            )?;
            Ok(Some(()))
        }
        Some(event) => Err(format!("expected budget failure, got {event:?}")),
        None => Ok(None),
    })?;
    require(
        worker.try_latest_material().is_none(),
        "failed plan published a partial bundle",
    )?;
    reservations(worker, 144 * MIB, 128 * MIB)?;
    worker
        .request_material(request(&source, 21, 3, 2, 1.0))
        .map_err(|e| e.to_string())?;
    let recovery = complete(worker)?;
    require(
        recovery.request_id == 3 && recovery.outputs.len() == 2,
        "worker did not recover from budget error",
    )?;
    drop(recovery);
    checks.push(json!({"name":"low_budget","passed":true}));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    let stats = worker.diagnostics();
    require(
        errors.lock().unwrap().is_empty() && stats.validation_errors == 0,
        "GPU validation error",
    )?;
    let manifest = json!({"scenario":"material_resources","size":extent.width,"seed":42,
        "chain_levels":48,"cancellation_chain_levels":120,"normal_strength":"request_id / 12, ids 2..111",
        "viewport_bytes":144*MIB,"readback_bytes":128*MIB});
    let mut serialized = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while serialized.len() % 4 != 0 {
        serialized.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &serialized,
        Extent::new((serialized.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .unwrap()
    .content_hash
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    let result = json!({"passed":true,"suite":"resources","size":extent.width,
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"seed":42,
        "manifest":manifest,"manifest_hash":hash,"checks":checks,
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":timing.measurements,
            "p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,
            "peak_bytes":stats.peak_bytes,"cancellation_errors":stats.stale_publications,
            "scope":"material request through coherent publication, changed normal key, shared cached height; includes worker scheduling and polling",
            "submission_scope":"sum of actual command finish and queue submit CPU durations per material"},
        "resources":{"budget_bytes":BUDGET,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,
            "cancellation_requests":stats.cancellation_requests,"cancelled_publications":stats.stale_publications,
            "cancelled_pass_batches":stats.cancelled_pass_batches,"validation_errors":stats.validation_errors,
            "viewport_reserved_bytes":stats.viewport_reserved_bytes,"readback_reserved_bytes":stats.readback_reserved_bytes,
            "accounting":"source, cache, live scratch, queue staging, uniforms, retired leases, viewport/IBL reservation, serialized readback reservation; driver heaps and eframe surfaces excluded"}});
    contract::validate_resources_report(&result).map_err(|e| format!("{e}; report={result}"))?;
    Ok(result)
}
