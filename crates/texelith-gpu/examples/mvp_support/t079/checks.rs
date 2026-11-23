//! Original numerical fixtures, shared by the dedicated test and hardware suite.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;

type Result<T> = std::result::Result<T, String>;
pub const BUDGET: u64 = 1024 * 1024 * 1024;
pub fn wait<T>(mut next: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(value) = next()? {
            return Ok(value);
        }
        if Instant::now() >= deadline {
            return Err("T-079 worker timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn source(extent: Extent, value: Option<f32>) -> TypedSourceUpload {
    let pixels = TypedSourcePixels::HeightF32(
        (0..extent.width * extent.height)
            .map(|i| value.unwrap_or((i % 5) as f32 / 4.0))
            .collect::<Vec<_>>()
            .into(),
    );
    TypedSourceUpload {
        descriptor: pixels.descriptor(extent).unwrap(),
        pixels,
    }
}
fn install(worker: &GpuWorker, upload: TypedSourceUpload) -> Result<()> {
    let hash = upload.descriptor.content_hash;
    worker
        .install_typed_source(upload)
        .map_err(|e| e.to_string())?;
    wait(|| match worker.try_event() {
        Some(WorkerEvent::SourceReady { content_hash }) if content_hash == hash => Ok(Some(())),
        Some(event) => Err(format!("source installation: {event:?}")),
        None => Ok(None),
    })
}
fn snapshot(source: SourceDescriptor, generation: u64) -> Arc<EvaluationSnapshot> {
    let output = OutputRef {
        node: NodeId(1),
        port: OutputPort::Height,
    };
    let graph = Graph::try_from(MaterialGraphDto {
        schema_version: MATERIAL_GRAPH_SCHEMA_VERSION,
        graph: vec![MaterialNode {
            id: NodeId(1),
            params: MaterialNodeParams::HeightSource(source),
            inputs: vec![],
        }],
        outputs: [(MapKind::Height, output)].into(),
        selected_output: output,
    })
    .unwrap();
    Arc::new(EvaluationSnapshot { generation, graph })
}
fn output(name: &str, format: ExportFormat, channels: usize) -> ExportOutput {
    ExportOutput {
        name: name.into(),
        format,
        channels: (0..channels)
            .map(|c| ChannelSpec {
                source: if c == 0 {
                    ChannelSource::Map(
                        OutputRef {
                            node: NodeId(1),
                            port: OutputPort::Height,
                        },
                        Component::R,
                    )
                } else {
                    ChannelSource::Constant(c as f32 / 4.0)
                },
                invert: false,
                transfer: Transfer::Linear,
            })
            .collect(),
        normal_has_negative_z: false,
        quantization: None,
        import_hints: None,
    }
}
fn request(job: u64, snapshot: &EvaluationSnapshot, outputs: Vec<ExportOutput>) -> ExportRequest {
    let plan = ExportPlan {
        source_extent: snapshot
            .graph
            .material_output_extent(OutputRef {
                node: NodeId(1),
                port: OutputPort::Height,
            })
            .unwrap(),
        size: SizePolicy::Keep,
        naming: "T_{name}_{map}".into(),
        profile: EngineProfile {
            id: "custom".into(),
            revision: 1,
            engine_safe_srgb: false,
            normal_convention: NormalConvention::OpenGl,
        },
        outputs,
        displacement: DisplacementSettings::default(),
        padding: Boundary::Clamp,
    };
    ExportRequest::new(
        JobId(job),
        ExportCapture::new(snapshot, 79, &plan).unwrap(),
        plan,
    )
    .unwrap()
}
fn plane(worker: &GpuWorker) -> Result<ExportPlane> {
    wait(|| {
        if let Some(progress) = worker.try_export_progress()
            && matches!(progress.stage, ExportStage::Failed | ExportStage::Cancelled)
        {
            return Err(format!("unexpected export termination: {progress:?}"));
        }
        if let Some(event) = worker.try_event() {
            return Err(format!("unexpected worker event: {event:?}"));
        }
        Ok(worker.try_export_plane())
    })
}
fn stage(worker: &GpuWorker, expected: ExportStage) -> Result<ExportProgress> {
    wait(|| {
        let progress = worker.try_export_progress();
        if let Some(p) = &progress {
            if p.stage == expected {
                return Ok(progress);
            }
            if matches!(p.stage, ExportStage::Cancelled | ExportStage::Failed) {
                return Err(format!("unexpected stage: {p:?}"));
            }
        }
        Ok(None)
    })
}
fn finish(worker: &GpuWorker, job: u64) -> Result<()> {
    stage(worker, ExportStage::ReadyToPublish)?;
    require(
        worker.finish_export(JobId(job), Ok(())).is_err(),
        "publication requires explicit begin",
    )?;
    worker
        .begin_export_publication(JobId(job))
        .map_err(|e| e.to_string())?;
    stage(worker, ExportStage::Publishing)?;
    worker
        .finish_export(JobId(job), Ok(()))
        .map_err(|e| e.to_string())?;
    stage(worker, ExportStage::Completed)?;
    Ok(())
}
fn verify_plane(
    plane: &ExportPlane,
    channels: usize,
    format: ExportFormat,
    value: Option<f32>,
) -> Result<()> {
    let extent = plane.metadata.extent;
    let bytes = (format.bit_depth() / 8) as usize;
    require(
        plane.pixels.len() == extent.width as usize * extent.height as usize * channels * bytes,
        "pitch or carrier channels leaked into CPU plane",
    )?;
    for (i, pixel) in plane.pixels.chunks_exact(channels * bytes).enumerate() {
        for c in 0..channels {
            let input = if c == 0 {
                value.unwrap_or((i % 5) as f32 / 4.0)
            } else {
                c as f32 / 4.0
            };
            let correct = match format {
                ExportFormat::Png8 => pixel[c] == (input * 255.0).round() as u8,
                ExportFormat::Png16 => {
                    u16::from_be_bytes(pixel[c * 2..c * 2 + 2].try_into().unwrap())
                        == (input * 65535.0).round() as u16
                }
                ExportFormat::Exr32 => {
                    f32::from_le_bytes(pixel[c * 4..c * 4 + 4].try_into().unwrap()) == input - 0.5
                }
                _ => false,
            };
            if !correct {
                return Err(format!("numeric mismatch at {i}/{c}: {format:?}"));
            }
        }
    }
    Ok(())
}

pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    size: u32,
) -> Result<serde_json::Value> {
    let worker = GpuWorker::spawn(device.clone(), queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let extent = Extent::new(257, 3).unwrap();
    let upload = source(extent, None);
    let original = snapshot(upload.descriptor.clone(), 7);
    install(&worker, upload)?;
    let formats = [
        (ExportFormat::Png8, 1),
        (ExportFormat::Png8, 3),
        (ExportFormat::Png8, 4),
        (ExportFormat::Png16, 1),
        (ExportFormat::Png16, 3),
        (ExportFormat::Png16, 4),
        (ExportFormat::Exr32, 1),
    ];
    let export_request = request(
        79,
        &original,
        formats
            .iter()
            .enumerate()
            .map(|(i, (format, channels))| output(&format!("item_{i}"), *format, *channels))
            .collect(),
    );
    let hash = export_request.capture().graph_hash;
    worker
        .request_export(export_request.clone())
        .map_err(|e| e.to_string())?;
    require(
        worker.request_export(export_request).err() == Some(SaveError::Busy),
        "second export must be Busy",
    )?;
    for (index, (format, channels)) in formats.iter().enumerate() {
        let result = plane(&worker)?;
        require(
            result.job_id == JobId(79)
                && result.item_index == index
                && result.output_name == format!("item_{index}"),
            "job/item identity changed",
        )?;
        require(
            result.generation == 7 && result.document_revision == 79 && result.graph_hash == hash,
            "immutable capture metadata changed",
        )?;
        verify_plane(&result, *channels, *format, None)?;
        if index == 0 {
            // Keep the IO consumer occupied while the live document and source
            // registry change enough to evict the original without protection.
            for revision in 1..=9 {
                let live_upload = source(extent, Some(revision as f32 / 10.0));
                let live = snapshot(live_upload.descriptor.clone(), 7 + revision);
                install(&worker, live_upload)?;
                worker
                    .request_material(MaterialRequest {
                        job_id: JobId(200),
                        request_id: revision,
                        snapshot: live.clone(),
                        outputs: live.graph.material_outputs().unwrap().clone(),
                        extent,
                        quality: Quality::Full,
                    })
                    .map_err(|e| e.to_string())?;
                let bundle = wait(|| Ok(worker.try_latest_material()))?;
                require(
                    bundle.generation == 7 + revision,
                    "interactive work starved behind export IO",
                )?;
                drop(bundle);
                require(
                    worker.try_export_plane().is_none(),
                    "a second plane escaped before IO acknowledgement",
                )?;
            }
            require(
                worker.begin_export_publication(JobId(79)).is_err(),
                "partial set became publishable",
            )?;
        }
        worker
            .acknowledge_export(result, Ok(()))
            .map_err(|e| e.to_string())?;
    }
    finish(&worker, 79)?;
    let normal_request = request_for_cancel(&original, 80);
    wait(|| match worker.request_export(normal_request.clone()) {
        Ok(()) => Ok(Some(())),
        Err(SaveError::Busy) => Ok(None),
        Err(e) => Err(e.to_string()),
    })?;
    let held = plane(&worker)?;
    worker.cancel_export(JobId(80)).map_err(|e| e.to_string())?;
    stage(&worker, ExportStage::Cancelled)?;
    require(
        worker.try_export_plane().is_none(),
        "cancelled job published another plane",
    )?;
    require(
        worker
            .request_export(request_for_cancel(&original, 81))
            .err()
            == Some(SaveError::Busy),
        "cancel bypassed externally held plane bound",
    )?;
    require(
        worker.acknowledge_export(held, Ok(())).err() == Some(SaveError::Cancelled),
        "late encode acknowledgement revived cancellation",
    )?;
    // A new request with a missing content identity must fail, never use the
    // most recently installed source with a matching extent.
    let missing = snapshot(source(extent, Some(0.123456)).descriptor, 40);
    let absent = request_for_cancel(&missing, 81);
    wait(|| match worker.request_export(absent.clone()) {
        Ok(()) => Ok(Some(())),
        Err(SaveError::Busy) => Ok(None),
        Err(e) => Err(e.to_string()),
    })?;
    let failure = stage(&worker, ExportStage::Failed)?;
    require(
        failure.job_id == JobId(81)
            && failure.output_name == "cancel"
            && failure
                .error
                .as_ref()
                .is_some_and(|e| e.contains("source hash")),
        "missing source lost output identity",
    )?;
    sizing_and_distinct_refs(&worker)?;
    let d = worker.diagnostics();
    worker.join().map_err(|e| e.to_string())?;
    require(
        worker.diagnostics().allocated_bytes == 0,
        "close did not release memory",
    )?;

    // Budget chosen to fit upload and packing, but not the carrier plus aligned
    // staging and CPU plane. A payload-only calculation would wrongly admit it.
    let small = Extent::new(17, 3).unwrap();
    let upload = source(small, None);
    let small_snapshot = snapshot(upload.descriptor.clone(), 1);
    let low = GpuWorker::spawn(
        device.clone(),
        queue.clone(),
        GpuConfig {
            budget_bytes: 2048,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    install(&low, upload)?;
    low.request_export(request(
        1,
        &small_snapshot,
        vec![output("budget_item", ExportFormat::Png8, 3)],
    ))
    .map_err(|e| e.to_string())?;
    let failed = stage(&low, ExportStage::Failed)?;
    require(
        failed.output_name == "budget_item"
            && failed
                .error
                .as_ref()
                .is_some_and(|e| e.contains("BudgetExceeded")),
        "aligned readback must fail bounded with item identity",
    )?;
    require(
        low.diagnostics().peak_bytes <= 2048,
        "low-budget readback overallocated",
    )?;
    low.join().map_err(|e| e.to_string())?;
    require(
        low.diagnostics().allocated_bytes == 0,
        "failed readback retained memory",
    )?;

    // Two 4K RGBA carriers and two staging planes alone would use the entire
    // 1 GiB budget. The live job must stream, not allocate per output in advance.
    let large = Extent::new(4096, 4096).unwrap();
    let big_upload = source(large, Some(0.25));
    let big_snapshot = snapshot(big_upload.descriptor.clone(), 50);
    let big = GpuWorker::spawn(device.clone(), queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    install(&big, big_upload)?;
    let big_request = request(
        1,
        &big_snapshot,
        vec![
            output("large_a", ExportFormat::Png16, 4),
            output("large_b", ExportFormat::Png16, 4),
        ],
    );
    big.request_export(big_request).map_err(|e| e.to_string())?;
    let first = plane(&big)?;
    verify_plane(&first, 4, ExportFormat::Png16, Some(0.25))?;
    let peak = big.diagnostics().peak_bytes;
    require(
        (512 * 1024 * 1024..=BUDGET).contains(&peak),
        "4K staging missing from shared budget",
    )?;
    big.acknowledge_export(first, Ok(()))
        .map_err(|e| e.to_string())?;
    wait(|| {
        if let Some(p) = big.try_export_progress()
            && p.stage == ExportStage::Failed
        {
            return Err(format!("4K export failed: {p:?}"));
        }
        Ok((big.diagnostics().readback_reserved_bytes > 0).then_some(()))
    })?;
    big.cancel_export(JobId(1)).map_err(|e| e.to_string())?;
    stage(&big, ExportStage::Cancelled)?;
    wait(|| {
        let d = big.diagnostics();
        Ok((d.in_flight == 0
            && d.readback_reserved_bytes == 0
            && d.allocated_bytes < 256 * 1024 * 1024)
            .then_some(()))
    })?;
    require(
        big.try_export_plane().is_none(),
        "in-flight cancellation published a partial set",
    )?;
    let closing = request_for_cancel(&big_snapshot, 2);
    wait(|| match big.request_export(closing.clone()) {
        Ok(()) => Ok(Some(())),
        Err(SaveError::Busy) => Ok(None),
        Err(e) => Err(e.to_string()),
    })?;
    wait(|| Ok((big.diagnostics().readback_reserved_bytes > 0).then_some(())))?;
    big.close();
    let closed = stage(&big, ExportStage::Cancelled)?;
    require(closed.job_id == JobId(2), "close lost export job identity")?;
    big.join().map_err(|e| e.to_string())?;
    require(
        big.diagnostics().allocated_bytes == 0,
        "cancelled staging did not retire",
    )?;
    require(
        big.try_export_plane().is_none(),
        "close published a late plane",
    )?;
    require(
        big.request_export(request_for_cancel(&big_snapshot, 3))
            .err()
            == Some(SaveError::Closed),
        "closed worker accepted export",
    )?;

    // A measured requested-size transfer, without file encode/publication timing.
    let timed_extent = Extent::new(size, size).map_err(|e| e.to_string())?;
    let timed_upload = source(timed_extent, Some(0.25));
    let timed_snapshot = snapshot(timed_upload.descriptor.clone(), 60);
    let timed = GpuWorker::spawn(device, queue, GpuConfig::default()).map_err(|e| e.to_string())?;
    install(&timed, timed_upload)?;
    let start = Instant::now();
    timed
        .request_export(request(
            1,
            &timed_snapshot,
            vec![output("timed", ExportFormat::Png8, 3)],
        ))
        .map_err(|e| e.to_string())?;
    let result = plane(&timed)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    verify_plane(&result, 3, ExportFormat::Png8, Some(0.25))?;
    timed
        .acknowledge_export(result, Ok(()))
        .map_err(|e| e.to_string())?;
    finish(&timed, 1)?;
    let td = timed.diagnostics();
    timed.join().map_err(|e| e.to_string())?;
    for diag in [d, big.diagnostics(), low.diagnostics(), td] {
        require(
            diag.validation_errors == 0
                && diag.stale_publications == 0
                && diag.peak_in_flight <= 1
                && diag.peak_bytes <= BUDGET,
            "worker validation, publication or memory bound failed",
        )?;
    }
    Ok(serde_json::json!({
        "passed":true, "budget_bytes":BUDGET, "peak_bytes":peak, "peak_in_flight":1,
        "validation_errors":0, "stale_publications":0, "shared_device":true,
        "formats_tested":["R8","RGB8","RGBA8","R16","RGB16","RGBA16","Y32F"],
        "odd_width":257, "large_extent":[4096,4096], "timed_size":size, "readback_ms":elapsed,
        "checks":["pitch_strip","immutable_snapshot","source_registry_churn","one_plane_backpressure","interactive_priority","job_item_identity","signed_float","low_budget","4k_staging_budget","cancel_retires_memory","close_retires_memory","missing_source_rejected","publication_phases","resize_pad_metadata","distinct_output_refs"]
    }))
}
fn request_for_cancel(snapshot: &EvaluationSnapshot, job: u64) -> ExportRequest {
    request(job, snapshot, vec![output("cancel", ExportFormat::Png8, 1)])
}

fn sizing_and_distinct_refs(worker: &GpuWorker) -> Result<()> {
    let extent = Extent::new(19, 5).unwrap();
    let upload = source(extent, Some(0.25));
    let snapshot = snapshot(upload.descriptor.clone(), 45);
    install(worker, upload.clone())?;
    for (job, size, expected) in [
        (
            82,
            SizePolicy::Resize {
                width: 17,
                height: 3,
            },
            Extent::new(17, 3).unwrap(),
        ),
        (
            83,
            SizePolicy::Pad {
                width: 23,
                height: 7,
            },
            Extent::new(23, 7).unwrap(),
        ),
    ] {
        let base = request(
            job,
            &snapshot,
            vec![output("sized", ExportFormat::Png16, 4)],
        );
        let mut plan = base.plan().clone();
        plan.size = size;
        let resized = ExportRequest::new(
            JobId(job),
            ExportCapture::new(&snapshot, 79, &plan).unwrap(),
            plan,
        )
        .unwrap();
        wait(|| match worker.request_export(resized.clone()) {
            Ok(()) => Ok(Some(())),
            Err(SaveError::Busy) => Ok(None),
            Err(e) => Err(e.to_string()),
        })?;
        let result = plane(worker)?;
        require(
            result.metadata.extent == expected,
            "export sizing metadata incorrect",
        )?;
        require(
            result.payload
                == export_shared::ExportPayload::Numeric {
                    channels: 4,
                    bit_depth: 16,
                },
            "numeric payload ABI incorrect",
        )?;
        if matches!(size, SizePolicy::Pad { .. }) {
            require(
                result.metadata.valid_content == [0, 0, 19, 5]
                    && result.metadata.uv_scale == [19.0 / 23.0, 5.0 / 7.0],
                "pad rectangle or UV metadata lost",
            )?;
        }
        verify_plane(&result, 4, ExportFormat::Png16, Some(0.25))?;
        worker
            .acknowledge_export(result, Ok(()))
            .map_err(|e| e.to_string())?;
        finish(worker, job)?;
    }
    let a = OutputRef {
        node: NodeId(1),
        port: OutputPort::Height,
    };
    let b = OutputRef {
        node: NodeId(2),
        port: OutputPort::Height,
    };
    let graph = Graph::try_from(MaterialGraphDto {
        schema_version: MATERIAL_GRAPH_SCHEMA_VERSION,
        graph: vec![
            MaterialNode {
                id: a.node,
                inputs: vec![],
                params: MaterialNodeParams::HeightSource(upload.descriptor),
            },
            MaterialNode {
                id: b.node,
                inputs: vec![InputEdge {
                    port: InputPort::Height,
                    source: a,
                }],
                params: MaterialNodeParams::Levels {
                    black: 0.0,
                    white: 0.5,
                    gamma: 1.0,
                    boundary: Boundary::Clamp,
                },
            },
        ],
        outputs: [(MapKind::Height, b)].into(),
        selected_output: b,
    })
    .unwrap();
    let snapshot = EvaluationSnapshot {
        generation: 46,
        graph,
    };
    let mut packed = output("distinct_refs", ExportFormat::Png8, 3);
    packed.channels[1].source = ChannelSource::Map(b, Component::R);
    packed.channels[2].source = ChannelSource::Constant(0.75);
    let request = request(84, &snapshot, vec![packed]);
    wait(|| match worker.request_export(request.clone()) {
        Ok(()) => Ok(Some(())),
        Err(SaveError::Busy) => Ok(None),
        Err(e) => Err(e.to_string()),
    })?;
    let result = plane(worker)?;
    require(
        result
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .all(|p| *p == [64, 128, 191]),
        "same-kind OutputRefs were collapsed",
    )?;
    worker
        .acknowledge_export(result, Ok(()))
        .map_err(|e| e.to_string())?;
    finish(worker, 84)
}
