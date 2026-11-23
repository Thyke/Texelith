//! Real headless command/worker gate; native interaction remains a separate gate.
use super::*;
use crate::startup::Config;
use std::{
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn pump(
    device: &wgpu::Device,
    batch: &mut BatchJobs,
    exports: &mut ExportJobs,
    c: &mut Controls,
    mut done: impl FnMut(&Controls) -> bool,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        device.poll(wgpu::PollType::Poll).map_err(err)?;
        batch.tick(c, exports);
        exports.tick(c);
        if done(c) {
            return Ok(());
        }
        if let Some(error) = &c.batch_status().error {
            return Err(error.clone());
        }
        if start.elapsed() > Duration::from_secs(180) {
            return Err("Batch diagnostic timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn finished(c: &Controls) -> bool {
    !c.batch_status().busy
}
fn count(c: &Controls, state: BatchState) -> usize {
    c.session()
        .batch()
        .unwrap()
        .items()
        .iter()
        .filter(|i| i.state == state)
        .count()
}
fn roughness(c: &mut Controls, value: f32) {
    let node = c
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_outputs()
        .unwrap()[&MapKind::Roughness]
        .node;
    c.action(UiAction::Change {
        address: ParameterAddress {
            node,
            parameter: ParameterId::Curve,
        },
        value: ParameterValue::Curve(vec![[0.0, value], [1.0, value]]),
    });
}
pub fn exercise(config: &Config) -> Result<serde_json::Value, String> {
    let parent = config
        .report
        .as_ref()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("target-evidence/T-087"));
    let root = parent.join(format!(
        "batch-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(err)?
            .as_nanos()
    ));
    let input = root.join("inputs");
    std::fs::create_dir_all(&input).map_err(err)?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = config.backends()?;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = crate::material_ui::block_on(instance.enumerate_adapters(config.backends()?))
        .into_iter()
        .find(|a| {
            config.adapter_name.as_ref().is_none_or(|name| {
                a.get_info()
                    .name
                    .to_lowercase()
                    .contains(&name.to_lowercase())
            })
        })
        .ok_or("Requested native GPU adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let (device, queue) =
        crate::material_ui::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 4096,
                ..Default::default()
            },
            ..Default::default()
        }))
        .map_err(err)?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let gpu =
        Arc::new(GpuWorker::spawn(device.clone(), queue.clone(), Default::default()).map_err(err)?);
    #[cfg(test)]
    super::regressions::register_context(&root, gpu.clone(), device.clone(), queue);
    let result = exercise_jobs(config, &root, &input, &device, gpu.clone());
    gpu.close();
    gpu.join().map_err(err)?;
    result.map(|mut report| {
        report["adapter"] = info.name.into();
        report["backend"] = format!("{:?}", info.backend).into();
        report["artifacts"] = root.to_string_lossy().into_owned().into();
        report
    })
}
fn exercise_jobs(
    config: &Config,
    root: &Path,
    input: &Path,
    device: &wgpu::Device,
    gpu: Arc<GpuWorker>,
) -> Result<serde_json::Value, String> {
    let size = config.size;
    let fixture = texelith_io::encoders::encode_png8(
        size,
        size,
        texelith_io::encoders::PngChannels::Rgb,
        texelith_io::encoders::PngColorIntent::Srgb,
        &[128, 64, 32].repeat(size as usize * size as usize),
    )
    .map_err(err)?;
    for name in ["a.png", "c.png"] {
        std::fs::write(input.join(name), &fixture).map_err(err)?;
    }
    let odd_width = (size / 2).max(1) + 1;
    let odd_height = (size / 2).max(1) + 3;
    let odd_fixture = texelith_io::encoders::encode_png8(
        odd_width,
        odd_height,
        texelith_io::encoders::PngChannels::Rgb,
        texelith_io::encoders::PngColorIntent::Srgb,
        &[32, 64, 128].repeat(odd_width as usize * odd_height as usize),
    )
    .map_err(err)?;
    std::fs::write(input.join("d.png"), odd_fixture).map_err(err)?;
    std::fs::write(input.join("b-bad.png"), b"invalid image").map_err(err)?;
    let mut c = Controls::new_material();
    c.open_image(input.join("a.png"));
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        return Err("Missing import effect".into());
    };
    let source = SourceDescriptor::from_rgba8(
        &[128, 64, 32, 255].repeat(size as usize * size as usize),
        Extent::new(size, size).map_err(err)?,
        InputInterpretation::SrgbPhoto,
    )
    .map_err(err)?;
    c.complete_import(token, source)?;
    c.export_options_mut().preset = "blender-v1".into();
    roughness(&mut c, 0.25);
    let selection = InputSelection::Folder {
        path: input.to_owned(),
        recursive: false,
    };
    let mut batch = BatchJobs::spawn(gpu.clone())?;
    let mut exports = ExportJobs::spawn(gpu.clone())?;
    c.queue_batch(
        selection.clone(),
        root.join("continue"),
        root.join("continue.json"),
        true,
    )?;
    roughness(&mut c, 0.75);
    let interactive = Arc::new(
        c.session()
            .snapshot()
            .ok_or("Missing interactive material")?,
    );
    let mut interactive_outputs = interactive
        .graph
        .material_outputs()
        .ok_or("Missing outputs")?
        .clone();
    interactive_outputs.retain(|kind, _| *kind == MapKind::Roughness);
    let mut interactive_sent = false;
    let mut interactive_error = None;
    pump(device, &mut batch, &mut exports, &mut c, |c| {
        #[cfg(not(test))]
        let interactive_allowed = true;
        #[cfg(test)]
        let interactive_allowed = super::regressions::interactive_allowed(root);
        if !interactive_sent && c.export_busy() && interactive_allowed {
            interactive_error = gpu
                .request_material(texelith_gpu::MaterialRequest {
                    job_id: texelith_gpu::JobId(u64::MAX - 87),
                    request_id: 87,
                    snapshot: interactive.clone(),
                    outputs: interactive_outputs.clone(),
                    extent: Extent {
                        width: size,
                        height: size,
                    },
                    quality: texelith_gpu::Quality::Full,
                })
                .err()
                .map(err);
            interactive_sent = true;
        }
        finished(c)
    })?;
    require(
        interactive_sent && interactive_error.is_none(),
        "Interactive GPU request was not admitted during batch export",
    )?;
    let started = Instant::now();
    loop {
        device.poll(wgpu::PollType::Poll).map_err(err)?;
        if let Some(frame) = gpu.try_latest_material() {
            require(
                frame.job_id == texelith_gpu::JobId(u64::MAX - 87)
                    && frame.generation == interactive.generation,
                "Batch crossed interactive publication identity",
            )?;
            break;
        }
        if started.elapsed() > Duration::from_secs(30) {
            return Err("Interactive GPU publication timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    require(
        count(&c, BatchState::Committed) == 3 && count(&c, BatchState::Failed) == 1,
        "Continue mode lost per-item outcomes",
    )?;
    require(
        c.session().batch().unwrap().recipe().graph != c.session().snapshot().unwrap().graph,
        "Interactive edits changed captured recipe",
    )?;
    let assignment = texelith_io::project_io::read_json(&root.join("continue/a/assignment.json"))
        .map_err(err)?;
    let plan: ExportPlan = serde_json::from_value(assignment["plan"].clone()).map_err(err)?;
    for (output, record) in plan
        .outputs
        .iter()
        .zip(assignment["outputs"].as_array().ok_or("Missing outputs")?)
    {
        if output.format == ExportFormat::Png8 {
            for (channel, spec) in output.channels.iter().enumerate() {
                if matches!(spec.source, ChannelSource::Map(reference, _) if reference.port == OutputPort::Roughness)
                {
                    let path = root
                        .join("continue/a")
                        .join(record["file"].as_str().ok_or("Missing file")?);
                    let image = texelith_io::decode_path(path, texelith_io::ColorIntent::PhotoSrgb)
                        .map_err(err)?;
                    require(
                        image
                            .rgba8
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|p| (i32::from(p[channel]) - 64).abs() <= 1),
                        "Export did not retain captured roughness",
                    )?;
                }
            }
        }
    }
    let recovered = recover_manifest(&root.join("continue.json"))?;
    let recovery_path = root.join("recovery.json");
    let recovery_paths = texelith_io::project_io::recovery_paths(&recovery_path).map_err(err)?;
    let valid_manifest = std::fs::read(root.join("continue.json")).map_err(err)?;
    let mut invalid = texelith_io::project_io::parse_json(&valid_manifest).map_err(err)?;
    invalid["recipe_digest"] = "corrupt".into();
    std::fs::write(&recovery_paths.backup, &valid_manifest).map_err(err)?;
    std::fs::write(
        &recovery_path,
        texelith_io::project_io::canonical_json(&invalid).map_err(err)?,
    )
    .map_err(err)?;
    require(
        recover_manifest(&recovery_path)?
            .job
            .items()
            .iter()
            .filter(|i| i.state == BatchState::Committed)
            .count()
            == 3,
        "Invalid manifest did not recover its valid backup",
    )?;
    let mut future = texelith_io::project_io::parse_json(&valid_manifest).map_err(err)?;
    future["schema_version"] = 2.into();
    let future_bytes = texelith_io::project_io::canonical_json(&future).map_err(err)?;
    std::fs::write(&recovery_paths.backup, &valid_manifest).map_err(err)?;
    std::fs::write(&recovery_path, &future_bytes).map_err(err)?;
    require(
        recover_manifest(&recovery_path).is_err()
            && recovery_paths.backup.exists()
            && std::fs::read(&recovery_path).map_err(err)? == future_bytes,
        "Future manifest silently fell back to an older version",
    )?;
    let odd_assignment =
        texelith_io::project_io::read_json(&root.join("continue/d/assignment.json"))
            .map_err(err)?;
    require(
        odd_assignment["outputs"]
            .as_array()
            .ok_or("Missing odd outputs")?
            .iter()
            .all(|o| o["extent"] == serde_json::json!({"width":odd_width,"height":odd_height})),
        "Keep size did not follow the substituted input extent",
    )?;
    require(
        recovered
            .job
            .items()
            .iter()
            .filter(|i| i.state == BatchState::Committed)
            .count()
            == 3,
        "Recovery lost committed sets",
    )?;
    require(
        recovered
            .job
            .resume_incomplete(recovered.job.id() + 1)
            .map_err(err_debug)?
            .items()
            .len()
            == 1,
        "Recovery requeued committed items",
    )?;
    let before_recovery = gpu.diagnostics().submitted;
    let mut recovery_controls = Controls::new_material();
    let mut recovery_worker = BatchJobs::spawn(gpu.clone())?;
    recovery_worker.recover(root.join("continue.json"))?;
    pump(
        device,
        &mut recovery_worker,
        &mut exports,
        &mut recovery_controls,
        finished,
    )?;
    require(
        count(&recovery_controls, BatchState::Committed) == 3
            && gpu.diagnostics().submitted == before_recovery,
        "Recovery automatically recomputed committed files",
    )?;
    c.queue_batch(
        selection.clone(),
        root.join("stop"),
        root.join("stop.json"),
        false,
    )?;
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        count(&c, BatchState::Committed) == 1
            && count(&c, BatchState::Failed) == 1
            && count(&c, BatchState::Queued) == 2,
        "Default stop policy scheduled later items",
    )?;
    let before = gpu.diagnostics().submitted;
    c.queue_batch(
        selection.clone(),
        root.join("continue"),
        root.join("collision.json"),
        true,
    )?;
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        c.batch_status()
            .error
            .as_ref()
            .is_some_and(|e| e.contains("collision"))
            && gpu.diagnostics().submitted == before,
        "Collision preflight submitted GPU work",
    )?;
    c.queue_batch(
        selection,
        root.join("cancel"),
        root.join("cancel.json"),
        true,
    )?;
    pump(device, &mut batch, &mut exports, &mut c, |c| {
        c.session()
            .batch()
            .is_some_and(|j| j.pending_work().is_some())
    })?;
    c.action(UiAction::Request(CommandRequest::BatchCancel {}));
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        count(&c, BatchState::Cancelled) == 4 && !root.join("cancel/a").exists(),
        "Cancellation published unscheduled work",
    )?;
    c.resume_batch()?;
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        count(&c, BatchState::Committed) == 3 && count(&c, BatchState::Failed) == 1,
        "Explicit resume failed",
    )?;
    std::fs::write(input.join("b-bad.png"), fixture).map_err(err)?;
    c.action(UiAction::Request(CommandRequest::BatchRetryFailed {}));
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        count(&c, BatchState::Committed) == 1 && c.session().batch().unwrap().items().len() == 1,
        "Retry did not isolate failed item",
    )?;
    c.queue_batch(
        InputSelection::Files(vec![input.join("a.png")]),
        root.join("cancel-gpu"),
        root.join("cancel-gpu.json"),
        false,
    )?;
    pump(
        device,
        &mut batch,
        &mut exports,
        &mut c,
        Controls::export_busy,
    )?;
    c.action(UiAction::Request(CommandRequest::BatchCancel {}));
    pump(device, &mut batch, &mut exports, &mut c, finished)?;
    require(
        count(&c, BatchState::Committed) + count(&c, BatchState::Cancelled) == 1,
        "Submitted cancellation lost item outcome",
    )?;
    require(
        root.join("cancel-gpu/a").exists() == (count(&c, BatchState::Committed) == 1),
        "Cancellation disagreed with atomic publication",
    )?;
    // Churn past the resident registry's eight entries, then edit the original
    // interactive document. Batch sources must never evict that document's input.
    for value in 1..=10_u8 {
        let pixels = texelith_gpu::TypedSourcePixels::Rgba8 {
            pixels: [value, 19, 73, 255].repeat(32 * 32).into(),
            interpretation: InputInterpretation::SrgbPhoto,
        };
        let descriptor = pixels
            .descriptor(Extent::new(32, 32).map_err(err)?)
            .map_err(err)?;
        let hash = descriptor.content_hash;
        gpu.install_typed_source(texelith_gpu::TypedSourceUpload { descriptor, pixels })
            .map_err(err)?;
        let started = Instant::now();
        loop {
            device.poll(wgpu::PollType::Poll).map_err(err)?;
            match gpu.try_event() {
                Some(texelith_gpu::WorkerEvent::SourceReady { content_hash })
                    if content_hash == hash =>
                {
                    break;
                }
                Some(texelith_gpu::WorkerEvent::SourceReady { .. }) | None => {}
                Some(event) => return Err(format!("Registry churn failed: {event:?}")),
            }
            if started.elapsed() > Duration::from_secs(30) {
                return Err("Registry churn timed out".into());
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    gpu.request_material(texelith_gpu::MaterialRequest {
        job_id: texelith_gpu::JobId(u64::MAX - 87),
        request_id: 88,
        snapshot: interactive.clone(),
        outputs: interactive_outputs,
        extent: Extent {
            width: size,
            height: size,
        },
        quality: texelith_gpu::Quality::Full,
    })
    .map_err(err)?;
    let started = Instant::now();
    loop {
        device.poll(wgpu::PollType::Poll).map_err(err)?;
        if let Some(frame) = gpu.try_latest_material() {
            require(
                frame.request_id == 88 && frame.generation == interactive.generation,
                "Interactive document lost its source lease",
            )?;
            break;
        }
        if let Some(event) = gpu.try_event() {
            return Err(format!("Interactive source was evicted: {event:?}"));
        }
        if started.elapsed() > Duration::from_secs(30) {
            return Err("Interactive source retention timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    // The legacy planner resolves the resident hash before considering caches,
    // so cached material outputs cannot hide a lost original source here.
    let source = interactive
        .graph
        .material_nodes()
        .iter()
        .find_map(|node| match &node.params {
            MaterialNodeParams::ColorSource(source) => Some(source.clone()),
            _ => None,
        })
        .ok_or("Missing original interactive source")?;
    let graph = Graph::new(
        vec![Node {
            id: NodeId(0),
            input: None,
            params: NodeParams::Source(source.clone()),
        }],
        NodeId(0),
    )
    .map_err(err)?;
    gpu.request(texelith_gpu::RenderRequest {
        request_id: 89,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: interactive.generation,
            graph,
        }),
        output: NodeId(0),
        extent: source.extent,
        quality: texelith_gpu::Quality::Full,
    })
    .map_err(err)?;
    let started = Instant::now();
    loop {
        device.poll(wgpu::PollType::Poll).map_err(err)?;
        if let Some(frame) = gpu.try_latest() {
            require(
                frame.request_id == 89 && frame.generation == interactive.generation,
                "Resident source lookup crossed documents",
            )?;
            break;
        }
        if let Some(event) = gpu.try_event() {
            return Err(format!("Resident source lookup failed: {event:?}"));
        }
        if started.elapsed() > Duration::from_secs(30) {
            return Err("Resident source lookup timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    // Retry and restart reserve their own durable names before scheduling work.
    // Unrelated records at either derived name must survive unchanged.
    for (resume, label, original) in [
        (false, "retry-collision", "continue.json"),
        (true, "resume-collision", "continue.json"),
    ] {
        let base = root.join(format!("{label}.json"));
        let original_bytes = std::fs::read(root.join(original)).map_err(err)?;
        std::fs::write(&base, &original_bytes).map_err(err)?;
        let mut controls = Controls::new_material();
        let mut worker = BatchJobs::spawn(gpu.clone())?;
        worker.recover(base.clone())?;
        pump(device, &mut worker, &mut exports, &mut controls, finished)?;
        require(
            controls.session().batch().is_some(),
            &format!(
                "Collision regression recovery failed: {:?}",
                controls.batch_status().error
            ),
        )?;
        let id = controls.session().batch().unwrap().id() + 1;
        let path = root.join(format!("{label}.job-{id}.json"));
        let sentinel = b"unrelated durable manifest";
        std::fs::write(&path, sentinel).map_err(err)?;
        let submitted = gpu.diagnostics().submitted;
        if resume {
            controls.resume_batch()?;
        } else {
            controls.action(UiAction::Request(CommandRequest::BatchRetryFailed {}));
        }
        pump(device, &mut worker, &mut exports, &mut controls, finished)?;
        // Repeated ticks cannot start a job whose manifest reservation failed.
        for _ in 0..5 {
            worker.tick(&mut controls, &mut exports);
        }
        require(
            controls.batch_status().error.is_some()
                && std::fs::read(&path).map_err(err)? == sentinel
                && std::fs::read(&base).map_err(err)? == original_bytes
                && gpu.diagnostics().submitted == submitted,
            "Retry/resume overwrote a manifest or scheduled unreserved work",
        )?;
    }
    let diagnostics = gpu.diagnostics();
    require(
        diagnostics.validation_errors == 0
            && diagnostics.peak_in_flight <= 1
            && diagnostics.peak_bytes <= 1024 * 1024 * 1024,
        "GPU bounds or validation failed",
    )?;
    Ok(
        serde_json::json!({ "task":"T-087", "scenario":"batch", "passed":true,
        "continue_on_error":true,"stop_on_error":true,"collisions":true,"cancel_retry":true,"immutable_recipe":true,"recovery":true,
        "peak_active":batch.status.peak_active,"peak_prefetch":batch.status.peak_prefetch,"cpu_reserved_bytes":batch.status.cpu_reserved_bytes,
        "cpu_retained_budget_bytes":INTERACTIVE_RESERVATION,"cpu_scratch_peak_bytes":memory::scratch_peak_bytes(),
        "gpu_peak_bytes":diagnostics.peak_bytes,"validation_errors":diagnostics.validation_errors,
        "source_residency":true,"mixed_extents":true,"interactive_gpu":true,"manifest_collisions":true,
        "cpu_measurement":"measured scratch ledger peak plus conservative retained-buffer headroom; process RSS not measured", "native_input":"deferred to orchestrator" }),
    )
}
pub fn run(config: &Config) -> Result<(), String> {
    let result = exercise(config);
    let value = result.as_ref().cloned().unwrap_or_else(
        |e| serde_json::json!({"task":"T-087","scenario":"batch","passed":false,"error":e}),
    );
    let path = config.report.as_ref().ok_or("Missing report")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&value).map_err(err)?).map_err(err)?;
    result.map(|_| ())
}
