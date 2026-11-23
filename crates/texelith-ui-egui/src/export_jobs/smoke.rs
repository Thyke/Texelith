//! Headless real-command/export gate. Native input is measured separately.
use super::*;
use crate::{command_adapter::UiAction, controls::Controls, startup::Config};
use std::{
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use texelith_gpu::*;

fn require(ok: bool, message: impl Into<String>) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn wait<T>(
    device: &wgpu::Device,
    mut next: impl FnMut() -> Result<Option<T>, String>,
) -> Result<T, String> {
    let start = Instant::now();
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        if let Some(result) = next()? {
            return Ok(result);
        }
        if start.elapsed() > Duration::from_secs(90) {
            return Err("Export diagnostic timed out".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn edit(c: &mut Controls, map: MapKind, parameter: ParameterId, value: ParameterValue) {
    let snapshot = c.session().snapshot().unwrap();
    let mut node = snapshot.graph.material_outputs().unwrap()[&map].node;
    // Photo height now ends in an explicit remap; edit its parameter-owning ancestor.
    while snapshot
        .graph
        .material_node(node)
        .unwrap()
        .params
        .parameter(parameter)
        .is_err()
    {
        node = snapshot.graph.material_node(node).unwrap().inputs[0]
            .source
            .node;
    }
    c.action(UiAction::Change {
        address: ParameterAddress { node, parameter },
        value,
    });
}
fn roughness(c: &mut Controls, value: f32) {
    edit(
        c,
        MapKind::Roughness,
        ParameterId::Curve,
        ParameterValue::Curve(vec![[0.0, value], [1.0, value]]),
    );
}
fn cancel_availability(c: &Controls, expected: bool) -> Result<(), String> {
    require(
        c.session()
            .availability()
            .iter()
            .any(|a| a.id == CommandId::CancelExport && a.enabled == expected),
        format!("CancelExport availability must be {expected}"),
    )
}
fn finish(
    jobs: &mut ExportJobs,
    c: &mut Controls,
    device: &wgpu::Device,
) -> Result<JobStatus, String> {
    wait(device, || {
        jobs.tick(c);
        cancel_availability(c, c.export_busy())?;
        if !c.export_busy() {
            Ok(c.export_status().cloned())
        } else {
            Ok(None)
        }
    })
}

pub fn exercise(config: &Config) -> Result<serde_json::Value, String> {
    let report = config.report.as_ref().ok_or("Missing report")?;
    let parent = report
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let root = parent.join(format!(
        "exports-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
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
            label: Some("T-082 export diagnostic"),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 4096,
                ..Default::default()
            },
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let gpu = Arc::new(
        GpuWorker::spawn(device.clone(), Arc::new(queue), Default::default())
            .map_err(|e| e.to_string())?,
    );
    let result = exercise_jobs(config, &root, &device, gpu.clone());
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
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
    device: &wgpu::Device,
    gpu: Arc<GpuWorker>,
) -> Result<serde_json::Value, String> {
    let extent = Extent::new(config.size, config.size).map_err(|e| e.to_string())?;
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: [128, 64, 32, 255]
            .repeat(config.size as usize * config.size as usize)
            .into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let fixture = texelith_io::encoders::encode_png8(
        config.size,
        config.size,
        texelith_io::encoders::PngChannels::Rgb,
        texelith_io::encoders::PngColorIntent::Srgb,
        &[128, 64, 32].repeat(config.size as usize * config.size as usize),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(root.join("source.png"), fixture).map_err(|e| e.to_string())?;
    let source = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    gpu.install_typed_source(TypedSourceUpload {
        descriptor: source.clone(),
        pixels,
    })
    .map_err(|e| e.to_string())?;
    wait(device, || match gpu.try_event() {
        Some(WorkerEvent::SourceReady { content_hash }) if content_hash == source.content_hash => {
            Ok(Some(()))
        }
        Some(e) => Err(format!("Source upload: {e:?}")),
        None => Ok(None),
    })?;
    let mut c = Controls::new_material();
    cancel_availability(&c, false)?;
    c.open_image("generated-export-fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        return Err("Missing import effect".into());
    };
    c.complete_import(token, source)?;
    edit(
        &mut c,
        MapKind::Height,
        ParameterId::CoarseWeight,
        ParameterValue::Float(0.0),
    );
    edit(
        &mut c,
        MapKind::Metallic,
        ParameterId::MaskMode,
        ParameterValue::MaskMode(MaskMode::Subtract),
    );
    let mut jobs = ExportJobs::spawn(gpu.clone())?;
    cancel_availability(&c, false)?;
    let mut presets = Vec::new();
    let mut one_job_bound = false;
    for preset in BuiltinExportPresets.presets() {
        roughness(&mut c, 0.25);
        c.export_options_mut().preset = preset.profile.id.clone();
        c.export_options_mut().destination = root.join(&preset.profile.id);
        c.request_material_export()?;
        let capture = c.session().snapshot().unwrap();
        let native_options = c.export_options().clone();
        c.export_options_mut().destination = root.join("edited-after-request");
        jobs.tick(&mut c);
        *c.export_options_mut() = native_options.clone();
        require(
            jobs.status().unwrap().destination == native_options.destination,
            "Destination changed after request capture",
        )?;
        require(jobs.busy(), "Export must be accepted before completion")?;
        cancel_availability(&c, true)?;
        require(
            jobs.status().unwrap().stage != ExportStage::Completed,
            "Premature completion",
        )?;
        c.action(UiAction::Request(CommandRequest::RequestExport {}));
        let duplicate = c.take_effects().remove(0);
        one_job_bound |= jobs.start(duplicate, c.export_options().clone()).is_err();
        roughness(&mut c, 0.75);
        require(
            c.session().snapshot().unwrap().generation != capture.generation,
            "Live edit did not apply",
        )?;
        let live = c.session().snapshot().unwrap();
        let request = crate::material_ui::request(
            live.clone(),
            MapKind::Normal,
            live.generation + 1000,
            Quality::Full,
        )?;
        gpu.request_material(request).map_err(|e| e.to_string())?;
        let bundle = wait(device, || {
            jobs.tick(&mut c);
            Ok(gpu.try_latest_material())
        })?;
        require(
            bundle.generation == live.generation,
            "Interactive request did not reflect live edits",
        )?;
        drop(bundle);
        let status = finish(&mut jobs, &mut c, device)?;
        cancel_availability(&c, false)?;
        require(
            status.stage == ExportStage::Completed,
            format!("{}: {:?}", preset.profile.id, status),
        )?;
        require(
            !status.destination.join(".texelith-transaction").exists(),
            "Successful export retained its transaction marker",
        )?;
        let manifest = verify_set(&status.destination, config.size)?;
        require(
            manifest["generation"] == capture.generation,
            "Export used the edited generation",
        )?;
        presets.push(serde_json::json!({ "preset": preset.profile.id, "outputs": manifest["outputs"], "passed": true }));
    }
    require(one_job_bound, "Second export was admitted")?;
    // Exercise the same parent-folder selection path as the native picker,
    // then complete actual GPU readback/encoding/publication for each folder.
    let folder_root = root.join("folder-regressions");
    let parents = [
        folder_root.join("folder with spaces"),
        folder_root.join("ç ğ ı ö ş ü İ"),
        folder_root.join("existing-non-empty"),
        folder_root.join("OneDrive - Texture Studio/Documents/Materials/Client Projects/Stone surfaces/Final approved textures/Exports"),
        folder_root.join(format!("trailing{}", std::path::MAIN_SEPARATOR)),
    ];
    let mut folder_paths = Vec::new();
    roughness(&mut c, 0.25);
    c.select_export_preset("unreal-v1")?;
    for parent in parents {
        std::fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
        let sentinel = parent.join("existing-user-file.txt");
        std::fs::write(&sentinel, b"keep this file").map_err(|e| e.to_string())?;
        c.select_export_folder(parent.clone());
        c.request_material_export()?;
        jobs.tick(&mut c);
        let status = finish(&mut jobs, &mut c, device)?;
        require(
            status.stage == ExportStage::Completed,
            format!("{status:?}"),
        )?;
        verify_set(&status.destination, config.size)?;
        require(
            std::fs::read(&sentinel).map_err(|e| e.to_string())? == b"keep this file",
            "Picked parent lost existing user data",
        )?;
        folder_paths.push(parent.display().to_string());
    }
    // Collision must fail without touching even one byte of an existing set.
    let existing = root.join("unreal-v1");
    let before = snapshot_files(&existing)?;
    c.export_options_mut().destination = existing.clone();
    c.request_material_export()?;
    jobs.tick(&mut c);
    let collision = finish(&mut jobs, &mut c, device)?;
    cancel_availability(&c, false)?;
    require(
        collision.stage == ExportStage::Failed,
        "Existing destination must fail",
    )?;
    require(
        snapshot_files(&existing)? == before,
        "Collision modified destination",
    )?;
    // Explicitly confirmed replacement uses the same transaction protocol and
    // retires both the backup and recovery marker before reporting success.
    let replacement_destination = root.join("confirmed-replacement");
    std::fs::create_dir(&replacement_destination).map_err(|e| e.to_string())?;
    std::fs::write(replacement_destination.join("old-set-sentinel"), b"old")
        .map_err(|e| e.to_string())?;
    c.export_options_mut().destination = replacement_destination.clone();
    c.export_options_mut().replace_existing = true;
    c.request_material_export()?;
    c.export_options_mut().replace_existing = false;
    jobs.tick(&mut c);
    let replaced = finish(&mut jobs, &mut c, device)?;
    require(
        replaced.stage == ExportStage::Completed,
        format!("Confirmed replacement failed: {replaced:?}"),
    )?;
    require(
        !replacement_destination
            .join(".texelith-transaction")
            .exists()
            && !replacement_destination.join("old-set-sentinel").exists(),
        "Replacement retained its recovery marker",
    )?;
    for entry in std::fs::read_dir(root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        require(
            !entry
                .file_name()
                .to_string_lossy()
                .starts_with(".texelith-"),
            "Successful exports retained staging or backup directories",
        )?;
    }
    // Cancel an admitted job after its first output; incomplete files stay staged.
    c.export_options_mut().preset = "unreal-v1".into();
    c.export_options_mut().destination = root.join("cancelled-set");
    c.request_material_export()?;
    jobs.tick(&mut c);
    wait(device, || {
        jobs.tick(&mut c);
        let status = jobs.status().ok_or("Missing cancel progress")?;
        require(
            !status.terminal(),
            "Export completed before cancellation probe",
        )?;
        Ok((status.completed_items >= 1).then_some(()))
    })?;
    c.action(UiAction::Request(CommandRequest::CancelExport {
        job_id: jobs.status().unwrap().job_id + 1,
    }));
    jobs.tick(&mut c);
    require(jobs.busy(), "Stale cancellation targeted the active job")?;
    c.cancel_material_export();
    jobs.tick(&mut c);
    let cancelled = finish(&mut jobs, &mut c, device)?;
    cancel_availability(&c, false)?;
    require(
        cancelled.stage == ExportStage::Cancelled,
        format!("Cancel failed: {cancelled:?}"),
    )?;
    require(
        !cancelled.destination.exists(),
        "Cancellation published a destination",
    )?;
    require(
        snapshot_files(&existing)? == before,
        "Cancellation modified an existing set",
    )?;
    // Invalid native naming fails before GPU admission and retains output identity.
    c.export_options_mut().destination = root.join("invalid-naming");
    c.export_options_mut().material_name = "../invalid".into();
    // Exercise the worker guard directly; the packing editor now rejects this
    // native name before requesting an export through its UI entry point.
    c.action(UiAction::Request(CommandRequest::RequestExport {}));
    jobs.tick(&mut c);
    let invalid = finish(&mut jobs, &mut c, device)?;
    require(
        invalid.stage == ExportStage::Failed
            && !invalid.output_name.is_empty()
            && invalid
                .error
                .as_ref()
                .is_some_and(|e| e.contains(&invalid.output_name)),
        "IO error lost output identity",
    )?;
    require(
        !invalid.destination.exists(),
        "Invalid naming published a set",
    )?;
    c.export_options_mut().material_name = "Material".into();
    // Float displacement uses the same route and preserves negative values.
    roughness(&mut c, 0.25);
    c.export_options_mut().preset = "blender-v1".into();
    c.export_options_mut().format = FormatChoice::FloatDisplacement;
    c.export_options_mut().destination = root.join("float-displacement");
    // Flat photo height is now .5. Keep the negative-displacement fixture at
    // -.5 by using midlevel 1, preserving the signed EXR assertion below.
    let mut float_preset = BuiltinExportPresets.find("blender-v1", 1).unwrap().clone();
    float_preset.displacement.midlevel = 1.0;
    c.packing.draft = Some(float_preset);
    c.request_material_export()?;
    jobs.tick(&mut c);
    let float = finish(&mut jobs, &mut c, device)?;
    require(
        float.stage == ExportStage::Completed,
        format!("Float export failed: {float:?}"),
    )?;
    verify_set(&float.destination, config.size)?;
    for entry in std::fs::read_dir(root).map_err(|e| e.to_string())? {
        require(
            !entry
                .map_err(|e| e.to_string())?
                .file_name()
                .to_string_lossy()
                .starts_with(".texelith-stage"),
            "Leaked cancelled staging directory",
        )?;
    }
    let diagnostics = gpu.diagnostics();
    require(diagnostics.validation_errors == 0, "GPU validation errors")?;
    require(
        diagnostics.peak_in_flight <= 1 && diagnostics.peak_pending <= 1,
        "Unbounded GPU work",
    )?;
    Ok(
        serde_json::json!({ "task": "T-082", "scenario": "export", "passed": true, "presets": presets,
        "immutable_edits": true, "cancel_preserves_destination": true, "collision_preserves_destination": true,
        "one_job_bound": one_job_bound, "error_identity": true, "float_displacement": true, "publication_only_completion": true,
        "cancel_availability": true, "folder_paths": folder_paths,
        "validation_errors": diagnostics.validation_errors, "peak_bytes": diagnostics.peak_bytes,
        "native_input": "not measured; deferred to orchestrator", "fixture": "constant sRGB photo, roughness .25, normalized height .5, float displacement midlevel 1, independent file-value checks" }),
    )
}

fn snapshot_files(directory: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        files.push((
            entry.file_name().to_string_lossy().into_owned(),
            std::fs::read(entry.path()).map_err(|e| e.to_string())?,
        ));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn expected(channel: &ChannelSpec) -> f32 {
    let mut value = match channel.source {
        ChannelSource::Constant(v) => v,
        ChannelSource::Map(r, c) => match r.port {
            OutputPort::Albedo => {
                let v = match c {
                    Component::R => 128.0,
                    Component::G => 64.0,
                    Component::B => 32.0,
                    Component::A => 255.0,
                } / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055_f32).powf(2.4)
                }
            }
            OutputPort::Normal => {
                if c == Component::B {
                    1.0
                } else {
                    0.5
                }
            }
            OutputPort::Ao => 1.0,
            OutputPort::Roughness => 0.25,
            OutputPort::Height => 0.5,
            _ => 0.0,
        },
    };
    if channel.invert {
        value = 1.0 - value;
    }
    if channel.transfer == Transfer::Srgb {
        value = if value <= 0.0031308 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
    }
    value
}
fn verify_set(destination: &Path, size: u32) -> Result<serde_json::Value, String> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(destination.join("assignment.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let plan: ExportPlan =
        serde_json::from_value(manifest["plan"].clone()).map_err(|e| e.to_string())?;
    let outputs = manifest["outputs"]
        .as_array()
        .ok_or("Missing output records")?;
    require(outputs.len() == plan.outputs.len(), "Output count mismatch")?;
    for (output, record) in plan.outputs.iter().zip(outputs) {
        let bytes =
            std::fs::read(destination.join(record["file"].as_str().ok_or("Missing filename")?))
                .map_err(|e| e.to_string())?;
        require(
            record["extent"] == serde_json::json!({"width":size,"height":size}),
            "Manifest size mismatch",
        )?;
        require(
            record["channels"] == output.channels.len()
                && record["semantics"]
                    == serde_json::to_value(output.semantics()).map_err(|e| e.to_string())?,
            "Numeric output metadata mismatch",
        )?;
        match output.format {
            ExportFormat::Png8 => {
                require(
                    bytes[24] == 8
                        && bytes[25]
                            == match output.channels.len() {
                                1 => 0,
                                3 => 2,
                                4 => 6,
                                _ => 255,
                            },
                    "PNG precision/channels mismatch",
                )?;
                let decoded =
                    texelith_io::decode_bytes(&bytes, texelith_io::ColorIntent::PhotoSrgb)
                        .map_err(|e| e.to_string())?;
                require(
                    decoded.width == size && decoded.height == size,
                    "PNG dimensions mismatch",
                )?;
                for pixel in decoded.rgba8.as_chunks::<4>().0 {
                    for (i, channel) in output.channels.iter().enumerate() {
                        if (i32::from(pixel[i]) - (expected(channel) * 255.0).round() as i32).abs()
                            > 1
                        {
                            return Err(format!(
                                "{} channel {i}: {} != {}",
                                output.name,
                                pixel[i],
                                expected(channel)
                            ));
                        }
                    }
                }
                require(
                    has_srgb(&bytes) == (output.semantics().color_tag == ExportColorTag::Srgb),
                    "PNG transfer tag mismatch",
                )?;
            }
            ExportFormat::Png16 => {
                let texelith_io::TypedImage::HeightU16 {
                    width,
                    height,
                    values,
                } = texelith_io::decode_typed_bytes(&bytes, texelith_io::TypedIntent::Height)
                    .map_err(|e| e.to_string())?
                else {
                    return Err("Height decode type mismatch".into());
                };
                require(
                    width == size
                        && height == size
                        && values.iter().all(|v| (i32::from(*v) - 32768).abs() <= 1),
                    "PNG16 normalized height values mismatch",
                )?;
            }
            ExportFormat::Exr32 => {
                let texelith_io::TypedImage::HeightF32 {
                    width,
                    height,
                    values,
                } = texelith_io::decode_typed_bytes(&bytes, texelith_io::TypedIntent::Height)
                    .map_err(|e| e.to_string())?
                else {
                    return Err("EXR decode type mismatch".into());
                };
                require(
                    width == size
                        && height == size
                        && values
                            .iter()
                            .all(|v| v.is_finite() && (*v + 0.5).abs() < 1e-6),
                    "EXR signed displacement mismatch",
                )?;
            }
            ExportFormat::DdsBc5 => {
                let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
                require(
                    &bytes[..4] == b"DDS "
                        && u32_at(12) == size
                        && u32_at(16) == size
                        && u32_at(128) == 83,
                    "DDS header mismatch",
                )?;
                let mut offset = 148;
                let mut extent = size;
                for _ in 0..u32_at(28) {
                    let len = extent.div_ceil(4).pow(2) as usize * 16;
                    let blocks = bytes.get(offset..offset + len).ok_or("Missing DDS mip")?;
                    for block in blocks.as_chunks::<8>().0 {
                        let a = f32::from(block[0]);
                        let b = f32::from(block[1]);
                        let mut palette = [a, b, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0];
                        let n = if a > b { 7 } else { 5 };
                        for i in 1..n {
                            palette[i + 1] = ((n - i) as f32 * a + i as f32 * b) / n as f32;
                        }
                        let bits = block[2..8]
                            .iter()
                            .enumerate()
                            .fold(0u64, |v, (i, b)| v | (u64::from(*b) << (i * 8)));
                        for i in 0..16 {
                            require(
                                (palette[((bits >> (i * 3)) & 7) as usize] - 128.0).abs() <= 1.0,
                                "DDS normal channel mismatch",
                            )?;
                        }
                    }
                    offset += len;
                    extent = (extent / 2).max(1);
                }
                require(
                    offset == bytes.len() && extent == 1,
                    "DDS mip chain length mismatch",
                )?;
            }
        }
    }
    Ok(manifest)
}
fn has_srgb(bytes: &[u8]) -> bool {
    let mut offset = 8;
    while offset + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        if &bytes[offset + 4..offset + 8] == b"sRGB" {
            return true;
        }
        offset += len + 12;
    }
    false
}
pub fn run(config: &Config) -> Result<(), String> {
    let result = exercise(config);
    let value = match &result {
        Ok(value) => value.clone(),
        Err(e) => {
            serde_json::json!({"task":"T-082", "scenario":"export", "passed":false, "error":e})
        }
    };
    let path = config.report.as_ref().ok_or("Missing report")?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    result.map(|_| ())
}
