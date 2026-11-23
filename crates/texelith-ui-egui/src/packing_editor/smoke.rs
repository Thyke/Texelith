//! Real Session effects, worker persistence and one GPU/file export; no native input claims.
use super::*;
use crate::{export_jobs::ExportJobs, startup::Config};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use texelith_gpu::*;

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn settle(c: &mut Controls) -> Result<(), String> {
    let start = Instant::now();
    while c.preset_io_busy() {
        c.poll_preset_io();
        if start.elapsed() > Duration::from_secs(10) {
            return Err("Preset IO timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    require(!c.packing.load_failed, c.preset_feedback())
}
fn material(source: SourceDescriptor) -> Result<Controls, String> {
    let mut c = Controls::new_material();
    c.open_image("packing-fixture.png".into());
    let Some(Effect::LoadImage { token, .. }) = c.take_effects().into_iter().next() else {
        return Err("Missing import effect".into());
    };
    c.complete_import(token, source)?;
    Ok(c)
}
pub fn exercise(config: &Config) -> Result<serde_json::Value, String> {
    let root = config
        .report
        .as_ref()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."))
        .join(format!(
            "packing-{}-{}",
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
            config
                .adapter_name
                .as_ref()
                .is_none_or(|n| a.get_info().name.to_lowercase().contains(&n.to_lowercase()))
        })
        .ok_or("Requested GPU adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let (device, queue) =
        crate::material_ui::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("T-083 packing"),
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
    let result = exercise_on(config, &root, &device, gpu.clone());
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
    result.map(|mut r| {
        r["adapter"] = info.name.into();
        r["backend"] = format!("{:?}", info.backend).into();
        r["artifacts"] = root.to_string_lossy().into_owned().into();
        r
    })
}
fn exercise_on(
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
    let source = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    gpu.install_typed_source(TypedSourceUpload {
        descriptor: source.clone(),
        pixels,
    })
    .map_err(|e| e.to_string())?;
    let upload_started = Instant::now();
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        match gpu.try_event() {
            Some(WorkerEvent::SourceReady { content_hash })
                if content_hash == source.content_hash =>
            {
                break;
            }
            Some(event) => return Err(format!("Source upload failed: {event:?}")),
            None => {}
        }
        if upload_started.elapsed() > Duration::from_secs(30) {
            return Err("Source upload timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let mut c = material(source.clone())?;
    c.open_packing_editor()?;
    c.packing_preset_mut().unwrap().outputs.truncate(1);
    let maps: Vec<_> = c
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_outputs()
        .unwrap()
        .keys()
        .copied()
        .collect();
    for map in maps {
        let output = &mut c.packing_preset_mut().unwrap().outputs[0];
        output.import_hints = None;
        output.channels = vec![
            PresetChannelSpec {
                source: PresetChannelSource::Map {
                    map,
                    component: Component::R,
                    missing: MissingMap::Error
                },
                invert: true,
                transfer: Transfer::Linear
            };
            4
        ];
        c.request_material_export()?;
        let effects = c.take_effects();
        let [Effect::Export { plan, capture }] = effects.as_slice() else {
            return Err("Missing routed export effect".into());
        };
        for ch in &plan.outputs[0].channels {
            require(
                ch.invert
                    && ch.source
                        == ChannelSource::Map(
                            capture.graph.material_outputs().unwrap()[&map],
                            Component::R,
                        ),
                "Map/channel routing mismatch",
            )?;
        }
    }
    let output = &mut c.packing_preset_mut().unwrap().outputs[0];
    output.channels[0] = PresetChannelSpec {
        source: PresetChannelSource::Map {
            map: MapKind::Albedo,
            component: Component::R,
            missing: MissingMap::Error,
        },
        invert: false,
        transfer: Transfer::Srgb,
    };
    output.channels[3] = PresetChannelSpec {
        source: PresetChannelSource::Constant(0.25),
        invert: true,
        transfer: Transfer::Linear,
    };
    require(
        c.packing_plan()?.outputs[0].semantics().warnings == vec![ExportWarning::MixedTransfer],
        "Mixed-transfer warning missing",
    )?;
    for invalid in 0..2 {
        c.export_options_mut().size = if invalid == 0 {
            SizePolicy::Resize {
                width: 0,
                height: 64,
            }
        } else {
            SizePolicy::Keep
        };
        c.export_options_mut().material_name = if invalid == 1 { "../bad" } else { "Stone" }.into();
        require(
            c.request_material_export().is_err() && c.take_effects().is_empty(),
            "Invalid export was accepted",
        )?;
    }
    c.export_options_mut().size = SizePolicy::Keep;
    c.export_options_mut().material_name = "Stone".into();
    let recipe = c.packing_preset_mut().unwrap();
    recipe.naming = "T_{name}_{map}_{index}".into();
    let mut second = recipe.outputs[0].clone();
    second.name = "alternate".into();
    recipe.outputs.push(second);
    c.start_preset_io(Some(root.join("presets.json")))?;
    settle(&mut c)?;
    c.save_packing_preset("Stone")?;
    settle(&mut c)?;
    let mut reopened = material(source)?;
    reopened.start_preset_io(Some(root.join("presets.json")))?;
    settle(&mut reopened)?;
    reopened.select_export_preset("custom-Stone")?;
    reopened.export_options_mut().material_name = "Stone".into();
    reopened.export_options_mut().destination = root.join("export");
    let preview = reopened.packing_names()?;
    reopened.request_material_export()?;
    let effect = reopened
        .take_effects()
        .into_iter()
        .next()
        .ok_or("Missing reloaded export effect")?;
    let mut jobs = ExportJobs::spawn(gpu.clone())?;
    jobs.start(effect, reopened.export_options().clone())?;
    let start = Instant::now();
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        jobs.tick(&mut reopened);
        if !jobs.busy() {
            break;
        }
        if start.elapsed() > Duration::from_secs(90) {
            return Err("Packing export timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    require(
        jobs.status()
            .is_some_and(|s| s.stage == ExportStage::Completed),
        "Packed export failed",
    )?;
    for name in &preview {
        require(
            root.join("export").join(name).is_file(),
            "Naming preview differs from published filename",
        )?;
    }
    let encoded =
        std::fs::read(root.join("export").join(&preview[0])).map_err(|e| e.to_string())?;
    let decoded = texelith_io::decode_bytes(&encoded, texelith_io::ColorIntent::LinearHeight)
        .map_err(|e| e.to_string())?;
    require(
        decoded.rgba8.as_chunks::<4>().0.iter().all(|p| p[3] == 191),
        "Constant inversion did not reach file alpha",
    )?;
    Ok(
        serde_json::json!({"task":"T-083", "scenario":"packing_ui", "passed":true,
        "every_map_rgba":true, "constants_invert":true, "preset_save_reload":true,
        "mixed_transfer_warning":true, "invalid_size_name_blocked":true, "naming_matches_files":true,
        "native_input":"not measured", "size":config.size}),
    )
}
pub fn run(config: &Config) -> Result<(), String> {
    let result = exercise(config);
    let value = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            serde_json::json!({"task":"T-083", "scenario":"packing_ui", "passed":false, "error":e})
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
