//! Real shared-device inspector gate. Native pointer/focus responsiveness remains
//! a separate orchestrator check; these assertions exercise production routes.
use crate::{
    command_adapter::UiAction,
    controls::Controls,
    material_ui::{MAPS, MaterialIntegration, block_on},
    startup::Config,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;
fn dispatch(c: &mut Controls, request: CommandRequest) {
    c.action(UiAction::Request(request));
}
fn wait(
    runtime: &mut MaterialIntegration,
    c: &mut Controls,
    device: &wgpu::Device,
    mut done: impl FnMut(&MaterialIntegration, &Controls) -> bool,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        runtime.tick(c);
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        if let Some(error) = &runtime.error {
            return Err(error.clone());
        }
        if done(runtime, c) {
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(90) {
            return Err(format!(
                "Inspector diagnostic timed out: {}",
                runtime.status
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn select(
    runtime: &mut MaterialIntegration,
    c: &mut Controls,
    device: &wgpu::Device,
    map: MapKind,
) -> Result<(), String> {
    let before = runtime.publications;
    dispatch(c, CommandRequest::SelectMap { map });
    wait(runtime, c, device, |r, _| r.publications > before)
}
fn parameter(c: &Controls, id: ParameterId) -> Result<ParameterAddress, String> {
    let s = c.session().snapshot().ok_or("No material")?;
    Ok(ParameterAddress {
        node: s
            .graph
            .material_nodes()
            .iter()
            .find(|n| n.params.parameter(id).is_ok())
            .ok_or("Parameter unavailable")?
            .id,
        parameter: id,
    })
}
fn value(c: &Controls, address: ParameterAddress) -> ParameterValue {
    c.session()
        .snapshot()
        .unwrap()
        .graph
        .material_node(address.node)
        .unwrap()
        .params
        .parameter(address.parameter)
        .unwrap()
}
pub fn run(config: &Config) -> Result<(), String> {
    let report = config.report.as_ref().ok_or("missing report")?;
    let parent = report
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = config.backends()?;
    let instance = wgpu::Instance::new(desc);
    let adapter = block_on(instance.enumerate_adapters(config.backends()?))
        .into_iter()
        .find(|a| {
            config
                .adapter_name
                .as_ref()
                .is_none_or(|n| a.get_info().name.to_lowercase().contains(&n.to_lowercase()))
        })
        .ok_or("Requested native GPU adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("T-072 inspectors"),
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let gpu = Arc::new(
        GpuWorker::spawn(device.clone(), queue.clone(), Default::default())
            .map_err(|e| e.to_string())?,
    );
    // Presentation and picker readbacks are serialized and reserved in this ledger.
    let mut runtime = MaterialIntegration::new(gpu.clone(), device.clone(), queue)?;
    let mut c = Controls::new_material();
    let mut checks = Vec::new();
    let result = (|| {
        let size = config.size;
        let mut pixels = Vec::with_capacity((size * size * 3) as usize);
        for y in 0..size {
            for x in 0..size {
                pixels.extend_from_slice(&[
                    (32 + (x * 191 / size)) as u8,
                    (24 + (y * 199 / size)) as u8,
                    (60 + ((x / 16 + y / 16) % 2) * 120) as u8,
                ]);
            }
        }
        let fixture = parent.join("inspector-source.png");
        let bytes = texelith_io::encoders::encode_png8(
            size,
            size,
            texelith_io::encoders::PngChannels::Rgb,
            texelith_io::encoders::PngColorIntent::Srgb,
            &pixels,
        )
        .map_err(|e| e.to_string())?;
        texelith_io::publish_png(&fixture, &bytes).map_err(|e| e.to_string())?;
        c.open_image(fixture);
        wait(&mut runtime, &mut c, &device, |r, c| {
            r.publications > 0 && c.session().snapshot().is_some()
        })?;
        checks.push("photo-command-effect-worker-import".to_owned());
        let selected = c.session().snapshot().unwrap().graph.selected_output_ref();
        wait(&mut runtime, &mut c, &device, |_, c| {
            c.map_thumbnails_current()
        })?;
        if c.session().snapshot().unwrap().graph.selected_output_ref() != selected {
            return Err("Thumbnail population changed map selection".into());
        }
        checks.push("seven-background-thumbnails-without-selection".into());
        for map in MAPS {
            select(&mut runtime, &mut c, &device, map)?;
            checks.push(format!("{map:?}-real-gpu-preview"));
        }
        wait(&mut runtime, &mut c, &device, |_, c| {
            MAPS.into_iter()
                .all(|m| c.thumbnail_generation(m) == c.session().snapshot().map(|s| s.generation))
        })?;
        if !MAPS.into_iter().all(|m| {
            c.map_thumbnail_size(m)
                .is_some_and(|s| s[0] <= 128 && s[1] <= 128)
        }) {
            return Err("Thumbnail bounds".into());
        }
        checks.push("seven-current-generation-cached-thumbnails".into());
        select(&mut runtime, &mut c, &device, MapKind::Metallic)?;
        let color = parameter(&c, ParameterId::PickedColor)?;
        let request = c.request_color_pick(color, [0.123, 0.234])?;
        wait(&mut runtime, &mut c, &device, |_, c| {
            c.inspectors.pending.is_none()
        })?;
        let x = (request.uv[0] * size as f32).floor() as usize;
        let y = (request.uv[1] * size as f32).floor() as usize;
        let expected = std::array::from_fn::<_, 3, _>(|i| {
            let v = pixels[(y * size as usize + x) * 3 + i] as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        });
        let ParameterValue::Color(picked) = value(&c, color) else {
            return Err("color type".into());
        };
        if picked
            .iter()
            .zip(expected)
            .any(|(a, b)| (*a - b).abs() > 0.001)
        {
            return Err(format!(
                "Original linear source pick mismatch {picked:?} versus {expected:?}"
            ));
        }
        checks.push("original-linear-source-one-pixel-gpu-readback".into());
        let stale = c.request_color_pick(color, [0.7, 0.8])?;
        runtime.tick(&mut c);
        let current = c.request_color_pick(color, [0.2, 0.3])?;
        if c.complete_color_pick(&stale, [1.0; 3]) {
            return Err("Stale picker applied".into());
        }
        wait(&mut runtime, &mut c, &device, |_, c| {
            c.inspectors.pending.is_none()
        })?;
        if c.complete_color_pick(&current, [1.0; 3]) {
            return Err("Completed picker applied twice".into());
        }
        checks.push("out-of-order-picker-token-rejected".into());
        let address = parameter(&c, ParameterId::Strength)?;
        let before = c.session().history().undo_entries;
        c.action(UiAction::DragBegin { address });
        for v in [1.2, 1.6, 2.0] {
            c.action(UiAction::Change {
                address,
                value: ParameterValue::Float(v),
            });
        }
        c.action(UiAction::FocusLost);
        if c.session().history().undo_entries != before + 1 {
            return Err("Slider history not coalesced".into());
        }
        checks.push("slider-focus-loss-one-history-entry".into());
        let address = parameter(&c, ParameterId::Curve)?;
        let original = value(&c, address);
        c.action(UiAction::DragBegin { address });
        c.action(UiAction::Change {
            address,
            value: ParameterValue::Curve(vec![[0.0, 0.2], [0.4, 0.7], [1.0, 0.9]]),
        });
        c.action(UiAction::Escape);
        if value(&c, address) != original {
            return Err("Curve Escape did not restore original".into());
        }
        checks.push("typed-curve-edit-cancel".into());
        dispatch(
            &mut c,
            CommandRequest::ConfigureMaterial {
                seamless: true,
                painted_mask: true,
            },
        );
        select(&mut runtime, &mut c, &device, MapKind::Mask)?;
        c.set_brush_enabled(true);
        let before = c.session().history().undo_entries;
        c.paint_at([0.2, 0.3], super::PaintPhase::Begin);
        c.paint_at([0.6, 0.7], super::PaintPhase::Append);
        c.action(UiAction::FocusLost);
        if c.session().history().undo_entries != before + 1 {
            return Err("Stroke history".into());
        }
        let previous = runtime.publications;
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.publications > previous
        })?;
        if !runtime.last_nonconstant {
            return Err("Stroke did not reach the GPU mask kernel".into());
        }
        checks.push("pointer-stroke-gpu-mask-focus-commit-one-history-entry".into());
        c.paint_at([0.3, 0.4], super::PaintPhase::Begin);
        c.action(UiAction::Escape);
        if c.session().history().undo_entries != before + 1
            || c.session().history().active_stroke.is_some()
        {
            return Err("Escape stroke contract".into());
        }
        checks.push("stroke-escape-cancel".into());
        c.action(UiAction::UndoShortcut);
        let previous = runtime.publications;
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.publications > previous
        })?;
        if runtime.last_nonconstant {
            return Err("Undo retained painted mask".into());
        }
        c.action(UiAction::RedoShortcut);
        let previous = runtime.publications;
        wait(&mut runtime, &mut c, &device, |r, _| {
            r.publications > previous
        })?;
        if !runtime.last_nonconstant {
            return Err("Redo lost painted mask".into());
        }
        checks.push("stroke-gpu-undo-redo".into());
        dispatch(
            &mut c,
            CommandRequest::SetLocked {
                map: MapKind::Mask,
                locked: true,
            },
        );
        c.paint_at([0.2, 0.3], super::PaintPhase::Begin);
        if c.session().history().active_stroke.is_some() {
            return Err("Locked map painted".into());
        }
        checks.push("locked-map-rejects-paint".into());
        Ok(())
    })();
    let diagnostics = gpu.diagnostics();
    let passed = result.is_ok() && diagnostics.validation_errors == 0 && checks.len() >= 17;
    let report_value = serde_json::json!({"task":"T-072","scenario":"inspectors","passed":passed,"assertions":checks.len(),"checks":checks,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"size":config.size,"error":result.as_ref().err(),"gpu_validation_errors":diagnostics.validation_errors,"peak_bytes":diagnostics.peak_bytes,"picker_staging_bytes":256,"native_input":"deferred to orchestrator","poisson_residual":"not supplied by current CompletedMaterial API"});
    std::fs::write(
        report,
        serde_json::to_vec_pretty(&report_value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    drop(runtime);
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
    if passed {
        Ok(())
    } else {
        Err(result
            .err()
            .unwrap_or_else(|| "Incomplete assertions or GPU validation errors".into()))
    }
}
