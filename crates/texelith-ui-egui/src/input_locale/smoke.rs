//! Real IO/GPU command-effect diagnostic; no foreground window is required.
use crate::{
    controls::Controls,
    material_ui::{MaterialIntegration, block_on},
    startup::Config,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::GpuWorker;

fn wait(
    runtime: &mut MaterialIntegration,
    c: &mut Controls,
    device: &wgpu::Device,
    mut done: impl FnMut(&MaterialIntegration, &Controls) -> bool,
) -> Result<(), String> {
    let started = Instant::now();
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
        if started.elapsed() > Duration::from_secs(90) {
            return Err(format!("Input diagnostic timed out: {}", runtime.status));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

pub fn run(config: &Config) -> Result<(), String> {
    let report = config.report.as_ref().ok_or("missing report")?;
    let parent = report
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = config.backends()?;
    let instance = wgpu::Instance::new(descriptor);
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
        label: Some("T-073 input diagnostic"),
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
    let mut runtime = MaterialIntegration::new(gpu.clone(), device.clone(), queue)?;
    let mut c = Controls::new_material();
    let mut checks = vec![];
    let result = (|| {
        checks.extend(super::contract::exercise()?);
        let size = config.size;
        let pixels: Vec<u8> = (0..size)
            .flat_map(|y| {
                (0..size).flat_map(move |x| {
                    [
                        (32 + x * 191 / size) as u8,
                        (24 + y * 199 / size) as u8,
                        (64 + ((x / 16 + y / 16) % 2) * 120) as u8,
                    ]
                })
            })
            .collect();
        let fixture = parent.join("input-source.png");
        let bytes = texelith_io::encoders::encode_png8(
            size,
            size,
            texelith_io::encoders::PngChannels::Rgb,
            texelith_io::encoders::PngColorIntent::Srgb,
            &pixels,
        )
        .map_err(|e| e.to_string())?;
        texelith_io::publish_png(&fixture, &bytes).map_err(|e| e.to_string())?;
        c.drop_paths(vec![fixture.clone()]);
        wait(&mut runtime, &mut c, &device, |r, c| {
            r.publications > 0 && c.session().snapshot().is_some()
        })?;
        checks.push("drop-image-real-IO-decode-GPU-upload-publication".into());
        let ctx = egui::Context::default();
        for (key, map) in [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
            egui::Key::Num5,
            egui::Key::Num6,
            egui::Key::Num7,
        ]
        .into_iter()
        .zip(crate::material_ui::MAPS)
        {
            let before = runtime.publications;
            super::contract::frame(
                &mut c,
                &ctx,
                vec![super::contract::key(key, egui::Modifiers::NONE)],
                &[],
                false,
            );
            wait(&mut runtime, &mut c, &device, |r, _| {
                r.publications > before
            })?;
            if c.session()
                .snapshot()
                .unwrap()
                .graph
                .selected_output_ref()
                .port
                .map_kind()
                != map
            {
                return Err("shortcut did not reach GPU selection".into());
            }
            checks.push(format!("{key:?}-{map:?}-real-GPU-publication"));
        }
        let revision = c.session().document_revision();
        c.drop_paths(vec![parent.join("not-present.texelith.json")]);
        if c.pending_replacement().is_none() {
            return Err("project drop did not guard unsaved material".into());
        }
        c.resolve_dirty(super::DirtyChoice::Discard);
        runtime.tick(&mut c);
        if c.input_feedback_key() != Some("project-open-pending")
            || c.session().document_revision() != revision
        {
            return Err("project pending handler lost document".into());
        }
        checks.push("project-open-effect-explicit-unavailable-retains-document".into());
        for (paths, expected) in [
            (vec!["a.png", "b.jpg"], "batch-offer"),
            (vec!["studio.hdr"], "environment-pending"),
            (vec!["mesh.glb"], "mesh-pending"),
            (vec!["bad.txt"], "drop-unsupported"),
        ] {
            c.drop_paths(paths.into_iter().map(Into::into).collect());
            if c.input_feedback_key() != Some(expected)
                || !c.take_effects().is_empty()
                || c.session().document_revision() != revision
            {
                return Err(format!("incorrect typed drop handling: {expected}"));
            }
            checks.push(expected.into());
        }
        c.action(crate::command_adapter::UiAction::Request(
            CommandRequest::SetLocked {
                map: MapKind::Normal,
                locked: true,
            },
        ));
        c.drop_paths(vec![fixture]);
        if c.pending_replacement().is_none() {
            return Err("dirty GPU document was not guarded".into());
        }
        c.resolve_dirty(super::DirtyChoice::Cancel);
        if !c.take_effects().is_empty() || !c.session().is_dirty() {
            return Err("cancel lost dirty document".into());
        }
        checks.push("dirty-GPU-document-cancel-retains-material".into());
        Ok(())
    })();
    let diagnostics = gpu.diagnostics();
    let passed = result.is_ok() && diagnostics.validation_errors == 0 && checks.len() >= 30;
    let value = serde_json::json!({"task":"T-073","scenario":"input","passed":passed,"assertions":checks.len(),"checks":checks,
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"size":config.size,
        "gpu_validation_errors":diagnostics.validation_errors,"peak_bytes":diagnostics.peak_bytes,
        "error":result.as_ref().err(),"native_input":"deferred to orchestrator; event-shape tests are automated"});
    std::fs::write(
        report,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
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
