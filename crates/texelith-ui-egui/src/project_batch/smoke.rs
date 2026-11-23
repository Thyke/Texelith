use super::*;
use crate::startup::Config;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_gpu::{GpuWorker, WorkerEvent};
fn err(e: impl ToString) -> String {
    e.to_string()
}
fn check(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn pump(c: &mut Controls, device: &wgpu::Device, gpu: &GpuWorker) -> Result<(), String> {
    let start = Instant::now();
    while c.project_busy() {
        c.poll_project_io();
        if let Some(upload) = c.take_project_upload() {
            gpu.install_typed_source(upload).map_err(err)?;
        }
        device.poll(wgpu::PollType::Poll).map_err(err)?;
        while let Some(event) = gpu.try_event() {
            match event {
                WorkerEvent::SourceReady { content_hash } => {
                    c.ack_project_upload(content_hash);
                }
                other => return Err(format!("Project GPU upload: {other:?}")),
            }
        }
        if start.elapsed() > Duration::from_secs(60) {
            return Err("Project diagnostic timed out".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
pub fn exercise(config: &Config) -> Result<serde_json::Value, String> {
    let parent = config
        .report
        .as_ref()
        .and_then(|p| p.parent())
        .unwrap_or(std::path::Path::new("target-evidence/T-088"));
    let root = parent.join(format!(
        "projects-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(err)?
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).map_err(err)?;
    let root = std::fs::canonicalize(root).map_err(err)?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = config.backends()?;
    let instance = wgpu::Instance::new(desc);
    let adapter = crate::material_ui::block_on(instance.enumerate_adapters(config.backends()?))
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
    let gpu = GpuWorker::spawn(device.clone(), Arc::new(queue), Default::default()).map_err(err)?;
    let result = (|| {
        let source = root.join("source.png");
        texelith_io::write_normal_png(
            &source,
            config.size,
            config.size,
            &[128, 64, 32].repeat(config.size as usize * config.size as usize),
            false,
        )
        .map_err(err)?;
        let upload = crate::material_ui::presenter::decode(
            &source,
            crate::material_ui::ImportKind::Photo,
            NormalConvention::OpenGl,
        )?;
        let mut c = Controls::new_material();
        c.open_image(source.clone());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            return Err("Missing import effect".into());
        };
        c.complete_import(token, upload.descriptor).map_err(err)?;
        c.action(UiAction::Request(CommandRequest::SetLocked {
            map: MapKind::Normal,
            locked: true,
        }));
        let project = root.join("material.texelith.json");
        c.save_project_to(project.clone(), true)?;
        pump(&mut c, &device, &gpu)?;
        check(!c.session().is_dirty(), c.project_feedback())?;
        let exact = c.session().snapshot().unwrap().graph;
        let mut reopened = Controls::new_material();
        reopened.open_project_path(project.clone());
        pump(&mut reopened, &device, &gpu)?;
        check(
            reopened.session().snapshot().is_some(),
            reopened.project_feedback(),
        )?;
        check(
            MaterialGraphDto::from(reopened.session().snapshot().unwrap().graph)
                == MaterialGraphDto::from(exact),
            "Recipe roundtrip changed graph",
        )?;
        check(
            reopened.session().view_state().maps[&MapKind::Normal].locked,
            "Lock was not restored",
        )?;
        let before = reopened.session().snapshot().unwrap();
        reopened.open_project_path(root.join("missing.texelith.json"));
        pump(&mut reopened, &device, &gpu)?;
        check(
            reopened.session().snapshot().unwrap() == before,
            "Failed open replaced the document",
        )?;
        reopened.save_project_to(project.clone(), false)?;
        reopened.action(UiAction::Request(CommandRequest::SetLocked {
            map: MapKind::Height,
            locked: true,
        }));
        pump(&mut reopened, &device, &gpu)?;
        check(
            reopened.session().is_dirty(),
            "Save cleaned edits made after capture",
        )?;
        // Two accepted opens race; the invalid newest token must retain the old recipe.
        reopened.open_project_path(project.clone());
        reopened.poll_project_io();
        reopened.open_project_path(root.join("newest-missing.texelith.json"));
        pump(&mut reopened, &device, &gpu)?;
        check(
            reopened.session().view_state().maps[&MapKind::Height].locked,
            "Stale open crossed the newest token",
        )?;
        texelith_io::write_normal_png(
            &source,
            config.size,
            config.size,
            &[200, 120, 40].repeat(config.size as usize * config.size as usize),
            true,
        )
        .map_err(err)?;
        let mut changed = Controls::new_material();
        changed.open_project_path(project.clone());
        pump(&mut changed, &device, &gpu)?;
        check(
            changed.session().snapshot().is_none() && !changed.project_asset_issues().is_empty(),
            "Changed asset was accepted without a prompt",
        )?;
        changed.resolve_project_asset(AssetChoice::UseChanged)?;
        pump(&mut changed, &device, &gpu)?;
        check(
            changed.session().snapshot().is_some() && changed.session().is_dirty(),
            "Use changed did not publish a dirty updated recipe",
        )?;
        std::fs::rename(&source, root.join("relinked.png")).map_err(err)?;
        let mut missing = Controls::new_material();
        missing.open_project_path(project);
        pump(&mut missing, &device, &gpu)?;
        check(
            missing
                .project_asset_issues()
                .first()
                .is_some_and(|i| !i.changed),
            "Missing asset was not offered for relink",
        )?;
        missing.resolve_project_asset(AssetChoice::Relink(root.join("relinked.png")))?;
        pump(&mut missing, &device, &gpu)?;
        check(
            missing
                .project_asset_issues()
                .first()
                .is_some_and(|i| i.changed),
            "Relink bypassed changed hash choice",
        )?;
        missing.resolve_project_asset(AssetChoice::Cancel)?;
        exercise_restart(&mut changed)?;
        Ok(
            serde_json::json!({"task":"T-088","passed":true,"adapter":info.name,"backend":format!("{:?}",info.backend),"size":config.size,"artifacts":root,"checks":["save/reopen exact recipe and locks","failed-open preservation","save during edits","pending-open token races","changed-asset choice","missing-asset relink and cancel","restart incomplete keep-stopped/retry choices and registry start","real typed GPU source upload"]}),
        )
    })();
    gpu.close();
    gpu.join().map_err(err)?;
    result
}
fn exercise_restart(c: &mut Controls) -> Result<(), String> {
    let snapshot = c.session.snapshot().ok_or("Missing restart fixture")?;
    let settings = c
        .session
        .export_settings()
        .ok_or("Missing captured settings")?
        .clone();
    let capture =
        ExportCapture::new(&snapshot, c.session.document_revision(), &settings).map_err(err)?;
    let items = (0..2)
        .map(|i| BatchInput {
            source: OpaqueLocator(format!("restart-{i}")),
            source_digest: [i as u8; 32],
            destinations: (0..settings.outputs.len())
                .map(|o| OpaqueLocator(format!("restart-{i}-{o}")))
                .collect(),
        })
        .collect();
    let job = BatchJob::new(88, capture, settings, items, false).map_err(|e| format!("{e:?}"))?;
    let mut checkpoint = job.checkpoint();
    checkpoint.items[0].state = BatchState::Committed;
    checkpoint.items[1].state = BatchState::Evaluating;
    let stopped = BatchJob::restore(checkpoint).map_err(|e| format!("{e:?}"))?;
    let recipe = stopped.recipe().clone();
    c.session.set_batch(stopped).map_err(|e| format!("{e:?}"))?;
    c.resolve_batch_restart(RestartChoice::KeepStopped)?;
    check(
        !c.session.batch().unwrap().can_start(),
        "Restart began without consent",
    )?;
    c.resolve_batch_restart(RestartChoice::RetryIncomplete)?;
    let resumed = c.session.batch().unwrap();
    check(
        resumed.items().len() == 1
            && resumed.items()[0].index == 1
            && resumed.recipe() == &recipe
            && resumed.pending_work().is_some(),
        "Restart requeued completed work or lost captured recipe/start command",
    )
}
pub fn run(config: &Config) -> Result<(), String> {
    let result = if config.workflow_smoke {
        (|| {
            let export = crate::export_jobs::smoke::exercise(config)?;
            let packing = crate::packing_editor::smoke::exercise(config)?;
            let batch = crate::batch_jobs::smoke::exercise(config)?;
            let projects = exercise(config)?;
            Ok(
                serde_json::json!({"task":"T-088","scenario":"workflow","passed":true,"export":export,"packing_ui":packing,"batch":batch,"projects":projects}),
            )
        })()
    } else {
        exercise(config)
    };
    let report = match &result {
        Ok(v) => v.clone(),
        Err(e) => serde_json::json!({"task":"T-088","passed":false,"error":e}),
    };
    let path = config.report.as_ref().ok_or("Missing report")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    result.map(|_| ())
}
