//! Test-only barriers pause real exports without relying on GPU/disk timings.
use crate::gpu_availability;

use super::*;
use std::{
    path::Path,
    sync::{Mutex, mpsc},
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Context {
    gpu: Arc<GpuWorker>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}
struct Phase {
    name: &'static str,
    context: Context,
    capture: ExportCapture,
    release: mpsc::Sender<()>,
}
struct Hook {
    root: PathBuf,
    context: Option<Context>,
    send: mpsc::Sender<Phase>,
    seen: Vec<&'static str>,
    paused: bool,
}
static HOOK: Mutex<Option<Hook>> = Mutex::new(None);
pub(super) fn register_context(
    root: &Path,
    gpu: Arc<GpuWorker>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
) {
    if let Some(hook) = HOOK.lock().unwrap().as_mut()
        && std::fs::canonicalize(root).unwrap().starts_with(&hook.root)
    {
        hook.context = Some(Context { gpu, device, queue });
    }
}
pub(super) fn interactive_allowed(root: &Path) -> bool {
    HOOK.lock().unwrap().as_ref().is_none_or(|h| {
        !std::fs::canonicalize(root).unwrap().starts_with(&h.root)
            || (h.seen.len() == 2 && !h.paused)
    })
}
pub(crate) fn export_phase(destination: &Path, name: &'static str, capture: &ExportCapture) {
    let mut guard = HOOK.lock().unwrap();
    let Some(hook) = guard
        .as_mut()
        .filter(|h| destination.starts_with(&h.root) && !h.seen.contains(&name))
    else {
        return;
    };
    let Some(context) = hook.context.clone() else {
        return;
    };
    hook.seen.push(name);
    hook.paused = true;
    let (release, wait) = mpsc::channel();
    let _ = hook.send.send(Phase {
        name,
        context,
        capture: capture.clone(),
        release,
    });
    drop(guard);
    let _ = wait.recv_timeout(Duration::from_secs(20));
    if let Some(hook) = HOOK.lock().unwrap().as_mut() {
        hook.paused = false;
    }
}

#[test]
fn preview_completes_while_export_is_paused_before_gpu_and_publication() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/evidence/T-087")
        .join(format!("phase-regression-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut config = crate::startup::Config::parse(
        [
            "--mvp-smoke",
            "batch",
            "--size",
            "32",
            "--report",
            "unused.json",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    config.report = Some(root.join("report.json"));
    let (send, receive) = mpsc::channel();
    *HOOK.lock().unwrap() = Some(Hook {
        root: std::fs::canonicalize(root).unwrap(),
        context: None,
        send,
        seen: Vec::new(),
        paused: false,
    });
    let export = std::thread::spawn(move || {
        let result = super::smoke::exercise(&config);
        if let Err(error) = &result {
            eprintln!("Phase regression export: {error}");
        }
        result
    });
    let mut results = Vec::new();
    for id in 1..=2 {
        let phase = receive
            .recv_timeout(Duration::from_secs(20))
            .expect("export reached phase barrier");
        let (send, completion) = mpsc::channel();
        let preview = std::thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                if id == 1 {
                    bounded_4k_heatmap(&phase.context, &phase.capture)?;
                }
                let snapshot = EvaluationSnapshot {
                    generation: phase.capture.generation,
                    graph: phase.capture.graph,
                };
                let output = snapshot.graph.material_outputs().unwrap()[&MapKind::Roughness];
                phase
                    .context
                    .gpu
                    .request_material(texelith_gpu::MaterialRequest {
                        job_id: texelith_gpu::JobId(u64::MAX - 88),
                        request_id: id + 1,
                        extent: phase.capture.sources[0].extent,
                        snapshot: Arc::new(snapshot),
                        outputs: [(MapKind::Roughness, output)].into(),
                        quality: texelith_gpu::Quality::Full,
                    })
                    .map_err(err)?;
                let start = Instant::now();
                let bundle = loop {
                    phase
                        .context
                        .device
                        .poll(wgpu::PollType::Poll)
                        .map_err(err)?;
                    if let Some(bundle) = phase.context.gpu.try_latest_material() {
                        break bundle;
                    }
                    if start.elapsed() > Duration::from_secs(10) {
                        return Err("interactive GPU timeout".into());
                    }
                    std::thread::sleep(Duration::from_millis(2));
                };
                let presented = crate::material_ui::presenter::present(
                    phase.context.device,
                    phase.context.queue,
                    bundle,
                    MapKind::Roughness,
                    false,
                    None,
                    None,
                )?;
                if presented.image.pixels.is_empty() {
                    return Err("empty presentation".into());
                }
                Ok(())
            })();
            let _ = send.send(result);
        });
        let outcome = completion.recv_timeout(Duration::from_secs(15));
        // Always release the export, including on regression, so no worker leaks.
        let _ = phase.release.send(());
        preview.join().unwrap();
        results.push((phase.name, outcome));
    }
    let exported = export.join().unwrap();
    *HOOK.lock().unwrap() = None;
    assert!(exported.is_ok(), "{exported:?}");
    for (phase, outcome) in results {
        assert!(
            matches!(outcome, Ok(Ok(()))),
            "preview waited on active {phase} export: {outcome:?}"
        );
    }
}

fn bounded_4k_heatmap(context: &Context, capture: &ExportCapture) -> Result<(), String> {
    let extent = Extent::new(4096, 4096).map_err(err)?;
    let pixels = texelith_gpu::TypedSourcePixels::Rgba8 {
        pixels: vec![127; 4096 * 4096 * 4].into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let source = pixels.descriptor(extent).map_err(err)?;
    let hash = source.content_hash;
    let upload = TypedSourceUpload {
        descriptor: source.clone(),
        pixels,
    };
    let start = Instant::now();
    loop {
        match context.gpu.install_typed_source(upload.clone()) {
            Ok(()) => break,
            Err(error)
                if error.to_string().contains("Busy")
                    && start.elapsed() < Duration::from_secs(10) =>
            {
                context.device.poll(wgpu::PollType::Poll).map_err(err)?;
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(err(error)),
        }
    }
    drop(upload);
    let start = Instant::now();
    loop {
        context.device.poll(wgpu::PollType::Poll).map_err(err)?;
        if matches!(context.gpu.try_event(), Some(texelith_gpu::WorkerEvent::SourceReady { content_hash }) if content_hash == hash)
        {
            break;
        }
        if start.elapsed() > Duration::from_secs(10) {
            return Err("4K source upload timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let graph = instantiate_recipe(&capture.graph, source).map_err(err)?;
    let output = graph.material_outputs().unwrap()[&MapKind::Roughness];
    context
        .gpu
        .request_material(texelith_gpu::MaterialRequest {
            job_id: texelith_gpu::JobId(u64::MAX - 88),
            request_id: 1,
            snapshot: Arc::new(EvaluationSnapshot {
                generation: capture.generation,
                graph,
            }),
            outputs: [(MapKind::Roughness, output)].into(),
            extent,
            quality: texelith_gpu::Quality::Full,
        })
        .map_err(err)?;
    let start = Instant::now();
    let bundle = loop {
        context.device.poll(wgpu::PollType::Poll).map_err(err)?;
        if let Some(bundle) = context.gpu.try_latest_material() {
            break bundle;
        }
        if start.elapsed() > Duration::from_secs(10) {
            return Err("4K evaluation timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    if bundle.extent != extent {
        return Err("4K regression received a proxy".into());
    }
    let result = crate::material_ui::presenter::present(
        context.device.clone(),
        context.queue.clone(),
        bundle,
        MapKind::Roughness,
        true,
        None,
        None,
    );
    let presented = result?;
    if presented.image.size != [1024, 1024] || presented.metrics.is_none() {
        return Err("4K heatmap failed bounded presentation during active export".into());
    }
    Ok(())
}

#[path = "manifest_tests.rs"]
mod manifests;
