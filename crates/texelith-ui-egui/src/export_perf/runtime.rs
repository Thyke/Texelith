use super::{Config, Metric, fixture_graph, fixture_pixels, metrics, scenario_plan};
use crate::export_jobs::{ExportJobs, ExportOptions, JobStatus};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::*;

fn wait<T>(
    device: &wgpu::Device,
    mut next: impl FnMut() -> Result<Option<T>, String>,
) -> Result<T, String> {
    let start = Instant::now();
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| e.to_string())?;
        if let Some(value) = next()? {
            return Ok(value);
        }
        if start.elapsed() > Duration::from_secs(180) {
            return Err("Export diagnostic timed out".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn effect(graph: &Graph, plan: &ExportPlan) -> Result<Effect, String> {
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 1,
        graph: graph.clone(),
    });
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: plan.clone(),
        })
        .map_err(|e| e.to_string())?;
    session
        .dispatch(CommandRequest::RequestExport {})
        .map_err(|e| e.to_string())?
        .effects
        .into_iter()
        .next()
        .ok_or("Missing immutable export effect".into())
}

fn job(
    jobs: &mut ExportJobs,
    device: &wgpu::Device,
    graph: &Graph,
    plan: &ExportPlan,
    destination: &Path,
    measured: bool,
) -> Result<(JobStatus, Option<Metric>), String> {
    let effect = effect(graph, plan)?;
    if measured {
        metrics::arm()?;
    }
    let result = (|| {
        jobs.start(
            effect,
            ExportOptions {
                destination: destination.into(),
                material_name: "Perf".into(),
                ..Default::default()
            },
        )?;
        wait(device, || Ok(jobs.status().filter(|s| s.terminal())))
    })();
    let metric = if measured {
        Some(metrics::finish(
            result
                .as_ref()
                .is_ok_and(|s| s.stage == ExportStage::Completed),
        ))
    } else {
        None
    };
    let status = result?;
    if status.stage != ExportStage::Completed {
        return Err(format!("Export {:?}: {:?}", status.stage, status.error));
    }
    Ok((status, metric.transpose()?))
}

/// Verify published file identities and headers outside the measured interval.
/// Numeric preset correctness is covered by T-095's engine_profiles suite.
fn verify(destination: &Path, plan: &ExportPlan, size: u32) -> Result<serde_json::Value, String> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(destination.join("assignment.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let published: ExportPlan =
        serde_json::from_value(manifest["plan"].clone()).map_err(|e| e.to_string())?;
    if published != *plan {
        return Err("Published plan differs from immutable request".into());
    }
    let records = manifest["outputs"]
        .as_array()
        .ok_or("Missing output records")?;
    if records.len() != plan.outputs.len() {
        return Err("Incomplete set".into());
    }
    let mut files = Vec::new();
    for (record, output) in records.iter().zip(&plan.outputs) {
        let file = record["file"].as_str().ok_or("Missing filename")?;
        let path = destination.join(file);
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        if record["name"] != output.name
            || record["encoded_bytes"] != bytes.len()
            || record["extent"] != serde_json::json!({"width":size,"height":size})
        {
            return Err(format!("Invalid published record {file}"));
        }
        match output.format {
            ExportFormat::Png8 | ExportFormat::Png16 => {
                if bytes.len() < 33
                    || &bytes[..8] != b"\x89PNG\r\n\x1a\n"
                    || u32::from_be_bytes(bytes[16..20].try_into().unwrap()) != size
                    || u32::from_be_bytes(bytes[20..24].try_into().unwrap()) != size
                    || bytes[24] != output.format.bit_depth()
                {
                    return Err(format!("Invalid PNG {file}"));
                }
            }
            ExportFormat::DdsBc5 => {
                if bytes.len() < 148
                    || &bytes[..4] != b"DDS "
                    || u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != size
                    || u32::from_le_bytes(bytes[16..20].try_into().unwrap()) != size
                    || u32::from_le_bytes(bytes[128..132].try_into().unwrap()) != 83
                {
                    return Err(format!("Invalid DDS {file}"));
                }
            }
            ExportFormat::Exr32 => {
                if !bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
                    return Err(format!("Invalid EXR {file}"));
                }
            }
        }
        files.push(serde_json::json!({"file":file,"bytes":bytes.len(),"format":output.format}));
    }
    Ok(serde_json::json!(files))
}

pub fn exercise(config: &Config) -> Result<serde_json::Value, String> {
    let root = config.report.with_extension("artifacts");
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let backend = crate::startup::Config {
        backend: config.backend.clone(),
        ..Default::default()
    }
    .backends()?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backend;
    let instance = wgpu::Instance::new(desc);
    let adapter = crate::material_ui::block_on(instance.enumerate_adapters(backend))
        .into_iter()
        .find(|a| {
            config
                .adapter
                .as_ref()
                .is_none_or(|s| a.get_info().name.to_lowercase().contains(&s.to_lowercase()))
        })
        .ok_or("Requested native adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let setup = Instant::now();
    let (device, queue) =
        crate::material_ui::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
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
    let result = (|| {
        let fixture_started = Instant::now();
        let pixels = fixture_pixels(config.size, 0);
        let TypedSourcePixels::Rgba8 { pixels, .. } = pixels else {
            unreachable!()
        };
        let png = texelith_io::encoders::encode_png8(
            config.size,
            config.size,
            texelith_io::encoders::PngChannels::Rgba,
            texelith_io::encoders::PngColorIntent::Srgb,
            &pixels,
        )
        .map_err(|e| e.to_string())?;
        std::fs::write(root.join("source.png"), &png).map_err(|e| e.to_string())?;
        let fixture_ns = fixture_started.elapsed().as_nanos() as u64;
        let decode = Instant::now();
        let decoded = texelith_io::decode_bytes(&png, texelith_io::ColorIntent::PhotoSrgb)
            .map_err(|e| e.to_string())?;
        let decode_ns = decode.elapsed().as_nanos() as u64;
        drop(png);
        drop(pixels);
        let pixels = TypedSourcePixels::Rgba8 {
            pixels: decoded.rgba8.into(),
            interpretation: InputInterpretation::SrgbPhoto,
        };
        let source = pixels
            .descriptor(Extent::new(config.size, config.size).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let graph = fixture_graph(source.clone(), config.maximum)?;
        let plan = scenario_plan(&graph, &config.profile)?;
        gpu.install_typed_source(TypedSourceUpload {
            descriptor: source.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
        wait(&device, || match gpu.try_event() {
            Some(WorkerEvent::SourceReady { content_hash })
                if content_hash == source.content_hash =>
            {
                Ok(Some(()))
            }
            Some(e) => Err(format!("Source upload failed: {e:?}")),
            None => Ok(None),
        })?;
        let setup_ns = setup.elapsed().as_nanos() as u64;
        let mut jobs = ExportJobs::spawn(gpu.clone())?;
        let warmup = Instant::now();
        job(
            &mut jobs,
            &device,
            &graph,
            &plan,
            &root.join("warmup"),
            false,
        )?;
        let warmup_ns = warmup.elapsed().as_nanos() as u64;
        let invalidated = gpu.invalidate_result_caches().map_err(|e| e.to_string())?;
        wait(&device, || match invalidated.try_recv() {
            Ok(()) => Ok(Some(())),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(_) => Err("Cache invalidation acknowledgement disconnected".into()),
        })?;
        let before = gpu.diagnostics();
        let (status, metric) = job(
            &mut jobs,
            &device,
            &graph,
            &plan,
            &root.join("measured"),
            true,
        )?;
        let metric = metric.ok_or("Missing publication metric")?;
        let after = gpu.diagnostics();
        let files = verify(&status.destination, &plan, config.size)?;
        let submissions = after.submitted.saturating_sub(before.submitted);
        let passed = config.size == 4096
            && metric.request_to_disk_ns < 2_000_000_000
            && submissions > 0
            && after.validation_errors == 0;
        Ok(serde_json::json!({
            "task":"T-096", "passed":passed, "profile":config.profile,
            "quality":if config.maximum {"maximum"} else {"default"}, "size":config.size,
            "target_ns":2_000_000_000u64, "metric":metric, "cache_invalidated":true,
            "compute_submissions":submissions, "validation_errors":after.validation_errors,
            "cache_misses":after.cache_misses.saturating_sub(before.cache_misses),
            "cache_hits":after.cache_hits.saturating_sub(before.cache_hits),
            "gpu_peak_bytes":after.peak_bytes, "adapter":info.name, "backend":format!("{:?}",info.backend),
            "cpu_budget_bytes":crate::batch_jobs::CPU_BUDGET,
            "cpu_scratch_peak_bytes":crate::batch_jobs::memory::scratch_peak_bytes(),
            "cpu_retained_headroom_bytes":crate::batch_jobs::INTERACTIVE_RESERVATION,
            "driver":info.driver, "driver_info":info.driver_info,
            "fixture_generation_encode_ns":fixture_ns, "decode_ns":decode_ns, "setup_upload_ns":setup_ns,
            "cold_compile_and_full_warmup_ns":warmup_ns, "files":files, "plan":plan,
            "graph":MaterialGraphDto::from(graph), "source_hash":source.content_hash,
            "artifacts":root, "gpu_timestamp_ns":null,
            "gpu_timing_scope":"Not measured: production worker has no GPU timestamp instrumentation. Per-output evaluation_readback_ns is host wall time, includes scheduling, CPU setup, GPU compute and readback.",
            "warmup_scope":"Full same-profile export before explicit cache invalidation; driver shader cache warm. Node-private kernels evicted with results can recompile inside the measured interval.",
            "quality_scope":"Full resolution both modes; maximum raises AO from 8x8 to 32x32 samples. BC5 uses production Default fit in both modes.",
            "measurement_boundary":"Immutable ExportJobs acceptance -> publish_set returns after all encode/write/sync/final rename; verification and decode excluded."
        }))
    })();
    gpu.close();
    gpu.join().map_err(|e| e.to_string())?;
    result
}
