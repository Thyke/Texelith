//! Acceptance driver of the production worker, never a simulated stress result.
//! The headless bootstrap alone creates the shared device. No UI timing claim.
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, Node, NodeId, NodeParams,
    SourceDescriptor,
};
use texelith_gpu::{GpuConfig, GpuWorker, Quality, RenderRequest, SourceUpload, WorkerEvent};

struct Config {
    backend: String,
    adapter: String,
    size: u32,
    updates: u64,
    budget: u64,
    expect_budget_error: bool,
    report: PathBuf,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self {
            backend: "dx12".into(),
            adapter: "RTX 5070 Ti".into(),
            size: 4096,
            updates: 10_000,
            budget: 1024 * 1024 * 1024,
            expect_budget_error: false,
            report: "target/evidence/T-017/stress.json".into(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--expect-budget-error" {
                config.expect_budget_error = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--backend" => config.backend = value,
                "--adapter-name" => config.adapter = value,
                "--size" => config.size = value.parse().map_err(|_| "invalid size")?,
                "--updates" => {
                    config.updates = value.parse().map_err(|_| "invalid update count")?
                }
                "--budget-mib" => {
                    config.budget = value
                        .parse::<u64>()
                        .map_err(|_| "invalid budget")?
                        .checked_mul(1024 * 1024)
                        .ok_or("budget overflow")?
                }
                "--report" => config.report = value.into(),
                _ => return Err(format!("unknown argument {key}")),
            }
        }
        if config.size == 0
            || config.size > 4096
            || config.updates == 0
            || config.updates > 1_000_000
            || config.budget == 0
            || config.adapter.trim().is_empty()
        {
            return Err(
                "size 1..4096, updates 1..1000000, nonzero budget and adapter name required".into(),
            );
        }
        config.backends()?;
        Ok(config)
    }

    fn backends(&self) -> Result<wgpu::Backends, String> {
        match self.backend.as_str() {
            "dx12" => Ok(wgpu::Backends::DX12),
            "vulkan" => Ok(wgpu::Backends::VULKAN),
            "metal" => Ok(wgpu::Backends::METAL),
            _ => Err("backend must be dx12, vulkan or metal".into()),
        }
    }
}

fn request(id: u64, source: &SourceDescriptor, quality: Quality) -> RenderRequest {
    let params = [
        NodeParams::Source(source.clone()),
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Blur {
            sigma_source_px: 1.25,
        },
        NodeParams::NormalFromHeight {
            strength: (id % 101) as f32 / 10.0,
        },
    ];
    let nodes = params
        .into_iter()
        .enumerate()
        .map(|(index, params)| Node {
            id: NodeId(index as u64 + 1),
            input: (index != 0).then_some(NodeId(index as u64)),
            params,
        })
        .collect();
    RenderRequest {
        request_id: id,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: id,
            graph: Graph::new(nodes, NodeId(4)).unwrap(),
        }),
        output: NodeId(4),
        extent: if quality == Quality::Full {
            source.extent
        } else {
            texelith_gpu::proxy_extent(source.extent).unwrap()
        },
        quality,
    }
}

fn execute(config: &Config, report: &mut Value) -> Result<(), String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = config.backends()?;
    descriptor.flags |= wgpu::InstanceFlags::VALIDATION;
    let instance = wgpu::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(config.backends()?));
    report["available_adapters"] = json!(
        adapters
            .iter()
            .map(|a| a.get_info().name)
            .collect::<Vec<_>>()
    );
    let adapter = adapters
        .into_iter()
        .find(|a| {
            let info = a.get_info();
            info.device_type != wgpu::DeviceType::Cpu
                && info
                    .name
                    .to_lowercase()
                    .contains(&config.adapter.to_lowercase())
        })
        .ok_or("requested native GPU adapter unavailable")?;
    let info = adapter.get_info();
    report["adapter"] = json!({ "name": info.name, "backend": format!("{:?}", info.backend), "driver": info.driver, "driver_info": info.driver_info });
    let limits = wgpu::Limits {
        max_texture_dimension_2d: 4096,
        ..wgpu::Limits::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("T-017 shared headless device"),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let worker = GpuWorker::spawn(
        Arc::clone(&device),
        queue,
        GpuConfig {
            budget_bytes: config.budget,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let validation = Arc::new(Mutex::new(Vec::<String>::new()));
    let callback_errors = Arc::clone(&validation);
    device.on_uncaptured_error(Arc::new(move |error| {
        callback_errors.lock().unwrap().push(error.to_string());
    }));
    let extent = Extent {
        width: config.size,
        height: config.size,
    };
    let mut pixels = vec![0u8; config.size as usize * config.size as usize * 4];
    for (index, pixel) in pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let value = ((index * 73 + index / config.size as usize * 19) % 256) as u8;
        pixel.copy_from_slice(&[value, value, value, 255]);
    }
    let source = SourceDescriptor::from_rgba8(&pixels, extent, InputInterpretation::LinearHeight)
        .map_err(|e| e.to_string())?;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(60);
    worker
        .install_source(SourceUpload {
            descriptor: source.clone(),
            pixels: pixels.into(),
        })
        .map_err(|e| e.to_string())?;
    let mut submitted_final = false;
    let mut source_ready = false;
    let mut budget_rejected = false;
    let mut current_final = false;
    let mut stale_received = 0;
    let mut terminal_error = None;
    loop {
        while let Some(event) = worker.try_event() {
            match event {
                WorkerEvent::SourceReady { content_hash }
                    if content_hash == source.content_hash =>
                {
                    source_ready = true
                }
                WorkerEvent::BudgetExceeded {
                    budget_bytes,
                    required_bytes,
                } => {
                    report["budget_rejection"] =
                        json!({ "budget_bytes": budget_bytes, "required_bytes": required_bytes });
                    budget_rejected =
                        budget_bytes == config.budget && required_bytes > budget_bytes;
                    if !budget_rejected {
                        terminal_error = Some("invalid structured budget rejection".into());
                    }
                }
                WorkerEvent::DeviceLost => terminal_error = Some("device lost".into()),
                WorkerEvent::Error(error) => terminal_error = Some(error.to_string()),
                _ => {}
            }
        }
        if source_ready && !submitted_final {
            for id in 1..=config.updates {
                worker
                    .request(request(id, &source, Quality::Proxy))
                    .map_err(|e| e.to_string())?;
            }
            let mut full = request(config.updates + 1, &source, Quality::Full);
            // Release promotes quality without changing the final graph/generation.
            full.snapshot = request(config.updates, &source, Quality::Full).snapshot;
            worker.request(full).map_err(|e| e.to_string())?;
            submitted_final = true;
        }
        if let Some(frame) = worker.try_latest() {
            if frame.request_id != config.updates + 1
                || frame.generation != config.updates
                || frame.quality != Quality::Full
                || frame.extent != extent
            {
                stale_received += 1;
            } else {
                if frame.lease.texture().format() != wgpu::TextureFormat::Rgba8Unorm
                    || frame.lease.texture().width() != config.size
                    || frame.lease.texture().height() != config.size
                {
                    terminal_error =
                        Some("published display lease has wrong texture shape/format".into());
                }
                if !(frame.requested_at <= frame.submitted_at
                    && frame.submitted_at <= frame.completed_at
                    && frame.completed_at <= Instant::now())
                {
                    terminal_error = Some("invalid monotonic worker timestamps".into());
                }
                report["final_request_id"] = json!(frame.request_id);
                report["final_generation"] = json!(frame.generation);
                current_final = true;
            }
        }
        if terminal_error.is_some()
            || budget_rejected
            || current_final
            || Instant::now() >= deadline
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    worker.close();
    let stats = worker.diagnostics();
    report["elapsed_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    report["worker"] = json!({ "peak_in_flight": stats.peak_in_flight, "peak_pending": stats.peak_pending, "peak_bytes": stats.peak_bytes, "allocated_bytes": stats.allocated_bytes, "submitted": stats.submitted, "coalesced": stats.coalesced, "stale_publications": stats.stale_publications, "validation_errors": stats.validation_errors });
    report["stale_received"] = json!(stale_received);
    report["validation_errors"] = json!(*validation.lock().unwrap());
    if let Some(error) = terminal_error {
        return Err(error);
    }
    if stats.peak_in_flight > 1
        || stats.peak_pending > 1
        || stats.peak_bytes > config.budget
        || stats.stale_publications != 0
        || stats.validation_errors != 0
        || stale_received != 0
        || !validation.lock().unwrap().is_empty()
    {
        return Err("boundedness, stale publication, budget or validation assertion failed".into());
    }
    if config.expect_budget_error {
        if !budget_rejected || current_final {
            return Err("expected structured budget rejection before deadline".into());
        }
    } else if budget_rejected || !current_final || stats.submitted == 0 {
        return Err("current final full output not published before deadline".into());
    }
    Ok(())
}

fn main() {
    let config = match Config::parse() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let mut report = json!({ "passed": false, "role": "test-author", "requested_backend": config.backend, "requested_adapter": config.adapter, "size": config.size, "updates": config.updates, "budget_bytes": config.budget, "expect_budget_error": config.expect_budget_error, "profile": if cfg!(debug_assertions) { "debug" } else { "release" }, "worktree": env!("CARGO_MANIFEST_DIR"), "ui_performance_gate": false });
    let outcome = execute(&config, &mut report);
    report["passed"] = json!(outcome.is_ok());
    if let Err(error) = &outcome {
        report["error"] = json!(error);
    }
    let written = (|| -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = config.report.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&config.report, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    })();
    if let Err(error) = written {
        eprintln!("cannot write report: {error}");
        std::process::exit(2);
    }
    if outcome.is_err() {
        eprintln!("FAIL: {}", report["error"]);
        std::process::exit(1);
    }
    println!("PASS: {}", config.report.display());
}
