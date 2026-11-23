//! Original CPU oracles plus native AO node and worker measurements.
use super::contract::{aggregate_timings, validate_report};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{
    AoParameters, AoResources, HorizonAoKernel, MaterialPlan, Quality, ResourceReservations,
    TypedSourcePixels, WorkerEvent,
    pool::{AllocationDesc, ResourcePool},
};
#[allow(dead_code)]
#[path = "../../../tests/mvp_t051/oracle.rs"]
mod oracle;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t051/support.rs"]
mod support;
use oracle::{Controls, ao, image, seeded};
use support::{GIB, Gpu, request, wait};
type Result<T> = std::result::Result<T, String>;

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn compare(actual: &[f32], expected: &[f64]) -> Result<()> {
    require(
        !actual.is_empty() && actual.len() == expected.len(),
        "AO oracle extent",
    )?;
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| {
            if !a.is_finite() || !(0.0..=1.0).contains(&a) || !b.is_finite() {
                f64::INFINITY
            } else {
                (f64::from(a) - b).abs()
            }
        })
        .fold(0.0_f64, f64::max);
    require(error <= 2e-5, &format!("AO CPU oracle error {error}"))
}
fn check(checks: &mut Vec<Value>, name: &str) {
    checks.push(json!({"name":name,"passed":true}));
}
fn numerics(g: &Gpu, checks: &mut Vec<Value>) -> Result<()> {
    let worker = g.worker(GIB);
    let mut id = 0;
    for (w, h, kind) in [
        (1, 1, "flat"),
        (17, 13, "flat"),
        (1, 23, "singleton"),
        (23, 1, "singleton"),
        (17, 11, "ramp"),
        (31, 19, "impulse"),
        (257, 193, "npot"),
    ] {
        let e = Extent::new(w, h).map_err(|e| e.to_string())?;
        let pixels = match kind {
            "flat" => vec![-2.0; (w * h) as usize],
            "ramp" => image(e, |x, y| (x as f32 * 0.7 + y as f32 * 0.3) / 1024.0 - 0.02),
            "impulse" => image(e, |x, y| if x == 0 && y == 0 { 0.05 } else { 0.0 }),
            _ => seeded(e),
        };
        let source = g.install(
            &worker,
            e,
            TypedSourcePixels::HeightF32(pixels.clone().into()),
        );
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            for (directions, steps, radius, bias) in
                [(8, 8, 32.0, 0.01), (7, 11, 5.5, 0.0), (32, 32, 512.0, 0.01)]
            {
                id += 1;
                let c = Controls {
                    directions,
                    steps,
                    radius,
                    bias,
                    boundary,
                };
                let actual = g.evaluate(&worker, request(&source, id, c, Quality::Full));
                compare(&actual, &ao(&pixels, e, e, c))?;
                if kind == "flat" {
                    require(actual.iter().all(|&v| v == 1.0), "flat AO must equal 1")?;
                }
            }
        }
        check(checks, &format!("cpu_oracle_{kind}_{w}x{h}"));
    }
    let e = Extent::new(33, 33).unwrap();
    let pixels = image(e, |x, y| {
        if (x as i32 - 16).abs() <= 3 && (y as i32 - 16).abs() <= 3 {
            0.0
        } else {
            0.1
        }
    });
    let source = g.install(
        &worker,
        e,
        TypedSourcePixels::HeightF32(pixels.clone().into()),
    );
    let c = Controls {
        radius: 12.0,
        ..Default::default()
    };
    id += 1;
    let actual = g.evaluate(&worker, request(&source, id, c, Quality::Full));
    compare(&actual, &ao(&pixels, e, e, c))?;
    require(
        actual[16 * 33 + 16] < actual[0] && actual[0] == 1.0,
        "cavity vs plateau",
    )?;
    check(checks, "cavity_darker_than_plateau");
    let e = Extent::new(32, 24).unwrap();
    let pixels = image(e, |x, y| {
        let u = std::f64::consts::TAU * (f64::from(x) + 0.5) / 32.0;
        let v = std::f64::consts::TAU * (f64::from(y) + 0.5) / 24.0;
        (0.02 * (u.cos() + v.cos())) as f32
    });
    let source = g.install(
        &worker,
        e,
        TypedSourcePixels::HeightF32(pixels.clone().into()),
    );
    let c = Controls {
        radius: 7.0,
        ..Default::default()
    };
    id += 1;
    let actual = g.evaluate(&worker, request(&source, id, c, Quality::Full));
    compare(&actual, &ao(&pixels, e, e, c))?;
    require(
        (0..24).all(|y| (actual[y * 32] - actual[y * 32 + 31]).abs() <= 2e-5)
            && (0..32).all(|x| (actual[x] - actual[23 * 32 + x]).abs() <= 2e-5),
        "wrap continuity",
    )?;
    id += 1;
    let cached = g.evaluate(&worker, request(&source, id, c, Quality::Full));
    require(
        bytemuck::cast_slice::<f32, u8>(&actual) == bytemuck::cast_slice::<f32, u8>(&cached),
        "determinism",
    )?;
    check(checks, "wrap_continuity");
    check(checks, "determinism");
    let e = Extent::new(2048, 32).unwrap();
    let pixels = image(e, |x, _| {
        if (x / 16).is_multiple_of(2) {
            0.0
        } else {
            0.01
        }
    });
    let source = g.install(&worker, e, TypedSourcePixels::HeightF32(pixels.into()));
    let proxy = texelith_gpu::proxy_extent(e).map_err(|e| e.to_string())?;
    let pixels = image(
        proxy,
        |x, _| if (x / 8).is_multiple_of(2) { 0.0 } else { 0.01 },
    );
    for radius in [8.0, 32.0] {
        id += 1;
        let c = Controls {
            radius,
            ..Default::default()
        };
        compare(
            &g.evaluate(&worker, request(&source, id, c, Quality::Proxy)),
            &ao(&pixels, proxy, e, c),
        )?;
    }
    check(checks, "source_radius_proxy_rectangular_uv");
    require(
        worker.diagnostics().validation_errors == 0,
        "numerical validation errors",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(())
}

fn worker_contracts(g: &Gpu, checks: &mut Vec<Value>) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = g.worker(256 * 1024);
    let e = Extent::new(129, 129).unwrap();
    let source = g.install(
        &worker,
        e,
        TypedSourcePixels::HeightF32(vec![0.0; 129 * 129].into()),
    );
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 256 * 1024,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    let submitted = worker.diagnostics().submitted;
    worker
        .request_material(request(&source, 1, Controls::default(), Quality::Full))
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event()),WorkerEvent::BudgetExceeded {required_bytes,budget_bytes} if required_bytes>budget_bytes),
        "budget rejection",
    )?;
    require(
        worker.diagnostics().submitted == submitted && worker.try_latest_material().is_none(),
        "budget rejected after dispatch",
    )?;
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    require(
        g.evaluate(
            &worker,
            request(&source, 2, Controls::default(), Quality::Full),
        )
        .iter()
        .all(|&v| v == 1.0),
        "budget recovery",
    )?;
    require(
        worker.diagnostics().peak_bytes <= 256 * 1024,
        "low budget peak",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    check(checks, "budget_rejection_and_recovery");
    let worker = g.worker(GIB);
    let e = Extent::new(1024, 513).unwrap();
    let source = g.install(&worker, e, TypedSourcePixels::HeightF32(seeded(e).into()));
    let before = worker.diagnostics();
    worker
        .request_material(request(
            &source,
            1,
            Controls {
                directions: 32,
                steps: 32,
                radius: 512.0,
                ..Default::default()
            },
            Quality::Full,
        ))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(60);
    while worker.diagnostics().submitted == before.submitted {
        if let Some(event) = worker.try_event() {
            return Err(format!("before AO submission: {event:?}"));
        }
        require(Instant::now() < deadline, "AO never submitted")?;
        std::thread::yield_now();
    }
    for id in 2..=32 {
        worker
            .request_material(request(
                &source,
                id,
                Controls {
                    bias: 1e10,
                    ..Default::default()
                },
                Quality::Full,
            ))
            .map_err(|e| e.to_string())?;
    }
    let bundle = wait(|| {
        if let Some(event) = worker.try_event() {
            panic!("AO cancellation: {event:?}");
        }
        worker.try_latest_material()
    });
    require(
        bundle.request_id == 32 && bundle.generation == 32,
        "stale AO generation",
    )?;
    require(
        g.read(bundle.outputs[&texelith_core::MapKind::Ao].texture())
            .iter()
            .all(|&v| v == 1.0),
        "cancelled AO published",
    )?;
    let stats = worker.diagnostics();
    require(
        stats.cancellation_requests > before.cancellation_requests
            && stats.cancelled_pass_batches > before.cancelled_pass_batches
            && stats.stale_publications == 0
            && stats.validation_errors == 0
            && stats.peak_bytes <= GIB
            && stats.peak_pending == 1
            && stats.peak_in_flight == 1,
        "cancellation diagnostics",
    )?;
    drop(bundle);
    worker.join().map_err(|e| e.to_string())?;
    check(checks, "real_worker_cancellation");
    Ok(stats)
}

fn params(c: Controls) -> AoParameters {
    AoParameters {
        directions: c.directions,
        steps: c.steps,
        radius_source_px: c.radius,
        bias: c.bias,
        boundary: c.boundary,
    }
}
fn poll(g: &Gpu) -> Result<()> {
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn measure(g: &Gpu, e: Extent, c: Controls) -> Result<Value> {
    let params = params(c);
    let table = params.sample_bytes().map_err(|e| e.to_string())?;
    let payload = u64::from(e.width) * u64::from(e.height) * 4;
    let mut ledger = ResourcePool::new(GIB);
    let desc = AllocationDesc::Texture {
        extent: e,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
    };
    let ids = ledger
        .allocate_plan(&[
            desc,
            desc,
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Buffer { bytes: table },
            AllocationDesc::Staging { bytes: 64 + table },
            AllocationDesc::Staging { bytes: payload },
        ])
        .map_err(|e| e.to_string())?;
    let texture = || {
        g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("AO measured payload"),
            size: wgpu::Extent3d {
                width: e.width,
                height: e.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let input = texture();
    let output = texture();
    let buffer = |size| {
        g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AO measured uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let uniform = buffer(64);
    let samples = buffer(table);
    let pixels = seeded(e);
    g.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 4),
            rows_per_image: Some(e.height),
        },
        input.size(),
    );
    let cold = Instant::now();
    let kernel =
        HorizonAoKernel::new(g.device.clone(), c.directions, c.steps).map_err(|e| e.to_string())?;
    let pass = kernel
        .prepare(
            &g.queue,
            AoResources {
                input: &input,
                output: &output,
                uniform: &uniform,
                samples: &samples,
            },
            e,
            params,
        )
        .map_err(|e| e.to_string())?;
    let cold_prepare_ms = cold.elapsed().as_secs_f64() * 1000.0;
    g.queue.submit([g
        .device
        .create_command_encoder(&Default::default())
        .finish()]);
    poll(g)?;
    // Upload and preparation staging retire before the first warmup.
    ledger.release(ids[4]).map_err(|e| e.to_string())?;
    ledger.release(ids[5]).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let start = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        pass.encode(&mut encoder);
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        poll(g)?;
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = aggregate_timings(&rows, 10)?;
    let peak = ledger.diagnostics().peak_bytes;
    require(peak <= GIB, "AO measurement budget")?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":timing.measurements,
        "p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"peak_bytes":peak,
        "cancellation_errors":0,"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>(),
        "cold_prepare_ms":cold_prepare_ms,"directions":c.directions,"steps":c.steps,"radius_source_px":c.radius}),
    )
}
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
) -> Result<Value> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let g = Gpu { device, queue };
    let mut checks = Vec::new();
    numerics(&g, &mut checks)?;
    let stats = worker_contracts(&g, &mut checks)?;
    let source = SourceDescriptor {
        content_hash: [51; 32],
        extent,
        interpretation: InputInterpretation::LinearHeight,
    };
    let plan = MaterialPlan::new(
        &request(&source, 1, Controls::default(), Quality::Full),
        GIB,
        ResourceReservations {
            viewport_bytes: 144 * 1024 * 1024,
            readback_bytes: 128 * 1024 * 1024,
        },
    )
    .map_err(|e| e.to_string())?;
    require(plan.peak_bytes() <= GIB, "shared 4K budget")?;
    check(&mut checks, "shared_budget_plan");
    let mut timing = measure(&g, extent, Controls::default())?;
    let maximum = measure(
        &g,
        extent,
        Controls {
            directions: 32,
            steps: 32,
            radius: 512.0,
            ..Default::default()
        },
    )?;
    timing["peak_bytes"] = json!(timing["peak_bytes"].as_u64().unwrap().max(stats.peak_bytes));
    timing["cancellation_errors"] = json!(stats.stale_publications);
    poll(&g)?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "AO GPU validation error",
    )?;
    let manifest = json!({"directions":8,"steps":8,"radius_source_px":32.0,"bias":0.01,"boundary":"repeat",
        "seed":51,"size":extent.width,"algorithm_revision":1,"step_placement":"radius*k/steps; k=1..steps",
        "direction_phase":"+X; toward +image Y","distance":"Euclidean UV using original source extent"});
    let mut bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &bytes,
        Extent::new((bytes.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash;
    let mut report = json!({"passed":true,"suite":"ao","size":extent.width,
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver,"driver_info":info.driver_info},
        "seed":51,"manifest":manifest,"manifest_hash":hash.iter().map(|v|format!("{v:02x}")).collect::<String>(),
        "manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four",
        "checks":checks,"timing":timing,"maximum_quality":maximum,
        "worker":{"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,"cancellation_requests":stats.cancellation_requests,
            "cancelled_pass_batches":stats.cancelled_pass_batches,"validation_errors":stats.validation_errors,"peak_bytes":stats.peak_bytes},
        "shared_plan_peak_bytes":plan.peak_bytes(),
        "timing_scope":"Every AO sample and direction, CPU encoding, queue submission and completion wait; input is numeric height. No output cache or readback in timed samples. Preparation/compile is reported separately.",
        "accounting_scope":"Input/output R32 textures, uniform/sample buffers and all queue-write payloads admitted before allocation. Worker cancellation measured separately; sequential tiny oracle readbacks and driver heaps excluded."});
    if let Err(error) = validate_report(&report) {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> std::result::Result<serde_json::Value, String> {
    run(
        context.device.clone(),
        context.queue.clone(),
        context.extent,
        context.info,
    )
}
