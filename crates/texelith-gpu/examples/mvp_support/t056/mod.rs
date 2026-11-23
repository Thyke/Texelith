//! Serial numerical coverage and completion-bound, cold-replay full-node timing.
use crate::contract::{aggregate_timings, validate_report};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use texelith_core::{Boundary, BrushSample, BrushStroke, Extent, MapKind, MaskMode, stroke_digest};
use texelith_gpu::{PaintKernel, PreparedPaint, Quality, ResourceReservations, WorkerEvent};
#[allow(dead_code)]
#[path = "../../../tests/mvp_t056/reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t056/support.rs"]
mod support;
use support::{GIB, Gpu, dot, graph, request, request_graph, wait, with_boundary};
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn compare(a: &[f32], b: &[f64]) -> Result<f64> {
    require(!a.is_empty() && a.len() == b.len(), "paint dimensions")?;
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| {
            if a.is_finite() && (0.0..=1.0).contains(&a) {
                (f64::from(a) - b).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0., f64::max);
    require(error <= 2e-5, "paint CPU oracle mismatch")?;
    Ok(error)
}
fn numerics(g: &Gpu) -> Result<(f64, f64)> {
    let worker = g.worker(GIB);
    let mut id = 0;
    let mut error = 0.0f64;
    for e in [
        Extent::new(37, 29).unwrap(),
        Extent::new(257, 193).unwrap(),
        Extent::new(1, 19).unwrap(),
        Extent::new(23, 1).unwrap(),
        Extent::new(1, 1).unwrap(),
    ] {
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            for strokes in [
                vec![],
                vec![dot(1, [0.; 2], 0.181, 0.7, MaskMode::Add)],
                vec![
                    dot(1, [0.5; 2], 0.231, 0.6, MaskMode::Add),
                    dot(2, [0.5; 2], 0.231, 0.25, MaskMode::Subtract),
                    dot(3, [0.5; 2], 0.231, 0.8, MaskMode::Replace),
                ],
            ] {
                id += 1;
                let actual = g.evaluate(
                    &worker,
                    request_graph(
                        with_boundary(graph(e, &strokes), boundary),
                        id,
                        Quality::Full,
                    ),
                );
                error = error.max(compare(&actual, &reference::raster(e, &strokes, boundary))?);
            }
        }
    }
    let e = Extent::new(65, 65).unwrap();
    for opacity in [0., 0.5001, 1.] {
        for hardness in [0., 1.] {
            let mut s = dot(1, [0.5; 2], 0.241, opacity, MaskMode::Replace);
            s.brush.hardness = hardness;
            id += 1;
            let a = g.evaluate(&worker, request(e, &[s], id, Quality::Full));
            require(
                (a[32 * 65 + 32] - opacity).abs() < 2e-5,
                "paint hardness/opacity center",
            )?;
            require(
                a.iter()
                    .all(|v| v.is_finite() && (0.0..=opacity).contains(v)),
                "paint bounds",
            )?;
            for x in 32..64 {
                require(
                    a[32 * 65 + x] + 2e-5 >= a[32 * 65 + x + 1],
                    "paint radial monotonicity",
                )?;
            }
        }
    }
    let strokes = [
        dot(1, [0.07, 0.5], 0.181, 0.6, MaskMode::Replace),
        dot(2, [0.92, 0.5], 0.221, 0.2, MaskMode::Subtract),
    ];
    for q in [Quality::Proxy, Quality::Full] {
        id += 1;
        let r = request(Extent::new(2051, 7).unwrap(), &strokes, id, q);
        let e = r.extent;
        error = error.max(compare(
            &g.evaluate(&worker, r),
            &reference::raster(e, &strokes, Boundary::Repeat),
        )?);
    }
    // Twenty commits, undo, redo, then divergent suffix. Real Session commands
    // also have dedicated integration coverage; here retain immutable prefixes.
    let mut strokes = Vec::new();
    let mut snapshots = Vec::new();
    for n in 1..=20 {
        strokes.push(dot(
            n,
            [0.5; 2],
            0.181,
            if n % 2 == 0 { 0.1 } else { 0.2 },
            if n % 2 == 0 {
                MaskMode::Subtract
            } else {
                MaskMode::Add
            },
        ));
        id += 1;
        let actual = g.evaluate(&worker, request(e, &strokes, id, Quality::Full));
        error = error.max(compare(
            &actual,
            &reference::raster(e, &strokes, Boundary::Repeat),
        )?);
        snapshots.push(actual);
    }
    let mut replay_error = 0.0f64;
    for count in [9, 10, 9] {
        id += 1;
        let a = g.evaluate(&worker, request(e, &strokes[..count], id, Quality::Full));
        replay_error = replay_error.max(
            a.iter()
                .zip(&snapshots[count - 1])
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0., f64::max),
        );
    }
    strokes.truncate(9);
    strokes.push(dot(99, [0.2, 0.3], 0.12, 0.85, MaskMode::Replace));
    id += 1;
    let a = g.evaluate(&worker, request(e, &strokes, id, Quality::Full));
    error = error.max(compare(
        &a,
        &reference::raster(e, &strokes, Boundary::Repeat),
    )?);
    let digest = stroke_digest(&strokes).unwrap();
    let mut renamed = strokes.clone();
    renamed[0].id.0 = 123;
    require(
        digest == stroke_digest(&renamed).unwrap(),
        "paint digest edit IDs",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    let cold = g.worker(GIB);
    id += 1;
    require(
        a == g.evaluate(&cold, request(e, &strokes, id, Quality::Full)),
        "paint cold determinism",
    )?;
    cold.join().map_err(|e| e.to_string())?;
    require(replay_error <= 2e-5, "paint undo replay")?;
    Ok((error, replay_error))
}
fn cancellation(g: &Gpu) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = g.worker(GIB);
    let e = Extent::new(511, 257).unwrap();
    let mut long = dot(1, [0.5; 2], 0.02, 0.001, MaskMode::Add);
    long.points = (0..8192)
        .map(|i| BrushSample {
            uv: [0.5 + 0.15 * (i as f32 * 0.01).sin(), 0.5],
            pressure: 1.,
        })
        .collect();
    worker
        .request_material(request(e, &[long], 1, Quality::Full))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().submitted == 0 {
        if let Some(event) = worker.try_event() {
            return Err(format!("paint cancellation {event:?}"));
        }
        require(Instant::now() < deadline, "paint did not start")?;
        std::thread::yield_now();
    }
    let latest = [dot(2, [0.5; 2], 0.201, 0.7, MaskMode::Replace)];
    worker
        .request_material(request(e, &latest, 2, Quality::Full))
        .map_err(|e| e.to_string())?;
    let bundle = wait(|| worker.try_latest_material());
    require(bundle.request_id == 2, "stale paint published")?;
    compare(
        &g.read(bundle.outputs[&MapKind::Mask].texture()),
        &reference::raster(e, &latest, Boundary::Repeat),
    )?;
    let stats = worker.diagnostics();
    require(
        stats.cancelled_pass_batches > 0
            && stats.cancellation_requests > 0
            && stats.stale_publications == 0
            && stats.validation_errors == 0,
        "real paint cancellation not observed",
    )?;
    drop(bundle);
    worker.join().map_err(|e| e.to_string())?;
    Ok(stats)
}
fn budget(g: &Gpu) -> Result<()> {
    let worker = g.worker(65536);
    let e = Extent::new(31, 17).unwrap();
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 65536,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    worker
        .request_material(request(e, &[], 1, Quality::Full))
        .map_err(|e| e.to_string())?;
    require(
        matches!(
            wait(|| worker.try_event()),
            WorkerEvent::BudgetExceeded { .. }
        ),
        "paint low budget",
    )?;
    require(
        worker.diagnostics().submitted == 0,
        "budget rejected after dispatch",
    )?;
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    require(
        g.evaluate(&worker, request(e, &[], 2, Quality::Full))
            .iter()
            .all(|v| *v == 0.),
        "budget recovery",
    )?;
    worker.join().map_err(|e| e.to_string())
}
fn texture(g: &Gpu, e: Extent) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paint timing"),
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
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
fn submit(g: &Gpu, encode: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<f64> {
    let mut encoder = g.device.create_command_encoder(&Default::default());
    encode(&mut encoder);
    let now = Instant::now();
    g.queue.submit([encoder.finish()]);
    let ms = now.elapsed().as_secs_f64() * 1000.;
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })
        .map_err(|e| e.to_string())?;
    Ok(ms)
}
fn measure(g: &Gpu, p: &PreparedPaint, strokes: &[BrushStroke]) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submission_ms = 0.;
    for sample in 0..110 {
        let now = Instant::now();
        let mut submit_ms = submit(g, |e| p.initialize(&g.queue, e, false))?;
        for stroke in strokes {
            let mut offset = 0;
            loop {
                submit_ms += submit(g, |e| {
                    offset = p.batch(&g.queue, e, stroke, Boundary::Repeat, offset)
                })?;
                if offset >= stroke.points.len() {
                    break;
                }
            }
        }
        submit_ms += submit(g, |e| p.finish(&g.queue, e))?;
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.]);
        if sample >= 10 {
            submission_ms += submit_ms;
        }
    }
    let timing = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submission_ms/100.,"samples_ms":rows[10..].iter().map(|v|v[0]).collect::<Vec<_>>() }),
    )
}
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (scalar_error, replay_error) = numerics(&g)?;
    budget(&g)?;
    let stats = cancellation(&g)?;
    let e = context.extent;
    let pixels = u64::from(e.width) * u64::from(e.height);
    let direct_bytes = pixels * 12
        + u64::from((e.width * 4).div_ceil(256) * 256) * u64::from(e.height)
        + 16384
        + 132;
    let mut ledger = texelith_gpu::pool::ResourcePool::new(GIB);
    let reservation = ledger
        .allocate(texelith_gpu::pool::AllocationDesc::Staging {
            bytes: direct_bytes,
        })
        .map_err(|e| e.to_string())?;
    let input = texture(&g, Extent::new(1, 1).unwrap());
    let output = texture(&g, e);
    let kernel = PaintKernel::new(&g.device);
    let p = kernel
        .prepare(&g.device, &input, &output)
        .map_err(|e| e.to_string())?;
    // Default workload: twenty soft strokes, 1,024 valid samples in total.
    let mut strokes = Vec::new();
    for n in 0..20 {
        let mut s = dot(
            n + 1,
            [0.2 + (n % 5) as f32 * 0.14, 0.2 + (n / 5) as f32 * 0.14],
            0.02,
            0.75,
            MaskMode::Add,
        );
        s.brush.hardness = 0.5;
        let count = if n < 4 { 52 } else { 51 };
        let uv = s.points[0].uv;
        s.points = (0..count)
            .map(|i| BrushSample {
                uv: [uv[0] + i as f32 * 0.0005, uv[1]],
                pressure: 1.,
            })
            .collect();
        strokes.push(s);
    }
    let mut timing = measure(&g, &p, &strokes)?;
    // Maximum sample-count stress: 100,000 samples, small-radius curved path.
    // This measures the supported point cap, not every possible worst geometry.
    let mut max = dot(1, [0.5; 2], 0.002, 0.75, MaskMode::Add);
    max.brush.hardness = 0.5;
    max.points = (0..100000)
        .map(|i| BrushSample {
            uv: [
                0.5 + 0.001 * (i as f32 * 0.01).sin(),
                0.5 + 0.001 * (i as f32 * 0.01).cos(),
            ],
            pressure: 1.,
        })
        .collect();
    let mut maximum = measure(&g, &p, &[max])?;
    maximum["point_count"] = json!(100000);
    maximum["radius_uv"] = json!(0.002);
    // Direct harness reservation includes padded readback. Worker peak includes
    // checkpoints and was measured in a separate, completed phase.
    let peak = direct_bytes.max(stats.peak_bytes);
    require(peak <= GIB, "paint harness budget")?;
    timing["peak_bytes"] = json!(peak);
    timing["cancellation_errors"] = json!(stats.stale_publications);
    timing["cache_hits_used_for_timing"] = json!(0);
    let (hits, bytes, segments) = texelith_gpu::paint_mask::diagnostics();
    require(
        hits > 0 && segments > 0 && segments <= 256,
        "paint bounded batches/checkpoints",
    )?;
    let values = g.read(&output);
    require(
        values
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && values.iter().any(|v| *v > 0.),
        "maximum paint output is invalid",
    )?;
    drop((p, kernel, input, output));
    ledger.release(reservation).map_err(|e| e.to_string())?;
    require(
        ledger.diagnostics().allocated_bytes == 0,
        "paint harness reservation leaked",
    )?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "paint GPU validation error",
    )?;
    let digest = stroke_digest(&strokes)
        .map_err(|e| e.to_string())?
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let manifest = json!({"boundary":"Repeat","radius_uv":0.02,"hardness":0.5,"opacity":0.75,"mode":"Add","stroke_digest":digest,"stroke_count":20,"point_count":1024,"algorithm_revision":1});
    let mut encoded = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !encoded.len().is_multiple_of(4) {
        encoded.push(0);
    }
    let hash = texelith_core::SourceDescriptor::from_rgba8(
        &encoded,
        Extent::new((encoded.len() / 4) as u32, 1).unwrap(),
        texelith_core::InputInterpretation::LinearHeight,
    )
    .unwrap()
    .content_hash
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    let mut report = json!({"passed":true,"suite":"paint","size":e.width,"seed":56,"adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend)},"manifest":manifest,"manifest_hash":hash,"checks":(["cpu_oracle", "stroke_order", "opacity_hardness_endpoints", "wrapped_edge", "clamp_edge", "undo_replay", "proxy_full_alignment", "stable_digest", "npot", "singleton_axes", "cancellation", "low_budget", "bounded_batches"].iter().map(|n|json!({"name":n,"passed":true})).collect::<Vec<_>>()),"timing":timing,"maximum_quality":maximum,
        "paint":{"budget_bytes":GIB,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,"cancellation_requests":stats.cancellation_requests,"cancelled_pass_batches":stats.cancelled_pass_batches,"cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,"low_budget_rejections":1,"max_segments_per_batch":256,"peak_uploaded_segments":segments,"checkpoint_hits":hits,"checkpoint_bytes":bytes,"replay_max_error":replay_error,"scalar_max_error":scalar_error}});
    if let Err(error) = validate_report(&report) {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
