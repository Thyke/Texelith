//! T-053 native numerical and full-node measurement runner.
use crate::contract::{aggregate_timings, validate_roughness_report};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{Extent, InputInterpretation, MaterialNodeParams as P, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, MaterialRequest, Pass, PreparedRoughness,
    ResourceReservations, TypedSourcePixels, TypedSourceUpload, WorkerEvent,
    pool::{AllocationDesc, FenceId, ResourcePool},
};
#[allow(dead_code)]
#[path = "../../../tests/mvp_t053/gpu.rs"]
mod gpu;
#[allow(dead_code, unused_imports)]
#[path = "../../../tests/mvp_t053/material.rs"]
mod material;
#[allow(dead_code)]
mod oracle;
use material::{IDENTITY, request};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn wait<T>(mut f: impl FnMut() -> Option<T>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(result) = f() {
            return Ok(result);
        }
        if Instant::now() > deadline {
            return Err("roughness worker timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn install(worker: &GpuWorker, extent: Extent, bytes: Vec<u8>) -> Result<SourceDescriptor> {
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: bytes.into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    worker
        .install_typed_source(TypedSourceUpload {
            descriptor: descriptor.clone(),
            pixels,
        })
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event())?, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
        "source install failed",
    )?;
    Ok(descriptor)
}
fn evaluate(g: &gpu::Gpu, worker: &GpuWorker, r: MaterialRequest) -> Result<Vec<f32>> {
    let id = r.request_id;
    let extent = r.extent;
    worker.request_material(r).map_err(|e| e.to_string())?;
    let bundle = wait(|| worker.try_latest_material())?;
    if let Some(event) = worker.try_event() {
        return Err(format!("worker error {event:?}"));
    }
    require(
        bundle.request_id == id && bundle.extent == extent,
        "wrong material identity/extent",
    )?;
    Ok(g.read_scalar(bundle.outputs[&texelith_core::MapKind::Roughness].texture()))
}
fn compare(a: &[f32], b: &[f64]) -> Result<()> {
    require(
        !a.is_empty()
            && a.len() == b.len()
            && a.iter().zip(b).all(|(a, b)| {
                a.is_finite() && (0.0..=1.0).contains(a) && (f64::from(*a) - b).abs() <= 2e-5
            }),
        "roughness CPU oracle mismatch",
    )
}
fn check(checks: &mut Vec<Value>, name: &str) {
    checks.push(json!({"name":name,"passed":true}));
}
fn prepared(
    g: &gpu::Gpu,
    input: &wgpu::Texture,
    mask: Option<&wgpu::Texture>,
    params: &P,
) -> Result<Vec<f32>> {
    let e = Extent::new(input.width(), input.height()).map_err(|e| e.to_string())?;
    let output = g.texture(e, wgpu::TextureFormat::R32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut node = PreparedRoughness::new(&g.executor, input, mask, &output, params, pool.clone())
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    for pass in node.passes() {
        pass.encode(&mut encoder);
    }
    node.mark_submitted(FenceId(1)).map_err(|e| e.to_string())?;
    g.queue.submit([encoder.finish()]);
    g.wait();
    node.complete(FenceId(1));
    let values = g.read_scalar(&output);
    drop(node);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "roughness scratch leak",
    )?;
    Ok(values)
}
fn numerics(g: &gpu::Gpu) -> Result<Vec<Value>> {
    let mut checks = Vec::new();
    let e = Extent::new(9, 1).unwrap();
    let samples: Vec<u16> = (0..9)
        .flat_map(|i| {
            let bits = [
                0, 0x3000, 0x3400, 0x3600, 0x3800, 0x3900, 0x3a00, 0x3b00, 0x3c00,
            ][i];
            [bits, bits, bits, 0x3c00]
        })
        .collect();
    let input = g.upload(
        e,
        wgpu::TextureFormat::Rgba16Float,
        bytemuck::cast_slice(&samples),
    );
    let luma: Vec<_> = (0..9).map(|i| f64::from(i) / 8.0).collect();
    let masks = [0.0f32, 1.0, 0.25, 0.75, 0.5, 0.0, 1.0, 0.25, 0.75];
    let mask = g.upload(
        e,
        wgpu::TextureFormat::R32Float,
        bytemuck::cast_slice(&masks),
    );
    let ones = g.upload(
        e,
        wgpu::TextureFormat::R32Float,
        bytemuck::cast_slice(&[1.0f32; 9]),
    );
    let curve = [[0.25, 0.75], [0.5, 0.25], [0.75, 1.0]];
    for invert in [false, true] {
        let params = P::Roughness {
            curve: curve.to_vec(),
            base: 0.375,
            invert,
        };
        let a = prepared(g, &input, Some(&mask), &params)?;
        compare(
            &a,
            &oracle::roughness(&luma, Some(&masks), &curve, 0.375, invert),
        )?;
        let missing = prepared(g, &input, None, &params)?;
        compare(
            &missing,
            &oracle::roughness(&luma, None, &curve, 0.375, invert),
        )?;
        let explicit = prepared(g, &input, Some(&ones), &params)?;
        require(missing == explicit, "missing mask is not one")?;
    }
    for name in [
        "knot_interpolation",
        "curve_endpoints",
        "mask_zero_one_blend",
        "optional_mask_one",
        "invert_after_blend",
    ] {
        check(&mut checks, name);
    }
    let r = prepared(
        g,
        &input,
        None,
        &P::Roughness {
            curve: IDENTITY.to_vec(),
            base: 0.5,
            invert: false,
        },
    )?;
    let s = prepared(
        g,
        &input,
        None,
        &P::Roughness {
            curve: IDENTITY.to_vec(),
            base: 0.5,
            invert: true,
        },
    )?;
    require(
        r.iter().zip(&s).all(|(r, s)| *s == 1.0 - *r),
        "inversion not exact complement",
    )?;
    let twice = prepared(
        g,
        &input,
        None,
        &P::Roughness {
            curve: vec![[0.0, 1.0], [1.0, 0.0]],
            base: 0.5,
            invert: true,
        },
    )?;
    compare(&twice, &luma)?;
    check(&mut checks, "invert_twice_identity");
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [10, 10, 10],
        [11, 11, 11],
        [128, 128, 128],
        [255, 255, 255],
        [0, 0, 0],
    ];
    let bytes: Vec<u8> = colors
        .into_iter()
        .flat_map(|[r, g, b]| [[r, g, b, 0], [r, g, b, 255]])
        .flatten()
        .collect();
    let expected = oracle::luminance(&bytes);
    let source = install(&worker, Extent::new(16, 1).unwrap(), bytes)?;
    let actual = evaluate(g, &worker, request(&source, 1, &IDENTITY, 0.5, false))?;
    compare(&actual, &expected)?;
    require(
        actual
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| v[0].to_bits() == v[1].to_bits()),
        "alpha affected roughness",
    )?;
    require(
        (actual[10] - 128.0 / 255.0).abs() > 0.1,
        "accidental srgb output",
    )?;
    for name in ["linear_rec709", "alpha_ignored", "no_srgb_encode"] {
        check(&mut checks, name);
    }
    let mut serial = 1;
    let curve: Vec<_> = (0..16)
        .map(|i| [i as f32 / 15.0, if i % 2 == 0 { 0.125 } else { 0.875 }])
        .collect();
    for (w, h, kind) in [
        (1, 1, "constant"),
        (9, 7, "constant"),
        (1, 17, "singleton_axes"),
        (17, 1, "singleton_axes"),
        (31, 17, "ramp"),
        (31, 17, "impulse"),
        (257, 193, "npot_257x193"),
    ] {
        let mut state = 53u32;
        let bytes: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = match kind {
                    "constant" => 128,
                    "ramp" => (i * 255 / (w * h - 1).max(1)) as u8,
                    "impulse" => {
                        if i == w * h - 1 {
                            255
                        } else {
                            0
                        }
                    }
                    _ => (state >> 24) as u8,
                };
                [v, v, v, 255]
            })
            .collect();
        let expected = oracle::luminance(&bytes);
        let source = install(&worker, Extent::new(w, h).unwrap(), bytes)?;
        serial += 1;
        let r = request(&source, serial, &curve, 0.375, true);
        let a = evaluate(g, &worker, r.clone())?;
        compare(&a, &oracle::roughness(&expected, None, &curve, 0.375, true))?;
        let before = worker.diagnostics();
        serial += 1;
        let b = evaluate(g, &worker, request(&source, serial, &curve, 0.375, true))?;
        require(
            a == b
                && worker.diagnostics().cache_hits > before.cache_hits
                && worker.diagnostics().submitted == before.submitted,
            "determinism/cache miss",
        )?;
        serial += 1;
        let edited = evaluate(
            g,
            &worker,
            request(&source, serial, &IDENTITY, 0.375, false),
        )?;
        compare(&edited, &expected)?;
        serial += 1;
        require(
            a == evaluate(g, &worker, request(&source, serial, &curve, 0.375, true))?,
            "restored output changed",
        )?;
        check(&mut checks, kind);
    }
    for name in [
        "sixteen_knots",
        "bounded_finite",
        "determinism",
        "cache_invalidation",
    ] {
        check(&mut checks, name);
    }
    require(
        worker.diagnostics().validation_errors == 0,
        "worker validation errors",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(checks)
}

fn budget(g: &gpu::Gpu) -> Result<()> {
    let worker = GpuWorker::spawn(
        g.device.clone(),
        g.queue.clone(),
        GpuConfig {
            budget_bytes: 65536,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let source = install(
        &worker,
        Extent::new(16, 16).unwrap(),
        vec![128; 16 * 16 * 4],
    )?;
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 60 * 1024,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    worker
        .request_material(request(&source, 1, &IDENTITY, 0.5, false))
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event())?,WorkerEvent::BudgetExceeded{required_bytes,budget_bytes:65536} if required_bytes>65536),
        "budget rejection missing",
    )?;
    require(
        worker.try_latest_material().is_none(),
        "partial budget publication",
    )?;
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    compare(
        &evaluate(g, &worker, request(&source, 2, &IDENTITY, 0.5, false))?,
        &oracle::luminance(&vec![128; 16 * 16 * 4]),
    )?;
    require(
        worker.diagnostics().peak_bytes <= 65536,
        "low budget overshoot",
    )?;
    worker.join().map_err(|e| e.to_string())
}

fn cancellation(g: &gpu::Gpu, extent: Extent) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let source = install(
        &worker,
        extent,
        vec![128; (extent.width * extent.height * 4) as usize],
    )?;
    worker
        .request_material(request(&source, 1, &IDENTITY, 0.5, false))
        .map_err(|e| e.to_string())?;
    // Observe a real active worker generation before replacing it. A single-pass
    // roughness kernel can finish atomically, but stale publication must be fenced.
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().scheduled == 0 {
        if Instant::now() > deadline {
            return Err("worker did not start cancellation fixture".into());
        }
        std::thread::yield_now();
    }
    let values = evaluate(g, &worker, request(&source, 2, &IDENTITY, 0.5, true))?;
    compare(
        &values,
        &oracle::roughness(
            &oracle::luminance(&[128, 128, 128, 128]),
            None,
            &IDENTITY,
            0.5,
            true,
        )
        .repeat(values.len()),
    )?;
    let stats = worker.diagnostics();
    require(
        stats.cancellation_requests > 0
            && stats.stale_publications == 0
            && stats.peak_in_flight == 1
            && stats.validation_errors == 0,
        "worker cancellation not exercised",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(stats)
}
fn measure(g: &gpu::Gpu, node: &mut PreparedRoughness, serial: &mut u64) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        for pass in node.passes() {
            pass.encode(&mut encoder);
        }
        *serial += 1;
        let fence = FenceId(*serial);
        node.mark_submitted(fence).map_err(|e| e.to_string())?;
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        g.wait();
        node.complete(fence);
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
) -> Result<Value> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let g = gpu::Gpu {
        executor: pollster::block_on(GpuExecutor::new(device.clone(), queue.clone()))
            .map_err(|e| e.to_string())?,
        device,
        queue,
    };
    let mut checks = numerics(&g)?;
    budget(&g)?;
    check(&mut checks, "low_budget");
    let stats = cancellation(&g, extent)?;
    check(&mut checks, "cancellation");
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let pixels = u64::from(extent.width) * u64::from(extent.height);
    // Admit every harness-owned texture and all upload/readback bytes before allocation.
    let mut payload = Vec::new();
    for bytes in [
        pixels * 4,
        pixels * 8,
        pixels * 4,
        pixels * 4,
        pixels * 8 + 128,
    ] {
        payload.push(
            pool.lock()
                .unwrap()
                .allocate(AllocationDesc::Staging { bytes })
                .map_err(|e| e.to_string())?,
        );
    }
    let bytes: Vec<u8> = (0..pixels)
        .flat_map(|i| {
            let v = (i.wrapping_mul(53) % 256) as u8;
            [v, v, v, 255]
        })
        .collect();
    let raw = g.upload(extent, wgpu::TextureFormat::Rgba8Unorm, &bytes);
    let color = g.texture(extent, wgpu::TextureFormat::Rgba16Float);
    let output = g.texture(extent, wgpu::TextureFormat::R32Float);
    let masks: Vec<f32> = (0..pixels).map(|i| (i % 5) as f32 * 0.25).collect();
    let mask = g.upload(
        extent,
        wgpu::TextureFormat::R32Float,
        bytemuck::cast_slice(&masks),
    );
    let conversion = g
        .executor
        .prepare(Pass::ColorSource, &raw, &color, extent)
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    conversion.encode(&mut encoder);
    g.queue.submit([encoder.finish()]);
    g.wait();
    let mut node = PreparedRoughness::new(
        &g.executor,
        &color,
        None,
        &output,
        &P::Roughness {
            curve: IDENTITY.to_vec(),
            base: 0.5,
            invert: false,
        },
        pool.clone(),
    )
    .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut timing = measure(&g, &mut node, &mut serial)?;
    compare(&g.read_scalar(&output), &oracle::luminance(&bytes))?;
    drop(node);
    let curve: Vec<_> = (0..16)
        .map(|i| [i as f32 / 15.0, if i % 2 == 0 { 0.125 } else { 0.875 }])
        .collect();
    let mut node = PreparedRoughness::new(
        &g.executor,
        &color,
        Some(&mask),
        &output,
        &P::Roughness {
            curve: curve.clone(),
            base: 0.5,
            invert: true,
        },
        pool.clone(),
    )
    .map_err(|e| e.to_string())?;
    let mut maximum = measure(&g, &mut node, &mut serial)?;
    maximum["knot_count"] = json!(16);
    compare(
        &g.read_scalar(&output),
        &oracle::roughness(&oracle::luminance(&bytes), Some(&masks), &curve, 0.5, true),
    )?;
    let peak = pool
        .lock()
        .unwrap()
        .diagnostics()
        .peak_bytes
        .max(stats.peak_bytes);
    timing["peak_bytes"] = json!(peak);
    timing["cancellation_errors"] = json!(stats.stale_publications);
    drop(node);
    drop(conversion);
    drop((raw, color, output, mask));
    for id in payload {
        pool.lock()
            .unwrap()
            .release(id)
            .map_err(|e| e.to_string())?;
    }
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "harness resources leaked",
    )?;
    g.wait();
    require(
        pollster::block_on(scope.pop()).is_none(),
        "roughness GPU validation error",
    )?;
    let manifest = json!({"curve":IDENTITY,"base":0.5,"invert":false,"mask":"constant_one","seed":53,"size":extent.width});
    let mut encoded = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !encoded.len().is_multiple_of(4) {
        encoded.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &encoded,
        Extent::new((encoded.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    let mut report = json!({"passed":true,"suite":"roughness","size":extent.width,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"seed":53,"manifest":manifest,"manifest_hash":hash,"manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four","checks":checks,"timing":timing,"maximum_quality":maximum,"roughness":{"budget_bytes":BUDGET,"peak_in_flight":stats.peak_in_flight,"cancellation_requests":stats.cancellation_requests,"cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,"low_budget_rejections":1},"accounting_scope":"source, color, output, explicit mask, scratch, uploads and padded scalar readback; excludes driver allocations; full-node starts at typed linear color"});
    if let Err(error) = validate_roughness_report(&report) {
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
