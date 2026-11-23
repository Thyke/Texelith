//! T-048 native runner. Fixtures and f64 oracles are shared with this task's
//! tests only; timing always executes the production prepared node.
use super::contract::{aggregate_timings, validate_report};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, Extent, InputInterpretation, MapKind, NormalConvention, NormalFilter,
    SourceDescriptor,
};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, NormalParameters, Pass, PreparedNormal, Quality,
    TypedSourcePixels, WorkerEvent,
    pool::{AllocationDesc, FenceId, ResourcePool},
};
#[allow(dead_code, unused_imports)]
#[path = "../../../tests/mvp_t048/normal.rs"]
mod cases;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t048/gpu.rs"]
mod gpu;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t048/reference.rs"]
mod reference;
use crate::reference::abi;
use cases::{Controls, evaluate, install, install_f32, request, wait};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;
fn require(value: bool, error: &str) -> Result<()> {
    if value { Ok(()) } else { Err(error.into()) }
}

fn numerics(g: &gpu::Gpu) -> Result<Vec<Value>> {
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut run = |source: &SourceDescriptor, c: &Controls| {
        serial += 1;
        evaluate(g, &worker, request(source, serial, c), c.convention)
    };
    for (w, h, kind) in [
        (1, 1, "constant"),
        (1, 17, "ramp"),
        (17, 1, "ramp"),
        (9, 7, "constant"),
        (37, 23, "ramp"),
        (31, 17, "impulse"),
        (257, 193, "seeded"),
    ] {
        let e = Extent::new(w, h).map_err(|e| e.to_string())?;
        let data = match kind {
            "constant" => vec![-2.0; (w * h) as usize],
            "ramp" => reference::ramp(e, 0.5, -0.25),
            "impulse" => {
                let mut data = vec![0.0; (w * h) as usize];
                data[0] = 0.125;
                data
            }
            _ => reference::seeded(e),
        };
        let source = install_f32(&worker, e, data.clone());
        for filter in [
            NormalFilter::Central,
            NormalFilter::Sobel,
            NormalFilter::Scharr,
        ] {
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                    let c = Controls {
                        filter,
                        boundary,
                        convention,
                        ..Default::default()
                    };
                    let bytes = run(&source, &c);
                    reference::compare(
                        &gpu::decode(&bytes),
                        &reference::normals(e, &data, filter, boundary, c.strength, convention),
                    );
                    require(bytes == run(&source, &c), "nondeterministic cached normal")?;
                    if filter == NormalFilter::Central
                        && boundary == Boundary::Clamp
                        && convention == NormalConvention::OpenGl
                    {
                        let input = g.scalar(e, &data);
                        let output = g.texture(e, wgpu::TextureFormat::Rgba16Float);
                        let pass = g
                            .executor
                            .prepare(
                                Pass::Normal {
                                    strength: c.strength,
                                },
                                &input,
                                &output,
                                e,
                            )
                            .map_err(|e| e.to_string())?;
                        let mut encoder = g.device.create_command_encoder(&Default::default());
                        pass.encode(&mut encoder);
                        g.queue.submit([encoder.finish()]);
                        require(bytes == g.read(&output), "legacy central output changed")?;
                    }
                }
            }
        }
        let gl = Controls::default();
        let dx = Controls {
            convention: NormalConvention::DirectX,
            ..gl.clone()
        };
        let a = gpu::decode(&run(&source, &gl));
        let b = gpu::decode(&run(&source, &dx));
        require(
            a.iter()
                .zip(&b)
                .all(|(a, b)| a[0] == b[0] && a[1] == -b[1] && a[2] == b[2] && a[3] == b[3]),
            "GL/DX double flip",
        )?;
        require(
            b == gpu::decode(&run(&source, &dx)) && a == gpu::decode(&run(&source, &gl)),
            "cached convention changed",
        )?;
    }
    // Adjacent u16 input codes retain their slopes and agree with linear f32.
    let e = Extent::new(17, 13).unwrap();
    let codes: Vec<u16> = (0..e.width * e.height)
        .map(|i| 32760 + (i % 17) as u16)
        .collect();
    let floats: Vec<f32> = codes.iter().map(|v| f32::from(*v) / 65535.0).collect();
    let u16_source = install(&worker, e, TypedSourcePixels::HeightU16(codes.into()));
    let f32_source = install_f32(&worker, e, floats.clone());
    let c = Controls {
        boundary: Boundary::Clamp,
        ..Default::default()
    };
    let a = gpu::decode(&run(&u16_source, &c));
    reference::compare(
        &a,
        &reference::normals(e, &floats, c.filter, c.boundary, c.strength, c.convention),
    );
    require(a.iter().any(|n| n[0].abs() > 0.0001), "u16 slope lost")?;
    require(a == gpu::decode(&run(&f32_source, &c)), "u16/f32 mismatch")?;
    // Recover unnormalized slopes from isolated scales, then blend independently.
    let e = Extent::new(65, 49).unwrap();
    let data = reference::image(e, |x, y| {
        0.07 * ((x as f32 + 0.5) / 65.0 * std::f32::consts::TAU).sin()
            + 0.02 * ((y as f32 + 0.5) / 49.0 * std::f32::consts::TAU * 5.0).cos()
    });
    let source = install_f32(&worker, e, data);
    for filter in [NormalFilter::Sobel, NormalFilter::Scharr] {
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            let mut c = Controls {
                filter,
                boundary,
                weights: vec![0.0; 3],
                ..Default::default()
            };
            let mut basis = Vec::new();
            for scale in 0..3 {
                c.weights = vec![0.0; 3];
                c.weights[scale] = 1.0;
                basis.push(gpu::decode(&run(&source, &c)));
                require(
                    basis[scale].iter().any(|n| n[0].abs() + n[1].abs() > 0.05),
                    "scale ignored",
                )?;
            }
            require(
                basis[0]
                    .iter()
                    .zip(&basis[2])
                    .any(|(a, b)| (a[1] - b[1]).abs() > 0.01),
                "scale radii identical",
            )?;
            for weights in [[0.25, 0.5, -0.125], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]] {
                c.weights = weights.to_vec();
                let actual = gpu::decode(&run(&source, &c));
                let expected: Vec<_> = (0..actual.len())
                    .map(|i| {
                        let mut du = 0.0;
                        let mut dv = 0.0;
                        for s in 0..3 {
                            du += f64::from(weights[s]) * -basis[s][i][0] / basis[s][i][2];
                            dv += f64::from(weights[s]) * basis[s][i][1] / basis[s][i][2];
                        }
                        reference::unit(du, dv, 1.0, c.convention)
                    })
                    .collect();
                reference::compare(&actual, &expected);
            }
        }
    }
    drop(run);
    // Proxy comparison uses an analytic UV ramp far from the clamped edges.
    let e = Extent::new(2048, 128).unwrap();
    let source = install_f32(&worker, e, reference::ramp(e, 0.5, -0.25));
    for filter in [
        NormalFilter::Central,
        NormalFilter::Sobel,
        NormalFilter::Scharr,
    ] {
        for weights in [vec![1.0], vec![0.25, 0.5, 0.25]] {
            let c = Controls {
                filter,
                weights,
                boundary: Boundary::Clamp,
                ..Default::default()
            };
            serial += 1;
            let full = gpu::decode(&evaluate(
                g,
                &worker,
                request(&source, serial, &c),
                c.convention,
            ));
            serial += 1;
            let mut r = request(&source, serial, &c);
            r.quality = Quality::Proxy;
            r.extent = texelith_gpu::proxy_extent(e).unwrap();
            let proxy = gpu::decode(&evaluate(g, &worker, r, c.convention));
            for y in 12..52 {
                for x in (16..1008).step_by(31) {
                    let a = proxy[y * 1024 + x];
                    let b = full[2 * y * 2048 + 2 * x];
                    require(
                        (0..3).all(|i| (a[i] - b[i]).abs() <= 0.002),
                        "proxy UV scaling mismatch",
                    )?;
                    let dot = (0..3).map(|i| a[i] * b[i]).sum::<f64>();
                    let norm = |n: [f64; 4]| n[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
                    require(
                        (dot / (norm(a) * norm(b))).clamp(-1.0, 1.0).acos() < 0.005,
                        "proxy angular mismatch",
                    )?;
                }
            }
        }
    }
    require(
        worker.diagnostics().validation_errors == 0,
        "numerical GPU validation",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok([
        "legacy_central",
        "sobel",
        "scharr",
        "multi_scale",
        "gradient_before_normalize",
        "uv_ramp_y_sign",
        "unit_length",
        "gl_dx_metadata",
        "no_double_flip",
        "proxy_scaling",
        "linear_height_precision",
        "constant",
        "ramp",
        "impulse",
        "npot_257x193",
        "singleton_axes",
        "repeat_clamp",
        "determinism",
    ]
    .into_iter()
    .map(|name| json!({"name":name,"passed":true}))
    .collect())
}

fn runtime_checks(g: &gpu::Gpu) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let e = Extent::new(1024, 1024).unwrap();
    let source = install_f32(&worker, e, reference::seeded(e));
    let c = Controls {
        filter: NormalFilter::Scharr,
        weights: vec![0.25; 16],
        ..Default::default()
    };
    worker
        .request_material(request(&source, 1, &c))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().submitted < 2 {
        require(Instant::now() < deadline, "cancellation timeout")?;
        require(
            worker.try_event().is_none() && worker.try_latest_material().is_none(),
            "cancellation setup failed",
        )?;
        std::thread::yield_now();
    }
    let zero = Controls {
        weights: vec![0.0; 16],
        ..c
    };
    for serial in 2..=65 {
        worker
            .request_material(request(&source, serial, &zero))
            .map_err(|e| e.to_string())?;
    }
    let bundle = wait(|| {
        assert!(worker.try_event().is_none());
        worker.try_latest_material()
    });
    require(
        bundle.request_id == 65 && bundle.generation == 65,
        "stale normal published",
    )?;
    let actual = gpu::decode(&g.read(bundle.outputs[&MapKind::Normal].texture()));
    reference::compare(&actual, &vec![[0.0, 0.0, 1.0]; actual.len()]);
    let stats = worker.diagnostics();
    require(
        stats.cancelled_pass_batches > 0
            && stats.cancellation_requests > 0
            && stats.stale_publications == 0
            && stats.validation_errors == 0
            && stats.peak_in_flight == 1
            && stats.peak_pending == 1
            && stats.peak_bytes <= BUDGET,
        "invalid cancellation diagnostics",
    )?;
    drop(bundle);
    worker.join().map_err(|e| e.to_string())?;
    let worker = GpuWorker::spawn(
        g.device.clone(),
        g.queue.clone(),
        GpuConfig {
            budget_bytes: 600 * 1024,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let e = Extent::new(257, 193).unwrap();
    let source = install_f32(&worker, e, reference::seeded(e));
    worker
        .request_material(request(&source, 1, &zero))
        .map_err(|e| e.to_string())?;
    require(
        matches!(
            wait(|| worker.try_event()),
            WorkerEvent::BudgetExceeded { .. }
        ),
        "low budget was not rejected",
    )?;
    require(
        worker.try_latest_material().is_none() && worker.diagnostics().submitted == 0,
        "partial budget output",
    )?;
    let e = Extent::new(17, 13).unwrap();
    let data = reference::ramp(e, 0.5, 0.25);
    let source = install_f32(&worker, e, data.clone());
    let c = Controls::default();
    let actual = gpu::decode(&evaluate(g, &worker, request(&source, 2, &c), c.convention));
    reference::compare(
        &actual,
        &reference::normals(e, &data, c.filter, c.boundary, c.strength, c.convention),
    );
    require(
        worker.diagnostics().peak_bytes <= 600 * 1024,
        "low budget recovery exceeded budget",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(stats)
}

fn measure(g: &gpu::Gpu, node: &mut PreparedNormal, serial: &mut u64) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        // Every sample recomputes the whole requested pyramid and final gradient
        // blend. Neither output-cache nor pyramid-cache hits enter this timing.
        for pass in node.pyramid_passes() {
            pass.encode(&mut encoder);
        }
        node.encode(&mut encoder);
        *serial += 1;
        let fence = FenceId(*serial);
        node.mark_submitted(fence).map_err(|e| e.to_string())?;
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        g.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| e.to_string())?;
        node.complete(fence, true);
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","includes_pyramid":true,
        "warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,
        "submission_ms":submissions[10..].iter().sum::<f64>()/100.0,
        "pyramid_passes":node.pyramid_passes().len(),"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
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
    let stats = runtime_checks(&g)?;
    checks.push(json!({"name":"cancellation","passed":true}));
    checks.push(json!({"name":"low_budget","passed":true}));
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let desc = |format| AllocationDesc::Texture {
        extent,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
    };
    let mut ids = Vec::new();
    for d in [
        desc(wgpu::TextureFormat::R32Float),
        desc(wgpu::TextureFormat::Rgba16Float),
        AllocationDesc::Staging {
            bytes: u64::from(extent.width) * u64::from(extent.height) * 4,
        },
    ] {
        ids.push(
            pool.lock()
                .unwrap()
                .allocate(d)
                .map_err(|e| e.to_string())?,
        );
    }
    let data = reference::seeded(extent);
    let input = g.scalar(extent, &data);
    let source = TypedSourcePixels::HeightF32(data.into())
        .descriptor(extent)
        .map_err(|e| e.to_string())?;
    let output = g.texture(extent, wgpu::TextureFormat::Rgba16Float);
    g.wait();
    pool.lock()
        .unwrap()
        .release(ids.pop().unwrap())
        .map_err(|e| e.to_string())?;
    let mut params = NormalParameters {
        strength: 1.0,
        filter: NormalFilter::Sobel,
        scale_weights: vec![1.0],
        boundary: Boundary::Repeat,
        convention: NormalConvention::OpenGl,
    };
    let cold = Instant::now();
    let mut node =
        PreparedNormal::new(&g.executor, &input, &output, &source, &params, pool.clone())
            .map_err(|e| e.to_string())?;
    let cold_prepare_ms = cold.elapsed().as_secs_f64() * 1000.0;
    let mut serial = 0;
    let mut timing = measure(&g, &mut node, &mut serial)?;
    let default_peak = pool.lock().unwrap().diagnostics().peak_bytes;
    drop(node);
    params.filter = NormalFilter::Scharr;
    params.scale_weights = vec![0.25; 16];
    let mut node =
        PreparedNormal::new(&g.executor, &input, &output, &source, &params, pool.clone())
            .map_err(|e| e.to_string())?;
    let mut maximum = measure(&g, &mut node, &mut serial)?;
    maximum["filter"] = json!("scharr");
    maximum["scales"] = json!(16);
    maximum["scale_weights"] = json!(params.scale_weights);
    maximum["peak_bytes"] = json!(pool.lock().unwrap().diagnostics().peak_bytes);
    timing["peak_bytes"] = json!(default_peak.max(stats.peak_bytes));
    timing["cancellation_errors"] = json!(stats.stale_publications);
    drop(node);
    for id in ids {
        pool.lock()
            .unwrap()
            .release(id)
            .map_err(|e| e.to_string())?;
    }
    g.wait();
    require(
        pollster::block_on(scope.pop()).is_none(),
        "normal GPU validation error",
    )?;
    let manifest = json!({"filter":"sobel","scale_weights":[1.0],"strength":1.0,"boundary":"repeat","convention":"opengl","filter_revision":1,"size":extent.width,"seed":48});
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
    .content_hash;
    let mut result = json!({"passed":true,"suite":"normal","size":extent.width,"seed":48,
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver},
        "manifest":manifest,"manifest_hash":hash.iter().map(|b|format!("{b:02x}")).collect::<String>(),
        "manifest_hash_encoding":"BLAKE3 of compact JSON padded to four bytes","checks":checks,
        "timing":timing,"maximum_quality":maximum,"cold_prepare_ms":cold_prepare_ms,
        "normal":{"budget_bytes":BUDGET,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,
            "cancellation_requests":stats.cancellation_requests,"cancelled_pass_batches":stats.cancelled_pass_batches,
            "cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,"low_budget_rejections":1},
        "accounting_scope":"R32 input and upload staging, RGBA16 normal output, Gaussian levels/scratch, uniforms and staging; separate serial runtime peak. Driver allocations excluded.",
        "timing_scope":"CPU encoding, queue submission and completion of all requested Gaussian pyramid passes and final gradient blend. Source upload is a separate graph node."});
    if let Err(e) = validate_report(&result) {
        result["passed"] = json!(false);
        result["failure"] = json!(e);
    }
    Ok(result)
}

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> std::result::Result<serde_json::Value, String> {
    let legacy = crate::baseline(
        context.device.clone(),
        context.queue.clone(),
        context.extent,
        context.info,
    )?;
    let mut report = run(
        context.device.clone(),
        context.queue.clone(),
        context.extent,
        context.info,
    )?;
    report["legacy_baseline"] = legacy;
    Ok(report)
}
