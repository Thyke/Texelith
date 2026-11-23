//! T-054 numerical contracts and completion-bound full-node timings.
use crate::contract::{aggregate_timings, validate_report};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, Pass, Quality, ResourceReservations, TypedSourcePixels,
    WorkerEvent,
    metallic::MetallicKernel,
    pool::{AllocationDesc, ResourcePool},
};
#[allow(dead_code)]
#[path = "../../../tests/mvp_t054/reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t054/support.rs"]
mod support;
use reference::{Combine, Controls};
use support::{GIB, Gpu, request};
type Result<T> = std::result::Result<T, String>;

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn check(checks: &mut Vec<Value>, name: &str) {
    checks.push(json!({"name":name,"passed":true}));
}
fn compare(a: &[f32], b: &[f64]) -> Result<()> {
    require(
        !a.is_empty()
            && a.len() == b.len()
            && a.iter().zip(b).all(|(&a, &b)| {
                a.is_finite() && (0.0..=1.0).contains(&a) && (f64::from(a) - b).abs() <= 2e-5
            }),
        "metallic CPU oracle mismatch",
    )
}
fn wait<T>(mut next: impl FnMut() -> Option<T>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(v) = next() {
            return Ok(v);
        }
        if Instant::now() >= deadline {
            return Err("metallic worker timed out".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn texture(g: &Gpu, e: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("T-054 fixture"),
        size: wgpu::Extent3d {
            width: e.width,
            height: e.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
fn upload(g: &Gpu, t: &wgpu::Texture, bytes: &[u8], stride: u32) {
    g.queue.write_texture(
        t.as_image_copy(),
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(t.width() * stride),
            rows_per_image: Some(t.height()),
        },
        t.size(),
    );
}
fn uniform(g: &Gpu) -> wgpu::Buffer {
    g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-054 uniform"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn finish(g: &Gpu) -> Result<()> {
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}
fn evaluate(
    g: &Gpu,
    worker: &GpuWorker,
    source: &SourceDescriptor,
    id: u64,
    c: Controls,
    q: Quality,
) -> Result<Vec<f32>> {
    let r = request(source, id, c, q);
    let extent = r.extent;
    worker.request_material(r).map_err(|e| e.to_string())?;
    let bundle = wait(|| {
        if let Some(event) = worker.try_event() {
            return Some(Err(format!("worker error: {event:?}")));
        }
        worker
            .try_latest_material()
            .filter(|b| b.request_id == id)
            .map(Ok)
    })??;
    require(
        bundle.request_id == id
            && bundle.generation == id
            && bundle.extent == extent
            && bundle.quality == q,
        "wrong material identity/extent",
    )?;
    Ok(g.read(bundle.outputs[&MapKind::Metallic].texture()))
}
fn photo(g: &Gpu, worker: &GpuWorker, e: Extent, bytes: Vec<u8>) -> SourceDescriptor {
    g.install(
        worker,
        e,
        TypedSourcePixels::Rgba8 {
            pixels: bytes.into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
}

fn numerics(g: &Gpu, kernel: &MetallicKernel) -> Result<Vec<Value>> {
    let mut checks = Vec::new();
    // Exactly representable half-float input tests the threshold equalities.
    let e = Extent::new(8, 1).unwrap();
    let color = texture(g, e, wgpu::TextureFormat::Rgba16Float);
    let output = texture(g, e, wgpu::TextureFormat::R32Float);
    let mask = texture(g, e, wgpu::TextureFormat::R32Float);
    let zero_mask = texture(g, e, wgpu::TextureFormat::R32Float);
    let one_mask = texture(g, e, wgpu::TextureFormat::R32Float);
    upload(g, &one_mask, bytemuck::cast_slice(&[1.0f32; 8]), 4);
    let halves = [0, 0x3000, 0x3400, 0x3600, 0x3800, 0x3900, 0x3a00, 0x3c00];
    let colors: Vec<u16> = halves.into_iter().flat_map(|r| [r, 0, 0, 0x3c00]).collect();
    upload(g, &color, bytemuck::cast_slice(&colors), 8);
    let masks = [0.0f32, 0.1234567, 0.5, 0.9999, 1.0, 0.25, 0.75, 1.0];
    upload(g, &mask, bytemuck::cast_slice(&masks), 4);
    let uniform = uniform(g);
    let base = Controls {
        picked: [0.; 3],
        radius: 0.25,
        feather: 0.5,
        mode: Combine::Add,
        ..Default::default()
    };
    for (c, expected) in [
        (base, vec![1., 1., 1., 1., 1., 0., 0., 0.]),
        (
            Controls {
                threshold: 0.5001,
                ..base
            },
            vec![1., 1., 1., 1., 0., 0., 0., 0.],
        ),
        (
            Controls {
                threshold: 1.,
                ..base
            },
            vec![1., 1., 1., 0., 0., 0., 0., 0.],
        ),
        (
            Controls {
                threshold: 0.,
                ..base
            },
            vec![1.; 8],
        ),
        (
            Controls {
                feather: 0.,
                ..base
            },
            vec![1., 1., 1., 0., 0., 0., 0., 0.],
        ),
        (
            Controls {
                radius: 0.,
                feather: 0.,
                ..base
            },
            vec![1., 0., 0., 0., 0., 0., 0., 0.],
        ),
        (
            Controls {
                feather: f32::MAX,
                ..base
            },
            vec![1.; 8],
        ),
    ] {
        let pass = kernel
            .prepare(
                &g.device,
                &color,
                &output,
                Some(&zero_mask),
                &uniform,
                &c.params(),
            )
            .map_err(|e| e.to_string())?;
        let mut encoder = g.device.create_command_encoder(&Default::default());
        pass.encode(&mut encoder);
        g.queue.submit([encoder.finish()]);
        compare(&g.read(&output), &expected)?;
    }
    for name in [
        "picked_center",
        "radius_equality",
        "threshold_equality",
        "threshold_endpoints",
        "zero_feather",
        "zero_radius",
        "extreme_feather",
    ] {
        check(&mut checks, name);
    }
    for mode in [Combine::Replace, Combine::Add, Combine::Subtract] {
        let c = Controls { mode, ..base };
        let mut absent = Vec::new();
        for painted in [None, Some(&one_mask)] {
            let pass = kernel
                .prepare(&g.device, &color, &output, painted, &uniform, &c.params())
                .map_err(|e| e.to_string())?;
            let mut encoder = g.device.create_command_encoder(&Default::default());
            pass.encode(&mut encoder);
            g.queue.submit([encoder.finish()]);
            let actual = g.read(&output);
            compare(
                &actual,
                &[if matches!(mode, Combine::Subtract) {
                    0.0
                } else {
                    1.0
                }; 8],
            )?;
            if painted.is_none() {
                absent = actual;
            } else {
                require(actual == absent, "absent mask differs from ConstantOne")?;
            }
        }
        for painted in [None, Some(&mask)] {
            let pass = kernel
                .prepare(&g.device, &color, &output, painted, &uniform, &c.params())
                .map_err(|e| e.to_string())?;
            let mut encoder = g.device.create_command_encoder(&Default::default());
            pass.encode(&mut encoder);
            g.queue.submit([encoder.finish()]);
            let expected: Vec<_> = [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.]
                .into_iter()
                .enumerate()
                .map(|(i, r)| reference::metallic([r, 0., 0.], painted.map(|_| masks[i]), c))
                .collect();
            compare(&g.read(&output), &expected)?;
        }
    }
    for name in [
        "mask_replace",
        "mask_add",
        "mask_subtract",
        "fractional_mask",
        "absent_mask_constant_one",
    ] {
        check(&mut checks, name);
    }
    let worker = g.worker(GIB);
    let mut id = 0;
    for (w, h, kind, name) in [
        (1, 1, 0, "constant"),
        (17, 13, 0, "constant"),
        (31, 19, 1, "ramp"),
        (17, 13, 2, "impulse"),
        (257, 193, 3, "npot_257x193"),
        (1, 31, 3, "singleton_axes"),
        (31, 1, 3, "singleton_axes"),
    ] {
        let bytes = reference::photo(w, h, kind);
        let source = photo(g, &worker, Extent::new(w, h).unwrap(), bytes.clone());
        for c in [
            Controls::default(),
            base,
            Controls {
                picked: [1.; 3],
                radius: 0.75,
                threshold: 0.75,
                ..base
            },
        ] {
            id += 1;
            let a = evaluate(g, &worker, &source, id, c, Quality::Full)?;
            compare(&a, &reference::photo_reference(&bytes, c))?;
            id += 1;
            let before = worker.diagnostics();
            let b = evaluate(g, &worker, &source, id, c, Quality::Full)?;
            require(
                a == b
                    && worker.diagnostics().submitted == before.submitted
                    && worker.diagnostics().cache_hits > before.cache_hits,
                "determinism/cache reuse failed",
            )?;
        }
        check(&mut checks, name);
    }
    // Black, white and each primary prove that the metric uses all RGB lanes.
    let bytes = vec![
        0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 1, 0, 255, 0, 128, 0, 0, 255, 254,
    ];
    for (i, picked) in [[0.; 3], [1.; 3], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        .into_iter()
        .enumerate()
    {
        let c = Controls {
            picked,
            radius: 0.,
            feather: 0.,
            ..base
        };
        compare(
            &g.photo_selection(Extent::new(5, 1).unwrap(), &bytes, c),
            &(0..5)
                .map(|j| if j == i { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        )?;
    }
    let bytes = vec![
        128, 128, 128, 0, 128, 128, 128, 255, 0, 0, 0, 0, 255, 255, 255, 255,
    ];
    let c = Controls {
        picked: [reference::linear_half(128) as f32; 3],
        radius: 0.,
        feather: 0.,
        ..base
    };
    compare(
        &g.photo_selection(Extent::new(4, 1).unwrap(), &bytes, c),
        &[1., 1., 0., 0.],
    )?;
    compare(
        &g.photo_selection(
            Extent::new(4, 1).unwrap(),
            &bytes,
            Controls {
                picked: [128. / 255.; 3],
                ..c
            },
        ),
        &[0.; 4],
    )?;
    let e = Extent::new(2048, 32).unwrap();
    let bytes = (0..e.width * e.height)
        .flat_map(|i| {
            if (i % e.width / 16).is_multiple_of(2) {
                [0, 0, 0, 255]
            } else {
                [255; 4]
            }
        })
        .collect();
    let source = photo(g, &worker, e, bytes);
    id += 1;
    let proxy = texelith_gpu::proxy_extent(e).unwrap();
    let expected = vec![1.0; (proxy.width * proxy.height) as usize];
    compare(
        &evaluate(
            g,
            &worker,
            &source,
            id,
            Controls {
                feather: 0.,
                ..base
            },
            Quality::Proxy,
        )?,
        &expected,
    )?;
    for name in [
        "cpu_oracle",
        "rgb_metric_extremes",
        "linear_picker",
        "alpha_ignored",
        "single_srgb_decode",
        "determinism",
        "cache_invalidation",
        "proxy_extent",
    ] {
        check(&mut checks, name);
    }
    require(
        worker.diagnostics().peak_bytes <= GIB && worker.diagnostics().validation_errors == 0,
        "numerical worker budget/validation error",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(checks)
}

fn budget(g: &Gpu) -> Result<()> {
    let worker = GpuWorker::spawn(
        g.device.clone(),
        g.queue.clone(),
        GpuConfig {
            budget_bytes: 65536,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let source = photo(
        g,
        &worker,
        Extent::new(16, 16).unwrap(),
        vec![0; 16 * 16 * 4],
    );
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 65536,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    let before = worker.diagnostics().submitted;
    let c = Controls {
        picked: [0.; 3],
        ..Default::default()
    };
    worker
        .request_material(request(&source, 1, c, Quality::Full))
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event())?, WorkerEvent::BudgetExceeded { required_bytes, budget_bytes: 65536 } if required_bytes > 65536)
            && worker.diagnostics().submitted == before
            && worker.try_latest_material().is_none(),
        "budget rejection missing",
    )?;
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    compare(
        &evaluate(g, &worker, &source, 2, c, Quality::Full)?,
        &vec![1.; 256],
    )?;
    require(worker.diagnostics().peak_bytes <= 65536, "budget overshoot")?;
    worker.join().map_err(|e| e.to_string())
}
fn cancellation(g: &Gpu, extent: Extent) -> Result<texelith_gpu::WorkerDiagnostics> {
    let worker = g.worker(GIB);
    let source = photo(
        g,
        &worker,
        extent,
        vec![0; (extent.width * extent.height * 4) as usize],
    );
    let c = Controls {
        picked: [1.; 3],
        feather: 0.,
        ..Default::default()
    };
    worker
        .request_material(request(&source, 1, c, Quality::Full))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().scheduled == 0 {
        require(
            Instant::now() < deadline,
            "cancellation generation did not start",
        )?;
        std::thread::yield_now();
    }
    let c = Controls {
        picked: [0.; 3],
        ..c
    };
    let values = evaluate(g, &worker, &source, 2, c, Quality::Full)?;
    require(
        values.iter().all(|&v| v == 1.),
        "latest parameters not published",
    )?;
    let stats = worker.diagnostics();
    require(
        stats.cancellation_requests > 0
            && stats.stale_publications == 0
            && stats.peak_in_flight == 1
            && stats.validation_errors == 0,
        "cancellation not exercised",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(stats)
}
fn measure(g: &Gpu, prepared: &texelith_gpu::PreparedPass) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder);
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.);
        finish(g)?;
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.]);
    }
    let t = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":t.measurements,"p95_ms":t.p95_ms,
        "submission_ms":submissions[10..].iter().sum::<f64>()/100., "samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}

pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let now = Instant::now();
    let kernel = MetallicKernel::new(&g.device, g.queue.clone());
    let cold_pipeline_ms = now.elapsed().as_secs_f64() * 1000.;
    let mut checks = numerics(&g, &kernel)?;
    budget(&g)?;
    check(&mut checks, "low_budget");
    let stats = cancellation(&g, context.extent)?;
    check(&mut checks, "cancellation");
    let extent = context.extent;
    let pixels = u64::from(extent.width) * u64::from(extent.height);
    let pool = Arc::new(Mutex::new(ResourcePool::new(GIB)));
    // Reserve all fixture payloads, upload staging and padded readback before
    // creating them. The production worker accounts its own resources above.
    let pitch = (extent.width * 4).div_ceil(256) * 256;
    let bytes_reserved =
        pixels * (4 + 8 + 4 + 4 + 4 + 4) + u64::from(pitch) * u64::from(extent.height) + 128 + 128;
    let reservation = pool
        .lock()
        .unwrap()
        .allocate(AllocationDesc::Staging {
            bytes: bytes_reserved,
        })
        .map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = (0..pixels)
        .flat_map(|i| {
            let v = (i.wrapping_mul(54) % 256) as u8;
            [v, v, v, 255]
        })
        .collect();
    let raw = texture(&g, extent, wgpu::TextureFormat::Rgba8Unorm);
    upload(&g, &raw, &bytes, 4);
    let color = texture(&g, extent, wgpu::TextureFormat::Rgba16Float);
    let output = texture(&g, extent, wgpu::TextureFormat::R32Float);
    let masks: Vec<f32> = (0..pixels).map(|i| (i % 5) as f32 * 0.25).collect();
    let mask = texture(&g, extent, wgpu::TextureFormat::R32Float);
    upload(&g, &mask, bytemuck::cast_slice(&masks), 4);
    let executor = pollster::block_on(GpuExecutor::new(g.device.clone(), g.queue.clone()))
        .map_err(|e| e.to_string())?;
    let conversion = executor
        .prepare(Pass::ColorSource, &raw, &color, extent)
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    conversion.encode(&mut encoder);
    g.queue.submit([encoder.finish()]);
    finish(&g)?;
    let uniform = uniform(&g);
    let default = Controls::default();
    let prepared = kernel
        .prepare(
            &g.device,
            &color,
            &output,
            None,
            &uniform,
            &default.params(),
        )
        .map_err(|e| e.to_string())?;
    let mut timing = measure(&g, &prepared)?;
    compare(
        &g.read(&output),
        &reference::photo_reference(&bytes, default),
    )?;
    drop(prepared);
    // One pass has no quality-dependent iteration count. The maximal work adds
    // an explicit full-resolution numeric mask to the full RGB/feather path.
    let maximum_controls = Controls {
        picked: [1.; 3],
        radius: 0.75,
        feather: 0.5,
        threshold: 0.75,
        mode: Combine::Subtract,
    };
    let prepared = kernel
        .prepare(
            &g.device,
            &color,
            &output,
            Some(&mask),
            &uniform,
            &maximum_controls.params(),
        )
        .map_err(|e| e.to_string())?;
    let mut maximum = measure(&g, &prepared)?;
    maximum["mask_mode"] = json!("Subtract");
    let expected: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .zip(&masks)
        .map(|(p, &m)| {
            reference::metallic(
                [
                    reference::linear_half(p[0]),
                    reference::linear_half(p[1]),
                    reference::linear_half(p[2]),
                ],
                Some(m),
                maximum_controls,
            )
        })
        .collect();
    compare(&g.read(&output), &expected)?;
    let peak = bytes_reserved.max(stats.peak_bytes);
    require(peak <= GIB, "4K budget exceeded")?;
    timing["peak_bytes"] = json!(peak);
    timing["cancellation_errors"] = json!(stats.stale_publications);
    drop((prepared, conversion, uniform, raw, color, output, mask));
    pool.lock()
        .unwrap()
        .release(reservation)
        .map_err(|e| e.to_string())?;
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "harness reservations leaked",
    )?;
    finish(&g)?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "metallic GPU validation error",
    )?;
    let manifest = json!({"picked_color":default.picked,"radius":default.radius,"threshold":default.threshold,"feather":default.feather,"mask_mode":"Replace","mask":"absent","seed":54,"size":extent.width});
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
    let maximum_manifest = json!({"picked_color":maximum_controls.picked,"radius":maximum_controls.radius,"threshold":maximum_controls.threshold,"feather":maximum_controls.feather,"mask_mode":"Subtract","mask":"explicit R32Float"});
    let mut report = json!({"passed":true,"suite":"metallic","size":extent.width,"adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend)},
        "seed":54,"manifest":manifest,"manifest_hash":hash,"manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four",
        "maximum_manifest":maximum_manifest,"checks":checks,"timing":timing,"maximum_quality":maximum,"cold_pipeline_ms":cold_pipeline_ms,
        "metallic":{"budget_bytes":GIB,"peak_in_flight":stats.peak_in_flight,"cancellation_requests":stats.cancellation_requests,"cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,"low_budget_rejections":1,"scratch_bytes":128},
        "accounting_scope":"source/color/output/mask textures, uploads, padded scalar readback and uniforms; excludes driver allocations; full-node begins at typed linear color"});
    if let Err(error) = validate_report(&report) {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
