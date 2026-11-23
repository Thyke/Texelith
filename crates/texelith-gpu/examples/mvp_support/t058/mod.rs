//! CPU-oracle checks and completion-bound full-node measurements for T-058.
use crate::reference::abi;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t058/gpu.rs"]
mod gpu;
pub mod reference;
#[path = "contract.rs"]
mod report;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t058/support.rs"]
mod support;
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use support::{Controls, evaluate, height, install, request, wait};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, MaterialError, MaterialPlan, Pass, ResourceReservations,
    TypedSourcePixels, WorkerEvent,
    pool::{AllocationDesc, FenceId, ResourcePool},
    seamless_splat::{SplatKernels, SplatTransform},
};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn error(actual: impl Iterator<Item = f64>, expected: &[f64], tolerance: f64) -> Result<f64> {
    let actual: Vec<_> = actual.collect();
    require(
        !actual.is_empty() && actual.len() == expected.len(),
        "oracle extent",
    )?;
    let mut maximum = 0.0f64;
    for (a, b) in actual.into_iter().zip(expected) {
        require(a.is_finite() && b.is_finite(), "nonfinite output")?;
        maximum = maximum.max((a - b).abs());
    }
    require(
        maximum <= tolerance,
        &format!("oracle maximum {maximum} exceeds {tolerance}"),
    )?;
    Ok(maximum)
}
fn render(e: Extent, output: Extent, values: &[f64], c: Controls) -> Vec<f64> {
    reference::render(
        e,
        output,
        values,
        c.seed,
        c.grid,
        f64::from(c.overlap),
        c.boundary,
    )
}

fn numerics(g: &gpu::Gpu) -> Result<(Value, Vec<Value>)> {
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut scalar_error = 0.0f64;
    let mut constant_error = 0.0f64;
    let mut color_error = 0.0f64;
    let mut correlation_error = 0.0f64;
    let mut period_error = 0.0f64;
    let mut minimum_coverage = f64::INFINITY;
    for (w, h, kind) in [
        (1, 1, "constant"),
        (1, 17, "ramp"),
        (17, 1, "seeded"),
        (7, 5, "constant"),
        (7, 5, "ramp"),
        (17, 9, "impulse"),
        (257, 193, "seeded"),
        (1025, 193, "seeded"),
        (1, 1025, "seeded"),
        (1025, 1, "seeded"),
    ] {
        let e = Extent::new(w, h).unwrap();
        let input = reference::fixture(e, kind);
        let source = height(&worker, e, input.clone());
        let values: Vec<_> = input.into_iter().map(f64::from).collect();
        for (seed, grid, overlap) in [(0, 2, 0.1), (58, 4, 0.25), (u32::MAX, 8, 0.5)] {
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                let c = Controls {
                    seed,
                    grid,
                    overlap,
                    boundary,
                };
                serial += 1;
                let mut r = request(std::slice::from_ref(&source), serial, Some(c));
                if w > 1024 || h > 1024 {
                    r.quality = texelith_gpu::Quality::Proxy;
                    r.extent = texelith_gpu::proxy_extent(e).unwrap();
                }
                let output = r.extent;
                let actual = evaluate(g, &worker, r);
                let err = error(
                    actual[&MapKind::Height].iter().map(|p| p[0]),
                    &render(e, output, &values, c),
                    2e-5,
                )?;
                scalar_error = scalar_error.max(err);
                if kind == "constant" {
                    constant_error = constant_error.max(err);
                }
            }
        }
    }
    for value in [-3.25f32, 1.75, 0.500001] {
        let e = Extent::new(9, 7).unwrap();
        let source = height(&worker, e, vec![value; 63]);
        serial += 1;
        let actual = evaluate(
            g,
            &worker,
            request(&[source], serial, Some(Controls::default())),
        );
        scalar_error = scalar_error.max(error(
            actual[&MapKind::Height].iter().map(|p| p[0]),
            &[f64::from(value); 63],
            3e-7,
        )?);
    }
    // Imported RGBA8 linear height also samples the original source in proxies.
    let e = Extent::new(1025, 193).unwrap();
    let bytes: Vec<_> = (0..e.width * e.height)
        .flat_map(|i| [(i.wrapping_mul(71) % 256) as u8, 255, 128, 0])
        .collect();
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f64::from(p[0]) / 255.0)
        .collect();
    let source = install(
        &worker,
        e,
        TypedSourcePixels::Rgba8 {
            pixels: bytes.into(),
            interpretation: InputInterpretation::LinearHeight,
        },
    );
    serial += 1;
    let mut r = request(&[source], serial, Some(Controls::default()));
    r.quality = texelith_gpu::Quality::Proxy;
    r.extent = texelith_gpu::proxy_extent(e).unwrap();
    let output = r.extent;
    let actual = evaluate(g, &worker, r);
    scalar_error = scalar_error.max(error(
        actual[&MapKind::Height].iter().map(|p| p[0]),
        &render(e, output, &values, Controls::default()),
        2e-5,
    )?);
    let e = Extent::new(17, 11).unwrap();
    let values: Vec<_> = reference::fixture(e, "seeded")
        .into_iter()
        .map(f64::from)
        .collect();
    for grid in 2..=8 {
        for overlap in [0.1, 0.25, 0.5] {
            let transform = SplatTransform::new(58, grid, overlap).map_err(|e| e.to_string())?;
            let field = |uv| {
                let taps = transform.samples(uv);
                let sum: f64 = taps.iter().map(|s| f64::from(s.weight)).sum();
                taps.iter()
                    .map(|s| {
                        f64::from(s.weight)
                            * reference::bilinear(e, &values, s.uv.map(f64::from), Boundary::Repeat)
                    })
                    .sum::<f64>()
                    / sum.max(1e-8)
            };
            for y in 0..=32 {
                for x in 0..=32 {
                    let uv = [x as f32 / 32.0, y as f32 / 32.0];
                    let taps = transform.samples(uv);
                    require(
                        taps.len() <= 9
                            && taps.iter().all(|s| s.weight.is_finite() && s.weight >= 0.0),
                        "bounded finite neighbors",
                    )?;
                    minimum_coverage =
                        minimum_coverage.min(taps.iter().map(|s| f64::from(s.weight)).sum());
                    error(
                        std::iter::once(field(uv)),
                        &[reference::at(
                            e,
                            &values,
                            uv.map(f64::from),
                            58,
                            grid,
                            f64::from(overlap),
                            Boundary::Repeat,
                        )],
                        2e-5,
                    )?;
                    for shift in [[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]] {
                        period_error = period_error
                            .max((field(uv) - field([uv[0] + shift[0], uv[1] + shift[1]])).abs());
                    }
                }
            }
        }
    }
    require(
        minimum_coverage > 1e-8 && period_error <= 2e-5,
        "coverage/period",
    )?;
    let e = Extent::new(31, 19).unwrap();
    let codes = [0, 10, 11, 64, 128, 192, 255];
    let bytes: Vec<_> = (0..31 * 19)
        .flat_map(|i| {
            [
                codes[i % 7],
                codes[(i * 3 + 1) % 7],
                codes[(i * 5 + 2) % 7],
                13,
            ]
        })
        .collect();
    let channels: [Vec<f64>; 3] = std::array::from_fn(|c| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| reference::half(reference::linear(p[c])))
            .collect()
    });
    let color = install(
        &worker,
        e,
        TypedSourcePixels::Rgba8 {
            pixels: bytes.into(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    );
    let scalar = height(&worker, e, channels[0].iter().map(|v| *v as f32).collect());
    for (seed, grid, overlap) in [(0, 2, 0.1), (58, 4, 0.25), (u32::MAX, 8, 0.5)] {
        let c = Controls {
            seed,
            grid,
            overlap,
            ..Default::default()
        };
        serial += 1;
        let actual = evaluate(
            g,
            &worker,
            request(&[color.clone(), scalar.clone()], serial, Some(c)),
        );
        for (channel, values) in channels.iter().enumerate() {
            color_error = color_error.max(error(
                actual[&MapKind::Albedo].iter().map(|p| p[channel]),
                &render(e, e, values, c),
                5e-4,
            )?);
        }
        correlation_error = correlation_error.max(error(
            actual[&MapKind::Albedo].iter().map(|p| p[0]),
            &actual[&MapKind::Height]
                .iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            5e-4,
        )?);
        require(
            actual[&MapKind::Albedo].iter().all(|p| p[3] == 1.0),
            "alpha",
        )?;
    }
    for (w, h) in [(1, 1), (1, 17), (17, 1), (257, 193)] {
        let e = Extent::new(w, h).unwrap();
        serial += 1;
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: [128, 64, 255, 13].repeat((w * h) as usize).into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let actual = evaluate(
            g,
            &worker,
            request(&[source], serial, Some(Controls::default())),
        );
        for (c, code) in [128, 64, 255].iter().enumerate() {
            constant_error = constant_error.max(error(
                actual[&MapKind::Albedo].iter().map(|p| p[c]),
                &vec![reference::half(reference::linear(*code)); (w * h) as usize],
                2e-5,
            )?);
        }
    }
    let mut seams = Vec::new();
    for (w, h) in [(256, 256), (257, 193)] {
        let e = Extent::new(w, h).unwrap();
        let input = reference::fixture(e, "discontinuous_noise");
        let source = height(&worker, e, input.clone());
        serial += 1;
        let actual = evaluate(
            g,
            &worker,
            request(&[source], serial, Some(Controls::default())),
        );
        let input: Vec<_> = input.into_iter().map(f64::from).collect();
        let output: Vec<_> = actual[&MapKind::Height].iter().map(|p| p[0]).collect();
        scalar_error = scalar_error.max(error(
            output.iter().copied(),
            &render(e, e, &input, Controls::default()),
            2e-5,
        )?);
        let (before, interior) = reference::seam_rms(e, &input);
        let (after, detail) = reference::seam_rms(e, &output);
        require(
            before > 0.01 && interior > 0.01 && after <= before * 0.5 && detail >= interior * 0.5,
            "seam/detail",
        )?;
        seams.push(json!({"fixture":format!("discontinuous_noise_{w}x{h}"),"source_boundary_rms":before,"source_interior_rms":interior,"result_boundary_rms":after,"result_interior_rms":detail}));
    }
    let e = Extent::new(64, 48).unwrap();
    let input = reference::fixture(e, "structured");
    let source = height(&worker, e, input.clone());
    serial += 1;
    let original = evaluate(
        g,
        &worker,
        request(std::slice::from_ref(&source), serial, None),
    );
    require(
        original[&MapKind::Height]
            .iter()
            .map(|p| p[0])
            .eq(input.iter().copied().map(f64::from)),
        "disabled lossless",
    )?;
    serial += 1;
    let changed = evaluate(
        g,
        &worker,
        request(
            std::slice::from_ref(&source),
            serial,
            Some(Controls::default()),
        ),
    );
    let difference = changed[&MapKind::Height]
        .iter()
        .zip(&input)
        .map(|(p, v)| (p[0] - f64::from(*v)).abs())
        .sum::<f64>()
        / input.len() as f64;
    require(difference > 0.03, "structured tradeoff")?;
    serial += 1;
    require(
        original == evaluate(g, &worker, request(&[source], serial, None)),
        "restore source",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok((
        json!({"constant_max_error":constant_error,"scalar_max_error":scalar_error,"color_max_error":color_error,"period_max_error":period_error,"minimum_coverage":minimum_coverage,"maximum_neighbors":9,"same_seed_identical":true,"correlation_max_error":correlation_error,"structured_mean_absolute_change":difference}),
        seams,
    ))
}

fn scheduling(g: &gpu::Gpu) -> Result<(Value, f64)> {
    let request = |sources: &[SourceDescriptor], serial, overlap: Option<f32>, boundary| {
        support::request(
            sources,
            serial,
            overlap.map(|overlap| Controls {
                overlap,
                boundary,
                ..Default::default()
            }),
        )
    };
    let mut different_seed_rms = 0.0;

    let extent = Extent::new(1024, 512).unwrap();
    let input = reference::fixture(extent, "seeded");
    let mut baseline = None;
    let mut stats = texelith_gpu::WorkerDiagnostics::default();
    for _ in 0..2 {
        let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
            .map_err(|e| e.to_string())?;
        let source = height(&worker, extent, input.clone());
        worker
            .request_material(request(
                std::slice::from_ref(&source),
                1,
                Some(0.1),
                Boundary::Repeat,
            ))
            .map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(30);
        while worker.diagnostics().in_flight != 1 {
            require(
                worker.try_event().is_none()
                    && worker.try_latest_material().is_none()
                    && Instant::now() < deadline,
                "cold request never active for cancellation",
            )?;
            std::thread::yield_now();
        }
        let result = evaluate(
            g,
            &worker,
            request(
                std::slice::from_ref(&source),
                2,
                Some(0.25),
                Boundary::Repeat,
            ),
        );
        let expected = reference::render(
            extent,
            extent,
            &input.iter().copied().map(f64::from).collect::<Vec<_>>(),
            58,
            4,
            0.25,
            Boundary::Repeat,
        );
        error(
            result[&MapKind::Height].iter().map(|p| p[0]),
            &expected,
            2e-5,
        )?;
        let before = worker.diagnostics();
        let warm = evaluate(
            g,
            &worker,
            request(
                std::slice::from_ref(&source),
                3,
                Some(0.25),
                Boundary::Repeat,
            ),
        );
        require(
            result == warm
                && before.submitted == worker.diagnostics().submitted
                && worker.diagnostics().cache_hits > before.cache_hits,
            "warm cache/determinism",
        )?;
        if let Some(previous) = &baseline {
            require(previous == &result, "cold determinism")?;
        }
        let changed = evaluate(
            g,
            &worker,
            support::request(
                std::slice::from_ref(&source),
                10,
                Some(Controls {
                    seed: 59,
                    ..Default::default()
                }),
            ),
        );
        different_seed_rms = (result[&MapKind::Height]
            .iter()
            .zip(&changed[&MapKind::Height])
            .map(|(a, b)| (a[0] - b[0]).powi(2))
            .sum::<f64>()
            / (extent.width * extent.height) as f64)
            .sqrt();
        require(different_seed_rms > 1e-3, "different seeds")?;
        baseline = Some(result);
        let key = |overlap, boundary, serial| -> Result<_> {
            let plan = MaterialPlan::new(
                &request(
                    std::slice::from_ref(&source),
                    serial,
                    Some(overlap),
                    boundary,
                ),
                BUDGET,
                ResourceReservations::default(),
            )
            .map_err(|e| e.to_string())?;
            Ok(*plan.output_keys().values().next().unwrap())
        };
        require(
            key(0.25, Boundary::Repeat, 4)? == key(0.25, Boundary::Repeat, 5)?
                && key(0.25, Boundary::Repeat, 4)? != key(0.5, Boundary::Repeat, 5)?
                && key(0.25, Boundary::Repeat, 4)? != key(0.25, Boundary::Clamp, 5)?,
            "cache identity",
        )?;
        stats = worker.diagnostics();
        require(
            stats.cancellation_requests > 0
                && stats.stale_publications == 0
                && stats.peak_in_flight == 1
                && stats.peak_pending == 1
                && stats.validation_errors == 0
                && stats.peak_bytes <= BUDGET,
            "runtime cancellation/budget",
        )?;
        worker.join().map_err(|e| e.to_string())?;
    }
    let worker = GpuWorker::spawn(
        g.device.clone(),
        g.queue.clone(),
        GpuConfig {
            budget_bytes: 65536,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let extent = Extent::new(16, 16).unwrap();
    let input = reference::fixture(extent, "seeded");
    let source = height(&worker, extent, input.clone());
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 65536,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    worker
        .request_material(request(
            std::slice::from_ref(&source),
            1,
            Some(0.25),
            Boundary::Repeat,
        ))
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event()),WorkerEvent::BudgetExceeded{budget_bytes:65536,required_bytes} if required_bytes>65536),
        "low budget rejection",
    )?;
    require(
        worker.try_latest_material().is_none(),
        "partial budget publication",
    )?;
    wait(|| (worker.diagnostics().in_flight == 0).then_some(()));
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    let result = evaluate(
        g,
        &worker,
        request(&[source], 2, Some(0.25), Boundary::Repeat),
    );
    error(
        result[&MapKind::Height].iter().map(|p| p[0]),
        &reference::render(
            extent,
            extent,
            &input.into_iter().map(f64::from).collect::<Vec<_>>(),
            58,
            4,
            0.25,
            Boundary::Repeat,
        ),
        2e-5,
    )?;
    require(
        worker.diagnostics().peak_bytes <= 65536,
        "low budget recovery",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    // Admission for scalar, color, and a correlated pair with the viewport share.
    let extent = Extent::new(4096, 4096).unwrap();
    let scalar = SourceDescriptor {
        content_hash: [1; 32],
        extent,
        interpretation: InputInterpretation::LinearHeight,
    };
    let color = SourceDescriptor {
        content_hash: [2; 32],
        extent,
        interpretation: InputInterpretation::SrgbPhoto,
    };
    for sources in [
        vec![scalar.clone()],
        vec![color.clone()],
        vec![color, scalar],
    ] {
        let request = request(&sources, 1, Some(0.25), Boundary::Repeat);
        let reservations = ResourceReservations {
            viewport_bytes: 144 * 1024 * 1024,
            readback_bytes: 0,
        };
        let plan = MaterialPlan::new(&request, BUDGET, reservations).map_err(|e| e.to_string())?;
        require(
            matches!(
                MaterialPlan::new(&request, plan.peak_bytes() - 1, reservations),
                Err(MaterialError::BudgetExceeded { .. })
            ),
            "one-byte admission",
        )?;
    }
    Ok((
        json!({"budget_bytes":BUDGET,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,"cancellation_requests":stats.cancellation_requests,"cancelled_pass_batches":stats.cancelled_pass_batches,"low_budget_rejections":1,"cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors}),
        different_seed_rms,
    ))
}

fn measure(g: &gpu::Gpu, extent: Extent, grid: u32, overlap: f32) -> Result<Value> {
    // Each actual payload has exactly one admitted ledger allocation. The
    // immutable decoded upload is excluded from node time; sRGB preparation is
    // recomputed in every timed batch, followed by all four overlap-add copies.
    let mut pool = ResourcePool::new(BUDGET);
    let usage = wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::STORAGE_BINDING
        | wgpu::TextureUsages::COPY_DST;
    let mut ids = Vec::new();
    let mut textures = Vec::new();
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba16Float,
    ] {
        let actual_usage = if format == wgpu::TextureFormat::Rgba8Unorm {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        } else {
            usage
        };
        ids.push(
            pool.allocate(AllocationDesc::Texture {
                extent,
                format,
                usage: actual_usage,
            })
            .map_err(|e| e.to_string())?,
        );
        textures.push(g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-058 timed payload"),
            size: wgpu::Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: actual_usage,
            view_formats: &[],
        }));
    }
    for desc in [
        AllocationDesc::Buffer { bytes: 64 },
        AllocationDesc::Buffer { bytes: 64 },
        AllocationDesc::Staging { bytes: 128 },
        AllocationDesc::Staging {
            bytes: u64::from(extent.width) * u64::from(extent.height) * 4,
        },
    ] {
        ids.push(pool.allocate(desc).map_err(|e| e.to_string())?);
    }
    let bytes: Vec<u8> = reference::fixture(extent, "seeded")
        .into_iter()
        .flat_map(|v| {
            let c = (v * 255.0) as u8;
            [c, c.wrapping_add(71), c.wrapping_add(137), 255]
        })
        .collect();
    g.queue.write_texture(
        textures[0].as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * 4),
            rows_per_image: Some(extent.height),
        },
        textures[0].size(),
    );
    let executor = pollster::block_on(GpuExecutor::new(g.device.clone(), g.queue.clone()))
        .map_err(|e| e.to_string())?;
    let source = executor
        .prepare(Pass::ColorSource, &textures[0], &textures[1], extent)
        .map_err(|e| e.to_string())?;
    let uniform = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("splat timed uniform"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let kernels = SplatKernels::new(&g.device, g.queue.clone());
    let splat = kernels
        .prepare(
            &g.device,
            &textures[1],
            &textures[2],
            &uniform,
            SplatTransform::new(58, grid, overlap).map_err(|e| e.to_string())?,
            Boundary::Repeat,
        )
        .map_err(|e| e.to_string())?;
    // Flush the initial upload and uniform writes under their charged fence.
    for id in &ids {
        pool.mark_submitted(*id, FenceId(1))
            .map_err(|e| e.to_string())?;
    }
    g.queue.submit([g
        .device
        .create_command_encoder(&Default::default())
        .finish()]);
    g.wait();
    pool.complete(FenceId(1));
    pool.release(ids.pop().unwrap())
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for i in 0..110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        source.encode(&mut encoder);
        splat.encode(&mut encoder);
        let fence = FenceId(i + 2);
        for id in &ids {
            pool.mark_submitted(*id, fence).map_err(|e| e.to_string())?;
        }
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        g.wait();
        pool.complete(fence);
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = crate::contract::aggregate_timings(&rows, 10)?;
    let peak = pool.diagnostics().peak_bytes;
    for id in ids {
        pool.release(id).map_err(|e| e.to_string())?;
    }
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","includes_source":true,"warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"peak_bytes":peak,"cache_hits_used_for_timing":0,"cancellation_errors":0,"overlap":overlap,"grid":grid,"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>(),"scope_detail":"linear albedo source preparation and nine-neighbor periodic splat, encoding through completion; excludes immutable upload and pipeline preparation"}),
    )
}

pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
    requested_name: &str,
    requested_backend: &str,
) -> Result<Value> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let g = gpu::Gpu { device, queue };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
        let (numerical, seams) = numerics(&g)?;
        let (resources, seed_rms) = scheduling(&g)?;
        let mut numerical = numerical;
        numerical["different_seed_rms"] = json!(seed_rms);
        let timing = measure(&g, extent, 4, 0.25)?;
        let maximum = measure(&g, extent, 8, 0.5)?;
        let manifest = json!({"overlap":0.25,"boundary":"repeat","epsilon":1e-8,"window":"separable_cubic_compact_pow4","seed":58,"grid":4,"hash":"wrapped_u32_avalanche_v1","max_neighbors":9,"algorithm_revision":1});
        let checks: Vec<_> = report::REQUIRED_CHECKS
            .iter()
            .map(|n| json!({"name":n,"passed":true}))
            .collect();
        Ok(
            json!({"passed":true,"suite":"seamless_splat","size":extent.width,"seed":58,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},"requested_adapter":{"name":requested_name,"backend":requested_backend},"manifest_hash":report::manifest_hash(&manifest)?,"manifest":manifest,"checks":checks,"timing":timing,"maximum_quality":maximum,"numerical":numerical,"seams":seams,"resources":resources}),
        )
    }));
    g.wait();
    let gpu_error = pollster::block_on(scope.pop());
    let result = match outcome {
        Ok(value) => value,
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "behavioral check panicked".into())),
    };
    let mut result = match result {
        Ok(value) => value,
        Err(message) => {
            return Ok(
                json!({"passed":false,"suite":"seamless_splat","failure":message,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)}}),
            );
        }
    };
    let validation = crate::contract::validate_completed_report(
        &result,
        gpu_error.map_or(Ok(()), |e| Err(e.to_string())),
    )
    .and_then(|()| report::validate(&result));
    if let Err(message) = validation {
        result["passed"] = json!(false);
        result["failure"] = json!(message);
    }
    Ok(result)
}

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> std::result::Result<serde_json::Value, String> {
    run(
        context.device.clone(),
        context.queue.clone(),
        context.extent,
        context.info,
        context.adapter_name,
        context.backend,
    )
}
