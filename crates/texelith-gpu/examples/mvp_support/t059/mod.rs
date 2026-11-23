//! Real worker oracles and cold, completion-bound full-node measurements.
use crate::reference::abi;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t059/gpu.rs"]
mod gpu;
pub mod reference;
use crate::contract::histogram_contract as report_contract;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t059/support.rs"]
mod support;
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use support::{BASES, Base, Controls, evaluate, height, install, request, wait};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, MaterialError, MaterialPlan, Pass, ResourceReservations,
    TypedSourcePixels, WorkerEvent,
    pool::{AllocationDesc, FenceId, ResourcePool},
    seamless_histogram::{HistogramKernels, HistogramResources, diagnostics, scratch_bytes},
};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;
fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}
fn error(a: impl Iterator<Item = f64>, b: &[f64], bound: f64) -> Result<f64> {
    let a: Vec<_> = a.collect();
    require(!a.is_empty() && a.len() == b.len(), "oracle extent")?;
    let mut maximum = 0.0f64;
    for (a, b) in a.into_iter().zip(b) {
        require(a.is_finite() && b.is_finite(), "nonfinite pixel")?;
        maximum = maximum.max((a - b).abs());
    }
    require(
        maximum <= bound,
        &format!("oracle error {maximum} > {bound}"),
    )?;
    Ok(maximum)
}
fn texture(g: &gpu::Gpu, e: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("T-059 admitted fixture"),
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
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
fn read_buffer(g: &gpu::Gpu, source: &wgpu::Buffer) -> Result<Vec<u32>> {
    let buffer = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("small histogram oracle readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    g.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.wait();
    rx.recv_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let values = mapped
        .as_chunks::<4>().0.iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}
fn integer_mapping(g: &gpu::Gpu) -> Result<()> {
    let e = Extent::new(257, 193).unwrap();
    let values = reference::fixture(e, "seeded");
    let mut ledger = ResourcePool::new(BUDGET);
    let ids = ledger
        .allocate_plan(&[AllocationDesc::Staging {
            bytes: scratch_bytes(e) + u64::from(e.width) * u64::from(e.height) * 12 + 2056 * 4,
        }])
        .map_err(|e| e.to_string())?;
    let input = texture(g, e, wgpu::TextureFormat::R32Float);
    let output = texture(g, e, wgpu::TextureFormat::R32Float);
    g.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&values),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 4),
            rows_per_image: Some(e.height),
        },
        input.size(),
    );
    let resources = HistogramResources::new(&g.device, &g.queue, e);
    let kernels = HistogramKernels::new(&g.device);
    let pass = kernels
        .prepare(
            &g.device,
            &g.queue,
            &input,
            &output,
            &resources,
            &BASES[0].core(),
            Boundary::Repeat,
            1.0,
        )
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    pass.encode(&mut encoder, true);
    g.queue.submit([encoder.finish()]);
    g.wait();
    let mapping = read_buffer(g, &resources.cdf)?;
    let oracle =
        reference::Histogram::new(&values.iter().copied().map(f64::from).collect::<Vec<_>>());
    let mut count = 0u32;
    for i in 0..1024 {
        require(
            mapping[1032 + i] == oracle.counts[i],
            "integer histogram mismatch",
        )?;
        require(
            (f64::from(f32::from_bits(mapping[i])) - f64::from(count) / values.len() as f64).abs()
                < 1e-7,
            "CDF mismatch",
        )?;
        count += oracle.counts[i];
    }
    require(
        count == e.width * e.height && f32::from_bits(mapping[1024]) == 1.0,
        "histogram conservation",
    )?;
    require(
        (f64::from(f32::from_bits(mapping[1025])) - oracle.variance).abs() < 1e-6,
        "population variance",
    )?;
    for id in ids {
        ledger.release(id).map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn numerics(g: &gpu::Gpu) -> Result<(Value, Vec<Value>, Vec<Value>)> {
    integer_mapping(g)?;
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut scalar = 0.0f64;
    let mut constant = 0.0f64;
    let mut precision = 0.0f64;
    for (w, h, kind) in [
        (1, 1, "constant"),
        (1, 17, "ramp"),
        (17, 1, "seeded"),
        (7, 5, "ramp"),
        (17, 9, "impulse"),
        (257, 193, "seeded"),
        (1025, 193, "seeded"),
        (1, 1025, "seeded"),
        (1025, 1, "seeded"),
    ] {
        let e = Extent::new(w, h).unwrap();
        let input = reference::fixture(e, kind);
        let rgb: Vec<_> = input.iter().map(|v| [f64::from(*v); 3]).collect();
        let source = height(&worker, e, input);
        for base in [
            BASES[0],
            BASES[1],
            Base::Offset(0.1),
            Base::Splat {
                seed: u32::MAX,
                grid: 8,
                overlap: 0.5,
            },
        ] {
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                for strength in [0.0, 0.375, 1.0] {
                    serial += 1;
                    let mut req = request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            boundary,
                            strength,
                        }),
                    );
                    if w > 1024 || h > 1024 {
                        req.extent = texelith_gpu::proxy_extent(e).unwrap();
                        req.quality = texelith_gpu::Quality::Proxy;
                    }
                    let out = req.extent;
                    let actual = evaluate(g, &worker, req);
                    let expected = reference::render(
                        e,
                        out,
                        &rgb,
                        &base.core(),
                        boundary,
                        f64::from(strength),
                        false,
                    );
                    let err = error(
                        actual[&MapKind::Height].iter().map(|v| v[0]),
                        &expected.iter().map(|v| v[0]).collect::<Vec<_>>(),
                        2e-5,
                    )?;
                    scalar = scalar.max(err);
                    if kind == "constant" {
                        constant = constant.max(err);
                    }
                }
            }
        }
    }
    let e = Extent::new(19, 7).unwrap();
    for value in [0.0f32, 1e-12, 0.500001, -3.25, 1.75] {
        let source = height(&worker, e, vec![value; (e.width * e.height) as usize]);
        for base in BASES {
            serial += 1;
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    serial,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            let err = error(
                actual[&MapKind::Height].iter().map(|v| v[0]),
                &vec![f64::from(value); (e.width * e.height) as usize],
                3e-7,
            )?;
            precision = precision.max(err);
            constant = constant.max(err);
        }
    }
    for amplitude in [1e-8f32, 0.0005] {
        let values: Vec<_> = reference::fixture(e, "seeded")
            .iter()
            .map(|v| 0.2 + amplitude * v)
            .collect();
        let source = height(&worker, e, values);
        for base in BASES {
            serial += 1;
            let actual = evaluate(
                g,
                &worker,
                request(
                    std::slice::from_ref(&source),
                    serial,
                    Some(Controls {
                        base,
                        ..Default::default()
                    }),
                ),
            );
            serial += 1;
            let plain = evaluate(
                g,
                &worker,
                support::request_method(
                    std::slice::from_ref(&source),
                    serial,
                    Some((base.method(), Boundary::Repeat)),
                ),
            );
            error(
                actual[&MapKind::Height].iter().map(|v| v[0]),
                &plain[&MapKind::Height]
                    .iter()
                    .map(|v| v[0])
                    .collect::<Vec<_>>(),
                3e-7,
            )?;
        }
    }
    let mut color_error = 0.0f64;
    let mut changes = 0.0;
    let mut clipped = 0;
    let mut pixels = 0;
    let mut ratio = 0.0f64;
    for (e, bytes) in [
        (
            Extent::new(31, 19).unwrap(),
            (0..31 * 19)
                .flat_map(|i| {
                    [
                        (i * 71 % 256) as u8,
                        (i * 17 % 256) as u8,
                        (i * 43 % 256) as u8,
                        13,
                    ]
                })
                .collect::<Vec<_>>(),
        ),
        (Extent::new(7, 5).unwrap(), [0, 0, 0, 0].repeat(35)),
        (Extent::new(7, 5).unwrap(), [1, 0, 2, 255].repeat(35)),
    ] {
        let rgb: Vec<_> = bytes
            .as_chunks::<4>().0.iter()
            .map(|p| std::array::from_fn(|c| reference::half(reference::linear(p[c]))))
            .collect();
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        for base in BASES {
            let mut zero = None;
            for strength in [0.0, 0.375, 1.0] {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            strength,
                            ..Default::default()
                        }),
                    ),
                );
                let expected = reference::render(
                    e,
                    e,
                    &rgb,
                    &base.core(),
                    Boundary::Repeat,
                    f64::from(strength),
                    true,
                );
                for c in 0..3 {
                    color_error = color_error.max(error(
                        actual[&MapKind::Albedo].iter().map(|v| v[c]),
                        &expected.iter().map(|v| v[c]).collect::<Vec<_>>(),
                        5e-4,
                    )?);
                }
                if strength == 0.0 {
                    zero = Some(actual[&MapKind::Albedo].clone());
                }
                for (p, b) in actual[&MapKind::Albedo].iter().zip(zero.as_ref().unwrap()) {
                    require(
                        p[3] == 1.0 && p[..3].iter().all(|v| (0.0..=1.0).contains(v)),
                        "SDR/chroma finite",
                    )?;
                    let l = reference::luminance([b[0], b[1], b[2]]);
                    let restored = reference::luminance([p[0], p[1], p[2]]);
                    // Report the bounded reconstruction ratio before half storage.
                    ratio = ratio.max((restored / l.max(1e-8)).min(16.0));
                    clipped += usize::from(p[..3].contains(&1.0));
                    pixels += 1;
                    changes += (p[0] - b[0]).abs() + (p[1] - b[1]).abs() + (p[2] - b[2]).abs();
                }
            }
        }
    }
    let mut cdfs = Vec::new();
    let mut seams = Vec::new();
    for (i, base) in BASES.into_iter().enumerate() {
        let name = if i == 0 { "offset" } else { "splat" };
        let e = Extent::new(1024, 1024).unwrap();
        let input = reference::fixture(e, "noise");
        let original: Vec<_> = input.iter().copied().map(f64::from).collect();
        let source = height(&worker, e, input);
        serial += 1;
        let actual = evaluate(
            g,
            &worker,
            request(
                &[source],
                serial,
                Some(Controls {
                    base,
                    ..Default::default()
                }),
            ),
        );
        let distance = reference::cdf_distance(
            &original,
            &actual[&MapKind::Height]
                .iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
        );
        require(distance <= 0.05, "noise luminance CDF")?;
        cdfs.push(json!({"base":name,"fixture":"value_noise_seed59_1024","size":1024,"bins":1024,"max_distance":distance}));
        let e = Extent::new(257, 193).unwrap();
        let input = reference::fixture(e, "discontinuous_noise");
        let (before, interior) =
            reference::seam_rms(e, &input.iter().copied().map(f64::from).collect::<Vec<_>>());
        let source = height(&worker, e, input);
        serial += 1;
        let actual = evaluate(
            g,
            &worker,
            request(
                &[source],
                serial,
                Some(Controls {
                    base,
                    ..Default::default()
                }),
            ),
        );
        let (after, detail) = reference::seam_rms(
            e,
            &actual[&MapKind::Height]
                .iter()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
        );
        require(
            after <= before * 0.5 && detail >= interior * 0.5,
            "seam/detail",
        )?;
        seams.push(json!({"base":name,"fixture":"discontinuous_noise_257x193","source_boundary_rms":before,"source_interior_rms":interior,"result_boundary_rms":after,"result_interior_rms":detail}));
    }
    worker.join().map_err(|e| e.to_string())?;
    Ok((
        json!({"scalar_max_error":scalar,"constant_max_error":constant,"height_precision_max_error":precision,"color_max_error":color_error,
        "nonfinite_pixels":0,"histogram_count_mismatch":0,"max_luminance_ratio":ratio,"clipped_pixel_fraction":clipped as f64/pixels as f64,"color_mean_absolute_change":changes/(pixels*3)as f64}),
        cdfs,
        seams,
    ))
}
fn scheduling(g: &gpu::Gpu) -> Result<Value> {
    let before = diagnostics();
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let e = Extent::new(1024, 512).unwrap();
    let values = reference::fixture(e, "seeded");
    let source = height(&worker, e, values);
    worker
        .request_material(request(
            std::slice::from_ref(&source),
            1,
            Some(Controls::default()),
        ))
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while worker.diagnostics().in_flight != 1 {
        require(
            worker.try_event().is_none()
                && worker.try_latest_material().is_none()
                && Instant::now() < deadline,
            "active cancellation not exercised",
        )?;
        std::thread::yield_now();
    }
    let c = Controls {
        base: BASES[1],
        strength: 0.5,
        ..Default::default()
    };
    let first = evaluate(
        g,
        &worker,
        request(std::slice::from_ref(&source), 2, Some(c)),
    );
    let d = worker.diagnostics();
    let warm = evaluate(
        g,
        &worker,
        request(std::slice::from_ref(&source), 3, Some(c)),
    );
    require(
        first == warm && worker.diagnostics().submitted == d.submitted,
        "warm output determinism",
    )?;
    let changed = evaluate(
        g,
        &worker,
        request(
            std::slice::from_ref(&source),
            4,
            Some(Controls {
                strength: 0.75,
                ..c
            }),
        ),
    );
    require(first != changed, "strength must reconstruct")?;
    let stats = worker.diagnostics();
    require(
        stats.cancellation_requests > 0
            && stats.peak_in_flight == 1
            && stats.peak_pending == 1
            && stats.stale_publications == 0
            && stats.validation_errors == 0
            && stats.peak_bytes <= BUDGET,
        "cancellation/ledger",
    )?;
    let e = Extent::new(37, 29).unwrap();
    let values = reference::fixture(e, "seeded");
    let source = height(&worker, e, values.clone());
    let original = evaluate(
        g,
        &worker,
        request(std::slice::from_ref(&source), 5, Some(Controls::default())),
    );
    let altered = height(&worker, e, values.iter().map(|v| 0.1 + 0.4 * v).collect());
    let result = evaluate(
        g,
        &worker,
        request(&[altered], 6, Some(Controls::default())),
    );
    require(result != original, "source mapping invalidation")?;
    let restored = evaluate(g, &worker, request(&[source], 7, Some(Controls::default())));
    require(original == restored, "restored source mapping")?;
    worker.join().map_err(|e| e.to_string())?;
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let source = height(&worker, e, values);
    let cold = evaluate(g, &worker, request(&[source], 1, Some(Controls::default())));
    require(cold == original, "cold backend determinism")?;
    worker.join().map_err(|e| e.to_string())?;
    let after = diagnostics();
    require(
        after.0 > before.0 && after.1 > before.1 + 1,
        "source LUT reuse/invalidation counters",
    )?;
    let worker = GpuWorker::spawn(
        g.device.clone(),
        g.queue.clone(),
        GpuConfig {
            budget_bytes: 262144,
            max_cache_entries: 8,
        },
    )
    .map_err(|e| e.to_string())?;
    let e = Extent::new(16, 16).unwrap();
    let source = height(&worker, e, reference::fixture(e, "seeded"));
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 262144,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    worker
        .request_material(request(
            std::slice::from_ref(&source),
            1,
            Some(Controls::default()),
        ))
        .map_err(|e| e.to_string())?;
    require(
        matches!(wait(||worker.try_event()),WorkerEvent::BudgetExceeded{budget_bytes:262144,required_bytes} if required_bytes>262144),
        "atomic budget rejection",
    )?;
    require(
        worker.try_latest_material().is_none(),
        "partial publication",
    )?;
    wait(|| (worker.diagnostics().in_flight == 0).then_some(()));
    worker
        .set_resource_reservations(Default::default())
        .map_err(|e| e.to_string())?;
    evaluate(g, &worker, request(&[source], 2, Some(Controls::default())));
    require(
        worker.diagnostics().peak_bytes <= 262144,
        "low budget recovery",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    let e = Extent::new(4096, 4096).unwrap();
    for color in [false, true] {
        let source = texelith_core::SourceDescriptor {
            extent: e,
            content_hash: [u8::from(color); 32],
            interpretation: if color {
                InputInterpretation::SrgbPhoto
            } else {
                InputInterpretation::LinearHeight
            },
        };
        for base in BASES {
            let r = request(
                std::slice::from_ref(&source),
                1,
                Some(Controls {
                    base,
                    ..Default::default()
                }),
            );
            let reservations = ResourceReservations {
                viewport_bytes: 144 * 1024 * 1024,
                readback_bytes: 0,
            };
            let plan = MaterialPlan::new(&r, BUDGET, reservations).map_err(|e| e.to_string())?;
            require(
                matches!(
                    MaterialPlan::new(&r, plan.peak_bytes() - 1, reservations),
                    Err(MaterialError::BudgetExceeded { .. })
                ),
                "one byte admission",
            )?;
        }
    }
    Ok(
        json!({"budget_bytes":BUDGET,"scratch_peak_bytes":scratch_bytes(e),"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,
        "cancellation_requests":stats.cancellation_requests,"cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,
        "low_budget_rejections":1,"source_lut_cache_hits":after.0-before.0,"source_lut_invalidations":after.1-before.1-1}),
    )
}
fn periodic() -> Result<f64> {
    let e = Extent::new(31, 19).unwrap();
    let values: Vec<_> = reference::fixture(e, "seeded")
        .iter()
        .map(|v| [f64::from(*v); 3])
        .collect();
    let hist = reference::Histogram::new(&values.iter().map(|v| v[0]).collect::<Vec<_>>());
    let mut maximum = 0.0f64;
    for base in BASES {
        let samples = |uv| -> Vec<([f32; 2], f32)> {
            match base {
                Base::Offset(o) => texelith_gpu::seamless_offset::OffsetTransform::new(o)
                    .unwrap()
                    .samples(uv)
                    .iter()
                    .map(|s| (s.uv, s.weight))
                    .collect(),
                Base::Splat {
                    seed,
                    grid,
                    overlap,
                } => texelith_gpu::seamless_splat::SplatTransform::new(seed, grid, overlap)
                    .unwrap()
                    .samples(uv)
                    .iter()
                    .map(|s| (s.uv, s.weight))
                    .collect(),
            }
        };
        for uv in [[0.0, 0.0], [0.125, 0.375], [0.25, 0.5], [-0.25, 2.5]] {
            let taps = samples(uv);
            let oracle = reference::taps(uv.map(f64::from), &base.core());
            for ((coord, w), tap) in taps.iter().zip(oracle) {
                require((f64::from(*w) - tap.weight).abs() < 2e-5, "spatial weights")?;
                for (a, b) in coord.iter().zip(tap.uv) {
                    require((f64::from(*a) - b).abs() < 2e-5, "spatial coordinates")?;
                }
            }
            let field = |uv| {
                let taps = samples(uv);
                let total = taps.iter().map(|t| f64::from(t.1)).sum::<f64>();
                let mut latent = 0.0;
                let mut squares = 0.0;
                for (uv, w) in taps {
                    let w = f64::from(w) / total;
                    latent += w * reference::to_gaussian(hist.probability(
                        reference::bilinear(e, &values, uv.map(f64::from), Boundary::Repeat)[0],
                    ));
                    squares += w * w;
                }
                hist.inverse(reference::from_gaussian(latent / squares.sqrt()))
            };
            for shift in [[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]] {
                maximum =
                    maximum.max((field(uv) - field([uv[0] + shift[0], uv[1] + shift[1]])).abs());
            }
        }
    }
    require(maximum <= 2e-5, "periodic transformed field")?;
    Ok(maximum)
}
fn measure(g: &gpu::Gpu, e: Extent, base: Base) -> Result<Value> {
    let mut pool = ResourcePool::new(BUDGET);
    let mut ids = Vec::new();
    let mut textures = Vec::new();
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba16Float,
    ] {
        let usage = if format == wgpu::TextureFormat::Rgba8Unorm {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
        };
        ids.push(
            pool.allocate(AllocationDesc::Texture {
                extent: e,
                format,
                usage,
            })
            .map_err(|e| e.to_string())?,
        );
        textures.push(g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("histogram timed payload"),
            size: wgpu::Extent3d {
                width: e.width,
                height: e.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }));
    }
    for desc in [
        AllocationDesc::Buffer { bytes: 64 },
        AllocationDesc::Staging { bytes: 64 },
        AllocationDesc::Staging {
            bytes: scratch_bytes(e),
        },
        AllocationDesc::Staging {
            bytes: u64::from(e.width) * u64::from(e.height) * 4,
        },
    ] {
        ids.push(pool.allocate(desc).map_err(|e| e.to_string())?);
    }
    let bytes: Vec<u8> = reference::fixture(e, "seeded")
        .iter()
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
            bytes_per_row: Some(e.width * 4),
            rows_per_image: Some(e.height),
        },
        textures[0].size(),
    );
    let executor = pollster::block_on(GpuExecutor::new(g.device.clone(), g.queue.clone()))
        .map_err(|e| e.to_string())?;
    let source = executor
        .prepare(Pass::ColorSource, &textures[0], &textures[1], e)
        .map_err(|e| e.to_string())?;
    let resources = HistogramResources::new(&g.device, &g.queue, e);
    let kernels = HistogramKernels::new(&g.device);
    let prepared = kernels
        .prepare(
            &g.device,
            &g.queue,
            &textures[1],
            &textures[2],
            &resources,
            &base.core(),
            Boundary::Repeat,
            1.0,
        )
        .map_err(|e| e.to_string())?;
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
        let start = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        source.encode(&mut encoder);
        prepared.encode(&mut encoder, true);
        let fence = FenceId(i + 2);
        for id in &ids {
            pool.mark_submitted(*id, fence).map_err(|e| e.to_string())?;
        }
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        g.wait();
        pool.complete(fence);
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = crate::contract::aggregate_timings(&rows, 10)?;
    let peak = pool.diagnostics().peak_bytes;
    for id in ids {
        pool.release(id).map_err(|e| e.to_string())?;
    }
    let mut result = json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,
        "submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"peak_bytes":peak,"cancellation_errors":0,"output_cache_hits_used_for_timing":0,"source_lut_cache_hits_used_for_timing":0,"cpu_pixel_conversions":0,
        "samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>(),"strength":1.0,
        "scope_detail":"sRGB source conversion, integer histogram, reduction, source CDF/LUT construction, Gaussian blend and inverse; immutable upload/fixed Gaussian table and pipeline setup excluded"});
    for pass in report_contract::PASSES {
        result[format!("includes_{pass}")] = json!(true);
    }
    match base {
        Base::Offset(overlap) => {
            result["base"] = json!("offset");
            result["overlap"] = json!(overlap);
        }
        Base::Splat { grid, overlap, .. } => {
            result["base"] = json!("splat");
            result["grid"] = json!(grid);
            result["overlap"] = json!(overlap);
        }
    }
    Ok(result)
}
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let g = gpu::Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
        let (mut numerical, cdf, seams) = numerics(&g)?;
        let resources = scheduling(&g)?;
        numerical["period_max_error"] = json!(periodic()?);
        numerical["same_backend_identical"] = json!(true);
        let timing = measure(&g, context.extent, BASES[0])?;
        let maximum = measure(
            &g,
            context.extent,
            Base::Splat {
                seed: 59,
                grid: 8,
                overlap: 0.5,
            },
        )?;
        let manifest = report_contract::default_manifest();
        Ok(
            json!({"passed":true,"suite":"seamless_histogram","size":context.extent.width,"seed":59,
            "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend)},
            "manifest_hash":report_contract::manifest_hash(&manifest)?,"manifest":manifest,"numerical":numerical,"cdf":cdf,"seams":seams,"resources":resources,
            "checks":report_contract::REQUIRED_CHECKS.iter().map(|name|json!({"name":name,"passed":true})).collect::<Vec<_>>(),"timing":timing,"maximum_quality":maximum}),
        )
    }));
    g.wait();
    let gpu_error = pollster::block_on(scope.pop());
    let outcome = match outcome {
        Ok(value) => value,
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|v| v.to_string()))
            .unwrap_or_else(|| "behavioral check panicked".into())),
    };
    let mut result = match outcome {
        Ok(value) => value,
        Err(failure) => {
            return Ok(
                json!({"passed":false,"suite":"seamless_histogram","failure":failure,"adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend)}}),
            );
        }
    };
    if let Err(failure) = crate::contract::validate_completed_report(
        &result,
        gpu_error.map_or(Ok(()), |e| Err(e.to_string())),
    )
    .and_then(|()| report_contract::validate(&result))
    {
        result["passed"] = json!(false);
        result["failure"] = json!(failure);
    }
    Ok(result)
}

