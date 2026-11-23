//! Native height node verification. CPU references are test-only f64 oracles.
use super::contract::{aggregate_timings, validate_height_report};
use super::pyramid::oracle;
use crate::reference::abi;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{
    GpuConfig, GpuExecutor, GpuWorker, HeightParameters, Pass, PreparedHeight, PreparedMinMaxRemap,
    Quality, ResourceReservations, WorkerEvent,
    pool::{AllocationDesc, FenceId, ResourcePool},
};
#[allow(dead_code)]
#[path = "../../tests/mvp_t047/gpu.rs"]
mod gpu;
#[path = "../../tests/mvp_t047/reference.rs"]
mod reference;
// Reuse the exact behavioral fixtures and request construction. Test-only
// imports in this module are unused in a normal example build.
#[allow(dead_code, unused_imports)]
#[path = "../../tests/mvp_t047/height.rs"]
mod cases;
use cases::{evaluate, fixture, install, request};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;

fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn compare(actual: &[f32], expected: &[f64]) -> Result<()> {
    require(
        !actual.is_empty()
            && actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && b.is_finite() && (f64::from(*a) - b).abs() <= 2e-5),
        "height CPU oracle mismatch",
    )
}
fn check(checks: &mut Vec<Value>, name: &str) {
    checks.push(json!({"name":name,"passed":true}));
}
fn numerics(g: &gpu::Gpu) -> Result<(Vec<Value>, u64, u64)> {
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let mut checks = Vec::new();
    let mut serial = 0;
    for (w, h, kind) in [
        (1, 1, "constant"),
        (9, 7, "constant"),
        (1, 17, "singleton_axes"),
        (17, 1, "singleton_axes"),
        (31, 17, "ramp"),
        (31, 17, "impulse"),
        (257, 193, "npot_257x193"),
    ] {
        let extent = Extent::new(w, h).map_err(|e| e.to_string())?;
        let bytes = fixture(w, h, if kind == "constant" { "flat" } else { kind });
        let image = reference::luminance(extent, &bytes);
        let source = install(&worker, extent, bytes);
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            serial += 1;
            compare(
                &evaluate(
                    g,
                    &worker,
                    request(&source, serial, &[1.0], 1.0, false, boundary, None),
                ),
                &image.pixels,
            )?;
        }
        check(&mut checks, kind);
    }
    check(&mut checks, "unit_reconstruction");
    check(&mut checks, "repeat_clamp");
    let extent = Extent::new(33, 19).unwrap();
    let bytes = fixture(33, 19, "seeded");
    let image = reference::luminance(extent, &bytes);
    let source = install(&worker, extent, bytes);
    for boundary in [Boundary::Repeat, Boundary::Clamp] {
        serial += 1;
        let unit = evaluate(
            g,
            &worker,
            request(&source, serial, &[1.0], 1.0, false, boundary, None),
        );
        let count = reference::pyramid(&image, boundary).len() - 1;
        for selected in 0..count {
            let mut weights = vec![1.0; count];
            weights[selected] = -0.75;
            serial += 1;
            let actual = evaluate(
                g,
                &worker,
                request(&source, serial, &weights, 1.0, false, boundary, None),
            );
            compare(
                &actual,
                &reference::reconstruct(&image, &weights, 1.0, boundary).pixels,
            )?;
            let mut isolated = vec![0.0; count];
            isolated[selected] = -1.75;
            let delta: Vec<_> = actual.iter().zip(&unit).map(|(a, b)| a - b).collect();
            compare(
                &delta,
                &reference::reconstruct(&image, &isolated, 0.0, boundary).pixels,
            )?;
        }
    }
    check(&mut checks, "band_isolation");
    serial += 1;
    let removed = reference::reconstruct(&image, &[1.0], 0.0, Boundary::Repeat);
    let actual = evaluate(
        g,
        &worker,
        request(&source, serial, &[1.0], 3.0, true, Boundary::Repeat, None),
    );
    compare(&actual, &removed.pixels)?;
    require(actual.iter().any(|v| *v < 0.0), "signed height was clamped")?;
    check(&mut checks, "coarse_removal");
    serial += 1;
    compare(
        &evaluate(
            g,
            &worker,
            request(
                &source,
                serial,
                &[1.0],
                3.0,
                true,
                Boundary::Repeat,
                Some((0.0, 0.5, 2.0)),
            ),
        ),
        &reference::levels(&removed, 0.0, 0.5, 2.0),
    )?;
    check(&mut checks, "explicit_levels");
    serial += 1;
    let r = request(
        &source,
        serial,
        &[0.5, 1.25, -0.25],
        0.8,
        false,
        Boundary::Repeat,
        None,
    );
    let first = evaluate(g, &worker, r.clone());
    let before = worker.diagnostics();
    serial += 1;
    let second = evaluate(
        g,
        &worker,
        request(
            &source,
            serial,
            &[0.5, 1.25, -0.25],
            0.8,
            false,
            Boundary::Repeat,
            None,
        ),
    );
    let cached = worker.diagnostics();
    require(
        first
            .iter()
            .map(|v| v.to_bits())
            .eq(second.iter().map(|v| v.to_bits())),
        "height nondeterminism",
    )?;
    require(
        cached.cache_hits > before.cache_hits && cached.submitted == before.submitted,
        "output cache missed",
    )?;
    serial += 1;
    let changed = evaluate(
        g,
        &worker,
        request(
            &source,
            serial,
            &[0.6, 1.25, -0.25],
            0.8,
            false,
            Boundary::Repeat,
            None,
        ),
    );
    require(
        changed
            .iter()
            .zip(&first)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "stale height cache",
    )?;
    let stats = worker.diagnostics();
    require(
        stats.pyramid_cache_hits > cached.pyramid_cache_hits,
        "pyramid cache missed weight-only edit",
    )?;
    check(&mut checks, "cache_reuse");
    check(&mut checks, "determinism");
    let proxy_extent = Extent::new(2048, 16).unwrap();
    let bytes = fixture(2048, 16, "proxy");
    let image = reference::luminance(proxy_extent, &bytes);
    let source = install(&worker, proxy_extent, bytes);
    serial += 1;
    let mut proxy = request(
        &source,
        serial,
        &[0.0, 1.0],
        1.0,
        false,
        Boundary::Repeat,
        None,
    );
    proxy.quality = Quality::Proxy;
    proxy.extent = texelith_gpu::proxy_extent(proxy_extent).map_err(|e| e.to_string())?;
    compare(&evaluate(g, &worker, proxy), &image.reduce().pixels)?;
    serial += 1;
    compare(
        &evaluate(
            g,
            &worker,
            request(
                &source,
                serial,
                &[0.0, 1.0],
                1.0,
                false,
                Boundary::Repeat,
                None,
            ),
        ),
        &reference::reconstruct(&image, &[0.0, 1.0], 1.0, Boundary::Repeat).pixels,
    )?;
    check(&mut checks, "proxy_scaling");
    let extent = Extent::new(16, 1).unwrap();
    let bytes: Vec<u8> = [0, 10, 11, 32, 64, 128, 192, 255]
        .into_iter()
        .flat_map(|v| [[v, v, v, 0], [v, v, v, 255]])
        .flatten()
        .collect();
    let image = reference::luminance(extent, &bytes);
    let source = install(&worker, extent, bytes);
    serial += 1;
    let actual = evaluate(
        g,
        &worker,
        request(&source, serial, &[1.0], 1.0, false, Boundary::Repeat, None),
    );
    compare(&actual, &image.pixels)?;
    require(
        (actual[10] - 128.0 / 255.0).abs() > 0.1,
        "height reencoded as sRGB",
    )?;
    require(
        actual
            .as_chunks::<2>()
            .0
            .iter()
            .all(|p| (p[0] - p[1]).abs() < 2e-6),
        "height depends on alpha",
    )?;
    // A linear-height ramp uses the scalar source path without a transfer.
    let ramp = oracle::Image::new(17, 1, (0..17).map(|i| f64::from(i * 15) / 255.0).collect());
    let bytes: Vec<u8> = (0..17)
        .flat_map(|i| [i * 15, i * 15, i * 15, 255])
        .collect();
    let input = g.upload(ramp.extent, wgpu::TextureFormat::Rgba8Unorm, &bytes);
    let output = g.texture(ramp.extent, wgpu::TextureFormat::R32Float);
    g.dispatch(
        &g.executor
            .prepare(
                Pass::Source(InputInterpretation::LinearHeight),
                &input,
                &output,
                ramp.extent,
            )
            .map_err(|e| e.to_string())?,
    );
    compare(&g.read_scalar(&output), &ramp.pixels)?;
    check(&mut checks, "srgb_linear_ramp");
    require(
        worker.diagnostics().validation_errors == 0,
        "worker GPU validation error",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok((
        checks,
        stats.pyramid_cache_hits - cached.pyramid_cache_hits,
        cached.cache_hits - before.cache_hits,
    ))
}

fn remap(g: &gpu::Gpu, checks: &mut Vec<Value>) -> Result<()> {
    for (w, h, value, midlevel) in [
        (1, 1, -7.0, 0.5),
        (1, 17, 0.0, 0.25),
        (17, 1, 2.0, 0.75),
        (257, 193, 0.123, 0.5),
    ] {
        let image = oracle::Image::new(w, h, vec![value; (w * h) as usize]);
        remap_image(
            g,
            &image,
            midlevel,
            &vec![f64::from(midlevel); (w * h) as usize],
        )?;
    }
    check(checks, "flat_min_max");
    let mut pixels = vec![2.0; 257 * 193];
    pixels[0] = -2.0;
    *pixels.last_mut().unwrap() = 6.0;
    let image = oracle::Image::new(257, 193, pixels);
    remap_image(
        g,
        &image,
        0.5,
        &image
            .pixels
            .iter()
            .map(|v| (v + 2.0) / 8.0)
            .collect::<Vec<_>>(),
    )?;
    let adjacent = f32::from_bits(0.5f32.to_bits() + 1);
    remap_image(
        g,
        &oracle::Image::new(2, 1, vec![0.5, f64::from(adjacent)]),
        0.5,
        &[0.0, 1.0],
    )?;
    remap_image(
        g,
        &oracle::Image::new(3, 1, vec![-f64::from(f32::MAX), 0.0, f64::from(f32::MAX)]),
        0.5,
        &[0.0, 0.5, 1.0],
    )?;
    check(checks, "min_max_remap");
    Ok(())
}
fn remap_image(g: &gpu::Gpu, image: &oracle::Image, midlevel: f32, expected: &[f64]) -> Result<()> {
    let input = g.scalar(image);
    let output = g.texture(image.extent, wgpu::TextureFormat::R32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut node = PreparedMinMaxRemap::new(&g.executor, &input, &output, midlevel, pool.clone())
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    for pass in node.passes() {
        pass.encode(&mut encoder);
    }
    node.mark_submitted(FenceId(1)).map_err(|e| e.to_string())?;
    g.queue.submit([encoder.finish()]);
    g.wait();
    node.complete(FenceId(1));
    compare(&g.read_scalar(&output), expected)?;
    drop(node);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "remap scratch leaked",
    )
}
fn budget(g: &gpu::Gpu, checks: &mut Vec<Value>) -> Result<()> {
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
    let bytes = fixture(16, 16, "seeded");
    let image = reference::luminance(extent, &bytes);
    let source = install(&worker, extent, bytes);
    worker
        .set_resource_reservations(ResourceReservations {
            viewport_bytes: 60 * 1024,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
    worker
        .request_material(request(
            &source,
            1,
            &[1.0],
            1.0,
            false,
            Boundary::Repeat,
            None,
        ))
        .map_err(|e| e.to_string())?;
    let event = cases::wait(|| worker.try_event());
    require(
        matches!(event,WorkerEvent::BudgetExceeded{required_bytes,budget_bytes:65536} if required_bytes>65536),
        "expected real budget rejection",
    )?;
    require(
        worker.try_latest_material().is_none(),
        "budget failure published partial height",
    )?;
    worker
        .set_resource_reservations(ResourceReservations::default())
        .map_err(|e| e.to_string())?;
    compare(
        &evaluate(
            g,
            &worker,
            request(&source, 2, &[1.0], 1.0, false, Boundary::Repeat, None),
        ),
        &image.pixels,
    )?;
    require(
        worker.diagnostics().peak_bytes <= 65536,
        "worker exceeded low budget",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    check(checks, "low_budget");
    Ok(())
}

fn measure(g: &gpu::Gpu, node: &mut PreparedHeight, serial: &mut u64) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        // Deliberately recompute luminance and every pyramid level on every
        // sample, even for all-one weights. No warm output/pyramid cache here.
        for pass in node.passes() {
            pass.encode(&mut encoder);
        }
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
        json!({"kind":"completion_upper_bound","scope":"full_node","includes_pyramid":true,"warmups":10,
        "measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,
        "levels":node.levels(),"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
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
    let (mut checks, pyramid_hits, output_hits) = numerics(&g)?;
    remap(&g, &mut checks)?;
    budget(&g, &mut checks)?;
    let bytes = fixture(extent.width, extent.height, "seeded");
    let source = SourceDescriptor::from_rgba8(&bytes, extent, InputInterpretation::SrgbPhoto)
        .map_err(|e| e.to_string())?;
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let desc = |format| AllocationDesc::Texture {
        extent,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
    };
    let mut payload = Vec::new();
    for allocation in [
        desc(wgpu::TextureFormat::Rgba8Unorm),
        desc(wgpu::TextureFormat::Rgba16Float),
        desc(wgpu::TextureFormat::R32Float),
        AllocationDesc::Staging {
            bytes: u64::from(extent.width) * u64::from(extent.height) * 4 + 128,
        },
    ] {
        payload.push(
            pool.lock()
                .unwrap()
                .allocate(allocation)
                .map_err(|e| e.to_string())?,
        );
    }
    let raw_texture = g.texture(extent, wgpu::TextureFormat::Rgba8Unorm);
    let color_texture = g.texture(extent, wgpu::TextureFormat::Rgba16Float);
    let output_texture = g.texture(extent, wgpu::TextureFormat::R32Float);
    let raw = &raw_texture;
    let color = &color_texture;
    let output = &output_texture;
    g.queue.write_texture(
        raw.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * 4),
            rows_per_image: Some(extent.height),
        },
        raw.size(),
    );
    g.dispatch(
        &g.executor
            .prepare(Pass::ColorSource, raw, color, extent)
            .map_err(|e| e.to_string())?,
    );
    g.wait();
    pool.lock()
        .unwrap()
        .release(payload[3])
        .map_err(|e| e.to_string())?;
    let mut params = HeightParameters {
        band_weights: vec![1.0],
        coarse_weight: 1.0,
        remove_coarse: false,
        boundary: Boundary::Repeat,
    };
    let prepare = Instant::now();
    let mut node = PreparedHeight::new(&g.executor, &source, color, output, &params, pool.clone())
        .map_err(|e| e.to_string())?;
    let cold_prepare_ms = prepare.elapsed().as_secs_f64() * 1000.0;
    let mut serial = 0;
    let mut timing = measure(&g, &mut node, &mut serial)?;
    let levels = node.levels();
    let scratch = node.scratch_bytes();
    let peak = pool.lock().unwrap().diagnostics().peak_bytes;
    require(
        scratch <= 192 * 1024 * 1024,
        "height exceeded ADR scratch ceiling",
    )?;
    require(peak <= BUDGET, "height exceeded application budget")?;
    // There is one fixed five-tap filter and no adjustable quality/radius. The
    // largest accepted weight vector uses all 16 controls and the same full
    // chain. Measure it independently, without changing the default manifest.
    params.band_weights = (0..16)
        .map(|i| if i % 2 == 0 { 2.0 } else { -1.0 })
        .collect();
    params.remove_coarse = true;
    node.reweight(&g.executor, output, &params)
        .map_err(|e| e.to_string())?;
    let mut maximum = measure(&g, &mut node, &mut serial)?;
    maximum["parameters"] = json!({"band_weights":params.band_weights,"remove_coarse":true,"quality":"fixed binomial5 and all levels; no additional quality knob"});
    drop(node);
    for id in &payload[..3] {
        pool.lock()
            .unwrap()
            .release(*id)
            .map_err(|e| e.to_string())?;
    }
    drop(raw_texture);
    drop(color_texture);
    drop(output_texture);
    // The pyramid pool retains reusable backing after lease release. Destroy
    // that pool before starting a separately budgeted runtime measurement.
    drop(pool);
    // Separate real runtime run, after timed resources retire. Wait for a
    // height submission, supersede it, then check coherent final zero height.
    let worker = GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default())
        .map_err(|e| e.to_string())?;
    let source = install(&worker, extent, bytes);
    let stats = cases::cancel_runtime(&g, &worker, &source, 1000);
    worker.join().map_err(|e| e.to_string())?;
    check(&mut checks, "cancellation");
    timing["peak_bytes"] = json!(peak.max(stats.peak_bytes));
    timing["cancellation_errors"] = json!(stats.stale_publications);
    let manifest = json!({"band_weights":[1.0],"coarse_weight":1.0,"remove_coarse":false,"boundary":"repeat","remap":"none","filter_revision":1,"filter":"binomial5","size":extent.width,"seed":47});
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
    g.wait();
    require(
        pollster::block_on(scope.pop()).is_none(),
        "height GPU validation error",
    )?;
    let mut result = json!({"passed":true,"suite":"height","size":extent.width,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver,"driver_info":info.driver_info},
        "seed":47,"manifest":manifest,"manifest_hash":hash.iter().map(|v|format!("{v:02x}")).collect::<String>(),
        "manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four","checks":checks,"timing":timing,"maximum_quality":maximum,"cold_prepare_ms":cold_prepare_ms,
        "height":{"levels":levels,"budget_bytes":BUDGET,"scratch_peak_bytes":scratch,"peak_in_flight":stats.peak_in_flight,"cancellation_requests":stats.cancellation_requests,"cancelled_pass_batches":stats.cancelled_pass_batches,
            "cancelled_publications":stats.stale_publications,"validation_errors":stats.validation_errors,"low_budget_rejections":1,"shared_pyramid_cache_hits":pyramid_hits,"output_cache_hits":output_hits},
        "accounting_scope":"RGBA8 source and upload staging, binary16 linear color, R32 output, all luminance/pyramid/reconstruction scratch and uniforms; separate runtime cancellation peak. Tiny sequential oracle fixtures and driver-owned allocations excluded.",
        "timing_scope":"Prepared HeightFromAlbedo node: linear luminance, full Gaussian chain, all weighted reconstruction passes, CPU encoding, queue submission and completion wait. ColorSource decode is a separate graph node prepared before timing."});
    // Preserve real measurements even if a gate fails, so the orchestrator can
    // inspect the failure. The caller still exits unsuccessfully on validation.
    if let Err(error) = validate_height_report(&result) {
        result["passed"] = json!(false);
        result["failure"] = json!(error);
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
    )
}
