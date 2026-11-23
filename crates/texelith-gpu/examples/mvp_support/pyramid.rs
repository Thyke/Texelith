//! Hardware validation of the shared primitives and the budgeted Gaussian chain.
use super::contract::{aggregate_timings, fixture, validate_pyramid_report};
use crate::reference::abi;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputInterpretation, NodeId, SourceDescriptor,
};
use texelith_gpu::{
    Axis, GpuExecutor, Pass, PreparedPyramid, PyramidPass as P, Quality, RenderRequest,
    TypedSourcePixels,
    cache::{CacheAddress, EvaluationCache},
    pool::{AllocationDesc, FenceId, ResourcePool},
    scheduler::{Completion, Scheduler},
};
#[allow(dead_code)]
#[path = "../../tests/mvp_t046/gpu.rs"]
mod gpu;
#[allow(dead_code)]
#[path = "../../tests/mvp_t046/oracle.rs"]
pub(super) mod oracle;
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn compare(actual: &[f32], expected: &[f64]) -> Result<()> {
    require(
        actual.len() == expected.len()
            && !actual.is_empty()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && (f64::from(*a) - b).abs() <= 2e-5),
        "pyramid CPU oracle mismatch",
    )
}
fn primitive(
    g: &gpu::Gpu,
    op: P,
    input: &wgpu::Texture,
    coarse: Option<&wgpu::Texture>,
    extent: Extent,
    boundary: Boundary,
) -> Result<wgpu::Texture> {
    let output = g.texture(extent, wgpu::TextureFormat::R32Float);
    let pass = g
        .executor
        .prepare_pyramid(op, input, coarse, &output, boundary)
        .map_err(|e| e.to_string())?;
    g.dispatch(&pass);
    Ok(output)
}
fn numerics(g: &gpu::Gpu) -> Result<Vec<Value>> {
    let mut checks = Vec::new();
    for (w, h, kind) in [
        (1, 1, "constant"),
        (9, 7, "constant"),
        (1, 17, "singleton_axes"),
        (17, 1, "singleton_axes"),
        (31, 17, "ramp"),
        (5, 3, "impulse"),
        (257, 193, "npot_257x193"),
    ] {
        let data: Vec<f64> = fixture(w, h, 42)?
            .into_iter()
            .enumerate()
            .map(|(i, v)| match kind {
                "constant" => -0.375,
                "ramp" => i as f64 / f64::from(w * h) - 0.5,
                "impulse" => {
                    if i == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => f64::from(v) * 2.0 - 1.0,
            })
            .collect();
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            let mut refs = vec![oracle::Image::new(w, h, data.clone())];
            let mut levels = vec![g.scalar(&refs[0])];
            let mut bands = Vec::new();
            while refs.last().unwrap().extent != Extent::new(1, 1).unwrap() {
                let last = refs.last().unwrap();
                let xr = last.gaussian(false, boundary);
                let yr = xr.gaussian(true, boundary);
                let down = yr.reduce();
                let x = primitive(
                    g,
                    P::Gaussian(Axis::X),
                    levels.last().unwrap(),
                    None,
                    last.extent,
                    boundary,
                )?;
                compare(&g.read_scalar(&x), &xr.pixels)?;
                let y = primitive(g, P::Gaussian(Axis::Y), &x, None, last.extent, boundary)?;
                compare(&g.read_scalar(&y), &yr.pixels)?;
                let d = primitive(g, P::AreaDownsample, &y, None, down.extent, boundary)?;
                compare(&g.read_scalar(&d), &down.pixels)?;
                let up = primitive(g, P::BilinearUpsample, &d, None, last.extent, boundary)?;
                compare(
                    &g.read_scalar(&up),
                    &down.expand(last.extent, boundary).pixels,
                )?;
                let band = primitive(
                    g,
                    P::LaplacianBand,
                    levels.last().unwrap(),
                    Some(&d),
                    last.extent,
                    boundary,
                )?;
                let br: Vec<_> = last
                    .pixels
                    .iter()
                    .zip(down.expand(last.extent, boundary).pixels)
                    .map(|(a, b)| a - b)
                    .collect();
                compare(&g.read_scalar(&band), &br)?;
                let weighted = primitive(
                    g,
                    P::WeightedReconstruct {
                        band_weight: 0.25,
                        coarse_weight: 0.5,
                    },
                    &band,
                    Some(&d),
                    last.extent,
                    boundary,
                )?;
                let wr: Vec<_> = br
                    .iter()
                    .zip(down.expand(last.extent, boundary).pixels)
                    .map(|(a, b)| 0.25 * a + 0.5 * b)
                    .collect();
                compare(&g.read_scalar(&weighted), &wr)?;
                bands.push(band);
                refs.push(down);
                levels.push(d);
            }
            let mut reconstruction = levels.last().unwrap().clone();
            for i in (0..bands.len()).rev() {
                reconstruction = primitive(
                    g,
                    P::WeightedReconstruct {
                        band_weight: 1.0,
                        coarse_weight: 1.0,
                    },
                    &bands[i],
                    Some(&reconstruction),
                    refs[i].extent,
                    boundary,
                )?;
                compare(&g.read_scalar(&reconstruction), &refs[i].pixels)?;
            }
            compare(&g.read_scalar(&reconstruction), &data)?;
            let a = primitive(
                g,
                P::Gaussian(Axis::X),
                &levels[0],
                None,
                refs[0].extent,
                boundary,
            )?;
            let b = primitive(
                g,
                P::Gaussian(Axis::X),
                &levels[0],
                None,
                refs[0].extent,
                boundary,
            )?;
            require(g.read(&a) == g.read(&b), "nondeterministic pyramid")?;
            if kind == "npot_257x193" {
                let source = TypedSourcePixels::HeightF32(Arc::from(
                    data.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                ))
                .descriptor(refs[0].extent)
                .map_err(|e| e.to_string())?;
                let pool = Arc::new(Mutex::new(ResourcePool::new(1_073_741_824)));
                let source_id = pool
                    .lock()
                    .unwrap()
                    .allocate(AllocationDesc::Texture {
                        extent: source.extent,
                        format: wgpu::TextureFormat::R32Float,
                        usage: levels[0].usage(),
                    })
                    .map_err(|e| e.to_string())?;
                let mut chain =
                    PreparedPyramid::new(&g.executor, &source, &levels[0], boundary, pool.clone())
                        .map_err(|e| e.to_string())?;
                let mut encoder = g.device.create_command_encoder(&Default::default());
                for pass in chain.passes() {
                    pass.encode(&mut encoder);
                }
                chain
                    .mark_submitted(FenceId(1))
                    .map_err(|e| e.to_string())?;
                g.queue.submit([encoder.finish()]);
                wait(&g.device)?;
                chain.complete(FenceId(1), true);
                for (texture, expected) in chain.levels().iter().zip(&refs) {
                    compare(&g.read_scalar(texture), &expected.pixels)?;
                }
                require(
                    chain.levels().len() == refs.len(),
                    "budgeted NPOT chain extent mismatch",
                )?;
                drop(chain);
                pool.lock()
                    .unwrap()
                    .release(source_id)
                    .map_err(|e| e.to_string())?;
            }
        }
        checks.push(json!({"name":kind,"passed":true}));
    }
    for name in [
        "repeat_clamp",
        "manual_bilinear",
        "reconstruction",
        "determinism",
    ] {
        checks.push(json!({"name":name,"passed":true}));
    }
    let e = Extent::new(4, 1).unwrap();
    let bytes = [
        0, 10, 11, 0, 128, 64, 255, 1, 255, 128, 0, 255, 10, 11, 12, 99,
    ];
    let input = g.upload(e, wgpu::TextureFormat::Rgba8Unorm, &bytes);
    let linear = |v: u8| {
        let v = f64::from(v) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    for op in [
        Pass::ColorSource,
        Pass::Source(InputInterpretation::SrgbPhoto),
        Pass::Source(InputInterpretation::LinearHeight),
    ] {
        let output = g.texture(e, op.output_format());
        g.dispatch(
            &g.executor
                .prepare(op, &input, &output, e)
                .map_err(|e| e.to_string())?,
        );
        if matches!(op, Pass::ColorSource) {
            for (p, c) in g
                .read_rgba(&output)
                .iter()
                .zip(bytes.as_chunks::<4>().0.iter())
            {
                require(
                    p[3] == 1.0 && (0..3).all(|i| (p[i] - linear(c[i])).abs() <= 0.0005),
                    "color transfer or alpha",
                )?;
            }
        } else {
            let expected: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| {
                    if matches!(op, Pass::Source(InputInterpretation::SrgbPhoto)) {
                        0.2126 * linear(c[0]) + 0.7152 * linear(c[1]) + 0.0722 * linear(c[2])
                    } else {
                        f64::from(c[0]) / 255.0
                    }
                })
                .collect();
            compare(&g.read_scalar(&output), &expected)?;
        }
        let mut opaque = bytes;
        for p in opaque.as_chunks_mut::<4>().0 {
            p[3] = 255;
        }
        let input2 = g.upload(e, wgpu::TextureFormat::Rgba8Unorm, &opaque);
        let output2 = g.texture(e, op.output_format());
        g.dispatch(
            &g.executor
                .prepare(op, &input2, &output2, e)
                .map_err(|e| e.to_string())?,
        );
        require(g.read(&output) == g.read(&output2), "alpha changed data")?;
    }
    checks.push(json!({"name":"source_transfer_alpha","passed":true}));
    let hash = |v| {
        TypedSourcePixels::HeightF32(Arc::from(v))
            .descriptor(Extent::new(2, 1).unwrap())
            .map(|d| d.content_hash)
            .map_err(|e| e.to_string())
    };
    require(
        hash([-0.0, -0.25])? == hash([0.0, -0.25])?
            && hash([-0.25, 2.0])? != hash([0.25, 2.0])?
            && hash([-0.25, 2.0])? != hash([-0.25, f32::from_bits(2.0f32.to_bits() + 1)])?,
        "source hash collision",
    )?;
    checks.push(json!({"name":"signed_source_hashes","passed":true}));
    Ok(checks)
}
fn wait(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn measure(g: &gpu::Gpu, chain: &mut PreparedPyramid, serial: &mut u64) -> Result<Value> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let start = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        for pass in chain.passes() {
            pass.encode(&mut encoder);
        }
        *serial += 1;
        let fence = FenceId(*serial);
        chain.mark_submitted(fence).map_err(|e| e.to_string())?;
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        wait(&g.device)?;
        chain.complete(fence, true);
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let t = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":t.measurements,
        "p95_ms":t.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"levels":chain.levels().len()}),
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
    let data = fixture(extent.width, extent.height, 42)?;
    let source = TypedSourcePixels::HeightF32(Arc::from(data.clone()))
        .descriptor(extent)
        .map_err(|e| e.to_string())?;
    let pool = Arc::new(Mutex::new(ResourcePool::new(1_073_741_824)));
    let input_desc = AllocationDesc::Texture {
        extent,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    };
    let source_id = pool
        .lock()
        .unwrap()
        .allocate(input_desc)
        .map_err(|e| e.to_string())?;
    let staging_id = pool
        .lock()
        .unwrap()
        .allocate(AllocationDesc::Staging {
            bytes: input_desc.byte_len().map_err(|e| e.to_string())?,
        })
        .map_err(|e| e.to_string())?;
    let input = g.upload(
        extent,
        wgpu::TextureFormat::R32Float,
        bytemuck::cast_slice(&data),
    );
    g.queue.submit([]);
    wait(&g.device)?;
    pool.lock()
        .unwrap()
        .release(staging_id)
        .map_err(|e| e.to_string())?;
    let low = Arc::new(Mutex::new(ResourcePool::new(64)));
    require(
        PreparedPyramid::new(&g.executor, &source, &input, Boundary::Repeat, low).is_err(),
        "low budget accepted",
    )?;
    checks.push(json!({"name":"low_budget","passed":true}));
    let mut chain =
        PreparedPyramid::new(&g.executor, &source, &input, Boundary::Repeat, pool.clone())
            .map_err(|e| e.to_string())?;
    let request = |id| RenderRequest {
        request_id: id,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: id,
            graph: Graph::new(
                vec![texelith_core::Node {
                    id: NodeId(1),
                    input: None,
                    params: texelith_core::NodeParams::Source(source.clone()),
                }],
                NodeId(1),
            )
            .unwrap(),
        }),
        output: NodeId(1),
        extent,
        quality: Quality::Full,
    };
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1)).map_err(|e| e.to_string())?;
    let work = scheduler.start_next().ok_or("no scheduled pyramid")?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    if let Some(pass) = chain.passes().first() {
        pass.encode(&mut encoder);
    }
    chain
        .mark_submitted(FenceId(1))
        .map_err(|e| e.to_string())?;
    g.queue.submit([encoder.finish()]);
    scheduler.request(request(2)).map_err(|e| e.to_string())?;
    require(
        !scheduler.should_continue(work.id),
        "obsolete pyramid continued",
    )?;
    wait(&g.device)?;
    chain.complete(FenceId(1), false);
    require(
        scheduler.complete(work.id) == Completion::Discarded
            && scheduler.try_latest().is_none()
            && !chain.is_ready(),
        "cancelled pyramid published",
    )?;
    checks.push(json!({"name":"cancellation","passed":true}));
    let mut serial = 1;
    let mut timing = measure(&g, &mut chain, &mut serial)?;
    let maximum = measure(&g, &mut chain, &mut serial)?;
    require(chain.is_ready(), "pyramid cache not ready")?;
    let scratch = chain.scratch_bytes();
    let key = chain.key;
    let mut cache = EvaluationCache::default();
    let address = CacheAddress {
        content: key,
        format: wgpu::TextureFormat::R32Float,
    };
    cache.insert(address, Arc::new(chain));
    let first = cache.get(&address).ok_or("missing shared pyramid")?.clone();
    let second = cache
        .get(&address)
        .ok_or("missing second consumer")?
        .clone();
    require(
        Arc::ptr_eq(&first, &second) && first.is_ready(),
        "pyramid was recomputed per consumer",
    )?;
    checks.push(json!({"name":"shared_cache","passed":true}));
    let peak = pool.lock().unwrap().diagnostics().peak_bytes;
    timing["peak_bytes"] = json!(peak);
    timing["cancellation_errors"] = json!(0);
    let manifest = json!({"filter":"binomial5","filter_revision":1,"boundary":"repeat","band_weight":1.0,"levels":first.levels().len(),
        "maximum_quality":"same fixed five-tap filter and all levels; no adjustable quality in this primitive","size":extent.width,"seed":42});
    let mut bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &bytes,
        Extent::new((bytes.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash;
    let hash = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    wait(&g.device)?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "pyramid GPU validation error",
    )?;
    let stats = scheduler.diagnostics();
    let result = json!({"passed":true,"suite":"pyramid","size":extent.width,"adapter":{"name":info.name,"backend":format!("{:?}",info.backend)},
        "seed":42,"manifest":manifest,"manifest_hash":hash,"checks":checks,"timing":timing,"maximum_quality":maximum,
        "pyramid":{"levels":first.levels().len(),"budget_bytes":1_073_741_824u64,"scratch_peak_bytes":scratch,
            "peak_in_flight":stats.peak_in_flight,"cancellation_requests":stats.cancellation_requests,"cancelled_publications":stats.stale_publications,
            "validation_errors":0,"low_budget_rejections":1,"shared_cache_hits":2},
        "accounting_scope":"source, upload staging, persistent Gaussian levels, one scratch image per level, all NodeUniforms; excludes driver-owned allocations and tiny oracle fixtures"});
    validate_pyramid_report(&result)?;
    drop(first);
    drop(second);
    cache.clear();
    pool.lock()
        .unwrap()
        .release(source_id)
        .map_err(|e| e.to_string())?;
    Ok(result)
}
