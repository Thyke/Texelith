use super::{
    oracle,
    support::{BUDGET, Gpu},
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use texelith_gpu::{
    PreparedSeamHeatmap, TypedSourcePixels,
    pool::{FenceId, ResourcePool},
};

fn close(a: f64, b: f64) -> Result<(), String> {
    if !a.is_finite() || (a - b).abs() > 2e-5 * b.max(1.0) {
        return Err(format!("seam oracle mismatch: {a} vs {b}"));
    }
    Ok(())
}
fn inspect(
    gpu: &Gpu,
    w: usize,
    h: usize,
    values: TypedSourcePixels,
    levels: bool,
) -> Result<oracle::Diagnostic, String> {
    let (worker, bundle, kind) = gpu.material(w, h, values, levels);
    let input = bundle.outputs[&kind].texture();
    let original = gpu.read_bytes(input);
    let expected = oracle::inspect(w, h, &gpu.pixels(input));
    let output = gpu.texture(w, h, wgpu::TextureFormat::R32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut prepared = PreparedSeamHeatmap::new(&gpu.executor, input, &output, pool.clone())
        .map_err(|e| e.to_string())?;
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for pass in prepared.passes() {
        pass.encode(&mut encoder);
    }
    prepared
        .mark_submitted(FenceId(1))
        .map_err(|e| e.to_string())?;
    gpu.queue.submit([encoder.finish()]);
    let actual = prepared.read_metrics().map_err(|e| e.to_string())?;
    prepared.complete(FenceId(1));
    for (a, b) in [
        (f64::from(actual.boundary_rms), expected.boundary_rms),
        (f64::from(actual.interior_rms), expected.interior_rms),
        (f64::from(actual.ratio), expected.ratio),
    ] {
        close(a, b)?;
    }
    for (p, &e) in gpu.pixels(&output).iter().zip(&expected.heatmap) {
        close(p[0], e)?;
    }
    if original != gpu.read_bytes(input) {
        return Err("diagnostic mutated material pixels".into());
    }
    drop(prepared);
    if pool.lock().unwrap().diagnostics().allocated_bytes != 0 {
        return Err("diagnostic scratch leaked".into());
    }
    drop(bundle);
    worker.join().map_err(|e| e.to_string())?;
    Ok(expected)
}
fn scalar(gpu: &Gpu, w: usize, h: usize, values: Vec<f32>) -> Result<oracle::Diagnostic, String> {
    inspect(
        gpu,
        w,
        h,
        TypedSourcePixels::HeightF32(values.into()),
        false,
    )
}

pub(super) fn cancellation_budget(gpu: &Gpu) -> Result<(), String> {
    let input = gpu.texture(33, 19, wgpu::TextureFormat::R32Float);
    let output = gpu.texture(33, 19, wgpu::TextureFormat::R32Float);
    let low = Arc::new(Mutex::new(ResourcePool::new(0)));
    if PreparedSeamHeatmap::new(&gpu.executor, &input, &output, low.clone()).is_ok()
        || low.lock().unwrap().diagnostics().allocated_bytes != 0
    {
        return Err("low budget accepted/leaked".into());
    }
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut node = PreparedSeamHeatmap::new(&gpu.executor, &input, &output, pool.clone())
        .map_err(|e| e.to_string())?;
    if node.read_metrics().is_ok() {
        return Err("unsubmitted metrics accepted".into());
    }
    let generation = AtomicU64::new(1);
    for stop_after in 0..=node.passes().len() {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut checks = 0;
        let encoded = node.encode_if_current(&mut encoder, || {
            if checks == stop_after {
                generation.store(2, Ordering::Release);
            }
            checks += 1;
            generation.load(Ordering::Acquire) == 1
        });
        if encoded {
            return Err("stale generation encoded for publication".into());
        }
        drop(encoder);
        generation.store(1, Ordering::Release);
    }
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    if !node.encode_if_current(&mut encoder, || generation.load(Ordering::Acquire) == 1) {
        return Err("current generation rejected".into());
    }
    node.mark_submitted(FenceId(1)).map_err(|e| e.to_string())?;
    gpu.queue.submit([encoder.finish()]);
    generation.store(2, Ordering::Release);
    let _ = node.read_metrics().map_err(|e| e.to_string())?;
    let publish = generation.load(Ordering::Acquire) == 1;
    if publish {
        return Err("stale completion published".into());
    }
    // Dropping submitted state retains its charge until the queue fence retires.
    drop(node);
    if pool.lock().unwrap().diagnostics().allocated_bytes == 0 {
        return Err("submitted scratch retired early".into());
    }
    pool.lock().unwrap().complete(FenceId(1));
    if pool.lock().unwrap().diagnostics().allocated_bytes != 0 {
        return Err("cancelled scratch leaked".into());
    }
    Ok(())
}

pub(super) fn unsubmitted_cancellation_budget(gpu: &Gpu) -> Result<(), String> {
    let input = gpu.texture(33, 19, wgpu::TextureFormat::R32Float);
    let output = gpu.texture(33, 19, wgpu::TextureFormat::R32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    // Drain unrelated uploads before measuring. No submissions are made during
    // the cancellation loop: a pending queue write would otherwise be hidden.
    gpu.queue.submit([]);
    gpu.wait();
    let baseline = gpu
        .device
        .generate_allocator_report()
        .ok_or("cancellation regression requires native allocator reporting")?
        .total_allocated_bytes;
    for _ in 0..8 {
        for stop_after in 0..=3 {
            let node = PreparedSeamHeatmap::new(&gpu.executor, &input, &output, pool.clone())
                .map_err(|e| e.to_string())?;
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let mut checks = 0;
            if node.encode_if_current(&mut encoder, || {
                let current = checks < stop_after;
                checks += 1;
                current
            }) {
                return Err("cancelled preparation accepted".into());
            }
            drop(encoder);
            drop(node);
            if pool.lock().unwrap().diagnostics().allocated_bytes != 0 {
                return Err("unsubmitted cancellation leaked pool reservations".into());
            }
            gpu.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| e.to_string())?;
            let allocated = gpu
                .device
                .generate_allocator_report()
                .ok_or("native allocator report disappeared")?
                .total_allocated_bytes;
            if allocated > baseline {
                return Err(format!(
                    "unsubmitted cancellation retained unaccounted GPU memory: {allocated} > {baseline}"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn run(gpu: &Gpu) -> Result<Vec<serde_json::Value>, String> {
    for (w, h) in [
        (32, 24),
        (257, 193),
        (17, 11),
        (1, 29),
        (31, 1),
        (1, 1),
        (3, 2),
    ] {
        let flat = scalar(gpu, w, h, vec![0.375; w * h])?;
        if flat.boundary_rms != 0.0 || flat.ratio != 0.0 {
            return Err("nonzero flat diagnostic".into());
        }
        let mut seed = 60u32;
        scalar(
            gpu,
            w,
            h,
            oracle::image(w, h, |_, _| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 16) as f32 / 65535.0
            }),
        )?;
        scalar(
            gpu,
            w,
            h,
            oracle::image(
                w,
                h,
                |x, y| if x == w / 2 && y == h / 2 { 1.0 } else { 0.0 },
            ),
        )?;
        scalar(gpu, w, h, oracle::periodic(w, h))?;
    }
    for axis in 0..2 {
        let ramp = scalar(
            gpu,
            32,
            32,
            oracle::image(32, 32, |x, y| (if axis == 0 { x } else { y }) as f32 / 32.0),
        )?;
        close(ramp.ratio, 11.0)?;
        close(
            ramp.heatmap[if axis == 0 { 31 } else { 31 * 32 }],
            31.0 / 32.0,
        )?;
        let step = scalar(
            gpu,
            32,
            32,
            oracle::image(32, 32, |x, y| {
                if (if axis == 0 { x } else { y }) >= 16 {
                    1.0
                } else {
                    0.0
                }
            }),
        )?;
        close(step.boundary_rms, 0.25)?;
        close(step.interior_rms, 0.25)?;
        for (position, b, i) in [
            (3, 2.0_f64, 0.0_f64),
            (4, 1.0, 0.0),
            (5, 0.0, 0.0),
            (12, 0.0, 1.0),
            (13, 0.0, 2.0),
            (20, 0.0, 1.0),
            (21, 0.0, 0.0),
            (27, 0.0, 0.0),
            (28, 1.0, 0.0),
            (29, 2.0, 0.0),
        ] {
            let d = scalar(
                gpu,
                32,
                32,
                oracle::image(32, 32, |x, y| {
                    if (if axis == 0 { x } else { y }) == position {
                        1.0
                    } else {
                        0.0
                    }
                }),
            )?;
            close(d.boundary_rms, (b / 16.0).sqrt())?;
            close(d.interior_rms, (i / 16.0).sqrt())?;
        }
        let (w, h) = (256, 192);
        let mut p = oracle::periodic(w, h);
        let good = scalar(gpu, w, h, p.clone())?;
        for y in 0..h {
            for x in 0..w {
                if (axis == 0 && x == 0) || (axis == 1 && y == 0) {
                    p[y * w + x] += 1.0;
                }
            }
        }
        let bad = scalar(gpu, w, h, p)?;
        if good.boundary_rms >= 5e-4 || bad.boundary_rms <= 0.2 || bad.ratio <= 100.0 {
            return Err("periodic/injected seam score".into());
        }
    }
    let zero = scalar(
        gpu,
        32,
        32,
        oracle::image(32, 32, |x, _| if x == 0 { 1.0 } else { 0.0 }),
    )?;
    if zero.interior_rms != 0.0 || !zero.ratio.is_finite() || zero.ratio <= 1e7 {
        return Err("zero denominator".into());
    }
    inspect(
        gpu,
        33,
        19,
        TypedSourcePixels::HeightF32(
            oracle::image(33, 19, |x, _| 0.01 + 0.8 * x as f32 / 32.0).into(),
        ),
        true,
    )?;
    inspect(
        gpu,
        33,
        19,
        TypedSourcePixels::HeightU16(
            (0..33 * 19)
                .map(|i| (10000 + i * 7) as u16)
                .collect::<Vec<_>>()
                .into(),
        ),
        false,
    )?;
    inspect(
        gpu,
        32,
        24,
        TypedSourcePixels::Rgba8 {
            pixels: (0..32 * 24)
                .flat_map(|i| {
                    if i % 32 < 16 {
                        [128, 0, 0, 0]
                    } else {
                        [0, 76, 0, 255]
                    }
                })
                .collect::<Vec<_>>()
                .into(),
            interpretation: texelith_core::InputInterpretation::SrgbPhoto,
        },
        false,
    )?;
    cancellation_budget(gpu)?;
    unsubmitted_cancellation_budget(gpu)?;
    Ok([
        "cpu_oracle",
        "seam_location",
        "four_pixel_bands",
        "flat_zero_denominator",
        "periodic_and_injected_seam",
        "typed_material_output",
        "input_unchanged",
        "npot_singleton",
        "cancellation",
        "unsubmitted_cancellation_no_pending_uploads",
        "budget",
    ]
    .into_iter()
    .map(|name| serde_json::json!({"name":name,"passed":true}))
    .collect())
}
