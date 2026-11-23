use super::{oracle, support::Gpu};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use texelith_gpu::{
    PreparedSeamHeatmap,
    pool::{AllocationDesc, FenceId, ResourcePool},
};

pub(super) fn run(context: &crate::SuiteContext<'_>, gpu: &Gpu) -> Result<(Value, Value), String> {
    let e = context.extent;
    let pixels = u64::from(e.width) * u64::from(e.height);
    let pool = Arc::new(Mutex::new(ResourcePool::new(1_073_741_824)));
    // Caller-owned input/output and the larger upload or heatmap-readback peak
    // are explicitly reserved alongside the node's actual backed scratch.
    let mut reservations = Vec::new();
    for bytes in [
        pixels * 4,
        pixels * 4,
        (u64::from((e.width * 4).div_ceil(256) * 256) * u64::from(e.height)).max(pixels * 4),
    ] {
        reservations.push(
            pool.lock()
                .unwrap()
                .allocate(AllocationDesc::Staging { bytes })
                .map_err(|e| e.to_string())?,
        );
    }
    let input = gpu.texture(
        e.width as usize,
        e.height as usize,
        wgpu::TextureFormat::R32Float,
    );
    let output = gpu.texture(
        e.width as usize,
        e.height as usize,
        wgpu::TextureFormat::R32Float,
    );
    let data = crate::contract::fixture(e.width, e.height, 60)?;
    gpu.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 4),
            rows_per_image: Some(e.height),
        },
        input.size(),
    );
    let mut node = PreparedSeamHeatmap::new(&gpu.executor, &input, &output, pool.clone())
        .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut measure = || -> Result<Value, String> {
        let mut rows = Vec::new();
        let mut submissions = Vec::new();
        for _ in 0..110 {
            let start = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            if !node.encode_if_current(&mut encoder, || true) {
                return Err("current diagnostic cancelled".into());
            }
            serial += 1;
            let fence = FenceId(serial);
            node.mark_submitted(fence).map_err(|e| e.to_string())?;
            let submit = Instant::now();
            gpu.queue.submit([encoder.finish()]);
            submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
            let metrics = node.read_metrics().map_err(|e| e.to_string())?;
            node.complete(fence);
            if !metrics.ratio.is_finite() {
                return Err("nonfinite measured metrics".into());
            }
            rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
        }
        let t = crate::contract::aggregate_timings(&rows, 10)?;
        Ok(
            json!({"scope":"full_node","kind":"completion_upper_bound","warmups":10,"measurements":t.measurements,
            "p95_ms":t.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,
            "samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>(),
            "peak_bytes":pool.lock().unwrap().diagnostics().peak_bytes,"cancellation_errors":0,
            "includes":"encoding, all three compute passes, submission, scalar copy/map and completion"}),
        )
    };
    let timing = measure()?;
    let maximum = measure()?;
    let metrics = node.read_metrics().map_err(|e| e.to_string())?;
    // The measured 4K fixture is checked independently as well as small inputs.
    let heatmap = gpu.read_bytes(&output);
    let mut heatmap_matches = true;
    let expected = oracle::inspect_scalar_streaming(
        e.width as usize,
        e.height as usize,
        &data,
        |i, expected| {
            let actual = f32::from_le_bytes(heatmap[i * 4..i * 4 + 4].try_into().unwrap());
            heatmap_matches &= actual.is_finite() && (f64::from(actual) - expected).abs() <= 2e-5;
        },
    );
    for (a, b) in [
        (f64::from(metrics.boundary_rms), expected[0]),
        (f64::from(metrics.interior_rms), expected[1]),
        (f64::from(metrics.ratio), expected[2]),
    ] {
        if (a - b).abs() > 2e-5 * b.max(1.0) {
            return Err("4K seam metrics mismatch".into());
        }
    }
    if !heatmap_matches {
        return Err("4K heatmap mismatch".into());
    }
    drop(heatmap);
    if gpu.read_bytes(&input) != bytemuck::cast_slice::<_, u8>(&data) {
        return Err("4K source mutated".into());
    }
    drop(node);
    for id in reservations {
        pool.lock()
            .unwrap()
            .release(id)
            .map_err(|e| e.to_string())?;
    }
    if pool.lock().unwrap().diagnostics().allocated_bytes != 0 {
        return Err("measured scratch leak".into());
    }
    Ok((timing, maximum))
}
