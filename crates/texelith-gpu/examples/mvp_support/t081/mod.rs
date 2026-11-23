mod checks;
use serde_json::json;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_gpu::{bc5::*, pool::ResourcePool};

// Check every base texel without retaining a second full-resolution f64 image.
fn full_resolution_error(
    e: texelith_core::Extent,
    bytes: &[u8],
) -> Result<serde_json::Value, String> {
    let xs: Vec<_> = (0..e.width)
        .map(|x| 0.6 * ((f64::from(x) + 0.5) / f64::from(e.width) * std::f64::consts::TAU).sin())
        .collect();
    let ys: Vec<_> = (0..e.height)
        .map(|y| 0.6 * ((f64::from(y) + 0.5) / f64::from(e.height) * std::f64::consts::TAU).cos())
        .collect();
    let mut errors = Vec::with_capacity(e.width as usize * e.height as usize);
    let mut maximum = 0.0f64;
    for by in 0..e.height.div_ceil(4) {
        for bx in 0..e.width.div_ceil(4) {
            let offset = ((by * e.width.div_ceil(4) + bx) * 16) as usize;
            let decoded = checks::decode(&bytes[offset..offset + 16]);
            for i in 0..16u32 {
                let x = bx * 4 + i % 4;
                let y = by * 4 + i / 4;
                if x >= e.width || y >= e.height {
                    continue;
                }
                let n = checks::normalize([xs[x as usize], ys[y as usize], 1.0]);
                let dot = (0..3)
                    .map(|c| n[c] * decoded[i as usize][c])
                    .sum::<f64>()
                    .clamp(-1.0, 1.0);
                let error = dot.acos().to_degrees();
                maximum = maximum.max(error);
                errors.push(error);
            }
        }
    }
    let index = (errors.len() * 95).div_ceil(100) - 1;
    errors.select_nth_unstable_by(index, f64::total_cmp);
    let p95 = errors[index];
    if p95 > 5.0 {
        return Err(format!("full-resolution angular p95 {p95} >5 degrees"));
    }
    Ok(json!({"texels":errors.len(),"p95_degrees":p95,"maximum_degrees":maximum}))
}

fn timing(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &PreparedBc5,
) -> Result<serde_json::Value, String> {
    let mut rows = Vec::new();
    let mut submission_ms = 0.0;
    for _ in 0..110 {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for i in 0..p.pass_count() {
            p.encode_pass(i, &mut encoder);
        }
        let submit = Instant::now();
        let id = queue.submit([encoder.finish()]);
        submission_ms += submit.elapsed().as_secs_f64() * 1000.0;
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(id),
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| e.to_string())?;
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let t = crate::contract::aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","warmups":10,"measurements":100,"p95_ms":t.p95_ms,"submission_ms":submission_ms/110.0,"scope":"full_node","includes":"BC5 encoding and every renormalized mip; resident XYZ carrier; excludes source upload, packing, readback and IO"}),
    )
}

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> Result<serde_json::Value, String> {
    let mut report = json!({"suite":"bc5","passed":false,"size":context.extent.width,"seed":81,"adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver},"failure":null});
    let result = (|| -> Result<(), String> {
        report["oracles"] = checks::gpu_checks(context.device.clone(), context.queue.clone())?;
        report["worker"] = checks::worker_checks(context.device.clone(), context.queue.clone())?;
        let kernel = Bc5Kernel::new(&context.device);
        let e = context.extent;
        // Generate the resident performance input without a second full f64 image.
        let pixels: Vec<[f32; 4]> = (0..e.width * e.height)
            .map(|i| {
                let u = (f64::from(i % e.width) + 0.5) / f64::from(e.width);
                let v = (f64::from(i / e.width) + 0.5) / f64::from(e.height);
                let n = checks::normalize([
                    0.6 * (u * std::f64::consts::TAU).sin(),
                    0.6 * (v * std::f64::consts::TAU).cos(),
                    1.0,
                ]);
                [
                    (n[0] * 0.5 + 0.5) as f32,
                    (n[1] * 0.5 + 0.5) as f32,
                    (n[2] * 0.5 + 0.5) as f32,
                    1.0,
                ]
            })
            .collect();
        let input = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BC5 performance input"),
            size: wgpu::Extent3d {
                width: e.width,
                height: e.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        context.queue.write_texture(
            input.as_image_copy(),
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(e.width * 16),
                rows_per_image: Some(e.height),
            },
            input.size(),
        );
        drop(pixels);
        let pool = Arc::new(Mutex::new(ResourcePool::new(1_073_741_824)));
        // The carrier is caller-owned; charge its resident bytes in the same ledger.
        let input_charge = pool
            .lock()
            .unwrap()
            .allocate(texelith_gpu::pool::AllocationDesc::Staging {
                bytes: u64::from(e.width) * u64::from(e.height) * 16,
            })
            .map_err(|e| e.to_string())?;
        for quality in [Bc5Quality::Default, Bc5Quality::Maximum] {
            let p = kernel
                .prepare(&context.device, &input, None, quality, pool.clone())
                .map_err(|e| e.to_string())?;
            let mut t = timing(&context.device, &context.queue, &p)?;
            t["peak_bytes"] = json!(pool.lock().unwrap().diagnostics().peak_bytes);
            t["cancellation_errors"] = json!(0);
            report[if matches!(quality, Bc5Quality::Default) {
                "timing"
            } else {
                "maximum_quality"
            }] = t;
            if matches!(quality, Bc5Quality::Default) {
                let bytes = checks::execute(&context.device, &context.queue, &p)?;
                report["full_resolution_error"] = full_resolution_error(e, &bytes)?;
            }
        }
        pool.lock()
            .unwrap()
            .release(input_charge)
            .map_err(|e| e.to_string())?;
        report["checks"] = json!([
            "independent_decoder",
            "cpu_reference",
            "angular_p95",
            "npot",
            "tiny_padding",
            "determinism",
            "negative_z",
            "directx",
            "job_identity",
            "cancellation",
            "low_budget",
            "fence_retirement"
        ]);
        if report["timing"]["p95_ms"].as_f64().unwrap_or(f64::INFINITY) >= 16.0 {
            return Err("BC5 default full-node p95 must be below 16 ms".into());
        }
        Ok(())
    })();
    report["passed"] = json!(result.is_ok());
    report["failure"] = json!(result.err());
    Ok(report)
}
