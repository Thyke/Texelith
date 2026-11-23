use super::{
    abi, oracle,
    support::{GIB, PORTS},
};
use crate::contract::aggregate_timings;
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{Boundary, NormalConvention};
use texelith_gpu::{
    CurvatureParameters, GpuExecutor, PreparedCurvature,
    pool::{AllocationDesc, FenceId, ResourcePool},
};

pub(super) fn wait(device: &wgpu::Device) -> Result<(), String> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}
fn measure(
    context: &crate::SuiteContext<'_>,
    nodes: &mut [PreparedCurvature],
    serial: &mut u64,
) -> Result<Value, String> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for _ in 0..110 {
        let start = Instant::now();
        let mut encoder = context.device.create_command_encoder(&Default::default());
        for node in nodes.iter() {
            for pass in node.passes() {
                pass.encode(&mut encoder);
            }
        }
        *serial += 1;
        let fence = FenceId(*serial);
        for node in nodes.iter_mut() {
            node.mark_submitted(fence).map_err(|e| e.to_string())?;
        }
        let submit = Instant::now();
        context.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        wait(&context.device)?;
        for node in nodes.iter_mut() {
            node.complete(fence);
        }
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","scope":"full_node","outputs":3,"warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}
fn read(context: &crate::SuiteContext<'_>, texture: &wgpu::Texture) -> Result<Vec<f32>, String> {
    let (tight, pitch) = abi::row_layout(texture.width(), 4)?;
    let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("curvature scalar readback"),
        size: u64::from(pitch) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    context.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    wait(&context.device)?;
    rx.recv_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let values = mapped
        .chunks_exact(pitch as usize)
        .flat_map(|r| {
            r[..tight as usize]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
        })
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}
pub(super) fn run(context: &crate::SuiteContext<'_>) -> Result<(Value, Value), String> {
    let e = context.extent;
    let pixels = u64::from(e.width) * u64::from(e.height);
    let pool = Arc::new(Mutex::new(ResourcePool::new(GIB)));
    // Reserve payloads and peak staging in the same ledger as prepared uniforms.
    // Upload staging retires before readback; one conservative token covers both.
    let staging =
        (pixels * 8).max(u64::from((e.width * 4).div_ceil(256) * 256) * u64::from(e.height));
    let mut reservations = Vec::new();
    for bytes in [pixels * 8, pixels * 4, pixels * 4, pixels * 4, staging] {
        reservations.push(
            pool.lock()
                .unwrap()
                .allocate(AllocationDesc::Staging { bytes })
                .map_err(|e| e.to_string())?,
        );
    }
    let texture = |format| {
        context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("curvature measured plane"),
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
    };
    // Exactly representable half components, amplitude 1/128. At 4K this
    // bounds |du-dv| by 64, whose R32 spacing (<=2^-17) fits the absolute 2e-5
    // oracle contract. The previous amplitude 1/2 could produce values above
    // 2048, where R32 spacing alone exceeds that contract. All pixels still
    // execute the same normalization, four taps and output branches; large
    // tilts and steep normals are checked separately in the numerical suite.
    let half = [
        0x0000u16, 0x1800, 0x1c00, 0x1e00, 0x2000, 0x9800, 0x9c00, 0x9e00, 0xa000,
    ];
    // Give Y one sixteenth the X amplitude: nonconstant axis derivatives cannot
    // cancel at the f32 noise floor and then saturate under f32::MAX scales.
    // Both positive and negative divergence remain present, with four taps per
    // pixel and two independently varying tangent components.
    let half_y = [
        0x0000u16, 0x0800, 0x0c00, 0x0e00, 0x1000, 0x8800, 0x8c00, 0x8e00, 0x9000,
    ];
    let sample = |x: u32, y: u32| [half[x as usize % 9], half_y[y as usize % 9], 0x3c00, 0x3c00];
    let input = texture(wgpu::TextureFormat::Rgba16Float);
    let data = oracle::image(e, sample);
    context.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 8),
            rows_per_image: Some(e.height),
        },
        input.size(),
    );
    wait(&context.device)?;
    drop(data);
    let outputs = PORTS.map(|_| texture(wgpu::TextureFormat::R32Float));
    let executor = pollster::block_on(GpuExecutor::new(
        context.device.clone(),
        context.queue.clone(),
    ))
    .map_err(|e| e.to_string())?;
    let mut serial = 0;
    let mut timings = Vec::new();
    for maximum in [false, true] {
        let controls = if maximum {
            oracle::Controls {
                edge: f32::MAX,
                cavity: f32::MAX,
                boundary: Boundary::Repeat,
            }
        } else {
            oracle::Controls::default()
        };
        let mut nodes = Vec::new();
        for (port, output) in PORTS.into_iter().zip(&outputs) {
            let parameters = CurvatureParameters::from_material(
                &controls.params(),
                NormalConvention::OpenGl,
                port,
            )
            .map_err(|e| e.to_string())?;
            nodes.push(
                PreparedCurvature::new(&executor, &input, output, &parameters, pool.clone())
                    .map_err(|e| e.to_string())?,
            );
        }
        let mut timing = measure(context, &mut nodes, &mut serial)?;
        // The input repeats every nine pixels; only boundary positions differ.
        // Build the full independent oracle once per configuration for readback.
        let normals = oracle::image(e, |x, y| {
            let n = sample(x, y);
            [abi::decode_half(n[0]), abi::decode_half(n[1]), 1.0]
        });
        let expected = oracle::curvature(&normals, e, NormalConvention::OpenGl, controls);
        for (output, reference) in outputs.iter().zip(&expected) {
            oracle::compare(&read(context, output)?, reference);
        }
        timing["edge_scale"] = json!(controls.edge);
        timing["cavity_scale"] = json!(controls.cavity);
        timings.push(timing);
        drop(nodes);
    }
    timings[0]["peak_bytes"] = json!(pool.lock().unwrap().diagnostics().peak_bytes);
    timings[1]["quality_note"] = json!(
        "No adjustable sample/filter quality; same three passes with maximum finite edge/cavity scales"
    );
    drop((input, outputs));
    for id in reservations {
        pool.lock()
            .unwrap()
            .release(id)
            .map_err(|e| e.to_string())?;
    }
    if pool.lock().unwrap().diagnostics().allocated_bytes != 0 {
        return Err("curvature harness scratch leak".into());
    }
    let maximum = timings.pop().unwrap();
    Ok((timings.pop().unwrap(), maximum))
}
