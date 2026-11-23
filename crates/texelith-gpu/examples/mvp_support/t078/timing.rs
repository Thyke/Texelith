use super::{fixtures::*, gpu::Gpu, reference, stress::BUDGET};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use texelith_core::*;
use texelith_gpu::{
    ExportPackingSource, PreparedExportPacking,
    pool::{AllocationDesc, FenceId, ResourcePool},
};

fn measure(
    g: &Gpu,
    node: &mut PreparedExportPacking,
    pool: &Arc<Mutex<ResourcePool>>,
) -> Result<Value, String> {
    let mut rows = Vec::new();
    let mut submissions = Vec::new();
    for serial in 1..=110 {
        let now = Instant::now();
        let mut encoder = g.device.create_command_encoder(&Default::default());
        for pass in node.passes() {
            pass.encode(&mut encoder);
        }
        node.mark_submitted(FenceId(serial))
            .map_err(|e| e.to_string())?;
        let submit = Instant::now();
        g.queue.submit([encoder.finish()]);
        submissions.push(submit.elapsed().as_secs_f64() * 1000.0);
        g.wait();
        node.complete(FenceId(serial));
        rows.push(vec![now.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = crate::contract::aggregate_timings(&rows, 10)?;
    Ok(
        json!({"kind":"completion_upper_bound","warmups":10,"measurements":timing.measurements,
        "p95_ms":timing.p95_ms,"submission_ms":submissions[10..].iter().sum::<f64>()/100.0,
        "scope":"full_node","includes_source":true,"cache_hits_used_for_timing":0,"cancellation_errors":0,
        "peak_bytes":pool.lock().unwrap().diagnostics().peak_bytes,
        "passes":node.passes().len(),"samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}

fn value(x: u32, y: u32, channel: u32) -> f32 {
    ((x * 37 + y * 113 + channel * 251 + 78) % 1024) as f32 / 1024.0
}

// Stream the mapped bytes: do not allocate a second CPU copy of the 4K plane.
fn verify(g: &Gpu, texture: &wgpu::Texture, plan: &ExportPlan) -> Result<(), String> {
    let width = texture.width();
    let height = texture.height();
    let pitch = (width * 16).div_ceil(256) * 256;
    let buffer = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("packing measured plane"),
        size: u64::from(pitch) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = g.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    g.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.wait();
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let out = &plan.outputs[0];
    let bits = out.format.bit_depth();
    let quant = out.quantization.unwrap_or_default();
    let sx = f64::from(plan.source_extent.width) / f64::from(width);
    let sy = f64::from(plan.source_extent.height) / f64::from(height);
    let mut max_lsb: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            for c in 0..4 {
                let offset = (y * pitch + x * 16 + c * 4) as usize;
                let actual = f64::from(f32::from_le_bytes(
                    mapped[offset..offset + 4].try_into().unwrap(),
                ));
                let expected = if c == 3 {
                    1.0
                } else {
                    let mut sum = 0.0;
                    // Keep, or exact pixel-footprint area reduction on each axis.
                    let left = f64::from(x) * sx;
                    let right = f64::from(x + 1) * sx;
                    let top = f64::from(y) * sy;
                    let bottom = f64::from(y + 1) * sy;
                    for iy in top.floor() as u32..bottom.ceil() as u32 {
                        for ix in left.floor() as u32..right.ceil() as u32 {
                            let weight = (right.min(f64::from(ix + 1)) - left.max(f64::from(ix)))
                                * (bottom.min(f64::from(iy + 1)) - top.max(f64::from(iy)))
                                / (sx * sy);
                            sum += f64::from(value(ix, iy, c)) * weight;
                        }
                    }
                    reference::quantize(sum, bits, [0.0, 1.0], quant.dither, x, y)
                };
                let error = (actual - expected).abs() * f64::from((1u32 << bits) - 1);
                if !actual.is_finite() || error > 1.01 {
                    return Err(format!(
                        "4K packing mismatch {x},{y},{c}: {actual} != {expected}"
                    ));
                }
                max_lsb = max_lsb.max(error);
            }
        }
    }
    drop(mapped);
    buffer.unmap();
    println!(
        "packing {}x{} complete-plane oracle: max error {max_lsb:.6} LSB",
        width, height
    );
    Ok(())
}

pub(super) fn run(g: &Gpu, extent: Extent) -> Result<(Value, Value), String> {
    let mut reports = Vec::new();
    for maximum in [false, true] {
        let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
        let pixels = u64::from(extent.width) * u64::from(extent.height);
        let mut p = plan(extent, vec![packed_outputs().remove(0)]);
        if maximum {
            p.size = SizePolicy::Resize {
                width: extent.width.div_ceil(2),
                height: extent.height.div_ceil(2),
            };
            p.outputs[0].format = ExportFormat::Png16;
            p.outputs[0].channels.push(constant(1.0));
            p.outputs[0].quantization = Some(IntegerQuantization {
                dither: IntegerDither::Ordered4x4,
                ..Default::default()
            });
        }
        let target = reference::destination(&p);
        let pitch = (u64::from(target.width) * 16).div_ceil(256) * 256;
        let payload = pool
            .lock()
            .unwrap()
            .allocate_plan(&[
                AllocationDesc::Staging { bytes: pixels * 12 }, // Three resident scalar maps.
                AllocationDesc::Staging {
                    bytes: u64::from(target.width) * u64::from(target.height) * 16,
                },
                AllocationDesc::Staging {
                    bytes: pitch * u64::from(target.height),
                },
            ])
            .map_err(|e| e.to_string())?;
        let upload = pool
            .lock()
            .unwrap()
            .allocate(AllocationDesc::Staging { bytes: pixels * 4 })
            .map_err(|e| e.to_string())?;
        let mut textures = Vec::new();
        for c in 0..3 {
            let data: Vec<f32> = (0..extent.height)
                .flat_map(|y| (0..extent.width).map(move |x| value(x, y, c)))
                .collect();
            let texture = g.texture(extent, wgpu::TextureFormat::R32Float);
            g.queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(extent.width * 4),
                    rows_per_image: Some(extent.height),
                },
                texture.size(),
            );
            g.wait();
            textures.push(texture);
        }
        pool.lock()
            .unwrap()
            .release(upload)
            .map_err(|e| e.to_string())?;
        let bindings: Vec<_> = [OutputPort::Ao, OutputPort::Roughness, OutputPort::Metallic]
            .into_iter()
            .zip(&textures)
            .map(|(port, texture)| ExportPackingSource {
                reference: reference(port),
                texture,
                boundary: Boundary::Clamp,
                normal_convention: NormalConvention::OpenGl,
            })
            .collect();
        let output = g.texture(target, wgpu::TextureFormat::Rgba32Float);
        let mut node = g
            .kernel
            .prepare(&g.executor, &p, 0, &bindings, &output, pool.clone())
            .map_err(|e| e.to_string())?;
        let mut timing = measure(g, &mut node, &pool)?;
        timing["source_extent"] = json!([extent.width, extent.height]);
        timing["output_extent"] = json!([target.width, target.height]);
        if maximum {
            timing["configuration"] = json!("rgba16_ordered4x4_resize");
        }
        verify(g, &output, &p)?;
        drop(node);
        drop(bindings);
        drop(textures);
        drop(output);
        for id in payload {
            pool.lock()
                .unwrap()
                .release(id)
                .map_err(|e| e.to_string())?;
        }
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        reports.push(timing);
    }
    let maximum = reports.pop().unwrap();
    Ok((reports.pop().unwrap(), maximum))
}
