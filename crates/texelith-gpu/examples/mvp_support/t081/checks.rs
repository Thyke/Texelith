//! Independent decoder, vector-area mip oracle and live worker checks.
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::*;
use texelith_gpu::{
    bc5::*,
    pool::{FenceId, ResourcePool},
    *,
};
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

pub fn decode_rg(block: &[u8]) -> [[f64; 2]; 16] {
    let mut xy = [[0.0; 2]; 16];
    for c in 0..2 {
        let bytes = &block[c * 8..c * 8 + 8];
        let a = f64::from(bytes[0]);
        let b = f64::from(bytes[1]);
        let mut table = [a, b, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0];
        let denominator = if a > b { 7 } else { 5 };
        for k in 1..denominator {
            table[k + 1] = ((denominator - k) as f64 * a + k as f64 * b) / denominator as f64;
        }
        let mut bits = 0u64;
        for (i, v) in bytes[2..].iter().enumerate() {
            bits |= u64::from(*v) << (i * 8);
        }
        for i in 0..16 {
            xy[i][c] = table[((bits >> (i * 3)) & 7) as usize] / 127.5 - 1.0;
        }
    }
    xy
}
pub fn decode(block: &[u8]) -> [[f64; 3]; 16] {
    decode_rg(block).map(|[x, y]| normalize([x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]))
}
pub fn normalize(n: [f64; 3]) -> [f64; 3] {
    let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
    if length < 1e-20 {
        [0.0, 0.0, 1.0]
    } else {
        n.map(|v| v / length)
    }
}
pub fn fixture(e: Extent, steep: bool) -> Vec<[f64; 3]> {
    (0..e.width * e.height)
        .map(|i| {
            let u = (f64::from(i % e.width) + 0.5) / f64::from(e.width);
            let v = (f64::from(i / e.width) + 0.5) / f64::from(e.height);
            let k = if steep { 8.0 } else { 0.6 };
            normalize([
                k * (u * std::f64::consts::TAU).sin(),
                k * (v * std::f64::consts::TAU).cos(),
                1.0,
            ])
        })
        .collect()
}
pub fn mip_oracle(e: Extent, values: &[[f64; 3]]) -> (Extent, Vec<[f64; 3]>) {
    let next = Extent::new((e.width / 2).max(1), (e.height / 2).max(1)).unwrap();
    let mut output = Vec::new();
    for y in 0..next.height {
        for x in 0..next.width {
            let x0 = f64::from(x) * f64::from(e.width) / f64::from(next.width);
            let x1 = f64::from(x + 1) * f64::from(e.width) / f64::from(next.width);
            let y0 = f64::from(y) * f64::from(e.height) / f64::from(next.height);
            let y1 = f64::from(y + 1) * f64::from(e.height) / f64::from(next.height);
            let mut n = [0.0; 3];
            for sy in y0.floor() as u32..y1.ceil() as u32 {
                for sx in x0.floor() as u32..x1.ceil() as u32 {
                    let w = (x1.min(f64::from(sx + 1)) - x0.max(f64::from(sx)))
                        * (y1.min(f64::from(sy + 1)) - y0.max(f64::from(sy)));
                    for c in 0..3 {
                        n[c] += values[(sy * e.width + sx) as usize][c] * w;
                    }
                }
            }
            output.push(normalize(n));
        }
    }
    (next, output)
}
pub fn angular(e: Extent, expected: &[[f64; 3]], bytes: &[u8]) -> Result<(f64, f64)> {
    let mut errors = Vec::new();
    require(
        bytes.len() == (e.width.div_ceil(4) * e.height.div_ceil(4) * 16) as usize,
        "BC5 block byte count",
    )?;
    for by in 0..e.height.div_ceil(4) {
        for bx in 0..e.width.div_ceil(4) {
            let offset = ((by * e.width.div_ceil(4) + bx) * 16) as usize;
            let block = decode(&bytes[offset..offset + 16]);
            for i in 0..16u32 {
                let x = bx * 4 + i % 4;
                let y = by * 4 + i / 4;
                if x < e.width && y < e.height {
                    let n = expected[(y * e.width + x) as usize];
                    let dot = (0..3)
                        .map(|c| n[c] * block[i as usize][c])
                        .sum::<f64>()
                        .clamp(-1.0, 1.0);
                    errors.push(dot.acos().to_degrees());
                }
            }
        }
    }
    errors.sort_by(f64::total_cmp);
    Ok((
        errors[(errors.len() * 95).div_ceil(100) - 1],
        *errors.last().unwrap(),
    ))
}
pub fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    e: Extent,
    normals: &[[f64; 3]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("T-081 oracle input"),
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
    let pixels: Vec<[f32; 4]> = normals
        .iter()
        .map(|n| {
            [
                (n[0] * 0.5 + 0.5) as f32,
                (n[1] * 0.5 + 0.5) as f32,
                (n[2] * 0.5 + 0.5) as f32,
                1.0,
            ]
        })
        .collect();
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.width * 16),
            rows_per_image: Some(e.height),
        },
        texture.size(),
    );
    texture
}
pub fn execute(device: &wgpu::Device, queue: &wgpu::Queue, p: &PreparedBc5) -> Result<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-081 oracle readback"),
        size: p.staging_bytes(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for i in 0..p.pass_count() {
        p.encode_pass(i, &mut encoder);
    }
    p.encode_readback(&mut encoder, &staging);
    let submission = queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        })
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let result = p.readback_bytes(
        &staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?,
    );
    staging.unmap();
    result.map_err(|e| e.to_string())
}
pub fn gpu_checks(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<serde_json::Value> {
    let kernel = Bc5Kernel::new(&device);
    let pool = Arc::new(Mutex::new(ResourcePool::new(1024 * 1024 * 1024)));
    let contract: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../docs/evidence/T-081/bc5-contract.json"
    ))
    .map_err(|e| e.to_string())?;
    let flat = vec![[0.0, 0.0, 1.0]; 16];
    let texture = upload(&device, &queue, Extent::new(4, 4).unwrap(), &flat);
    let prepared = kernel
        .prepare(&device, &texture, None, Bc5Quality::Default, pool.clone())
        .map_err(|e| e.to_string())?;
    let bytes = execute(&device, &queue, &prepared)?;
    let expected: Vec<u8> = contract["block_bytes_each_mip"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    require(
        bytes == expected.repeat(3),
        "committed BC5 block/mip contract",
    )?;
    let (p95, max) = angular(Extent::new(4, 4).unwrap(), &flat, &bytes[..16])?;
    require(
        (p95 - contract["angular_error_degrees"]["p95"].as_f64().unwrap()).abs() < 1e-6
            && (max - p95).abs() < 1e-6,
        "committed BC5 decoded error metrics",
    )?;
    drop(prepared);
    let mut seed = 81u32;
    for case in 0..24 {
        let pixels: [[u8; 2]; 16] = std::array::from_fn(|i| {
            std::array::from_fn(|c| {
                if case == 0 {
                    64 + (if c == 0 { i % 4 } else { i / 4 }) as u8 * 32
                } else {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    64 + (seed >> 25) as u8
                }
            })
        });
        let values: Vec<_> = pixels
            .iter()
            .map(|p| {
                let x = f64::from(p[0]) / 127.5 - 1.0;
                let y = f64::from(p[1]) / 127.5 - 1.0;
                [x, y, (1.0 - x * x - y * y).sqrt()]
            })
            .collect();
        let input = upload(&device, &queue, Extent::new(4, 4).unwrap(), &values);
        let prepared = kernel
            .prepare(&device, &input, None, Bc5Quality::Maximum, pool.clone())
            .map_err(|e| e.to_string())?;
        let bytes = execute(&device, &queue, &prepared)?;
        let cpu = decode_rg(&encode_block(&pixels));
        let gpu = decode_rg(&bytes[..16]);
        let error = |decoded: &[[f64; 2]; 16]| {
            (0..16)
                .flat_map(|i| {
                    (0..2).map(move |c| {
                        (decoded[i][c] * 127.5 + 127.5 - f64::from(pixels[i][c])).powi(2)
                    })
                })
                .sum::<f64>()
        };
        require(
            error(&gpu) <= error(&cpu) + 1.0,
            "seeded maximum-quality GPU/CPU error bound",
        )?;
    }
    let mut smooth_p95 = 0.0f64;
    let mut smooth_max = 0.0f64;
    let mut steep_max = 0.0f64;
    for (w, h) in [(4, 4), (12, 20), (257, 193), (1, 13), (13, 1), (3, 2)] {
        let e = Extent::new(w, h).unwrap();
        for steep in [false, true] {
            let values = fixture(e, steep);
            let input = upload(&device, &queue, e, &values);
            let prepared = kernel
                .prepare(&device, &input, None, Bc5Quality::Default, pool.clone())
                .map_err(|e| e.to_string())?;
            let bytes = execute(&device, &queue, &prepared)?;
            require(
                bytes == execute(&device, &queue, &prepared)?,
                "GPU determinism",
            )?;
            let mut expected = values.clone();
            let mut extent = e;
            for mip in prepared.mips() {
                require(mip.extent == extent, "DDS logical mip dimensions")?;
                let block = &bytes[mip.offset as usize..(mip.offset + mip.byte_len) as usize];
                let (p95, max) = angular(extent, &expected, block)?;
                if steep {
                    steep_max = steep_max.max(max);
                } else {
                    smooth_p95 = smooth_p95.max(p95);
                    smooth_max = smooth_max.max(max);
                }
                // Odd/singleton mip blocks must replicate valid edge texels.
                for by in 0..extent.height.div_ceil(4) {
                    for bx in 0..extent.width.div_ceil(4) {
                        let at = ((by * extent.width.div_ceil(4) + bx) * 16) as usize;
                        let decoded = decode(&block[at..at + 16]);
                        for i in 0..16u32 {
                            let local_x = (extent.width - 1 - bx * 4).min(i % 4);
                            let local_y = (extent.height - 1 - by * 4).min(i / 4);
                            require(
                                decoded[i as usize] == decoded[(local_y * 4 + local_x) as usize],
                                "tiny mip replicated edge",
                            )?;
                        }
                    }
                }
                (extent, expected) = mip_oracle(extent, &expected);
            }
            // CPU maximum search is an independent upper bound on channel SSE;
            // default has a documented angular bound rather than byte identity.
            if w == 4 && h == 4 {
                let pixels = std::array::from_fn(|i| {
                    std::array::from_fn(|c| ((values[i][c] * 0.5 + 0.5) * 255.0).round() as u8)
                });
                let cpu = encode_block(&pixels);
                let maximum = kernel
                    .prepare(&device, &input, None, Bc5Quality::Maximum, pool.clone())
                    .map_err(|e| e.to_string())?;
                let max_bytes = execute(&device, &queue, &maximum)?;
                let cpu_decoded = decode_rg(&cpu);
                let gpu_decoded = decode_rg(&max_bytes[..16]);
                // Exact rational GPU distances can break a CPU f32 tie differently.
                let sse = |ns: &[[f64; 2]; 16]| {
                    (0..16)
                        .flat_map(|i| {
                            (0..2).map(move |c| {
                                (ns[i][c] * 127.5 + 127.5 - f64::from(pixels[i][c])).powi(2)
                            })
                        })
                        .sum::<f64>()
                };
                require(
                    sse(&gpu_decoded) <= sse(&cpu_decoded) + 1.0,
                    "maximum GPU/CPU channel error bound",
                )?;
            }
        }
    }
    require(
        smooth_p95 <= 5.0,
        "smooth positive-Z angular p95 >5 degrees",
    )?;
    let input = upload(
        &device,
        &queue,
        Extent::new(4, 4).unwrap(),
        &vec![[0.0, 0.0, 1.0]; 16],
    );
    let small = Arc::new(Mutex::new(ResourcePool::new(16)));
    require(
        kernel
            .prepare(&device, &input, None, Bc5Quality::Default, small.clone())
            .is_err(),
        "low budget must reject codec",
    )?;
    require(
        small.lock().unwrap().diagnostics().allocated_bytes == 0,
        "failed admission leaked budget",
    )?;
    // Drop-before-fence retains the complete prepared allocation until completion.
    let mut retirement = kernel
        .prepare(&device, &input, None, Bc5Quality::Default, pool.clone())
        .map_err(|e| e.to_string())?;
    retirement
        .mark_submitted(FenceId(81))
        .map_err(|e| e.to_string())?;
    drop(retirement);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes > 0,
        "pending fence accounting",
    )?;
    pool.lock().unwrap().complete(FenceId(81));
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "retired codec leaked allocation",
    )?;
    Ok(
        serde_json::json!({"smooth_p95_degrees":smooth_p95,"smooth_max_degrees":smooth_max,"steep_max_degrees":steep_max,"npot":true,"tiny_padding":true,"cpu_reference":true,"low_budget":true,"fence_retirement":true}),
    )
}

fn wait<T>(mut next: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(v) = next()? {
            return Ok(v);
        }
        if Instant::now() > deadline {
            return Err("T-081 worker timeout".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
pub fn worker_checks(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
) -> Result<serde_json::Value> {
    let worker =
        GpuWorker::spawn(device, queue, GpuConfig::default()).map_err(|e| e.to_string())?;
    let extent = Extent::new(12, 20).unwrap();
    for negative in [false, true] {
        let values = fixture(extent, false);
        let pixels: Vec<u8> = values
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                n.iter().enumerate().map(move |(c, v)| {
                    (((if negative && i == 0 && c == 2 { -v } else { *v }) * 0.5 + 0.5) * 255.0)
                        .round() as u8
                })
            })
            .collect();
        let pixels = TypedSourcePixels::NormalRgb8 {
            pixels: pixels.into(),
            convention: NormalConvention::OpenGl,
        };
        let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
        let reference = OutputRef {
            node: NodeId(1),
            port: OutputPort::Normal,
        };
        let graph = Graph::try_from(MaterialGraphDto {
            schema_version: MATERIAL_GRAPH_SCHEMA_VERSION,
            graph: vec![MaterialNode {
                id: NodeId(1),
                params: MaterialNodeParams::NormalSource {
                    source: descriptor.clone(),
                    convention: NormalConvention::OpenGl,
                },
                inputs: vec![],
            }],
            outputs: [(MapKind::Normal, reference)].into(),
            selected_output: reference,
        })
        .map_err(|e| e.to_string())?;
        worker
            .install_typed_source(TypedSourceUpload {
                descriptor: descriptor.clone(),
                pixels,
            })
            .map_err(|e| e.to_string())?;
        wait(|| match worker.try_event() {
            Some(WorkerEvent::SourceReady { content_hash })
                if content_hash == descriptor.content_hash =>
            {
                Ok(Some(()))
            }
            Some(e) => Err(format!("source: {e:?}")),
            None => Ok(None),
        })?;
        let snapshot = EvaluationSnapshot {
            generation: 81,
            graph,
        };
        let plan = ExportPlan {
            source_extent: extent,
            size: if negative {
                SizePolicy::Resize {
                    width: 4,
                    height: 4,
                }
            } else {
                SizePolicy::Keep
            },
            naming: "T_{name}_{map}".into(),
            profile: EngineProfile {
                id: "custom".into(),
                revision: 1,
                engine_safe_srgb: false,
                normal_convention: NormalConvention::DirectX,
            },
            outputs: vec![ExportOutput {
                name: "normal_bc5".into(),
                format: ExportFormat::DdsBc5,
                channels: [Component::R, Component::G]
                    .map(|c| ChannelSpec {
                        source: ChannelSource::Map(reference, c),
                        invert: false,
                        transfer: Transfer::Linear,
                    })
                    .to_vec(),
                normal_has_negative_z: false,
                quantization: None,
                import_hints: None,
            }],
            displacement: DisplacementSettings::default(),
            padding: Boundary::Clamp,
        };
        let capture = ExportCapture::new(&snapshot, 81, &plan).map_err(|e| e.to_string())?;
        let job = JobId(if negative { 82 } else { 81 });
        let request = ExportRequest::new(job, capture.clone(), plan).map_err(|e| e.to_string())?;
        wait(|| {
            worker.try_export_progress();
            match worker.request_export(request.clone()) {
                Ok(()) => Ok(Some(())),
                Err(SaveError::Busy) => Ok(None),
                Err(e) => Err(e.to_string()),
            }
        })?;
        let result = wait(|| {
            if let Some(p) = worker.try_export_progress()
                && p.stage == ExportStage::Failed
            {
                return Ok(Some(Err(p.error.unwrap_or_default())));
            }
            if let Some(e) = worker.try_event() {
                return Err(format!("worker event: {e:?}"));
            }
            Ok(worker.try_export_plane().map(Ok))
        })?;
        if negative {
            require(
                result.err().is_some_and(|e| e.contains("negative-Z")),
                "negative-Z source must fail even if descriptor flag is false",
            )?;
        } else {
            let plane = result?;
            require(
                plane.job_id == job
                    && plane.graph_hash == capture.graph_hash
                    && plane.document_revision == 81,
                "immutable BC5 job identity",
            )?;
            let export_shared::ExportPayload::Bc5 { mips } = &plane.payload else {
                return Err("numeric payload for BC5".into());
            };
            require(
                mips == &block_mips(extent).map_err(|e| e.to_string())?,
                "BC5 payload mip layout",
            )?;
            let expected: Vec<_> = values.iter().map(|n| [n[0], -n[1], n[2]]).collect();
            require(
                angular(
                    extent,
                    &expected,
                    &plane.pixels[..mips[0].byte_len as usize],
                )?
                .0 <= 5.0,
                "DX convention converted once",
            )?;
            worker.cancel_export(job).map_err(|e| e.to_string())?;
            drop(plane);
            require(
                worker.begin_export_publication(job).is_err(),
                "cancelled BC5 cannot publish",
            )?;
        }
    }
    require(
        worker.diagnostics().validation_errors == 0,
        "worker GPU validation errors",
    )?;
    worker.join().map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"job_identity":true,"directx":true,"negative_z":true,"cancellation":true}),
    )
}
