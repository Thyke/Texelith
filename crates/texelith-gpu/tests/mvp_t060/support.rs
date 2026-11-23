//! Hardware, public typed upload/material requests, and readback only.
use crate::gpu_availability;

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, EvaluationSnapshot, Extent, Graph, InputEdge, InputPort, MapKind, MaterialGraphDto,
    MaterialNode, MaterialNodeParams, NodeId, OutputPort, OutputRef,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuExecutor, GpuWorker, JobId, MaterialRequest, Quality,
    TypedSourcePixels, TypedSourceUpload, WorkerEvent,
};

pub const BUDGET: u64 = 64 * 1024 * 1024;
pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub executor: GpuExecutor,
}

pub fn with_gpu(run: impl FnOnce(&Gpu)) {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    static SHARED: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let gpu = SHARED
        .get_or_init(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = if cfg!(windows) {
                wgpu::Backends::DX12
            } else {
                wgpu::Backends::PRIMARY
            };
            let instance = wgpu::Instance::new(descriptor);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect(
                    "T-060 requires native GPU hardware; missing hardware is not a passing skip",
                );
            assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-060 numerical adapter: {:?}", adapter.get_info());
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 4096,
                        ..Default::default()
                    },
                    ..Default::default()
                }))
                .unwrap();
            let device = Arc::new(device);
            let queue = Arc::new(queue);
            let executor =
                pollster::block_on(GpuExecutor::new(device.clone(), queue.clone())).unwrap();
            Mutex::new(Gpu {
                device,
                queue,
                executor,
            })
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&gpu)));
    gpu.wait();
    let validation = pollster::block_on(scope.pop());
    drop(gpu);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(
        validation.is_none(),
        "T-060 GPU validation error: {validation:?}"
    );
}

fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-060 material worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

impl Gpu {
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    /// Returns real public material output; Levels proves inspection observes
    /// processed pixels rather than rescanning an upload or content hash.
    pub fn material(
        &self,
        w: usize,
        h: usize,
        pixels: TypedSourcePixels,
        levels: bool,
    ) -> (GpuWorker, CompletedMaterial, MapKind) {
        let extent = Extent::new(w as u32, h as u32).unwrap();
        let color = matches!(&pixels, TypedSourcePixels::Rgba8 { .. });
        assert!(!color || !levels);
        let descriptor = pixels.descriptor(extent).unwrap();
        let worker = GpuWorker::spawn(
            self.device.clone(),
            self.queue.clone(),
            GpuConfig {
                budget_bytes: BUDGET,
                max_cache_entries: 8,
            },
        )
        .unwrap();
        worker
            .install_typed_source(TypedSourceUpload {
                descriptor: descriptor.clone(),
                pixels,
            })
            .unwrap();
        let ready = wait(|| worker.try_event());
        assert!(
            matches!(ready, WorkerEvent::SourceReady { content_hash }
            if content_hash == descriptor.content_hash),
            "{ready:?}"
        );
        let kind = if color {
            MapKind::Albedo
        } else {
            MapKind::Height
        };
        let port = if color {
            OutputPort::Albedo
        } else {
            OutputPort::Height
        };
        let source = OutputRef {
            node: NodeId(1),
            port,
        };
        let output = OutputRef {
            node: NodeId(if levels { 2 } else { 1 }),
            port,
        };
        let mut nodes = vec![MaterialNode {
            id: source.node,
            inputs: vec![],
            params: if color {
                MaterialNodeParams::ColorSource(descriptor)
            } else {
                MaterialNodeParams::HeightSource(descriptor)
            },
        }];
        if levels {
            nodes.push(MaterialNode {
                id: output.node,
                inputs: vec![InputEdge {
                    port: InputPort::Height,
                    source,
                }],
                params: MaterialNodeParams::Levels {
                    black: 0.0,
                    white: 1.0,
                    gamma: 2.0,
                    boundary: Boundary::Clamp,
                },
            });
        }
        let dto = MaterialGraphDto {
            schema_version: 2,
            graph: nodes,
            outputs: [(kind, output)].into(),
            selected_output: output,
        };
        worker
            .request_material(MaterialRequest {
                job_id: JobId(60),
                request_id: 1,
                extent,
                quality: Quality::Full,
                outputs: dto.outputs.clone(),
                snapshot: Arc::new(EvaluationSnapshot {
                    generation: 1,
                    graph: Graph::try_from(dto).unwrap(),
                }),
            })
            .unwrap();
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("T-060 material output: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(
            (bundle.job_id, bundle.request_id, bundle.generation),
            (JobId(60), 1, 1)
        );
        assert_eq!(bundle.extent, extent);
        assert_eq!(bundle.outputs.len(), 1);
        assert_eq!(
            bundle.outputs[&kind].texture().format(),
            if color {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            }
        );
        (worker, bundle, kind)
    }

    pub fn texture(&self, w: usize, h: usize, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-060 diagnostic output"),
            size: wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
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
    }

    pub fn read_bytes(&self, texture: &wgpu::Texture) -> Vec<u8> {
        let bpp = match texture.format() {
            wgpu::TextureFormat::R32Float => 4,
            wgpu::TextureFormat::Rgba16Float => 8,
            format => panic!("unexpected T-060 format {format:?}"),
        };
        let tight = texture.width() * bpp;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-060 padded readback"),
            size: u64::from(pitch) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
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
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.wait();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let bytes = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| row[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        bytes
    }

    pub fn pixels(&self, texture: &wgpu::Texture) -> Vec<[f64; 3]> {
        let bytes = self.read_bytes(texture);
        if texture.format() == wgpu::TextureFormat::R32Float {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| [f64::from(f32::from_le_bytes(*p)); 3])
                .collect()
        } else {
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|p| {
                    std::array::from_fn(|c| half(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]])))
                })
                .collect()
        }
    }
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    sign * match exponent {
        0 => fraction * 2.0_f64.powi(-24),
        31 => {
            if fraction == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => (1.0 + fraction / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15),
    }
}

#[test]
fn readback_half_decoder_covers_negative_subnormal_and_exact_linear_values() {
    assert_eq!(half(0x3800), 0.5);
    assert_eq!(half(0xbc00), -1.0);
    assert_eq!(half(1), 2.0_f64.powi(-24));
}

#[test]
fn typed_fixture_plumbing_reaches_real_height_color_and_processed_output_textures() {
    with_gpu(|gpu| {
        let values: Vec<f32> = (0..33 * 19)
            .map(|i| 0.01 + (i % 33) as f32 / 40.0)
            .collect();
        for levels in [false, true] {
            let (worker, bundle, kind) = gpu.material(
                33,
                19,
                TypedSourcePixels::HeightF32(values.clone().into()),
                levels,
            );
            let pixels = gpu.pixels(bundle.outputs[&kind].texture());
            for (pixel, &input) in pixels.iter().zip(&values) {
                let expected = if levels { input.sqrt() } else { input };
                assert!((pixel[0] - f64::from(expected)).abs() < 2e-5);
            }
            drop(bundle);
            worker.join().unwrap();
        }
        let (worker, bundle, kind) = gpu.material(
            1,
            1,
            TypedSourcePixels::Rgba8 {
                pixels: vec![128, 0, 0, 0].into(),
                interpretation: texelith_core::InputInterpretation::SrgbPhoto,
            },
            false,
        );
        let pixels = gpu.pixels(bundle.outputs[&kind].texture());
        assert!((pixels[0][0] - 0.2158605).abs() < 2e-4);
        assert_eq!([pixels[0][1], pixels[0][2]], [0.0, 0.0]);
        drop(bundle);
        worker.join().unwrap();
    });
}
