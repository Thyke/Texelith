//! Public graph requests, device plumbing and padded readback only.
use crate::gpu_availability;

use super::{abi::decode_half, oracle::Controls};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputEdge, InputPort, MapKind, MaterialGraphDto,
    MaterialNode, MaterialNodeParams, NodeId, NormalConvention, OutputPort, OutputRef,
    SourceDescriptor,
};
use texelith_gpu::{
    CompletedMaterial, GpuConfig, GpuWorker, JobId, MaterialRequest, Quality, TypedSourcePixels,
    TypedSourceUpload, WorkerEvent,
};

pub const GIB: u64 = 1024 * 1024 * 1024;
pub const PORTS: [OutputPort; 3] = [OutputPort::Curvature, OutputPort::Edge, OutputPort::Cavity];
pub const WITH_NORMAL: [OutputPort; 4] = [
    OutputPort::Normal,
    OutputPort::Curvature,
    OutputPort::Edge,
    OutputPort::Cavity,
];

pub fn output(port: OutputPort) -> OutputRef {
    OutputRef {
        node: NodeId(if port == OutputPort::Normal { 1 } else { 2 }),
        port,
    }
}

pub fn request(
    source: &SourceDescriptor,
    id: u64,
    c: Controls,
    convention: NormalConvention,
    ports: &[OutputPort],
) -> MaterialRequest {
    assert!(!ports.is_empty());
    let outputs = ports.iter().map(|&p| (p.map_kind(), output(p))).collect();
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: NodeId(1),
                inputs: vec![],
                params: MaterialNodeParams::NormalSource {
                    source: source.clone(),
                    convention,
                },
            },
            MaterialNode {
                id: NodeId(2),
                inputs: vec![InputEdge {
                    port: InputPort::Normal,
                    source: output(OutputPort::Normal),
                }],
                params: c.params(),
            },
        ],
        outputs,
        selected_output: output(ports[0]),
    };
    MaterialRequest {
        job_id: JobId(52),
        request_id: id,
        outputs: dto.outputs.clone(),
        extent: source.extent,
        quality: Quality::Full,
        snapshot: Arc::new(EvaluationSnapshot {
            generation: id,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}

pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(v) = next() {
            return v;
        }
        assert!(Instant::now() < deadline, "T-052 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}
pub struct Planes {
    pub scalar: BTreeMap<MapKind, Vec<f32>>,
    pub normal: Option<Vec<[f64; 3]>>,
}

pub fn with_gpu(run: impl FnOnce(&Gpu)) {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    static SHARED: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let gpu = SHARED
        .get_or_init(|| {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = if cfg!(windows) {
                wgpu::Backends::DX12
            } else {
                wgpu::Backends::PRIMARY
            };
            let instance = wgpu::Instance::new(desc);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect("T-052 requires native GPU hardware; absence is not a passing skip");
            let info = adapter.get_info();
            assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-052 numerical adapter: {info:?}");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 4096,
                        ..Default::default()
                    },
                    ..Default::default()
                }))
                .unwrap();
            Mutex::new(Gpu {
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&gpu)));
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    let validation = pollster::block_on(scope.pop());
    drop(gpu);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(validation.is_none(), "GPU validation error: {validation:?}");
}

impl Gpu {
    pub fn worker(&self, budget: u64) -> GpuWorker {
        GpuWorker::spawn(
            self.device.clone(),
            self.queue.clone(),
            GpuConfig {
                budget_bytes: budget,
                max_cache_entries: 16,
            },
        )
        .unwrap()
    }

    pub fn install(
        &self,
        worker: &GpuWorker,
        e: Extent,
        rgb: &[u8],
        convention: NormalConvention,
    ) -> SourceDescriptor {
        let pixels = TypedSourcePixels::NormalRgb8 {
            pixels: rgb.to_vec().into(),
            convention,
        };
        let descriptor = pixels.descriptor(e).unwrap();
        worker
            .install_typed_source(TypedSourceUpload {
                descriptor: descriptor.clone(),
                pixels,
            })
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::SourceReady { content_hash } if content_hash == descriptor.content_hash),
            "{event:?}"
        );
        descriptor
    }

    pub fn complete(&self, worker: &GpuWorker, r: MaterialRequest) -> CompletedMaterial {
        let expected = (
            r.job_id,
            r.request_id,
            r.snapshot.generation,
            r.extent,
            r.quality,
        );
        let kinds: Vec<_> = r.outputs.keys().copied().collect();
        let convention = r.outputs.get(&MapKind::Normal).map(|o| {
            r.snapshot
                .graph
                .output_normal_convention(*o)
                .unwrap()
                .unwrap()
        });
        worker
            .request_material(r)
            .expect("Curvature must be supported by the public material path");
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("curvature worker error: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(
            (
                bundle.job_id,
                bundle.request_id,
                bundle.generation,
                bundle.extent,
                bundle.quality
            ),
            expected
        );
        assert_eq!(bundle.outputs.keys().copied().collect::<Vec<_>>(), kinds);
        assert_eq!(bundle.normal_convention, convention);
        bundle
    }

    pub fn evaluate(&self, worker: &GpuWorker, r: MaterialRequest) -> Planes {
        let bundle = self.complete(worker, r);
        let mut scalar = BTreeMap::new();
        let mut normal = None;
        for (&kind, lease) in &bundle.outputs {
            let texture = lease.texture();
            assert_eq!(
                (texture.width(), texture.height()),
                (bundle.extent.width, bundle.extent.height)
            );
            let bytes = self.read(texture);
            if kind == MapKind::Normal {
                assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
                normal = Some(
                    bytes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|p| {
                            let n = std::array::from_fn(|c| {
                                decode_half(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]]))
                            });
                            assert!(
                                (n.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() <= 2e-3
                            );
                            assert_eq!(decode_half(u16::from_le_bytes([p[6], p[7]])), 1.0);
                            n
                        })
                        .collect(),
                );
            } else {
                assert_eq!(
                    texture.format(),
                    wgpu::TextureFormat::R32Float,
                    "{kind:?} must stay linear R32Float"
                );
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| f32::from_le_bytes(*p))
                    .collect();
                assert!(values.iter().all(|v| v.is_finite()));
                if matches!(kind, MapKind::Edge | MapKind::Cavity) {
                    assert!(values.iter().all(|v| (0.0..=1.0).contains(v)));
                }
                scalar.insert(kind, values);
            }
        }
        Planes { scalar, normal }
    }

    fn read(&self, texture: &wgpu::Texture) -> Vec<u8> {
        let bytes_per_pixel = if texture.format() == wgpu::TextureFormat::Rgba16Float {
            8
        } else {
            4
        };
        let tight = texture.width() * bytes_per_pixel;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-052 test readback"),
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
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let bytes = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|r| r[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        bytes
    }
}
