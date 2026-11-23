use crate::gpu_availability;

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use texelith_core::{
    Boundary, Brush, BrushSample, BrushStroke, EvaluationSnapshot, Extent, Graph, MapKind,
    MaskMode, MaterialGraphDto, MaterialNode, MaterialNodeParams, NodeId, OutputPort, OutputRef,
    StrokeId, stroke_digest,
};
use texelith_gpu::{GpuConfig, GpuWorker, JobId, MaterialRequest, Quality};

pub const GIB: u64 = 1 << 30;
pub const NODE: NodeId = NodeId(56);

pub fn dot(id: u64, uv: [f32; 2], radius: f32, opacity: f32, mode: MaskMode) -> BrushStroke {
    BrushStroke {
        id: StrokeId(id),
        brush: Brush {
            radius,
            opacity,
            mode,
            hardness: 1.0,
        },
        points: vec![BrushSample { uv, pressure: 1.0 }],
    }
}

pub fn graph(extent: Extent, strokes: &[BrushStroke]) -> Graph {
    let mask = OutputRef {
        node: NODE,
        port: OutputPort::Mask,
    };
    Graph::try_from(MaterialGraphDto {
        schema_version: 2,
        graph: vec![MaterialNode {
            id: NODE,
            inputs: vec![],
            params: MaterialNodeParams::PaintMask {
                boundary: texelith_core::Boundary::Repeat,
                extent,
                stroke_digest: stroke_digest(strokes).unwrap(),
                strokes: strokes.to_vec(),
            },
        }],
        outputs: [(MapKind::Mask, mask)].into(),
        selected_output: mask,
    })
    .unwrap()
}

pub fn request_graph(graph: Graph, id: u64, quality: Quality) -> MaterialRequest {
    let output = graph.selected_output_ref();
    let source_extent = graph.material_output_extent(output).unwrap();
    MaterialRequest {
        job_id: JobId(56),
        request_id: id,
        extent: if quality == Quality::Full {
            source_extent
        } else {
            texelith_gpu::proxy_extent(source_extent).unwrap()
        },
        quality,
        outputs: [(MapKind::Mask, output)].into(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: id,
            graph,
        }),
    }
}

pub fn request(
    extent: Extent,
    strokes: &[BrushStroke],
    id: u64,
    quality: Quality,
) -> MaterialRequest {
    request_graph(graph(extent, strokes), id, quality)
}

/// Proposed minimal serialization seam for the missing PaintMask boundary API.
/// Round-trip the field so serde cannot silently ignore a Clamp request.
pub fn with_boundary(graph: Graph, boundary: Boundary) -> Graph {
    let mut dto = MaterialGraphDto::from(graph);
    let mut params = serde_json::to_value(&dto.graph[0].params).unwrap();
    params["PaintMask"]["boundary"] = serde_json::to_value(boundary).unwrap();
    let decoded: MaterialNodeParams = serde_json::from_value(params.clone())
        .expect("PaintMask needs an explicit repeat/clamp boundary API");
    let encoded = serde_json::to_value(&decoded).unwrap();
    if boundary == Boundary::Repeat {
        params["PaintMask"]
            .as_object_mut()
            .unwrap()
            .remove("boundary");
        assert_eq!(encoded, params, "Repeat must retain legacy serialization");
        let again: MaterialNodeParams = serde_json::from_value(encoded).unwrap();
        assert!(matches!(
            again,
            MaterialNodeParams::PaintMask {
                boundary: Boundary::Repeat,
                ..
            }
        ));
    } else {
        assert_eq!(encoded, params, "Clamp must retain its explicit selector");
    }
    dto.graph[0].params = decoded;
    Graph::try_from(dto).unwrap()
}

pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-056 worker timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
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
                .expect("T-056 requires a hardware GPU; absence is a failure");
            let info = adapter.get_info();
            assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-056 numerical adapter: {info:?}");
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
        .unwrap_or_else(|error| error.into_inner());
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&gpu)));
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    let error = pollster::block_on(scope.pop());
    drop(gpu);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(error.is_none(), "GPU validation error: {error:?}");
}

impl Gpu {
    pub fn worker(&self, budget_bytes: u64) -> GpuWorker {
        GpuWorker::spawn(
            self.device.clone(),
            self.queue.clone(),
            GpuConfig {
                budget_bytes,
                max_cache_entries: 8,
            },
        )
        .unwrap()
    }

    pub fn evaluate(&self, worker: &GpuWorker, request: MaterialRequest) -> Vec<f32> {
        let (id, extent, quality) = (request.request_id, request.extent, request.quality);
        worker
            .request_material(request)
            .expect("PaintMask must be supported by the public worker");
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("paint worker failed: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(
            (bundle.job_id, bundle.request_id, bundle.generation),
            (JobId(56), id, id)
        );
        assert_eq!((bundle.extent, bundle.quality), (extent, quality));
        assert_eq!(bundle.outputs.len(), 1);
        let texture = bundle.outputs[&MapKind::Mask].texture();
        assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
        assert_eq!(
            (texture.width(), texture.height()),
            (extent.width, extent.height)
        );
        self.read(texture)
    }

    pub fn read(&self, texture: &wgpu::Texture) -> Vec<f32> {
        let tight = texture.width() * 4;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-056 R32 readback"),
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
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let values = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| {
                row[..tight as usize]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        values
    }
}
