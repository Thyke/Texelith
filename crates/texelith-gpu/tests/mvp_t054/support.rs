//! Test resource plumbing; production computation uses only public node APIs.
use crate::gpu_availability;

use super::reference::{Combine, Controls};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputEdge, InputPort, MapKind, MaskMode, MaterialGraphDto,
    MaterialNode, MaterialNodeParams, NodeId, OutputPort, OutputRef, SourceDescriptor,
};
use texelith_gpu::{
    GpuConfig, GpuWorker, JobId, MaterialRequest, Quality, TypedSourcePixels, TypedSourceUpload,
    WorkerEvent,
};

pub const GIB: u64 = 1024 * 1024 * 1024;

impl Controls {
    pub fn params(self) -> MaterialNodeParams {
        MaterialNodeParams::Metallic {
            picked_color: self.picked,
            radius: self.radius,
            threshold: self.threshold,
            feather: self.feather,
            mask_mode: match self.mode {
                Combine::Replace => MaskMode::Replace,
                Combine::Add => MaskMode::Add,
                Combine::Subtract => MaskMode::Subtract,
            },
        }
    }
}

pub fn request(
    source: &SourceDescriptor,
    id: u64,
    c: Controls,
    quality: Quality,
) -> MaterialRequest {
    let color = OutputRef {
        node: NodeId(1),
        port: OutputPort::Albedo,
    };
    let metallic = OutputRef {
        node: NodeId(2),
        port: OutputPort::Metallic,
    };
    let dto = MaterialGraphDto {
        schema_version: 2,
        graph: vec![
            MaterialNode {
                id: color.node,
                inputs: vec![],
                params: MaterialNodeParams::ColorSource(source.clone()),
            },
            MaterialNode {
                id: metallic.node,
                inputs: vec![InputEdge {
                    port: InputPort::Color,
                    source: color,
                }],
                params: c.params(),
            },
        ],
        outputs: [(MapKind::Metallic, metallic)].into(),
        selected_output: metallic,
    };
    MaterialRequest {
        job_id: JobId(54),
        request_id: id,
        extent: if quality == Quality::Full {
            source.extent
        } else {
            texelith_gpu::proxy_extent(source.extent).unwrap()
        },
        quality,
        outputs: dto.outputs.clone(),
        snapshot: Arc::new(EvaluationSnapshot {
            generation: id,
            graph: Graph::try_from(dto).unwrap(),
        }),
    }
}
pub fn wait<T>(mut next: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(value) = next() {
            return value;
        }
        assert!(Instant::now() < deadline, "T-054 worker timed out");
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
                .expect("T-054 requires native GPU hardware; absence is not a passing skip");
            assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-054 numerical adapter: {:?}", adapter.get_info());
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 4096,
                        ..Default::default()
                    },
                    required_features: wgpu::Features::empty(),
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
    /// Exercise production photo decoding and metallic selection with an
    /// explicit zero/Add mask. Absent ConstantOne masks hide color selection.
    pub fn photo_selection(&self, extent: Extent, bytes: &[u8], c: Controls) -> Vec<f32> {
        let texture = |format| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("T-054 photo selection fixture"),
                size: wgpu::Extent3d {
                    width: extent.width,
                    height: extent.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let raw = texture(wgpu::TextureFormat::Rgba8Unorm);
        self.queue.write_texture(
            raw.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent.width * 4),
                rows_per_image: Some(extent.height),
            },
            raw.size(),
        );
        let color = texture(wgpu::TextureFormat::Rgba16Float);
        let output = texture(wgpu::TextureFormat::R32Float);
        // WebGPU initializes this explicit numeric mask to zero.
        let mask = texture(wgpu::TextureFormat::R32Float);
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-054 selection uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let executor = pollster::block_on(texelith_gpu::GpuExecutor::new(
            self.device.clone(),
            self.queue.clone(),
        ))
        .unwrap();
        let conversion = executor
            .prepare(texelith_gpu::Pass::ColorSource, &raw, &color, extent)
            .unwrap();
        let kernel = texelith_gpu::metallic::MetallicKernel::new(&self.device, self.queue.clone());
        let selected = kernel
            .prepare(
                &self.device,
                &color,
                &output,
                Some(&mask),
                &uniform,
                &Controls {
                    mode: Combine::Add,
                    ..c
                }
                .params(),
            )
            .unwrap();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        conversion.encode(&mut encoder);
        selected.encode(&mut encoder);
        self.queue.submit([encoder.finish()]);
        self.read(&output)
    }

    pub fn worker(&self, budget: u64) -> GpuWorker {
        GpuWorker::spawn(
            self.device.clone(),
            self.queue.clone(),
            GpuConfig {
                budget_bytes: budget,
                max_cache_entries: 8,
            },
        )
        .unwrap()
    }

    pub fn install(
        &self,
        worker: &GpuWorker,
        extent: Extent,
        pixels: TypedSourcePixels,
    ) -> SourceDescriptor {
        let descriptor = pixels.descriptor(extent).unwrap();
        worker
            .install_typed_source(TypedSourceUpload {
                descriptor: descriptor.clone(),
                pixels,
            })
            .unwrap();
        let event = wait(|| worker.try_event());
        assert!(
            matches!(event, WorkerEvent::SourceReady { content_hash }
            if content_hash == descriptor.content_hash),
            "{event:?}"
        );
        descriptor
    }

    pub fn evaluate(&self, worker: &GpuWorker, r: MaterialRequest) -> Vec<f32> {
        let id = r.request_id;
        let extent = r.extent;
        let quality = r.quality;
        worker
            .request_material(r)
            .expect("Metallic must be supported by the production material path");
        let bundle = wait(|| {
            if let Some(event) = worker.try_event() {
                panic!("Metallic worker produced an error: {event:?}");
            }
            worker.try_latest_material()
        });
        assert_eq!(
            (bundle.job_id, bundle.request_id, bundle.generation),
            (JobId(54), id, id)
        );
        assert_eq!(bundle.extent, extent);
        assert_eq!(bundle.quality, quality);
        assert_eq!(bundle.outputs.len(), 1);
        let texture = bundle.outputs[&MapKind::Metallic].texture();
        assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
        self.read(texture)
    }

    pub fn read(&self, texture: &wgpu::Texture) -> Vec<f32> {
        let tight = texture.width() * 4;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-054 scalar readback"),
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
        let values = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| {
                row[..tight as usize]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        values
    }
}
