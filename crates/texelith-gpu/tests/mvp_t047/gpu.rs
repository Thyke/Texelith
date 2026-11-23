//! Test-side resource I/O only. Every numerical dispatch uses GpuExecutor.
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use texelith_core::Extent;
use texelith_gpu::{GpuExecutor, PreparedPass};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub executor: GpuExecutor,
}

pub fn with_gpu(run: impl FnOnce(&Gpu)) {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    static SHARED: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let gpu = SHARED
        .get_or_init(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(descriptor);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    ..Default::default()
                }))
                .expect("T-047 DX12 checks require an adapter; absence is not a passing skip");
            assert_ne!(
                adapter.get_info().device_type,
                wgpu::DeviceType::Cpu,
                "software adapter is not hardware verification"
            );
            eprintln!("T-047 numerical adapter: {:?}", adapter.get_info());
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
        .unwrap();
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Finish the validation scope even when a red behavioral test panics. A
    // missing node must not poison every subsequent test's diagnostic result.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&gpu)));
    gpu.wait();
    let error = pollster::block_on(scope.pop());
    drop(gpu);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(error.is_none(), "GPU validation error: {error:?}");
}

impl Gpu {
    pub fn texture(&self, extent: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-047 tiny fixture"),
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
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub fn upload(
        &self,
        extent: Extent,
        format: wgpu::TextureFormat,
        bytes: &[u8],
    ) -> wgpu::Texture {
        let texture = self.texture(extent, format);
        let bpp = if format == wgpu::TextureFormat::Rgba16Float {
            8
        } else {
            4
        };
        assert_eq!(bytes.len(), (extent.width * extent.height * bpp) as usize);
        self.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent.width * bpp),
                rows_per_image: Some(extent.height),
            },
            texture.size(),
        );
        texture
    }

    pub fn scalar(&self, image: &super::oracle::Image) -> wgpu::Texture {
        let values: Vec<f32> = image.pixels.iter().map(|v| *v as f32).collect();
        self.upload(
            image.extent,
            wgpu::TextureFormat::R32Float,
            bytemuck::cast_slice(&values),
        )
    }

    pub fn dispatch(&self, prepared: &PreparedPass) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    pub fn read(&self, texture: &wgpu::Texture) -> Vec<u8> {
        let bpp = if texture.format() == wgpu::TextureFormat::Rgba16Float {
            8
        } else {
            4
        };
        let tight = texture.width() * bpp;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-047 padded readback"),
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
        let packed = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| row[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        packed
    }

    pub fn read_scalar(&self, texture: &wgpu::Texture) -> Vec<f32> {
        assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
        self.read(texture)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect()
    }

    pub fn read_rgba(&self, texture: &wgpu::Texture) -> Vec<[f64; 4]> {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
        self.read(texture)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|c| {
                    super::abi::decode_half(u16::from_le_bytes([p[2 * c], p[2 * c + 1]]))
                })
            })
            .collect()
    }
}
