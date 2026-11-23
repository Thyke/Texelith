//! Resource I/O only. Numerical work runs through the production worker/passes.
use crate::gpu_availability;

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use texelith_core::Extent;
use texelith_gpu::GpuExecutor;

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub executor: GpuExecutor,
}

pub fn with_gpu(run: impl FnOnce(&Gpu)) {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let gpu = GPU
        .get_or_init(|| {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(desc);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    ..Default::default()
                }))
                .expect("T-053 requires DX12 hardware; unavailable hardware is not PASS");
            assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-053 numerical adapter: {:?}", adapter.get_info());
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
        .unwrap();
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
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
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    pub fn texture(&self, extent: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-053 numeric fixture"),
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

    pub fn read_scalar(&self, texture: &wgpu::Texture) -> Vec<f32> {
        assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
        let tight = texture.width() * 4;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-053 padded readback"),
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
