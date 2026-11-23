//! Only device setup and padded readback; no test-side numerical GPU kernel.
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
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(d);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect("T-048 requires hardware DX12; missing hardware is not a passing skip");
            let info = adapter.get_info();
            assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-048 numerical adapter: {info:?}");
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
    let validation = pollster::block_on(scope.pop());
    // Drop the guard before resuming a red test so one failure does not poison others.
    drop(gpu);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(validation.is_none(), "GPU validation error: {validation:?}");
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
    pub fn texture(&self, e: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-048 synthetic fixture"),
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
    }
    pub fn scalar(&self, e: Extent, data: &[f32]) -> wgpu::Texture {
        let t = self.texture(e, wgpu::TextureFormat::R32Float);
        self.queue.write_texture(
            t.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(e.width * 4),
                rows_per_image: Some(e.height),
            },
            t.size(),
        );
        t
    }
    pub fn read(&self, t: &wgpu::Texture) -> Vec<u8> {
        assert_eq!(t.format(), wgpu::TextureFormat::Rgba16Float);
        let tight = t.width() * 8;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-048 readback"),
            size: u64::from(pitch) * u64::from(t.height()),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            t.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(t.height()),
                },
            },
            t.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
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

pub fn decode(bytes: &[u8]) -> Vec<[f64; 4]> {
    bytes
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
