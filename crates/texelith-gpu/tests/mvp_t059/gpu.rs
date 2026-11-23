//! Hardware I/O only. Numerical work is always sent to the real material worker.
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
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
            let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
            d.backends = if cfg!(target_os = "windows") {
                wgpu::Backends::DX12
            } else if cfg!(target_os = "macos") {
                wgpu::Backends::METAL
            } else {
                wgpu::Backends::VULKAN
            };
            let instance = wgpu::Instance::new(d);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    ..Default::default()
                }))
                .expect("T-059 requires a native adapter; missing hardware is not a passing skip");
            assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-059 adapter: {:?}", adapter.get_info());
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
        .unwrap();
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&gpu)));
    gpu.wait();
    let error = pollster::block_on(scope.pop());
    drop(gpu);
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
    assert!(error.is_none(), "GPU validation: {error:?}");
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

    pub fn read(&self, texture: &wgpu::Texture) -> Vec<[f64; 4]> {
        let color = texture.format() == wgpu::TextureFormat::Rgba16Float;
        assert!(color || texture.format() == wgpu::TextureFormat::R32Float);
        let bpp = if color { 8 } else { 4 };
        let tight = texture.width() * bpp;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-059 readback"),
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
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let values = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| {
                row[..tight as usize]
                    .chunks_exact(bpp as usize)
                    .map(|pixel| {
                        if color {
                            std::array::from_fn(|c| {
                                super::abi::decode_half(u16::from_le_bytes([
                                    pixel[2 * c],
                                    pixel[2 * c + 1],
                                ]))
                            })
                        } else {
                            [
                                f64::from(f32::from_le_bytes(pixel.try_into().unwrap())),
                                0.0,
                                0.0,
                                1.0,
                            ]
                        }
                    })
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        values
    }
}
