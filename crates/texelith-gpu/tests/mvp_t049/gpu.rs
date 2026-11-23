//! Test-only upload/readback. All operator dispatches call the production leaf.
use super::oracle::Image;
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use texelith_core::Extent;
use texelith_gpu::{GpuExecutor, PoissonGrid, PoissonOperators, PoissonPass, pool::ResourcePool};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub operators: PoissonOperators,
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
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(desc);
            let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
            let adapter = adapters
                .iter()
                .find(|a| a.get_info().name.contains("RTX 5070 Ti"))
                .expect(
                    "T-049 requires the requested native RTX 5070 Ti DX12 adapter; no passing skip",
                );
            assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-049 adapter: {:?}", adapter.get_info());
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
            let operators = PoissonOperators::new(
                device.clone(),
                Arc::new(Mutex::new(ResourcePool::new(201_326_592))),
            )
            .unwrap();
            Mutex::new(Gpu {
                device,
                queue,
                operators,
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

pub fn extent(image: &Image) -> Extent {
    Extent::new(image.w as u32, image.h as u32).unwrap()
}

impl Gpu {
    pub fn texture(&self, e: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-049 fixture"),
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

    pub fn upload(&self, texture: &wgpu::Texture, bytes: &[u8], bpp: u32) {
        self.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture.width() * bpp),
                rows_per_image: Some(texture.height()),
            },
            texture.size(),
        );
    }

    pub fn scalar(&self, image: &Image) -> wgpu::Texture {
        let texture = self.texture(extent(image), wgpu::TextureFormat::R32Float);
        let values: Vec<f32> = image.v.iter().map(|v| *v as f32).collect();
        self.upload(&texture, bytemuck::cast_slice(&values), 4);
        texture
    }

    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    pub fn run(
        &self,
        grid: &PoissonGrid,
        pass: PoissonPass,
        inputs: &[&wgpu::Texture],
        target: Extent,
    ) -> wgpu::Texture {
        let output = self.texture(target, wgpu::TextureFormat::R32Float);
        let prepared = self.operators.prepare(grid, pass, inputs, &output).unwrap();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder);
        self.queue.submit([encoder.finish()]);
        // Prepared reservations may be released only after completion; this driver
        // deliberately keeps the handle alive until the actual queue fence.
        self.wait();
        drop(prepared);
        output
    }

    pub fn read_bytes(&self, texture: &wgpu::Texture) -> Vec<u8> {
        let bpp = if texture.format() == wgpu::TextureFormat::Rgba16Float {
            8
        } else {
            4
        };
        let tight = texture.width() * bpp;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-049 readback"),
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
        let out = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| row[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        out
    }

    pub fn read(&self, texture: &wgpu::Texture) -> Image {
        assert_eq!(texture.format(), wgpu::TextureFormat::R32Float);
        Image::new(
            texture.width() as usize,
            texture.height() as usize,
            self.read_bytes(texture)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_le_bytes(*b)))
                .collect(),
        )
    }
}

pub fn compare(actual: &Image, expected: &Image, tolerance: f64) {
    assert_eq!((actual.w, actual.h), (expected.w, expected.h));
    for (i, (a, b)) in actual.v.iter().zip(&expected.v).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "pixel {i}: {a} != {b}, tolerance={tolerance}"
        );
    }
}
