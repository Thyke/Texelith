//! Device setup, synthetic upload and readback only; no solver dispatches here.
use super::oracle::Grid;
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
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU
        .get_or_init(|| {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::DX12;
            let instance = wgpu::Instance::new(desc);
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect("T-050 needs native DX12 hardware; missing hardware is not a passing skip");
            let info = adapter.get_info();
            assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-050 numerical adapter: {info:?}");
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
        .unwrap();
    let scope = guard.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&guard)));
    guard.wait();
    let validation = pollster::block_on(scope.pop());
    drop(guard);
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
    pub fn texture(&self, g: Grid, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-050 synthetic fixture"),
            size: wgpu::Extent3d {
                width: g.w as u32,
                height: g.h as u32,
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
    fn upload(&self, t: &wgpu::Texture, data: &[u8], bpp: u32) {
        self.queue.write_texture(
            t.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(t.width() * bpp),
                rows_per_image: Some(t.height()),
            },
            t.size(),
        );
    }
    pub fn scalar(&self, g: Grid, value: f32) -> wgpu::Texture {
        let t = self.texture(g, wgpu::TextureFormat::R32Float);
        self.upload(&t, bytemuck::cast_slice(&vec![value; g.len()]), 4);
        t
    }
    pub fn normal(&self, g: Grid, n: &[[f64; 3]]) -> (wgpu::Texture, Vec<[f64; 3]>) {
        assert_eq!(g.len(), n.len());
        let mut bits = Vec::with_capacity(g.len() * 4);
        let decoded: Vec<[f64; 3]> = n
            .iter()
            .map(|n| {
                let q = n.map(half);
                bits.extend_from_slice(&[q[0], q[1], q[2], half(1.0)]);
                q.map(unhalf)
            })
            .collect();
        assert!(
            decoded.iter().all(|n| {
                let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
                (length - 1.0).abs() <= 2e-3
            }),
            "half-float normal fixtures must remain normalized within the ADR tolerance"
        );
        let t = self.texture(g, wgpu::TextureFormat::Rgba16Float);
        self.upload(&t, bytemuck::cast_slice(&bits), 8);
        (t, decoded)
    }
    pub fn bytes(&self, t: &wgpu::Texture) -> Vec<u8> {
        assert_eq!(t.format(), wgpu::TextureFormat::R32Float);
        self.raw_bytes(t, 4)
    }
    pub fn read_normal(&self, t: &wgpu::Texture) -> Vec<[f64; 3]> {
        assert_eq!(t.format(), wgpu::TextureFormat::Rgba16Float);
        self.raw_bytes(t, 8)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| std::array::from_fn(|i| unhalf(u16::from_le_bytes([p[2 * i], p[2 * i + 1]]))))
            .collect()
    }
    fn raw_bytes(&self, t: &wgpu::Texture, bytes_per_pixel: u32) -> Vec<u8> {
        let tight = t.width() * bytes_per_pixel;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-050 padded readback"),
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
    pub fn read(&self, t: &wgpu::Texture) -> Vec<f64> {
        self.bytes(t)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect()
    }
}
fn half(x: f64) -> u16 {
    assert!(x.is_finite() && x.abs() <= 1.0);
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    let bits = if a < 2f64.powi(-14) {
        (a / 2f64.powi(-24)).round_ties_even() as u16
    } else {
        let e = a.log2().floor() as i32;
        ((e + 15) as u16) * 1024 + ((a / 2f64.powi(e) - 1.0) * 1024.0).round_ties_even() as u16
    };
    sign | bits
}
fn unhalf(bits: u16) -> f64 {
    let s = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (bits >> 10) & 31;
    let m = f64::from(bits & 1023);
    s * if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(i32::from(e) - 15)
    }
}

#[test]
fn fixture_half_encoding_has_independent_ieee_anchors() {
    for (value, bits) in [
        (0.0, 0),
        (1.0, 0x3c00),
        (-1.0, 0xbc00),
        (0.5, 0x3800),
        (2f64.powi(-24), 1),
        (2f64.powi(-25), 0),
        (1.0 / 3.0, 0x3555),
    ] {
        assert_eq!(half(value), bits);
    }
    assert_eq!(unhalf(0x3c00), 1.0);
    assert_eq!(unhalf(0xbc00), -1.0);
    assert_eq!(unhalf(1), 2f64.powi(-24));
}
