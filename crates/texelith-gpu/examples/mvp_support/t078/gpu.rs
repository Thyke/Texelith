//! Hardware resource I/O only. All tested conversion is production GPU work.
use super::reference::Source;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use texelith_core::{ExportPlan, Extent, NormalConvention};
use texelith_gpu::{
    ExportPackingKernel, ExportPackingSource, GpuExecutor,
    pool::{FenceId, ResourcePool},
};

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub executor: GpuExecutor,
    pub kernel: ExportPackingKernel,
}

// Fixture-only IEEE binary16 conversion. Inputs are finite and bounded by one.
fn half(value: f64) -> u16 {
    assert!(value.is_finite() && value.abs() <= 1.0);
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let value = value.abs();
    if value < 2.0f64.powi(-14) {
        return sign | (value * 2.0f64.powi(24)).round() as u16;
    }
    let exponent = value.log2().floor() as i32;
    let mantissa = ((value / 2.0f64.powi(exponent) - 1.0) * 1024.0).round() as u16;
    sign | (((exponent + 15) as u16) * 1024 + mantissa)
}

impl Gpu {
    pub fn texture(&self, extent: Extent, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T-078 numeric fixture"),
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

    pub fn upload(&self, source: &Source) -> wgpu::Texture {
        let vector = matches!(
            source.reference.port,
            texelith_core::OutputPort::Albedo | texelith_core::OutputPort::Normal
        );
        let texture = self.texture(
            source.extent,
            if vector {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            },
        );
        let bytes: Vec<u8> = if vector {
            source
                .pixels
                .iter()
                .flat_map(|p| p.iter().flat_map(|v| half(*v).to_le_bytes()))
                .collect()
        } else {
            source
                .pixels
                .iter()
                .flat_map(|p| (p[0] as f32).to_le_bytes())
                .collect()
        };
        self.queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(source.extent.width * if vector { 8 } else { 4 }),
                rows_per_image: Some(source.extent.height),
            },
            texture.size(),
        );
        texture
    }

    pub fn bindings<'a>(
        &self,
        sources: &[Source],
        textures: &'a [wgpu::Texture],
    ) -> Vec<ExportPackingSource<'a>> {
        sources
            .iter()
            .zip(textures)
            .map(|(source, texture)| ExportPackingSource {
                reference: source.reference,
                texture,
                boundary: source.boundary,
                normal_convention: source.normal_convention,
            })
            .collect()
    }

    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    pub fn read(&self, texture: &wgpu::Texture) -> Vec<[f64; 4]> {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba32Float);
        let tight = texture.width() * 16;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-078 readback"),
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
        let result = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| {
                row[..tight as usize].as_chunks::<16>().0.iter().map(|p| {
                    std::array::from_fn(|c| {
                        f64::from(f32::from_le_bytes(p[4 * c..4 * c + 4].try_into().unwrap()))
                    })
                })
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        result
    }

    pub fn run(&self, plan: &ExportPlan, index: usize, sources: &[Source]) -> Vec<[f64; 4]> {
        let textures: Vec<_> = sources.iter().map(|s| self.upload(s)).collect();
        let bindings = self.bindings(sources, &textures);
        let output = self.texture(
            super::reference::destination(plan),
            wgpu::TextureFormat::Rgba32Float,
        );
        let pool = Arc::new(Mutex::new(ResourcePool::new(64 * 1024 * 1024)));
        let mut prepared = self
            .kernel
            .prepare(
                &self.executor,
                plan,
                index,
                &bindings,
                &output,
                pool.clone(),
            )
            .unwrap();
        assert!(
            !prepared.passes().is_empty(),
            "zero dispatches is not packing verification"
        );
        let metadata = prepared.metadata();
        assert_eq!(metadata.extent, super::reference::destination(plan));
        assert_eq!(metadata.semantics, plan.outputs[index].semantics());
        assert_eq!(metadata.displacement, plan.displacement);
        let target = metadata.extent;
        if matches!(plan.size, texelith_core::SizePolicy::Pad { .. }) {
            assert_eq!(
                metadata.valid_content,
                [0, 0, plan.source_extent.width, plan.source_extent.height]
            );
            assert_eq!(metadata.padding, plan.padding);
            assert_eq!(
                metadata.uv_scale,
                [
                    plan.source_extent.width as f32 / target.width as f32,
                    plan.source_extent.height as f32 / target.height as f32
                ]
            );
        } else {
            assert_eq!(metadata.valid_content, [0, 0, target.width, target.height]);
            assert_eq!(metadata.uv_scale, [1.0, 1.0]);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for pass in prepared.passes() {
            pass.encode(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
        prepared.mark_submitted(FenceId(1)).unwrap();
        self.wait();
        prepared.complete(FenceId(1));
        let result = self.read(&output);
        drop(prepared);
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        result
    }
}

pub fn vector(port: texelith_core::OutputPort, w: u32, h: u32, pixels: Vec<[f64; 4]>) -> Source {
    assert_eq!(pixels.len(), (w * h) as usize);
    Source {
        reference: super::fixtures::reference(port),
        extent: super::fixtures::extent(w, h),
        pixels,
        boundary: texelith_core::Boundary::Clamp,
        normal_convention: NormalConvention::OpenGl,
    }
}
