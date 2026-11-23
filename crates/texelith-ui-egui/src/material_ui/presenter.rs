//! Worker-only typed decoding, numeric readback, display conversion and PNG publication.
use super::ImportKind;
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use texelith_core::*;
use texelith_gpu::*;

pub(crate) fn decode(
    path: &Path,
    kind: ImportKind,
    convention: NormalConvention,
) -> Result<TypedSourceUpload, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let (extent, pixels) = match kind {
        ImportKind::Photo => {
            let image = texelith_io::decode_bytes(&bytes, texelith_io::ColorIntent::PhotoSrgb)
                .map_err(|e| e.to_string())?;
            (
                Extent {
                    width: image.width,
                    height: image.height,
                },
                TypedSourcePixels::Rgba8 {
                    pixels: image.rgba8.into(),
                    interpretation: InputInterpretation::SrgbPhoto,
                },
            )
        }
        _ => {
            let intent = if kind == ImportKind::Height {
                texelith_io::TypedIntent::Height
            } else {
                texelith_io::TypedIntent::Normal(if convention == NormalConvention::OpenGl {
                    texelith_io::NormalConvention::OpenGl
                } else {
                    texelith_io::NormalConvention::DirectX
                })
            };
            match texelith_io::decode_typed_bytes(&bytes, intent).map_err(|e| e.to_string())? {
                texelith_io::TypedImage::HeightU16 {
                    width,
                    height,
                    values,
                } => (
                    Extent { width, height },
                    TypedSourcePixels::HeightU16(values.into()),
                ),
                texelith_io::TypedImage::HeightF32 {
                    width,
                    height,
                    values,
                } => (
                    Extent { width, height },
                    TypedSourcePixels::HeightF32(values.into()),
                ),
                texelith_io::TypedImage::NormalRgb8 {
                    width,
                    height,
                    values,
                    ..
                } => (
                    Extent { width, height },
                    TypedSourcePixels::NormalRgb8 {
                        pixels: values.into(),
                        convention,
                    },
                ),
            }
        }
    };
    let descriptor = pixels.descriptor(extent).map_err(|e| e.to_string())?;
    Ok(TypedSourceUpload { descriptor, pixels })
}

pub(crate) fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<[f32; 4]>, String> {
    let bpp = if texture.format() == wgpu::TextureFormat::R32Float {
        4
    } else if texture.format() == wgpu::TextureFormat::Rgba16Float {
        8
    } else {
        return Err("Unsupported material texture format".into());
    };
    let stride = (texture.width() * bpp).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("M1 bounded numeric readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|e| e.to_string())?;
    rx.recv_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let mut pixels = Vec::with_capacity(texture.width() as usize * texture.height() as usize);
    for row in mapped.chunks_exact(stride as usize) {
        for p in row[..(texture.width() * bpp) as usize].chunks_exact(bpp as usize) {
            let value = if bpp == 4 {
                [f32::from_le_bytes(p.try_into().unwrap()), 0.0, 0.0, 1.0]
            } else {
                std::array::from_fn(|i| half(u16::from_le_bytes([p[i * 2], p[i * 2 + 1]])))
            };
            if value.iter().any(|v| !v.is_finite()) {
                return Err("Nonfinite GPU output".into());
            }
            pixels.push(value);
        }
    }
    drop(mapped);
    buffer.unmap();
    Ok(pixels)
}
pub(super) fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    sign * if exponent == 0 {
        f32::from(fraction) * 2.0f32.powi(-24)
    } else if exponent == 31 {
        if fraction == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15)
    }
}
fn srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}
fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
pub(crate) fn rgb(pixel: [f32; 4], map: MapKind, convention: Option<NormalConvention>) -> [u8; 3] {
    match map {
        MapKind::Albedo => [
            byte(srgb(pixel[0])),
            byte(srgb(pixel[1])),
            byte(srgb(pixel[2])),
        ],
        MapKind::Normal => {
            let y = if convention == Some(NormalConvention::DirectX) {
                -pixel[1]
            } else {
                pixel[1]
            };
            [
                byte(pixel[0] * 0.5 + 0.5),
                byte(y * 0.5 + 0.5),
                byte(pixel[2] * 0.5 + 0.5),
            ]
        }
        MapKind::Curvature => [byte(srgb(pixel[0] * 0.5 + 0.5)); 3],
        // egui stores display colors as sRGB bytes. Scalar maps are linear;
        // applying this transfer here leaves their numeric exports untouched.
        _ => [byte(srgb(pixel[0])); 3],
    }
}

/// Scalar previews stretch varying data to the display range. Constants retain
/// their meaning (metallic zero stays black, unoccluded AO stays white).
pub(super) fn scalar_range(pixels: &[[f32; 4]], map: MapKind) -> Option<[f32; 2]> {
    if matches!(map, MapKind::Albedo | MapKind::Normal | MapKind::Curvature) {
        return None;
    }
    let [low, high] = pixels
        .iter()
        .map(|p| p[0])
        .filter(|v| v.is_finite())
        .fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], v| {
            [lo.min(v), hi.max(v)]
        });
    (high > low).then_some([low, high])
}

pub(super) fn display_rgb(
    mut pixel: [f32; 4],
    map: MapKind,
    convention: Option<NormalConvention>,
    range: Option<[f32; 2]>,
) -> [u8; 3] {
    if let Some([low, high]) = range {
        // f64 avoids overflow for signed/HDR imported height ranges.
        pixel[0] =
            ((f64::from(pixel[0]) - f64::from(low)) / (f64::from(high) - f64::from(low))) as f32;
    }
    rgb(pixel, map, convention)
}

pub(crate) struct Presentation {
    pub image: egui::ColorImage,
    pub metrics: Option<SeamMetrics>,
    pub saved: Option<std::path::PathBuf>,
    pub nonconstant: bool,
}

/// Read only an admitted proxy, then convert the samples needed by the card.
/// This avoids the full-size readback and 1024-square display image in `present`.
pub(crate) fn present_thumbnail(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bundle: CompletedMaterial,
    map: MapKind,
    current: &std::sync::atomic::AtomicU64,
) -> Result<Option<egui::ColorImage>, String> {
    let is_current = || current.load(std::sync::atomic::Ordering::Acquire) == bundle.generation;
    if !is_current() {
        return Ok(None);
    }
    let texture = bundle
        .outputs
        .get(&map)
        .ok_or("Missing thumbnail output")?
        .texture();
    if texture.width().max(texture.height()) > 1024 {
        return Err("Thumbnail readback requires a bounded proxy".into());
    }
    let pixels = read_texture(device, queue, texture)?;
    if !is_current() {
        return Ok(None);
    }
    let scale = (128.0 / texture.width().max(texture.height()) as f32).min(1.0);
    let size = [
        (texture.width() as f32 * scale).round().max(1.0) as usize,
        (texture.height() as f32 * scale).round().max(1.0) as usize,
    ];
    let mut result = Vec::with_capacity(size[0] * size[1]);
    let range = scalar_range(&pixels, map);
    for y in 0..size[1] {
        for x in 0..size[0] {
            let sx = ((2 * x + 1) * texture.width() as usize / (2 * size[0]))
                .min(texture.width() as usize - 1);
            let sy = ((2 * y + 1) * texture.height() as usize / (2 * size[1]))
                .min(texture.height() as usize - 1);
            let [r, g, b] = display_rgb(
                pixels[sy * texture.width() as usize + sx],
                map,
                bundle.normal_convention,
                range,
            );
            result.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    Ok(Some(egui::ColorImage::new(size, result)))
}

pub(crate) fn present(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bundle: CompletedMaterial,
    map: MapKind,
    heatmap: bool,
    save: Option<&Path>,
    current: Option<&std::sync::atomic::AtomicU64>,
) -> Result<Presentation, String> {
    let lease = bundle.outputs.get(&map).ok_or("Missing material output")?;
    let texture = lease.texture();
    let extent = bundle.extent;
    let _cpu = crate::batch_jobs::memory::reserve_presentation(
        texture.width(),
        texture.height(),
        heatmap,
        save.is_some(),
    )?;
    let mut diagnostic = None;
    let mut heat = None;
    if heatmap {
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("M1 seam heatmap"),
            size: texture.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let executor = super::block_on(GpuExecutor::new(device.clone(), queue.clone()))
            .map_err(|e| e.to_string())?;
        let pool = Arc::new(Mutex::new(pool::ResourcePool::new(32 * 1024 * 1024)));
        let mut prepared = PreparedSeamHeatmap::new(&executor, texture, &output, pool)
            .map_err(|e| e.to_string())?;
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode_if_current(&mut encoder, || true);
        prepared
            .mark_submitted(pool::FenceId(1))
            .map_err(|e| e.to_string())?;
        queue.submit([encoder.finish()]);
        diagnostic = Some(prepared.read_metrics().map_err(|e| e.to_string())?);
        prepared.complete(pool::FenceId(1));
        heat = Some(output);
    }
    // Numeric save always reads the original output, never the diagnostic display.
    if let Some(path) = save {
        crate::batch_jobs::presentation::save_png(
            &device,
            &queue,
            texture,
            path,
            map,
            bundle.normal_convention,
            || {
                current.is_none_or(|g| {
                    g.load(std::sync::atomic::Ordering::Acquire) == bundle.generation
                })
            },
        )?;
    }
    let (pixels, nonconstant) = crate::batch_jobs::presentation::sample(&device, &queue, texture)?;
    let heat_pixels = heat
        .as_ref()
        .map(|t| crate::batch_jobs::presentation::sample(&device, &queue, t).map(|(p, _)| p))
        .transpose()?;
    let [width, height] =
        crate::batch_jobs::presentation::display_size(extent.width, extent.height);
    let mut bytes = Vec::with_capacity(width as usize * height as usize * 3);
    let range = scalar_range(&pixels, map);
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            bytes.extend_from_slice(&if let Some(h) = &heat_pixels {
                let v = (h[index][0] * 8.0).clamp(0.0, 1.0);
                [byte(v), byte(v * v * 0.5), 0]
            } else {
                display_rgb(pixels[index], map, bundle.normal_convention, range)
            });
        }
    }
    Ok(Presentation {
        image: egui::ColorImage::from_rgb([width as usize, height as usize], &bytes),
        metrics: diagnostic,
        saved: save.map(Path::to_path_buf),
        nonconstant,
    })
}

#[cfg(test)]
pub(crate) fn save_png(
    path: &Path,
    extent: Extent,
    map: MapKind,
    convention: Option<NormalConvention>,
    pixels: &[[f32; 4]],
    current: impl Fn() -> bool,
) -> Result<(), String> {
    let encoded = if map == MapKind::Height {
        texelith_io::encode_height_png(
            extent.width,
            extent.height,
            &pixels
                .iter()
                .map(|p| (p[0].clamp(0.0, 1.0) * 65535.0).round() as u16)
                .collect::<Vec<_>>(),
        )
    } else if matches!(map, MapKind::Albedo | MapKind::Normal) {
        let values: Vec<u8> = pixels
            .iter()
            .flat_map(|p| rgb(*p, map, convention))
            .collect();
        texelith_io::encoders::encode_png8(
            extent.width,
            extent.height,
            texelith_io::encoders::PngChannels::Rgb,
            if map == MapKind::Albedo {
                texelith_io::encoders::PngColorIntent::Srgb
            } else {
                texelith_io::encoders::PngColorIntent::Linear
            },
            &values,
        )
    } else {
        texelith_io::encoders::encode_png8(
            extent.width,
            extent.height,
            texelith_io::encoders::PngChannels::Scalar,
            texelith_io::encoders::PngColorIntent::Linear,
            &pixels.iter().map(|p| byte(p[0])).collect::<Vec<_>>(),
        )
    }
    .map_err(|e| e.to_string())?;
    texelith_io::publish_png_if_current(path, &encoded, current).map_err(|e| e.to_string())
}
