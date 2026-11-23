//! Bounded presentation readback. GPU precision is preserved; only sampled
//! display pixels are expanded to f32. Numeric saves convert one pixel at a time.
use crate::material_ui::presenter::rgb;
use std::{path::Path, time::Duration};
use texelith_core::{MapKind, NormalConvention};

pub(crate) const TILE_ROWS: u32 = 64;

pub(crate) fn display_size(width: u32, height: u32) -> [u32; 2] {
    let longest = width.max(height).max(1024);
    [
        (width * 1024 / longest).max(1),
        (height * 1024 / longest).max(1),
    ]
}

fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mut visit: impl FnMut(u32, u32, [f32; 4]),
) -> Result<(), String> {
    let bpp = match texture.format() {
        wgpu::TextureFormat::R32Float => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        _ => return Err("Unsupported material texture format".into()),
    };
    let stride = (texture.width() * bpp).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T-087 tiled presentation readback"),
        size: u64::from(stride) * u64::from(texture.height().min(TILE_ROWS)),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for first_row in (0..texture.height()).step_by(TILE_ROWS as usize) {
        let rows = (texture.height() - first_row).min(TILE_ROWS);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut source = texture.as_image_copy();
        source.origin.y = first_row;
        encoder.copy_texture_to_buffer(
            source,
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::Extent3d {
                width: texture.width(),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = send.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| e.to_string())?;
        receive
            .recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        for (y, row) in mapped
            .chunks_exact(stride as usize)
            .take(rows as usize)
            .enumerate()
        {
            for (x, p) in row[..(texture.width() * bpp) as usize]
                .chunks_exact(bpp as usize)
                .enumerate()
            {
                let value = if bpp == 4 {
                    [f32::from_le_bytes(p.try_into().unwrap()), 0.0, 0.0, 1.0]
                } else {
                    std::array::from_fn(|i| half(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])))
                };
                if value.iter().any(|v| !v.is_finite()) {
                    return Err("Nonfinite GPU output".into());
                }
                visit(x as u32, first_row + y as u32, value);
            }
        }
        drop(mapped);
        buffer.unmap();
    }
    Ok(())
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    sign * match exponent {
        0 => f32::from(fraction) * 2.0f32.powi(-24),
        31 if fraction == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15),
    }
}

pub(crate) fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<(Vec<[f32; 4]>, bool), String> {
    let [width, height] = display_size(texture.width(), texture.height());
    let mut pixels = vec![[0.0; 4]; width as usize * height as usize];
    let mut first = None;
    let mut nonconstant = false;
    read(device, queue, texture, |x, y, value| {
        nonconstant |= *first.get_or_insert(value) != value;
        // Invert the presenter's floor(dx * source / display) sampling rule.
        let dx = (x * width).div_ceil(texture.width());
        let dy = (y * height).div_ceil(texture.height());
        if dx < width
            && dy < height
            && dx * texture.width() / width == x
            && dy * texture.height() / height == y
        {
            pixels[(dy * width + dx) as usize] = value;
        }
    })?;
    Ok((pixels, nonconstant))
}

pub(crate) fn save_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    path: &Path,
    map: MapKind,
    convention: Option<NormalConvention>,
    current: impl Fn() -> bool,
) -> Result<(), String> {
    use texelith_io::encoders::*;
    let n = texture.width() as usize * texture.height() as usize;
    let encoded = if map == MapKind::Height {
        let mut values = Vec::with_capacity(n);
        read(device, queue, texture, |_, _, p| {
            values.push((p[0].clamp(0.0, 1.0) * 65535.0).round() as u16)
        })?;
        texelith_io::encode_height_png(texture.width(), texture.height(), &values)
    } else {
        let color = matches!(map, MapKind::Albedo | MapKind::Normal);
        let mut values = Vec::with_capacity(n * if color { 3 } else { 1 });
        read(device, queue, texture, |_, _, p| {
            if color {
                values.extend_from_slice(&rgb(p, map, convention));
            } else {
                values.push((p[0].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        })?;
        encode_png8(
            texture.width(),
            texture.height(),
            if color {
                PngChannels::Rgb
            } else {
                PngChannels::Scalar
            },
            if map == MapKind::Albedo {
                PngColorIntent::Srgb
            } else {
                PngColorIntent::Linear
            },
            &values,
        )
    }
    .map_err(|e| e.to_string())?;
    texelith_io::publish_png_if_current(path, &encoded, current).map_err(|e| e.to_string())
}

#[cfg(test)]
#[path = "presentation_tests.rs"]
mod tests;
