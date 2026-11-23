use exr::prelude::{FlatSamples, ReadChannels, ReadLayers};
use std::io::Cursor;

#[derive(Debug)]
pub struct EnvironmentImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<[f32; 3]>,
}

#[derive(Debug)]
pub enum EnvironmentError {
    Invalid,
    Unsupported,
    LimitExceeded,
}
impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EnvironmentError {}
type Result<T> = std::result::Result<T, EnvironmentError>;
const INPUT_CAP: u64 = 128 * 1024 * 1024;
const DEFAULT_BUDGET: u64 = 256 * 1024 * 1024;
fn dimensions(w: u32, h: u32) -> Result<usize> {
    if w == 0 || h == 0 || w > 2048 || h > 1024 {
        return Err(EnvironmentError::LimitExceeded);
    }
    Ok(w as usize * h as usize)
}
fn budget(input: usize, count: usize, extra: u64, limit: u64) -> Result<()> {
    let required = (input as u64)
        .checked_add(
            (count as u64)
                .checked_mul(24)
                .ok_or(EnvironmentError::LimitExceeded)?,
        )
        .and_then(|v| v.checked_add(extra))
        .ok_or(EnvironmentError::LimitExceeded)?;
    if required > limit {
        Err(EnvironmentError::LimitExceeded)
    } else {
        Ok(())
    }
}
/// Decode linear RGB in top-to-bottom row-major order, with a 256 MiB working budget.
pub fn decode_environment_bytes(bytes: &[u8]) -> Result<EnvironmentImage> {
    decode_environment_bytes_with_limit(bytes, DEFAULT_BUDGET)
}
/// The budget includes the input, decoded pixels and conservative decoder scratch.
pub fn decode_environment_bytes_with_limit(bytes: &[u8], limit: u64) -> Result<EnvironmentImage> {
    if bytes.len() as u64 > INPUT_CAP {
        return Err(EnvironmentError::LimitExceeded);
    }
    let mut pos = 0;
    if matches!(line(bytes, &mut pos), Ok("#?RADIANCE" | "#?RGBE")) {
        hdr(bytes, limit)
    } else if bytes.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        exr(bytes, limit)
    } else {
        Err(EnvironmentError::Unsupported)
    }
}
fn line<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a str> {
    let rest = bytes.get(*pos..).ok_or(EnvironmentError::Invalid)?;
    let n = rest
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(EnvironmentError::Invalid)?;
    *pos = pos.checked_add(n + 1).ok_or(EnvironmentError::Invalid)?;
    if *pos > 1024 * 1024 {
        return Err(EnvironmentError::LimitExceeded);
    }
    std::str::from_utf8(&rest[..n])
        .map(|s| s.trim_end_matches('\r'))
        .map_err(|_| EnvironmentError::Invalid)
}
fn hdr(bytes: &[u8], limit: u64) -> Result<EnvironmentImage> {
    let mut pos = 0;
    line(bytes, &mut pos)?;
    let mut format = false;
    loop {
        let s = line(bytes, &mut pos)?;
        if s.is_empty() {
            break;
        }
        if s == "FORMAT=32-bit_rle_rgbe" {
            format = true;
        }
    }
    if !format {
        return Err(EnvironmentError::Unsupported);
    }
    let mut parts = line(bytes, &mut pos)?.split_whitespace();
    let y_axis = parts.next().ok_or(EnvironmentError::Invalid)?;
    let height = parts.next().ok_or(EnvironmentError::Invalid)?;
    let x_axis = parts.next().ok_or(EnvironmentError::Invalid)?;
    let width = parts.next().ok_or(EnvironmentError::Invalid)?;
    if parts.next().is_some() || !matches!(y_axis, "-Y" | "+Y") || !matches!(x_axis, "+X" | "-X") {
        return Err(EnvironmentError::Unsupported);
    }
    let h: u32 = height.parse().map_err(|_| EnvironmentError::Invalid)?;
    let w: u32 = width.parse().map_err(|_| EnvironmentError::Invalid)?;
    let count = dimensions(w, h)?;
    budget(bytes.len(), count, (w as u64) * 4 + 4096, limit)?;
    let mut rgb = vec![[0.0; 3]; count];
    let mut scan = vec![[0u8; 4]; w as usize];
    for row in 0..h as usize {
        let marker = bytes.get(pos..pos + 4).ok_or(EnvironmentError::Invalid)?;
        if (8..=32767).contains(&w) && marker[0..2] == [2, 2] && marker[2] & 128 == 0 {
            if usize::from(u16::from_be_bytes([marker[2], marker[3]])) != w as usize {
                return Err(EnvironmentError::Invalid);
            }
            pos += 4;
            for c in 0..4 {
                let mut x = 0;
                while x < w as usize {
                    let code = *bytes.get(pos).ok_or(EnvironmentError::Invalid)?;
                    pos += 1;
                    if code == 0 {
                        return Err(EnvironmentError::Invalid);
                    }
                    if code > 128 {
                        let n = (code - 128) as usize;
                        let value = *bytes.get(pos).ok_or(EnvironmentError::Invalid)?;
                        pos += 1;
                        if x + n > w as usize {
                            return Err(EnvironmentError::Invalid);
                        }
                        for pixel in &mut scan[x..x + n] {
                            pixel[c] = value;
                        }
                        x += n;
                    } else {
                        let n = code as usize;
                        let src = bytes.get(pos..pos + n).ok_or(EnvironmentError::Invalid)?;
                        if x + n > w as usize {
                            return Err(EnvironmentError::Invalid);
                        }
                        for (pixel, &v) in scan[x..x + n].iter_mut().zip(src) {
                            pixel[c] = v;
                        }
                        pos += n;
                        x += n;
                    }
                }
            }
        } else {
            // Legacy RGBE repeat markers encode runs of the preceding pixel.
            let mut x = 0;
            let mut shift = 0u32;
            while x < w as usize {
                let p: [u8; 4] = bytes
                    .get(pos..pos + 4)
                    .ok_or(EnvironmentError::Invalid)?
                    .try_into()
                    .unwrap();
                pos += 4;
                if p[..3] == [1, 1, 1] {
                    if x == 0 || shift >= usize::BITS {
                        return Err(EnvironmentError::Invalid);
                    }
                    let n = (p[3] as usize)
                        .checked_shl(shift)
                        .ok_or(EnvironmentError::Invalid)?;
                    if n > w as usize - x {
                        return Err(EnvironmentError::Invalid);
                    }
                    let previous = scan[x - 1];
                    scan[x..x + n].fill(previous);
                    x += n;
                    shift += 8;
                } else {
                    scan[x] = p;
                    x += 1;
                    shift = 0;
                }
            }
        }
        let y = if y_axis == "-Y" {
            row
        } else {
            h as usize - 1 - row
        };
        for (col, p) in scan.iter().enumerate() {
            let x = if x_axis == "+X" {
                col
            } else {
                w as usize - 1 - col
            };
            let factor = if p[3] == 0 {
                0.0
            } else {
                2f32.powi(i32::from(p[3]) - 136)
            };
            let v = [
                p[0] as f32 * factor,
                p[1] as f32 * factor,
                p[2] as f32 * factor,
            ];
            if !v.iter().all(|n| n.is_finite()) {
                return Err(EnvironmentError::Invalid);
            }
            rgb[y * w as usize + x] = v;
        }
    }
    Ok(EnvironmentImage {
        width: w,
        height: h,
        rgb,
    })
}
fn text<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let rest = bytes.get(*pos..).ok_or(EnvironmentError::Invalid)?;
    let n = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or(EnvironmentError::Invalid)?;
    if n > 255 {
        return Err(EnvironmentError::Invalid);
    }
    *pos += n + 1;
    Ok(&rest[..n])
}
fn exr(bytes: &[u8], limit: u64) -> Result<EnvironmentImage> {
    if bytes.len() < 8 || bytes[5] & !2 != 0 || bytes[6..8] != [0, 0] {
        return Err(EnvironmentError::Unsupported);
    }
    let mut pos = 8;
    let mut attributes = 0;
    loop {
        if pos >= bytes.len() || pos > 1024 * 1024 || attributes > 256 {
            return Err(EnvironmentError::LimitExceeded);
        }
        if bytes[pos] == 0 {
            break;
        }
        attributes += 1;
        let name = text(bytes, &mut pos)?;
        let kind = text(bytes, &mut pos)?;
        let len = u32::from_le_bytes(
            bytes
                .get(pos..pos + 4)
                .ok_or(EnvironmentError::Invalid)?
                .try_into()
                .unwrap(),
        ) as usize;
        let end = pos
            .checked_add(4)
            .and_then(|n| n.checked_add(len))
            .ok_or(EnvironmentError::Invalid)?;
        if end > 1024 * 1024 {
            return Err(EnvironmentError::LimitExceeded);
        }
        let value = bytes.get(pos + 4..end).ok_or(EnvironmentError::Invalid)?;
        if name == b"dataWindow" && kind == b"box2i" && value.len() == 16 {
            let coord: Vec<i64> = value
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i64::from(i32::from_le_bytes(*c)))
                .collect();
            let w =
                u32::try_from(coord[2] - coord[0] + 1).map_err(|_| EnvironmentError::Invalid)?;
            let h =
                u32::try_from(coord[3] - coord[1] + 1).map_err(|_| EnvironmentError::Invalid)?;
            dimensions(w, h)?;
        }
        if name == b"tiles" && kind == b"tiledesc" && value.len() == 9 {
            let w = u32::from_le_bytes(value[..4].try_into().unwrap());
            let h = u32::from_le_bytes(value[4..8].try_into().unwrap());
            dimensions(w, h)?;
        }
        pos = end;
    }
    budget(bytes.len(), 0, 48 * 1024 * 1024, limit)?;
    let meta = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false)
        .map_err(|_| EnvironmentError::Invalid)?;
    if meta.headers.len() != 1 {
        return Err(EnvironmentError::Unsupported);
    }
    let header = &meta.headers[0];
    let w = u32::try_from(header.layer_size.0).map_err(|_| EnvironmentError::Invalid)?;
    let h = u32::try_from(header.layer_size.1).map_err(|_| EnvironmentError::Invalid)?;
    let count = dimensions(w, h)?;
    let blocks_ok = match header.blocks {
        exr::meta::BlockDescription::ScanLines => true,
        exr::meta::BlockDescription::Tiles(tile) => {
            tile.level_mode == exr::meta::attribute::LevelMode::Singular
                && tile.tile_size.0 <= 2048
                && tile.tile_size.1 <= 1024
        }
    };
    let channels = &header.channels.list;
    if header.deep
        || !blocks_ok
        || channels.len() != 3
        || !["R", "G", "B"]
            .iter()
            .all(|name| channels.iter().any(|c| c.name.to_string() == *name))
        || channels.iter().any(|c| {
            !matches!(
                c.sample_type,
                exr::meta::attribute::SampleType::F16 | exr::meta::attribute::SampleType::F32
            ) || c.sampling != exr::math::Vec2(1, 1)
        })
    {
        return Err(EnvironmentError::Unsupported);
    }
    use exr::compression::Compression;
    if !matches!(
        header.compression,
        Compression::Uncompressed | Compression::RLE
    ) {
        return Err(EnvironmentError::Unsupported);
    }
    let block = header.max_block_pixel_size();
    let block_bytes = (block.0 as u64)
        .checked_mul(block.1 as u64)
        .and_then(|v| v.checked_mul(12))
        .ok_or(EnvironmentError::LimitExceeded)?;
    let scratch = if header.compression == Compression::RLE {
        let expanded = block_bytes
            .checked_mul(64)
            .ok_or(EnvironmentError::LimitExceeded)?;
        if expanded > 16 * 1024 * 1024 {
            return Err(EnvironmentError::LimitExceeded);
        }
        block_bytes + 4 * expanded
    } else {
        block_bytes
    };
    let tables = (header.chunk_count as u64)
        .checked_mul(6 * 8 + 3 * std::mem::size_of::<exr::meta::TileIndices>() as u64)
        .ok_or(EnvironmentError::LimitExceeded)?;
    budget(
        bytes.len(),
        count,
        48 * 1024 * 1024 + scratch + tables,
        limit,
    )?;
    drop(meta);
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .non_parallel()
        .from_buffered(Cursor::new(bytes))
        .map_err(|_| EnvironmentError::Invalid)?;
    let mut rgb = vec![[0.0; 3]; count];
    for channel in image.layer_data.channel_data.list {
        let index = match channel.name.to_string().as_str() {
            "R" => 0,
            "G" => 1,
            "B" => 2,
            _ => return Err(EnvironmentError::Invalid),
        };
        match channel.sample_data {
            FlatSamples::F32(values) => {
                if values.len() != count {
                    return Err(EnvironmentError::Invalid);
                }
                for (pixel, v) in rgb.iter_mut().zip(values) {
                    if !v.is_finite() {
                        return Err(EnvironmentError::Invalid);
                    }
                    pixel[index] = v;
                }
            }
            FlatSamples::F16(values) => {
                if values.len() != count {
                    return Err(EnvironmentError::Invalid);
                }
                for (pixel, v) in rgb.iter_mut().zip(values) {
                    let v = v.to_f32();
                    if !v.is_finite() {
                        return Err(EnvironmentError::Invalid);
                    }
                    pixel[index] = v;
                }
            }
            _ => return Err(EnvironmentError::Unsupported),
        }
    }
    Ok(EnvironmentImage {
        width: w,
        height: h,
        rgb,
    })
}
