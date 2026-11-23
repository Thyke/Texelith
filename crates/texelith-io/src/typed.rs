use crate::{IoError, MAX_ALLOC, MAX_AXIS, MAX_INPUT, pixels, source};
use exr::prelude::{FlatSamples, ReadChannels, ReadLayers};
use image::{ImageDecoder, ImageFormat, ImageReader};
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalConvention {
    OpenGl,
    DirectX,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedIntent {
    Height,
    Normal(NormalConvention),
}

#[derive(Debug)]
pub enum TypedImage {
    HeightU16 {
        width: u32,
        height: u32,
        values: Vec<u16>,
    },
    HeightF32 {
        width: u32,
        height: u32,
        values: Vec<f32>,
    },
    NormalRgb8 {
        width: u32,
        height: u32,
        values: Vec<u8>,
        convention: NormalConvention,
    },
}

/// Decodes numeric maps without color conversion. Only grayscale PNG16, single-channel
/// float32 EXR (uncompressed or RLE) and RGB8 PNG normals are accepted.
pub fn decode_typed_bytes(bytes: &[u8], intent: TypedIntent) -> Result<TypedImage, IoError> {
    if bytes.len() as u64 > MAX_INPUT {
        return Err(IoError::LimitExceeded {
            reason: "compressed input",
        });
    }
    match image::guess_format(bytes).map_err(|_| IoError::UnsupportedFormat)? {
        ImageFormat::Png => decode_png(bytes, intent),
        ImageFormat::OpenExr if intent == TypedIntent::Height => decode_exr(bytes),
        _ => Err(IoError::UnsupportedFormat),
    }
}

fn decode_png(bytes: &[u8], intent: TypedIntent) -> Result<TypedImage, IoError> {
    let header = bytes.get(..26).ok_or(IoError::InvalidPixels)?;
    match intent {
        TypedIntent::Height if (header[24], header[25]) != (16, 0) => {
            return Err(IoError::UnsupportedFormat);
        }
        TypedIntent::Normal(_) if (header[24], header[25]) != (8, 2) => {
            return Err(IoError::UnsupportedFormat);
        }
        _ => {}
    }
    let prepared = source::prepare_typed_png(bytes)?;
    let mut reader = ImageReader::with_format(Cursor::new(&prepared.bytes), ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_AXIS);
    limits.max_image_height = Some(MAX_AXIS);
    limits.max_alloc = Some(prepared.decoder_budget);
    reader.limits(limits);
    let decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    if (width, height) != prepared.dimensions {
        return Err(IoError::InvalidPixels);
    }
    let decoded = image::DynamicImage::from_decoder(decoder)?;
    match (intent, decoded) {
        (TypedIntent::Height, image::DynamicImage::ImageLuma16(data)) => {
            Ok(TypedImage::HeightU16 {
                width,
                height,
                values: data.into_raw(),
            })
        }
        (TypedIntent::Normal(convention), image::DynamicImage::ImageRgb8(data)) => {
            Ok(TypedImage::NormalRgb8 {
                width,
                height,
                values: data.into_raw(),
                convention,
            })
        }
        _ => Err(IoError::InvalidPixels),
    }
}

fn decode_exr(bytes: &[u8]) -> Result<TypedImage, IoError> {
    // Bound metadata before the EXR parser can allocate attribute payloads.
    if bytes.len() < 8 || bytes[5] & !2 != 0 || bytes[6..8] != [0, 0] {
        return Err(IoError::UnsupportedFormat);
    }
    let mut offset = 8usize;
    let mut attributes = 0;
    loop {
        if offset >= bytes.len() || offset > 1024 * 1024 || attributes > 256 {
            return Err(IoError::LimitExceeded {
                reason: "exr metadata",
            });
        }
        if bytes[offset] == 0 {
            break;
        }
        attributes += 1;
        let name = exr_text(bytes, &mut offset)?;
        let kind = exr_text(bytes, &mut offset)?;
        let size = bytes
            .get(offset..offset + 4)
            .ok_or(IoError::InvalidPixels)?;
        let len = u32::from_le_bytes(size.try_into().unwrap()) as usize;
        let end = offset
            .checked_add(4)
            .and_then(|n| n.checked_add(len))
            .ok_or(IoError::InvalidPixels)?;
        if end > 1024 * 1024 {
            return Err(IoError::LimitExceeded {
                reason: "exr metadata",
            });
        }
        let value = bytes.get(offset + 4..end).ok_or(IoError::InvalidPixels)?;
        // exr computes chunk counts while parsing the header, before validating
        // it. Bound dimensions first to avoid overflow/division by zero there.
        match (name, kind, value) {
            (b"dataWindow", b"box2i", value) if value.len() == 16 => {
                let mut coordinates = value
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| i64::from(i32::from_le_bytes(*v)));
                let x = coordinates.next().unwrap();
                let y = coordinates.next().unwrap();
                let width = coordinates.next().unwrap() - x + 1;
                let height = coordinates.next().unwrap() - y + 1;
                let width = u32::try_from(width).map_err(|_| IoError::InvalidPixels)?;
                let height = u32::try_from(height).map_err(|_| IoError::InvalidPixels)?;
                pixels(width, height, 1)?;
            }
            (b"tiles", b"tiledesc", value) if value.len() == 9 => {
                let width = u32::from_le_bytes(value[..4].try_into().unwrap());
                let height = u32::from_le_bytes(value[4..8].try_into().unwrap());
                pixels(width, height, 1)?;
            }
            _ => {}
        }
        offset = end;
    }
    let meta = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false)
        .map_err(|_| IoError::InvalidPixels)?;
    if meta.headers.len() != 1 {
        return Err(IoError::UnsupportedFormat);
    }
    let header = &meta.headers[0];
    let width = u32::try_from(header.layer_size.0).map_err(|_| IoError::InvalidPixels)?;
    let height = u32::try_from(header.layer_size.1).map_err(|_| IoError::InvalidPixels)?;
    let count = pixels(width, height, 1)?;
    let channels = &header.channels.list;
    let blocks_ok = match header.blocks {
        exr::meta::BlockDescription::ScanLines => true,
        exr::meta::BlockDescription::Tiles(tile) => {
            tile.level_mode == exr::meta::attribute::LevelMode::Singular
                && tile.tile_size.0 <= MAX_AXIS as usize
                && tile.tile_size.1 <= MAX_AXIS as usize
        }
    };
    if header.deep
        || !blocks_ok
        || channels.len() != 1
        || channels[0].sample_type != exr::meta::attribute::SampleType::F32
        || channels[0].sampling != exr::math::Vec2(1, 1)
    {
        return Err(IoError::UnsupportedFormat);
    }
    let block_scratch = exr_block_scratch(header)?;
    // exr 1.74.3 grows BOTH offset tables geometrically. Six table lengths
    // cover their capacities and old/new allocations during growth. The block
    // iterator also materializes a TileIndices Vec; charge three capacities.
    // The sample reader owns one F32 frame; charge two frames conservatively. Fixed scratch
    // covers metadata/parser storage (32 MiB) and retained thread-local RLE
    // interleave storage from earlier calls (at most 16 MiB, enforced below).
    // Re-audit these bounds when upgrading the decoder.
    let bytes_per_chunk = 6 * 8 + 3 * std::mem::size_of::<exr::meta::TileIndices>() as u64;
    let required = (header.chunk_count as u64)
        .checked_mul(bytes_per_chunk)
        .and_then(|n| n.checked_add(bytes.len() as u64))
        .and_then(|n| n.checked_add(count as u64 * 8))
        .and_then(|n| n.checked_add(48 * 1024 * 1024))
        .and_then(|n| n.checked_add(block_scratch));
    if required.is_none_or(|n| n > MAX_ALLOC) {
        return Err(IoError::LimitExceeded {
            reason: "decoder allocation",
        });
    }
    drop(meta);
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .non_parallel()
        .from_buffered(Cursor::new(bytes))
        .map_err(|_| IoError::InvalidPixels)?;
    let channel = image
        .layer_data
        .channel_data
        .list
        .into_iter()
        .next()
        .ok_or(IoError::InvalidPixels)?;
    match channel.sample_data {
        FlatSamples::F32(values)
            if values.len() == count && values.iter().all(|v| v.is_finite()) =>
        {
            Ok(TypedImage::HeightF32 {
                width,
                height,
                values,
            })
        }
        _ => Err(IoError::InvalidPixels),
    }
}

fn exr_text<'a>(bytes: &'a [u8], offset: &mut usize) -> Result<&'a [u8], IoError> {
    let remaining = bytes.get(*offset..).ok_or(IoError::InvalidPixels)?;
    let len = remaining
        .iter()
        .position(|&b| b == 0)
        .ok_or(IoError::InvalidPixels)?;
    if len > 255 {
        return Err(IoError::InvalidPixels);
    }
    *offset += len + 1;
    Ok(&remaining[..len])
}

fn exr_block_scratch(header: &exr::meta::header::Header) -> Result<u64, IoError> {
    use exr::compression::Compression;

    // Explicit allowlist: DWA contains attacker-controlled section allocation
    // hints; other codecs have not been audited for malformed-stream expansion.
    // Never let a new/unaudited codec reach exr's pixel decoder implicitly.
    if !matches!(
        header.compression,
        Compression::Uncompressed | Compression::RLE
    ) {
        return Err(IoError::UnsupportedFormat);
    }
    let block = header.max_block_pixel_size();
    let bytes = (block.0 as u64)
        .checked_mul(block.1 as u64)
        .and_then(|n| n.checked_mul(4))
        .ok_or(IoError::InvalidPixels)?;
    if header.compression == Compression::Uncompressed {
        // The chunk reader hard-limits its allocation to this header-derived size.
        return Ok(bytes);
    }
    // Malformed RLE can overshoot the expected size and keep decoding. Each
    // two-byte repeat token emits at most 128 bytes, hence <=64*compressed bytes.
    // The compressed chunk itself is hard-limited to `bytes`. Charge Vec growth
    // (including old + new allocations) and interleave scratch at this expanded
    // size, NOT the expected pixel size. Cap the retained TLS buffer as well.
    let expanded = bytes.checked_mul(64).ok_or(IoError::InvalidPixels)?;
    if expanded > 16 * 1024 * 1024 {
        return Err(IoError::LimitExceeded {
            reason: "decoder allocation",
        });
    }
    Ok(bytes + 4 * expanded)
}
