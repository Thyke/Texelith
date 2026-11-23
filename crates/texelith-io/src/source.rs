//! Allocation-free validation precedes any decoder construction or compressed-data copy.
use image::ImageFormat;

use crate::{IoError, pixels};

// Bounds for the locked image 0.25.10 / png 0.18.1 / zune-jpeg 0.5.15 paths.
// PNG retains only fixed-size headers/palette/transparency and streams IDAT. Its row,
// inflate and unfilter buffers at 4096 pixels / 8 bpc fit comfortably within 8 MiB.
// JPEG has <=4 components, sampling factors <=4, strides <=4128 i16s, and row
// buffers of <=256 strides/component. 32 MiB covers these (including Vec growth),
// upsampling scratch, fixed Huffman/quantization tables and small parser allocations.
// Full-image JPEG coefficient storage is charged separately below, even for baseline
// JPEGs: non-interleaved scans can require it too. Re-audit on decoder upgrades.
const PNG_SCRATCH: u64 = 8 * 1024 * 1024;
const JPEG_SCRATCH: u64 = 32 * 1024 * 1024;

pub(super) struct Prepared {
    pub bytes: Vec<u8>,
    pub dimensions: (u32, u32),
    pub decoder_budget: u64,
}

struct Header {
    dimensions: (u32, u32),
    coefficients: u64,
}

fn allocation_limit() -> IoError {
    IoError::LimitExceeded {
        reason: "decoder allocation",
    }
}

pub(super) fn prepare(
    bytes: &[u8],
    format: ImageFormat,
    max_alloc: u64,
    input_cost: u64,
) -> Result<Prepared, IoError> {
    // Two passes keep the scanner allocation-free and the sanitized Vec exact-sized.
    let mut retained = 0usize;
    let header = scan(bytes, format, false, |part| retained += part.len())?;
    let rgba = pixels(header.dimensions.0, header.dimensions.1, 4)? as u64;
    let scratch = if format == ImageFormat::Png {
        PNG_SCRATCH
    } else {
        JPEG_SCRATCH
    };
    // During decode, JPEG owns a second copy of the sanitized stream. During RGBA
    // conversion, decoder/coefficient storage has been dropped, leaving native+RGBA.
    // Charge both pixel buffers conservatively even when into_rgba8 reuses one.
    let compressed_copies = if format == ImageFormat::Jpeg { 2 } else { 1 };
    let pixel_peak = (header.coefficients + rgba).max(2 * rgba);
    let required = input_cost
        .checked_add(retained as u64 * compressed_copies)
        .and_then(|n| n.checked_add(pixel_peak))
        .and_then(|n| n.checked_add(scratch))
        .ok_or_else(allocation_limit)?;
    if required > max_alloc {
        return Err(allocation_limit());
    }
    let mut filtered = Vec::with_capacity(retained);
    scan(bytes, format, false, |part| {
        filtered.extend_from_slice(part)
    })?;
    // Account for the allocator's actual reported capacity, not merely the length.
    if required + (filtered.capacity() - retained) as u64 * compressed_copies > max_alloc {
        return Err(allocation_limit());
    }
    Ok(Prepared {
        bytes: filtered,
        dimensions: header.dimensions,
        decoder_budget: scratch,
    })
}

pub(super) fn prepare_typed_png(bytes: &[u8]) -> Result<Prepared, IoError> {
    let mut retained = 0usize;
    let header = scan(bytes, ImageFormat::Png, true, |part| retained += part.len())?;
    let size = pixels(header.dimensions.0, header.dimensions.1, 4)? as u64;
    if bytes.len() as u64 + retained as u64 + 2 * size + PNG_SCRATCH > crate::MAX_ALLOC {
        return Err(allocation_limit());
    }
    let mut filtered = Vec::with_capacity(retained);
    scan(bytes, ImageFormat::Png, true, |part| {
        filtered.extend_from_slice(part)
    })?;
    Ok(Prepared {
        bytes: filtered,
        dimensions: header.dimensions,
        decoder_budget: PNG_SCRATCH,
    })
}

fn scan(
    bytes: &[u8],
    format: ImageFormat,
    typed: bool,
    keep: impl FnMut(&[u8]),
) -> Result<Header, IoError> {
    match format {
        ImageFormat::Png => scan_png(bytes, typed, keep),
        ImageFormat::Jpeg => scan_jpeg(bytes, keep),
        _ => Err(IoError::UnsupportedFormat),
    }
}

fn scan_png(bytes: &[u8], typed: bool, mut keep: impl FnMut(&[u8])) -> Result<Header, IoError> {
    let ihdr = bytes.get(..33).ok_or(IoError::InvalidPixels)?;
    if &ihdr[8..16] != b"\0\0\0\rIHDR" {
        return Err(IoError::InvalidPixels);
    }
    // Check the source depth before palette expansion or any decoder allocation.
    if ihdr[24] != 8 && !(typed && ihdr[24] == 16) {
        return Err(IoError::UnsupportedBitDepth { bits: ihdr[24] });
    }
    let width = u32::from_be_bytes(ihdr[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(ihdr[20..24].try_into().unwrap());
    pixels(width, height, 4)?;
    keep(&ihdr[..8]);
    let mut offset = 8usize;
    let mut palette = false;
    let mut transparency = false;
    while offset < bytes.len() {
        let header = bytes
            .get(offset..offset + 8)
            .ok_or(IoError::InvalidPixels)?;
        let length = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let end = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(length))
            .ok_or(IoError::InvalidPixels)?;
        let chunk = bytes.get(offset..end).ok_or(IoError::InvalidPixels)?;
        let kind = &header[4..8];
        match kind {
            b"IHDR" if offset == 8 => keep(chunk),
            b"PLTE" if !palette && length <= 768 => {
                palette = true;
                keep(chunk);
            }
            b"tRNS" if !transparency && length <= 256 => {
                transparency = true;
                keep(chunk);
            }
            b"IDAT" => keep(chunk),
            b"IEND" if length == 0 => {
                keep(chunk);
                return Ok(Header {
                    dimensions: (width, height),
                    coefficients: 0,
                });
            }
            // Only metadata is discarded; never discard malformed/unknown critical chunks.
            _ if kind == b"tRNS" || kind[0] & 0x20 == 0 => return Err(IoError::InvalidPixels),
            _ => {}
        }
        offset = end;
    }
    Err(IoError::InvalidPixels)
}

fn scan_jpeg(bytes: &[u8], mut keep: impl FnMut(&[u8])) -> Result<Header, IoError> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err(IoError::InvalidPixels);
    }
    keep(&bytes[..2]);
    let mut offset = 2;
    let mut frame = None;
    let mut components = [0; 4];
    let mut component_count = 0;
    let mut in_scan = false;
    let mut seen_scan = false;
    while offset < bytes.len() {
        if in_scan {
            let start = offset;
            // FF00 is an escaped entropy byte; restart markers stay inside the scan.
            // All other markers return to the outer parser, including between scans.
            while offset < bytes.len() {
                if bytes[offset] != 0xff {
                    offset += 1;
                    continue;
                }
                let marker_start = offset;
                while bytes.get(offset) == Some(&0xff) {
                    offset += 1;
                }
                let code = *bytes.get(offset).ok_or(IoError::InvalidPixels)?;
                if code == 0 || (0xd0..=0xd7).contains(&code) {
                    offset += 1;
                } else {
                    offset = marker_start;
                    break;
                }
            }
            keep(&bytes[start..offset]);
            in_scan = false;
        }
        let start = offset;
        if bytes.get(offset) != Some(&0xff) {
            return Err(IoError::InvalidPixels);
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).ok_or(IoError::InvalidPixels)?;
        offset += 1;
        if marker == 0xd9 {
            if !seen_scan {
                return Err(IoError::InvalidPixels);
            }
            keep(&bytes[start..offset]);
            return frame.ok_or(IoError::InvalidPixels);
        }
        let len = bytes
            .get(offset..offset + 2)
            .ok_or(IoError::InvalidPixels)?;
        let len = u16::from_be_bytes(len.try_into().unwrap()) as usize;
        if len < 2 {
            return Err(IoError::InvalidPixels);
        }
        let end = offset.checked_add(len).ok_or(IoError::InvalidPixels)?;
        let payload = bytes.get(offset + 2..end).ok_or(IoError::InvalidPixels)?;
        match marker {
            0xc0..=0xc2 => {
                if frame.is_some() || payload.len() < 6 {
                    return Err(IoError::InvalidPixels);
                }
                if payload[0] != 8 {
                    return Err(IoError::UnsupportedBitDepth { bits: payload[0] });
                }
                let height = u16::from_be_bytes(payload[1..3].try_into().unwrap()) as u32;
                let width = u16::from_be_bytes(payload[3..5].try_into().unwrap()) as u32;
                pixels(width, height, 4)?;
                component_count = payload[5] as usize;
                if !matches!(component_count, 1 | 3 | 4) || payload.len() != 6 + 3 * component_count
                {
                    return Err(IoError::InvalidPixels);
                }
                let mut h_max = 1u64;
                let mut v_max = 1u64;
                let mut blocks = 0u64;
                for (i, component) in payload[6..].as_chunks::<3>().0.iter().enumerate() {
                    let h = u64::from(component[1] >> 4);
                    let v = u64::from(component[1] & 15);
                    if !matches!(h, 1 | 2 | 4)
                        || !matches!(v, 1 | 2 | 4)
                        || component[2] > 3
                        || components[..i].contains(&component[0])
                    {
                        return Err(IoError::InvalidPixels);
                    }
                    components[i] = component[0];
                    h_max = h_max.max(h);
                    v_max = v_max.max(v);
                    blocks += h * v;
                }
                let coefficients = u64::from(width).div_ceil(8 * h_max)
                    * u64::from(height).div_ceil(8 * v_max)
                    * blocks
                    * 64
                    * 2;
                frame = Some(Header {
                    dimensions: (width, height),
                    coefficients,
                });
                keep(&bytes[start..end]);
            }
            0xda => {
                if frame.is_none() || payload.is_empty() {
                    return Err(IoError::InvalidPixels);
                }
                let count = payload[0] as usize;
                if count == 0 || count > component_count || payload.len() != 4 + 2 * count {
                    return Err(IoError::InvalidPixels);
                }
                let mut selected = [0; 4];
                for (i, component) in payload[1..1 + 2 * count]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                {
                    if !components[..component_count].contains(&component[0])
                        || selected[..i].contains(&component[0])
                        || component[1] >> 4 > 3
                        || component[1] & 15 > 3
                    {
                        return Err(IoError::InvalidPixels);
                    }
                    selected[i] = component[0];
                }
                keep(&bytes[start..end]);
                in_scan = true;
                seen_scan = true;
            }
            // Adobe's fixed-size transform selector affects CMYK/YCCK/RGB decoding;
            // it is pixel coding information, unlike ICC, EXIF, XMP and comments.
            0xee if payload.len() == 12 && payload.starts_with(b"Adobe") && payload[11] <= 2 => {
                keep(&bytes[start..end]);
            }
            0xe0..=0xef | 0xfe => {}
            0xc4 | 0xdb | 0xdd => keep(&bytes[start..end]),
            _ => return Err(IoError::UnsupportedFormat),
        }
        offset = end;
    }
    Err(IoError::InvalidPixels)
}
