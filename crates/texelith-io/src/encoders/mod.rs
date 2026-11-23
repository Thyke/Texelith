use crate::{IoError, pixels};
use exr::prelude::{
    Encoding, Image, Layer, LayerAttributes, SpecificChannels, Vec2, WritableImage,
};
use image::ImageEncoder;
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngChannels {
    Scalar,
    Rgb,
    Rgba,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngColorIntent {
    Linear,
    Srgb,
}

impl PngChannels {
    fn count(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
    fn color(self, sixteen: bool) -> image::ExtendedColorType {
        use image::ExtendedColorType::*;
        match (self, sixteen) {
            (Self::Scalar, false) => L8,
            (Self::Rgb, false) => Rgb8,
            (Self::Rgba, false) => Rgba8,
            (Self::Scalar, true) => L16,
            (Self::Rgb, true) => Rgb16,
            (Self::Rgba, true) => Rgba16,
        }
    }
}

fn png(
    width: u32,
    height: u32,
    channels: PngChannels,
    intent: PngColorIntent,
    bytes: &[u8],
    sixteen: bool,
) -> Result<Vec<u8>, IoError> {
    if channels == PngChannels::Rgba && !sixteen {
        // Bound the destination independently of Vec growth. Level 1 uses one
        // zlib buffer, avoiding fdeflate's simultaneous fallback buffers.
        let filtered = bytes.len() + height as usize;
        let bound = filtered + filtered.div_ceil(1000) + 65536;
        let mut out = vec![0; bound];
        let mut writer = Cursor::new(out.as_mut_slice());
        image::codecs::png::PngEncoder::new_with_quality(
            &mut writer,
            image::codecs::png::CompressionType::Level(1),
            image::codecs::png::FilterType::Sub,
        )
        .write_image(bytes, width, height, channels.color(false))?;
        let len = writer.position() as usize;
        out.truncate(len);
        if intent == PngColorIntent::Srgb {
            out.splice(
                33..33,
                [
                    0, 0, 0, 1, b's', b'R', b'G', b'B', 0, 0xae, 0xce, 0x1c, 0xe9,
                ],
            );
        }
        return Ok(out);
    }
    let mut out = Vec::new();
    // A fixed lossless predictor avoids scoring five filters on every scanline.
    image::codecs::png::PngEncoder::new_with_quality(
        &mut out,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Sub,
    )
    .write_image(bytes, width, height, channels.color(sixteen))?;
    if intent == PngColorIntent::Srgb {
        // PNG sRGB rendering intent 0 (perceptual); insert immediately after IHDR.
        let chunk = [
            0, 0, 0, 1, b's', b'R', b'G', b'B', 0, 0xae, 0xce, 0x1c, 0xe9,
        ];
        out.splice(33..33, chunk);
    }
    Ok(out)
}

/// Encodes interleaved integer samples without applying a transfer function.
pub fn encode_png8(
    width: u32,
    height: u32,
    channels: PngChannels,
    intent: PngColorIntent,
    values: &[u8],
) -> Result<Vec<u8>, IoError> {
    if pixels(width, height, channels.count())? != values.len() {
        return Err(IoError::InvalidPixels);
    }
    png(width, height, channels, intent, values, false)
}
/// Encodes native u16 samples; PNG serialization uses network byte order.
pub fn encode_png16(
    width: u32,
    height: u32,
    channels: PngChannels,
    intent: PngColorIntent,
    values: &[u16],
) -> Result<Vec<u8>, IoError> {
    if pixels(width, height, channels.count())? != values.len() {
        return Err(IoError::InvalidPixels);
    }
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    png(width, height, channels, intent, &bytes, true)
}

/// Encodes finite linear displacement as one lossless float32 Y channel.
pub fn encode_exr32(width: u32, height: u32, values: &[f32]) -> Result<Vec<u8>, IoError> {
    if pixels(width, height, 1)? != values.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(IoError::InvalidPixels);
    }
    let channels = SpecificChannels::build()
        .with_channel("Y")
        .with_pixel_fn(|p: Vec2<usize>| (values[p.y() * width as usize + p.x()],));
    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::default(),
        Encoding::FAST_LOSSLESS,
        channels,
    );
    let image = Image::from_layer(layer);
    let mut out = Cursor::new(Vec::new());
    image
        .write()
        .non_parallel()
        .to_buffered(&mut out)
        .map_err(|e| IoError::Io(std::io::Error::other(e.to_string())))?;
    Ok(out.into_inner())
}

/// Wraps preencoded BC5 blocks in a DX10 DDS texture with a contiguous mip chain.
pub fn encode_dds_bc5(width: u32, height: u32, mips: &[&[u8]]) -> Result<Vec<u8>, IoError> {
    pixels(width, height, 1)?;
    if mips.is_empty() {
        return Err(IoError::InvalidPixels);
    }
    let max_mips = 32 - width.max(height).leading_zeros();
    if mips.len() > max_mips as usize {
        return Err(IoError::InvalidPixels);
    }
    let (mut w, mut h) = (width, height);
    let mut size = 148usize;
    for mip in mips {
        let expected = (w.div_ceil(4) as usize)
            .checked_mul(h.div_ceil(4) as usize)
            .and_then(|n| n.checked_mul(16))
            .ok_or(IoError::InvalidPixels)?;
        if mip.len() != expected {
            return Err(IoError::InvalidPixels);
        }
        size = size.checked_add(expected).ok_or(IoError::InvalidPixels)?;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    let mut out = vec![0u8; 148];
    out[..4].copy_from_slice(b"DDS ");
    let mut set =
        |offset: usize, value: u32| out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    set(4, 124);
    set(8, 0x81007 | if mips.len() > 1 { 0x20000 } else { 0 });
    set(12, height);
    set(16, width);
    set(20, mips[0].len() as u32);
    set(28, mips.len() as u32);
    set(76, 32);
    set(80, 4);
    set(84, u32::from_le_bytes(*b"DX10"));
    set(108, 0x1000 | if mips.len() > 1 { 0x400008 } else { 0 });
    set(128, 83);
    set(132, 3);
    set(140, 1);
    out.reserve_exact(size - 148);
    for mip in mips {
        out.extend_from_slice(mip);
    }
    Ok(out)
}
