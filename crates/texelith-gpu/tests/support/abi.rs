use bytemuck::{Pod, Zeroable};
use texelith_core::Extent;

/// Test-side wire representation of ADR 0003, not the production Rust ABI.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NodeUniform {
    pub dimensions: [u32; 4],
    pub parameters: [f32; 4],
    pub flags: [u32; 4],
    pub reserved: [u32; 4],
}

impl NodeUniform {
    pub fn new(evaluation: Extent, source: Extent) -> Self {
        Self {
            dimensions: [
                evaluation.width,
                evaluation.height,
                source.width,
                source.height,
            ],
            ..Self::zeroed()
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlurWeights {
    pub slots: [[f32; 4]; 8],
}

impl BlurWeights {
    pub fn new(weights: &[f64]) -> Self {
        assert!(weights.len() <= 31);
        let mut block = Self::zeroed();
        for (i, &weight) in weights.iter().enumerate() {
            block.slots[i / 4][i % 4] = weight as f32;
        }
        block
    }
}

pub fn workgroups(extent: Extent) -> Result<[u32; 3], String> {
    extent.validate().map_err(|error| error.to_string())?;
    Ok([extent.width.div_ceil(8), extent.height.div_ceil(8), 1])
}

/// Copy-buffer rows must be padded, but decoded pixels must not include padding.
pub fn row_layout(width: u32, bytes_per_pixel: u32) -> Result<(u32, u32), String> {
    if width == 0 || ![4, 8].contains(&bytes_per_pixel) {
        return Err("invalid readback layout".into());
    }
    let tight = width.checked_mul(bytes_per_pixel).ok_or("row overflow")?;
    let padded = tight.checked_add(255).ok_or("padding overflow")? / 256 * 256;
    Ok((tight, padded))
}

pub fn strip_rows(
    bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, String> {
    let (tight, padded) = row_layout(width, bytes_per_pixel)?;
    let expected = (padded as usize)
        .checked_mul(height as usize)
        .ok_or("size overflow")?;
    if height == 0 || bytes.len() != expected {
        return Err("invalid readback byte count".into());
    }
    Ok(bytes
        .chunks_exact(padded as usize)
        .flat_map(|row| row[..tight as usize].iter().copied())
        .collect())
}

/// Decode IEEE binary16 without adding a dependency or unsafe casts.
pub fn decode_half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let mantissa = f64::from(bits & 1023);
    sign * match exponent {
        0 => mantissa * 2.0_f64.powi(-24),
        31 if mantissa == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15),
    }
}
