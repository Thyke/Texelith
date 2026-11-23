use crate::GpuError;
use std::sync::Arc;
use texelith_core::{Extent, InputInterpretation, NormalConvention, SourceDescriptor};

#[derive(Clone)]
pub enum TypedSourcePixels {
    Rgba8 {
        pixels: Arc<[u8]>,
        interpretation: InputInterpretation,
    },
    HeightU16(Arc<[u16]>),
    HeightF32(Arc<[f32]>),
    NormalRgb8 {
        pixels: Arc<[u8]>,
        convention: NormalConvention,
    },
}
#[derive(Clone)]
pub struct TypedSourceUpload {
    pub descriptor: SourceDescriptor,
    pub pixels: TypedSourcePixels,
}
impl TypedSourcePixels {
    pub(crate) fn validate_length(&self, extent: Extent) -> Result<(), GpuError> {
        extent.validate()?;
        let count = extent.width as usize * extent.height as usize;
        let valid = match self {
            Self::Rgba8 {
                pixels,
                interpretation,
            } => pixels.len() == count * 4 && *interpretation != InputInterpretation::LinearNormal,
            Self::HeightU16(pixels) => pixels.len() == count,
            Self::HeightF32(pixels) => pixels.len() == count,
            Self::NormalRgb8 { pixels, .. } => pixels.len() == count * 3,
        };
        if valid {
            Ok(())
        } else {
            Err(GpuError(
                "invalid typed source length or interpretation".into(),
            ))
        }
    }
    /// Worker/import-worker operation. No paths, host endian bytes, or quantization.
    pub fn descriptor(&self, extent: Extent) -> Result<SourceDescriptor, GpuError> {
        self.validate_length(extent)?;
        if let Self::Rgba8 {
            pixels,
            interpretation,
        } = self
        {
            return Ok(SourceDescriptor::from_rgba8(
                pixels,
                extent,
                *interpretation,
            )?);
        }
        // Reuse core's BLAKE3 boundary without adding a dependency upward or
        // changing the frozen legacy hash. Four canonical bytes per pixel bound
        // temporary storage to 64 MiB; a second domain-separated digest adds type.
        let mut bytes = Vec::with_capacity(extent.width as usize * extent.height as usize * 4);
        let (tag, convention, interpretation) = match self {
            Self::HeightU16(pixels) => {
                for value in pixels.iter() {
                    bytes.extend_from_slice(&u32::from(*value).to_le_bytes());
                }
                (1u32, 0u32, InputInterpretation::LinearHeight)
            }
            Self::HeightF32(pixels) => {
                for value in pixels.iter() {
                    if !value.is_finite() {
                        return Err(GpuError("nonfinite height source".into()));
                    }
                    bytes.extend_from_slice(
                        &(if *value == 0.0 { 0.0 } else { *value }).to_le_bytes(),
                    );
                }
                (2, 0, InputInterpretation::LinearHeight)
            }
            Self::NormalRgb8 { pixels, convention } => {
                for pixel in pixels.as_chunks::<3>().0 {
                    bytes.extend_from_slice(pixel);
                    bytes.push(0);
                }
                (
                    3,
                    u32::from(*convention == NormalConvention::DirectX),
                    InputInterpretation::LinearNormal,
                )
            }
            Self::Rgba8 { .. } => unreachable!(),
        };
        let payload = SourceDescriptor::from_rgba8(&bytes, extent, interpretation)?.content_hash;
        let mut identity = Vec::from(&b"texelith.typed-source.v1\0\0"[..]);
        for value in [tag, convention, extent.width, extent.height] {
            identity.extend_from_slice(&value.to_le_bytes());
        }
        identity.extend_from_slice(&payload);
        while identity.len() % 4 != 0 {
            identity.push(0);
        }
        let hash = SourceDescriptor::from_rgba8(
            &identity,
            Extent::new((identity.len() / 4) as u32, 1)?,
            interpretation,
        )?
        .content_hash;
        Ok(SourceDescriptor {
            content_hash: hash,
            extent,
            interpretation,
        })
    }
}
