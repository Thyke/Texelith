use texelith_core::{ExportColorTag, ExportFormat};
use texelith_gpu::{ExportPlane, export_shared::ExportPayload};
use texelith_io::encoders::*;

pub(super) fn encode(plane: &ExportPlane) -> Result<Vec<u8>, String> {
    let extent = plane.metadata.extent;
    let result = match &plane.payload {
        ExportPayload::Bc5 { mips } if plane.format == ExportFormat::DdsBc5 => {
            let slices: Result<Vec<_>, _> = mips
                .iter()
                .map(|m| {
                    let end = m
                        .offset
                        .checked_add(m.byte_len)
                        .ok_or("Invalid BC5 mip offset")?;
                    plane
                        .pixels
                        .get(m.offset as usize..end as usize)
                        .ok_or("Invalid BC5 mip length")
                })
                .collect();
            encode_dds_bc5(extent.width, extent.height, &slices?)
        }
        ExportPayload::Numeric {
            channels,
            bit_depth,
        } if *channels == plane.channels && *bit_depth == plane.format.bit_depth() => {
            let channels = match channels {
                1 => PngChannels::Scalar,
                3 => PngChannels::Rgb,
                4 => PngChannels::Rgba,
                _ => return Err("Unsupported channel count".into()),
            };
            let intent = if plane.metadata.semantics.color_tag == ExportColorTag::Srgb {
                PngColorIntent::Srgb
            } else {
                PngColorIntent::Linear
            };
            match plane.format {
                ExportFormat::Png8 => {
                    encode_png8(extent.width, extent.height, channels, intent, &plane.pixels)
                }
                ExportFormat::Png16 => {
                    if !plane.pixels.len().is_multiple_of(2) {
                        return Err("Invalid PNG16 plane length".into());
                    }
                    let values: Vec<_> = plane
                        .pixels
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| u16::from_be_bytes(*v))
                        .collect();
                    encode_png16(extent.width, extent.height, channels, intent, &values)
                }
                ExportFormat::Exr32 => {
                    if !plane.pixels.len().is_multiple_of(4) {
                        return Err("Invalid EXR plane length".into());
                    }
                    let values: Vec<_> = plane
                        .pixels
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect();
                    encode_exr32(extent.width, extent.height, &values)
                }
                ExportFormat::DdsBc5 => return Err("BC5 payload required".into()),
            }
        }
        _ => return Err("Export payload contradicts format".into()),
    };
    result.map_err(|e| e.to_string())
}
