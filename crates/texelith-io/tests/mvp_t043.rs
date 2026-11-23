use exr::prelude::*;
use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb};
use std::io::Cursor;
use texelith_io::{
    ColorIntent, IoError, NormalConvention, TypedImage, TypedIntent, decode_bytes,
    decode_typed_bytes, encode_height_png, encode_normal_png,
};

// Proposed IO-owned contract: typed decoding is separate from the legacy RGBA8 API.
fn exr32(width: usize, height: usize, channels: &[(&str, Vec<f32>)]) -> Vec<u8> {
    let channels = AnyChannels::sort(
        channels
            .iter()
            .map(|(name, values)| AnyChannel::new(*name, FlatSamples::F32(values.clone())))
            .collect(),
    );
    let layer = Layer::new(
        (width, height),
        LayerAttributes::default(),
        Encoding::default(),
        channels,
    );
    let mut bytes = Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .non_parallel()
        .to_buffered(&mut bytes)
        .unwrap();
    bytes.into_inner()
}

fn png8(width: u32, height: u32, values: Vec<u8>) -> Vec<u8> {
    let image = DynamicImage::ImageRgb8(
        ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, values).unwrap(),
    );
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    bytes.into_inner()
}

#[test]
fn png16_height_preserves_all_bits_and_singleton_extent() {
    let values = [0, 1, 255, 256, 32768, 65534, 65535];
    let bytes = encode_height_png(1, 7, &values).unwrap();
    let image = decode_typed_bytes(&bytes, TypedIntent::Height).unwrap();
    match image {
        TypedImage::HeightU16 {
            width,
            height,
            values: actual,
        } => {
            assert_eq!((width, height), (1, 7));
            assert_eq!(actual, values);
        }
        other => panic!("expected linear u16 height, got {other:?}"),
    }
    assert!(matches!(
        decode_bytes(&bytes, ColorIntent::LinearHeight),
        Err(IoError::UnsupportedBitDepth { .. })
    ));
}

#[test]
fn exr32_height_preserves_signed_unclamped_finite_values_and_npot() {
    let values = vec![-4.5, -0.001, 0.0, 0.25, 1.0, 100.25];
    let bytes = exr32(3, 2, &[("Y", values.clone())]);
    match decode_typed_bytes(&bytes, TypedIntent::Height).unwrap() {
        TypedImage::HeightF32 {
            width,
            height,
            values: actual,
        } => {
            assert_eq!((width, height), (3, 2));
            assert_eq!(actual, values);
        }
        other => panic!("expected linear f32 height, got {other:?}"),
    }
}

#[test]
fn exr_nonfinite_values_and_multichannel_height_are_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bytes = exr32(1, 1, &[("Y", vec![value])]);
        assert!(decode_typed_bytes(&bytes, TypedIntent::Height).is_err());
    }
    let bytes = exr32(
        1,
        1,
        &[("R", vec![0.25]), ("G", vec![0.5]), ("B", vec![1.0])],
    );
    assert!(decode_typed_bytes(&bytes, TypedIntent::Height).is_err());
}

#[test]
fn normal_rgb8_preserves_encoded_channels_and_convention() {
    let values = vec![128, 128, 255, 0, 255, 128, 255, 0, 1];
    let bytes = encode_normal_png(3, 1, &values).unwrap();
    for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
        match decode_typed_bytes(&bytes, TypedIntent::Normal(convention)).unwrap() {
            TypedImage::NormalRgb8 {
                width,
                height,
                values: actual,
                convention: actual_convention,
            } => {
                assert_eq!((width, height), (3, 1));
                assert_eq!(actual, values);
                assert_eq!(actual_convention, convention);
            }
            other => panic!("expected RGB8 normal, got {other:?}"),
        }
    }
}

#[test]
fn mismatched_interpretations_and_excessive_dimensions_are_errors() {
    let height = encode_height_png(1, 1, &[12345]).unwrap();
    let normal = png8(1, 1, vec![10, 20, 30]);
    let exr = exr32(1, 1, &[("Y", vec![0.5])]);
    assert!(decode_typed_bytes(&height, TypedIntent::Normal(NormalConvention::OpenGl)).is_err());
    assert!(decode_typed_bytes(&normal, TypedIntent::Height).is_err());
    assert!(decode_typed_bytes(&exr, TypedIntent::Normal(NormalConvention::OpenGl)).is_err());
    let oversized = png8(4097, 1, vec![0; 4097 * 3]);
    assert!(matches!(
        decode_typed_bytes(&oversized, TypedIntent::Normal(NormalConvention::OpenGl)),
        Err(IoError::LimitExceeded { .. })
    ));
    let oversized_exr = exr32(1, 4097, &[("Y", vec![0.0; 4097])]);
    assert!(matches!(
        decode_typed_bytes(&oversized_exr, TypedIntent::Height),
        Err(IoError::LimitExceeded { .. })
    ));
}

#[test]
fn legacy_png_and_jpeg_still_decode_as_rgba8() {
    let image = DynamicImage::ImageLuma8(
        ImageBuffer::<Luma<u8>, _>::from_raw(3, 1, vec![0, 127, 255]).unwrap(),
    );
    for format in [ImageFormat::Png, ImageFormat::Jpeg] {
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, format).unwrap();
        let decoded = decode_bytes(bytes.get_ref(), ColorIntent::PhotoSrgb).unwrap();
        assert_eq!((decoded.width, decoded.height), (3, 1));
        assert_eq!(decoded.rgba8.len(), 12);
        assert_eq!(decoded.color_intent, ColorIntent::PhotoSrgb);
    }
}

#[test]
fn oversized_compressed_input_is_rejected_before_metadata_processing() {
    let mut bytes = encode_height_png(1, 1, &[1]).unwrap();
    // A valid PNG prefix followed by oversized ancillary payload still counts as input.
    bytes.resize(128 * 1024 * 1024 + 1, 0);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded { .. })
    ));
}

#[test]
fn exr_tiny_tiles_are_charged_before_offset_table_allocation() {
    let channels = AnyChannels::sort(
        [AnyChannel::new("Y", FlatSamples::F32(vec![0.0; 2]))]
            .into_iter()
            .collect(),
    );
    let encoding = Encoding {
        blocks: Blocks::Tiles(Vec2(2, 1)),
        ..Encoding::default()
    };
    let layer = Layer::new((2, 1), LayerAttributes::default(), encoding, channels);
    let mut output = Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .non_parallel()
        .to_buffered(&mut output)
        .unwrap();
    let mut bytes = output.into_inner();
    for name in [b"dataWindow\0".as_slice(), b"displayWindow\0".as_slice()] {
        let start = bytes
            .windows(name.len())
            .position(|part| part == name)
            .unwrap();
        let value = start + name.len() + b"box2i\0".len() + 4;
        bytes[value + 8..value + 12].copy_from_slice(&4095i32.to_le_bytes());
        bytes[value + 12..value + 16].copy_from_slice(&4095i32.to_le_bytes());
    }
    // Complete offset table, no chunks: a decoder must reject on its budget
    // before constructing the two large in-memory offset tables.
    let mut cursor = 8;
    while bytes[cursor] != 0 {
        for _ in 0..2 {
            cursor += bytes[cursor..].iter().position(|&b| b == 0).unwrap() + 1;
        }
        let len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + len;
    }
    bytes.truncate(cursor + 1);
    bytes.resize(cursor + 1 + 2048 * 4096 * 8, 0);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "decoder allocation"
        })
    ));
}

#[test]
fn oversized_exr_metadata_is_rejected_within_input_limit() {
    let original = exr32(1, 1, &[("Y", vec![0.0])]);
    let mut bytes = original[..8].to_vec();
    bytes.extend_from_slice(b"extra\0opaque\0");
    bytes.extend_from_slice(&(1024u32 * 1024).to_le_bytes());
    bytes.resize(bytes.len() + 1024 * 1024, 0);
    bytes.extend_from_slice(&original[8..]);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "exr metadata"
        })
    ));
}

fn encoded_exr(compression: Compression, blocks: Blocks) -> Vec<u8> {
    let values = vec![-4.5, -0.001, 0.0, 0.25, 1.0, 100.25];
    let channels = AnyChannels::sort(
        [AnyChannel::new("Y", FlatSamples::F32(values))]
            .into_iter()
            .collect(),
    );
    let layer = Layer::new(
        (1, 6),
        LayerAttributes::default(),
        Encoding {
            compression,
            blocks,
            line_order: LineOrder::Increasing,
        },
        channels,
    );
    let mut output = Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .non_parallel()
        .to_buffered(&mut output)
        .unwrap();
    output.into_inner()
}

fn exr_attribute(bytes: &[u8], attribute: &[u8]) -> std::ops::Range<usize> {
    let mut offset = 8;
    while bytes[offset] != 0 {
        let name_end = offset + bytes[offset..].iter().position(|&b| b == 0).unwrap();
        let name = &bytes[offset..name_end];
        offset = name_end + 1;
        offset += bytes[offset..].iter().position(|&b| b == 0).unwrap() + 1;
        let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if name == attribute {
            return offset..offset + len;
        }
        offset += len;
    }
    panic!("missing EXR attribute");
}

fn assert_supported_exr(compression: Compression) {
    for blocks in [Blocks::ScanLines, Blocks::Tiles(Vec2(3, 5))] {
        let bytes = encoded_exr(compression, blocks);
        match decode_typed_bytes(&bytes, TypedIntent::Height).unwrap() {
            TypedImage::HeightF32 {
                width,
                height,
                values,
            } => {
                assert_eq!((width, height), (1, 6));
                assert_eq!(values, [-4.5, -0.001, 0.0, 0.25, 1.0, 100.25]);
            }
            other => panic!("expected f32 height, got {other:?}"),
        }
    }
}

#[test]
fn exr_uncompressed_scanlines_and_tiles_preserve_values() {
    assert_supported_exr(Compression::Uncompressed);
}

#[test]
fn exr_rle_scanlines_and_tiles_preserve_values() {
    assert_supported_exr(Compression::RLE);
}

// Use a complete raw block: EXR permits raw fallback even when the header names
// a compressor. It must still be rejected explicitly, rather than reaching a
// decoder and failing accidentally on corrupt compressed bytes.
fn assert_unsupported_exr(compression_id: u8) {
    let mut bytes = encoded_exr(Compression::Uncompressed, Blocks::Tiles(Vec2(3, 6)));
    let compression = exr_attribute(&bytes, b"compression");
    bytes[compression.start] = compression_id;
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::UnsupportedFormat)
    ));
}

macro_rules! unsupported_exr_compression {
    ($name:ident, $id:expr) => {
        #[test]
        fn $name() {
            assert_unsupported_exr($id);
        }
    };
}

unsupported_exr_compression!(exr_zip1_is_explicitly_unsupported, 2);
unsupported_exr_compression!(exr_zip16_is_explicitly_unsupported, 3);
unsupported_exr_compression!(exr_piz_is_explicitly_unsupported, 4);
unsupported_exr_compression!(exr_pxr24_is_explicitly_unsupported, 5);
unsupported_exr_compression!(exr_b44_is_explicitly_unsupported, 6);
unsupported_exr_compression!(exr_b44a_is_explicitly_unsupported, 7);
unsupported_exr_compression!(exr_dwaa_is_explicitly_unsupported, 8);
unsupported_exr_compression!(exr_dwab_is_explicitly_unsupported, 9);
unsupported_exr_compression!(exr_htj2k32_is_explicitly_unsupported, 10);
unsupported_exr_compression!(exr_htj2k256_is_explicitly_unsupported, 11);

#[test]
fn exr_dwa_section_size_cannot_request_half_a_gibibyte() {
    let channels = AnyChannels::sort(
        [AnyChannel::new("Y", FlatSamples::F32(vec![0.0; 64]))]
            .into_iter()
            .collect(),
    );
    let layer = Layer::new(
        (64, 1),
        LayerAttributes::default(),
        Encoding::UNCOMPRESSED,
        channels,
    );
    let mut output = Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .non_parallel()
        .to_buffered(&mut output)
        .unwrap();
    let original = output.into_inner();
    // One scanline chunk follows the single u64 offset entry. Locate it through
    // the EXR reader's header-only API; no pixel decompression is performed here.
    let mut cursor = Cursor::new(original.as_slice());
    exr::meta::MetaData::read_from_buffered(&mut cursor, false).unwrap();
    let table = cursor.position() as usize;
    let chunk = u64::from_le_bytes(original[table..table + 8].try_into().unwrap()) as usize;
    for compression_id in [8, 9] {
        let mut bytes = original[..chunk + 8].to_vec();
        let compression = exr_attribute(&bytes, b"compression");
        bytes[compression.start] = compression_id;
        // Version 1, UNKNOWN uncompressed size = 512 MiB, compressed size = 8.
        // The remaining DWA section sizes are zero. This fits in a 96-byte block.
        bytes[chunk + 4..chunk + 8].copy_from_slice(&96i32.to_le_bytes());
        for counter in [1u64, 512 * 1024 * 1024, 8, 0, 0, 0, 0, 0, 0, 0, 0] {
            bytes.extend_from_slice(&counter.to_le_bytes());
        }
        bytes.extend_from_slice(&[0; 8]);
        assert!(matches!(
            decode_typed_bytes(&bytes, TypedIntent::Height),
            Err(IoError::UnsupportedFormat)
        ));
    }
}

#[test]
fn exr_rle_large_tile_scratch_is_rejected_before_reading_chunks() {
    let mut bytes = encoded_exr(Compression::RLE, Blocks::Tiles(Vec2(3, 6)));
    let tile = exr_attribute(&bytes, b"tiles");
    bytes[tile.start..tile.start + 4].copy_from_slice(&4096u32.to_le_bytes());
    bytes[tile.start + 4..tile.start + 8].copy_from_slice(&4096u32.to_le_bytes());
    // Tiny image, tiny payload, but the chunk reader trusts the advertised tile
    // size. Reject its worst-case RLE expansion even though actual edge pixels fit.
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "decoder allocation"
        })
    ));
}

#[test]
fn exr_uncompressed_large_block_is_included_in_peak_budget() {
    let mut bytes = encoded_exr(Compression::Uncompressed, Blocks::Tiles(Vec2(3, 6)));
    for name in [b"dataWindow".as_slice(), b"displayWindow".as_slice()] {
        let window = exr_attribute(&bytes, name);
        bytes[window.start + 8..window.start + 12].copy_from_slice(&4095i32.to_le_bytes());
        bytes[window.start + 12..window.start + 16].copy_from_slice(&4095i32.to_le_bytes());
    }
    let tile = exr_attribute(&bytes, b"tiles");
    bytes[tile.start..tile.start + 4].copy_from_slice(&4096u32.to_le_bytes());
    bytes[tile.start + 4..tile.start + 8].copy_from_slice(&4096u32.to_le_bytes());
    bytes.resize(17 * 1024 * 1024, 0);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "decoder allocation"
        })
    ));
}

#[test]
fn exr_block_coordinate_vector_is_budgeted_alongside_offset_tables() {
    let mut bytes = encoded_exr(Compression::Uncompressed, Blocks::Tiles(Vec2(16, 1)));
    for name in [b"dataWindow".as_slice(), b"displayWindow".as_slice()] {
        let window = exr_attribute(&bytes, name);
        bytes[window.start + 8..window.start + 12].copy_from_slice(&4095i32.to_le_bytes());
        bytes[window.start + 12..window.start + 16].copy_from_slice(&4095i32.to_le_bytes());
    }
    let mut cursor = Cursor::new(bytes.as_slice());
    exr::meta::MetaData::read_from_buffered(&mut cursor, false).unwrap();
    let table = cursor.position() as usize;
    // A complete 8 MiB table passes an offsets-only estimate. exr also collects
    // over a million TileIndices before filtering those offsets.
    bytes.truncate(table);
    bytes.resize(table + 256 * 4096 * 8, 0);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "decoder allocation"
        })
    ));
}

#[test]
fn exr_malformed_rle_that_overshoots_expected_size_is_an_error() {
    let original = encoded_exr(Compression::RLE, Blocks::Tiles(Vec2(3, 6)));
    let mut cursor = Cursor::new(original.as_slice());
    exr::meta::MetaData::read_from_buffered(&mut cursor, false).unwrap();
    let table = cursor.position() as usize;
    let chunk = u64::from_le_bytes(original[table..table + 8].try_into().unwrap()) as usize;
    // A tile chunk has four i32 coordinates followed by its i32 payload size.
    let mut bytes = original[..chunk + 20].to_vec();
    bytes[chunk + 16..chunk + 20].copy_from_slice(&8i32.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 127, 0, 127, 0, 127, 0]);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::InvalidPixels)
    ));
}

#[test]
fn exr_header_dimensions_are_bounded_before_dependency_chunk_count_arithmetic() {
    let original = encoded_exr(Compression::RLE, Blocks::Tiles(Vec2(3, 6)));
    let mut oversized = original.clone();
    let window = exr_attribute(&oversized, b"dataWindow");
    oversized[window.start + 8..window.start + 12].copy_from_slice(&i32::MAX.to_le_bytes());
    oversized[window.start + 12..window.start + 16].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(matches!(
        decode_typed_bytes(&oversized, TypedIntent::Height),
        Err(IoError::LimitExceeded { .. })
    ));
    for axis in [0, 4] {
        let mut zero_tile = original.clone();
        let tile = exr_attribute(&zero_tile, b"tiles");
        zero_tile[tile.start + axis..tile.start + axis + 4].fill(0);
        assert!(decode_typed_bytes(&zero_tile, TypedIntent::Height).is_err());
    }
}

#[test]
fn exr_metadata_attribute_count_is_bounded() {
    let original = encoded_exr(Compression::Uncompressed, Blocks::ScanLines);
    let mut bytes = original[..8].to_vec();
    for index in 0..257 {
        bytes.extend_from_slice(format!("extra{index}\0opaque\0").as_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    bytes.extend_from_slice(&original[8..]);
    assert!(matches!(
        decode_typed_bytes(&bytes, TypedIntent::Height),
        Err(IoError::LimitExceeded {
            reason: "exr metadata"
        })
    ));
}
