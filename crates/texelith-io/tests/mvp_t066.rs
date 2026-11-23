use exr::prelude::*;
use std::io::Cursor;
use texelith_io::{
    EnvironmentError, decode_environment_bytes, decode_environment_bytes_with_limit,
};

// Proposed public contract: row-major top-to-bottom linear RGB f32 pixels.
fn exr_rgb(width: usize, height: usize, rgb: &[[f32; 3]]) -> Vec<u8> {
    let channels = AnyChannels::sort(
        ["R", "G", "B"]
            .into_iter()
            .enumerate()
            .map(|(c, name)| {
                AnyChannel::new(name, FlatSamples::F32(rgb.iter().map(|p| p[c]).collect()))
            })
            .collect(),
    );
    let layer = Layer::new(
        (width, height),
        LayerAttributes::default(),
        Encoding::UNCOMPRESSED,
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

// Radiance RGBE, old-style uncompressed scanlines. Pixel values are exact powers of two.
fn hdr(width: usize, height: usize, pixels: &[[u8; 4]]) -> Vec<u8> {
    let mut out =
        format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n").into_bytes();
    for pixel in pixels {
        out.extend_from_slice(pixel);
    }
    out
}

fn sky() -> Vec<[f32; 3]> {
    // Distinct sky and ground rows expose row inversion and channel swaps.
    vec![
        [4.0, 2.0, 1.0],
        [2.0, 1.0, 0.5],
        [0.25, 0.5, 1.0],
        [0.5, 1.0, 2.0],
    ]
}

#[test]
fn exr_linear_rgb_preserves_hdr_range_and_row_order() {
    let expected = sky();
    let image = decode_environment_bytes(&exr_rgb(2, 2, &expected)).unwrap();
    assert_eq!((image.width, image.height), (2, 2));
    assert_eq!(image.rgb, expected);
}

#[test]
fn radiance_linear_rgb_preserves_hdr_range_and_row_order() {
    let pixels = [
        [128, 64, 32, 131],
        [128, 64, 32, 130],
        [32, 64, 128, 129],
        [32, 64, 128, 130],
    ];
    let expected = sky();
    let image = decode_environment_bytes(&hdr(2, 2, &pixels)).unwrap();
    assert_eq!((image.width, image.height), (2, 2));
    for (actual, expected) in image.rgb.iter().zip(&expected) {
        for c in 0..3 {
            assert!(
                (actual[c] - expected[c]).abs() <= 0.032,
                "{actual:?} != {expected:?}"
            );
        }
    }
    assert_eq!(image.rgb.len(), expected.len());
}

#[test]
fn radiance_accepts_crlf_headers_for_both_signatures() {
    for signature in ["RADIANCE", "RGBE"] {
        let mut bytes =
            format!("#?{signature}\r\nFORMAT=32-bit_rle_rgbe\r\n\r\n-Y 1 +X 1\r\n").into_bytes();
        bytes.extend_from_slice(&[128, 64, 32, 131]);
        assert_eq!(
            decode_environment_bytes(&bytes).unwrap().rgb,
            vec![[4.0, 2.0, 1.0]]
        );
    }
}

#[test]
fn exr_rejects_nonfinite_in_every_channel() {
    for channel in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut pixels = sky();
            pixels[2][channel] = value;
            assert!(decode_environment_bytes(&exr_rgb(2, 2, &pixels)).is_err());
        }
    }
}

#[test]
fn oversized_headers_are_rejected_before_pixels_are_available() {
    for (width, height) in [(2049, 1), (1, 1025), (u32::MAX as usize, 1)] {
        let bytes = hdr(width, height, &[]);
        assert!(decode_environment_bytes(&bytes).is_err());
    }
    let original = exr_rgb(1, 1, &[[1.0; 3]]);
    for (width, height) in [(2049i32, 1i32), (1, 1025), (i32::MAX, i32::MAX)] {
        let mut bytes = original.clone();
        for name in [b"dataWindow\0".as_slice(), b"displayWindow\0".as_slice()] {
            let start = bytes.windows(name.len()).position(|x| x == name).unwrap()
                + name.len()
                + b"box2i\0".len()
                + 4;
            bytes[start + 8..start + 12].copy_from_slice(&(width - 1).to_le_bytes());
            bytes[start + 12..start + 16].copy_from_slice(&(height - 1).to_le_bytes());
        }
        assert!(decode_environment_bytes(&bytes).is_err());
    }
}

#[test]
fn cap_is_inclusive_for_radiance() {
    // RLE scanlines keep the generated maximum-size fixture small.
    let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1024 +X 2048\n".to_vec();
    for _ in 0..1024 {
        bytes.extend_from_slice(&[2, 2, 8, 0]);
        for value in [128u8, 64, 32, 131] {
            for _ in 0..16 {
                bytes.extend_from_slice(&[255, value]);
            }
            bytes.extend_from_slice(&[144, value]);
        }
    }
    let image = decode_environment_bytes(&bytes).unwrap();
    assert_eq!((image.width, image.height), (2048, 1024));
    assert_eq!(image.rgb.len(), 2048 * 1024);
    assert!(image.rgb[0][0] > 1.0);
}

#[test]
fn working_memory_budget_is_enforced_for_both_formats() {
    let exr = exr_rgb(2, 2, &sky());
    let radiance = hdr(2, 2, &[[128, 64, 32, 131]; 4]);
    for bytes in [&exr, &radiance] {
        assert!(decode_environment_bytes_with_limit(bytes, 1).is_err());
        assert!(decode_environment_bytes_with_limit(bytes, 64 * 1024 * 1024).is_ok());
    }
}

fn tiled_exr(width: usize, height: usize, tile: usize, compression: Compression) -> Vec<u8> {
    let pixels = vec![[1.0; 3]; width * height];
    let channels = AnyChannels::sort(
        ["R", "G", "B"]
            .into_iter()
            .enumerate()
            .map(|(c, name)| {
                AnyChannel::new(
                    name,
                    FlatSamples::F32(pixels.iter().map(|p| p[c]).collect()),
                )
            })
            .collect(),
    );
    let layer = Layer::new(
        (width, height),
        LayerAttributes::default(),
        Encoding {
            compression,
            blocks: Blocks::Tiles(Vec2(tile, tile)),
            line_order: LineOrder::Increasing,
        },
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

#[test]
fn header_derived_working_memory_is_charged() {
    let radiance = hdr(64, 64, &[[128, 64, 32, 131]; 4096]);
    let frame_limit = radiance.len() as u64 + 4096;
    assert!(matches!(
        decode_environment_bytes_with_limit(&radiance, frame_limit),
        Err(EnvironmentError::LimitExceeded)
    ));
    assert!(decode_environment_bytes(&radiance).is_ok());

    let base = 48 * 1024 * 1024u64;
    let scanlines = exr_rgb(128, 128, &vec![[1.0; 3]; 128 * 128]);
    assert!(matches!(
        decode_environment_bytes_with_limit(&scanlines, base + scanlines.len() as u64 + 100_000),
        Err(EnvironmentError::LimitExceeded)
    ));
    assert!(decode_environment_bytes(&scanlines).is_ok());
    for (tile, compression, allowance) in [
        (1, Compression::Uncompressed, 128 * 128 * 24 + 100_000),
        (64, Compression::RLE, 128 * 128 * 24 + 100_000),
    ] {
        let bytes = tiled_exr(128, 128, tile, compression);
        let limit = base + bytes.len() as u64 + allowance;
        assert!(bytes.len() as u64 <= limit);
        assert!(matches!(
            decode_environment_bytes_with_limit(&bytes, limit),
            Err(EnvironmentError::LimitExceeded)
        ));
        assert!(decode_environment_bytes(&bytes).is_ok());
    }
}
