use image::{DynamicImage, ImageFormat};
use std::io::Cursor;
use texelith_io::encoders::{
    PngChannels, PngColorIntent, encode_dds_bc5, encode_exr32, encode_png8, encode_png16,
};
use texelith_io::{TypedImage, TypedIntent, decode_typed_bytes};

// Proposed leaf API: tightly packed interleaved samples and complete mip slices.
fn chunks(bytes: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let mut result = Vec::new();
    let mut pos = 8;
    while pos < bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = bytes[pos + 4..pos + 8].try_into().unwrap();
        result.push((kind, bytes[pos + 8..pos + 8 + len].to_vec()));
        pos += len + 12;
    }
    result
}

#[test]
fn png8_layout_depth_tags_and_samples() {
    for (channels, count, color_type) in [
        (PngChannels::Scalar, 1, 0),
        (PngChannels::Rgb, 3, 2),
        (PngChannels::Rgba, 4, 6),
    ] {
        let input: Vec<u8> = (0..2 * count).map(|x| (x * 31 + 1) as u8).collect();
        let bytes = encode_png8(2, 1, channels, PngColorIntent::Linear, &input).unwrap();
        let c = chunks(&bytes);
        assert_eq!(&c[0].1[8..10], &[8, color_type]);
        assert!(
            !c.iter()
                .any(|(name, _)| name == b"sRGB" || name == b"gAMA" || name == b"iCCP")
        );
        let decoded = image::load_from_memory_with_format(&bytes, ImageFormat::Png).unwrap();
        assert_eq!(decoded.as_bytes(), input);
    }
    let bytes = encode_png8(
        1,
        1,
        PngChannels::Rgb,
        PngColorIntent::Srgb,
        &[20, 100, 220],
    )
    .unwrap();
    assert!(chunks(&bytes).iter().any(|(name, _)| name == b"sRGB"));
    assert_eq!(
        image::load_from_memory(&bytes).unwrap().as_bytes(),
        &[20, 100, 220]
    );
}

#[test]
fn png16_layout_precision_and_linear_tag() {
    for (channels, count, color_type) in [
        (PngChannels::Scalar, 1, 0),
        (PngChannels::Rgb, 3, 2),
        (PngChannels::Rgba, 4, 6),
    ] {
        let samples = [0, 1, 255, 256, 32768, 65535, 12345, 54321];
        let input = &samples[..2 * count];
        let bytes = encode_png16(2, 1, channels, PngColorIntent::Linear, input).unwrap();
        let c = chunks(&bytes);
        assert_eq!(&c[0].1[8..10], &[16, color_type]);
        assert!(
            !c.iter()
                .any(|(name, _)| name == b"sRGB" || name == b"gAMA" || name == b"iCCP")
        );
        let decoded = image::load_from_memory(&bytes).unwrap();
        let actual: Vec<u16> = match decoded {
            DynamicImage::ImageLuma16(i) => i.into_raw(),
            DynamicImage::ImageRgb16(i) => i.into_raw(),
            DynamicImage::ImageRgba16(i) => i.into_raw(),
            other => panic!("unexpected decoded layout: {other:?}"),
        };
        assert_eq!(actual, input);
    }
}

#[test]
fn exr_is_single_y_float_and_preserves_signed_values() {
    let values = [-4.5, -0.001, 0.0, 0.25, 1.0, 100.25];
    let bytes = encode_exr32(3, 2, &values).unwrap();
    let metadata =
        exr::meta::MetaData::read_from_buffered(&mut Cursor::new(&bytes), false).unwrap();
    let channels = &metadata.headers[0].channels.list;
    assert_eq!(channels.len(), 1);
    assert_eq!(channels[0].name.to_string(), "Y");
    assert_eq!(
        channels[0].sample_type,
        exr::meta::attribute::SampleType::F32
    );
    match decode_typed_bytes(&bytes, TypedIntent::Height).unwrap() {
        TypedImage::HeightF32 {
            width,
            height,
            values: actual,
        } => {
            assert_eq!((width, height), (3, 2));
            assert_eq!(actual, values);
        }
        other => panic!("unexpected EXR: {other:?}"),
    }
}

fn le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn dds_dx10_bc5_mips_have_exact_headers_and_payloads() {
    let base = vec![0x13; 64]; // 8x8: four BC5 blocks
    let mip1 = vec![0x57; 16]; // 4x4
    let mip2 = vec![0x9b; 16]; // 2x2, one padded block
    let mip3 = vec![0xdf; 16]; // 1x1, one padded block
    let bytes = encode_dds_bc5(8, 8, &[&base, &mip1, &mip2, &mip3]).unwrap();
    assert_eq!(&bytes[..4], b"DDS ");
    assert_eq!(le(&bytes, 4), 124);
    assert_eq!((le(&bytes, 12), le(&bytes, 16)), (8, 8));
    assert_eq!(le(&bytes, 20), 64);
    assert_eq!(le(&bytes, 28), 4);
    assert_eq!(le(&bytes, 76), 32);
    assert_eq!(&bytes[84..88], b"DX10");
    assert_eq!(le(&bytes, 128), 83); // DXGI_FORMAT_BC5_UNORM
    assert_eq!(le(&bytes, 132), 3); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    assert_eq!(le(&bytes, 140), 1); // array size
    assert_eq!(&bytes[148..], [&base[..], &mip1, &mip2, &mip3].concat());
}

#[test]
fn invalid_dimensions_lengths_and_mips_are_rejected() {
    for (w, h) in [(0, 1), (1, 0), (4097, 1), (u32::MAX, u32::MAX)] {
        assert!(encode_png8(w, h, PngChannels::Rgba, PngColorIntent::Linear, &[]).is_err());
        assert!(encode_png16(w, h, PngChannels::Rgba, PngColorIntent::Linear, &[]).is_err());
        assert!(encode_exr32(w, h, &[]).is_err());
        assert!(encode_dds_bc5(w, h, &[]).is_err());
    }
    assert!(encode_png8(2, 1, PngChannels::Rgb, PngColorIntent::Linear, &[0; 5]).is_err());
    assert!(encode_png16(2, 1, PngChannels::Rgba, PngColorIntent::Linear, &[0; 9]).is_err());
    assert!(encode_exr32(2, 1, &[0.0]).is_err());
    assert!(encode_exr32(1, 1, &[f32::NAN]).is_err());
    assert!(encode_dds_bc5(6, 4, &[&[0; 32]]).is_ok());
    assert!(encode_dds_bc5(4, 4, &[&[0; 15]]).is_err());
    assert!(encode_dds_bc5(4, 4, &[&[0; 16], &[0; 16], &[0; 16], &[0; 16]]).is_err());
}

#[test]
fn odd_bc5_dds_keeps_logical_dimensions_and_complete_mips() {
    for (width, height) in [(1u32, 1u32), (3, 5), (1023, 1001), (4095, 4093)] {
        let (mut w, mut h) = (width, height);
        let mut mips = Vec::new();
        loop {
            mips.push(vec![0x80; (w.div_ceil(4) * h.div_ceil(4) * 16) as usize]);
            if w == 1 && h == 1 {
                break;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        let slices: Vec<_> = mips.iter().map(Vec::as_slice).collect();
        let bytes = encode_dds_bc5(width, height, &slices).unwrap();
        assert_eq!((le(&bytes, 16), le(&bytes, 12)), (width, height));
        assert_eq!(le(&bytes, 20) as usize, mips[0].len());
        assert_eq!(le(&bytes, 28) as usize, mips.len());
        assert_eq!(&bytes[148..], mips.concat());
    }
}
