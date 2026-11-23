use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb, Rgba};
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use texelith_io::{
    ColorIntent, IoError, decode_bytes, decode_path, encode_height_png, encode_normal_png,
    write_height_png, write_normal_png,
};

// Proposed public contract for the implementation pass. Pixels are unassociated RGBA8.
fn encoded(image: DynamicImage, format: ImageFormat) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, format).unwrap();
    bytes.into_inner()
}

fn temp_path(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "texelith-io-{label}-{}-{}.png",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn assert_no_srgb_metadata(png: &[u8]) {
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let mut offset = 8;
    while offset + 12 <= png.len() {
        let len = u32::from_be_bytes(png[offset..offset + 4].try_into().unwrap()) as usize;
        let end = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(len))
            .unwrap();
        assert!(end <= png.len());
        let kind = &png[offset + 4..offset + 8];
        assert_ne!(kind, b"sRGB", "numeric maps must not carry sRGB metadata");
        if kind == b"IEND" {
            return;
        }
        offset = end;
    }
    panic!("missing IEND");
}

#[test]
fn png_detects_odd_width_and_preserves_unassociated_alpha_and_intent() {
    let pixels = vec![200, 100, 50, 0, 120, 90, 60, 128, 3, 2, 1, 255];
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(3, 1, pixels.clone()).unwrap();
    let bytes = encoded(DynamicImage::ImageRgba8(image), ImageFormat::Png);
    for intent in [ColorIntent::PhotoSrgb, ColorIntent::LinearHeight] {
        let decoded = decode_bytes(&bytes, intent).unwrap();
        assert_eq!((decoded.width, decoded.height), (3, 1));
        assert_eq!(decoded.rgba8, pixels);
        assert_eq!(decoded.color_intent, intent);
    }
}

#[test]
fn grayscale_png_expands_to_opaque_rgba() {
    let gray = ImageBuffer::<Luma<u8>, _>::from_raw(3, 1, vec![0, 127, 255]).unwrap();
    let bytes = encoded(DynamicImage::ImageLuma8(gray), ImageFormat::Png);
    let decoded = decode_bytes(&bytes, ColorIntent::LinearHeight).unwrap();
    assert_eq!(
        decoded.rgba8,
        [0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255]
    );
}

#[test]
fn jpeg_is_detected_from_content_and_decoded_through_path() {
    let rgb = ImageBuffer::from_pixel(3, 2, Rgb([80u8, 140, 200]));
    let bytes = encoded(DynamicImage::ImageRgb8(rgb), ImageFormat::Jpeg);
    let path = temp_path("jpeg-content"); // deliberately has a .png extension
    std::fs::write(&path, &bytes).unwrap();
    let from_bytes = decode_bytes(&bytes, ColorIntent::PhotoSrgb).unwrap();
    let from_path = decode_path(&path, ColorIntent::PhotoSrgb).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!((from_path.width, from_path.height), (3, 2));
    assert_eq!(from_path.rgba8, from_bytes.rgba8);
    assert!(
        from_path
            .rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[3] == 255)
    );
}

#[test]
fn failed_install_removes_sibling_temporary_file() {
    let directory = temp_path("cleanup");
    std::fs::create_dir(&directory).unwrap();
    let destination = directory.join("occupied");
    std::fs::create_dir(&destination).unwrap();
    assert!(write_height_png(&destination, 1, 1, &[1], false).is_err());
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    std::fs::remove_dir(&destination).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn overwrite_preserves_directory_and_contents() {
    let directory = temp_path("directory-overwrite");
    std::fs::create_dir(&directory).unwrap();
    let child = directory.join("important");
    std::fs::write(&child, b"keep").unwrap();
    assert!(write_height_png(&directory, 1, 1, &[1], true).is_err());
    assert_eq!(std::fs::read(&child).unwrap(), b"keep");
    std::fs::remove_file(child).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn low_bit_depth_png_is_rejected() {
    let png = encode_normal_png(1, 1, &[1, 2, 3]).unwrap();
    for bits in [1, 2, 4] {
        let mut source = png.clone();
        source[24] = bits;
        assert!(
            matches!(decode_bytes(&source, ColorIntent::PhotoSrgb), Err(IoError::UnsupportedBitDepth { bits: b }) if b == bits)
        );
    }
}

#[test]
fn corrupt_truncated_and_unsupported_sources_are_errors() {
    let png = encoded(
        DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([1u8, 2, 3, 4]))),
        ImageFormat::Png,
    );
    for input in [&b"not an image"[..], &png[..png.len() / 2], &png[..8]] {
        assert!(decode_bytes(input, ColorIntent::PhotoSrgb).is_err());
    }
    let gray16 = ImageBuffer::<Luma<u16>, _>::from_raw(1, 1, vec![12345]).unwrap();
    let source = encoded(DynamicImage::ImageLuma16(gray16), ImageFormat::Png);
    assert!(matches!(
        decode_bytes(&source, ColorIntent::LinearHeight),
        Err(IoError::UnsupportedBitDepth { .. })
    ));
}

#[test]
fn dimensions_and_compressed_input_are_bounded() {
    for (width, height) in [(4097, 1), (1, 4097)] {
        let image = ImageBuffer::from_pixel(width, height, Luma([0u8]));
        let png = encoded(DynamicImage::ImageLuma8(image), ImageFormat::Png);
        assert!(matches!(
            decode_bytes(&png, ColorIntent::LinearHeight),
            Err(IoError::LimitExceeded { .. })
        ));
    }
    let oversized = vec![0; 128 * 1024 * 1024 + 1];
    assert!(matches!(
        decode_bytes(&oversized, ColorIntent::PhotoSrgb),
        Err(IoError::LimitExceeded { .. })
    ));
    let path = temp_path("oversized-input");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(128 * 1024 * 1024 + 1).unwrap();
    drop(file);
    let result = decode_path(&path, ColorIntent::PhotoSrgb);
    std::fs::remove_file(path).unwrap();
    assert!(matches!(result, Err(IoError::LimitExceeded { .. })));
}

#[test]
fn numeric_png_roundtrips_keep_precision_and_have_no_srgb_tag() {
    let heights = [0u16, 1, 255, 256, 32768, 65535];
    let png = encode_height_png(3, 2, &heights).unwrap();
    assert_no_srgb_metadata(&png);
    let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).unwrap();
    assert_eq!(decoded.color(), image::ColorType::L16);
    assert_eq!(decoded.to_luma16().into_raw(), heights);

    let normals = [0u8, 127, 255, 12, 128, 240, 255, 1, 127];
    let png = encode_normal_png(3, 1, &normals).unwrap();
    assert_no_srgb_metadata(&png);
    let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgb8);
    assert_eq!(decoded.to_rgb8().into_raw(), normals);
}

#[test]
fn invalid_output_sizes_fail_without_overflow_or_large_allocation() {
    assert!(encode_height_png(u32::MAX, u32::MAX, &[]).is_err());
    assert!(encode_normal_png(u32::MAX, u32::MAX, &[]).is_err());
    assert!(encode_height_png(0, 1, &[]).is_err());
    assert!(encode_height_png(2, 2, &[1, 2, 3]).is_err());
    assert!(encode_normal_png(2, 1, &[1, 2, 3]).is_err());
}

#[test]
fn writes_require_explicit_overwrite_and_preserve_existing_on_failure() {
    let path = temp_path("overwrite");
    std::fs::write(&path, b"original").unwrap();
    assert!(write_height_png(&path, 1, 1, &[1234], false).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    assert!(write_normal_png(&path, 2, 1, &[0, 0, 0], true).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    write_height_png(&path, 1, 1, &[1234], true).unwrap();
    assert_eq!(image::open(&path).unwrap().to_luma16().into_raw(), [1234]);
    std::fs::remove_file(&path).unwrap();
    write_normal_png(&path, 1, 1, &[1, 2, 3], false).unwrap();
    assert_eq!(image::open(&path).unwrap().to_rgb8().into_raw(), [1, 2, 3]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn fixture_and_inspector_examples_report_reproducible_4k_content() {
    let path = temp_path("fixture");
    let run_fixture = || {
        std::process::Command::new(env!("CARGO"))
            .args([
                "run",
                "-p",
                "texelith-io",
                "--example",
                "make_fixture",
                "--locked",
                "--",
                "--size",
                "4096",
                "--output",
            ])
            .arg(&path)
            .output()
            .unwrap()
    };
    let first = run_fixture();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let pixels = image::open(&path).unwrap().to_rgba8();
    assert_eq!(pixels.dimensions(), (4096, 4096));
    assert_eq!(pixels.as_raw().len(), 64 * 1024 * 1024);
    assert!(pixels.pixels().any(|pixel| pixel != pixels.get_pixel(0, 0)));
    let second = run_fixture();
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(image::open(&path).unwrap().to_rgba8(), pixels);
    let first_report = String::from_utf8(first.stdout).unwrap();
    let second_report = String::from_utf8(second.stdout).unwrap();
    assert_eq!(first_report, second_report);
    assert!(
        first_report.to_lowercase().contains("hash")
            || first_report.to_lowercase().contains("checksum")
    );
    let inspected = std::process::Command::new(env!("CARGO"))
        .args([
            "run",
            "-p",
            "texelith-io",
            "--example",
            "inspect_image",
            "--locked",
            "--",
        ])
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stderr)
    );
    let report = String::from_utf8(inspected.stdout).unwrap().to_lowercase();
    assert!(report.contains("4096") && (report.contains("64 mib") || report.contains("67108864")));
    assert!(report.contains("alpha") && (report.contains("srgb") || report.contains("color")));
}
