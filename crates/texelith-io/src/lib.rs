use image::{ImageDecoder, ImageEncoder, ImageFormat, ImageReader};
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// FAZ1_IO_MODULES: T-066
mod environment;
pub use environment::{
    EnvironmentError, EnvironmentImage, decode_environment_bytes,
    decode_environment_bytes_with_limit,
};

mod source;
mod typed;
pub use typed::{NormalConvention, TypedImage, TypedIntent, decode_typed_bytes};

// FAZ1_IO_MODULES: T-064 begin
pub mod mesh;
// FAZ1_IO_MODULES: T-064 end

const MAX_INPUT: u64 = 128 * 1024 * 1024;
const MAX_ALLOC: u64 = 256 * 1024 * 1024;
const MAX_AXIS: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorIntent {
    PhotoSrgb,
    LinearHeight,
}

/// Unassociated RGBA8 pixels. ICC and EXIF are ignored; downstream processing ignores alpha.
/// Photo RGB is interpreted as sRGB; linear-height input uses the red channel directly.
#[derive(Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub color_intent: ColorIntent,
}

#[derive(Debug)]
pub enum IoError {
    Io(std::io::Error),
    Image(image::ImageError),
    UnsupportedFormat,
    UnsupportedBitDepth { bits: u8 },
    LimitExceeded { reason: &'static str },
    InvalidPixels,
    DestinationExists,
}
impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for IoError {}
impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<image::ImageError> for IoError {
    fn from(e: image::ImageError) -> Self {
        if matches!(e, image::ImageError::Limits(_)) {
            Self::LimitExceeded {
                reason: "decoder limit",
            }
        } else {
            Self::Image(e)
        }
    }
}
fn pixels(width: u32, height: u32, channels: usize) -> Result<usize, IoError> {
    if width == 0 || height == 0 || width > MAX_AXIS || height > MAX_AXIS {
        return Err(IoError::LimitExceeded {
            reason: "image dimensions",
        });
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(IoError::LimitExceeded {
            reason: "image size",
        })
}
/// Decode only 8-bit PNG or JPEG; color intent is recorded without conversion.
/// The 256 MiB working budget includes the input slice, sanitized compressed data,
/// decoder scratch, decoded pixels and RGBA conversion. Inputs that cannot fit are rejected.
pub fn decode_bytes(bytes: &[u8], color_intent: ColorIntent) -> Result<DecodedImage, IoError> {
    decode_with_limit(bytes, color_intent, MAX_ALLOC)
}

fn decode_with_limit(
    bytes: &[u8],
    color_intent: ColorIntent,
    max_alloc: u64,
) -> Result<DecodedImage, IoError> {
    decode_with_input_cost(bytes, color_intent, max_alloc, bytes.len() as u64)
}

fn decode_with_input_cost(
    bytes: &[u8],
    color_intent: ColorIntent,
    max_alloc: u64,
    input_cost: u64,
) -> Result<DecodedImage, IoError> {
    if bytes.len() as u64 > MAX_INPUT {
        return Err(IoError::LimitExceeded {
            reason: "compressed input",
        });
    }
    let format = image::guess_format(bytes).map_err(|_| IoError::UnsupportedFormat)?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return Err(IoError::UnsupportedFormat);
    }
    let prepared = source::prepare(bytes, format, max_alloc, input_cost)?;
    let mut reader = ImageReader::with_format(Cursor::new(prepared.bytes.as_slice()), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_AXIS);
    limits.max_image_height = Some(MAX_AXIS);
    limits.max_alloc = Some(prepared.decoder_budget);
    reader.limits(limits);
    let decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    let size = pixels(width, height, 4)?;
    if (width, height) != prepared.dimensions || decoder.total_bytes() > size as u64 {
        return Err(IoError::InvalidPixels);
    }
    if decoder.color_type().bits_per_pixel() / decoder.color_type().channel_count() as u16 > 8 {
        return Err(IoError::UnsupportedBitDepth { bits: 16 });
    }
    let rgba8 = image::DynamicImage::from_decoder(decoder)?
        .into_rgba8()
        .into_raw();
    if rgba8.len() != size {
        return Err(IoError::InvalidPixels);
    }
    Ok(DecodedImage {
        width,
        height,
        rgba8,
        color_intent,
    })
}

pub fn decode_path(path: impl AsRef<Path>, intent: ColorIntent) -> Result<DecodedImage, IoError> {
    let mut file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length > MAX_INPUT {
        return Err(IoError::LimitExceeded {
            reason: "compressed input",
        });
    }
    let mut bytes = Vec::with_capacity(length as usize);
    // Exact growth avoids read_to_end's geometric capacity overshoot at the 128 MiB boundary.
    // Checking every read also bounds files that grow after metadata was read.
    let mut chunk = [0; 8192];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        if bytes.len() as u64 + read as u64 > MAX_INPUT {
            return Err(IoError::LimitExceeded {
                reason: "compressed input",
            });
        }
        bytes.reserve_exact(read);
        bytes.extend_from_slice(&chunk[..read]);
    }
    decode_with_input_cost(&bytes, intent, MAX_ALLOC, bytes.capacity() as u64)
}
pub fn encode_height_png(width: u32, height: u32, values: &[u16]) -> Result<Vec<u8>, IoError> {
    if pixels(width, height, 1)? != values.len() {
        return Err(IoError::InvalidPixels);
    }
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(
        &bytes,
        width,
        height,
        image::ExtendedColorType::L16,
    )?;
    Ok(out)
}
pub fn encode_normal_png(width: u32, height: u32, values: &[u8]) -> Result<Vec<u8>, IoError> {
    if pixels(width, height, 3)? != values.len() {
        return Err(IoError::InvalidPixels);
    }
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(
        values,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(out)
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn write_png(path: &Path, data: &[u8], overwrite: bool) -> Result<(), IoError> {
    write_png_checked(path, data, overwrite, || true)
}

fn write_png_checked(
    path: &Path,
    data: &[u8],
    overwrite: bool,
    current: impl Fn() -> bool,
) -> Result<(), IoError> {
    if overwrite && fs::metadata(path).is_ok_and(|metadata| metadata.is_dir()) {
        return Err(IoError::Io(std::io::Error::new(
            std::io::ErrorKind::IsADirectory,
            "destination is a directory",
        )));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temp, mut file) = loop {
        let name = format!(
            ".texelith-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let temp = Temp(parent.join(name));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp.0)
        {
            Ok(file) => break (temp, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                std::mem::forget(temp);
            }
            Err(e) => return Err(e.into()),
        }
    };
    file.write_all(data)?;
    file.flush()?;
    drop(file);
    if !current() {
        return Err(IoError::Io(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "Save cancelled: document changed",
        )));
    }
    if overwrite {
        // On Windows rename cannot replace a destination. Keep a sibling backup for rollback.
        if path.exists() {
            let backup = temp.0.with_extension("bak");
            fs::rename(path, &backup)?;
            if let Err(e) = fs::rename(&temp.0, path) {
                let _ = fs::rename(&backup, path);
                return Err(e.into());
            }
            fs::remove_file(backup)?;
        } else {
            fs::rename(&temp.0, path)?;
        }
    } else {
        install_new(&temp.0, path, false)?;
    }
    Ok(())
}
fn install_new(temp: &Path, path: &Path, force_copy: bool) -> Result<(), IoError> {
    if !force_copy {
        match fs::hard_link(temp, path) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(IoError::DestinationExists);
            }
            Err(_) => {}
        }
    }
    copy_new(fs::File::open(temp)?, path)
}

fn copy_new(mut source: impl Read, path: &Path) -> Result<(), IoError> {
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                IoError::DestinationExists
            } else {
                IoError::Io(e)
            }
        })?;
    if let Err(e) = std::io::copy(&mut source, &mut destination).and_then(|_| destination.flush()) {
        drop(destination);
        let _ = fs::remove_file(path);
        return Err(e.into());
    }
    Ok(())
}
pub fn write_height_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    values: &[u16],
    overwrite: bool,
) -> Result<(), IoError> {
    write_png(
        path.as_ref(),
        &encode_height_png(width, height, values)?,
        overwrite,
    )
}
pub fn write_normal_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    values: &[u8],
    overwrite: bool,
) -> Result<(), IoError> {
    write_png(
        path.as_ref(),
        &encode_normal_png(width, height, values)?,
        overwrite,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoder_allocation_limit_is_enforced() {
        let png = encode_normal_png(16, 16, &[42; 16 * 16 * 3]).unwrap();
        assert!(matches!(
            decode_with_limit(&png, ColorIntent::PhotoSrgb, 64),
            Err(IoError::LimitExceeded { .. })
        ));
    }
    #[test]
    fn metadata_is_removed_before_decoder_allocation() {
        let png = encode_normal_png(1, 1, &[1, 2, 3]).unwrap();
        let mut decorated = png[..33].to_vec();
        let metadata = vec![0xff; 1024 * 1024];
        decorated.extend_from_slice(&(metadata.len() as u32).to_be_bytes());
        decorated.extend_from_slice(b"tEXt");
        decorated.extend_from_slice(&metadata);
        decorated.extend_from_slice(&[0; 4]);
        decorated.extend_from_slice(&png[33..]);
        // The working budget now includes the original input as well as decoder memory.
        assert!(matches!(
            decode_with_limit(&decorated, ColorIntent::PhotoSrgb, 4096),
            Err(IoError::LimitExceeded { .. })
        ));
        let prepared = source::prepare(
            &decorated,
            ImageFormat::Png,
            10 * 1024 * 1024,
            decorated.len() as u64,
        )
        .unwrap();
        assert_eq!(prepared.bytes, png);
        assert_eq!(
            decode_with_limit(&decorated, ColorIntent::PhotoSrgb, 10 * 1024 * 1024)
                .unwrap()
                .rgba8,
            [1, 2, 3, 255]
        );
    }
    fn jpeg(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut out)
            .encode(
                &vec![127; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        out
    }

    fn jpeg_segment(out: &mut Vec<u8>, marker: u8, data: &[u8]) {
        out.extend_from_slice(&[0xff, marker]);
        out.extend_from_slice(&u16::try_from(data.len() + 2).unwrap().to_be_bytes());
        out.extend_from_slice(data);
    }

    #[test]
    fn jpeg_extended_xmp_cannot_allocate_from_its_advertised_size() {
        let original = jpeg(1, 1);
        let mut decorated = original[..2].to_vec();
        let mut xmp = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
        xmp.extend_from_slice(b"0123456789abcdef0123456789abcdef");
        xmp.extend_from_slice(&u32::MAX.to_be_bytes());
        xmp.extend_from_slice(&0u32.to_be_bytes());
        xmp.push(42);
        jpeg_segment(&mut decorated, 0xe1, &xmp);
        decorated.extend_from_slice(&original[2..]);
        let decoded = decode_bytes(&decorated, ColorIntent::PhotoSrgb).unwrap();
        assert_eq!(
            decoded.rgba8,
            decode_bytes(&original, ColorIntent::PhotoSrgb)
                .unwrap()
                .rgba8
        );
    }

    #[test]
    fn jpeg_large_extended_xmp_and_all_app_segments_are_stripped() {
        let original = jpeg(3, 1);
        let total = 96 * 1024 * 1024u32;
        let mut decorated = Vec::with_capacity(total as usize + 512 * 1024);
        decorated.extend_from_slice(&original[..2]);
        let mut offset = 0u32;
        while offset < total {
            let mut xmp = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
            xmp.extend_from_slice(b"0123456789abcdef0123456789abcdef");
            xmp.extend_from_slice(&total.to_be_bytes());
            xmp.extend_from_slice(&offset.to_be_bytes());
            let count = (total - offset).min(60 * 1024);
            xmp.resize(xmp.len() + count as usize, 42);
            jpeg_segment(&mut decorated, 0xe1, &xmp);
            offset += count;
        }
        for marker in (0xe0..=0xef).chain([0xfe]) {
            jpeg_segment(&mut decorated, marker, &[0; 32]);
        }
        decorated.extend_from_slice(&original[2..]);
        let prepared = source::prepare(
            &decorated,
            ImageFormat::Jpeg,
            MAX_ALLOC,
            decorated.len() as u64,
        )
        .unwrap();
        let plain = source::prepare(
            &original,
            ImageFormat::Jpeg,
            MAX_ALLOC,
            original.len() as u64,
        )
        .unwrap();
        assert_eq!(prepared.bytes, plain.bytes);
        // 128 MiB cannot hold the 96 MiB input plus 32 MiB scratch and pixels.
        assert!(matches!(
            decode_with_limit(&decorated, ColorIntent::PhotoSrgb, 128 * 1024 * 1024),
            Err(IoError::LimitExceeded { .. })
        ));
        assert_eq!(
            decode_bytes(&decorated, ColorIntent::PhotoSrgb)
                .unwrap()
                .rgba8,
            decode_bytes(&original, ColorIntent::PhotoSrgb)
                .unwrap()
                .rgba8
        );
    }

    #[test]
    fn progressive_jpeg_strips_metadata_between_scans() {
        // A 3x1 constant grayscale image: DC category zero, then AC end-of-block.
        let mut encoded = vec![0xff, 0xd8];
        let mut quantization = vec![1; 65];
        quantization[0] = 0;
        jpeg_segment(&mut encoded, 0xdb, &quantization);
        jpeg_segment(&mut encoded, 0xc2, &[8, 0, 1, 0, 3, 1, 1, 0x11, 0]);
        let mut table = [0; 18];
        table[1] = 1; // One 1-bit code, with symbol zero.
        jpeg_segment(&mut encoded, 0xc4, &table);
        table[0] = 0x10;
        jpeg_segment(&mut encoded, 0xc4, &table);
        jpeg_segment(&mut encoded, 0xda, &[1, 1, 0, 0, 0, 0]);
        encoded.push(0x7f);
        let split = encoded.len();
        jpeg_segment(&mut encoded, 0xda, &[1, 1, 0, 1, 63, 0]);
        encoded.extend_from_slice(&[0x7f, 0xff, 0xd9]);
        let mut decorated = encoded[..split].to_vec();
        jpeg_segment(&mut decorated, 0xe1, b"Exif\0\0invalid EXIF is ignored");
        jpeg_segment(&mut decorated, 0xe2, b"ICC_PROFILE\0invalid ICC is ignored");
        jpeg_segment(&mut decorated, 0xfe, b"ignored comment");
        decorated.extend_from_slice(&encoded[split..]);
        let prepared = source::prepare(
            &decorated,
            ImageFormat::Jpeg,
            MAX_ALLOC,
            decorated.len() as u64,
        )
        .unwrap();
        assert_eq!(prepared.bytes, encoded);
        let image = decode_bytes(&decorated, ColorIntent::LinearHeight).unwrap();
        assert_eq!(image.rgba8, [128, 128, 128, 255].repeat(3));
    }

    #[test]
    fn preflight_accounts_for_input_copies_pixels_and_coefficients() {
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let encoded = if format == ImageFormat::Png {
                encode_normal_png(3, 1, &[127; 9]).unwrap()
            } else {
                jpeg(3, 1)
            };
            let scratch = if format == ImageFormat::Png { 8 } else { 32 } * 1024 * 1024;
            let copies = if format == ImageFormat::Png { 1 } else { 2 };
            let pixels = if format == ImageFormat::Png {
                24
            } else {
                3 * 64 * 2 + 12
            };
            let prepared =
                source::prepare(&encoded, format, MAX_ALLOC, encoded.len() as u64).unwrap();
            let required =
                encoded.len() as u64 + prepared.bytes.len() as u64 * copies + pixels + scratch;
            assert!(source::prepare(&encoded, format, required, encoded.len() as u64).is_ok());
            assert!(matches!(
                source::prepare(&encoded, format, required - 1, encoded.len() as u64),
                Err(IoError::LimitExceeded { .. })
            ));
            // A path input whose Vec capacity exceeds its length pays for that capacity.
            assert!(matches!(
                source::prepare(&encoded, format, required, encoded.len() as u64 + 1),
                Err(IoError::LimitExceeded { .. })
            ));
        }
        // A tiny stream can advertise a full 4K frame. Charge full-frame coefficients
        // for baseline as well as progressive/non-interleaved JPEGs before constructing it.
        let mut large = jpeg(1, 1);
        let sof = large
            .windows(2)
            .position(|part| part == [0xff, 0xc0])
            .unwrap();
        large[sof + 5..sof + 9].copy_from_slice(&[0x10, 0, 0x10, 0]);
        for scheme in [0xc0, 0xc1, 0xc2] {
            large[sof + 1] = scheme;
            assert!(matches!(
                source::prepare(
                    &large,
                    ImageFormat::Jpeg,
                    160 * 1024 * 1024,
                    large.len() as u64
                ),
                Err(IoError::LimitExceeded { .. })
            ));
            assert!(
                source::prepare(&large, ImageFormat::Jpeg, MAX_ALLOC, large.len() as u64).is_ok()
            );
        }
    }

    #[test]
    fn jpeg_preflight_rejects_bad_frames_and_truncation() {
        let original = jpeg(1, 1);
        let sof = original
            .windows(2)
            .position(|part| part == [0xff, 0xc0])
            .unwrap();
        for (relative, value) in [(4, 16), (5, 0x10), (7, 0x10), (11, 0x44)] {
            let mut modified = original.clone();
            modified[sof + relative] = value;
            if relative == 4 {
                assert!(matches!(
                    decode_bytes(&modified, ColorIntent::PhotoSrgb),
                    Err(IoError::UnsupportedBitDepth { bits: 16 })
                ));
            } else if matches!(relative, 5 | 7) {
                assert!(matches!(
                    decode_bytes(&modified, ColorIntent::PhotoSrgb),
                    Err(IoError::LimitExceeded { .. })
                ));
            } else {
                // Large, valid sampling factors are fully charged, even in a tiny image.
                let prepared = source::prepare(
                    &modified,
                    ImageFormat::Jpeg,
                    MAX_ALLOC,
                    modified.len() as u64,
                )
                .unwrap();
                assert_eq!(prepared.dimensions, (1, 1));
            }
        }
        for end in 0..original.len() {
            assert!(
                decode_bytes(&original[..end], ColorIntent::PhotoSrgb).is_err(),
                "prefix {end}"
            );
        }
        for payload in [&[][..], &[0][..], &[0, 1][..], &[255, 255, 1][..]] {
            let mut malformed = original[..2].to_vec();
            malformed.extend_from_slice(&[0xff, 0xe1]);
            malformed.extend_from_slice(payload);
            assert!(decode_bytes(&malformed, ColorIntent::PhotoSrgb).is_err());
        }
    }
    #[test]
    fn copy_fallback_does_not_replace_existing_destination() {
        let base = std::env::temp_dir().join(format!(
            "texelith-copy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&base).unwrap();
        let source = base.join("source");
        let dest = base.join("dest");
        fs::write(&source, b"new").unwrap();
        install_new(&source, &dest, true).unwrap();
        assert!(matches!(
            install_new(&source, &dest, true),
            Err(IoError::DestinationExists)
        ));
        assert_eq!(fs::read(&dest).unwrap(), b"new");
        fs::remove_file(source).unwrap();
        fs::remove_file(dest).unwrap();
        fs::remove_dir(base).unwrap();
    }

    #[test]
    fn copy_failure_removes_partial_destination() {
        struct FailingReader(bool);
        impl Read for FailingReader {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("injected copy failure"));
                }
                self.0 = true;
                out[..4].copy_from_slice(b"part");
                Ok(4)
            }
        }
        let path = std::env::temp_dir().join(format!(
            "texelith-copy-failure-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let error = copy_new(FailingReader(false), &path).unwrap_err();
        assert!(
            matches!(error, IoError::Io(error) if error.to_string() == "injected copy failure")
        );
        assert!(!path.exists());
    }
}

// FAZ1_IO_MODULES: T-080
pub mod encoders;

// FAZ1_IO_MODULES: T-077 begin
pub mod export_transactions;
// FAZ1_IO_MODULES: T-077 end

// FAZ1_IO_MODULES: T-085 begin
pub mod project_io;
// FAZ1_IO_MODULES: T-085 end

/// Publish a complete encoded single-map PNG using the existing overwrite transaction.
pub fn publish_png(path: &Path, encoded: &[u8]) -> Result<(), IoError> {
    if !encoded.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(IoError::InvalidPixels);
    }
    write_png(path, encoded, true)
}

/// Cancel publication after encoding/staging if the captured document has changed.
pub fn publish_png_if_current(
    path: &Path,
    encoded: &[u8],
    current: impl Fn() -> bool,
) -> Result<(), IoError> {
    if !encoded.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(IoError::InvalidPixels);
    }
    write_png_checked(path, encoded, true, current)
}
