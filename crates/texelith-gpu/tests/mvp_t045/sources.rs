use std::sync::Arc;
use texelith_core::{Extent, InputInterpretation, NormalConvention, SourceDescriptor};
use texelith_gpu::{GpuError, GpuWorker, TypedSourcePixels, TypedSourceUpload};

#[test]
fn typed_upload_is_owned_sendable_and_worker_install_is_additive() {
    fn send<T: Send>() {}
    send::<TypedSourceUpload>();
    let _: fn(&GpuWorker, TypedSourceUpload) -> Result<(), GpuError> =
        GpuWorker::install_typed_source;
}

#[test]
fn legacy_rgba8_hashes_remain_compatible_and_pixels_are_not_reinterpreted() {
    let extent = Extent::new(2, 1).unwrap();
    let pixels: Arc<[u8]> = Arc::from([0, 128, 255, 7, 9, 10, 11, 255]);
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
    ] {
        let expected = SourceDescriptor::from_rgba8(&pixels, extent, interpretation).unwrap();
        let source = TypedSourcePixels::Rgba8 {
            pixels: pixels.clone(),
            interpretation,
        };
        assert_eq!(source.descriptor(extent).unwrap(), expected);
    }
}

#[test]
fn typed_identity_includes_precision_extent_content_and_normal_convention() {
    let e = Extent::new(2, 1).unwrap();
    let u16_source = TypedSourcePixels::HeightU16(Arc::from([0, 32768]));
    let first = u16_source.descriptor(e).unwrap();
    assert_eq!(first.interpretation, InputInterpretation::LinearHeight);
    assert_eq!(first, u16_source.descriptor(e).unwrap());
    assert_ne!(
        first.content_hash,
        u16_source
            .descriptor(Extent::new(1, 2).unwrap())
            .unwrap()
            .content_hash
    );
    assert_ne!(
        first.content_hash,
        TypedSourcePixels::HeightU16(Arc::from([0, 32769]))
            .descriptor(e)
            .unwrap()
            .content_hash
    );
    // Same byte payload, different channel layout and numeric interpretation.
    let rgb = Arc::<[u8]>::from([0, 0, 0, 128]);
    let rgba = TypedSourcePixels::Rgba8 {
        pixels: rgb,
        interpretation: InputInterpretation::LinearHeight,
    };
    assert_ne!(
        first.content_hash,
        rgba.descriptor(Extent::new(1, 1).unwrap())
            .unwrap()
            .content_hash
    );
    let float = TypedSourcePixels::HeightF32(Arc::from([0.0, 0.5]));
    assert_ne!(
        first.content_hash,
        float.descriptor(e).unwrap().content_hash
    );
    let one_pixel = Extent::new(1, 1).unwrap();
    let same_bytes_rgba = TypedSourcePixels::Rgba8 {
        pixels: Arc::from(0.5f32.to_le_bytes()),
        interpretation: InputInterpretation::LinearHeight,
    };
    let same_bytes_float = TypedSourcePixels::HeightF32(Arc::from([0.5]));
    assert_ne!(
        same_bytes_rgba.descriptor(one_pixel).unwrap().content_hash,
        same_bytes_float.descriptor(one_pixel).unwrap().content_hash
    );
    let normal = |convention| TypedSourcePixels::NormalRgb8 {
        pixels: Arc::from([128, 128, 255, 128, 128, 255]),
        convention,
    };
    let gl = normal(NormalConvention::OpenGl).descriptor(e).unwrap();
    let dx = normal(NormalConvention::DirectX).descriptor(e).unwrap();
    assert_eq!(gl.interpretation, InputInterpretation::LinearNormal);
    assert_ne!(gl.content_hash, dx.content_hash);
}

#[test]
fn float_height_identity_is_deterministic_without_quantizing() {
    let e = Extent::new(2, 1).unwrap();
    let hash = |values| {
        TypedSourcePixels::HeightF32(Arc::from(values))
            .descriptor(e)
            .unwrap()
            .content_hash
    };
    assert_eq!(hash([0.0, 0.5]), hash([0.0, 0.5]));
    assert_ne!(
        hash([0.0, 0.5]),
        hash([0.0, f32::from_bits(0.5f32.to_bits() + 1)])
    );
}

#[test]
fn malformed_or_nonfinite_typed_sources_fail_before_allocation() {
    let e = Extent::new(2, 1).unwrap();
    for source in [
        TypedSourcePixels::HeightU16(Arc::from([1])),
        TypedSourcePixels::HeightF32(Arc::from([0.0, f32::NAN])),
        TypedSourcePixels::HeightF32(Arc::from([f32::INFINITY, 0.0])),
        TypedSourcePixels::NormalRgb8 {
            pixels: Arc::from([128, 128, 255]),
            convention: NormalConvention::OpenGl,
        },
        TypedSourcePixels::Rgba8 {
            pixels: Arc::from([0, 0, 0, 255]),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    ] {
        assert!(source.descriptor(e).is_err());
    }
    let source = TypedSourcePixels::HeightU16(Arc::from([1]));
    for bad in [
        Extent {
            width: 0,
            height: 1,
        },
        Extent {
            width: 4097,
            height: 1,
        },
    ] {
        assert!(source.descriptor(bad).is_err());
    }
}
