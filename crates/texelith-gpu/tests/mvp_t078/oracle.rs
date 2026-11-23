use super::{fixtures::*, reference::*};
use texelith_core::*;

pub fn scalar(port: OutputPort, w: u32, h: u32, values: &[f64]) -> Source {
    assert_eq!(values.len(), (w * h) as usize);
    Source {
        reference: reference(port),
        extent: extent(w, h),
        pixels: values.iter().map(|v| [*v, 0.0, 0.0, 1.0]).collect(),
        boundary: Boundary::Clamp,
        normal_convention: NormalConvention::OpenGl,
    }
}

#[test]
fn oracle_has_closed_form_area_bilinear_boundary_and_mixed_axis_anchors() {
    let source = scalar(OutputPort::Height, 3, 1, &[0.0, 3.0, 12.0]);
    let mut p = plan(source.extent, vec![]);
    p.size = SizePolicy::Resize {
        width: 2,
        height: 1,
    };
    assert!((sample(&source, &p, 0, 0)[0] - 1.0).abs() < 1e-12);
    assert!((sample(&source, &p, 1, 0)[0] - 9.0).abs() < 1e-12);
    let source = scalar(OutputPort::Height, 2, 1, &[0.0, 1.0]);
    p.source_extent = source.extent;
    p.size = SizePolicy::Resize {
        width: 4,
        height: 1,
    };
    let actual: Vec<_> = (0..4).map(|x| sample(&source, &p, x, 0)[0]).collect();
    assert_eq!(actual, [0.0, 0.25, 0.75, 1.0]);
    let mut wrapped = source;
    wrapped.boundary = Boundary::Repeat;
    assert_eq!(sample(&wrapped, &p, 0, 0)[0], 0.25);
    assert_eq!(sample(&wrapped, &p, 3, 0)[0], 0.75);
    let source = scalar(OutputPort::Height, 3, 2, &[0.0, 3.0, 12.0, 6.0, 9.0, 18.0]);
    p.source_extent = source.extent;
    p.size = SizePolicy::Resize {
        width: 2,
        height: 4,
    };
    assert!((sample(&source, &p, 1, 1)[0] - 10.5).abs() < 1e-12);
}

#[test]
fn oracle_transfer_quantization_and_signed_float_anchors_are_independent() {
    assert!((srgb(0.003) - 0.03876).abs() < 1e-12);
    assert!((srgb(0.18) - 0.46135612950044164).abs() < 1e-12);
    assert_eq!(
        quantize(-1.0, 8, [0.0, 1.0], IntegerDither::None, 0, 0),
        0.0
    );
    assert_eq!(
        quantize(2.0, 16, [0.0, 1.0], IntegerDither::None, 0, 0),
        1.0
    );
    assert_eq!(
        quantize(0.5, 8, [0.0, 1.0], IntegerDither::None, 0, 0),
        128.0 / 255.0
    );
    // Clamp retains the normalized domain; it does not stretch [0.2,0.8] to [0,1].
    assert_eq!(
        quantize(-1.0, 8, [0.2, 0.8], IntegerDither::None, 0, 0),
        51.0 / 255.0
    );
    assert_eq!(
        quantize(2.0, 8, [0.2, 0.8], IntegerDither::None, 0, 0),
        204.0 / 255.0
    );
    let source = scalar(OutputPort::Height, 3, 1, &[0.0, 0.5, 1.0]);
    let mut p = plan(
        source.extent,
        vec![output(
            "displacement",
            ExportFormat::Exr32,
            vec![map(OutputPort::Height, Component::R)],
        )],
    );
    p.displacement.scale = 4.0;
    assert_eq!(
        evaluate(&p, 0, &[source])
            .iter()
            .map(|v| v[0])
            .collect::<Vec<_>>(),
        [-2.0, 0.0, 2.0]
    );
}

#[test]
fn oracle_dither_is_balanced_fixed_and_anchored_to_output_coordinates() {
    let mut sum = 0.0;
    let mut lows = 0;
    for y in 0..4 {
        for x in 0..4 {
            let v = quantize(0.5, 8, [0.0, 1.0], IntegerDither::Ordered4x4, x, y);
            assert!(v == 127.0 / 255.0 || v == 128.0 / 255.0);
            lows += usize::from(v == 127.0 / 255.0);
            sum += v;
            assert_eq!(
                v,
                quantize(0.5, 8, [0.0, 1.0], IntegerDither::Ordered4x4, x + 4, y + 4)
            );
        }
    }
    assert_eq!(lows, 8);
    assert_eq!(sum / 16.0, 0.5);
}

#[test]
fn explicit_engine_pack_fixtures_validate_without_preset_lookup() {
    let p = plan(extent(17, 13), packed_outputs());
    validate_export(&p).unwrap();
    let sources = [
        scalar(OutputPort::Ao, 17, 13, &vec![0.2; 221]),
        scalar(OutputPort::Roughness, 17, 13, &vec![0.6; 221]),
        scalar(OutputPort::Metallic, 17, 13, &vec![0.8; 221]),
    ];
    let expected = [
        [51, 153, 204, 255],
        [204, 51, 255, 102],
        [51, 153, 204, 255],
        [204, 0, 0, 102],
    ];
    for (i, codes) in expected.iter().enumerate() {
        assert_eq!(p.outputs[i].semantics().color_tag, ExportColorTag::Linear);
        for pixel in evaluate(&p, i, &sources) {
            for c in 0..4 {
                assert!((pixel[c] - f64::from(codes[c]) / 255.0).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn core_size_contract_preserves_npot_for_bc5() {
    let mut p = plan(
        extent(17, 13),
        vec![output(
            "normal",
            ExportFormat::DdsBc5,
            vec![
                map(OutputPort::Normal, Component::R),
                map(OutputPort::Normal, Component::G),
            ],
        )],
    );
    assert!(validate_export(&p).is_ok());
    for size in [
        SizePolicy::Resize {
            width: 20,
            height: 16,
        },
        SizePolicy::Pad {
            width: 32,
            height: 16,
        },
    ] {
        p.size = size;
        validate_export(&p).unwrap();
    }
    p.outputs[0] = output(
        "height",
        ExportFormat::Png16,
        vec![map(OutputPort::Height, Component::R)],
    );
    p.size = SizePolicy::Keep;
    validate_export(&p).unwrap();
    assert_eq!(destination(&p), extent(17, 13));
    for size in [
        SizePolicy::Resize {
            width: 0,
            height: 16,
        },
        SizePolicy::Resize {
            width: 4097,
            height: 16,
        },
        SizePolicy::Pad {
            width: 16,
            height: 16,
        },
        SizePolicy::Pad {
            width: 32,
            height: 0,
        },
    ] {
        p.size = size;
        assert!(validate_export(&p).is_err(), "accepted {size:?}");
    }
}
