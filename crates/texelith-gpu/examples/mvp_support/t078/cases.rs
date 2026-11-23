//! Executable copies of the tests-first behavioral cases, kept assertion-for-assertion.
use super::{fixtures::*, gpu::vector, oracle::scalar, reference};
use texelith_core::*;

fn compare(actual: &[[f64; 4]], expected: &[[f64; 4]], tolerance: f64) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        for c in 0..4 {
            assert!(
                a[c].is_finite() && (a[c] - e[c]).abs() <= tolerance,
                "pixel {i} channel {c}: {} != {} (tolerance {tolerance})",
                a[c],
                e[c]
            );
        }
    }
}

fn code(value: f64, bits: u8) -> u32 {
    let max = f64::from((1u32 << bits) - 1);
    assert!(value.is_finite() && (0.0..=1.0).contains(&value));
    assert!(
        (value * max - (value * max).round()).abs() < 0.004,
        "value {value} is not on the {bits}-bit integer grid"
    );
    (value * max).round() as u32
}

pub(super) fn orm_hdrp_mask_gltf_and_urp_pack_exact_distinct_channels_and_constants(
    gpu: &super::gpu::Gpu,
) {
    {
        let p = plan(extent(17, 13), packed_outputs());
        let sources = [
            scalar(OutputPort::Metallic, 17, 13, &vec![0.8; 221]),
            scalar(OutputPort::Ao, 17, 13, &vec![0.2; 221]),
            scalar(OutputPort::Roughness, 17, 13, &vec![0.6; 221]),
        ];
        let expected = [
            [51, 153, 204, 255],
            [204, 51, 255, 102],
            [51, 153, 204, 255],
            [204, 0, 0, 102],
        ];
        for (i, expected) in expected.iter().enumerate() {
            for pixel in gpu.run(&p, i, &sources) {
                assert_eq!(
                    pixel.map(|v| code(v, 8)),
                    *expected,
                    "{}",
                    p.outputs[i].name
                );
            }
        }
    }
}

pub(super) fn arbitrary_swizzles_use_full_output_ref_and_apply_invert_before_transfer(
    gpu: &super::gpu::Gpu,
) {
    {
        let mut red = map(OutputPort::Albedo, Component::R);
        red.invert = true;
        red.transfer = Transfer::Srgb;
        let mut other = scalar(OutputPort::Roughness, 1, 1, &[0.25]);
        other.reference.node = NodeId(99);
        let mut other_channel = map(OutputPort::Roughness, Component::R);
        other_channel.source = ChannelSource::Map(other.reference, Component::R);
        let p = plan(
            extent(1, 1),
            vec![output(
                "swizzle",
                ExportFormat::Png16,
                vec![
                    map(OutputPort::Albedo, Component::B),
                    red,
                    other_channel,
                    constant(0.75),
                ],
            )],
        );
        let sources = [
            vector(OutputPort::Albedo, 1, 1, vec![[0.25, 0.5, 0.75, 0.0]]),
            scalar(OutputPort::Roughness, 1, 1, &[0.875]),
            other,
        ];
        assert_eq!(p.outputs[0].semantics().color_tag, ExportColorTag::Raw);
        assert_eq!(
            p.outputs[0].semantics().warnings,
            [ExportWarning::MixedTransfer]
        );
        compare(
            &gpu.run(&p, 0, &sources),
            &reference::evaluate(&p, 0, &sources),
            1.01 / 65535.0,
        );
    }
}

pub(super) fn albedo_is_encoded_once_after_linear_resampling_data_and_alpha_stay_linear(
    gpu: &super::gpu::Gpu,
) {
    {
        let color = vector(
            OutputPort::Albedo,
            2,
            1,
            vec![[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.25]],
        );
        let channels = [Component::R, Component::G, Component::B].map(|c| {
            let mut channel = map(OutputPort::Albedo, c);
            channel.transfer = Transfer::Srgb;
            channel
        });
        let mut p = plan(
            extent(2, 1),
            vec![output(
                "albedo",
                ExportFormat::Png8,
                channels
                    .into_iter()
                    .chain([map(OutputPort::Albedo, Component::A)])
                    .collect(),
            )],
        );
        p.size = SizePolicy::Resize {
            width: 1,
            height: 1,
        };
        assert_eq!(p.outputs[0].semantics().color_tag, ExportColorTag::Srgb);
        assert_eq!(
            gpu.run(&p, 0, &[color])[0].map(|v| code(v, 8)),
            [188, 188, 188, 255]
        );
        let mut channels = vec![
            map(OutputPort::Albedo, Component::R),
            map(OutputPort::Ao, Component::R),
            map(OutputPort::Roughness, Component::R),
            map(OutputPort::Albedo, Component::A),
        ];
        channels[0].transfer = Transfer::Srgb;
        channels[3].invert = true;
        p = plan(
            extent(1, 1),
            vec![output("mixed", ExportFormat::Png16, channels)],
        );
        let sources = [
            vector(OutputPort::Albedo, 1, 1, vec![[0.125, 0.0, 0.0, 0.5]]),
            scalar(OutputPort::Ao, 1, 1, &[0.125]),
            scalar(OutputPort::Roughness, 1, 1, &[0.75]),
        ];
        let actual = gpu.run(&p, 0, &sources);
        compare(
            &actual,
            &reference::evaluate(&p, 0, &sources),
            1.01 / 65535.0,
        );
        assert!(actual[0][0] > 0.38 && actual[0][1] < 0.126);
        assert_eq!(
            actual[0][3], 0.0,
            "ignored source alpha is one, then inverted"
        );
    }
}

pub(super) fn srgb_knee_and_linear_data_are_numerically_independent(gpu: &super::gpu::Gpu) {
    {
        let values = [0.0, 3.0 / 1024.0, 4.0 / 1024.0, 0.125, 0.5, 1.0];
        let color = vector(
            OutputPort::Albedo,
            6,
            1,
            values.map(|v| [v, v, v, 0.0]).to_vec(),
        );
        let mut channels = vec![
            map(OutputPort::Albedo, Component::R),
            map(OutputPort::Albedo, Component::G),
            map(OutputPort::Albedo, Component::B),
        ];
        for channel in &mut channels {
            channel.transfer = Transfer::Srgb;
        }
        let p = plan(
            extent(6, 1),
            vec![output("color", ExportFormat::Png16, channels)],
        );
        compare(
            &gpu.run(&p, 0, std::slice::from_ref(&color)),
            &reference::evaluate(&p, 0, &[color]),
            1.01 / 65535.0,
        );
    }
}

pub(super) fn normals_convert_y_once_select_components_after_conversion_and_renormalize(
    gpu: &super::gpu::Gpu,
) {
    {
        // Distinct unit vectors whose average is not a unit vector.
        let source = vector(
            OutputPort::Normal,
            2,
            1,
            vec![[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        );
        for from in [NormalConvention::OpenGl, NormalConvention::DirectX] {
            for to in [NormalConvention::OpenGl, NormalConvention::DirectX] {
                let mut source = source.clone();
                source.normal_convention = from;
                let mut p = plan(
                    extent(2, 1),
                    vec![output(
                        "normal",
                        ExportFormat::Png16,
                        [Component::R, Component::G, Component::B]
                            .map(|c| map(OutputPort::Normal, c))
                            .to_vec(),
                    )],
                );
                p.profile.normal_convention = to;
                p.size = SizePolicy::Resize {
                    width: 1,
                    height: 1,
                };
                let actual = gpu.run(&p, 0, std::slice::from_ref(&source));
                compare(
                    &actual,
                    &reference::evaluate(&p, 0, &[source.clone()]),
                    1.01 / 65535.0,
                );
                let n = actual[0][..3]
                    .iter()
                    .map(|v| 2.0 * v - 1.0)
                    .collect::<Vec<_>>();
                assert!((n.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() < 2e-3);
                assert!((n[1].abs() - 0.5f64.sqrt()).abs() < 2e-3);
                assert_eq!(n[1] > 0.0, from == to);
                p.outputs[0].channels = vec![
                    map(OutputPort::Normal, Component::G),
                    map(OutputPort::Normal, Component::B),
                    map(OutputPort::Normal, Component::R),
                ];
                compare(
                    &gpu.run(&p, 0, std::slice::from_ref(&source)),
                    &reference::evaluate(&p, 0, &[source]),
                    1.01 / 65535.0,
                );
            }
        }
    }
}

pub(super) fn png16_preserves_adjacent_height_codes_that_png8_cannot_represent(
    gpu: &super::gpu::Gpu,
) {
    {
        let values: Vec<_> = (32750..32782).map(|v| f64::from(v) / 65535.0).collect();
        let sources = [scalar(OutputPort::Height, 32, 1, &values)];
        let mut p = plan(
            extent(32, 1),
            vec![output(
                "height",
                ExportFormat::Png16,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        let actual = gpu.run(&p, 0, &sources);
        let codes: Vec<_> = actual.iter().map(|v| code(v[0], 16)).collect();
        assert_eq!(codes, (32750..32782).collect::<Vec<_>>());
        p.outputs[0].format = ExportFormat::Png8;
        let actual = gpu.run(&p, 0, &sources);
        assert!(
            actual
                .windows(2)
                .any(|pair| code(pair[0][0], 8) == code(pair[1][0], 8))
        );
    }
}

pub(super) fn explicit_integer_clamp_does_not_remap_and_float_displacement_is_never_clamped(
    gpu: &super::gpu::Gpu,
) {
    {
        let sources = [scalar(
            OutputPort::Height,
            5,
            1,
            &[-0.25, 0.0, 0.5, 1.0, 1.25],
        )];
        let mut p = plan(
            extent(5, 1),
            vec![output(
                "height",
                ExportFormat::Png16,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        p.displacement = DisplacementSettings {
            midlevel: 0.5,
            scale: 4.0,
            units: DisplacementUnits::Centimeters,
        };
        p.outputs[0].quantization = Some(IntegerQuantization {
            clamp_min: 0.2,
            clamp_max: 0.8,
            dither: IntegerDither::None,
        });
        compare(
            &gpu.run(&p, 0, &sources),
            &reference::evaluate(&p, 0, &sources),
            1.01 / 65535.0,
        );
        p.outputs[0].format = ExportFormat::Exr32;
        p.outputs[0].quantization = None;
        let actual = gpu.run(&p, 0, &sources);
        compare(&actual, &reference::evaluate(&p, 0, &sources), 2e-5);
        assert_eq!(
            actual.iter().map(|p| p[0]).collect::<Vec<_>>(),
            [-3.0, -2.0, 0.0, 2.0, 3.0]
        );
        // Values separated by <half-float ULP must survive float export.
        let sources = [scalar(
            OutputPort::Height,
            5,
            1,
            &[0.500001, 0.500002, 0.500003, 0.500004, 0.500005],
        )];
        let actual = gpu.run(&p, 0, &sources);
        compare(&actual, &reference::evaluate(&p, 0, &sources), 2.5e-7);
        assert!(actual.windows(2).all(|pair| pair[0][0] < pair[1][0]));
    }
}

pub(super) fn ordered_dither_is_balanced_repeatable_and_integer_only(gpu: &super::gpu::Gpu) {
    {
        for format in [ExportFormat::Png8, ExportFormat::Png16] {
            let mut p = plan(
                extent(8, 8),
                vec![output("dither", format, vec![constant(0.5)])],
            );
            p.outputs[0].quantization = Some(IntegerQuantization {
                dither: IntegerDither::Ordered4x4,
                ..Default::default()
            });
            let a = gpu.run(&p, 0, &[]);
            let b = gpu.run(&p, 0, &[]);
            assert_eq!(a, b);
            let bits = format.bit_depth();
            let lower = ((1u32 << bits) - 1) / 2;
            let codes: Vec<_> = a.iter().map(|v| code(v[0], bits)).collect();
            assert_eq!(codes.iter().filter(|v| **v == lower).count(), 32);
            assert_eq!(codes.iter().filter(|v| **v == lower + 1).count(), 32);
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(codes[y * 8 + x], codes[(y % 4) * 8 + x % 4]);
                }
            }
            compare(&a, &reference::evaluate(&p, 0, &[]), 1e-7);
        }
    }
}

pub(super) fn area_bilinear_and_mixed_axis_resize_match_oracle_on_npot_singletons_and_impulses(
    gpu: &super::gpu::Gpu,
) {
    {
        for (w, h, dw, dh) in [
            (17, 13, 7, 5),
            (7, 5, 17, 13),
            (17, 5, 7, 13),
            (1, 17, 1, 7),
            (17, 1, 31, 1),
            (1, 1, 7, 5),
            (257, 193, 129, 97),
        ] {
            for boundary in [Boundary::Repeat, Boundary::Clamp] {
                for impulse in [false, true] {
                    let values: Vec<_> = (0..w * h)
                        .map(|i| {
                            if impulse {
                                f64::from(u8::from(i == 0))
                            } else {
                                f64::from((i * 37 + i / w * 11) % 1024) / 1024.0
                            }
                        })
                        .collect();
                    let mut source = scalar(OutputPort::Height, w, h, &values);
                    source.boundary = boundary;
                    let mut p = plan(
                        extent(w, h),
                        vec![output(
                            "resize",
                            ExportFormat::Png16,
                            vec![map(OutputPort::Height, Component::R)],
                        )],
                    );
                    p.size = SizePolicy::Resize {
                        width: dw,
                        height: dh,
                    };
                    compare(
                        &gpu.run(&p, 0, std::slice::from_ref(&source)),
                        &reference::evaluate(&p, 0, &[source]),
                        1.01 / 65535.0,
                    );
                }
            }
        }
    }
}

pub(super) fn keep_is_lossless_in_size_and_pad_is_top_left_repeat_or_clamp_with_uv_metadata(
    gpu: &super::gpu::Gpu,
) {
    {
        let values: Vec<_> = (0..17 * 13).map(|i| f64::from(i % 256) / 255.0).collect();
        let sources = [scalar(OutputPort::Height, 17, 13, &values)];
        let mut p = plan(
            extent(17, 13),
            vec![output(
                "height",
                ExportFormat::Png8,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        let kept = gpu.run(&p, 0, &sources);
        assert_eq!(kept.len(), 17 * 13);
        compare(&kept, &reference::evaluate(&p, 0, &sources), 1e-7);
        p.size = SizePolicy::Pad {
            width: 32,
            height: 16,
        };
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            p.padding = boundary;
            let actual = gpu.run(&p, 0, &sources);
            compare(&actual, &reference::evaluate(&p, 0, &sources), 1e-7);
            for y in 0..13 {
                for x in 0..17 {
                    assert_eq!(actual[y * 32 + x], kept[y * 17 + x]);
                }
            }
            assert_eq!(
                code(actual[17][0], 8),
                if boundary == Boundary::Repeat { 0 } else { 16 }
            );
        }
    }
}

pub(super) fn bc5_preparation_requires_four_multiple_but_does_not_force_power_of_two(
    gpu: &super::gpu::Gpu,
) {
    {
        let sources = [vector(
            OutputPort::Normal,
            17,
            13,
            vec![[0.0, 0.0, 1.0, 1.0]; 17 * 13],
        )];
        let mut p = plan(
            extent(17, 13),
            vec![output(
                "bc5",
                ExportFormat::DdsBc5,
                vec![
                    map(OutputPort::Normal, Component::R),
                    map(OutputPort::Normal, Component::G),
                ],
            )],
        );
        p.size = SizePolicy::Resize {
            width: 20,
            height: 16,
        };
        let actual = gpu.run(&p, 0, &sources);
        assert_eq!(actual.len(), 20 * 16);
        compare(&actual, &reference::evaluate(&p, 0, &sources), 1.01 / 255.0);
    }
}

pub(super) fn same_full_resolution_input_and_parameters_produce_identical_content_hashes(
    gpu: &super::gpu::Gpu,
) {
    {
        let values: Vec<_> = (0..17 * 13)
            .map(|i| f64::from((i * 53) % 1024) / 1024.0)
            .collect();
        let sources = [scalar(OutputPort::Height, 17, 13, &values)];
        let p = plan(
            extent(17, 13),
            vec![output(
                "height",
                ExportFormat::Png16,
                vec![map(OutputPort::Height, Component::R)],
            )],
        );
        let hash = |pixels: &[[f64; 4]]| {
            let bytes: Vec<_> = pixels
                .iter()
                .flat_map(|p| p.iter().flat_map(|v| (*v as f32).to_le_bytes()))
                .collect();
            SourceDescriptor::from_rgba8(
                &bytes,
                extent(17 * 4, 13),
                InputInterpretation::LinearHeight,
            )
            .unwrap()
            .content_hash
        };
        let a = gpu.run(&p, 0, &sources);
        let b = gpu.run(&p, 0, &sources);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
        let mut changed = p;
        changed.outputs[0].channels[0].invert = true;
        assert_ne!(hash(&a), hash(&gpu.run(&changed, 0, &sources)));
    }
}

pub(super) fn large_npot_resize_preserves_footprint_precision_on_both_axes(gpu: &super::gpu::Gpu) {
    {
        for (w, h, dw, dh) in [
            (4093, 1, 2047, 1),
            (1, 4093, 1, 2047),
            (4096, 1, 4095, 1),
            (1, 4096, 1, 4095),
            (4093, 193, 2047, 97),
            (193, 4093, 97, 2047),
        ] {
            let values: Vec<_> = (0..w * h)
                .map(|i| f64::from(((i % w) * 37 + (i / w) * 113 + 329) % 1024) / 1024.0)
                .collect();
            let source = scalar(OutputPort::Height, w, h, &values);
            for format in [ExportFormat::Png16, ExportFormat::Exr32] {
                let mut p = plan(
                    extent(w, h),
                    vec![output(
                        "large_npot",
                        format,
                        vec![map(OutputPort::Height, Component::R)],
                    )],
                );
                p.size = SizePolicy::Resize {
                    width: dw,
                    height: dh,
                };
                compare(
                    &gpu.run(&p, 0, std::slice::from_ref(&source)),
                    &reference::evaluate(&p, 0, std::slice::from_ref(&source)),
                    if format == ExportFormat::Exr32 {
                        2e-7
                    } else {
                        1.01 / 65535.0
                    },
                );
            }
        }
    }
}
