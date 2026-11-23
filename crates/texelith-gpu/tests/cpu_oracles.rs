mod support;

use std::mem::{align_of, offset_of, size_of};
use support::{abi::*, oracle::*};
use texelith_core::{Extent, InputInterpretation, MapFormat, MapKind, SourceDescriptor};

fn extent(w: u32, h: u32) -> Extent {
    Extent::new(w, h).unwrap()
}
fn close(a: f64, b: f64) {
    compare(&[a], &[b], 1e-12).unwrap();
}

#[test]
fn common_uniform_is_four_aligned_slots_with_zero_reserved_bytes() {
    assert_eq!(size_of::<NodeUniform>(), 64);
    assert_eq!(align_of::<NodeUniform>(), 16);
    assert_eq!(offset_of!(NodeUniform, dimensions), 0);
    assert_eq!(offset_of!(NodeUniform, parameters), 16);
    assert_eq!(offset_of!(NodeUniform, flags), 32);
    assert_eq!(offset_of!(NodeUniform, reserved), 48);
    let u = NodeUniform::new(extent(17, 11), extent(34, 22));
    assert_eq!(u.dimensions, [17, 11, 34, 22]);
    assert_eq!(&bytemuck::bytes_of(&u)[16..], &[0; 48]);
    assert_eq!(size_of::<BlurWeights>(), 128);
    assert_eq!(align_of::<BlurWeights>(), 16);
    let weights = gaussian(5.0);
    let block = BlurWeights::new(&weights);
    assert_eq!(block.slots[7][3], 0.0);
    for (i, &weight) in weights.iter().enumerate() {
        assert_eq!(block.slots[i / 4][i % 4], weight as f32);
    }
}

#[test]
fn dispatch_ceil_division_and_guard_cover_every_valid_pixel_once() {
    for e in [
        extent(1, 1),
        extent(1, 13),
        extent(13, 1),
        extent(7, 9),
        extent(17, 11),
        extent(256, 256),
        extent(4096, 4096),
    ] {
        let [gx, gy, gz] = workgroups(e).unwrap();
        assert_eq!((gx, gy, gz), (e.width.div_ceil(8), e.height.div_ceil(8), 1));
        assert!(gx * 8 >= e.width && gx * 8 - e.width < 8);
        assert!(gy * 8 >= e.height && gy * 8 - e.height < 8);
        if e.width <= 256 && e.height <= 256 {
            let mut visits = vec![0u8; e.width as usize * e.height as usize];
            for y in 0..gy * 8 {
                for x in 0..gx * 8 {
                    if x < e.width && y < e.height {
                        visits[(y * e.width + x) as usize] += 1;
                    }
                }
            }
            assert!(visits.iter().all(|&n| n == 1));
        }
    }
    for invalid in [
        Extent {
            width: 0,
            height: 1,
        },
        Extent {
            width: 1,
            height: 0,
        },
        Extent {
            width: 4097,
            height: 1,
        },
        Extent {
            width: u32::MAX,
            height: u32::MAX,
        },
    ] {
        assert!(workgroups(invalid).is_err());
    }
}

#[test]
fn readback_padding_strips_sentinels_and_checks_lengths_and_overflow() {
    for bpp in [4, 8] {
        for width in [1, 13, 32, 63, 64, 65, 255, 256] {
            let (tight, padded) = row_layout(width, bpp).unwrap();
            assert_eq!(padded % 256, 0);
            assert!(padded >= tight && padded - tight < 256);
            let mut bytes = vec![0xED; padded as usize * 3];
            for y in 0..3 {
                bytes[y * padded as usize..y * padded as usize + tight as usize].fill(y as u8);
            }
            let stripped = strip_rows(&bytes, width, 3, bpp).unwrap();
            assert_eq!(stripped.len(), tight as usize * 3);
            for y in 0..3 {
                assert!(
                    stripped[y * tight as usize..(y + 1) * tight as usize]
                        .iter()
                        .all(|&v| v == y as u8)
                );
            }
            assert!(strip_rows(&bytes[..bytes.len() - 1], width, 3, bpp).is_err());
        }
    }
    assert!(row_layout(0, 4).is_err());
    assert!(row_layout(u32::MAX, 8).is_err());
    assert!(strip_rows(&[], 1, 0, 4).is_err());
}

#[test]
fn half_decoder_preserves_sign_subnormals_and_specials() {
    close(decode_half(0x3c00), 1.0);
    close(decode_half(0xbc00), -1.0);
    close(decode_half(1), 2.0_f64.powi(-24));
    close(decode_half(0x7bff), 65504.0);
    assert!(decode_half(0x8000).is_sign_negative());
    assert!(decode_half(0x7c00).is_infinite());
    assert!(decode_half(0x7e00).is_nan());
}

#[test]
fn photo_transfer_then_luminance_is_distinct_from_linear_red_and_ignores_alpha() {
    close(srgb_inverse(0.0), 0.0);
    close(srgb_inverse(0.04045), 0.0031308049535603713);
    close(srgb_inverse(0.5), 0.21404114048223255);
    close(srgb_inverse(1.0), 1.0);
    let rgba = [
        255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 64, 128, 128, 128, 0, 10, 10, 10, 255, 11, 11, 11,
        255,
    ];
    let photo = prepare(&rgba, extent(6, 1), InputInterpretation::SrgbPhoto);
    compare(
        &photo.pixels[..4],
        &[0.2126, 0.7152, 0.0722, 0.21586050011389926],
        1e-12,
    )
    .unwrap();
    close(photo.pixels[4], (10.0 / 255.0) / 12.92);
    close(
        photo.pixels[5],
        ((11.0 / 255.0 + 0.055) / 1.055_f64).powf(2.4),
    );
    let linear = prepare(&rgba, extent(6, 1), InputInterpretation::LinearHeight);
    compare(
        &linear.pixels,
        &[1.0, 0.0, 0.0, 128.0 / 255.0, 10.0 / 255.0, 11.0 / 255.0],
        1e-12,
    )
    .unwrap();
    let mut opaque = rgba;
    for p in opaque.as_chunks_mut::<4>().0 {
        p[3] = 255;
    }
    assert_eq!(
        photo.pixels,
        prepare(&opaque, extent(6, 1), InputInterpretation::SrgbPhoto).pixels
    );
    assert_eq!(
        linear.pixels,
        prepare(&opaque, extent(6, 1), InputInterpretation::LinearHeight).pixels
    );
}

#[test]
fn levels_have_analytical_clamp_endpoints_and_inverse_gamma() {
    let input = Height::new(
        extent(7, 1),
        vec![-1.0, 0.25, 0.28125, 0.375, 0.5, 0.75, 2.0],
    );
    let output = levels(&input, 0.25, 0.75, 2.0);
    compare(
        &output.pixels,
        &[0.0, 0.0, 0.25, 0.5, 0.5_f64.sqrt(), 1.0, 1.0],
        1e-12,
    )
    .unwrap();
    let dense = Height::generate(extent(256, 1), |x, _| 0.4 + f64::from(x) * 1e-5);
    let identity = levels(&dense, 0.0, 1.0, 1.0);
    assert_eq!(dense.pixels, identity.pixels);
    assert!(identity.pixels.windows(2).all(|w| w[1] > w[0]));
    assert_eq!(MapKind::Height.format(), MapFormat::R32Float);
    assert_eq!(MapKind::Normal.format(), MapFormat::Rgba16Float);
}

#[test]
fn gaussian_weights_are_normalized_symmetric_capped_and_zero_is_identity() {
    for sigma in [0.0, 0.01, 0.25, 1.0, 2.5, 5.0, 10.0] {
        let weights = gaussian(sigma);
        assert_eq!(
            weights.len(),
            2 * (3.0 * sigma).ceil().min(15.0) as usize + 1
        );
        close(weights.iter().sum(), 1.0);
        assert!(weights.iter().all(|w| *w >= 0.0));
        assert_eq!(weights, weights.iter().rev().copied().collect::<Vec<_>>());
    }
    for fixture in fixtures() {
        assert_eq!(
            blur(&fixture.height, fixture.height.extent, 0.0).pixels,
            fixture.height.pixels
        );
    }
}

#[test]
fn blur_impulse_has_analytical_gaussian_ratio_and_separable_outer_product() {
    let image = Height::generate(extent(9, 9), |x, y| f64::from(x == 4 && y == 4));
    let blurred = blur(&image, image.extent, 1.0);
    let weights = gaussian(1.0);
    close(blurred.pixels.iter().sum(), 1.0);
    close(blurred.at(5, 4) / blurred.at(4, 4), (-0.5_f64).exp());
    close(blurred.at(5, 5) / blurred.at(4, 4), (-1.0_f64).exp());
    for y in 1..8 {
        for x in 1..8 {
            close(
                blurred.at(x, y),
                weights[(x - 1) as usize] * weights[(y - 1) as usize],
            );
        }
    }
    let constant = Height::generate(extent(17, 11), |_, _| 0.375);
    compare(
        &blur(&constant, constant.extent, 5.0).pixels,
        &constant.pixels,
        1e-12,
    )
    .unwrap();
}

#[test]
fn blur_clamps_instead_of_wrapping_and_scales_sigma_per_axis() {
    let corner = Height::new(
        extent(9, 1),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let blurred = blur(&corner, corner.extent, 1.0);
    let w = gaussian(1.0);
    close(blurred.at(0, 0), w[..=3].iter().sum());
    close(blurred.at(8, 0), 0.0);
    let image = Height::generate(extent(17, 11), |x, y| f64::from(x == 8 && y == 5));
    let result = blur(&image, extent(34, 44), 2.0);
    let expected = blur_axis(&blur_axis(&image, 1.0, false), 0.5, true);
    assert_eq!(result.pixels, expected.pixels);
    close(result.at(9, 5) / result.at(8, 5), (-0.5_f64).exp());
    close(result.at(8, 6) / result.at(8, 5), (-2.0_f64).exp());
}

#[test]
fn normals_are_signed_unit_vectors_with_opengl_positive_y_and_clamped_edges() {
    let image = Height::generate(extent(17, 11), |x, y| {
        0.2 + 0.25 * (f64::from(x) + 0.5) / 17.0 + 0.125 * (f64::from(y) + 0.5) / 11.0
    });
    let normals = normal(&image, 2.0);
    let length = (0.5_f64.powi(2) + 0.25_f64.powi(2) + 1.0).sqrt();
    compare(
        &normals[5 * 17 + 8],
        &[-0.5 / length, 0.25 / length, 1.0 / length, 1.0],
        1e-12,
    )
    .unwrap();
    let edge = normals[5 * 17];
    close(edge[0] / edge[2], -0.25);
    close(edge[1] / edge[2], 0.25);
    assert!(
        normals
            .iter()
            .all(|n| n[0] < 0.0 && n[1] > 0.0 && n[2] > 0.0 && n[3] == 1.0)
    );
    for n in normals {
        close(n[..3].iter().map(|c| c * c).sum::<f64>().sqrt(), 1.0);
    }
    for fixture in fixtures() {
        assert!(
            normal(&fixture.height, 0.0)
                .iter()
                .all(|n| *n == [0.0, 0.0, 1.0, 1.0])
        );
        let normals = normal(&fixture.height, 10.0);
        if fixture.height.extent.width == 1 {
            assert!(normals.iter().all(|n| n[0] == 0.0));
        }
        if fixture.height.extent.height == 1 {
            assert!(normals.iter().all(|n| n[1] == 0.0));
        }
    }
}

#[test]
fn area_downsample_has_fractional_npot_expectation_and_converts_before_averaging() {
    let row = Height::new(extent(5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    compare(&downsample(&row, extent(2, 1)).pixels, &[0.8, 3.2], 1e-12).unwrap();
    let col = Height::new(extent(1, 5), row.pixels.clone());
    compare(&downsample(&col, extent(1, 2)).pixels, &[0.8, 3.2], 1e-12).unwrap();
    let photo = prepare(
        &[0, 0, 0, 255, 255, 255, 255, 0],
        extent(2, 1),
        InputInterpretation::SrgbPhoto,
    );
    close(downsample(&photo, extent(1, 1)).pixels[0], 0.5);
    assert!((srgb_inverse(0.5) - 0.5).abs() > 0.25);
    for fixture in fixtures() {
        assert_eq!(
            downsample(&fixture.height, fixture.height.extent).pixels,
            fixture.height.pixels
        );
    }
}

#[test]
fn proxy_extent_preserves_aspect_to_one_texel_and_uv_slopes_match_full() {
    for source in [
        extent(4096, 2048),
        extent(2048, 4096),
        extent(4096, 1),
        extent(1, 4096),
        extent(2049, 1001),
        extent(17, 11),
        extent(1, 1),
    ] {
        let proxy = proxy_extent(source);
        assert!(proxy.width >= 1 && proxy.height >= 1 && proxy.width.max(proxy.height) <= 1024);
        let scale = (1024.0 / f64::from(source.width.max(source.height))).min(1.0);
        assert!((f64::from(proxy.width) - f64::from(source.width) * scale).abs() < 1.0);
        assert!((f64::from(proxy.height) - f64::from(source.height) * scale).abs() < 1.0);
    }
    // A 4:1 analytic fixture is small; compare away from clamped boundaries.
    let full = Height::generate(extent(64, 32), |x, y| {
        0.1 + 0.25 * (f64::from(x) + 0.5) / 64.0 + 0.125 * (f64::from(y) + 0.5) / 32.0
    });
    let proxy = downsample(&full, extent(16, 8));
    let full_normal = normal(&blur(&full, full.extent, 1.0), 3.0);
    let proxy_normal = normal(&blur(&proxy, full.extent, 1.0), 3.0);
    compare(&full_normal[16 * 64 + 32], &proxy_normal[4 * 16 + 8], 1e-12).unwrap();
}

#[test]
fn display_codes_are_opaque_data_codes_without_srgb_transfer_or_y_flip() {
    let image = Height::new(extent(5, 1), vec![-0.1, 0.0, 0.5, 1.0, 1.1]);
    assert_eq!(
        display_height(&image),
        vec![
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [128, 128, 128, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ]
    );
    assert_eq!(
        display_normal(&[
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ]),
        vec![
            [0, 128, 128, 255],
            [128, 255, 128, 255],
            [128, 128, 255, 255]
        ]
    );
}

#[test]
fn symmetric_tileable_height_edges_remain_equal_with_clamped_blur() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.name == "tileable_symmetric")
        .unwrap();
    close(wrapped_edge_score(&fixture.height), 0.0);
    let output = blur(
        &levels(&fixture.height, 0.1, 0.9, 1.5),
        fixture.height.extent,
        1.0,
    );
    assert!(wrapped_edge_score(&output) < 1e-12);
    let non_tileable = Height::generate(extent(17, 11), |x, _| f64::from(x) / 16.0);
    close(wrapped_edge_score(&non_tileable), 1.0);
}

#[test]
fn references_repeat_exactly_and_source_hash_uses_real_core_contract() {
    let extent = extent(17, 11);
    let rgba: Vec<u8> = (0..extent.width * extent.height)
        .flat_map(|i| [(i % 251) as u8, 10, 128, 255])
        .collect();
    let a = SourceDescriptor::from_rgba8(&rgba, extent, InputInterpretation::SrgbPhoto).unwrap();
    let b = SourceDescriptor::from_rgba8(&rgba, extent, InputInterpretation::SrgbPhoto).unwrap();
    assert_eq!(a.content_hash, b.content_hash);
    let chain = || {
        normal(
            &blur(
                &levels(&prepare(&rgba, extent, a.interpretation), 0.1, 0.9, 1.25),
                extent,
                1.0,
            ),
            2.0,
        )
    };
    let first = chain();
    let second = chain();
    assert_eq!(
        first
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn tolerance_comparison_rejects_nan_infinity_length_mismatch_and_bad_channels() {
    assert!(compare(&[f64::NAN], &[0.0], HEIGHT_TOLERANCE).is_err());
    assert!(compare(&[f64::INFINITY], &[0.0], HEIGHT_TOLERANCE).is_err());
    assert!(compare(&[], &[0.0], HEIGHT_TOLERANCE).is_err());
    assert!(compare(&[HEIGHT_TOLERANCE * 1.01], &[0.0], HEIGHT_TOLERANCE).is_err());
    assert!(compare(&[NORMAL_TOLERANCE * 1.01], &[0.0], NORMAL_TOLERANCE).is_err());
}
