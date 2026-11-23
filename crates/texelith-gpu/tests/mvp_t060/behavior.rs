use super::{
    node::{self, ResultData},
    oracle,
    support::{Gpu, with_gpu},
};
use texelith_core::InputInterpretation;
use texelith_gpu::TypedSourcePixels;

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(actual.is_finite(), "nonfinite {label}: {actual}");
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual} expected={expected} tolerance={tolerance}"
    );
}

fn check(actual: &ResultData, expected: &oracle::Diagnostic) {
    assert_eq!(actual.heatmap.len(), expected.heatmap.len());
    for (i, (&a, &b)) in actual.heatmap.iter().zip(&expected.heatmap).enumerate() {
        close(a, b, 2e-5, &format!("heatmap[{i}]"));
        assert!(a >= 0.0);
    }
    close(
        actual.boundary_rms,
        expected.boundary_rms,
        2e-5,
        "boundary RMS",
    );
    close(
        actual.interior_rms,
        expected.interior_rms,
        2e-5,
        "interior RMS",
    );
    close(
        actual.ratio,
        expected.ratio,
        2e-5 * expected.ratio.max(1.0),
        "RMS ratio",
    );
}

fn material(
    gpu: &Gpu,
    w: usize,
    h: usize,
    pixels: TypedSourcePixels,
    levels: bool,
) -> (ResultData, Vec<[f64; 3]>) {
    let (worker, bundle, kind) = gpu.material(w, h, pixels, levels);
    let input = bundle.outputs[&kind].texture();
    let actual_pixels = gpu.pixels(input);
    let result = node::evaluate(gpu, input);
    check(&result, &oracle::inspect(w, h, &actual_pixels));
    drop(bundle);
    worker.join().unwrap();
    (result, actual_pixels)
}

fn scalar(gpu: &Gpu, w: usize, h: usize, values: &[f32]) -> ResultData {
    let (result, pixels) = material(
        gpu,
        w,
        h,
        TypedSourcePixels::HeightF32(values.to_vec().into()),
        false,
    );
    assert_eq!(
        pixels,
        oracle::scalar(values),
        "numeric height acquired a transfer/quantization"
    );
    check(&result, &oracle::inspect(w, h, &oracle::scalar(values)));
    result
}

#[test]
fn raw_unprocessed_ramps_locate_the_wrap_seam_and_match_hand_calculated_rms_ratio() {
    with_gpu(|gpu| {
        for axis in 0..2 {
            let (w, h) = (32, 24);
            let n = if axis == 0 { w } else { h };
            let values = oracle::image(w, h, |x, y| {
                (if axis == 0 { x } else { y }) as f32 / n as f32
            });
            let result = scalar(gpu, w, h, &values);
            close(
                result.ratio,
                (((n - 1).pow(2) + 7) as f64 / 8.0).sqrt(),
                2e-5,
                "literal ramp ratio",
            );
            for y in 0..h {
                for x in 0..w {
                    let at_wrap = if axis == 0 { x == w - 1 } else { y == h - 1 };
                    close(
                        result.heatmap[y * w + x],
                        if at_wrap {
                            (n - 1) as f64 / n as f64
                        } else {
                            1.0 / n as f64
                        },
                        2e-5,
                        "seam location",
                    );
                }
            }
        }
    });
}

#[test]
fn synthetic_step_has_high_absolute_score_at_both_the_interior_step_and_wrapped_edge() {
    with_gpu(|gpu| {
        for axis in 0..2 {
            let values = oracle::image(32, 32, |x, y| {
                if (if axis == 0 { x } else { y }) >= 16 {
                    1.0
                } else {
                    0.0
                }
            });
            let result = scalar(gpu, 32, 32, &values);
            close(result.boundary_rms, 0.25, 2e-5, "literal step boundary");
            close(result.interior_rms, 0.25, 2e-5, "literal step interior");
            close(result.ratio, 1.0, 2e-5, "step ratio");
            for y in 0..32 {
                for x in 0..32 {
                    let position = if axis == 0 { x } else { y };
                    assert_eq!(
                        result.heatmap[y * 32 + x],
                        if position == 15 || position == 31 {
                            1.0
                        } else {
                            0.0
                        }
                    );
                }
            }
        }
    });
}

#[test]
fn exactly_four_boundary_pixels_per_side_and_equal_width_interior_bands_are_used() {
    with_gpu(|gpu| {
        for axis in 0..2 {
            // Forward edges are at p-1 and p. These positions distinguish widths
            // three, four, five and whole-image reductions on BOTH boundaries.
            for (position, boundary_edges, interior_edges) in [
                (3, 2.0_f64, 0.0_f64),
                (4, 1.0, 0.0),
                (5, 0.0, 0.0),
                (12, 0.0, 1.0),
                (13, 0.0, 2.0),
                (20, 0.0, 1.0),
                (21, 0.0, 0.0),
                (27, 0.0, 0.0),
                (28, 1.0, 0.0),
                (29, 2.0, 0.0),
            ] {
                let values = oracle::image(32, 32, |x, y| {
                    if (if axis == 0 { x } else { y }) == position {
                        1.0
                    } else {
                        0.0
                    }
                });
                let result = scalar(gpu, 32, 32, &values);
                close(
                    result.boundary_rms,
                    (boundary_edges / 16.0).sqrt(),
                    2e-5,
                    "boundary membership",
                );
                close(
                    result.interior_rms,
                    (interior_edges / 16.0).sqrt(),
                    2e-5,
                    "interior membership",
                );
            }
        }
    });
}

#[test]
fn flat_fields_and_nonzero_boundary_with_zero_interior_have_finite_explicit_ratios() {
    with_gpu(|gpu| {
        for (w, h) in [(32, 24), (1, 1), (1, 19), (23, 1), (2, 3), (7, 5)] {
            for value in [-0.125, 0.0, 0.5, 2.0] {
                let result = scalar(gpu, w, h, &vec![value; w * h]);
                assert_eq!(
                    (result.boundary_rms, result.interior_rms, result.ratio),
                    (0.0, 0.0, 0.0)
                );
                assert!(result.heatmap.iter().all(|&v| v == 0.0));
            }
        }
        let values = oracle::image(32, 32, |x, _| if x == 0 { 1.0 } else { 0.0 });
        let result = scalar(gpu, 32, 32, &values);
        close(
            result.boundary_rms,
            0.125_f64.sqrt(),
            2e-5,
            "zero-interior seam",
        );
        assert_eq!(result.interior_rms, 0.0);
        assert!(result.ratio > 1e7 && result.ratio.is_finite());
    });
}

#[test]
fn analytic_periodic_output_scores_near_zero_and_injected_seams_score_high_on_each_axis() {
    with_gpu(|gpu| {
        let (w, h) = (256, 192);
        let values = oracle::periodic(w, h);
        let good = scalar(gpu, w, h, &values);
        assert!(good.boundary_rms < 5e-4);
        for axis in 0..2 {
            let mut damaged = values.clone();
            for y in 0..h {
                for x in 0..w {
                    if (axis == 0 && x == 0) || (axis == 1 && y == 0) {
                        damaged[y * w + x] += 1.0;
                    }
                }
            }
            let bad = scalar(gpu, w, h, &damaged);
            assert!(bad.boundary_rms > 0.2 && bad.ratio > 100.0);
            assert!(bad.boundary_rms > 400.0 * good.boundary_rms);
            let wrap = if axis == 0 { w - 1 } else { (h - 1) * w };
            assert!(
                bad.heatmap[wrap] > 0.99,
                "missing seam at opposite edge for axis {axis}"
            );
        }
    });
}

#[test]
fn seeded_npot_impulse_ramp_and_singleton_inputs_match_every_oracle_pixel() {
    with_gpu(|gpu| {
        for (w, h) in [(257, 193), (17, 11), (1, 29), (31, 1), (1, 1), (3, 2)] {
            let mut state = 60_u32;
            let noise = oracle::image(w, h, |_, _| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 16) as f32 / 65535.0
            });
            scalar(gpu, w, h, &noise);
            let impulse = oracle::image(
                w,
                h,
                |x, y| if x == w / 2 && y == h / 2 { 0.75 } else { 0.0 },
            );
            scalar(gpu, w, h, &impulse);
            scalar(gpu, w, h, &oracle::periodic(w, h));
        }
    });
}

#[test]
fn diagnostic_uses_actual_processed_material_output_and_preserves_source_height_precision() {
    with_gpu(|gpu| {
        let (w, h) = (33, 19);
        let values = oracle::image(w, h, |x, _| 0.01 + 0.8 * x as f32 / (w - 1) as f32);
        let raw = scalar(gpu, w, h, &values);
        let (processed, pixels) = material(
            gpu,
            w,
            h,
            TypedSourcePixels::HeightF32(values.clone().into()),
            true,
        );
        assert!(
            pixels
                .iter()
                .zip(&values)
                .any(|(p, &v)| (p[0] - f64::from(v)).abs() > 0.1)
        );
        assert!(
            processed
                .heatmap
                .iter()
                .zip(&raw.heatmap)
                .any(|(a, b)| (a - b).abs() > 0.01)
        );
        let u16_values: Vec<u16> = (0..w * h).map(|i| (10000 + i * 7) as u16).collect();
        let (_, pixels) = material(
            gpu,
            w,
            h,
            TypedSourcePixels::HeightU16(u16_values.clone().into()),
            false,
        );
        for (p, v) in pixels.iter().zip(u16_values) {
            close(p[0], f64::from(v) / 65535.0, 1e-7, "linear u16 import");
        }
        assert!(
            pixels.windows(2).all(|p| p[0] != p[1]),
            "height was reduced to 8 bits"
        );
    });
}

#[test]
fn linear_rgb_differences_include_chromatic_seams_and_ignore_photo_alpha() {
    with_gpu(|gpu| {
        let (w, h) = (32, 24);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                if i % w < w / 2 {
                    [128, 0, 0, 0]
                } else {
                    [0, 76, 0, 255]
                }
            })
            .collect();
        let (srgb, pixels) = material(
            gpu,
            w,
            h,
            TypedSourcePixels::Rgba8 {
                pixels: rgba.clone().into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
            false,
        );
        let red = ((128.0_f64 / 255.0 + 0.055) / 1.055).powf(2.4);
        close(
            pixels[0][0],
            red,
            2e-4,
            "one sRGB decode before diagnostics",
        );
        assert!(
            srgb.boundary_rms > 0.02,
            "RGB seam was hidden by near-equal luminance"
        );
        let mut changed_alpha = rgba.clone();
        for p in changed_alpha.as_chunks_mut::<4>().0 {
            p[3] = 255 - p[3];
        }
        let (alpha, _) = material(
            gpu,
            w,
            h,
            TypedSourcePixels::Rgba8 {
                pixels: changed_alpha.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
            false,
        );
        assert_eq!(srgb, alpha);
    });
}

#[test]
fn same_typed_input_is_bit_deterministic_and_new_input_does_not_reuse_old_metrics() {
    with_gpu(|gpu| {
        let (w, h) = (31, 17);
        let values = oracle::image(w, h, |x, y| ((x * 13 + y * 7) % 31) as f32 / 31.0);
        let first = scalar(gpu, w, h, &values);
        let flat = scalar(gpu, w, h, &vec![0.375; w * h]);
        assert_eq!(flat.boundary_rms, 0.0);
        let again = scalar(gpu, w, h, &values);
        assert_eq!(
            first, again,
            "same backend/input must produce identical diagnostic bytes"
        );
        assert_ne!(first, flat);
    });
}
