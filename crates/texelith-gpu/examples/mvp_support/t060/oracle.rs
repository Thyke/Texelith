//! Independent f64 reference, usable by the numerical suite and standalone rustc.
//! This file never imports GPU code or consumes implementation-produced metrics.
pub const BAND: usize = 4;
pub const DENOMINATOR_FLOOR: f64 = 1e-8;

#[derive(Debug)]
pub struct Diagnostic {
    pub heatmap: Vec<f64>,
    pub boundary_rms: f64,
    pub interior_rms: f64,
    pub ratio: f64,
    pub boundary_samples: usize,
    pub interior_samples: usize,
}

/// Forward differences in pixel units; singleton axes contribute no samples.
/// Color distance is RMS over linear RGB, with alpha excluded.
pub fn inspect(width: usize, height: usize, pixels: &[[f64; 3]]) -> Diagnostic {
    assert!(width > 0 && height > 0);
    assert_eq!(pixels.len(), width * height);
    assert!(pixels.iter().flatten().all(|v| v.is_finite()));
    let mut energy = vec![0.0; pixels.len()];
    let mut boundary = Vec::new();
    let mut interior = Vec::new();
    for (axis, length) in [width, height].into_iter().enumerate() {
        if length == 1 {
            continue;
        }
        // Small extents shrink symmetrically; masks are sets, never duplicated.
        let band = BAND.min(length / 2);
        let middle = length / 2 - band;
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let (coordinate, next) = if axis == 0 {
                    (x, y * width + (x + 1) % width)
                } else {
                    (y, ((y + 1) % height) * width + x)
                };
                let squared = pixels[index]
                    .iter()
                    .zip(pixels[next])
                    .map(|(a, b)| (b - a).powi(2))
                    .sum::<f64>()
                    / 3.0;
                energy[index] += squared;
                if coordinate < band || coordinate >= length - band {
                    boundary.push(squared);
                }
                if (middle..middle + 2 * band).contains(&coordinate) {
                    interior.push(squared);
                }
            }
        }
    }
    let rms = |samples: &[f64]| {
        if samples.is_empty() {
            0.0
        } else {
            (samples.iter().sum::<f64>() / samples.len() as f64).sqrt()
        }
    };
    let boundary_rms = rms(&boundary);
    let interior_rms = rms(&interior);
    Diagnostic {
        heatmap: energy.into_iter().map(f64::sqrt).collect(),
        boundary_rms,
        interior_rms,
        ratio: boundary_rms / interior_rms.max(DENOMINATOR_FLOOR),
        boundary_samples: boundary.len(),
        interior_samples: interior.len(),
    }
}

pub fn scalar(values: &[f32]) -> Vec<[f64; 3]> {
    values.iter().map(|&v| [f64::from(v); 3]).collect()
}

pub fn image(width: usize, height: usize, mut pixel: impl FnMut(usize, usize) -> f32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| pixel(x, y))
        .collect()
}

pub fn periodic(width: usize, height: usize) -> Vec<f32> {
    image(width, height, |x, y| {
        let u = std::f64::consts::TAU * (x as f64 + 0.5) / width as f64;
        let v = std::f64::consts::TAU * (y as f64 + 0.5) / height as f64;
        (0.5 + 0.125 * u.cos() + 0.125 * v.cos()) as f32
    })
}

/// Streaming scalar reference for full-resolution evidence. Avoid expanding a
/// 4K scalar fixture into RGB f64 pixels and multiple full-size f64 planes.
pub fn inspect_scalar_streaming(
    width: usize,
    height: usize,
    values: &[f32],
    mut heatmap: impl FnMut(usize, f64),
) -> [f64; 3] {
    assert_eq!(values.len(), width * height);
    let mut sums = [0.0_f64; 2];
    let mut counts = [0_usize; 2];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let a = f64::from(values[index]);
            let mut total = 0.0;
            for (n, coordinate, next) in [
                (width, x, y * width + (x + 1) % width),
                (height, y, ((y + 1) % height) * width + x),
            ] {
                if n == 1 {
                    continue;
                }
                let energy = (f64::from(values[next]) - a).powi(2);
                total += energy;
                let band = BAND.min(n / 2);
                let middle = n / 2 - band;
                for (mask, member) in [
                    coordinate < band || coordinate >= n - band,
                    (middle..middle + 2 * band).contains(&coordinate),
                ]
                .into_iter()
                .enumerate()
                {
                    if member {
                        sums[mask] += energy;
                        counts[mask] += 1;
                    }
                }
            }
            heatmap(index, total.sqrt());
        }
    }
    let rms: [f64; 2] = std::array::from_fn(|i| {
        if counts[i] == 0 {
            0.0
        } else {
            (sums[i] / counts[i] as f64).sqrt()
        }
    });
    [rms[0], rms[1], rms[0] / rms[1].max(DENOMINATOR_FLOOR)]
}

#[test]
fn streaming_scalar_reference_agrees_with_rgb_reference() {
    for (w, h) in [(257, 193), (32, 24), (1, 17), (19, 1), (1, 1), (3, 2)] {
        let values = image(w, h, |x, y| ((x * 13 + y * 7) % 31) as f32 / 31.0);
        let expected = inspect(w, h, &scalar(&values));
        let actual = inspect_scalar_streaming(w, h, &values, |i, v| {
            assert!((v - expected.heatmap[i]).abs() < 1e-14)
        });
        for (a, b) in
            actual
                .into_iter()
                .zip([expected.boundary_rms, expected.interior_rms, expected.ratio])
        {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn oracle_flat_zero_denominator_and_singleton_are_exact_zero() {
    for (w, h) in [(32, 24), (1, 1), (1, 19), (23, 1), (2, 3)] {
        let d = inspect(w, h, &vec![[0.25; 3]; w * h]);
        assert_eq!((d.boundary_rms, d.interior_rms, d.ratio), (0.0, 0.0, 0.0));
        assert!(d.heatmap.iter().all(|&v| v == 0.0));
    }
}

#[test]
fn oracle_hand_calculated_ramp_counts_wrap_and_all_four_bands() {
    let (w, h) = (32, 24);
    let values = image(w, h, |x, _| x as f32 / 32.0);
    let d = inspect(w, h, &scalar(&values));
    let count = 8 * (w + h);
    assert_eq!((d.boundary_samples, d.interior_samples), (count, count));
    let expected_boundary =
        ((h as f64 * (7.0 + 31.0_f64.powi(2))) / (count as f64 * 32.0_f64.powi(2))).sqrt();
    let expected_interior = ((8 * h) as f64 / (count as f64 * 32.0_f64.powi(2))).sqrt();
    assert!((d.boundary_rms - expected_boundary).abs() < 1e-14);
    assert!((d.interior_rms - expected_interior).abs() < 1e-14);
    assert!((d.ratio - (121.0_f64).sqrt()).abs() < 1e-14);
    assert_eq!(d.heatmap[w - 1], 31.0 / 32.0);
    assert_eq!(d.heatmap[w - 2], 1.0 / 32.0);
}

#[test]
fn oracle_fourth_band_pixel_is_included_fifth_is_excluded_on_both_sides_and_axes() {
    for axis in 0..2 {
        for (location, edges) in [
            (3, 2.0_f64),
            (4, 1.0),
            (5, 0.0),
            (27, 0.0),
            (28, 1.0),
            (29, 2.0),
        ] {
            // A single nonzero sample creates forward edges at p-1 and p.
            let values = image(32, 32, |x, y| {
                if (if axis == 0 { x } else { y }) == location {
                    1.0
                } else {
                    0.0
                }
            });
            let d = inspect(32, 32, &scalar(&values));
            assert!(
                (d.boundary_rms - (edges / 16.0).sqrt()).abs() < 1e-14,
                "axis={axis} p={location}"
            );
        }
    }
}

#[test]
fn oracle_periodic_input_is_near_zero_and_injected_seam_is_large() {
    let (w, h) = (256, 192);
    let input = periodic(w, h);
    let good = inspect(w, h, &scalar(&input));
    assert!(good.boundary_rms < 5e-4, "{good:?}");
    let mut damaged = input.clone();
    for y in 0..h {
        damaged[y * w] += 1.0;
    }
    let bad = inspect(w, h, &scalar(&damaged));
    assert!(bad.boundary_rms > 0.2 && bad.ratio > 100.0);
    assert!(bad.boundary_rms > 400.0 * good.boundary_rms);
}
