//! Test-only f64 reconstruction, using the independently anchored T-046 filters.
use super::oracle::Image;
use texelith_core::{Boundary, Extent};

pub fn linear(code: u8) -> f64 {
    let value = f64::from(code) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

// ColorSource's public output is binary16. Quantize the analytic transfer to
// that interface before applying the stricter R32 height tolerance.
fn half(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let spacing = 2.0_f64.powi((value.log2().floor() as i32 - 10).max(-24));
    (value / spacing).round_ties_even() * spacing
}

pub fn luminance(extent: Extent, rgba: &[u8]) -> Image {
    Image::new(
        extent.width,
        extent.height,
        rgba.as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                0.2126 * half(linear(p[0]))
                    + 0.7152 * half(linear(p[1]))
                    + 0.0722 * half(linear(p[2]))
            })
            .collect(),
    )
}

pub fn pyramid(source: &Image, boundary: Boundary) -> Vec<Image> {
    let mut levels = vec![source.clone()];
    while levels.last().unwrap().extent != Extent::new(1, 1).unwrap() {
        let next = levels
            .last()
            .unwrap()
            .gaussian(false, boundary)
            .gaussian(true, boundary)
            .reduce();
        levels.push(next);
    }
    levels
}

pub fn reconstruct(source: &Image, weights: &[f32], coarse: f32, boundary: Boundary) -> Image {
    assert!(!weights.is_empty());
    let levels = pyramid(source, boundary);
    let mut result = levels.last().unwrap().clone();
    result
        .pixels
        .iter_mut()
        .for_each(|v| *v *= f64::from(coarse));
    for l in (0..levels.len() - 1).rev() {
        let base = levels[l + 1].expand(levels[l].extent, boundary);
        let expanded = result.expand(levels[l].extent, boundary);
        let weight = f64::from(*weights.get(l).unwrap_or(weights.last().unwrap()));
        result = Image::new(
            levels[l].extent.width,
            levels[l].extent.height,
            levels[l]
                .pixels
                .iter()
                .zip(base.pixels)
                .zip(expanded.pixels)
                .map(|((fine, base), accumulated)| weight * (fine - base) + accumulated)
                .collect(),
        );
    }
    result
}

pub fn levels(source: &Image, black: f32, white: f32, gamma: f32) -> Vec<f64> {
    source
        .pixels
        .iter()
        .map(|value| {
            ((value - f64::from(black)) / f64::from(white - black))
                .clamp(0.0, 1.0)
                .powf(1.0 / f64::from(gamma))
        })
        .collect()
}

#[test]
fn oracle_has_closed_form_color_and_reconstruction_anchors() {
    let e = Extent::new(5, 1).unwrap();
    let source = luminance(
        e,
        &[
            0, 0, 0, 17, 255, 255, 255, 0, 255, 0, 0, 255, 0, 255, 0, 30, 0, 0, 255, 1,
        ],
    );
    assert_eq!(source.pixels, vec![0.0, 1.0, 0.2126, 0.7152, 0.0722]);
    assert!((linear(10) - (10.0 / 255.0) / 12.92).abs() < 1e-15);
    assert!(linear(11) > linear(10));
    for boundary in [Boundary::Repeat, Boundary::Clamp] {
        let unit = reconstruct(&source, &[1.0], 1.0, boundary);
        for (a, b) in unit.pixels.iter().zip(&source.pixels) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(
            reconstruct(&source, &[0.0], 0.0, boundary)
                .pixels
                .iter()
                .all(|v| *v == 0.0)
        );
        let flat = Image::new(5, 1, vec![0.25; 5]);
        assert_eq!(
            reconstruct(&flat, &[2.0, -1.0], 0.0, boundary).pixels,
            vec![0.0; 5]
        );
    }
}
