//! Independent f64 reference for the T-053 numerical runner and integration tests.
//! No GPU dispatch or production shader logic belongs here.

pub fn linear(code: u8) -> f64 {
    let value = f64::from(code) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Round to the published ColorSource Rgba16Float boundary, ties to even.
pub fn half(value: f64) -> f64 {
    if value == 0.0 {
        return value;
    }
    let spacing = 2.0_f64.powi((value.abs().log2().floor() as i32 - 10).max(-24));
    (value / spacing).round_ties_even() * spacing
}

pub fn luminance(rgba: &[u8]) -> Vec<f64> {
    assert!(!rgba.is_empty() && rgba.len().is_multiple_of(4));
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            0.2126 * half(linear(p[0])) + 0.7152 * half(linear(p[1])) + 0.0722 * half(linear(p[2]))
        })
        .collect()
}

/// Knots need not include x=0 or x=1: outside the domain, hold the endpoint.
pub fn remap(value: f64, curve: &[[f32; 2]]) -> f64 {
    if value <= f64::from(curve[0][0]) {
        return f64::from(curve[0][1]);
    }
    for pair in curve.windows(2) {
        let [x0, y0] = pair[0].map(f64::from);
        let [x1, y1] = pair[1].map(f64::from);
        if value <= x1 {
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0);
        }
    }
    f64::from(curve.last().unwrap()[1])
}

pub fn roughness(
    luma: &[f64],
    mask: Option<&[f32]>,
    curve: &[[f32; 2]],
    base: f32,
    invert: bool,
) -> Vec<f64> {
    if let Some(mask) = mask {
        assert_eq!(mask.len(), luma.len());
    }
    luma.iter()
        .enumerate()
        .map(|(i, value)| {
            let mask = mask.map_or(1.0, |m| f64::from(m[i]).clamp(0.0, 1.0));
            let mapped = remap(*value, curve);
            let result = f64::from(base) * (1.0 - mask) + mapped * mask;
            if invert { 1.0 - result } else { result }
        })
        .collect()
}

pub fn compare(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert!(!expected.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (0.0..=1.0).contains(actual),
            "pixel {i}: {actual}"
        );
        assert!(
            (f64::from(*actual) - expected).abs() <= tolerance,
            "pixel {i}: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_has_closed_form_anchors_for_transfer_curve_mask_and_inversion() {
        assert!((linear(128) - 0.21586050011389926).abs() < 1e-14);
        assert!((linear(10) - 10.0 / 255.0 / 12.92).abs() < 1e-14);
        assert_eq!(luminance(&[255, 0, 0, 0]), vec![0.2126]);
        assert_eq!(luminance(&[0, 255, 0, 255]), vec![0.7152]);
        assert_eq!(luminance(&[0, 0, 255, 7]), vec![0.0722]);
        let curve = [[0.25, 0.75], [0.5, 0.25], [0.75, 1.0]];
        assert_eq!(remap(0.0, &curve), 0.75);
        assert_eq!(remap(0.375, &curve), 0.5);
        assert_eq!(remap(0.625, &curve), 0.625);
        assert_eq!(remap(1.0, &curve), 1.0);
        let actual = roughness(&[0.375; 3], Some(&[0.0, 0.25, 1.0]), &curve, 0.25, true);
        assert_eq!(actual, vec![0.75, 0.6875, 0.5]);
    }
}
