//! Independent f64 oracle for hard circular brushes swept along UV segments.
//! Fixtures use hardness=1; the ADR does not prescribe a soft falloff curve.
use texelith_core::{Boundary, BrushSample, BrushStroke, Extent, MaskMode};

pub const TOLERANCE: f64 = 2e-5;

fn segment_alpha(p: [f64; 2], a: BrushSample, b: BrushSample, radius: f64) -> f64 {
    let a_uv = a.uv.map(f64::from);
    let b_uv = b.uv.map(f64::from);
    let delta = [b_uv[0] - a_uv[0], b_uv[1] - a_uv[1]];
    let length2 = delta[0] * delta[0] + delta[1] * delta[1];
    let t = if length2 == 0.0 {
        0.0
    } else {
        (((p[0] - a_uv[0]) * delta[0] + (p[1] - a_uv[1]) * delta[1]) / length2).clamp(0.0, 1.0)
    };
    let dx = p[0] - (a_uv[0] + delta[0] * t);
    let dy = p[1] - (a_uv[1] + delta[1] * t);
    if dx * dx + dy * dy >= radius * radius {
        0.0
    } else if length2 == 0.0 {
        f64::from(a.pressure.max(b.pressure))
    } else {
        f64::from(a.pressure) + (f64::from(b.pressure) - f64::from(a.pressure)) * t
    }
}

pub fn raster(extent: Extent, strokes: &[BrushStroke], boundary: Boundary) -> Vec<f64> {
    let mut output = Vec::with_capacity((extent.width * extent.height) as usize);
    for y in 0..extent.height {
        for x in 0..extent.width {
            let uv = [
                (f64::from(x) + 0.5) / f64::from(extent.width),
                (f64::from(y) + 0.5) / f64::from(extent.height),
            ];
            let mut value = 0.0_f64;
            for stroke in strokes {
                assert_eq!(stroke.brush.hardness, 1.0, "hard-brush oracle only");
                stroke.validate().unwrap();
                let mut coverage = 0.0_f64;
                let translations = if boundary == Boundary::Repeat {
                    -1..=1
                } else {
                    0..=0
                };
                for ty in translations.clone() {
                    for tx in translations.clone() {
                        let p = [uv[0] + f64::from(tx), uv[1] + f64::from(ty)];
                        let first = stroke.points[0];
                        coverage = coverage.max(segment_alpha(
                            p,
                            first,
                            first,
                            f64::from(stroke.brush.radius),
                        ));
                        for pair in stroke.points.windows(2) {
                            coverage = coverage.max(segment_alpha(
                                p,
                                pair[0],
                                pair[1],
                                f64::from(stroke.brush.radius),
                            ));
                        }
                    }
                }
                let alpha = coverage * f64::from(stroke.brush.opacity);
                value = match stroke.brush.mode {
                    MaskMode::Add => value + alpha,
                    MaskMode::Subtract => value - alpha,
                    MaskMode::Replace => value + (1.0 - value) * alpha,
                }
                .clamp(0.0, 1.0);
            }
            output.push(value);
        }
    }
    output
}

pub fn compare(actual: &[f32], expected: &[f64]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (0.0..=1.0).contains(&actual),
            "pixel {i}: {actual}"
        );
        assert!(
            (f64::from(actual) - expected).abs() <= TOLERANCE,
            "pixel {i}: GPU {actual}, CPU {expected}"
        );
    }
}
