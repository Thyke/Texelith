//! Original f64 reference, derived from the ADR, independent of GPU code.
use texelith_core::{Boundary, Extent, MaterialNodeParams};

#[derive(Clone, Copy, Debug)]
pub struct Controls {
    pub directions: u32,
    pub steps: u32,
    pub radius: f32,
    pub bias: f32,
    pub boundary: Boundary,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            directions: 8,
            steps: 8,
            radius: 32.0,
            bias: 0.01,
            boundary: Boundary::Repeat,
        }
    }
}

impl Controls {
    pub fn params(self) -> MaterialNodeParams {
        MaterialNodeParams::HorizonAo {
            directions: self.directions,
            steps: self.steps,
            radius_source_px: self.radius,
            bias: self.bias,
            boundary: self.boundary,
        }
    }
}

pub fn image(extent: Extent, f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
    (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y))
        .collect()
}

fn bilinear(data: &[f32], e: Extent, x: f64, y: f64, boundary: Boundary) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fetch = |x: i64, y: i64| {
        let address = |p: i64, size: u32| match boundary {
            Boundary::Repeat => p.rem_euclid(i64::from(size)) as usize,
            Boundary::Clamp => p.clamp(0, i64::from(size) - 1) as usize,
        };
        f64::from(data[address(y, e.height) * e.width as usize + address(x, e.width)])
    };
    let tx = x - ix as f64;
    let ty = y - iy as f64;
    let a = fetch(ix, iy) * (1.0 - tx) + fetch(ix + 1, iy) * tx;
    let b = fetch(ix, iy + 1) * (1.0 - tx) + fetch(ix + 1, iy + 1) * tx;
    a * (1.0 - ty) + b * ty
}

/// Steps k=1..=steps span the radius; direction zero is +X. Offsets are in
/// original source pixels, lengths in UV. Bias is subtracted from the slope,
/// before max with zero. cos(atan(slope))^2 = 1/(1+slope^2).
pub fn ao(data: &[f32], extent: Extent, source: Extent, c: Controls) -> Vec<f64> {
    assert_eq!(data.len(), extent.width as usize * extent.height as usize);
    let mut result = Vec::with_capacity(data.len());
    for y in 0..extent.height {
        for x in 0..extent.width {
            let center = f64::from(data[(y * extent.width + x) as usize]);
            let mut sum = 0.0;
            for d in 0..c.directions {
                let angle = std::f64::consts::TAU * f64::from(d) / f64::from(c.directions);
                let mut horizon = 0.0_f64;
                for k in 1..=c.steps {
                    let distance = f64::from(c.radius) * f64::from(k) / f64::from(c.steps);
                    let du = angle.cos() * distance / f64::from(source.width);
                    let dv = angle.sin() * distance / f64::from(source.height);
                    let sample = bilinear(
                        data,
                        extent,
                        f64::from(x) + du * f64::from(extent.width),
                        f64::from(y) + dv * f64::from(extent.height),
                        c.boundary,
                    );
                    horizon = horizon.max((sample - center) / du.hypot(dv) - f64::from(c.bias));
                }
                sum += 1.0 / (1.0 + horizon * horizon);
            }
            result.push(sum / f64::from(c.directions));
        }
    }
    result
}

pub fn compare(actual: &[f32], expected: &[f64]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (0.0..=1.0).contains(&a),
            "invalid AO at {i}: {a}"
        );
        assert!(
            b.is_finite() && (f64::from(a) - b).abs() <= 2e-5,
            "pixel {i}: GPU {a}, CPU {b}; tolerance 2e-5"
        );
    }
}

pub fn seeded(e: Extent) -> Vec<f32> {
    let mut state = 51_u32;
    (0..e.width * e.height)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Gentle heights preserve sensitivity to directions/steps and UV metric.
            (state >> 8) as f32 / 16777215.0 * 0.025 - 0.01
        })
        .collect()
}

#[test]
fn reference_matches_closed_form_cardinal_ramp_and_flat_plane() {
    let e = Extent::new(17, 9).unwrap();
    let c = Controls {
        directions: 4,
        steps: 4,
        radius: 4.0,
        bias: 0.0,
        boundary: Boundary::Clamp,
    };
    let ramp = image(e, |x, _| x as f32 / 32.0);
    let slope = 17.0_f64 / 32.0;
    let expected = (3.0 + 1.0 / (1.0 + slope * slope)) / 4.0;
    assert!((ao(&ramp, e, e, c)[4 * 17 + 8] - expected).abs() < 1e-12);
    for boundary in [Boundary::Clamp, Boundary::Repeat] {
        assert!(
            ao(&vec![-2.0; 153], e, e, Controls { boundary, ..c })
                .iter()
                .all(|&v| v == 1.0)
        );
    }
}

#[test]
fn fixtures_distinguish_sampling_controls_and_source_resolution() {
    let e = Extent::new(31, 23).unwrap();
    let pixels = seeded(e);
    let base = Controls {
        radius: 9.5,
        ..Default::default()
    };
    let reference = ao(&pixels, e, e, base);
    for c in [
        Controls {
            directions: 13,
            ..base
        },
        Controls { steps: 17, ..base },
        Controls {
            radius: 3.25,
            ..base
        },
        Controls { bias: 0.3, ..base },
    ] {
        assert!(
            ao(&pixels, e, e, c)
                .iter()
                .zip(&reference)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "ineffective fixture for {c:?}"
        );
    }
    let source = Extent::new(2048, 32).unwrap();
    let proxy = Extent::new(1024, 16).unwrap();
    let pixels = image(
        proxy,
        |x, _| if (x / 8).is_multiple_of(2) { 0.0 } else { 0.01 },
    );
    for radius in [8.0, 32.0] {
        let c = Controls {
            radius,
            ..Default::default()
        };
        assert!(
            ao(&pixels, proxy, source, c)
                .iter()
                .zip(ao(&pixels, proxy, proxy, c))
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }
    let e = Extent::new(19, 13).unwrap();
    let pixels = image(e, |x, _| (32768 + x * 61) as f32 / 65535.0);
    assert!(
        ao(
            &pixels,
            e,
            e,
            Controls {
                radius: 3.0,
                bias: 0.0,
                ..Default::default()
            }
        )
        .iter()
        .any(|&v| v < 1.0 - 4e-5)
    );
}
