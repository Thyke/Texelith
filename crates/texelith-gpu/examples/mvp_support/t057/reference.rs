//! Original f64 CPU oracle, independent of production transforms and shaders.
use texelith_core::{Boundary, Extent};

pub const EPSILON: f64 = 1e-8;

pub fn window(coordinate: f64, overlap: f64) -> f64 {
    let distance = coordinate.min(1.0 - coordinate);
    let t = (distance / overlap).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Continuous half-period translations; never round offsets to source texels.
pub fn samples(uv: [f64; 2], overlap: f64) -> [([f64; 2], f64); 4] {
    std::array::from_fn(|i| {
        let coordinate = [
            (uv[0] + if i & 1 == 0 { 0.5 } else { 0.0 }).rem_euclid(1.0),
            (uv[1] + if i & 2 == 0 { 0.5 } else { 0.0 }).rem_euclid(1.0),
        ];
        (
            coordinate,
            window(coordinate[0], overlap) * window(coordinate[1], overlap),
        )
    })
}

pub fn bilinear(extent: Extent, values: &[f64], uv: [f64; 2], boundary: Boundary) -> f64 {
    assert_eq!(values.len(), (extent.width * extent.height) as usize);
    let x = uv[0] * f64::from(extent.width) - 0.5;
    let y = uv[1] * f64::from(extent.height) - 0.5;
    let address = |p: i64, size: u32| match boundary {
        Boundary::Clamp => p.clamp(0, i64::from(size) - 1),
        Boundary::Repeat => p.rem_euclid(i64::from(size)),
    } as usize;
    let get = |dx, dy| {
        values[address(y.floor() as i64 + dy, extent.height) * extent.width as usize
            + address(x.floor() as i64 + dx, extent.width)]
    };
    let tx = x - x.floor();
    let ty = y - y.floor();
    (get(0, 0) * (1.0 - tx) + get(1, 0) * tx) * (1.0 - ty)
        + (get(0, 1) * (1.0 - tx) + get(1, 1) * tx) * ty
}

pub fn at(extent: Extent, values: &[f64], uv: [f64; 2], overlap: f64, boundary: Boundary) -> f64 {
    let taps = samples(uv, overlap);
    let total: f64 = taps.iter().map(|(_, weight)| weight).sum();
    taps.into_iter()
        .map(|(coordinate, weight)| weight * bilinear(extent, values, coordinate, boundary))
        .sum::<f64>()
        / total.max(EPSILON)
}

pub fn render(extent: Extent, values: &[f64], overlap: f64, boundary: Boundary) -> Vec<f64> {
    (0..extent.width * extent.height)
        .map(|i| {
            at(
                extent,
                values,
                [
                    (f64::from(i % extent.width) + 0.5) / f64::from(extent.width),
                    (f64::from(i / extent.width) + 0.5) / f64::from(extent.height),
                ],
                overlap,
                boundary,
            )
        })
        .collect()
}

pub fn scalar_fixture(extent: Extent, kind: &str) -> Vec<f32> {
    let mut state = 57u32;
    (0..extent.width * extent.height)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (i % extent.width) as f32 / extent.width as f32;
            let y = (i / extent.width) as f32 / extent.height as f32;
            let noise = (state >> 8) as f32 / 16777215.0;
            match kind {
                "constant" => 0.375,
                "ramp" => -0.25 + x + y * 0.5,
                "impulse" => {
                    if i == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "structured" => {
                    if (0.2..0.3).contains(&x) {
                        1.0
                    } else {
                        0.0
                    }
                }
                "discontinuous_noise" => 0.1 * noise + 0.5 * x + 0.3 * y,
                _ => noise,
            }
        })
        .collect()
}

/// Four-pixel bands on each side, including the wrap-crossing difference.
/// The comparison band contains exactly the same number of interior edges.
pub fn seam_rms(extent: Extent, values: &[f64]) -> (f64, f64) {
    assert!(extent.width >= 16 && extent.height >= 16);
    assert_eq!(values.len(), (extent.width * extent.height) as usize);
    let mut boundary_energy = 0.0;
    let mut interior_energy = 0.0;
    let mut count = 0;
    for axis in 0..2 {
        let (length, across) = if axis == 0 {
            (extent.width, extent.height)
        } else {
            (extent.height, extent.width)
        };
        let get = |x: u32, y: u32| {
            values[if axis == 0 {
                (y * extent.width + x) as usize
            } else {
                (x * extent.width + y) as usize
            }]
        };
        for row in 0..across {
            for delta in -3i64..=3 {
                let edge = delta.rem_euclid(i64::from(length)) as u32;
                let middle = (i64::from(length / 2) + delta) as u32;
                boundary_energy +=
                    (get(edge, row) - get((edge + length - 1) % length, row)).powi(2);
                interior_energy += (get(middle, row) - get(middle - 1, row)).powi(2);
                count += 1;
            }
        }
    }
    (
        (boundary_energy / f64::from(count)).sqrt(),
        (interior_energy / f64::from(count)).sqrt(),
    )
}

pub fn linear(code: u8) -> f64 {
    let x = f64::from(code) / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

pub fn half(value: f64) -> f64 {
    if value == 0.0 {
        return value;
    }
    let step = 2.0_f64.powi((value.abs().log2().floor() as i32 - 10).max(-24));
    (value / step).round_ties_even() * step
}

#[test]
fn oracle_has_closed_form_anchors_and_nontrivial_fixture_energy() {
    let e = Extent::new(3, 1).unwrap();
    // At u=0 only the continuously shifted copy contributes: center texel 1.
    assert_eq!(
        at(e, &[0.0, 3.0, 12.0], [0.0, 0.5], 0.25, Boundary::Repeat),
        3.0
    );
    // At u=.25 copies at .25/.75 have equal weight; samples .75 and 9.75.
    assert_eq!(
        at(e, &[0.0, 3.0, 12.0], [0.25, 0.5], 0.25, Boundary::Clamp),
        5.25
    );
    assert_eq!(window(0.0, 0.25), 0.0);
    assert_eq!(window(0.125, 0.25), 0.5);
    assert_eq!(window(0.25, 0.25), 1.0);
    let e = Extent::new(257, 193).unwrap();
    let input: Vec<f64> = scalar_fixture(e, "discontinuous_noise")
        .into_iter()
        .map(f64::from)
        .collect();
    let output = render(e, &input, 0.25, Boundary::Repeat);
    let (before, interior_before) = seam_rms(e, &input);
    let (after, interior_after) = seam_rms(e, &output);
    assert!(after <= before * 0.5, "oracle seam: {before} -> {after}");
    assert!(
        interior_after >= interior_before * 0.5,
        "oracle erased interior detail"
    );
}
