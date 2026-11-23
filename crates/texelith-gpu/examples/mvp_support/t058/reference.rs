//! Independent f64 test oracle; original fixtures, no production transform/math.
//!
//! This freezes the previously unspecified u32 hash/window as a tests-first
//! proposal. See docs/evidence/T-058/test-contract.md before implementing.
use texelith_core::{Boundary, Extent};

pub const MAX_NEIGHBORS: usize = 9;

#[derive(Clone, Copy, Debug)]
pub struct Tap {
    pub uv: [f64; 2],
    pub weight: f64,
}

pub fn assignment(seed: u32, cell: [i32; 2], grid: u32) -> ([f64; 2], u32) {
    let x = cell[0].rem_euclid(grid as i32) as u32;
    let y = cell[1].rem_euclid(grid as i32) as u32;
    let mut h = seed ^ x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (
        [
            f64::from(h & 0x7fff) / 32768.0,
            f64::from((h >> 15) & 0x7fff) / 32768.0,
        ],
        h >> 30,
    )
}

pub fn samples(uv: [f64; 2], seed: u32, grid: u32, overlap: f64) -> [Tap; MAX_NEIGHBORS] {
    let p = uv.map(|v| v.rem_euclid(1.0) * f64::from(grid));
    let cell = p.map(|v| v.floor() as i32);
    let radius = 1.0 + overlap;
    std::array::from_fn(|i| {
        let c = [cell[0] + i as i32 % 3 - 1, cell[1] + i as i32 / 3 - 1];
        let d = [p[0] - f64::from(c[0]) - 0.5, p[1] - f64::from(c[1]) - 0.5];
        let window = |d: f64| {
            let t = (1.0 - d.abs() / radius).clamp(0.0, 1.0);
            let smooth = t * t * (3.0 - 2.0 * t);
            // Concentrate contributions without losing compact C1 support;
            // a broad window erases over half the fixture's interior RMS.
            smooth.powi(4)
        };
        let (translation, rotation) = assignment(seed, c, grid);
        let rotated = match rotation {
            0 => d,
            1 => [-d[1], d[0]],
            2 => [-d[0], -d[1]],
            3 => [d[1], -d[0]],
            _ => unreachable!(),
        };
        Tap {
            uv: std::array::from_fn(|axis| {
                (translation[axis] + rotated[axis] / f64::from(grid)).rem_euclid(1.0)
            }),
            weight: window(d[0]) * window(d[1]),
        }
    })
}

pub fn bilinear(extent: Extent, values: &[f64], uv: [f64; 2], boundary: Boundary) -> f64 {
    assert_eq!(values.len(), (extent.width * extent.height) as usize);
    let p = [
        uv[0] * f64::from(extent.width) - 0.5,
        uv[1] * f64::from(extent.height) - 0.5,
    ];
    let base = p.map(|v| v.floor() as i64);
    let t = [p[0] - p[0].floor(), p[1] - p[1].floor()];
    let index = |v: i64, size: u32| match boundary {
        Boundary::Repeat => v.rem_euclid(i64::from(size)),
        Boundary::Clamp => v.clamp(0, i64::from(size) - 1),
    } as usize;
    let mut result = 0.0;
    for y in 0..2 {
        for x in 0..2 {
            let weight =
                if x == 0 { 1.0 - t[0] } else { t[0] } * if y == 0 { 1.0 - t[1] } else { t[1] };
            result += weight
                * values[index(base[1] + y, extent.height) * extent.width as usize
                    + index(base[0] + x, extent.width)];
        }
    }
    result
}

pub fn at(
    extent: Extent,
    values: &[f64],
    uv: [f64; 2],
    seed: u32,
    grid: u32,
    overlap: f64,
    boundary: Boundary,
) -> f64 {
    let taps = samples(uv, seed, grid, overlap);
    let coverage: f64 = taps.iter().map(|tap| tap.weight).sum();
    taps.iter()
        .map(|tap| tap.weight * bilinear(extent, values, tap.uv, boundary))
        .sum::<f64>()
        / coverage.max(1e-8)
}

pub fn render(
    source: Extent,
    output: Extent,
    values: &[f64],
    seed: u32,
    grid: u32,
    overlap: f64,
    boundary: Boundary,
) -> Vec<f64> {
    (0..output.width * output.height)
        .map(|i| {
            at(
                source,
                values,
                [
                    (f64::from(i % output.width) + 0.5) / f64::from(output.width),
                    (f64::from(i / output.width) + 0.5) / f64::from(output.height),
                ],
                seed,
                grid,
                overlap,
                boundary,
            )
        })
        .collect()
}

pub fn fixture(extent: Extent, kind: &str) -> Vec<f32> {
    let mut state = 58u32;
    (0..extent.width * extent.height)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (state >> 8) as f32 / 16777215.0;
            let x = (i % extent.width) as f32 / extent.width as f32;
            let y = (i / extent.width) as f32 / extent.height as f32;
            match kind {
                "constant" => 0.375,
                "ramp" => -0.25 + x + 0.5 * y,
                "impulse" => {
                    if i == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "discontinuous_noise" => 0.1 * noise + 0.5 * x + 0.3 * y,
                "structured" => {
                    if (0.2..0.3).contains(&x) {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => noise,
            }
        })
        .collect()
}

/// Seven differences spanning four pixels on each side of the wrap, compared
/// with an equal-size interior band. No duplicated-edge assumption.
pub fn seam_rms(e: Extent, values: &[f64]) -> (f64, f64) {
    assert!(e.width >= 16 && e.height >= 16);
    assert_eq!(values.len(), (e.width * e.height) as usize);
    let mut sums = [0.0; 2];
    let mut count = 0;
    for axis in 0..2 {
        let (length, rows) = if axis == 0 {
            (e.width, e.height)
        } else {
            (e.height, e.width)
        };
        let get = |x, row| {
            values[if axis == 0 {
                (row * e.width + x) as usize
            } else {
                (x * e.width + row) as usize
            }]
        };
        for row in 0..rows {
            for delta in -3i64..=3 {
                for (i, center) in [0, length / 2].into_iter().enumerate() {
                    let x = (i64::from(center) + delta).rem_euclid(i64::from(length)) as u32;
                    sums[i] += (get(x, row) - get((x + length - 1) % length, row)).powi(2);
                }
                count += 1;
            }
        }
    }
    sums.map(|sum| (sum / f64::from(count)).sqrt()).into()
}

pub fn linear(code: u8) -> f64 {
    let v = f64::from(code) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub fn half(value: f64) -> f64 {
    if value == 0.0 {
        return value;
    }
    let step = 2.0f64.powi((value.abs().log2().floor() as i32 - 10).max(-24));
    (value / step).round_ties_even() * step
}

#[test]
fn oracle_bilinear_and_hash_have_closed_form_anchors() {
    let e = Extent::new(2, 1).unwrap();
    assert_eq!(bilinear(e, &[2.0, 6.0], [0.25, 0.5], Boundary::Repeat), 2.0);
    assert_eq!(bilinear(e, &[2.0, 6.0], [0.0, 0.5], Boundary::Repeat), 4.0);
    assert_eq!(bilinear(e, &[2.0, 6.0], [0.0, 0.5], Boundary::Clamp), 2.0);
    assert_eq!(assignment(0, [0, 0], 4), ([0.0, 0.0], 0));
    assert_eq!(
        assignment(58, [0, 0], 4),
        ([0.38238525390625, 0.200225830078125], 0)
    );
    assert_eq!(
        assignment(58, [1, 2], 4),
        ([0.0079345703125, 0.4483642578125], 3)
    );
    assert_eq!(assignment(58, [-1, 4], 4), assignment(58, [3, 0], 4));
}

#[test]
fn oracle_coverage_periodicity_rotation_variety_and_convexity() {
    let e = Extent::new(17, 11).unwrap();
    let input: Vec<_> = fixture(e, "seeded").into_iter().map(f64::from).collect();
    let mut rotations = [false; 4];
    for seed in [0, 58, u32::MAX] {
        for grid in 2..=8 {
            for y in 0..grid {
                for x in 0..grid {
                    rotations[assignment(seed, [x as i32, y as i32], grid).1 as usize] = true;
                }
            }
            for overlap in [0.1, 0.25, 0.5] {
                for y in 0..17 {
                    for x in 0..19 {
                        let uv = [f64::from(x) / 19.0, f64::from(y) / 17.0];
                        let taps = samples(uv, seed, grid, overlap);
                        assert_eq!(taps.len(), MAX_NEIGHBORS);
                        assert!(taps.iter().all(|t| t.weight >= 0.0 && t.weight.is_finite()));
                        assert!(taps.iter().map(|t| t.weight).sum::<f64>() > 1e-8);
                        let a = at(e, &input, uv, seed, grid, overlap, Boundary::Repeat);
                        let b = at(
                            e,
                            &input,
                            [uv[0] - 2.0, uv[1] + 3.0],
                            seed,
                            grid,
                            overlap,
                            Boundary::Repeat,
                        );
                        assert!((a - b).abs() < 1e-12);
                        assert!((0.0..=1.0).contains(&a));
                    }
                }
            }
        }
    }
    assert!(rotations.into_iter().all(|present| present));
}

#[test]
fn oracle_fixed_fixture_meets_seam_and_detail_contract() {
    for (w, h) in [(256, 256), (257, 193)] {
        let e = Extent::new(w, h).unwrap();
        let input: Vec<_> = fixture(e, "discontinuous_noise")
            .into_iter()
            .map(f64::from)
            .collect();
        let result = render(e, e, &input, 58, 4, 0.25, Boundary::Repeat);
        let (before, interior_before) = seam_rms(e, &input);
        let (after, interior_after) = seam_rms(e, &result);
        assert!(before > 0.01 && interior_before > 0.01);
        assert!(after <= before * 0.5, "oracle boundary {before} -> {after}");
        assert!(
            interior_after >= interior_before * 0.5,
            "oracle interior {interior_before} -> {interior_after}"
        );
    }
}
