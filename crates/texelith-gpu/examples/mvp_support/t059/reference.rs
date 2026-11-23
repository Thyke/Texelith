//! Independent f64 oracle and generated fixtures, never a production fallback.
//! The discretization proposal is recorded in docs/evidence/T-059/test-contract.md.
use std::sync::OnceLock;
use texelith_core::{Boundary, Extent, SeamlessBase};

pub const BINS: usize = 1024;
pub const CLIP: f64 = 1.0 / 4096.0;
pub const EPSILON: f64 = 1e-8;
pub const RATIO_LIMIT: f64 = 16.0;
pub const VARIANCE_FLOOR: f64 = 1e-6;

pub fn luminance(rgb: [f64; 3]) -> f64 {
    // Integer Rec.709 coefficients sum exactly before division. In particular,
    // binary16 gray on a bin edge must not slip into the preceding bin because
    // the separately rounded decimal coefficients sum to less than one.
    (rgb[0] * 2126.0 + rgb[1] * 7152.0 + rgb[2] * 722.0) / 10000.0
}

// Numerical integration supplies an original, fixed Gaussian CDF table on
// [-8,8] at 1/512 steps. Forward/inverse use the SAME piecewise-linear table.
fn gaussian() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut cdf = vec![0.0; 8193];
        let density = |i: usize| (-0.5 * (i as f64 / 512.0 - 8.0).powi(2)).exp();
        for i in 1..cdf.len() {
            cdf[i] = cdf[i - 1] + (density(i - 1) + density(i)) / 1024.0;
        }
        let total = cdf[8192];
        for p in &mut cdf {
            *p /= total;
        }
        cdf
    })
}
pub fn to_gaussian(probability: f64) -> f64 {
    let p = probability.clamp(CLIP, 1.0 - CLIP);
    let cdf = gaussian();
    let hi = cdf.partition_point(|v| *v < p).clamp(1, 8192);
    let t = (p - cdf[hi - 1]) / (cdf[hi] - cdf[hi - 1]);
    (hi as f64 - 1.0 + t) / 512.0 - 8.0
}
pub fn from_gaussian(z: f64) -> f64 {
    let x = ((z + 8.0) * 512.0).clamp(0.0, 8192.0);
    let lo = (x.floor() as usize).min(8191);
    let t = x - lo as f64;
    gaussian()[lo] * (1.0 - t) + gaussian()[lo + 1] * t
}

pub struct Histogram {
    pub counts: [u32; BINS],
    pub variance: f64,
    cdf: Vec<f64>,
}
impl Histogram {
    pub fn new(values: &[f64]) -> Self {
        assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
        let mut counts = [0; BINS];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        for &v in values {
            counts[bin(v)] += 1;
        }
        let mut cdf = vec![0.0; BINS + 1];
        let mut total = 0u32;
        for i in 0..BINS {
            total += counts[i];
            cdf[i + 1] = f64::from(total) / values.len() as f64;
        }
        Self {
            counts,
            variance,
            cdf,
        }
    }
    pub fn probability(&self, v: f64) -> f64 {
        let x = v.clamp(0.0, 1.0) * BINS as f64;
        let i = bin(v);
        self.cdf[i] + (x - i as f64) * (self.cdf[i + 1] - self.cdf[i])
    }
    pub fn inverse(&self, probability: f64) -> f64 {
        let p = probability.clamp(CLIP, 1.0 - CLIP);
        let hi = self.cdf.partition_point(|v| *v < p).clamp(1, BINS);
        let lo = hi - 1;
        let t = (p - self.cdf[lo]) / (self.cdf[hi] - self.cdf[lo]).max(EPSILON);
        (lo as f64 + t) / BINS as f64
    }
}
pub fn bin(v: f64) -> usize {
    ((v.clamp(0.0, 1.0) * BINS as f64) as usize).min(BINS - 1)
}
pub fn cdf_distance(a: &[f64], b: &[f64]) -> f64 {
    let a = Histogram::new(a);
    let b = Histogram::new(b);
    a.cdf
        .iter()
        .zip(b.cdf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy)]
pub struct Tap {
    pub uv: [f64; 2],
    pub weight: f64,
}
pub fn taps(uv: [f64; 2], base: &SeamlessBase) -> Vec<Tap> {
    let smooth = |t: f64| {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    match *base {
        SeamlessBase::OffsetBlend { overlap } => [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.0, 0.0]]
            .map(|shift| {
                let s = std::array::from_fn(|i| (uv[i] + shift[i]).rem_euclid(1.0));
                let w = s.map(|v| smooth(v.min(1.0 - v) / f64::from(overlap)));
                Tap {
                    uv: s,
                    weight: w[0] * w[1],
                }
            })
            .to_vec(),
        SeamlessBase::Splat {
            seed,
            grid,
            overlap,
        } => {
            let p = uv.map(|v| v.rem_euclid(1.0) * f64::from(grid));
            let cell = p.map(|v| v.floor() as i32);
            (0..9)
                .map(|i| {
                    let c = [cell[0] + i % 3 - 1, cell[1] + i / 3 - 1];
                    let d = [p[0] - f64::from(c[0]) - 0.5, p[1] - f64::from(c[1]) - 0.5];
                    let x = c[0].rem_euclid(grid as i32) as u32;
                    let y = c[1].rem_euclid(grid as i32) as u32;
                    let mut h = seed ^ x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
                    h ^= h >> 16;
                    h = h.wrapping_mul(0x7feb_352d);
                    h ^= h >> 15;
                    h = h.wrapping_mul(0x846c_a68b);
                    h ^= h >> 16;
                    let shift = [
                        f64::from(h & 0x7fff) / 32768.0,
                        f64::from((h >> 15) & 0x7fff) / 32768.0,
                    ];
                    let r = match h >> 30 {
                        0 => d,
                        1 => [-d[1], d[0]],
                        2 => [-d[0], -d[1]],
                        _ => [d[1], -d[0]],
                    };
                    let w = d.map(|v| smooth(1.0 - v.abs() / (1.0 + f64::from(overlap))).powi(4));
                    Tap {
                        uv: std::array::from_fn(|a| {
                            (shift[a] + r[a] / f64::from(grid)).rem_euclid(1.0)
                        }),
                        weight: w[0] * w[1],
                    }
                })
                .collect()
        }
    }
}
pub fn bilinear(e: Extent, values: &[[f64; 3]], uv: [f64; 2], boundary: Boundary) -> [f64; 3] {
    let p = [
        uv[0] * f64::from(e.width) - 0.5,
        uv[1] * f64::from(e.height) - 0.5,
    ];
    let q = p.map(|v| v.floor() as i64);
    let t = p.map(|v| v - v.floor());
    let index = |v: i64, n: u32| match boundary {
        Boundary::Repeat => v.rem_euclid(i64::from(n)),
        Boundary::Clamp => v.clamp(0, i64::from(n) - 1),
    } as usize;
    let mut rgb = [0.0; 3];
    for y in 0..2 {
        for x in 0..2 {
            let w = if x == 0 { 1.0 - t[0] } else { t[0] } * if y == 0 { 1.0 - t[1] } else { t[1] };
            let v = values[index(q[1] + y, e.height) * e.width as usize + index(q[0] + x, e.width)];
            for c in 0..3 {
                rgb[c] += w * v[c];
            }
        }
    }
    rgb
}
#[allow(clippy::too_many_arguments)]
pub fn at(
    e: Extent,
    values: &[[f64; 3]],
    histogram: &Histogram,
    uv: [f64; 2],
    base: &SeamlessBase,
    boundary: Boundary,
    strength: f64,
    color: bool,
) -> [f64; 3] {
    let taps = taps(uv, base);
    let total = taps.iter().map(|t| t.weight).sum::<f64>().max(EPSILON);
    let mut rgb = [0.0; 3];
    let mut latent = 0.0;
    let mut squares = 0.0;
    for tap in taps {
        let w = tap.weight / total;
        let v = bilinear(e, values, tap.uv, boundary);
        for c in 0..3 {
            rgb[c] += w * v[c];
        }
        let l = if color { luminance(v) } else { v[0] };
        latent += w * to_gaussian(histogram.probability(l));
        squares += w * w;
    }
    if strength == 0.0 || histogram.variance < VARIANCE_FLOOR {
        return rgb;
    }
    let restored = histogram.inverse(from_gaussian(latent / squares.sqrt().max(EPSILON)));
    let l = if color { luminance(rgb) } else { rgb[0] };
    let target = l + strength * (restored - l);
    if color {
        let ratio = (target / l.max(EPSILON)).clamp(0.0, RATIO_LIMIT);
        rgb.map(|c| (c * ratio).clamp(0.0, 1.0))
    } else {
        [target; 3]
    }
}
pub fn render(
    source: Extent,
    output: Extent,
    values: &[[f64; 3]],
    base: &SeamlessBase,
    boundary: Boundary,
    strength: f64,
    color: bool,
) -> Vec<[f64; 3]> {
    assert_eq!(values.len(), (source.width * source.height) as usize);
    let l: Vec<_> = values
        .iter()
        .map(|v| if color { luminance(*v) } else { v[0] })
        .collect();
    let hist = Histogram::new(&l);
    (0..output.width * output.height)
        .map(|i| {
            at(
                source,
                values,
                &hist,
                [
                    (f64::from(i % output.width) + 0.5) / f64::from(output.width),
                    (f64::from(i / output.width) + 0.5) / f64::from(output.height),
                ],
                base,
                boundary,
                strength,
                color,
            )
        })
        .collect()
}
fn random(x: u32, y: u32) -> f64 {
    let mut h = 59 ^ x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    f64::from(h >> 8) / 16777215.0
}
/// Smooth seeded value noise on a 64x64 lattice for the fixed 1024-square CDF gate.
/// Independent of output transforms, with no production or sibling fixture calls.
pub fn fixture(e: Extent, kind: &str) -> Vec<f32> {
    (0..e.width * e.height)
        .map(|i| {
            let x = i % e.width;
            let y = i / e.width;
            match kind {
                "constant" => 0.375,
                "ramp" => {
                    0.1 + 0.6 * (x as f32 + 0.5) / e.width as f32
                        + 0.2 * (y as f32 + 0.5) / e.height as f32
                }
                "impulse" => {
                    if i == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "noise" => {
                    let p = [
                        (f64::from(x) + 0.5) * 64.0 / f64::from(e.width),
                        (f64::from(y) + 0.5) * 64.0 / f64::from(e.height),
                    ];
                    let cell = p.map(|v| v.floor() as u32);
                    let t = p.map(|v| {
                        let t = v - v.floor();
                        t * t * (3.0 - 2.0 * t)
                    });
                    let a = random(cell[0], cell[1]) * (1.0 - t[0])
                        + random(cell[0] + 1, cell[1]) * t[0];
                    let b = random(cell[0], cell[1] + 1) * (1.0 - t[0])
                        + random(cell[0] + 1, cell[1] + 1) * t[0];
                    (0.05 + 0.9 * (a * (1.0 - t[1]) + b * t[1])) as f32
                }
                "seeded" => random(x, y) as f32,
                "discontinuous_noise" => {
                    (0.1 * random(x, y)
                        + 0.5 * f64::from(x) / f64::from(e.width)
                        + 0.3 * f64::from(y) / f64::from(e.height)) as f32
                }
                _ => panic!("unknown T-059 fixture: {kind}"),
            }
        })
        .collect()
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
    let step = 2.0f64.powi((value.abs().log2().floor() as i32 - 10).max(-24));
    (value / step).round_ties_even() * step
}

/// Four-pixel wrap band and an equally sized interior first-difference band.
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
    (
        (sums[0] / f64::from(count)).sqrt(),
        (sums[1] / f64::from(count)).sqrt(),
    )
}

#[test]
fn oracle_integer_bins_quantiles_and_variance_normalization_have_independent_anchors() {
    let h = Histogram::new(&[0.0, 0.25, 0.5, 1.0]);
    assert_eq!(
        (h.counts[0], h.counts[256], h.counts[512], h.counts[1023]),
        (1, 1, 1, 1)
    );
    assert_eq!(h.counts.iter().sum::<u32>(), 4);
    assert_eq!(h.probability(0.5), 0.5);
    assert!((h.inverse(0.625) - 0.50048828125).abs() < 1e-12);
    assert!(to_gaussian(0.5).abs() < 1e-10);
    assert!((from_gaussian(1.0) - 0.841344746).abs() < 2e-7);
    assert!((to_gaussian(0.841344746) - 1.0).abs() < 2e-6);
    for p in [CLIP, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - CLIP] {
        assert!((from_gaussian(to_gaussian(p)) - p).abs() < 1e-12);
    }
    // Two independent equal-weight unit normals need sqrt(.5), not sum(w)=1.
    let z = (1.0 * 0.5 + 1.0 * 0.5) / (0.5f64).sqrt();
    assert!((z - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((from_gaussian(z) - from_gaussian(1.0)).abs() > 0.05);
    let flat = Histogram::new(&[0.500001; 17]);
    assert!(flat.variance < VARIANCE_FLOOR);
}

#[test]
fn oracle_fixed_noise_meets_cdf_and_seam_thresholds_for_both_bases() {
    let noise = Extent::new(1024, 1024).unwrap();
    let source: Vec<_> = fixture(noise, "noise").into_iter().map(f64::from).collect();
    let rgb: Vec<_> = source.iter().map(|v| [*v; 3]).collect();
    let seam = Extent::new(257, 193).unwrap();
    let discontinuous: Vec<_> = fixture(seam, "discontinuous_noise")
        .into_iter()
        .map(f64::from)
        .collect();
    let seam_rgb: Vec<_> = discontinuous.iter().map(|v| [*v; 3]).collect();
    for base in [
        SeamlessBase::OffsetBlend { overlap: 0.25 },
        SeamlessBase::Splat {
            seed: 59,
            grid: 4,
            overlap: 0.25,
        },
    ] {
        let result = render(noise, noise, &rgb, &base, Boundary::Repeat, 1.0, false);
        let distance = cdf_distance(&source, &result.iter().map(|v| v[0]).collect::<Vec<_>>());
        eprintln!("T-059 CPU oracle {base:?} CDF distance {distance}");
        assert!(distance <= 0.05, "oracle CDF {base:?}: {distance}");
        let output = render(seam, seam, &seam_rgb, &base, Boundary::Repeat, 1.0, false);
        let (before, interior) = seam_rms(seam, &discontinuous);
        let (after, detail) = seam_rms(seam, &output.iter().map(|v| v[0]).collect::<Vec<_>>());
        eprintln!(
            "T-059 CPU oracle {base:?} boundary {before} -> {after}; interior {interior} -> {detail}"
        );
        assert!(
            before > 0.01 && interior > 0.01 && after <= before * 0.5 && detail >= interior * 0.5,
            "oracle seam/detail: boundary {before} -> {after}; interior {interior} -> {detail}"
        );
    }
}
