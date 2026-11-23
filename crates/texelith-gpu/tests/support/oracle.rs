use texelith_core::{Extent, InputInterpretation};

pub const HEIGHT_TOLERANCE: f64 = 2e-5;
pub const NORMAL_TOLERANCE: f64 = 2e-3;
pub const LENGTH_TOLERANCE: f64 = 2e-3;
pub const DISPLAY_CODE_TOLERANCE: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct Height {
    pub extent: Extent,
    pub pixels: Vec<f64>,
}

impl Height {
    pub fn new(extent: Extent, pixels: Vec<f64>) -> Self {
        extent.validate().unwrap();
        assert_eq!(pixels.len(), extent.width as usize * extent.height as usize);
        assert!(pixels.iter().all(|p| p.is_finite()));
        Self { extent, pixels }
    }

    pub fn generate(extent: Extent, mut f: impl FnMut(u32, u32) -> f64) -> Self {
        let mut pixels = Vec::with_capacity(extent.width as usize * extent.height as usize);
        for y in 0..extent.height {
            for x in 0..extent.width {
                pixels.push(f(x, y));
            }
        }
        Self::new(extent, pixels)
    }

    pub fn at(&self, x: i64, y: i64) -> f64 {
        let x = x.clamp(0, i64::from(self.extent.width) - 1) as usize;
        let y = y.clamp(0, i64::from(self.extent.height) - 1) as usize;
        self.pixels[y * self.extent.width as usize + x]
    }
}

pub fn srgb_inverse(code: f64) -> f64 {
    if code <= 0.04045 {
        code / 12.92
    } else {
        ((code + 0.055) / 1.055).powf(2.4)
    }
}

pub fn prepare(rgba: &[u8], extent: Extent, interpretation: InputInterpretation) -> Height {
    assert_eq!(
        rgba.len(),
        extent.width as usize * extent.height as usize * 4
    );
    let pixels = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| match interpretation {
            InputInterpretation::LinearHeight => f64::from(p[0]) / 255.0,
            InputInterpretation::SrgbPhoto => [0.2126, 0.7152, 0.0722]
                .iter()
                .zip(p)
                .map(|(w, c)| w * srgb_inverse(f64::from(*c) / 255.0))
                .sum(),
            InputInterpretation::LinearNormal => {
                panic!("legacy height oracle does not support normal inputs")
            }
        })
        .collect();
    Height::new(extent, pixels)
}

pub fn levels(input: &Height, black: f64, white: f64, gamma: f64) -> Height {
    assert!(white - black >= 1e-6 && gamma > 0.0);
    Height::new(
        input.extent,
        input
            .pixels
            .iter()
            .map(|h| {
                ((h - black) / (white - black))
                    .clamp(0.0, 1.0)
                    .powf(1.0 / gamma)
            })
            .collect(),
    )
}

pub fn gaussian(sigma_axis: f64) -> Vec<f64> {
    assert!(sigma_axis.is_finite() && sigma_axis >= 0.0);
    if sigma_axis == 0.0 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma_axis).ceil().min(15.0) as i32;
    let mut weights: Vec<_> = (-radius..=radius)
        .map(|i| (-0.5 * (f64::from(i) / sigma_axis).powi(2)).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

pub fn blur_axis(input: &Height, sigma_axis: f64, vertical: bool) -> Height {
    let weights = gaussian(sigma_axis);
    let radius = weights.len() as i64 / 2;
    Height::generate(input.extent, |x, y| {
        weights
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let d = i as i64 - radius;
                let (dx, dy) = if vertical { (0, d) } else { (d, 0) };
                w * input.at(i64::from(x) + dx, i64::from(y) + dy)
            })
            .sum()
    })
}

pub fn blur(input: &Height, source: Extent, sigma_source_px: f64) -> Height {
    let x = sigma_source_px * f64::from(input.extent.width) / f64::from(source.width);
    let y = sigma_source_px * f64::from(input.extent.height) / f64::from(source.height);
    blur_axis(&blur_axis(input, x, false), y, true)
}

pub fn normal(input: &Height, strength: f64) -> Vec<[f64; 4]> {
    let mut normals = Vec::with_capacity(input.pixels.len());
    for y in 0..input.extent.height {
        for x in 0..input.extent.width {
            let (x, y) = (i64::from(x), i64::from(y));
            let du =
                (input.at(x + 1, y) - input.at(x - 1, y)) * f64::from(input.extent.width) / 2.0;
            let dv =
                (input.at(x, y + 1) - input.at(x, y - 1)) * f64::from(input.extent.height) / 2.0;
            let n = [-strength * du, strength * dv, 1.0];
            let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
            normals.push([n[0] / length, n[1] / length, n[2] / length, 1.0]);
        }
    }
    normals
}

/// Floor the short axis after aspect scaling, bounded to at least one texel.
pub fn proxy_extent(source: Extent) -> Extent {
    source.validate().unwrap();
    let longest = source.width.max(source.height);
    if longest <= 1024 {
        return source;
    }
    Extent::new(
        ((u64::from(source.width) * 1024) / u64::from(longest)).max(1) as u32,
        ((u64::from(source.height) * 1024) / u64::from(longest)).max(1) as u32,
    )
    .unwrap()
}

/// Integrate piecewise-constant source texel areas, including fractional edges.
pub fn downsample(input: &Height, output: Extent) -> Height {
    assert!(output.width <= input.extent.width && output.height <= input.extent.height);
    let sx = f64::from(input.extent.width) / f64::from(output.width);
    let sy = f64::from(input.extent.height) / f64::from(output.height);
    Height::generate(output, |x, y| {
        let (left, right) = (f64::from(x) * sx, f64::from(x + 1) * sx);
        let (top, bottom) = (f64::from(y) * sy, f64::from(y + 1) * sy);
        let mut sum = 0.0;
        for iy in top.floor() as i64..bottom.ceil() as i64 {
            for ix in left.floor() as i64..right.ceil() as i64 {
                let wx = (right.min((ix + 1) as f64) - left.max(ix as f64)).max(0.0);
                let wy = (bottom.min((iy + 1) as f64) - top.max(iy as f64)).max(0.0);
                sum += input.at(ix, iy) * wx * wy;
            }
        }
        sum / (sx * sy)
    })
}

pub fn code(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn display_height(input: &Height) -> Vec<[u8; 4]> {
    input
        .pixels
        .iter()
        .map(|&h| {
            let c = code(h);
            [c, c, c, 255]
        })
        .collect()
}

pub fn display_normal(input: &[[f64; 4]]) -> Vec<[u8; 4]> {
    input
        .iter()
        .map(|n| {
            [
                code(n[0] * 0.5 + 0.5),
                code(n[1] * 0.5 + 0.5),
                code(n[2] * 0.5 + 0.5),
                255,
            ]
        })
        .collect()
}

/// Diagnostic seam score; kernels still clamp and do not promise to tile photos.
pub fn wrapped_edge_score(input: &Height) -> f64 {
    let mut score: f64 = 0.0;
    for y in 0..input.extent.height {
        score = score.max(
            (input.at(0, i64::from(y)) - input.at(i64::from(input.extent.width) - 1, i64::from(y)))
                .abs(),
        );
    }
    for x in 0..input.extent.width {
        score = score.max(
            (input.at(i64::from(x), 0)
                - input.at(i64::from(x), i64::from(input.extent.height) - 1))
            .abs(),
        );
    }
    score
}

pub fn compare(actual: &[f64], expected: &[f64], tolerance: f64) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!("length {} != {}", actual.len(), expected.len()));
    }
    let mut max: f64 = 0.0;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || error > tolerance {
            return Err(format!(
                "sample {i}: {a} != {e}, error {error}, tolerance {tolerance}"
            ));
        }
        max = max.max(error);
    }
    Ok(max)
}

#[derive(Clone)]
pub struct Fixture {
    pub name: &'static str,
    pub height: Height,
}

pub fn fixtures() -> Vec<Fixture> {
    let extent = Extent::new(17, 11).unwrap();
    vec![
        Fixture {
            name: "constant_npot",
            height: Height::generate(extent, |_, _| 0.375),
        },
        Fixture {
            name: "uv_ramp_npot",
            height: Height::generate(extent, |x, y| {
                0.1 + 0.25 * (f64::from(x) + 0.5) / 17.0 + 0.125 * (f64::from(y) + 0.5) / 11.0
            }),
        },
        Fixture {
            name: "impulse",
            height: Height::generate(extent, |x, y| f64::from(x == 8 && y == 5)),
        },
        Fixture {
            name: "corner_impulse",
            height: Height::generate(extent, |x, y| f64::from(x == 0 && y == 0)),
        },
        Fixture {
            name: "one_by_n",
            height: Height::generate(Extent::new(1, 13).unwrap(), |_, y| f64::from(y) / 12.0),
        },
        Fixture {
            name: "n_by_one",
            height: Height::generate(Extent::new(13, 1).unwrap(), |x, _| f64::from(x) / 12.0),
        },
        Fixture {
            name: "singleton",
            height: Height::generate(Extent::new(1, 1).unwrap(), |_, _| 0.5),
        },
        Fixture {
            name: "clamp_and_display_range",
            height: Height::new(
                Extent::new(7, 1).unwrap(),
                vec![-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0],
            ),
        },
        Fixture {
            name: "tileable_symmetric",
            height: Height::generate(extent, |x, y| {
                0.5 + 0.125 * (std::f64::consts::TAU * f64::from(x) / 16.0).cos()
                    + 0.125 * (std::f64::consts::TAU * f64::from(y) / 10.0).cos()
            }),
        },
    ]
}
