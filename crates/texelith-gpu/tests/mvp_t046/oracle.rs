//! Independent f64 pixel-footprint integrals and pixel-center interpolation.
//! Test-only reference, never a CPU execution fallback.
use texelith_core::{Boundary, Extent};

#[derive(Clone, Debug)]
pub struct Image {
    pub extent: Extent,
    pub pixels: Vec<f64>,
}

impl Image {
    pub fn new(w: u32, h: u32, pixels: Vec<f64>) -> Self {
        assert_eq!(pixels.len(), (w * h) as usize);
        Self {
            extent: Extent::new(w, h).unwrap(),
            pixels,
        }
    }

    pub fn load(&self, x: i64, y: i64, boundary: Boundary) -> f64 {
        let address = |v: i64, size: u32| match boundary {
            Boundary::Clamp => v.clamp(0, i64::from(size) - 1),
            Boundary::Repeat => v.rem_euclid(i64::from(size)),
        };
        self.pixels[(address(y, self.extent.height) * i64::from(self.extent.width)
            + address(x, self.extent.width)) as usize]
    }

    pub fn gaussian(&self, vertical: bool, boundary: Boundary) -> Self {
        let mut out = self.clone();
        for y in 0..self.extent.height {
            for x in 0..self.extent.width {
                let value = [1.0, 4.0, 6.0, 4.0, 1.0]
                    .iter()
                    .enumerate()
                    .map(|(i, weight)| {
                        let offset = i as i64 - 2;
                        weight
                            * self.load(
                                i64::from(x) + if vertical { 0 } else { offset },
                                i64::from(y) + if vertical { offset } else { 0 },
                                boundary,
                            )
                    })
                    .sum::<f64>()
                    / 16.0;
                out.pixels[(y * self.extent.width + x) as usize] = value;
            }
        }
        out
    }

    pub fn reduce(&self) -> Self {
        let w = self.extent.width.div_ceil(2);
        let h = self.extent.height.div_ceil(2);
        let mut pixels = Vec::new();
        // Integrate overlap with unit-width source pixel cells. This formulation
        // deliberately does not assume a 2x2 footprint for odd dimensions.
        for y in 0..h {
            let top = f64::from(y) * f64::from(self.extent.height) / f64::from(h);
            let bottom = f64::from(y + 1) * f64::from(self.extent.height) / f64::from(h);
            for x in 0..w {
                let left = f64::from(x) * f64::from(self.extent.width) / f64::from(w);
                let right = f64::from(x + 1) * f64::from(self.extent.width) / f64::from(w);
                let mut sum = 0.0;
                for sy in top.floor() as u32..bottom.ceil() as u32 {
                    for sx in left.floor() as u32..right.ceil() as u32 {
                        let overlap_x =
                            (right.min(f64::from(sx + 1)) - left.max(f64::from(sx))).max(0.0);
                        let overlap_y =
                            (bottom.min(f64::from(sy + 1)) - top.max(f64::from(sy))).max(0.0);
                        if overlap_x > 0.0 && overlap_y > 0.0 {
                            sum += overlap_x
                                * overlap_y
                                * self.pixels[(sy * self.extent.width + sx) as usize];
                        }
                    }
                }
                pixels.push(sum / ((right - left) * (bottom - top)));
            }
        }
        Self::new(w, h, pixels)
    }

    pub fn expand(&self, target: Extent, boundary: Boundary) -> Self {
        let mut pixels = Vec::new();
        for y in 0..target.height {
            let sy = (f64::from(y) + 0.5) * f64::from(self.extent.height)
                / f64::from(target.height)
                - 0.5;
            for x in 0..target.width {
                let sx = (f64::from(x) + 0.5) * f64::from(self.extent.width)
                    / f64::from(target.width)
                    - 0.5;
                let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
                let (fx, fy) = (sx - sx.floor(), sy - sy.floor());
                let top =
                    self.load(ix, iy, boundary) * (1.0 - fx) + self.load(ix + 1, iy, boundary) * fx;
                let bottom = self.load(ix, iy + 1, boundary) * (1.0 - fx)
                    + self.load(ix + 1, iy + 1, boundary) * fx;
                pixels.push(top * (1.0 - fy) + bottom * fy);
            }
        }
        Self {
            extent: target,
            pixels,
        }
    }
}

pub fn compare(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (f64::from(actual) - expected).abs() <= tolerance,
            "pixel {index}: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn reference_has_hand_calculated_odd_area_and_center_mapping_anchors() {
    let ramp = Image::new(5, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    compare(&[0.4, 2.0, 3.6], &ramp.reduce().pixels, 2e-7);
    assert_eq!(
        ramp.gaussian(false, Boundary::Repeat).pixels[0],
        25.0 / 16.0
    );
    assert_eq!(ramp.gaussian(false, Boundary::Clamp).pixels[0], 6.0 / 16.0);
    let coarse = Image::new(3, 1, vec![0.0, 10.0, 20.0]);
    compare(
        &[0.0, 4.0, 10.0, 16.0, 20.0],
        &coarse.expand(ramp.extent, Boundary::Clamp).pixels,
        1e-12,
    );
    compare(
        &[4.0, 4.0, 10.0, 16.0, 16.0],
        &coarse.expand(ramp.extent, Boundary::Repeat).pixels,
        1e-12,
    );
    assert_eq!(ramp.load(-11, 999, Boundary::Repeat), 4.0);
}
