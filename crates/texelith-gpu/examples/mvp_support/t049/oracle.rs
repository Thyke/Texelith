//! Independent f64 finite-volume reference; no production imports or GPU code.
//! Kept std-only so its anchors can run before the proposed GPU API exists.

#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f64>,
}

impl Image {
    pub fn new(w: usize, h: usize, v: Vec<f64>) -> Self {
        assert!(w > 0 && h > 0 && v.len() == w * h);
        assert!(v.iter().all(|x| x.is_finite()));
        Self { w, h, v }
    }

    pub fn constant(w: usize, h: usize, value: f64) -> Self {
        Self::new(w, h, vec![value; w * h])
    }

    pub fn seeded(w: usize, h: usize, seed: u32, scale: f64) -> Self {
        let mut state = seed;
        let v = (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                // Dyadic samples survive f32 upload exactly; this is not a GPU oracle.
                (f64::from(state >> 16) / 32768.0 - 1.0) * scale
            })
            .collect();
        Self::new(w, h, v)
    }

    pub fn mean(&self) -> f64 {
        self.v.iter().sum::<f64>() / self.v.len() as f64
    }

    pub fn norm(&self) -> f64 {
        self.v.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn project(&self) -> Self {
        let mean = self.mean();
        Self::new(self.w, self.h, self.v.iter().map(|v| v - mean).collect())
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!((self.w, self.h), (other.w, other.h));
        Self::new(
            self.w,
            self.h,
            self.v.iter().zip(&other.v).map(|(a, b)| a - b).collect(),
        )
    }

    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!((self.w, self.h), (other.w, other.h));
        self.v.iter().zip(&other.v).map(|(a, b)| a * b).sum()
    }

    /// D has one oriented edge per pixel/axis. Missing Neumann edges are zero.
    pub fn difference(&self, repeat: bool, vertical: bool) -> Self {
        let mut out = Self::constant(self.w, self.h, 0.0);
        for y in 0..self.h {
            for x in 0..self.w {
                let (n, p) = if vertical { (self.h, y) } else { (self.w, x) };
                if n == 1 || (!repeat && p == n - 1) {
                    continue;
                }
                let j = if vertical {
                    ((y + 1) % self.h) * self.w + x
                } else {
                    y * self.w + (x + 1) % self.w
                };
                let i = y * self.w + x;
                out.v[i] = n as f64 * (self.v[j] - self.v[i]);
            }
        }
        out
    }

    /// Scatter each edge's two incidence coefficients. This independently
    /// anchors signs and ignores arbitrary data stored on outward Neumann edges.
    pub fn transpose(u: &Self, v: &Self, repeat: bool) -> Self {
        assert_eq!((u.w, u.h), (v.w, v.h));
        let mut out = Self::constant(u.w, u.h, 0.0);
        for y in 0..u.h {
            for x in 0..u.w {
                let i = y * u.w + x;
                if u.w > 1 && (repeat || x + 1 < u.w) {
                    let flux = u.w as f64 * u.v[i];
                    out.v[i] -= flux;
                    out.v[y * u.w + (x + 1) % u.w] += flux;
                }
                if u.h > 1 && (repeat || y + 1 < u.h) {
                    let flux = u.h as f64 * v.v[i];
                    out.v[i] -= flux;
                    out.v[((y + 1) % u.h) * u.w + x] += flux;
                }
            }
        }
        out
    }

    /// A = D^T D is positive semidefinite, i.e. minus the usual Laplacian.
    pub fn apply(&self, repeat: bool) -> Self {
        Self::transpose(
            &self.difference(repeat, false),
            &self.difference(repeat, true),
            repeat,
        )
    }

    /// Exact box-overlap area averages; no Gaussian prefilter for Poisson R.
    pub fn restrict(&self) -> Self {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let mut out = Self::constant(w, h, 0.0);
        for y in 0..h {
            let (top, bottom) = (
                y as f64 * self.h as f64 / h as f64,
                (y + 1) as f64 * self.h as f64 / h as f64,
            );
            for x in 0..w {
                let (left, right) = (
                    x as f64 * self.w as f64 / w as f64,
                    (x + 1) as f64 * self.w as f64 / w as f64,
                );
                for sy in top.floor() as usize..bottom.ceil() as usize {
                    for sx in left.floor() as usize..right.ceil() as usize {
                        let area = (right.min((sx + 1) as f64) - left.max(sx as f64)).max(0.0)
                            * (bottom.min((sy + 1) as f64) - top.max(sy as f64)).max(0.0);
                        out.v[y * w + x] +=
                            self.v[sy * self.w + sx] * area / ((right - left) * (bottom - top));
                    }
                }
            }
        }
        out
    }

    pub fn prolong(&self, w: usize, h: usize, repeat: bool) -> Self {
        assert_eq!((self.w, self.h), (w.div_ceil(2), h.div_ceil(2)));
        let index = |p: isize, n: usize| {
            if repeat {
                p.rem_euclid(n as isize) as usize
            } else {
                p.clamp(0, n as isize - 1) as usize
            }
        };
        let mut out = Self::constant(w, h, 0.0);
        for y in 0..h {
            let sy = (y as f64 + 0.5) * self.h as f64 / h as f64 - 0.5;
            for x in 0..w {
                let sx = (x as f64 + 0.5) * self.w as f64 / w as f64 - 0.5;
                let (ix, iy) = (sx.floor() as isize, sy.floor() as isize);
                let (fx, fy) = (sx - sx.floor(), sy - sy.floor());
                for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
                    for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
                        out.v[y * w + x] += wx
                            * wy
                            * self.v[index(iy + dy, self.h) * self.w + index(ix + dx, self.w)];
                    }
                }
            }
        }
        out
    }
}

/// `levels` starts at the finest extent and ends at the current coarse extent.
pub fn galerkin(x: &Image, levels: &[(usize, usize)], repeat: bool) -> Image {
    assert_eq!(levels.last().copied(), Some((x.w, x.h)));
    if levels.len() == 1 {
        return x.apply(repeat);
    }
    let (w, h) = levels[levels.len() - 2];
    galerkin(
        &x.prolong(w, h, repeat),
        &levels[..levels.len() - 1],
        repeat,
    )
    .restrict()
}

/// Basis-column diagonal is intentionally expensive and independent of a stencil.
pub fn jacobi(
    x: &Image,
    rhs: &Image,
    levels: &[(usize, usize)],
    repeat: bool,
    omega: f64,
) -> Image {
    let ax = galerkin(x, levels, repeat);
    let mut out = x.clone();
    for i in 0..x.v.len() {
        let mut basis = Image::constant(x.w, x.h, 0.0);
        basis.v[i] = 1.0;
        let diagonal = galerkin(&basis, levels, repeat).v[i];
        out.v[i] = if diagonal == 0.0 {
            x.v[i]
        } else {
            x.v[i] + omega * (rhs.v[i] - ax.v[i]) / diagonal
        };
    }
    out
}

/// True if either the denominator floor or either slope cap altered the pixel.
/// Count a pixel once even if both components need clamping.
pub fn normal_gradients(normals: &[[f64; 3]], direct_x: bool) -> (Vec<f64>, Vec<f64>, usize) {
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut count = 0;
    for &[x, y, z] in normals {
        let denominator = z.max(0.05);
        let du = -x / denominator;
        let dv = (if direct_x { -y } else { y }) / denominator;
        count += usize::from(z < 0.05 || du.abs() > 20.0 || dv.abs() > 20.0);
        u.push(du.clamp(-20.0, 20.0));
        v.push(dv.clamp(-20.0, 20.0));
    }
    (u, v, count)
}

#[cfg(test)]
mod anchors {
    use super::*;
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-12),
            "{a:?} != {b:?}"
        );
    }

    #[test]
    fn hand_computed_edge_sign_scaling_and_neumann_flux() {
        let h = Image::new(3, 1, vec![1.0, 2.0, 4.0]);
        close(&h.difference(false, false).v, &[3.0, 6.0, 0.0]);
        close(&h.difference(true, false).v, &[3.0, 6.0, -9.0]);
        close(&h.apply(false).v, &[-9.0, -9.0, 18.0]);
        close(&h.apply(true).v, &[-36.0, -9.0, 45.0]);
        let flux = Image::new(3, 1, vec![2.0, 3.0, 999.0]);
        close(&Image::transpose(&flux, &flux, false).v, &[-6.0, -3.0, 9.0]);
        close(&h.difference(false, true).v, &[0.0; 3]);
    }

    #[test]
    fn hand_computed_odd_area_and_pixel_center_weights() {
        let ramp = Image::new(5, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        close(&ramp.restrict().v, &[0.4, 2.0, 3.6]);
        let coarse = Image::new(3, 1, vec![0.0, 10.0, 20.0]);
        close(
            &coarse.prolong(5, 1, false).v,
            &[0.0, 4.0, 10.0, 16.0, 20.0],
        );
        close(&coarse.prolong(5, 1, true).v, &[4.0, 4.0, 10.0, 16.0, 16.0]);
        close(
            &galerkin(&Image::new(2, 1, vec![1.0, 0.0]), &[(3, 1), (2, 1)], false).v,
            &[3.0, -3.0],
        );
    }

    #[test]
    fn adjoint_nullspace_energy_and_recursive_coarsening() {
        for repeat in [false, true] {
            let x = Image::seeded(17, 13, 49, 1.0);
            let u = Image::seeded(17, 13, 51, 1.0);
            let v = Image::seeded(17, 13, 53, 1.0);
            let lhs = x.difference(repeat, false).dot(&u) + x.difference(repeat, true).dot(&v);
            let rhs = x.dot(&Image::transpose(&u, &v, repeat));
            assert!((lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0));
            assert!(x.dot(&x.apply(repeat)) > 0.0);
            let levels = [(17, 13), (9, 7), (5, 4)];
            close(
                &galerkin(&Image::constant(5, 4, 3.0), &levels, repeat).v,
                &[0.0; 20],
            );
        }
    }

    #[test]
    fn reduction_jacobi_and_normal_clamp_anchors() {
        let x = Image::new(3, 1, vec![1.0, 2.0, 3.0]);
        close(&x.project().v, &[-1.0, 0.0, 1.0]);
        assert_eq!(x.norm(), 14.0f64.sqrt());
        let impulse = Image::new(3, 1, vec![1.0, 0.0, 0.0]);
        close(
            &jacobi(
                &impulse,
                &Image::constant(3, 1, 0.0),
                &[(3, 1)],
                false,
                2.0 / 3.0,
            )
            .v,
            &[1.0 / 3.0, 1.0 / 3.0, 0.0],
        );
        let (u, v, count) = normal_gradients(
            &[
                [0.0, 0.0, 1.0],
                [0.5, -0.25, 0.5],
                [1.0, 1.0, 0.0],
                [-2.0, -2.0, -1.0],
            ],
            false,
        );
        close(&u, &[0.0, -1.0, -20.0, 20.0]);
        close(&v, &[0.0, -0.5, 20.0, -20.0]);
        assert_eq!(count, 2);
    }

    #[test]
    fn npot_galerkin_fixtures_detect_even_stencil_substitution() {
        for repeat in [false, true] {
            let mut levels: Vec<(usize, usize)> = vec![(257, 193)];
            for _ in 0..3 {
                let (w, h) = *levels.last().unwrap();
                levels.push((w.div_ceil(2), h.div_ceil(2)));
                let (w, h) = *levels.last().unwrap();
                let x = Image::seeded(w, h, 49, 2.0f64.powi(-20));
                let rap = galerkin(&x, &levels, repeat);
                assert!(rap.norm() > 1e-5);
                assert!(rap.sub(&x.apply(repeat)).norm() > 1e-4);
            }
        }
    }
}
