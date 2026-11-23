//! Original, std-only f64 reference. Sparse matrices assembled from incidence
//! edges and geometric overlap weights avoid sharing any production operators.
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub repeat: bool,
}
impl Grid {
    pub fn len(self) -> usize {
        self.w * self.h
    }
    pub fn sample(self, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        (0..self.len())
            .map(|i| {
                f(
                    (i % self.w) as f64 / self.w as f64,
                    (i / self.w) as f64 / self.h as f64,
                )
            })
            .collect()
    }
    pub fn gradient(self, h: &[f64]) -> [Vec<f64>; 2] {
        assert_eq!(h.len(), self.len());
        let mut g = [vec![0.0; self.len()], vec![0.0; self.len()]];
        self.edges(|i, j, axis, scale| g[axis][i] = scale * (h[j] - h[i]));
        g
    }
    pub fn transpose(self, g: &[Vec<f64>; 2]) -> Vec<f64> {
        assert!(g.iter().all(|v| v.len() == self.len()));
        let mut b = vec![0.0; self.len()];
        self.edges(|i, j, axis, scale| {
            b[i] -= scale * g[axis][i];
            b[j] += scale * g[axis][i];
        });
        b
    }
    pub fn apply(self, h: &[f64]) -> Vec<f64> {
        self.transpose(&self.gradient(h))
    }
    fn edges(self, mut visit: impl FnMut(usize, usize, usize, f64)) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                if self.w > 1 && (self.repeat || x + 1 < self.w) {
                    visit(i, y * self.w + (x + 1) % self.w, 0, self.w as f64);
                }
                if self.h > 1 && (self.repeat || y + 1 < self.h) {
                    visit(i, ((y + 1) % self.h) * self.w + x, 1, self.h as f64);
                }
            }
        }
    }
}
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
pub fn project(x: &mut [f64]) {
    let m = mean(x);
    for v in x {
        *v -= m;
    }
}
pub fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}
pub fn rms(a: &[f64], b: &[f64]) -> f64 {
    norm(&sub(a, b)) / (a.len() as f64).sqrt()
}
pub fn residual(grid: Grid, h: &[f64], rhs: &[f64]) -> f64 {
    norm(&sub(rhs, &grid.apply(h)))
}
pub fn analytic(grid: Grid) -> Vec<f64> {
    let k = if grid.repeat {
        std::f64::consts::TAU
    } else {
        std::f64::consts::PI
    };
    let mut h = grid.sample(|u, v| {
        0.04 * (k * (u + 0.5 / grid.w as f64)).cos()
            + 0.025 * (2.0 * k * (v + 0.5 / grid.h as f64)).cos()
    });
    project(&mut h);
    h
}
pub fn seeded(n: usize) -> Vec<f64> {
    let mut s = 50u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (f64::from(s >> 16) / 32768.0 - 1.0) * 0.025
        })
        .collect()
}
pub fn normals(g: &[Vec<f64>; 2], dx: bool) -> Vec<[f64; 3]> {
    g[0].iter()
        .zip(&g[1])
        .map(|(&u, &v)| {
            let length = (u * u + v * v + 1.0).sqrt();
            [
                -u / length,
                (if dx { -v } else { v }) / length,
                1.0 / length,
            ]
        })
        .collect()
}
pub fn slopes(n: &[[f64; 3]], dx: bool) -> ([Vec<f64>; 2], u32) {
    let mut g = [Vec::new(), Vec::new()];
    let mut count = 0;
    for &[x, y, z] in n {
        let u = -x / z.max(0.05);
        let v = (if dx { -y } else { y }) / z.max(0.05);
        count += u32::from(z < 0.05 || u.abs() > 20.0 || v.abs() > 20.0);
        g[0].push(u.clamp(-20.0, 20.0));
        g[1].push(v.clamp(-20.0, 20.0));
    }
    (g, count)
}

type Matrix = Vec<Vec<(usize, f64)>>;
fn multiply(a: &Matrix, x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().map(|&(j, v)| v * x[j]).sum())
        .collect()
}
fn product(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .map(|row| {
            let mut result = BTreeMap::<usize, f64>::new();
            for &(k, weight) in row {
                for &(j, v) in &b[k] {
                    *result.entry(j).or_default() += weight * v;
                }
            }
            result
                .into_iter()
                .filter(|(_, v)| v.abs() > 1e-13)
                .collect()
        })
        .collect()
}
fn transfers(f: Grid, c: Grid) -> (Matrix, Matrix) {
    let mut r = vec![Vec::new(); c.len()];
    for cy in 0..c.h {
        for cx in 0..c.w {
            let (l, t) = (
                cx as f64 * f.w as f64 / c.w as f64,
                cy as f64 * f.h as f64 / c.h as f64,
            );
            let (rr, b) = (
                (cx + 1) as f64 * f.w as f64 / c.w as f64,
                (cy + 1) as f64 * f.h as f64 / c.h as f64,
            );
            for y in t.floor() as usize..b.ceil() as usize {
                for x in l.floor() as usize..rr.ceil() as usize {
                    let area = (rr.min((x + 1) as f64) - l.max(x as f64)).max(0.0)
                        * (b.min((y + 1) as f64) - t.max(y as f64)).max(0.0);
                    if area > 0.0 {
                        r[cy * c.w + cx].push((y * f.w + x, area / ((rr - l) * (b - t))));
                    }
                }
            }
        }
    }
    let address = |p: isize, n: usize| {
        if f.repeat {
            p.rem_euclid(n as isize) as usize
        } else {
            p.clamp(0, n as isize - 1) as usize
        }
    };
    let mut p = vec![Vec::new(); f.len()];
    for y in 0..f.h {
        for x in 0..f.w {
            let sx = (x as f64 + 0.5) * c.w as f64 / f.w as f64 - 0.5;
            let sy = (y as f64 + 0.5) * c.h as f64 / f.h as f64 - 0.5;
            for (dy, wy) in [
                (0, 1.0 - sy.fract().rem_euclid(1.0)),
                (1, sy.fract().rem_euclid(1.0)),
            ] {
                for (dx, wx) in [
                    (0, 1.0 - sx.fract().rem_euclid(1.0)),
                    (1, sx.fract().rem_euclid(1.0)),
                ] {
                    p[y * f.w + x].push((
                        address(sy.floor() as isize + dy, c.h) * c.w
                            + address(sx.floor() as isize + dx, c.w),
                        wx * wy,
                    ));
                }
            }
        }
    }
    (r, p)
}
struct Level {
    a: Matrix,
    diagonal: Vec<f64>,
    restriction: Matrix,
    prolongation: Matrix,
}
pub struct Multigrid {
    levels: Vec<Level>,
}
impl Multigrid {
    pub fn new(mut grid: Grid) -> Self {
        let mut rows = vec![BTreeMap::<usize, f64>::new(); grid.len()];
        grid.edges(|i, j, _, s| {
            *rows[i].entry(i).or_default() += s * s;
            *rows[j].entry(j).or_default() += s * s;
            *rows[i].entry(j).or_default() -= s * s;
            *rows[j].entry(i).or_default() -= s * s;
        });
        let mut a: Matrix = rows.into_iter().map(|r| r.into_iter().collect()).collect();
        let mut levels = Vec::new();
        loop {
            let diagonal = a
                .iter()
                .enumerate()
                .map(|(i, r)| r.iter().filter(|(j, _)| *j == i).map(|(_, v)| *v).sum())
                .collect();
            if grid.len() == 1 {
                levels.push(Level {
                    a,
                    diagonal,
                    restriction: vec![],
                    prolongation: vec![],
                });
                break;
            }
            let coarse = Grid {
                w: grid.w.div_ceil(2),
                h: grid.h.div_ceil(2),
                ..grid
            };
            let (r, p) = transfers(grid, coarse);
            let next = product(&product(&r, &a), &p);
            levels.push(Level {
                a,
                diagonal,
                restriction: r,
                prolongation: p,
            });
            a = next;
            grid = coarse;
        }
        Self { levels }
    }
    fn smooth(level: &Level, h: &mut [f64], b: &[f64], count: u32) {
        for _ in 0..count {
            let ah = multiply(&level.a, h);
            for i in 0..h.len() {
                if level.diagonal[i].abs() > 1e-12 {
                    h[i] += (2.0 / 3.0) * (b[i] - ah[i]) / level.diagonal[i];
                }
            }
        }
    }
    fn cycle(&self, l: usize, h: &mut [f64], b: &[f64]) {
        let level = &self.levels[l];
        if l + 1 == self.levels.len() {
            Self::smooth(level, h, b, 32);
            return;
        }
        Self::smooth(level, h, b, 3);
        let residual = sub(b, &multiply(&level.a, h));
        let mut cb = multiply(&level.restriction, &residual);
        project(&mut cb);
        let mut correction = vec![0.0; cb.len()];
        self.cycle(l + 1, &mut correction, &cb);
        let correction = multiply(&level.prolongation, &correction);
        for (x, c) in h.iter_mut().zip(correction) {
            *x += c;
        }
        Self::smooth(level, h, b, 3);
    }
    pub fn solve(&self, rhs: &[f64], cycles: u32) -> (Vec<f64>, Vec<f64>) {
        let mut b = rhs.to_vec();
        project(&mut b);
        let mut h = vec![0.0; b.len()];
        let mut residuals = Vec::new();
        for _ in 0..cycles {
            self.cycle(0, &mut h, &b);
            project(&mut h);
            residuals.push(norm(&sub(&b, &multiply(&self.levels[0].a, &h))));
        }
        (h, residuals)
    }
}

/// Independent least-squares solution for arbitrary slope fields on tiny
/// periodic fixtures, using the exact discrete Fourier eigenvalues of D^T D.
pub fn periodic_spectral(grid: Grid, rhs: &[f64]) -> Vec<f64> {
    assert!(grid.repeat && grid.len() <= 1024);
    let mut h = vec![0.0; grid.len()];
    for ky in 0..grid.h {
        for kx in 0..grid.w {
            if kx == 0 && ky == 0 {
                continue;
            }
            let lambda = 4.0
                * ((grid.w as f64 * (std::f64::consts::PI * kx as f64 / grid.w as f64).sin())
                    .powi(2)
                    + (grid.h as f64 * (std::f64::consts::PI * ky as f64 / grid.h as f64).sin())
                        .powi(2));
            let phase = |i: usize| {
                std::f64::consts::TAU
                    * (kx as f64 * (i % grid.w) as f64 / grid.w as f64
                        + ky as f64 * (i / grid.w) as f64 / grid.h as f64)
            };
            let re = rhs
                .iter()
                .enumerate()
                .map(|(i, v)| v * phase(i).cos())
                .sum::<f64>()
                / grid.len() as f64
                / lambda;
            let im = rhs
                .iter()
                .enumerate()
                .map(|(i, v)| -v * phase(i).sin())
                .sum::<f64>()
                / grid.len() as f64
                / lambda;
            for (i, v) in h.iter_mut().enumerate() {
                *v += re * phase(i).cos() - im * phase(i).sin();
            }
        }
    }
    project(&mut h);
    h
}

/// Scalable f64 oracle, independent of the GPU's split/permuted FFT. Checked
/// against the direct DFT above. No shader arithmetic or twiddle tables shared.
pub fn periodic_fft(grid: Grid, rhs: &[f64]) -> Vec<f64> {
    assert!(grid.repeat && grid.w.is_power_of_two() && grid.h.is_power_of_two());
    fn fft(a: &mut [[f64; 2]], twiddles: &[[f64; 2]], inverse: bool) {
        let n = a.len();
        if n == 1 {
            return;
        }
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - n.ilog2());
            if j > i {
                a.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= n {
            for chunk in a.chunks_exact_mut(size) {
                for k in 0..size / 2 {
                    let [c, mut s] = twiddles[k * n / size];
                    if inverse {
                        s = -s;
                    }
                    let [x, y] = chunk[k + size / 2];
                    let b = [c * x - s * y, c * y + s * x];
                    let a = chunk[k];
                    chunk[k] = [a[0] + b[0], a[1] + b[1]];
                    chunk[k + size / 2] = [a[0] - b[0], a[1] - b[1]];
                }
            }
            size *= 2;
        }
        if inverse {
            for a in a {
                a[0] /= n as f64;
                a[1] /= n as f64;
            }
        }
    }
    let twiddles = |n: usize| {
        (0..n / 2)
            .map(|k| {
                let (s, c) = (-std::f64::consts::TAU * k as f64 / n as f64).sin_cos();
                [c, s]
            })
            .collect::<Vec<_>>()
    };
    let tx = twiddles(grid.w);
    let ty = twiddles(grid.h);
    let mut data: Vec<_> = rhs.iter().map(|b| [*b, 0.0]).collect();
    let mut column = vec![[0.0; 2]; grid.h];
    for inverse in [false, true] {
        for row in data.chunks_exact_mut(grid.w) {
            fft(row, &tx, inverse);
        }
        for x in 0..grid.w {
            for y in 0..grid.h {
                column[y] = data[y * grid.w + x];
            }
            fft(&mut column, &ty, inverse);
            for y in 0..grid.h {
                data[y * grid.w + x] = column[y];
            }
        }
        if !inverse {
            for (i, v) in data.iter_mut().enumerate() {
                let x = i % grid.w;
                let y = i / grid.w;
                if i == 0 {
                    *v = [0.0; 2];
                    continue;
                }
                let eigenvalue = 4.0
                    * ((grid.w as f64 * (std::f64::consts::PI * x as f64 / grid.w as f64).sin())
                        .powi(2)
                        + (grid.h as f64
                            * (std::f64::consts::PI * y as f64 / grid.h as f64).sin())
                        .powi(2));
                v[0] /= eigenvalue;
                v[1] /= eigenvalue;
            }
        }
    }
    let mut height: Vec<_> = data.into_iter().map(|v| v[0]).collect();
    project(&mut height);
    height
}

/// ||D h - g||, excluding nonexistent Neumann edges.
pub fn objective_residual(grid: Grid, height: &[f64], field: &[Vec<f64>; 2]) -> f64 {
    let mut sum = 0.0;
    grid.edges(|i, j, axis, scale| {
        sum += (scale * (height[j] - height[i]) - field[axis][i]).powi(2);
    });
    sum.sqrt()
}

/// sqrt(F(h)-F(h*)) without subtracting two nearly equal objective values.
pub fn optimality_gap(grid: Grid, height: &[f64], optimum: &[f64]) -> f64 {
    let mut sum = 0.0;
    grid.edges(|i, j, _, scale| {
        sum += (scale * ((height[j] - optimum[j]) - (height[i] - optimum[i]))).powi(2);
    });
    sum.sqrt()
}

pub fn field_norm(field: &[Vec<f64>; 2]) -> f64 {
    field.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod anchors {
    use super::*;
    #[test]
    fn fft_oracle_matches_direct_dft_and_objective_gap_rejects_wrong_height() {
        for (w, h) in [(1, 1), (1, 8), (16, 1), (16, 8), (8, 8)] {
            let g = Grid { w, h, repeat: true };
            let field = [seeded(g.len()), g.sample(|u, v| 0.2 * (9.0 * u + v).sin())];
            let rhs = g.transpose(&field);
            let exact = periodic_spectral(g, &rhs);
            let fast = periodic_fft(g, &rhs);
            assert!(rms(&exact, &fast) < 1e-14);
            let wrong = g.sample(|u, _| 0.1 * (std::f64::consts::TAU * u).cos());
            let optimum_residual = objective_residual(g, &fast, &field);
            let wrong_residual = objective_residual(g, &wrong, &field);
            let gap = optimality_gap(g, &wrong, &fast);
            assert!(
                (wrong_residual.powi(2) - optimum_residual.powi(2) - gap.powi(2)).abs() < 1e-10
            );
            if w > 1 {
                assert!(gap > 1e-3 * field_norm(&field));
            }
        }
    }
    #[test]
    fn hand_computed_incidence_and_constant_nullspace() {
        let g = Grid {
            w: 3,
            h: 1,
            repeat: true,
        };
        assert_eq!(g.apply(&[1.0, 2.0, 4.0]), vec![-36.0, -9.0, 45.0]);
        assert_eq!(
            Grid { repeat: false, ..g }.apply(&[1.0, 2.0, 4.0]),
            vec![-9.0, -9.0, 18.0]
        );
        assert_eq!(g.apply(&[17.0; 3]), vec![0.0; 3]);
    }
    #[test]
    fn cpu_multigrid_recovers_periodic_modes_and_removes_offset_each_cycle() {
        for (w, h) in [(16, 16), (32, 16), (17, 13)] {
            let g = Grid { w, h, repeat: true };
            let expected = analytic(g);
            let rhs = g.apply(&expected);
            let (actual, trace) = Multigrid::new(g).solve(&rhs, 8);
            assert!(
                rms(&actual, &expected) < 1e-3,
                "{w}x{h}: {}",
                rms(&actual, &expected)
            );
            assert!(
                trace.last().unwrap() / norm(&rhs) < 1e-3,
                "{w}x{h}: {trace:?}"
            );
            assert!(mean(&actual).abs() < 1e-14);
        }
    }
    #[test]
    fn spectral_solution_anchors_multigrid_for_nonintegrable_periodic_field() {
        let g = Grid {
            w: 16,
            h: 8,
            repeat: true,
        };
        let gradients = [
            seeded(g.len()),
            g.sample(|u, v| 0.15 * (7.0 * u + 3.0 * v).sin()),
        ];
        let rhs = g.transpose(&gradients);
        let expected = periodic_spectral(g, &rhs);
        assert!(residual(g, &expected, &rhs) / norm(&rhs) < 1e-12);
        let (actual, _) = Multigrid::new(g).solve(&rhs, 8);
        assert!(rms(&actual, &expected) < 2e-5);
        // Nonintegrable input has a remaining gradient error even at convergence.
        assert!(norm(&sub(&g.gradient(&expected)[0], &gradients[0])) > 1e-3);
    }
    #[test]
    fn boundary_and_slope_clamp_anchors() {
        let g = Grid {
            w: 3,
            h: 1,
            repeat: false,
        };
        assert_eq!(
            g.transpose(&[vec![1.0, 2.0, 999.0], vec![99.0; 3]]),
            vec![-3.0, -3.0, 6.0]
        );
        let (slopes, count) = slopes(&[[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]], false);
        assert_eq!(slopes, [vec![0.0, -20.0], vec![0.0, -20.0]]);
        assert_eq!(count, 1);
    }
}
