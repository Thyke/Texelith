use crate::GpuError;
use std::collections::BTreeMap;
use texelith_core::{Boundary, Extent};

type Row = BTreeMap<usize, f64>;
type Stencil = Vec<Row>;

/// A separable Galerkin grid. The two one-dimensional operator pairs retain
/// the original UV metric through any number of ceil-half coarsenings.
#[derive(Clone, Debug)]
pub struct PoissonGrid {
    pub(super) extent: Extent,
    pub(super) boundary: Boundary,
    pub(super) depth: usize,
    axes: [(Stencil, Stencil); 2],
}

impl PoissonGrid {
    pub fn new(extent: Extent, boundary: Boundary) -> Result<Self, GpuError> {
        Extent::new(extent.width, extent.height)?;
        let axes = [extent.width, extent.height].map(|n| {
            let n = n as usize;
            let mut a = vec![Row::new(); n];
            let mass = (0..n).map(|i| Row::from([(i, 1.0)])).collect();
            if n > 1 {
                for i in 0..n {
                    if boundary == Boundary::Repeat || i + 1 < n {
                        let j = (i + 1) % n;
                        let s = (n * n) as f64;
                        *a[i].entry(i).or_default() += s;
                        *a[j].entry(j).or_default() += s;
                        *a[i].entry(j).or_default() -= s;
                        *a[j].entry(i).or_default() -= s;
                    }
                }
            }
            (a, mass)
        });
        Ok(Self {
            extent,
            boundary,
            depth: 0,
            axes,
        })
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn coarsen(&self) -> Result<Self, GpuError> {
        if self.extent.width == 1 && self.extent.height == 1 {
            return Err(GpuError("cannot coarsen terminal Poisson grid".into()));
        }
        let axes = self.axes.each_ref().map(|(a, m)| {
            let n = a.len();
            let c = n.div_ceil(2);
            let r: Stencil = (0..c)
                .map(|i| {
                    // Integer overlap numerators avoid cancellation on odd grids.
                    let left = i * n;
                    let right = (i + 1) * n;
                    (left / c..right.div_ceil(c))
                        .map(|j| {
                            (
                                j,
                                (right.min((j + 1) * c) - left.max(j * c)) as f64 / n as f64,
                            )
                        })
                        .collect()
                })
                .collect();
            let p: Stencil = (0..n)
                .map(|i| {
                    let coordinate = (i as f64 + 0.5) * c as f64 / n as f64 - 0.5;
                    let lower = coordinate.floor() as isize;
                    let frac = coordinate - coordinate.floor();
                    let mut row = Row::new();
                    for (j, weight) in [(lower, 1.0 - frac), (lower + 1, frac)] {
                        let j = if self.boundary == Boundary::Repeat {
                            j.rem_euclid(c as isize)
                        } else {
                            j.clamp(0, c as isize - 1)
                        } as usize;
                        *row.entry(j).or_default() += weight;
                    }
                    row
                })
                .collect();
            (
                multiply(&multiply(&r, a), &p),
                multiply(&multiply(&r, m), &p),
            )
        });
        Ok(Self {
            extent: Extent::new(
                self.extent.width.div_ceil(2),
                self.extent.height.div_ceil(2),
            )?,
            boundary: self.boundary,
            depth: self.depth + 1,
            axes,
        })
    }

    // A = Ax tensor My + Mx tensor Ay. Only compact 1D stencils are stored;
    // no 2D matrix is assembled. Difference form makes the nullspace exact.
    pub(super) fn packed(&self) -> Vec<[f32; 4]> {
        let rows = (self.extent.width + self.extent.height) as usize;
        let mut data = vec![[0.0; 4]; rows];
        let mut row = 0;
        for (a, m) in &self.axes {
            for i in 0..a.len() {
                let mut entries = BTreeMap::<usize, (f64, f64)>::new();
                for (&j, &v) in &a[i] {
                    entries.entry(j).or_default().0 = v;
                }
                for (&j, &v) in &m[i] {
                    entries.entry(j).or_default().1 = v;
                }
                entries.remove(&i);
                let offset = data.len();
                let mut da = 0.0;
                let mut dm = 1.0;
                for (j, (av, mv)) in entries {
                    if av != 0.0 || mv != 0.0 {
                        da -= av;
                        dm -= mv;
                        data.push([j as f32, av as f32, mv as f32, 0.0]);
                    }
                }
                data[row] = [
                    offset as f32,
                    (data.len() - offset) as f32,
                    da as f32,
                    dm as f32,
                ];
                row += 1;
            }
        }
        data
    }
}

fn multiply(a: &Stencil, b: &Stencil) -> Stencil {
    a.iter()
        .map(|row| {
            let mut out = Row::new();
            for (&j, &v) in row {
                for (&k, &w) in &b[j] {
                    *out.entry(k).or_default() += v * w;
                }
            }
            out
        })
        .collect()
}
