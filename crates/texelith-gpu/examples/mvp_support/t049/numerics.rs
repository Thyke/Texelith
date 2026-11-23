use super::{
    Result,
    gpu::{Gpu, extent},
    oracle::{self, Image},
    require,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};
use texelith_core::{Boundary, Extent, NormalConvention};
use texelith_gpu::{Axis, PoissonGrid, PoissonOperators, PoissonPass as P, pool::ResourcePool};

#[derive(Default)]
pub struct Numerical {
    pub checks: Vec<Value>,
    pub scalar: f64,
    pub galerkin: f64,
    pub adjoint: f64,
}
fn compare(actual: &Image, expected: &Image) -> Result<f64> {
    require(
        (actual.w, actual.h) == (expected.w, expected.h),
        "oracle extent mismatch",
    )?;
    let mut max = 0.0f64;
    for (a, b) in actual.v.iter().zip(&expected.v) {
        require(a.is_finite() && b.is_finite(), "nonfinite oracle result")?;
        max = max.max((a - b).abs());
    }
    require(max <= 2e-5, &format!("Poisson oracle error {max}"))?;
    Ok(max)
}
pub fn run(g: &Gpu) -> Result<Numerical> {
    let mut result = Numerical::default();
    let mut checked = BTreeSet::new();
    let single = Extent::new(1, 1).unwrap();
    for (w, h) in [(1, 1), (1, 17), (17, 1), (31, 17), (257, 193)] {
        for boundary in [Boundary::Repeat, Boundary::Clamp] {
            let repeat = boundary == Boundary::Repeat;
            let e = Extent::new(w as u32, h as u32).unwrap();
            let grid = PoissonGrid::new(e, boundary).unwrap();
            let x = Image::seeded(w, h, 49, 1.0);
            let u = Image::seeded(w, h, 51, 1.0);
            let v = Image::seeded(w, h, 53, 1.0);
            let (xt, ut, vt) = (g.scalar(&x), g.scalar(&u), g.scalar(&v));
            let dx = g.run(&grid, P::Difference(Axis::X), &[&xt], e);
            let dy = g.run(&grid, P::Difference(Axis::Y), &[&xt], e);
            let adjoint = g.read(&g.run(&grid, P::Transpose, &[&ut, &vt], e));
            let lhs = g.read(&dx).dot(&u) + g.read(&dy).dot(&v);
            let rhs = x.dot(&adjoint);
            let relative = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-12);
            require(relative <= 2e-5, "adjoint identity failed")?;
            result.adjoint = result.adjoint.max(relative);
            checked.insert("adjoint");
            let mut impulse = Image::constant(w, h, 0.0);
            impulse.v[0] = 2.0f64.powi(-20);
            let ramp = Image::new(
                w,
                h,
                (0..w * h)
                    .map(|i| (i % w + 2 * (i / w)) as f64 * 2.0f64.powi(-20))
                    .collect(),
            );
            for (kind, x) in [
                ("impulse", impulse),
                ("ramp", ramp),
                ("seeded", Image::seeded(w, h, 49, 2.0f64.powi(-20))),
                ("constant_nullspace", Image::constant(w, h, -2.25)),
            ] {
                let input = g.scalar(&x);
                for (axis, vertical) in [(Axis::X, false), (Axis::Y, true)] {
                    result.scalar = result.scalar.max(compare(
                        &g.read(&g.run(&grid, P::Difference(axis), &[&input], e)),
                        &x.difference(repeat, vertical),
                    )?);
                }
                let a = g.run(&grid, P::Laplacian, &[&input], e);
                result.scalar = result.scalar.max(compare(&g.read(&a), &x.apply(repeat))?);
                if kind == "seeded" {
                    let b = g.run(&grid, P::Laplacian, &[&input], e);
                    require(
                        g.read_bytes(&a) == g.read_bytes(&b),
                        "nondeterministic operator",
                    )?;
                    checked.insert("determinism");
                    let zero = g.scalar(&Image::constant(w, h, 0.0));
                    let residual = g.run(&grid, P::Residual, &[&input, &zero], e);
                    let expected = Image::constant(w, h, 0.0).sub(&x.apply(repeat));
                    result.scalar = result.scalar.max(compare(&g.read(&residual), &expected)?);
                    let norm = g.read(&g.run(&grid, P::L2Norm, &[&residual], single)).v[0];
                    require(
                        (norm - expected.norm()).abs() <= 2e-5f64.max(expected.norm() * 2e-5),
                        "residual L2 mismatch",
                    )?;
                    checked.insert("residual_l2");
                } else {
                    checked.insert(kind);
                }
            }
            if w == 1 || h == 1 {
                checked.insert("singleton_axes");
            }
            let mean = g.run(&grid, P::Mean, &[&xt], single);
            result.scalar = result
                .scalar
                .max(compare(&g.read(&mean), &Image::constant(1, 1, x.mean()))?);
            let projected = g.read(&g.run(&grid, P::ProjectMean, &[&xt, &mean], e));
            result.scalar = result.scalar.max(compare(&projected, &x.project())?);
            require(projected.mean().abs() <= 2e-5, "mean projection failed")?;
            checked.insert("mean_zero");
            let restricted = x.restrict();
            let r = g.run(&grid, P::Restrict, &[&xt], extent(&restricted));
            result.scalar = result.scalar.max(compare(&g.read(&r), &restricted)?);
            result.scalar = result.scalar.max(compare(
                &g.read(&g.run(&grid, P::Prolong, &[&r], e)),
                &restricted.prolong(w, h, repeat),
            )?);
            checked.extend(["restriction", "prolongation"]);
            if !repeat {
                let mut u = Image::constant(w, h, 0.0);
                let mut v = u.clone();
                for y in 0..h {
                    u.v[y * w + w - 1] = 123.0;
                }
                for x in 0..w {
                    v.v[(h - 1) * w + x] = -456.0;
                }
                let (ut, vt) = (g.scalar(&u), g.scalar(&v));
                result.scalar = result.scalar.max(compare(
                    &g.read(&g.run(&grid, P::Transpose, &[&ut, &vt], e)),
                    &Image::constant(w, h, 0.0),
                )?);
                checked.insert("neumann_no_outward_flux");
            } else {
                for vertical in [false, true] {
                    let n = if vertical { h } else { w };
                    let image = Image::new(
                        w,
                        h,
                        (0..w * h)
                            .map(|i| {
                                let p = if vertical { i / w } else { i % w };
                                (std::f64::consts::TAU * p as f64 / n as f64).sin()
                                    / (w * w + h * h) as f64
                            })
                            .collect(),
                    );
                    let eigenvalue =
                        4.0 * (n * n) as f64 * (std::f64::consts::PI / n as f64).sin().powi(2);
                    let expected =
                        Image::new(w, h, image.v.iter().map(|v| v * eigenvalue).collect());
                    let input = g.scalar(&image);
                    result.scalar = result.scalar.max(compare(
                        &g.read(&g.run(&grid, P::Laplacian, &[&input], e)),
                        &expected,
                    )?);
                }
                checked.insert("periodic_sinusoids");
            }
        }
    }
    for boundary in [Boundary::Repeat, Boundary::Clamp] {
        let mut grid = PoissonGrid::new(Extent::new(257, 193).unwrap(), boundary).unwrap();
        let mut levels = vec![(257usize, 193usize)];
        for _ in 0..3 {
            grid = grid.coarsen().unwrap();
            let e = grid.extent();
            levels.push((e.width as usize, e.height as usize));
            let x = Image::seeded(e.width as usize, e.height as usize, 49, 2.0f64.powi(-20));
            let input = g.scalar(&x);
            result.galerkin = result.galerkin.max(compare(
                &g.read(&g.run(&grid, P::Laplacian, &[&input], e)),
                &oracle::galerkin(&x, &levels, boundary == Boundary::Repeat),
            )?);
        }
        let mut grid = PoissonGrid::new(Extent::new(7, 5).unwrap(), boundary).unwrap();
        let mut levels = vec![(7usize, 5usize)];
        loop {
            let e = grid.extent();
            let (w, h) = (e.width as usize, e.height as usize);
            let x = Image::seeded(w, h, 49, 0.25);
            let rhs = Image::seeded(w, h, 51, 0.25);
            let (xt, bt) = (g.scalar(&x), g.scalar(&rhs));
            result.scalar = result.scalar.max(compare(
                &g.read(&g.run(&grid, P::Jacobi { omega: 2.0 / 3.0 }, &[&xt, &bt], e)),
                &oracle::jacobi(&x, &rhs, &levels, boundary == Boundary::Repeat, 2.0 / 3.0),
            )?);
            if w == 1 && h == 1 {
                break;
            }
            grid = grid.coarsen().unwrap();
            levels.push((w.div_ceil(2), h.div_ceil(2)));
        }
    }
    checked.extend(["galerkin_257x193", "weighted_jacobi"]);
    let e = Extent::new(5, 1).unwrap();
    let grid = PoissonGrid::new(e, Boundary::Repeat).unwrap();
    let normal = g.texture(e, wgpu::TextureFormat::Rgba16Float);
    let halves: [[u16; 4]; 5] = [
        [0, 0, 0x3c00, 0x3c00],
        [0x3c00, 0, 0, 0x3c00],
        [0, 0x3c00, 0, 0x3c00],
        [0, 0, 0xbc00, 0x3c00],
        [0x3800, 0x3800, 0x39a8, 0x3c00],
    ];
    let bytes: Vec<u8> = halves
        .into_iter()
        .flatten()
        .flat_map(u16::to_le_bytes)
        .collect();
    g.upload(&normal, &bytes, 8);
    let normals = vec![
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.5, 0.70703125],
    ];
    for convention in [NormalConvention::OpenGl, NormalConvention::DirectX] {
        let (u, v, count) =
            oracle::normal_gradients(&normals, convention == NormalConvention::DirectX);
        for (axis, expected) in [(Axis::X, u), (Axis::Y, v)] {
            result.scalar = result.scalar.max(compare(
                &g.read(&g.run(&grid, P::NormalGradient { axis, convention }, &[&normal], e)),
                &Image::new(5, 1, expected),
            )?);
        }
        require(
            g.read(&g.run(&grid, P::ClampCount, &[&normal], single)).v[0] == count as f64,
            "clamp count mismatch",
        )?;
    }
    checked.extend([
        "normal_gradient",
        "steep_normal_clamp_count",
        "normal_convention_once",
    ]);
    let input = g.scalar(&Image::seeded(5, 1, 49, 0.25));
    let output = g.scalar(&Image::constant(5, 1, -19.0));
    let before = g.read_bytes(&output);
    let pool = Arc::new(Mutex::new(ResourcePool::new(super::SCRATCH)));
    let operators =
        PoissonOperators::new(g.device.clone(), pool.clone()).map_err(|e| e.to_string())?;
    let pass = operators
        .prepare(&grid, P::Laplacian, &[&input], &output)
        .map_err(|e| e.to_string())?;
    g.wait();
    require(g.read_bytes(&output) == before, "prepare submitted work")?;
    drop(pass);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "cancelled scratch leaked",
    )?;
    let low = PoissonOperators::new(g.device.clone(), Arc::new(Mutex::new(ResourcePool::new(0))))
        .map_err(|e| e.to_string())?;
    let error = low
        .prepare(&grid, P::Laplacian, &[&input], &output)
        .err()
        .ok_or("budget not enforced")?;
    require(
        error.to_string().to_lowercase().contains("budget"),
        "wrong low budget rejection",
    )?;
    require(g.read_bytes(&output) == before, "cancelled output mutated")?;
    checked.extend(["cancellation", "low_budget"]);
    // The callback owns the reservation after the caller drops its pass. The
    // callback fires only when polling observes real GPU completion.
    let pass = operators
        .prepare(&grid, P::Laplacian, &[&input], &output)
        .map_err(|e| e.to_string())?;
    let mut encoder = g.device.create_command_encoder(&Default::default());
    pass.encode(&mut encoder);
    g.queue.submit([encoder.finish()]);
    pass.retain_until_completed(&g.queue);
    drop(pass);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes > 0,
        "submitted scratch retired early",
    )?;
    g.wait();
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "completed scratch leaked",
    )?;
    for name in super::report_contract::CHECKS {
        require(
            checked.contains(name),
            &format!("missing measured check {name}"),
        )?;
        result.checks.push(json!({"name":name,"passed":true}));
    }
    Ok(result)
}
