//! Compiled when seamless_splat is present in the single node registry.
//! Proposed consumer API follows T-057's public reusable transform interface.
use super::reference;
use texelith_core::{Boundary, Extent};
use texelith_gpu::seamless_splat::SplatTransform;

#[test]
fn public_transform_validates_parameters_and_bounds_work_independently_of_texture_size() {
    for grid in [0, 1, 9, u32::MAX] {
        assert!(SplatTransform::new(58, grid, 0.25).is_err());
    }
    for overlap in [f32::NAN, f32::INFINITY, -0.1, 0.099, 0.501] {
        assert!(SplatTransform::new(58, 4, overlap).is_err());
    }
    for grid in 2..=8 {
        for overlap in [0.1, 0.25, 0.5] {
            let t = SplatTransform::new(u32::MAX, grid, overlap).unwrap();
            for y in 0..=32 {
                for x in 0..=32 {
                    let samples = t.samples([x as f32 / 32.0, y as f32 / 32.0]);
                    assert!(!samples.is_empty() && samples.len() <= reference::MAX_NEIGHBORS);
                    assert!(samples.iter().all(|s| s.weight.is_finite()
                        && s.weight >= 0.0
                        && s.uv.iter().all(|v| v.is_finite() && (0.0..1.0).contains(v))));
                    assert!(
                        samples.iter().map(|s| s.weight).sum::<f32>() > 1e-8,
                        "coverage hole: grid={grid}, overlap={overlap}, x={x}, y={y}"
                    );
                }
            }
        }
    }
}

#[test]
fn public_metadata_matches_oracle_is_periodic_and_can_be_reused_for_correlated_layers() {
    let e = Extent::new(17, 11).unwrap();
    let input: Vec<_> = reference::fixture(e, "seeded")
        .into_iter()
        .map(f64::from)
        .collect();
    let correlated: Vec<_> = input.iter().map(|v| 2.0 * v - 0.5).collect();
    for seed in [0, 58, u32::MAX] {
        for grid in [2, 4, 8] {
            for overlap in [0.1f32, 0.25, 0.5] {
                let t = SplatTransform::new(seed, grid, overlap).unwrap();
                let sample = |uv, values: &[f64]| {
                    let taps = t.samples(uv);
                    let total: f64 = taps.iter().map(|s| f64::from(s.weight)).sum();
                    taps.iter()
                        .map(|s| {
                            f64::from(s.weight)
                                * reference::bilinear(
                                    e,
                                    values,
                                    s.uv.map(f64::from),
                                    Boundary::Repeat,
                                )
                        })
                        .sum::<f64>()
                        / total.max(1e-8)
                };
                for uv in [[0.0, 0.0], [0.125, 0.25], [0.5, 0.875], [0.9375, 0.0625]] {
                    let value = sample(uv, &input);
                    let expected = reference::at(
                        e,
                        &input,
                        uv.map(f64::from),
                        seed,
                        grid,
                        f64::from(overlap),
                        Boundary::Repeat,
                    );
                    assert!(
                        (value - expected).abs() <= 2e-5,
                        "public patch assignments disagree with independent oracle"
                    );
                    for shift in [[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]] {
                        assert!(
                            (value - sample([uv[0] + shift[0], uv[1] + shift[1]], &input)).abs()
                                <= 2e-5
                        );
                    }
                    assert!((sample(uv, &correlated) - (2.0 * value - 0.5)).abs() <= 2e-5);
                    let repeated = SplatTransform::new(seed, grid, overlap)
                        .unwrap()
                        .samples(uv);
                    let original = t.samples(uv);
                    assert_eq!(original.len(), repeated.len());
                    for (a, b) in original.iter().zip(&repeated) {
                        assert_eq!(a.uv.map(f32::to_bits), b.uv.map(f32::to_bits));
                        assert_eq!(a.weight.to_bits(), b.weight.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn normalized_patch_field_is_continuous_across_cell_and_torus_boundaries() {
    let e = Extent::new(17, 11).unwrap();
    let values: Vec<_> = reference::fixture(e, "seeded")
        .into_iter()
        .map(f64::from)
        .collect();
    let t = SplatTransform::new(58, 4, 0.25).unwrap();
    let at = |uv| {
        let samples = t.samples(uv);
        let sum: f64 = samples.iter().map(|s| f64::from(s.weight)).sum();
        samples
            .iter()
            .map(|s| {
                f64::from(s.weight)
                    * reference::bilinear(e, &values, s.uv.map(f64::from), Boundary::Repeat)
            })
            .sum::<f64>()
            / sum
    };
    let step = 1.0 / 1048576.0;
    for cell in 0..=4 {
        let center = cell as f32 / 4.0;
        for axis in 0..2 {
            let mut a = [0.375, 0.625];
            let mut b = a;
            a[axis] = center - step;
            b[axis] = center + step;
            assert!(
                (at(a) - at(b)).abs() < 2e-4,
                "patch cutoff creates a discontinuity"
            );
        }
    }
}
