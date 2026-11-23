//! Proposed reusable API for T-058/T-059; see the handoff for exact signatures.
use super::reference;
use texelith_core::{Boundary, Extent};
use texelith_gpu::seamless_offset::OffsetTransform;

#[test]
fn reusable_transform_validates_overlap_and_covers_the_torus_with_four_smooth_windows() {
    for overlap in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.0,
        0.099,
        0.501,
    ] {
        assert!(
            OffsetTransform::new(overlap).is_err(),
            "accepted invalid overlap {overlap}"
        );
    }
    for overlap in [0.1f32, 0.25, 0.5] {
        let transform = OffsetTransform::new(overlap).unwrap();
        for y in 0..=32 {
            for x in 0..=32 {
                let uv = [x as f32 / 32.0, y as f32 / 32.0];
                let taps = transform.samples(uv);
                let expected = reference::samples(uv.map(f64::from), f64::from(overlap));
                assert_eq!(taps.len(), 4);
                let mut used = [false; 4];
                let mut total = 0.0;
                for tap in taps {
                    assert!(
                        tap.uv
                            .iter()
                            .all(|c| c.is_finite() && (0.0..1.0).contains(c))
                    );
                    assert!(tap.weight.is_finite() && tap.weight >= 0.0);
                    total += f64::from(tap.weight);
                    let index = expected
                        .iter()
                        .enumerate()
                        .position(|(i, (p, _))| {
                            !used[i]
                                && (f64::from(tap.uv[0]) - p[0]).abs() <= 2e-5
                                && (f64::from(tap.uv[1]) - p[1]).abs() <= 2e-5
                        })
                        .expect("incorrect continuous half-offset sample");
                    used[index] = true;
                    assert!((f64::from(tap.weight) - expected[index].1).abs() <= 2e-5);
                    if tap.uv.contains(&0.0) {
                        assert_eq!(
                            tap.weight, 0.0,
                            "window must vanish at its original boundary"
                        );
                    }
                }
                assert!(used.into_iter().all(|v| v));
                assert!(total > 1e-8, "uncovered pixel at {uv:?}");
                let normalized: f64 = transform
                    .samples(uv)
                    .iter()
                    .map(|s| f64::from(s.weight) / total.max(1e-8))
                    .sum();
                assert!((normalized - 1.0).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn reusable_transform_sampling_is_periodic_and_retains_channel_correspondence() {
    let e = Extent::new(17, 11).unwrap();
    let input: Vec<_> = reference::scalar_fixture(e, "seeded")
        .into_iter()
        .map(f64::from)
        .collect();
    let correlated: Vec<_> = input.iter().map(|h| 2.0 * h - 0.5).collect();
    for overlap in [0.1f32, 0.25, 0.5] {
        let transform = OffsetTransform::new(overlap).unwrap();
        let sample = |uv, values: &[f64]| {
            let taps = transform.samples(uv);
            let total: f64 = taps.iter().map(|s| f64::from(s.weight)).sum();
            taps.iter()
                .map(|s| {
                    f64::from(s.weight)
                        * reference::bilinear(e, values, s.uv.map(f64::from), Boundary::Repeat)
                })
                .sum::<f64>()
                / total.max(1e-8)
        };
        for uv in [[0.0, 0.0], [0.125, 0.25], [0.5, 0.875], [0.9375, 0.0625]] {
            let value = sample(uv, &input);
            for offset in [[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]] {
                let moved = sample([uv[0] + offset[0], uv[1] + offset[1]], &input);
                assert!(
                    (value - moved).abs() <= 2e-5,
                    "translated-period sampling changed value"
                );
            }
            assert!((sample(uv, &correlated) - (2.0 * value - 0.5)).abs() <= 2e-5);
        }
    }
}

#[test]
fn each_window_has_zero_first_derivative_where_its_copy_wraps() {
    let transform = OffsetTransform::new(0.25).unwrap();
    let step = 1.0 / 65536.0;
    // Locate the copy crossing zero in x; stable sample ordering is part of metadata reuse.
    let center = transform.samples([0.0, 0.25]);
    let left = transform.samples([-step, 0.25]);
    let right = transform.samples([step, 0.25]);
    let mut tested = 0;
    for i in 0..4 {
        if center[i].uv[0] == 0.0 {
            assert_eq!(center[i].weight, 0.0);
            assert!(left[i].weight / step < 0.002);
            assert!(right[i].weight / step < 0.002);
            tested += 1;
        }
    }
    assert_eq!(tested, 2);
}
