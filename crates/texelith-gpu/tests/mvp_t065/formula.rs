use super::oracle;
use texelith_gpu::pbr::*;
#[test]
fn analytic_ggx_matches_independent_correlated_smith_oracle() {
    for r in [0.05f32, 0.3, 0.7, 1.0] {
        for m in [0.0, 0.4, 1.0] {
            for (nv, nl, nh, vh) in [
                (1., 1., 1., 1.),
                (0.8, 0.6, 0.7, 0.9),
                (0.01, 0.02, 0.3, 0.5),
            ] {
                let color = [0.1, 0.4, 0.8];
                let got = direct_brdf(color, r, m, nv, nl, nh, vh);
                let expected = oracle::shade(
                    color.map(f64::from),
                    f64::from(r),
                    f64::from(m),
                    nv.into(),
                    nl.into(),
                    nh.into(),
                    vh.into(),
                );
                for (a, b) in got.into_iter().zip(expected) {
                    assert!(
                        (f64::from(a) - b).abs() <= b.abs() * 2e-4 + 1e-6,
                        "{a} != {b}"
                    );
                }
            }
        }
    }
    // Pure metal carries no diffuse, and zero metal uses exactly dielectric F0.
    let expected = 0.25 / std::f32::consts::PI;
    assert!((direct_brdf([1.; 3], 1., 1., 1., 1., 1., 1.)[0] - expected).abs() < 1e-6);
    assert_eq!(direct_brdf([1.; 3], 0., 0., 1., 0., 1., 1.), [0.; 3]);
}
#[test]
fn roughness_grazing_and_hdr_limits_stay_finite() {
    for r in [0., 1e-8, 1.] {
        for nv in [0., 1e-10, 0.01, 1.] {
            for nl in [0., 1e-10, 1.] {
                for m in [0., 1.] {
                    assert!(
                        direct_brdf([0.2, 0.5, 1.], r, m, nv, nl, 1., 0.)
                            .iter()
                            .all(|x| x.is_finite() && *x >= 0.)
                    );
                }
            }
        }
    }
    for x in [0., 0.0031308, 0.18, 1., 16., 65504., f32::MAX] {
        assert!(tone_map(x).is_finite());
        assert!((0.0..=1.0).contains(&tone_map(x)));
    }
}
#[test]
fn tone_curve_and_exact_single_srgb_transfer_match_known_ramp() {
    for i in 0..=1000 {
        let x = i as f32 / 1000.;
        assert!((linear_to_srgb(x) * 255. - f32::from(oracle::transfer(x.into()))).abs() <= 0.51);
        assert!((f64::from(tone_map(x)) - oracle::tone(x.into())).abs() < 2e-7);
    }
    assert!((linear_to_srgb(0.0031308) - 0.040449936).abs() < 1e-7);
    assert!((linear_to_srgb(0.18) - 0.46135613).abs() < 1e-6);
    assert!((tone_map(0.18) - 0.2668989).abs() < 1e-6);
}
