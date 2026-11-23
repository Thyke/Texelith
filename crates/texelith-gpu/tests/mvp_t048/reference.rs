//! f64 spatial convolution oracle, independent of production uniforms/shaders.
use texelith_core::{Boundary, Extent, NormalConvention, NormalFilter};

pub fn image(extent: Extent, f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
    (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y))
        .collect()
}

pub fn ramp(e: Extent, du: f32, dv: f32) -> Vec<f32> {
    image(e, |x, y| {
        -0.25 + du * (x as f32 + 0.5) / e.width as f32 + dv * (y as f32 + 0.5) / e.height as f32
    })
}

pub fn seeded(e: Extent) -> Vec<f32> {
    let mut state = 48u32;
    (0..e.width * e.height)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Small slopes avoid saturating normals and concealment by half precision.
            ((state >> 8) as f32 / 16777215.0 - 0.5) * 0.015
        })
        .collect()
}

pub fn unit(du: f64, dv: f64, strength: f64, convention: NormalConvention) -> [f64; 3] {
    let n = [-strength * du, strength * dv, 1.0];
    let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
    [
        n[0] / length,
        n[1] / length
            * if convention == NormalConvention::DirectX {
                -1.0
            } else {
                1.0
            },
        n[2] / length,
    ]
}

pub fn normals(
    e: Extent,
    data: &[f32],
    filter: NormalFilter,
    boundary: Boundary,
    strength: f32,
    convention: NormalConvention,
) -> Vec<[f64; 3]> {
    assert_eq!(data.len(), (e.width * e.height) as usize);
    let address = |p: i32, n: u32| match boundary {
        Boundary::Clamp => p.clamp(0, n as i32 - 1),
        Boundary::Repeat => p.rem_euclid(n as i32),
    } as usize;
    let sample =
        |x, y| f64::from(data[address(y, e.height) * e.width as usize + address(x, e.width)]);
    let (smooth, divisor) = match filter {
        NormalFilter::Central => ([0.0, 1.0, 0.0], 2.0),
        NormalFilter::Sobel => ([1.0, 2.0, 1.0], 8.0),
        NormalFilter::Scharr => ([3.0, 10.0, 3.0], 32.0),
    };
    (0..e.height as i32)
        .flat_map(|y| (0..e.width as i32).map(move |x| (x, y)))
        .map(|(x, y)| {
            let mut du = 0.0;
            let mut dv = 0.0;
            for (i, weight) in smooth.into_iter().enumerate() {
                let t = i as i32 - 1;
                du += weight * (sample(x + 1, y + t) - sample(x - 1, y + t));
                dv += weight * (sample(x + t, y + 1) - sample(x + t, y - 1));
            }
            unit(
                du * f64::from(e.width) / divisor,
                dv * f64::from(e.height) / divisor,
                f64::from(strength),
                convention,
            )
        })
        .collect()
}

pub fn compare(actual: &[[f64; 4]], expected: &[[f64; 3]]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        for c in 0..3 {
            assert!(
                a[c].is_finite() && (a[c] - b[c]).abs() <= 2e-3,
                "pixel {i} channel {c}: actual {a:?}, expected {b:?}"
            );
        }
        assert!(
            (a[..3].iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() <= 2e-3,
            "pixel {i} is not a unit normal: {a:?}"
        );
        assert_eq!(a[3], 1.0);
    }
}

#[test]
fn oracle_is_anchored_by_uv_ramp_and_off_axis_impulse_coefficients() {
    let e = Extent::new(9, 7).unwrap();
    for filter in [
        NormalFilter::Central,
        NormalFilter::Sobel,
        NormalFilter::Scharr,
    ] {
        let n = normals(
            e,
            &ramp(e, 0.5, -0.25),
            filter,
            Boundary::Clamp,
            2.0,
            NormalConvention::OpenGl,
        );
        let expected = unit(0.5, -0.25, 2.0, NormalConvention::OpenGl);
        for c in 0..3 {
            assert!((n[3 * 9 + 4][c] - expected[c]).abs() < 1e-7);
        }
    }
    let e = Extent::new(3, 3).unwrap();
    let mut impulse = vec![0.0; 9];
    impulse[0] = 1.0;
    for (filter, slope) in [
        (NormalFilter::Sobel, -3.0 / 8.0),
        (NormalFilter::Scharr, -9.0 / 32.0),
    ] {
        let n = normals(
            e,
            &impulse,
            filter,
            Boundary::Clamp,
            1.0,
            NormalConvention::OpenGl,
        )[4];
        let expected = unit(slope, slope, 1.0, NormalConvention::OpenGl);
        for c in 0..3 {
            assert!((n[c] - expected[c]).abs() < 1e-14);
        }
    }
}
