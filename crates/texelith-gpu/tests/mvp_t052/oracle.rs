//! Original f64 reference from ADR 0004, with no production numerical helpers.
use texelith_core::{Boundary, Extent, MaterialNodeParams, NormalConvention};

#[derive(Clone, Copy, Debug)]
pub struct Controls {
    pub edge: f32,
    pub cavity: f32,
    pub boundary: Boundary,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            edge: 1.0,
            cavity: 1.0,
            boundary: Boundary::Repeat,
        }
    }
}

impl Controls {
    pub fn params(self) -> MaterialNodeParams {
        MaterialNodeParams::Curvature {
            edge_scale: self.edge,
            cavity_scale: self.cavity,
            boundary: self.boundary,
        }
    }
}

pub fn image<T>(e: Extent, mut sample: impl FnMut(u32, u32) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity((e.width * e.height) as usize);
    for y in 0..e.height {
        for x in 0..e.width {
            out.push(sample(x, y));
        }
    }
    out
}

pub fn normalize(n: [f64; 3]) -> [f64; 3] {
    let len = n.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(len.is_finite() && len > 0.0);
    n.map(|v| v / len)
}

/// RGB8 is the public typed import API. Flip the encoded byte exactly so GL/DX
/// fixtures describe the same vector even at the two codes surrounding zero.
pub fn encode(normals: &[[f64; 3]], convention: NormalConvention) -> Vec<u8> {
    normals
        .iter()
        .flat_map(|&n| {
            let mut rgb = normalize(n).map(|v| ((v * 0.5 + 0.5) * 255.0).round() as u8);
            if convention == NormalConvention::DirectX {
                rgb[1] = 255 - rgb[1];
            }
            rgb
        })
        .collect()
}

pub fn decoded(rgb: &[u8]) -> Vec<[f64; 3]> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|p| normalize([p[0], p[1], p[2]].map(|v| f64::from(v) * 2.0 / 255.0 - 1.0)))
        .collect()
}

pub fn paraboloid(e: Extent, sign: f64) -> Vec<[f64; 3]> {
    image(e, |x, y| {
        let u = (f64::from(x) + 0.5) / f64::from(e.width) - 0.5;
        let v = (f64::from(y) + 0.5) / f64::from(e.height) - 0.5;
        // h=-sign*(u^2+v^2)/4; GL normal=(-dh/du,+dh/dv,1).
        normalize([sign * 0.5 * u, -sign * 0.5 * v, 1.0])
    })
}

pub fn seeded(e: Extent) -> Vec<[f64; 3]> {
    let mut state = 52_u32;
    image(e, |_, _| {
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            f64::from(state >> 8) / 16777215.0 - 0.5
        };
        normalize([next() * 0.2, next() * 0.2, 1.0])
    })
}

/// Read the actual Rgba16Float input plane through the public Normal output,
/// then normalize in f64 before differentiation. This keeps the scalar 2e-5
/// tolerance independent of source quantization amplified by the UV derivative.
pub fn curvature(
    normals: &[[f64; 3]],
    e: Extent,
    convention: NormalConvention,
    c: Controls,
) -> [Vec<f64>; 3] {
    assert_eq!(normals.len(), (e.width * e.height) as usize);
    let canonical: Vec<_> = normals
        .iter()
        .map(|&n| {
            let mut n = normalize(n);
            if convention == NormalConvention::DirectX {
                n[1] = -n[1];
            }
            n
        })
        .collect();
    let fetch = |x: i64, y: i64| {
        let address = |p: i64, n: u32| match c.boundary {
            Boundary::Clamp => p.clamp(0, i64::from(n) - 1),
            Boundary::Repeat => p.rem_euclid(i64::from(n)),
        };
        canonical[(address(y, e.height) * i64::from(e.width) + address(x, e.width)) as usize]
    };
    let signed = image(e, |x, y| {
        let (x, y) = (i64::from(x), i64::from(y));
        let du = if e.width == 1 {
            0.0
        } else {
            (fetch(x + 1, y)[0] - fetch(x - 1, y)[0]) * f64::from(e.width) * 0.5
        };
        let dv = if e.height == 1 {
            0.0
        } else {
            (fetch(x, y + 1)[1] - fetch(x, y - 1)[1]) * f64::from(e.height) * 0.5
        };
        du - dv
    });
    let edge = signed
        .iter()
        .map(|v| (v.max(0.0) * f64::from(c.edge)).clamp(0.0, 1.0))
        .collect();
    let cavity = signed
        .iter()
        .map(|v| ((-v).max(0.0) * f64::from(c.cavity)).clamp(0.0, 1.0))
        .collect();
    [signed, edge, cavity]
}

pub fn compare(actual: &[f32], expected: &[f64]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (f64::from(a) - b).abs() <= 2e-5,
            "pixel {i}: GPU {a}, CPU {b}, tolerance 2e-5"
        );
    }
}

#[test]
fn oracle_has_known_uv_sign_and_boundary_values() {
    let e = Extent::new(5, 3).unwrap();
    // Unit vectors with exactly linear tangent fields: interior divergence=.7.
    let n = image(e, |x, y| {
        let nx = 0.4 * (f64::from(x) + 0.5) / 5.0;
        let ny = -0.3 * (f64::from(y) + 0.5) / 3.0;
        [nx, ny, (1.0 - nx * nx - ny * ny).sqrt()]
    });
    let clamp = curvature(
        &n,
        e,
        NormalConvention::OpenGl,
        Controls {
            boundary: Boundary::Clamp,
            ..Default::default()
        },
    );
    assert!((clamp[0][7] - 0.7).abs() < 1e-12);
    assert!((clamp[0][0] - 0.35).abs() < 1e-12);
    let repeat = curvature(&n, e, NormalConvention::OpenGl, Controls::default());
    assert!((repeat[0][0] + 0.75).abs() < 1e-12);
    assert!(repeat[0].iter().sum::<f64>().abs() < 1e-12);
}
