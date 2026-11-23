//! Independent f64 oracle; not a production fallback. See the handoff for the
//! explicit feather/threshold interpretation where ADR 0004 leaves detail open.

#[derive(Clone, Copy, Debug)]
pub enum Combine {
    Replace,
    Add,
    Subtract,
}

#[derive(Clone, Copy, Debug)]
pub struct Controls {
    pub picked: [f32; 3],
    pub radius: f32,
    pub threshold: f32,
    pub feather: f32,
    pub mode: Combine,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            picked: [0.5; 3],
            radius: 0.25,
            threshold: 0.5,
            feather: 0.1,
            mode: Combine::Replace,
        }
    }
}

pub fn selection(rgb: [f64; 3], c: Controls) -> f64 {
    let distance = rgb
        .into_iter()
        .zip(c.picked)
        .map(|(a, b)| (a - f64::from(b)).powi(2))
        .sum::<f64>()
        .sqrt();
    if c.feather == 0.0 {
        return f64::from(distance <= f64::from(c.radius));
    }
    let t = ((distance - f64::from(c.radius)) / f64::from(c.feather)).clamp(0.0, 1.0);
    let coverage = 1.0 - t * t * (3.0 - 2.0 * t);
    f64::from(coverage >= f64::from(c.threshold))
}

pub fn metallic(rgb: [f64; 3], mask: Option<f32>, c: Controls) -> f64 {
    let selected = selection(rgb, c);
    let mask = f64::from(mask.unwrap_or(1.0));
    match c.mode {
        Combine::Replace => mask,
        Combine::Add => (selected + mask).clamp(0.0, 1.0),
        Combine::Subtract => (selected - mask).clamp(0.0, 1.0),
    }
}

pub fn srgb(byte: u8) -> f64 {
    let x = f64::from(byte) / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

// Round the independent transfer result to the actual Rgba16Float node input.
// For [0,1], exhaustively selecting the nearest half is simple and unambiguous;
// a LUT amortizes the work. Ties use the IEEE even significand.
pub fn linear_half(byte: u8) -> f64 {
    static LUT: std::sync::OnceLock<[f64; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let x = srgb(i as u8);
            let bits = (0..=0x3c00u16)
                .min_by(|&a, &b| {
                    (half(a) - x)
                        .abs()
                        .total_cmp(&(half(b) - x).abs())
                        .then_with(|| (a & 1).cmp(&(b & 1)))
                })
                .unwrap();
            half(bits)
        })
    })[usize::from(byte)]
}

fn half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let mantissa = f64::from(bits & 1023);
    if exponent == 0 {
        mantissa * 2.0_f64.powi(-24)
    } else {
        (1.0 + mantissa / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15)
    }
}

pub fn photo_reference(bytes: &[u8], c: Controls) -> Vec<f64> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            metallic(
                [linear_half(p[0]), linear_half(p[1]), linear_half(p[2])],
                None,
                c,
            )
        })
        .collect()
}

pub fn compare(actual: &[f32], expected: &[f64]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && (0.0..=1.0).contains(&a), "pixel {i}: {a}");
        assert!(
            (f64::from(a) - b).abs() <= 2e-5,
            "pixel {i}: got {a}, expected {b}"
        );
    }
}

pub fn photo(width: u32, height: u32, kind: u8) -> Vec<u8> {
    let mut state = 54u64;
    let mut bytes = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let rgb = match kind {
                0 => [128, 128, 128],
                1 => [
                    ((x * 255) / width.max(2).saturating_sub(1)) as u8,
                    ((y * 255) / height.max(2).saturating_sub(1)) as u8,
                    128,
                ],
                2 => {
                    if x == 0 && y == 0 {
                        [255; 3]
                    } else {
                        [0; 3]
                    }
                }
                _ => std::array::from_fn(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 56) as u8
                }),
            };
            bytes.extend_from_slice(&rgb);
            bytes.push(((x + y) % 256) as u8);
        }
    }
    bytes
}

#[test]
fn oracle_has_hand_computed_center_radius_feather_and_threshold_equalities() {
    let c = Controls {
        picked: [0.0; 3],
        radius: 0.25,
        feather: 0.5,
        ..Default::default()
    };
    assert_eq!(selection([0.0; 3], c), 1.0);
    assert_eq!(selection([0.25, 0.0, 0.0], c), 1.0);
    assert_eq!(selection([0.5, 0.0, 0.0], c), 1.0); // coverage == threshold
    assert_eq!(selection([0.625, 0.0, 0.0], c), 0.0);
    assert_eq!(selection([0.75, 0.0, 0.0], c), 0.0);
    assert_eq!(
        selection(
            [0.5, 0.0, 0.0],
            Controls {
                threshold: 0.5001,
                ..c
            }
        ),
        0.0
    );
    assert_eq!(
        selection(
            [1.0; 3],
            Controls {
                threshold: 0.0,
                ..c
            }
        ),
        1.0
    );
    assert_eq!(
        selection([0.25, 0.0, 0.0], Controls { feather: 0.0, ..c }),
        1.0
    );
    assert_eq!(
        selection([0.2501, 0.0, 0.0], Controls { feather: 0.0, ..c }),
        0.0
    );
}

#[test]
fn oracle_mask_modes_have_literal_saturated_and_fractional_answers() {
    let c = Controls {
        picked: [0.0; 3],
        feather: 0.0,
        ..Default::default()
    };
    for (mode, selected, unselected) in [
        (Combine::Replace, 0.25, 0.25),
        (Combine::Add, 1.0, 0.25),
        (Combine::Subtract, 0.75, 0.0),
    ] {
        assert_eq!(
            metallic([0.0; 3], Some(0.25), Controls { mode, ..c }),
            selected
        );
        assert_eq!(
            metallic([1.0; 3], Some(0.25), Controls { mode, ..c }),
            unselected
        );
        let absent = match mode {
            Combine::Replace | Combine::Add => 1.0,
            Combine::Subtract => 0.0,
        };
        for rgb in [[0.0; 3], [1.0; 3]] {
            assert_eq!(metallic(rgb, None, Controls { mode, ..c }), absent);
            assert_eq!(
                metallic(rgb, None, Controls { mode, ..c }),
                metallic(rgb, Some(1.0), Controls { mode, ..c })
            );
        }
    }
}

#[test]
fn transfer_oracle_known_values_and_fixture_seed_are_stable() {
    assert_eq!(linear_half(0), 0.0);
    assert_eq!(linear_half(255), 1.0);
    assert!((srgb(128) - 0.2158605001).abs() < 1e-10);
    assert_eq!(linear_half(128), 0.2158203125);
    assert_eq!(photo(17, 13, 3), photo(17, 13, 3));
    assert_ne!(photo(17, 13, 3), photo(17, 13, 2));
}
