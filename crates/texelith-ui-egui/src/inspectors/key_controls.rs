//! Explicit, map-specific presentation of existing typed parameters.
use texelith_core::{MapKind, MaterialNode, ParameterId};

pub(crate) fn parameters<'a>(
    map: MapKind,
    nodes: &[&'a MaterialNode],
) -> Vec<(&'a MaterialNode, ParameterId)> {
    use ParameterId::*;
    let ids: &[ParameterId] = match map {
        MapKind::Normal => &[Strength, ScaleWeights, Filter, Convention],
        MapKind::Height => &[BandWeights, CoarseWeight, RemoveCoarse, Black, White, Gamma],
        MapKind::Ao => &[RadiusSourcePx, Bias, Directions, Steps],
        MapKind::Roughness => &[Base, Invert, Curve],
        MapKind::Metallic => &[PickedColor, Radius, Threshold, Feather, MaskMode],
        MapKind::Edge | MapKind::Cavity | MapKind::Curvature => &[EdgeScale, CavityScale, Strength],
        MapKind::Albedo => &[SeamlessMethod, Boundary],
        _ => &[],
    };
    ids.iter()
        .filter_map(|&id| {
            nodes
                .iter()
                .find(|n| n.params.parameter(id).is_ok())
                .map(|&n| (n, id))
        })
        .collect()
}

pub(crate) fn label(id: ParameterId) -> &'static str {
    use ParameterId::*;
    match id {
        Strength => "strength",
        ScaleWeights | BandWeights => "detail-scale",
        Filter => "smoothing",
        Convention => "invert-y",
        CoarseWeight => "broad-relief",
        RemoveCoarse => "remove-lighting",
        Black => "shadows",
        White => "highlights",
        Gamma => "midtones",
        RadiusSourcePx => "ao-radius",
        Bias => "ao-bias",
        Directions => "ao-directions",
        Steps => "ao-steps",
        Base => "roughness-level",
        Invert => "invert-roughness",
        Curve => "roughness-response",
        PickedColor => "metal-color",
        Radius => "color-tolerance",
        Threshold => "metal-threshold",
        Feather => "softness",
        MaskMode => "mask-blend",
        EdgeScale => "edge-strength",
        CavityScale => "cavity-strength",
        SeamlessMethod => "seam-method",
        Boundary => "image-edges",
        _ => "strength",
    }
}

/// Octaves are spaced at powers of two in the existing normal/height kernels.
pub(crate) fn detail_scale(weights: &[f32]) -> f32 {
    let peak = weights.iter().map(|v| v.abs()).fold(0., f32::max);
    if peak == 0. {
        return 1.;
    }
    let sum: f32 = weights.iter().map(|v| v.abs() / peak).sum();
    let octave: f32 = weights
        .iter()
        .enumerate()
        .map(|(i, v)| i as f32 * (v.abs() / peak) / sum)
        .sum();
    2.0_f32.powf(octave).clamp(1., 8.)
}

pub(crate) fn scale_weights(scale: f32) -> Vec<f32> {
    let octave = scale.clamp(1., 8.).log2();
    let low = octave.floor() as usize;
    let mut weights = vec![0.; (low + 2).min(4)];
    weights[low] = 1. - octave.fract();
    if low + 1 < weights.len() {
        weights[low + 1] = octave.fract();
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_scale_round_trips_and_accepts_advanced_weights() {
        for scale in [1., 1.3, 2., 3.5, 4., 6., 8.] {
            let weights = scale_weights(scale);
            assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
            assert!((detail_scale(&weights) - scale).abs() < 1e-5);
        }
        assert_eq!(detail_scale(&[0.; 4]), 1.);
        assert_eq!(detail_scale(&[0., 0., 1.]), 4.);
        assert!((detail_scale(&[f32::MAX, -f32::MAX]) - 2.0_f32.sqrt()).abs() < 1e-5);
    }
}
