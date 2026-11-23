//! Typed material controls and discovery metadata. Edits reuse DTO validation.

use crate::{
    Boundary, CoreError, MaskMode, MaterialNodeParams, NormalConvention, NormalFilter, ParameterId,
    ParameterValue, SeamlessMethod,
};
use serde::Serialize;

/// Authoritative wire value shape, bounds and procedural defaults.
/// All floating-point payloads must be finite, including nested components.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum ParameterSchema {
    Float {
        min: f32,
        max: f32,
        min_exclusive: bool,
        default: f32,
    },
    UInt {
        min: u32,
        max: u32,
        default: u32,
    },
    Bool {
        default: bool,
    },
    FloatVector {
        min_len: usize,
        max_len: usize,
        min: f32,
        max: f32,
        default: &'static [f32],
    },
    Curve {
        min_len: usize,
        max_len: usize,
        min: f32,
        max: f32,
        strictly_increasing_x: bool,
        default: &'static [[f32; 2]],
    },
    Color {
        min: f32,
        max: f32,
        default: [f32; 3],
    },
    Boundary {
        choices: &'static [Boundary],
        default: Boundary,
    },
    Filter {
        choices: &'static [NormalFilter],
        default: NormalFilter,
    },
    Convention {
        choices: &'static [NormalConvention],
        default: NormalConvention,
    },
    MaskMode {
        choices: &'static [MaskMode],
        default: MaskMode,
    },
    SeamlessMethod {
        choices: &'static [&'static str],
        overlap: [f32; 2],
        grid: [u32; 2],
        seed: [u32; 2],
        strength: [f32; 2],
        default: SeamlessMethod,
    },
}

/// One entry per parameter ID. Node names are the stable MaterialNodeParams tags.
/// Legacy scalar metadata remains available through PARAMETER_DESCRIPTORS.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MaterialParameterDescriptor {
    pub id: ParameterId,
    pub node_kinds: &'static [&'static str],
    pub value_schema: ParameterSchema,
}

const fn float(min: f32, max: f32, default: f32) -> ParameterSchema {
    ParameterSchema::Float {
        min,
        max,
        min_exclusive: false,
        default,
    }
}
const fn positive(default: f32) -> ParameterSchema {
    ParameterSchema::Float {
        min: 0.0,
        max: f32::MAX,
        min_exclusive: true,
        default,
    }
}
const fn uint(min: u32, max: u32, default: u32) -> ParameterSchema {
    ParameterSchema::UInt { min, max, default }
}
const BANDS: ParameterSchema = ParameterSchema::FloatVector {
    min_len: 1,
    max_len: 16,
    min: -f32::MAX,
    max: f32::MAX,
    default: &[1.0],
};

pub static MATERIAL_PARAMETER_DESCRIPTORS: [MaterialParameterDescriptor; 33] = [
    MaterialParameterDescriptor {
        id: ParameterId::Black,
        node_kinds: &["Levels"],
        value_schema: float(0.0, 1.0, 0.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::White,
        node_kinds: &["Levels"],
        value_schema: float(0.0, 1.0, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Gamma,
        node_kinds: &["Levels"],
        value_schema: float(0.1, 10.0, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::SigmaSourcePx,
        node_kinds: &["Blur"],
        value_schema: float(0.0, 5.0, 0.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Strength,
        node_kinds: &["NormalFromHeight"],
        value_schema: float(0.0, 10.0, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::BandWeights,
        node_kinds: &["HeightFromAlbedo"],
        value_schema: BANDS,
    },
    MaterialParameterDescriptor {
        id: ParameterId::CoarseWeight,
        node_kinds: &["HeightFromAlbedo"],
        value_schema: float(-f32::MAX, f32::MAX, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::RemoveCoarse,
        node_kinds: &["HeightFromAlbedo"],
        value_schema: ParameterSchema::Bool { default: false },
    },
    MaterialParameterDescriptor {
        id: ParameterId::Boundary,
        node_kinds: &[
            "HeightFromAlbedo",
            "NormalFromHeight",
            "HeightFromNormal",
            "HorizonAo",
            "Curvature",
            "Levels",
            "Blur",
            "Seamless",
        ],
        value_schema: ParameterSchema::Boundary {
            choices: &[Boundary::Clamp, Boundary::Repeat],
            default: Boundary::Repeat,
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::Filter,
        node_kinds: &["NormalFromHeight"],
        value_schema: ParameterSchema::Filter {
            choices: &[
                NormalFilter::Central,
                NormalFilter::Sobel,
                NormalFilter::Scharr,
            ],
            default: NormalFilter::Sobel,
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::ScaleWeights,
        node_kinds: &["NormalFromHeight"],
        value_schema: BANDS,
    },
    MaterialParameterDescriptor {
        id: ParameterId::Convention,
        node_kinds: &["NormalFromHeight", "NormalSource"],
        value_schema: ParameterSchema::Convention {
            choices: &[NormalConvention::OpenGl, NormalConvention::DirectX],
            default: NormalConvention::OpenGl,
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::VCycles,
        node_kinds: &["HeightFromNormal"],
        value_schema: uint(1, 16, 8),
    },
    MaterialParameterDescriptor {
        id: ParameterId::PreSmooth,
        node_kinds: &["HeightFromNormal"],
        value_schema: uint(1, u32::MAX, 3),
    },
    MaterialParameterDescriptor {
        id: ParameterId::PostSmooth,
        node_kinds: &["HeightFromNormal"],
        value_schema: uint(1, u32::MAX, 3),
    },
    MaterialParameterDescriptor {
        id: ParameterId::CoarsestIterations,
        node_kinds: &["HeightFromNormal"],
        value_schema: uint(1, u32::MAX, 32),
    },
    MaterialParameterDescriptor {
        id: ParameterId::RelativeTolerance,
        node_kinds: &["HeightFromNormal"],
        value_schema: positive(1e-3),
    },
    MaterialParameterDescriptor {
        id: ParameterId::AbsoluteTolerance,
        node_kinds: &["HeightFromNormal"],
        value_schema: positive(1e-6),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Directions,
        node_kinds: &["HorizonAo"],
        value_schema: uint(4, 32, 8),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Steps,
        node_kinds: &["HorizonAo"],
        value_schema: uint(4, 32, 8),
    },
    MaterialParameterDescriptor {
        id: ParameterId::RadiusSourcePx,
        node_kinds: &["HorizonAo"],
        value_schema: float(1.0, 512.0, 32.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Bias,
        node_kinds: &["HorizonAo"],
        value_schema: float(0.0, f32::MAX, 0.01),
    },
    MaterialParameterDescriptor {
        id: ParameterId::EdgeScale,
        node_kinds: &["Curvature"],
        value_schema: float(0.0, f32::MAX, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::CavityScale,
        node_kinds: &["Curvature"],
        value_schema: float(0.0, f32::MAX, 1.0),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Curve,
        node_kinds: &["Roughness"],
        value_schema: ParameterSchema::Curve {
            min_len: 2,
            max_len: 16,
            min: 0.0,
            max: 1.0,
            strictly_increasing_x: true,
            default: &[[0.0, 0.0], [1.0, 1.0]],
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::Base,
        node_kinds: &["Roughness"],
        value_schema: float(0.0, 1.0, 0.5),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Invert,
        node_kinds: &["Roughness"],
        value_schema: ParameterSchema::Bool { default: false },
    },
    MaterialParameterDescriptor {
        id: ParameterId::PickedColor,
        node_kinds: &["Metallic"],
        value_schema: ParameterSchema::Color {
            min: 0.0,
            max: 1.0,
            default: [0.5; 3],
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::Radius,
        node_kinds: &["Metallic"],
        value_schema: float(0.0, 1.732_050_8, 0.25),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Threshold,
        node_kinds: &["Metallic"],
        value_schema: float(0.0, 1.0, 0.5),
    },
    MaterialParameterDescriptor {
        id: ParameterId::Feather,
        node_kinds: &["Metallic"],
        value_schema: float(0.0, f32::MAX, 0.1),
    },
    MaterialParameterDescriptor {
        id: ParameterId::MaskMode,
        node_kinds: &["Metallic"],
        value_schema: ParameterSchema::MaskMode {
            choices: &[MaskMode::Replace, MaskMode::Add, MaskMode::Subtract],
            default: MaskMode::Replace,
        },
    },
    MaterialParameterDescriptor {
        id: ParameterId::SeamlessMethod,
        node_kinds: &["Seamless"],
        value_schema: ParameterSchema::SeamlessMethod {
            choices: &["OffsetBlend", "Splat", "Histogram"],
            overlap: [0.1, 0.5],
            grid: [2, 8],
            seed: [0, u32::MAX],
            strength: [0.0, 1.0],
            default: SeamlessMethod::OffsetBlend { overlap: 0.25 },
        },
    },
];

impl ParameterId {
    pub fn material_descriptor(self) -> &'static MaterialParameterDescriptor {
        MATERIAL_PARAMETER_DESCRIPTORS
            .iter()
            .find(|entry| entry.id == self)
            .expect("every ParameterId has typed discovery metadata")
    }
}

// Generate both directions from the same field/type table. Validation remains
// on the complete replacement, so coupled constraints (levels, curves) are atomic.
macro_rules! material_parameters {
    ($(($node:ident, $field:ident, $id:ident, $value:ident)),* $(,)?) => {
        impl MaterialNodeParams {
            pub fn parameter(&self, parameter: ParameterId) -> Result<ParameterValue, CoreError> {
                match (self, parameter) {
                    $((Self::$node { $field, .. }, ParameterId::$id) =>
                        Ok(ParameterValue::$value($field.clone())),)*
                    _ => Err(CoreError::InvalidMaterial("parameter does not belong to material node")),
                }
            }

            pub fn with_parameter(&self, parameter: ParameterId, value: ParameterValue) -> Result<Self, CoreError> {
                let mut replacement = self.clone();
                match (&mut replacement, parameter, value) {
                    $((Self::$node { $field, .. }, ParameterId::$id, ParameterValue::$value(value)) => *$field = value,)*
                    _ => return Err(CoreError::InvalidMaterial("wrong material parameter or value type")),
                }
                replacement.validate()?;
                Ok(replacement)
            }
        }
    };
}

material_parameters! {
    (HeightFromAlbedo, band_weights, BandWeights, FloatVector),
    (HeightFromAlbedo, coarse_weight, CoarseWeight, Float),
    (HeightFromAlbedo, remove_coarse, RemoveCoarse, Bool),
    (HeightFromAlbedo, boundary, Boundary, Boundary),
    (NormalFromHeight, strength, Strength, Float),
    (NormalFromHeight, filter, Filter, Filter),
    (NormalFromHeight, scale_weights, ScaleWeights, FloatVector),
    (NormalFromHeight, boundary, Boundary, Boundary),
    (NormalFromHeight, convention, Convention, Convention),
    (NormalSource, convention, Convention, Convention),
    (HeightFromNormal, boundary, Boundary, Boundary),
    (HeightFromNormal, v_cycles, VCycles, UInt),
    (HeightFromNormal, pre_smooth, PreSmooth, UInt),
    (HeightFromNormal, post_smooth, PostSmooth, UInt),
    (HeightFromNormal, coarsest_iterations, CoarsestIterations, UInt),
    (HeightFromNormal, relative_tolerance, RelativeTolerance, Float),
    (HeightFromNormal, absolute_tolerance, AbsoluteTolerance, Float),
    (HorizonAo, directions, Directions, UInt),
    (HorizonAo, steps, Steps, UInt),
    (HorizonAo, radius_source_px, RadiusSourcePx, Float),
    (HorizonAo, bias, Bias, Float),
    (HorizonAo, boundary, Boundary, Boundary),
    (Curvature, edge_scale, EdgeScale, Float),
    (Curvature, cavity_scale, CavityScale, Float),
    (Curvature, boundary, Boundary, Boundary),
    (Roughness, curve, Curve, Curve),
    (Roughness, base, Base, Float),
    (Roughness, invert, Invert, Bool),
    (Metallic, picked_color, PickedColor, Color),
    (Metallic, radius, Radius, Float),
    (Metallic, threshold, Threshold, Float),
    (Metallic, feather, Feather, Float),
    (Metallic, mask_mode, MaskMode, MaskMode),
    (Levels, black, Black, Float),
    (Levels, white, White, Float),
    (Levels, gamma, Gamma, Float),
    (Levels, boundary, Boundary, Boundary),
    (Blur, sigma_source_px, SigmaSourcePx, Float),
    (Blur, boundary, Boundary, Boundary),
    (Seamless, method, SeamlessMethod, SeamlessMethod),
    (Seamless, boundary, Boundary, Boundary),
}
