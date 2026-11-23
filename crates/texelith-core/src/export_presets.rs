//! Graph-independent preset recipes. Runtime parsing belongs to IO, not core.
use crate::{
    Boundary, ChannelSource, ChannelSpec, Component, CoreError, DisplacementSettings,
    EngineProfile, ExportColorTag, ExportFormat, ExportInterpretation, ExportOutput, ExportPlan,
    Graph, IntegerQuantization, MapKind, NormalConvention, SizePolicy, Transfer,
    validate_export_for_graph,
};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextureUsage {
    Albedo,
    Normal,
    Height,
    Occlusion,
    Roughness,
    Smoothness,
    Metallic,
    DetailMask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureAssignment {
    pub usage: TextureUsage,
    pub component: Component,
}

/// Assignment instructions travel with the bound output and immutable job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportImportHints {
    pub color_tag: ExportColorTag,
    pub normal_map: bool,
    pub normal_compression: bool,
    pub assignments: Vec<TextureAssignment>,
    pub instructions: Vec<String>,
}

impl ExportImportHints {
    pub(crate) fn owned_bytes(&self) -> usize {
        self.assignments.len() * size_of::<TextureAssignment>()
            + self
                .instructions
                .iter()
                .map(|s| size_of::<String>() + s.len())
                .sum::<usize>()
    }

    pub(crate) fn validate(&self, output: &ExportOutput) -> Result<(), CoreError> {
        let semantics = output.semantics();
        if self.color_tag != semantics.color_tag
            || (self.normal_compression && !self.normal_map)
            || (self.normal_map && semantics.interpretation != ExportInterpretation::Normal)
            || self.assignments.iter().any(|assignment| {
                let index = match assignment.component {
                    Component::R => 0,
                    Component::G => 1,
                    Component::B => 2,
                    Component::A => 3,
                };
                index >= output.channels.len()
            })
        {
            return Err(CoreError::InvalidMaterial(
                "export import hints contradict output semantics",
            ));
        }
        Ok(())
    }
}

/// Each missing map has an explicit policy. OmitOutput supports optional files,
/// while Constant supplies a linear numeric default for optional packed data.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MissingMap {
    Error,
    Constant(f32),
    OmitOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PresetChannelSource {
    Map {
        map: MapKind,
        component: Component,
        missing: MissingMap,
    },
    Constant(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetChannelSpec {
    pub source: PresetChannelSource,
    pub invert: bool,
    pub transfer: Transfer,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetOutput {
    pub name: String,
    pub format: ExportFormat,
    pub channels: Vec<PresetChannelSpec>,
    pub quantization: Option<IntegerQuantization>,
    pub import_hints: Option<ExportImportHints>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportPreset {
    pub profile: EngineProfile,
    pub naming: String,
    pub size: SizePolicy,
    pub padding: Boundary,
    pub displacement: DisplacementSettings,
    pub outputs: Vec<PresetOutput>,
}

impl ExportPreset {
    /// Resolve semantic maps from this graph's named outputs and derive its extent.
    /// Known negative-Z information is supplied separately because graphs do not
    /// contain evaluated pixel-domain bounds; encoders must check actual samples.
    pub fn bind(
        &self,
        graph: &Graph,
        negative_z_maps: &[MapKind],
    ) -> Result<ExportPlan, CoreError> {
        let maps = graph.material_outputs().ok_or(CoreError::InvalidMaterial(
            "preset binding requires named material outputs",
        ))?;
        let source_extent = graph.material_output_extent(graph.selected_output_ref())?;
        let mut outputs = Vec::new();
        for output in &self.outputs {
            let mut channels = Vec::new();
            let mut omit = false;
            let mut normal_has_negative_z = false;
            for channel in &output.channels {
                let source = match channel.source {
                    PresetChannelSource::Constant(value) => ChannelSource::Constant(value),
                    PresetChannelSource::Map {
                        map,
                        component,
                        missing,
                    } => {
                        let valid_component = match map {
                            MapKind::Albedo => true,
                            MapKind::Normal => component != Component::A,
                            _ => component == Component::R,
                        };
                        if !valid_component {
                            return Err(CoreError::InvalidMaterial("invalid preset map component"));
                        }
                        if let MissingMap::Constant(value) = missing
                            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
                        {
                            return Err(CoreError::InvalidMaterial("invalid preset map fallback"));
                        }
                        match maps.get(&map) {
                            Some(reference) => {
                                normal_has_negative_z |=
                                    map == MapKind::Normal && negative_z_maps.contains(&map);
                                ChannelSource::Map(*reference, component)
                            }
                            None => match missing {
                                MissingMap::Error => {
                                    return Err(CoreError::InvalidMaterial(
                                        "required preset map is absent",
                                    ));
                                }
                                MissingMap::Constant(value) => ChannelSource::Constant(value),
                                MissingMap::OmitOutput => {
                                    omit = true;
                                    continue;
                                }
                            },
                        }
                    }
                };
                channels.push(ChannelSpec {
                    source,
                    invert: channel.invert,
                    transfer: channel.transfer,
                });
            }
            if !omit {
                outputs.push(ExportOutput {
                    name: output.name.clone(),
                    format: output.format,
                    channels,
                    normal_has_negative_z,
                    quantization: output.quantization,
                    import_hints: output.import_hints.clone(),
                });
            }
        }
        let plan = ExportPlan {
            source_extent,
            size: self.size,
            naming: self.naming.clone(),
            profile: self.profile.clone(),
            outputs,
            displacement: self.displacement,
            padding: self.padding,
        };
        validate_export_for_graph(&plan, graph)?;
        Ok(plan)
    }
}

/// Typed preset table supplied by T-076; entries contain no NodeIds or extents.
pub trait ExportPresetTable {
    fn presets(&self) -> &[ExportPreset];
    fn find(&self, id: &str, revision: u32) -> Option<&ExportPreset> {
        self.presets()
            .iter()
            .find(|preset| preset.profile.id == id && preset.profile.revision == revision)
    }
}

/// Built-in engine export contracts. JSON counterparts live in `presets/` as
/// reviewable golden fixtures; production uses these typed values directly.
pub struct BuiltinExportPresets;

impl ExportPresetTable for BuiltinExportPresets {
    fn presets(&self) -> &[ExportPreset] {
        &BUILTIN_EXPORT_PRESETS
    }
}

const fn channel(
    source: PresetChannelSource,
    invert: bool,
    transfer: Transfer,
) -> PresetChannelSpec {
    PresetChannelSpec {
        source,
        invert,
        transfer,
    }
}

const fn mapped(map: MapKind, component: Component, missing: MissingMap) -> PresetChannelSource {
    PresetChannelSource::Map {
        map,
        component,
        missing,
    }
}

fn output(
    name: &str,
    format: ExportFormat,
    channels: Vec<PresetChannelSpec>,
    hints: ExportImportHints,
) -> PresetOutput {
    PresetOutput {
        name: name.into(),
        format,
        channels,
        quantization: match format {
            ExportFormat::Png8 | ExportFormat::Png16 => Some(IntegerQuantization::default()),
            _ => None,
        },
        import_hints: Some(hints),
    }
}

fn hints(
    color_tag: ExportColorTag,
    normal_map: bool,
    normal_compression: bool,
    assignments: &[(TextureUsage, Component)],
    instructions: &[&str],
) -> ExportImportHints {
    ExportImportHints {
        color_tag,
        normal_map,
        normal_compression,
        assignments: assignments
            .iter()
            .map(|(usage, component)| TextureAssignment {
                usage: *usage,
                component: *component,
            })
            .collect(),
        instructions: instructions.iter().map(|text| (*text).into()).collect(),
    }
}

fn profile(id: &str, convention: NormalConvention) -> EngineProfile {
    EngineProfile {
        id: id.into(),
        revision: 1,
        engine_safe_srgb: true,
        normal_convention: convention,
    }
}

fn preset(profile: EngineProfile, outputs: Vec<PresetOutput>) -> ExportPreset {
    ExportPreset {
        profile,
        naming: "T_{name}_{map}".into(),
        size: SizePolicy::Keep,
        padding: Boundary::Repeat,
        displacement: DisplacementSettings::default(),
        outputs,
    }
}

fn rgb(map: MapKind, transfer: Transfer, missing: MissingMap) -> Vec<PresetChannelSpec> {
    [Component::R, Component::G, Component::B]
        .into_iter()
        .map(|c| channel(mapped(map, c, missing), false, transfer))
        .collect()
}

fn scalar(map: MapKind) -> PresetChannelSpec {
    channel(
        mapped(map, Component::R, MissingMap::Constant(1.0)),
        false,
        Transfer::Linear,
    )
}

fn albedo() -> PresetOutput {
    output(
        "albedo",
        ExportFormat::Png8,
        rgb(MapKind::Albedo, Transfer::Srgb, MissingMap::Error),
        hints(
            ExportColorTag::Srgb,
            false,
            false,
            &[
                (TextureUsage::Albedo, Component::R),
                (TextureUsage::Albedo, Component::G),
                (TextureUsage::Albedo, Component::B),
            ],
            &["Import as sRGB color texture."],
        ),
    )
}

fn normal(convention: NormalConvention, compression: bool) -> PresetOutput {
    output(
        "normal",
        ExportFormat::Png8,
        rgb(MapKind::Normal, Transfer::Linear, MissingMap::Error),
        hints(
            ExportColorTag::Linear,
            true,
            compression,
            &[
                (TextureUsage::Normal, Component::R),
                (TextureUsage::Normal, Component::G),
                (TextureUsage::Normal, Component::B),
            ],
            &[
                match convention {
                    NormalConvention::OpenGl => "OpenGL normal green orientation.",
                    NormalConvention::DirectX => "DirectX normal green orientation.",
                },
                "Disable sRGB; import as a normal map.",
            ],
        ),
    )
}

fn packed(
    name: &str,
    usages: &[(TextureUsage, Component)],
    channels: Vec<PresetChannelSpec>,
    instructions: &[&str],
) -> PresetOutput {
    output(
        name,
        ExportFormat::Png8,
        channels,
        hints(ExportColorTag::Linear, false, false, usages, instructions),
    )
}

fn build_presets() -> Vec<ExportPreset> {
    use Component::{A, B, G, R};
    use MapKind::{Ao, Metallic, Roughness};
    use MissingMap::{Constant, Error};
    use TextureUsage::{Metallic as MetalUsage, Occlusion, Roughness as RoughUsage, Smoothness};
    let orm = vec![scalar(Ao), scalar(Roughness), scalar(Metallic)];
    vec![
        preset(
            profile("unreal-v1", NormalConvention::DirectX),
            vec![
                albedo(),
                normal(NormalConvention::DirectX, true),
                packed(
                    "orm",
                    &[(Occlusion, R), (RoughUsage, G), (MetalUsage, B)],
                    orm.clone(),
                    &["ORM RGB: ambient occlusion, roughness, metallic; linear data."],
                ),
                output(
                    "height",
                    ExportFormat::Png16,
                    vec![scalar(MapKind::Height)],
                    hints(
                        ExportColorTag::Linear,
                        false,
                        false,
                        &[(TextureUsage::Height, R)],
                        &[
                            "Normalized height PNG16; displacement midlevel and scale are in the manifest.",
                        ],
                    ),
                ),
                output(
                    "normal_bc5",
                    ExportFormat::DdsBc5,
                    vec![
                        channel(mapped(MapKind::Normal, R, Error), false, Transfer::Linear),
                        channel(mapped(MapKind::Normal, G, Error), false, Transfer::Linear),
                    ],
                    hints(
                        ExportColorTag::Linear,
                        true,
                        true,
                        &[(TextureUsage::Normal, R)],
                        &["DDS BC5_UNORM; positive-Z normals only; sRGB off."],
                    ),
                ),
            ],
        ),
        preset(
            profile("unity-urp-v1", NormalConvention::OpenGl),
            vec![
                albedo(),
                normal(NormalConvention::OpenGl, false),
                packed(
                    "metallic_smoothness",
                    &[(MetalUsage, R), (Smoothness, A)],
                    vec![
                        scalar(Metallic),
                        channel(PresetChannelSource::Constant(0.0), false, Transfer::Linear),
                        channel(PresetChannelSource::Constant(0.0), false, Transfer::Linear),
                        channel(mapped(Roughness, R, Constant(1.0)), true, Transfer::Linear),
                    ],
                    &["RGBA: metallic in R, G/B zero, smoothness in A."],
                ),
                packed(
                    "occlusion",
                    &[(Occlusion, G)],
                    vec![
                        channel(PresetChannelSource::Constant(1.0), false, Transfer::Linear),
                        scalar(Ao),
                        channel(PresetChannelSource::Constant(1.0), false, Transfer::Linear),
                    ],
                    &["RGB: 1, ambient occlusion, 1; linear data."],
                ),
            ],
        ),
        preset(
            profile("unity-hdrp-v1", NormalConvention::OpenGl),
            vec![
                albedo(),
                normal(NormalConvention::OpenGl, false),
                packed(
                    "mask_map",
                    &[
                        (MetalUsage, R),
                        (Occlusion, G),
                        (TextureUsage::DetailMask, B),
                        (Smoothness, A),
                    ],
                    vec![
                        scalar(Metallic),
                        scalar(Ao),
                        channel(PresetChannelSource::Constant(1.0), false, Transfer::Linear),
                        channel(mapped(Roughness, R, Constant(1.0)), true, Transfer::Linear),
                    ],
                    &["HDRP Mask Map RGBA: metallic, occlusion, detail mask 1, smoothness."],
                ),
            ],
        ),
        preset(
            profile("godot-v1", NormalConvention::OpenGl),
            vec![
                albedo(),
                normal(NormalConvention::OpenGl, false),
                packed(
                    "orm",
                    &[(Occlusion, R), (RoughUsage, G), (MetalUsage, B)],
                    orm.clone(),
                    &["ORM RGB: ambient occlusion, roughness, metallic; linear data."],
                ),
            ],
        ),
        preset(
            profile("blender-v1", NormalConvention::OpenGl),
            vec![
                albedo(),
                normal(NormalConvention::OpenGl, false),
                packed(
                    "roughness",
                    &[(RoughUsage, R)],
                    vec![scalar(Roughness)],
                    &["Non-Color data."],
                ),
                packed(
                    "metallic",
                    &[(MetalUsage, R)],
                    vec![scalar(Metallic)],
                    &["Non-Color data."],
                ),
                packed(
                    "occlusion",
                    &[(Occlusion, R)],
                    vec![scalar(Ao)],
                    &["Non-Color data."],
                ),
                output(
                    "height",
                    ExportFormat::Png16,
                    vec![scalar(MapKind::Height)],
                    hints(
                        ExportColorTag::Linear,
                        false,
                        false,
                        &[(TextureUsage::Height, R)],
                        &[
                            "Normalized height PNG16; displacement midlevel and scale are in the manifest.",
                        ],
                    ),
                ),
            ],
        ),
        preset(
            profile("gltf2-v1", NormalConvention::OpenGl),
            vec![
                albedo(),
                normal(NormalConvention::OpenGl, false),
                packed(
                    "orm",
                    &[(Occlusion, R), (RoughUsage, G), (MetalUsage, B)],
                    orm.clone(),
                    &["ORM RGB: occlusion uses R, metallic-roughness uses G/B; linear data."],
                ),
            ],
        ),
    ]
}

pub static BUILTIN_EXPORT_PRESETS: LazyLock<Vec<ExportPreset>> = LazyLock::new(build_presets);
