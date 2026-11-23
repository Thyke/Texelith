//! Portable export settings and validation. No pixels, paths or encoders live here.
mod capture;
pub use capture::*;

use crate::{
    Boundary, CoreError, ExportImportHints, Extent, Graph, NodeParams, NormalConvention,
    OutputPort, OutputRef,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Component {
    R,
    G,
    B,
    A,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelSource {
    Map(OutputRef, Component),
    Constant(f32),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transfer {
    Linear,
    Srgb,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelSpec {
    pub source: ChannelSource,
    pub invert: bool,
    pub transfer: Transfer,
}

impl ChannelSpec {
    /// Imported source alpha is ignored in Faz 1. Inversion is applied afterwards.
    pub fn effective_source(self) -> ChannelSource {
        match self.source {
            ChannelSource::Map(
                OutputRef {
                    port: OutputPort::Albedo,
                    ..
                },
                Component::A,
            ) => ChannelSource::Constant(1.0),
            source => source,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Png8,
    Png16,
    Exr32,
    DdsBc5,
}

impl ExportFormat {
    /// Numeric channel precision; BC5's block representation uses UNORM8 endpoints.
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Png8 | Self::DdsBc5 => 8,
            Self::Png16 => 16,
            Self::Exr32 => 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportInterpretation {
    Color,
    Scalar,
    Normal,
    Packed,
    Displacement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportColorTag {
    Linear,
    Srgb,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportWarning {
    MixedTransfer,
}

/// Derived from declarations, so precision, interpretation and file tags cannot
/// contradict the selected format/channels. Raw mixtures require a manifest warning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportOutputSemantics {
    pub interpretation: ExportInterpretation,
    pub bit_depth: u8,
    pub color_tag: ExportColorTag,
    pub warnings: Vec<ExportWarning>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizePolicy {
    Keep,
    Resize { width: u32, height: u32 },
    Pad { width: u32, height: u32 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplacementUnits {
    #[default]
    Unitless,
    Meters,
    Centimeters,
    Millimeters,
}

/// EXR stores (h - midlevel) * scale; PNG height remains normalized h.
/// Both formats retain these settings for their assignment manifest.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplacementSettings {
    pub midlevel: f32,
    pub scale: f32,
    pub units: DisplacementUnits,
}

impl Default for DisplacementSettings {
    fn default() -> Self {
        Self {
            midlevel: 0.5,
            scale: 1.0,
            units: DisplacementUnits::Unitless,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegerDither {
    #[default]
    None,
    /// Fixed 4x4 Bayer matrix anchored at output pixel (0, 0), no random seed.
    Ordered4x4,
}

/// Clamp before integer quantization. This never clamps float displacement.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerQuantization {
    pub clamp_min: f32,
    pub clamp_max: f32,
    pub dither: IntegerDither,
}

impl Default for IntegerQuantization {
    fn default() -> Self {
        Self {
            clamp_min: 0.0,
            clamp_max: 1.0,
            dither: IntegerDither::None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineProfile {
    pub id: String,
    pub revision: u32,
    pub engine_safe_srgb: bool,
    pub normal_convention: NormalConvention,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportOutput {
    pub name: String,
    pub format: ExportFormat,
    /// Destination channels in RGBA order; an omitted alpha is generated as one.
    /// Normal components are selected after convention conversion, then UNORM encoded.
    pub channels: Vec<ChannelSpec>,
    /// Known source-domain information. Encoders must also reject discovered negative Z.
    pub normal_has_negative_z: bool,
    /// PNG only; None means the default [0, 1] clamp without dither.
    #[serde(default)]
    pub quantization: Option<IntegerQuantization>,
    #[serde(default)]
    pub import_hints: Option<ExportImportHints>,
}

fn albedo_rgb(channel: &ChannelSpec) -> bool {
    matches!(
        channel.source,
        ChannelSource::Map(
            OutputRef {
                port: OutputPort::Albedo,
                ..
            },
            Component::R | Component::G | Component::B
        )
    )
}

fn vector_channels(channels: &[ChannelSpec], port: OutputPort, components: &[Component]) -> bool {
    let Some(ChannelSpec {
        source: ChannelSource::Map(reference, _),
        ..
    }) = channels.first()
    else {
        return false;
    };
    channels.len() >= components.len()
        && reference.port == port
        && channels.iter().zip(components).all(|(channel, component)| {
            channel.source == ChannelSource::Map(*reference, *component)
        })
}

impl ExportOutput {
    pub fn semantics(&self) -> ExportOutputSemantics {
        let srgb = self
            .channels
            .iter()
            .any(|channel| channel.transfer == Transfer::Srgb);
        let srgb_rgb = self.channels.len() >= 3
            && self.channels[..3]
                .iter()
                .all(|channel| albedo_rgb(channel) && channel.transfer == Transfer::Srgb);
        let color_tag = if srgb_rgb {
            ExportColorTag::Srgb
        } else if srgb {
            ExportColorTag::Raw
        } else {
            ExportColorTag::Linear
        };
        let interpretation = if self.format == ExportFormat::Exr32 {
            ExportInterpretation::Displacement
        } else if vector_channels(
            &self.channels,
            OutputPort::Normal,
            if self.format == ExportFormat::DdsBc5 {
                &[Component::R, Component::G]
            } else {
                &[Component::R, Component::G, Component::B]
            },
        ) {
            ExportInterpretation::Normal
        } else if self.channels.len() >= 3 && self.channels[..3].iter().all(albedo_rgb) {
            ExportInterpretation::Color
        } else if self.channels.len() == 1 {
            ExportInterpretation::Scalar
        } else {
            ExportInterpretation::Packed
        };
        ExportOutputSemantics {
            interpretation,
            bit_depth: self.format.bit_depth(),
            color_tag,
            warnings: if color_tag == ExportColorTag::Raw {
                vec![ExportWarning::MixedTransfer]
            } else {
                vec![]
            },
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportPlan {
    pub source_extent: Extent,
    pub size: SizePolicy,
    pub naming: String,
    pub profile: EngineProfile,
    pub outputs: Vec<ExportOutput>,
    #[serde(default)]
    pub displacement: DisplacementSettings,
    /// Used only by Pad, anchored at the top-left valid-content rectangle.
    pub padding: Boundary,
}

impl ExportPlan {
    pub(crate) fn owned_bytes(&self) -> usize {
        self.naming.len()
            + self.profile.id.len()
            + self
                .outputs
                .iter()
                .map(|output| {
                    size_of::<ExportOutput>()
                        + output.name.len()
                        + output.channels.len() * size_of::<ChannelSpec>()
                        + output
                            .import_hints
                            .as_ref()
                            .map_or(0, ExportImportHints::owned_bytes)
                })
                .sum::<usize>()
    }
}

pub fn convert_normal_convention(
    mut normal: [f32; 3],
    from: NormalConvention,
    to: NormalConvention,
) -> [f32; 3] {
    if from != to {
        normal[1] = -normal[1];
    }
    normal
}

fn valid_name(name: &str, tokens: bool) -> bool {
    if name.trim().is_empty()
        || name.ends_with(['.', ' '])
        || name.contains("..")
        || name
            .chars()
            .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
    {
        return false;
    }
    let mut rest = name;
    while let Some(pos) = rest.find(['{', '}']) {
        if !tokens || rest.as_bytes()[pos] != b'{' {
            return false;
        }
        let tail = &rest[pos + 1..];
        let Some(end) = tail.find('}') else {
            return false;
        };
        if !matches!(&tail[..end], "name" | "map" | "preset" | "index") {
            return false;
        }
        rest = &tail[end + 1..];
    }
    let stem = name
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    !matches!(
        stem.as_str(),
        "CON"
            | "CONIN$"
            | "CONOUT$"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
            | "COM¹"
            | "COM²"
            | "COM³"
            | "LPT¹"
            | "LPT²"
            | "LPT³"
    )
}

pub fn validate_export(plan: &ExportPlan) -> Result<(), CoreError> {
    let invalid = CoreError::InvalidMaterial;
    if !plan.displacement.midlevel.is_finite()
        || !(0.0..=1.0).contains(&plan.displacement.midlevel)
        || !plan.displacement.scale.is_finite()
        || plan.displacement.scale <= 0.0
    {
        return Err(CoreError::InvalidMaterial("invalid displacement settings"));
    }
    plan.source_extent.validate()?;
    let extent = match plan.size {
        SizePolicy::Keep => plan.source_extent,
        SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
            Extent { width, height }
        }
    };
    extent.validate()?;
    if matches!(plan.size, SizePolicy::Pad { .. })
        && (extent.width < plan.source_extent.width || extent.height < plan.source_extent.height)
    {
        return Err(invalid(
            "Padding size must be at least the source image size",
        ));
    }
    if !valid_name(&plan.naming, true) {
        return Err(invalid(
            "Naming template contains invalid characters or tokens",
        ));
    }
    if !valid_name(&plan.profile.id, false) || plan.profile.revision == 0 {
        return Err(invalid("Export preset has an invalid name or revision"));
    }
    if plan.outputs.is_empty() {
        return Err(invalid("Select at least one map to export"));
    }
    let mut names = std::collections::BTreeSet::new();
    for output in &plan.outputs {
        if let Some(quantization) = output.quantization
            && (!matches!(output.format, ExportFormat::Png8 | ExportFormat::Png16)
                || !quantization.clamp_min.is_finite()
                || !quantization.clamp_max.is_finite()
                || quantization.clamp_min < 0.0
                || quantization.clamp_max > 1.0
                || quantization.clamp_min >= quantization.clamp_max)
        {
            return Err(CoreError::InvalidMaterial(
                "invalid integer quantization settings",
            ));
        }
        if !valid_name(&output.name, false)
            || !names.insert(output.name.to_lowercase())
            || !(1..=4).contains(&output.channels.len())
        {
            return Err(invalid(
                "Output names must be valid and unique, with one to four channels",
            ));
        }
        for (index, channel) in output.channels.iter().enumerate() {
            if let ChannelSource::Constant(value) = channel.source
                && (!value.is_finite() || !(0.0..=1.0).contains(&value))
            {
                return Err(invalid("Channel constants must be between zero and one"));
            }
            if let ChannelSource::Map(reference, component) = channel.source {
                let valid_component = match reference.port {
                    OutputPort::Albedo => true,
                    OutputPort::Normal => component != Component::A,
                    _ => component == Component::R,
                };
                if !valid_component {
                    return Err(CoreError::InvalidMaterial("invalid export map component"));
                }
            }
            if channel.transfer == Transfer::Srgb && (index == 3 || !albedo_rgb(channel)) {
                return Err(invalid(
                    "sRGB transfer is only supported for albedo RGB channels",
                ));
            }
        }
        if plan.profile.engine_safe_srgb && output.semantics().color_tag == ExportColorTag::Raw {
            return Err(invalid(
                "Preset mixes color and data channels; use a custom raw preset",
            ));
        }
        if output.format == ExportFormat::DdsBc5 && output.normal_has_negative_z {
            return Err(CoreError::InvalidMaterial(
                "negative-Z normals cannot be exported as BC5; use PNG for full XYZ",
            ));
        }
        match output.format {
            ExportFormat::Png8 | ExportFormat::Png16 if output.channels.len() == 2 => {
                return Err(CoreError::InvalidMaterial(
                    "PNG requires scalar, RGB or RGBA channels",
                ));
            }
            ExportFormat::Exr32
                if output.channels.len() != 1
                    || !matches!(
                        output.channels[0].source,
                        ChannelSource::Map(
                            OutputRef {
                                port: OutputPort::Height,
                                ..
                            },
                            Component::R
                        )
                    )
                    || output.channels[0].transfer != Transfer::Linear =>
            {
                return Err(invalid(
                    "EXR displacement requires one linear height channel",
                ));
            }
            ExportFormat::DdsBc5
                if output.channels.len() != 2
                    || !vector_channels(
                        &output.channels,
                        OutputPort::Normal,
                        &[Component::R, Component::G],
                    ) =>
            {
                return Err(invalid(
                    "BC5 normals require linear normal R and G channels",
                ));
            }
            _ => {}
        }
        if let Some(hints) = &output.import_hints {
            hints.validate(output)?;
        }
    }
    Ok(())
}

/// Structural validation plus reference/extent checks against the captured graph.
/// Settings and export requests both pass through this boundary.
pub fn validate_export_for_graph(plan: &ExportPlan, graph: &Graph) -> Result<(), CoreError> {
    validate_export(plan)?;
    let mut referenced = false;
    for output in &plan.outputs {
        for channel in &output.channels {
            if let ChannelSource::Map(reference, _) = channel.source {
                referenced = true;
                // This also checks port existence, including on legacy graphs.
                graph.output_normal_convention(reference)?;
                if export_extent(graph, reference)? != plan.source_extent {
                    return Err(CoreError::InvalidMaterial(
                        "export source extent does not match the captured graph",
                    ));
                }
            }
        }
    }
    if !referenced && export_extent(graph, graph.selected_output_ref())? != plan.source_extent {
        return Err(CoreError::InvalidMaterial(
            "export source extent does not match the captured graph",
        ));
    }
    Ok(())
}

fn export_extent(graph: &Graph, reference: OutputRef) -> Result<Extent, CoreError> {
    if graph.is_material() {
        return graph.material_output_extent(reference);
    }
    let mut node = graph
        .node(reference.node)
        .ok_or(CoreError::MissingNode(reference.node))?;
    loop {
        if let NodeParams::Source(source) = &node.params {
            return Ok(source.extent);
        }
        node = graph
            .node(node.input.ok_or(CoreError::MissingInput(node.id))?)
            .ok_or(CoreError::MissingNode(node.id))?;
    }
}
