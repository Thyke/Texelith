//! Schema-2 material DTOs. Only validated graphs expose these values as live state.

use crate::{
    CoreError, Extent, Graph, GraphDto, InputInterpretation, MapKind, Node, NodeId, NodeParams,
    ParameterId, SourceDescriptor,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MATERIAL_GRAPH_SCHEMA_VERSION: u32 = 2;
pub const MAX_MATERIAL_NODES: usize = 256;
pub const MAX_PYRAMID_BANDS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    Clamp,
    Repeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalConvention {
    OpenGl,
    DirectX,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalFilter {
    Central,
    Sobel,
    Scharr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskMode {
    Replace,
    Add,
    Subtract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputPort {
    Color,
    Height,
    Normal,
    Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputPort {
    Albedo,
    Height,
    Normal,
    Ao,
    Curvature,
    Edge,
    Cavity,
    Roughness,
    Metallic,
    Mask,
}

impl OutputPort {
    pub const fn map_kind(self) -> MapKind {
        match self {
            Self::Albedo => MapKind::Albedo,
            Self::Height => MapKind::Height,
            Self::Normal => MapKind::Normal,
            Self::Ao => MapKind::Ao,
            Self::Curvature => MapKind::Curvature,
            Self::Edge => MapKind::Edge,
            Self::Cavity => MapKind::Cavity,
            Self::Roughness => MapKind::Roughness,
            Self::Metallic => MapKind::Metallic,
            Self::Mask => MapKind::Mask,
        }
    }

    pub const fn input_port(self) -> Option<InputPort> {
        match self {
            Self::Albedo => Some(InputPort::Color),
            Self::Height => Some(InputPort::Height),
            Self::Normal => Some(InputPort::Normal),
            Self::Mask => Some(InputPort::Mask),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRef {
    pub node: NodeId,
    pub port: OutputPort,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputEdge {
    pub port: InputPort,
    pub source: OutputRef,
}

/// Resolved optional mask semantics; absence is a defined constant, not a lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskInput {
    ConstantOne,
    Output(OutputRef),
}

/// Named outputs are untrusted in a DTO and validated on conversion to `Graph`.
pub type MaterialOutputs = BTreeMap<MapKind, OutputRef>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SeamlessBase {
    OffsetBlend { overlap: f32 },
    Splat { seed: u32, grid: u32, overlap: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SeamlessMethod {
    OffsetBlend { overlap: f32 },
    Splat { seed: u32, grid: u32, overlap: f32 },
    Histogram { base: SeamlessBase, strength: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeamlessPlan {
    pub method: SeamlessMethod,
    pub boundary: Boundary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MaterialNodeParams {
    /// Schema-1 luminance/linear-red source, never promoted to color implicitly.
    LegacySource(SourceDescriptor),
    ColorSource(SourceDescriptor),
    HeightSource(SourceDescriptor),
    NormalSource {
        source: SourceDescriptor,
        convention: NormalConvention,
    },
    HeightFromAlbedo {
        band_weights: Vec<f32>,
        coarse_weight: f32,
        remove_coarse: bool,
        boundary: Boundary,
    },
    /// Explicit global min/max remap to [0, 1]; constant input becomes 0.5.
    /// Numeric height imports keep their original range unless this is requested.
    NormalizeHeight {},
    NormalFromHeight {
        strength: f32,
        filter: NormalFilter,
        scale_weights: Vec<f32>,
        boundary: Boundary,
        convention: NormalConvention,
    },
    HeightFromNormal {
        boundary: Boundary,
        v_cycles: u32,
        pre_smooth: u32,
        post_smooth: u32,
        coarsest_iterations: u32,
        relative_tolerance: f32,
        absolute_tolerance: f32,
    },
    HorizonAo {
        directions: u32,
        steps: u32,
        radius_source_px: f32,
        bias: f32,
        boundary: Boundary,
    },
    Curvature {
        edge_scale: f32,
        cavity_scale: f32,
        boundary: Boundary,
    },
    Roughness {
        curve: Vec<[f32; 2]>,
        base: f32,
        invert: bool,
    },
    Metallic {
        picked_color: [f32; 3],
        radius: f32,
        threshold: f32,
        feather: f32,
        mask_mode: MaskMode,
    },
    PaintMask {
        #[serde(default = "paint_boundary", skip_serializing_if = "is_paint_default")]
        boundary: Boundary,
        extent: Extent,
        stroke_digest: [u8; 32],
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        strokes: Vec<crate::BrushStroke>,
    },
    Levels {
        black: f32,
        white: f32,
        gamma: f32,
        boundary: Boundary,
    },
    Blur {
        sigma_source_px: f32,
        boundary: Boundary,
    },
    Seamless {
        method: SeamlessMethod,
        boundary: Boundary,
    },
}

fn paint_boundary() -> Boundary {
    Boundary::Repeat
}
fn is_paint_default(value: &Boundary) -> bool {
    *value == Boundary::Repeat
}

fn invalid(reason: &'static str) -> CoreError {
    CoreError::InvalidMaterial(reason)
}

fn finite(value: f32) -> Result<(), CoreError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid("material parameters must be finite"))
    }
}

fn range(value: f32, min: f32, max: f32) -> Result<(), CoreError> {
    finite(value)?;
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(invalid("material parameter outside its permitted range"))
    }
}

fn positive(value: f32) -> Result<(), CoreError> {
    finite(value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid("solver tolerances must be positive"))
    }
}

fn bands(values: &[f32]) -> Result<(), CoreError> {
    if !(1..=MAX_PYRAMID_BANDS).contains(&values.len()) {
        return Err(invalid("pyramid requires 1..=16 bands"));
    }
    values.iter().try_for_each(|value| finite(*value))
}

fn splat(grid: u32, overlap: f32) -> Result<(), CoreError> {
    if !(2..=8).contains(&grid) {
        return Err(invalid("splat grid must be in 2..=8"));
    }
    range(overlap, 0.1, 0.5)
}

impl SeamlessBase {
    pub fn validate(&self) -> Result<(), CoreError> {
        match *self {
            Self::OffsetBlend { overlap } => range(overlap, 0.1, 0.5),
            Self::Splat { grid, overlap, .. } => splat(grid, overlap),
        }
    }
}

impl SeamlessMethod {
    pub fn validate(&self) -> Result<(), CoreError> {
        match self {
            Self::OffsetBlend { overlap } => range(*overlap, 0.1, 0.5),
            Self::Splat { grid, overlap, .. } => splat(*grid, *overlap),
            Self::Histogram { base, strength } => {
                base.validate()?;
                range(*strength, 0.0, 1.0)
            }
        }
    }
}

impl MaterialNodeParams {
    pub fn validate(&self) -> Result<(), CoreError> {
        match self {
            Self::LegacySource(source) => NodeParams::Source(source.clone()).validate(),
            Self::ColorSource(source)
            | Self::HeightSource(source)
            | Self::NormalSource { source, .. } => {
                source.extent.validate()?;
                let expected = match self {
                    Self::ColorSource(_) => InputInterpretation::SrgbPhoto,
                    Self::HeightSource(_) => InputInterpretation::LinearHeight,
                    _ => InputInterpretation::LinearNormal,
                };
                if source.interpretation != expected {
                    return Err(invalid(
                        "source interpretation does not match its semantic type",
                    ));
                }
                Ok(())
            }
            Self::HeightFromAlbedo {
                band_weights,
                coarse_weight,
                ..
            } => {
                bands(band_weights)?;
                finite(*coarse_weight)
            }
            Self::NormalizeHeight {} => Ok(()),
            Self::NormalFromHeight {
                strength,
                scale_weights,
                ..
            } => {
                ParameterId::Strength.validate(*strength)?;
                bands(scale_weights)
            }
            Self::HeightFromNormal {
                v_cycles,
                pre_smooth,
                post_smooth,
                coarsest_iterations,
                relative_tolerance,
                absolute_tolerance,
                ..
            } => {
                if !(1..=16).contains(v_cycles)
                    || *pre_smooth == 0
                    || *post_smooth == 0
                    || *coarsest_iterations == 0
                {
                    return Err(invalid("invalid solver iteration budget"));
                }
                positive(*relative_tolerance)?;
                positive(*absolute_tolerance)
            }
            Self::HorizonAo {
                directions,
                steps,
                radius_source_px,
                bias,
                ..
            } => {
                if !(4..=32).contains(directions) || !(4..=32).contains(steps) {
                    return Err(invalid("AO directions and steps must be in 4..=32"));
                }
                range(*radius_source_px, 1.0, 512.0)?;
                range(*bias, 0.0, f32::MAX)
            }
            Self::Curvature {
                edge_scale,
                cavity_scale,
                ..
            } => {
                range(*edge_scale, 0.0, f32::MAX)?;
                range(*cavity_scale, 0.0, f32::MAX)
            }
            Self::Roughness { curve, base, .. } => {
                if !(2..=16).contains(&curve.len()) {
                    return Err(invalid("roughness curve requires 2..=16 knots"));
                }
                for [x, y] in curve {
                    range(*x, 0.0, 1.0)?;
                    range(*y, 0.0, 1.0)?;
                }
                if curve.windows(2).any(|pair| pair[0][0] >= pair[1][0]) {
                    return Err(invalid("roughness curve abscissas must strictly increase"));
                }
                range(*base, 0.0, 1.0)
            }
            Self::Metallic {
                picked_color,
                radius,
                threshold,
                feather,
                ..
            } => {
                for value in picked_color {
                    range(*value, 0.0, 1.0)?;
                }
                range(*radius, 0.0, 3.0_f32.sqrt())?;
                range(*threshold, 0.0, 1.0)?;
                range(*feather, 0.0, f32::MAX)
            }
            Self::PaintMask {
                extent,
                stroke_digest,
                strokes,
                ..
            } => {
                extent.validate()?;
                let mut ids = std::collections::BTreeSet::new();
                for stroke in strokes {
                    if !ids.insert(stroke.id.0) {
                        return Err(invalid("duplicate stroke ID"));
                    }
                }
                // Digest-only schema-2 masks remain readable with their original
                // cache identity. Vector-bearing masks must prove their identity.
                if !strokes.is_empty() && crate::stroke_digest(strokes)? != *stroke_digest {
                    return Err(invalid("stroke digest does not match vector data"));
                }
                Ok(())
            }
            Self::Levels {
                black,
                white,
                gamma,
                ..
            } => NodeParams::Levels {
                black: *black,
                white: *white,
                gamma: *gamma,
            }
            .validate(),
            Self::Blur {
                sigma_source_px, ..
            } => ParameterId::SigmaSourcePx.validate(*sigma_source_px),
            Self::Seamless { method, .. } => method.validate(),
        }
    }

    /// Convention of a normal-valued output. Consumers resolve it from the source
    /// reference and canonicalize once; no convention is inferred from a filename.
    pub const fn normal_convention(&self) -> Option<NormalConvention> {
        match self {
            Self::NormalSource { convention, .. } | Self::NormalFromHeight { convention, .. } => {
                Some(*convention)
            }
            _ => None,
        }
    }

    fn source_extent(&self) -> Option<Extent> {
        match self {
            Self::LegacySource(source)
            | Self::ColorSource(source)
            | Self::HeightSource(source)
            | Self::NormalSource { source, .. } => Some(source.extent),
            Self::PaintMask { extent, .. } => Some(*extent),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialNode {
    pub id: NodeId,
    /// Canonical order: required semantic input first, optional mask second.
    /// No mask edge means constant one. Explicit references are never defaulted.
    pub inputs: Vec<InputEdge>,
    pub params: MaterialNodeParams,
}

impl MaterialNode {
    pub fn mask_input(&self) -> Result<Option<MaskInput>, CoreError> {
        self.validate()?;
        if !matches!(
            self.params,
            MaterialNodeParams::Roughness { .. } | MaterialNodeParams::Metallic { .. }
        ) {
            return Ok(None);
        }
        Ok(Some(
            self.inputs.get(1).map_or(MaskInput::ConstantOne, |edge| {
                MaskInput::Output(edge.source)
            }),
        ))
    }

    pub fn primary_output(&self) -> OutputPort {
        match &self.params {
            MaterialNodeParams::ColorSource(_) => OutputPort::Albedo,
            MaterialNodeParams::NormalSource { .. }
            | MaterialNodeParams::NormalFromHeight { .. } => OutputPort::Normal,
            MaterialNodeParams::HorizonAo { .. } => OutputPort::Ao,
            MaterialNodeParams::Curvature { .. } => OutputPort::Curvature,
            MaterialNodeParams::Roughness { .. } => OutputPort::Roughness,
            MaterialNodeParams::Metallic { .. } => OutputPort::Metallic,
            MaterialNodeParams::PaintMask { .. } => OutputPort::Mask,
            MaterialNodeParams::Seamless { .. }
                if self
                    .inputs
                    .first()
                    .is_some_and(|input| input.port == InputPort::Color) =>
            {
                OutputPort::Albedo
            }
            _ => OutputPort::Height,
        }
    }

    pub fn has_output(&self, port: OutputPort) -> bool {
        self.primary_output() == port
            || (matches!(self.params, MaterialNodeParams::Curvature { .. })
                && matches!(port, OutputPort::Edge | OutputPort::Cavity))
    }

    /// Per-node validation also used by the cache boundary. Parent references and
    /// extents require the full graph and are checked during DTO conversion.
    pub fn validate(&self) -> Result<(), CoreError> {
        self.params.validate()?;
        use MaterialNodeParams as P;
        let (required, optional_mask) = match self.params {
            P::LegacySource(_)
            | P::ColorSource(_)
            | P::HeightSource(_)
            | P::NormalSource { .. }
            | P::PaintMask { .. } => (None, false),
            P::HeightFromAlbedo { .. } => (Some(InputPort::Color), false),
            P::NormalFromHeight { .. }
            | P::NormalizeHeight {}
            | P::HorizonAo { .. }
            | P::Levels { .. }
            | P::Blur { .. } => (Some(InputPort::Height), false),
            P::HeightFromNormal { .. } | P::Curvature { .. } => (Some(InputPort::Normal), false),
            P::Roughness { .. } | P::Metallic { .. } => (Some(InputPort::Color), true),
            P::Seamless { .. } => {
                let port = self
                    .inputs
                    .first()
                    .map(|edge| edge.port)
                    .ok_or(CoreError::MissingInput(self.id))?;
                if !matches!(port, InputPort::Color | InputPort::Height) {
                    return Err(invalid("seamless requires color or height"));
                }
                (Some(port), false)
            }
        };
        let minimum = usize::from(required.is_some());
        if self.inputs.len() < minimum || self.inputs.len() > minimum + usize::from(optional_mask) {
            return Err(invalid("incorrect material input arity"));
        }
        for (index, edge) in self.inputs.iter().enumerate() {
            let expected = if index == 0 {
                required
            } else {
                Some(InputPort::Mask)
            };
            if Some(edge.port) != expected || edge.source.port.input_port() != Some(edge.port) {
                return Err(invalid("incorrect material input semantic or port order"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialGraphDto {
    pub schema_version: u32,
    pub graph: Vec<MaterialNode>,
    pub outputs: MaterialOutputs,
    pub selected_output: OutputRef,
}

impl MaterialGraphDto {
    pub(crate) fn node(&self, id: NodeId) -> Option<&MaterialNode> {
        self.graph.iter().find(|node| node.id == id)
    }

    fn index(&self, id: NodeId) -> Result<usize, CoreError> {
        self.graph
            .iter()
            .position(|node| node.id == id)
            .ok_or(CoreError::MissingNode(id))
    }

    fn check_output(&self, output: OutputRef) -> Result<(), CoreError> {
        let node = self
            .node(output.node)
            .ok_or(CoreError::MissingNode(output.node))?;
        if !node.has_output(output.port) {
            return Err(invalid("node does not provide the referenced output port"));
        }
        Ok(())
    }

    fn topological_indices(&self) -> Result<Vec<usize>, CoreError> {
        // Kahn traversal covers disconnected branches and avoids recursive stacks.
        let mut pending: Vec<_> = self.graph.iter().map(|node| node.inputs.len()).collect();
        let mut consumers = vec![Vec::new(); self.graph.len()];
        for (index, node) in self.graph.iter().enumerate() {
            for input in &node.inputs {
                consumers[self.index(input.source.node)?].push(index);
            }
        }
        let mut ready: Vec<_> = pending
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count == 0).then_some(index))
            .collect();
        let mut ordered = Vec::with_capacity(self.graph.len());
        while let Some(index) = ready.pop() {
            ordered.push(index);
            for consumer in &consumers[index] {
                pending[*consumer] -= 1;
                if pending[*consumer] == 0 {
                    ready.push(*consumer);
                }
            }
        }
        if ordered.len() != self.graph.len() {
            let index = pending
                .iter()
                .position(|count| *count != 0)
                .expect("unvisited cycle");
            return Err(CoreError::Cycle(self.graph[index].id));
        }
        Ok(ordered)
    }

    fn extents(&self) -> Result<Vec<Extent>, CoreError> {
        let mut extents = vec![None; self.graph.len()];
        for index in self.topological_indices()? {
            let node = &self.graph[index];
            let mut extent = node.params.source_extent();
            for input in &node.inputs {
                let parent =
                    extents[self.index(input.source.node)?].expect("parent before consumer");
                if extent.is_some_and(|value| value != parent) {
                    return Err(invalid(
                        "material input extents must match; resizing is explicit",
                    ));
                }
                extent = Some(parent);
            }
            extents[index] = Some(extent.ok_or_else(|| invalid("node has no extent"))?);
        }
        Ok(extents
            .into_iter()
            .map(|extent| extent.expect("all nodes visited"))
            .collect())
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        if self.schema_version != MATERIAL_GRAPH_SCHEMA_VERSION {
            return Err(CoreError::UnknownSchema(self.schema_version));
        }
        if self.graph.is_empty() || self.graph.len() > MAX_MATERIAL_NODES {
            return Err(invalid("material graph requires 1..=256 nodes"));
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut brush_points = 0;
        for node in &self.graph {
            if !ids.insert(node.id) {
                return Err(CoreError::DuplicateNode(node.id));
            }
            node.validate()?;
            if let MaterialNodeParams::PaintMask { strokes, .. } = &node.params {
                brush_points += strokes
                    .iter()
                    .map(|stroke| stroke.points.len())
                    .sum::<usize>();
                if brush_points > crate::MAX_BRUSH_POINTS {
                    return Err(invalid("graph brush point limit exceeded"));
                }
            }
        }
        for node in &self.graph {
            for input in &node.inputs {
                self.check_output(input.source)?;
            }
        }
        if self.outputs.is_empty() {
            return Err(invalid("material graph requires named outputs"));
        }
        for (kind, output) in &self.outputs {
            self.check_output(*output)?;
            if *kind != output.port.map_kind() {
                return Err(invalid("named map does not match its output port"));
            }
        }
        self.check_output(self.selected_output)?;
        self.extents()?;
        Ok(())
    }
}

impl TryFrom<MaterialGraphDto> for Graph {
    type Error = CoreError;

    fn try_from(mut dto: MaterialGraphDto) -> Result<Self, Self::Error> {
        dto.validate()?;
        dto.graph.sort_unstable_by_key(|node| node.id);
        Ok(Self::from_material(dto))
    }
}

impl Graph {
    pub fn is_material(&self) -> bool {
        self.material.is_some()
    }

    /// Immutable material nodes sorted by stable ID, empty for a legacy graph.
    pub fn material_nodes(&self) -> &[MaterialNode] {
        self.material.as_ref().map_or(&[], |dto| &dto.graph)
    }

    pub fn material_node(&self, id: NodeId) -> Option<&MaterialNode> {
        self.material.as_ref()?.node(id)
    }

    pub fn material_outputs(&self) -> Option<&MaterialOutputs> {
        self.material.as_ref().map(|dto| &dto.outputs)
    }

    pub fn selected_output_ref(&self) -> OutputRef {
        self.material.as_ref().map_or_else(
            || OutputRef {
                node: self.selected_output(),
                port: legacy_output(
                    self.node(self.selected_output())
                        .expect("validated selection"),
                ),
            },
            |dto| dto.selected_output,
        )
    }

    pub fn topological_material_nodes(&self) -> Vec<&MaterialNode> {
        self.material.as_ref().map_or_else(Vec::new, |dto| {
            dto.topological_indices()
                .expect("validated graph")
                .into_iter()
                .map(|index| &dto.graph[index])
                .collect()
        })
    }

    pub fn material_output_extent(&self, output: OutputRef) -> Result<Extent, CoreError> {
        let dto = self
            .material
            .as_ref()
            .ok_or_else(|| invalid("requires a material graph"))?;
        dto.check_output(output)?;
        Ok(dto.extents()?[dto.index(output.node)?])
    }

    pub fn output_normal_convention(
        &self,
        output: OutputRef,
    ) -> Result<Option<NormalConvention>, CoreError> {
        if let Some(dto) = &self.material {
            dto.check_output(output)?;
            return Ok(dto
                .node(output.node)
                .expect("checked output")
                .params
                .normal_convention());
        }
        let node = self
            .node(output.node)
            .ok_or(CoreError::MissingNode(output.node))?;
        if legacy_output(node) != output.port {
            return Err(invalid("invalid legacy output port"));
        }
        Ok((output.port == OutputPort::Normal).then_some(NormalConvention::OpenGl))
    }

    /// Fallible schema-1 adapter for callers that may receive either graph kind.
    pub fn legacy_dto(&self) -> Result<GraphDto, CoreError> {
        GraphDto::try_from(self.clone())
    }
}

fn legacy_output(node: &Node) -> OutputPort {
    if matches!(node.params, NodeParams::NormalFromHeight { .. }) {
        OutputPort::Normal
    } else {
        OutputPort::Height
    }
}

impl From<Graph> for MaterialGraphDto {
    fn from(mut graph: Graph) -> Self {
        if let Some(dto) = graph.material.take() {
            return dto;
        }
        let selected_output = graph.selected_output_ref();
        let mut outputs = MaterialOutputs::new();
        for node in graph.topological_nodes() {
            let port = legacy_output(node);
            outputs.insert(
                port.map_kind(),
                OutputRef {
                    node: node.id,
                    port,
                },
            );
        }
        outputs.insert(selected_output.port.map_kind(), selected_output);
        let nodes = graph
            .nodes()
            .iter()
            .map(|node| {
                let params = match &node.params {
                    NodeParams::Source(source) => MaterialNodeParams::LegacySource(source.clone()),
                    NodeParams::Levels {
                        black,
                        white,
                        gamma,
                    } => MaterialNodeParams::Levels {
                        black: *black,
                        white: *white,
                        gamma: *gamma,
                        boundary: Boundary::Clamp,
                    },
                    NodeParams::Blur { sigma_source_px } => MaterialNodeParams::Blur {
                        sigma_source_px: *sigma_source_px,
                        boundary: Boundary::Clamp,
                    },
                    NodeParams::NormalFromHeight { strength } => {
                        MaterialNodeParams::NormalFromHeight {
                            strength: *strength,
                            filter: NormalFilter::Central,
                            scale_weights: vec![1.0],
                            boundary: Boundary::Clamp,
                            convention: NormalConvention::OpenGl,
                        }
                    }
                };
                MaterialNode {
                    id: node.id,
                    inputs: node
                        .input
                        .map(|parent| InputEdge {
                            port: InputPort::Height,
                            source: OutputRef {
                                node: parent,
                                port: OutputPort::Height,
                            },
                        })
                        .into_iter()
                        .collect(),
                    params,
                }
            })
            .collect();
        Self {
            schema_version: MATERIAL_GRAPH_SCHEMA_VERSION,
            graph: nodes,
            outputs,
            selected_output,
        }
    }
}

impl TryFrom<GraphDto> for MaterialGraphDto {
    type Error = CoreError;

    fn try_from(dto: GraphDto) -> Result<Self, Self::Error> {
        let migrated = Self::from(Graph::try_from(dto)?);
        migrated.validate()?;
        Ok(migrated)
    }
}

/// Builds a new Repeat material. The RGBA8 photo and legacy scalar factories are
/// deliberately separate. Imported normals default to OpenGL and integrate via
/// their own HeightFromNormal branch; edit the DTO to specify DirectX explicitly.
/// Uses raw procedural defaults; the application import factory adds its photo preset.
pub fn material_from_source(source: SourceDescriptor) -> Result<Graph, CoreError> {
    material_from_source_with_defaults(source, false)
}

fn material_from_source_with_defaults(
    source: SourceDescriptor,
    photo_defaults: bool,
) -> Result<Graph, CoreError> {
    source.extent.validate()?;
    let interpretation = source.interpretation;
    let mut nodes = Vec::new();
    let mut outputs = MaterialOutputs::new();
    let mut add = |params, inputs: Vec<InputEdge>, ports: &[OutputPort]| {
        let id = NodeId(nodes.len() as u64 + 1);
        nodes.push(MaterialNode { id, inputs, params });
        for port in ports {
            outputs.insert(
                port.map_kind(),
                OutputRef {
                    node: id,
                    port: *port,
                },
            );
        }
        OutputRef {
            node: id,
            port: ports[0],
        }
    };
    let edge = |source: OutputRef| InputEdge {
        port: source.port.input_port().expect("factory semantic input"),
        source,
    };
    let (height, imported_normal, albedo) = match interpretation {
        InputInterpretation::SrgbPhoto => {
            let color = add(
                MaterialNodeParams::ColorSource(source),
                vec![],
                &[OutputPort::Albedo],
            );
            let albedo = add(
                MaterialNodeParams::Seamless {
                    method: SeamlessMethod::OffsetBlend { overlap: 0.25 },
                    boundary: Boundary::Repeat,
                },
                vec![edge(color)],
                &[OutputPort::Albedo],
            );
            let height = add(
                MaterialNodeParams::HeightFromAlbedo {
                    // Suppress pixel grain, retaining the Gaussian shape bands.
                    band_weights: if photo_defaults {
                        vec![0.0, 0.5, 1.0]
                    } else {
                        vec![1.0]
                    },
                    coarse_weight: 1.0,
                    remove_coarse: false,
                    boundary: Boundary::Repeat,
                },
                vec![edge(albedo)],
                &[OutputPort::Height],
            );
            let height = if photo_defaults {
                add(
                    MaterialNodeParams::NormalizeHeight {},
                    vec![edge(height)],
                    &[OutputPort::Height],
                )
            } else {
                height
            };
            (height, None, Some(albedo))
        }
        InputInterpretation::LinearHeight => {
            let height = add(
                MaterialNodeParams::HeightSource(source),
                vec![],
                &[OutputPort::Height],
            );
            let height = add(
                MaterialNodeParams::Seamless {
                    method: SeamlessMethod::OffsetBlend { overlap: 0.25 },
                    boundary: Boundary::Repeat,
                },
                vec![edge(height)],
                &[OutputPort::Height],
            );
            (height, None, None)
        }
        InputInterpretation::LinearNormal => {
            let normal = add(
                MaterialNodeParams::NormalSource {
                    source,
                    convention: NormalConvention::OpenGl,
                },
                vec![],
                &[OutputPort::Normal],
            );
            let height = add(
                MaterialNodeParams::HeightFromNormal {
                    boundary: Boundary::Repeat,
                    v_cycles: 8,
                    pre_smooth: 3,
                    post_smooth: 3,
                    coarsest_iterations: 32,
                    relative_tolerance: 1e-3,
                    absolute_tolerance: 1e-6,
                },
                vec![edge(normal)],
                &[OutputPort::Height],
            );
            (height, Some(normal), None)
        }
    };
    let normal = if let Some(normal) = imported_normal {
        normal
    } else {
        add(
            MaterialNodeParams::NormalFromHeight {
                // Derivatives are in UV units, so photo relief needs a small scale.
                strength: if photo_defaults && albedo.is_some() {
                    0.015
                } else {
                    1.0
                },
                filter: NormalFilter::Sobel,
                scale_weights: if photo_defaults && albedo.is_some() {
                    vec![0.0, 0.0, 1.0]
                } else {
                    vec![1.0]
                },
                boundary: Boundary::Repeat,
                convention: NormalConvention::OpenGl,
            },
            vec![edge(height)],
            &[OutputPort::Normal],
        )
    };
    add(
        MaterialNodeParams::HorizonAo {
            directions: 8,
            steps: 8,
            radius_source_px: 32.0,
            bias: 0.01,
            boundary: Boundary::Repeat,
        },
        vec![edge(height)],
        &[OutputPort::Ao],
    );
    add(
        MaterialNodeParams::Curvature {
            edge_scale: 1.0,
            cavity_scale: 1.0,
            boundary: Boundary::Repeat,
        },
        vec![edge(normal)],
        &[OutputPort::Curvature, OutputPort::Edge, OutputPort::Cavity],
    );
    if let Some(albedo) = albedo {
        add(
            MaterialNodeParams::Roughness {
                curve: if photo_defaults {
                    vec![[0.0, 0.8], [0.25, 0.6], [1.0, 0.45]]
                } else {
                    vec![[0.0, 0.0], [1.0, 1.0]]
                },
                base: if photo_defaults { 0.65 } else { 0.5 },
                invert: false,
            },
            vec![edge(albedo)],
            &[OutputPort::Roughness],
        );
        add(
            MaterialNodeParams::Metallic {
                picked_color: [0.5; 3],
                radius: 0.25,
                threshold: 0.5,
                feather: 0.1,
                // With the defined absent-mask value of one, subtraction is
                // exactly zero for every photo. Metal must be explicitly enabled.
                mask_mode: if photo_defaults {
                    MaskMode::Subtract
                } else {
                    MaskMode::Replace
                },
            },
            vec![edge(albedo)],
            &[OutputPort::Metallic],
        );
    }
    Graph::try_from(MaterialGraphDto {
        schema_version: MATERIAL_GRAPH_SCHEMA_VERSION,
        graph: nodes,
        outputs,
        selected_output: normal,
    })
}

/// Application import factory with a photo preset and editable mask. Typed height
/// and normal imports retain their original ranges and procedural defaults.
pub fn demo_material_from_source(source: SourceDescriptor) -> Result<Graph, CoreError> {
    let extent = source.extent;
    let mut dto = MaterialGraphDto::from(material_from_source_with_defaults(source, true)?);
    // Levels clamps even at (0, 1, 1). New materials must retain signed/HDR
    // height, including mean-zero normal integration, until remapping is requested.
    // Photo-derived height already has its explicit NormalizeHeight stage.
    let id = NodeId(dto.graph.len() as u64 + 1);
    dto.graph.push(MaterialNode {
        id,
        params: MaterialNodeParams::PaintMask {
            boundary: Boundary::Repeat,
            extent,
            stroke_digest: [0; 32],
            strokes: vec![],
        },
        inputs: vec![],
    });
    dto.outputs.insert(
        MapKind::Mask,
        OutputRef {
            node: id,
            port: OutputPort::Mask,
        },
    );
    Graph::try_from(dto)
}

/// Rewire optional mask links and the seamless stage atomically, retaining node edits.
pub fn configure_material(
    graph: &Graph,
    seamless: bool,
    painted_mask: bool,
) -> Result<Graph, CoreError> {
    if !graph.is_material() {
        return Err(invalid("material configuration requires a material graph"));
    }
    let mut dto = MaterialGraphDto::from(graph.clone());
    let stages: Vec<_> = dto
        .graph
        .iter()
        .filter_map(|n| match n.params {
            MaterialNodeParams::Seamless { .. } => Some((
                OutputRef {
                    node: n.id,
                    port: n.inputs[0].source.port,
                },
                n.inputs[0].source,
            )),
            _ => None,
        })
        .collect();
    for (stage, original) in stages {
        let target = if seamless { stage } else { original };
        for node in &mut dto.graph {
            if node.id == stage.node {
                continue;
            }
            for edge in &mut node.inputs {
                if edge.source == stage || edge.source == original {
                    edge.source = target;
                }
            }
        }
        for output in dto.outputs.values_mut() {
            if *output == stage || *output == original {
                *output = target;
            }
        }
        if dto.selected_output == stage || dto.selected_output == original {
            dto.selected_output = target;
        }
    }
    let mask = dto.outputs.get(&MapKind::Mask).copied();
    for node in &mut dto.graph {
        if matches!(
            node.params,
            MaterialNodeParams::Roughness { .. } | MaterialNodeParams::Metallic { .. }
        ) {
            node.inputs.retain(|edge| edge.port != InputPort::Mask);
            if painted_mask {
                node.inputs.push(InputEdge {
                    port: InputPort::Mask,
                    source: mask.ok_or_else(|| invalid("material has no painted mask"))?,
                });
            }
        }
    }
    Graph::try_from(dto)
}

/// Discover the demo's optional seam stage and painted-mask connections.
pub fn material_options(graph: &Graph) -> (bool, bool) {
    let is_seam = |id| {
        graph
            .material_node(id)
            .is_some_and(|node| matches!(node.params, MaterialNodeParams::Seamless { .. }))
    };
    let seamless = graph
        .material_outputs()
        .is_some_and(|outputs| outputs.values().any(|output| is_seam(output.node)))
        || graph
            .material_nodes()
            .iter()
            .any(|node| node.inputs.iter().any(|edge| is_seam(edge.source.node)));
    let painted_mask = graph
        .material_nodes()
        .iter()
        .any(|node| node.inputs.iter().any(|edge| edge.port == InputPort::Mask));
    (seamless, painted_mask)
}
