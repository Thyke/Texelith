use serde::{Deserialize, Serialize};
use std::fmt;

pub const MAX_DIMENSION: u32 = 4096;
pub const GRAPH_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    pub fn new(width: u32, height: u32) -> Result<Self, CoreError> {
        let extent = Self { width, height };
        extent.validate()?;
        Ok(extent)
    }

    pub fn validate(self) -> Result<(), CoreError> {
        if (1..=MAX_DIMENSION).contains(&self.width) && (1..=MAX_DIMENSION).contains(&self.height) {
            Ok(())
        } else {
            Err(CoreError::InvalidExtent(self))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputInterpretation {
    SrgbPhoto,
    LinearHeight,
    LinearNormal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDescriptor {
    /// BLAKE3 of decoded RGBA8 pixels; see the crate's canonical encoding rule.
    pub content_hash: [u8; 32],
    pub extent: Extent,
    pub interpretation: InputInterpretation,
}

impl SourceDescriptor {
    pub fn from_rgba8(
        pixels: &[u8],
        extent: Extent,
        interpretation: InputInterpretation,
    ) -> Result<Self, CoreError> {
        extent.validate()?;
        // The 4096 cap makes this multiplication safe even on wasm32.
        let expected = extent.width as usize * extent.height as usize * 4;
        if pixels.len() != expected {
            return Err(CoreError::InvalidPixelLength {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            content_hash: *blake3::hash(pixels).as_bytes(),
            extent,
            interpretation,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MapKind {
    Height,
    Normal,
    Albedo,
    Ao,
    Curvature,
    Edge,
    Cavity,
    Roughness,
    Metallic,
    Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MapFormat {
    R32Float,
    Rgba16Float,
}

impl MapKind {
    pub const fn format(self) -> MapFormat {
        match self {
            Self::Normal | Self::Albedo => MapFormat::Rgba16Float,
            _ => MapFormat::R32Float,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Source,
    Levels,
    Blur,
    NormalFromHeight,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NodeParams {
    Source(SourceDescriptor),
    Levels { black: f32, white: f32, gamma: f32 },
    Blur { sigma_source_px: f32 },
    NormalFromHeight { strength: f32 },
}

impl NodeParams {
    pub const fn kind(&self) -> NodeKind {
        match self {
            Self::Source(_) => NodeKind::Source,
            Self::Levels { .. } => NodeKind::Levels,
            Self::Blur { .. } => NodeKind::Blur,
            Self::NormalFromHeight { .. } => NodeKind::NormalFromHeight,
        }
    }

    pub const fn output_kind(&self) -> MapKind {
        match self {
            Self::NormalFromHeight { .. } => MapKind::Normal,
            _ => MapKind::Height,
        }
    }

    pub fn default_for(kind: NodeKind, source: SourceDescriptor) -> Self {
        match kind {
            NodeKind::Source => Self::Source(source),
            NodeKind::Levels => Self::Levels {
                black: 0.0,
                white: 1.0,
                gamma: 1.0,
            },
            NodeKind::Blur => Self::Blur {
                sigma_source_px: 0.0,
            },
            NodeKind::NormalFromHeight => Self::NormalFromHeight { strength: 1.0 },
        }
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        match *self {
            Self::Source(ref source) => {
                source.extent.validate()?;
                if source.interpretation == InputInterpretation::LinearNormal {
                    return Err(CoreError::InvalidMaterial(
                        "legacy source cannot interpret normals",
                    ));
                }
                Ok(())
            }
            Self::Levels {
                black,
                white,
                gamma,
            } => {
                ParameterId::Black.validate(black)?;
                ParameterId::White.validate(white)?;
                ParameterId::Gamma.validate(gamma)?;
                if white - black < 1e-6 {
                    return Err(CoreError::InvalidLevelsInterval);
                }
                Ok(())
            }
            Self::Blur { sigma_source_px } => ParameterId::SigmaSourcePx.validate(sigma_source_px),
            Self::NormalFromHeight { strength } => ParameterId::Strength.validate(strength),
        }
    }

    pub fn parameter(&self, parameter: ParameterId) -> Result<f32, CoreError> {
        match (self, parameter) {
            (Self::Levels { black, .. }, ParameterId::Black) => Ok(*black),
            (Self::Levels { white, .. }, ParameterId::White) => Ok(*white),
            (Self::Levels { gamma, .. }, ParameterId::Gamma) => Ok(*gamma),
            (Self::Blur { sigma_source_px }, ParameterId::SigmaSourcePx) => Ok(*sigma_source_px),
            (Self::NormalFromHeight { strength }, ParameterId::Strength) => Ok(*strength),
            _ => Err(CoreError::WrongParameter {
                kind: self.kind(),
                parameter,
            }),
        }
    }

    /// Builds a validated replacement; the original value is unchanged on error.
    pub fn with_parameter(&self, parameter: ParameterId, value: f32) -> Result<Self, CoreError> {
        self.parameter(parameter)?;
        let mut replacement = self.clone();
        match (&mut replacement, parameter) {
            (Self::Levels { black, .. }, ParameterId::Black) => *black = value,
            (Self::Levels { white, .. }, ParameterId::White) => *white = value,
            (Self::Levels { gamma, .. }, ParameterId::Gamma) => *gamma = value,
            (Self::Blur { sigma_source_px }, ParameterId::SigmaSourcePx) => {
                *sigma_source_px = value
            }
            (Self::NormalFromHeight { strength }, ParameterId::Strength) => *strength = value,
            _ => unreachable!("parameter membership checked above"),
        }
        replacement.validate()?;
        Ok(replacement)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterId {
    Black,
    White,
    Gamma,
    SigmaSourcePx,
    Strength,
    BandWeights,
    CoarseWeight,
    RemoveCoarse,
    Boundary,
    Filter,
    ScaleWeights,
    Convention,
    VCycles,
    PreSmooth,
    PostSmooth,
    CoarsestIterations,
    RelativeTolerance,
    AbsoluteTolerance,
    Directions,
    Steps,
    RadiusSourcePx,
    Bias,
    EdgeScale,
    CavityScale,
    Curve,
    Base,
    Invert,
    PickedColor,
    Radius,
    Threshold,
    Feather,
    MaskMode,
    SeamlessMethod,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub id: ParameterId,
    pub node_kind: NodeKind,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

pub const PARAMETER_DESCRIPTORS: [ParameterDescriptor; 5] = [
    ParameterDescriptor {
        id: ParameterId::Black,
        node_kind: NodeKind::Levels,
        min: 0.0,
        max: 1.0,
        default: 0.0,
    },
    ParameterDescriptor {
        id: ParameterId::White,
        node_kind: NodeKind::Levels,
        min: 0.0,
        max: 1.0,
        default: 1.0,
    },
    ParameterDescriptor {
        id: ParameterId::Gamma,
        node_kind: NodeKind::Levels,
        min: 0.1,
        max: 10.0,
        default: 1.0,
    },
    ParameterDescriptor {
        id: ParameterId::SigmaSourcePx,
        node_kind: NodeKind::Blur,
        min: 0.0,
        max: 5.0,
        default: 0.0,
    },
    ParameterDescriptor {
        id: ParameterId::Strength,
        node_kind: NodeKind::NormalFromHeight,
        min: 0.0,
        max: 10.0,
        default: 1.0,
    },
];

impl ParameterId {
    /// Legacy scalar descriptor. Material consumers should use `material_descriptor`.
    ///
    /// # Panics
    /// Panics for IDs outside the five legacy scalar controls.
    pub fn descriptor(self) -> &'static ParameterDescriptor {
        &PARAMETER_DESCRIPTORS[match self {
            Self::Black => 0,
            Self::White => 1,
            Self::Gamma => 2,
            Self::SigmaSourcePx => 3,
            Self::Strength => 4,
            _ => panic!("material parameters use ParameterId::material_descriptor"),
        }]
    }

    pub fn validate(self, value: f32) -> Result<(), CoreError> {
        if !value.is_finite() {
            return Err(CoreError::NonFiniteParameter(self));
        }
        let crate::ParameterSchema::Float {
            min,
            max,
            min_exclusive,
            ..
        } = self.material_descriptor().value_schema
        else {
            return Err(CoreError::InvalidMaterial(
                "parameter requires a non-float value",
            ));
        };
        if !(min..=max).contains(&value) || (min_exclusive && value == min) {
            return Err(CoreError::ParameterOutOfRange(self));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// An untrusted construction/serialization value. Live graphs expose nodes read-only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub input: Option<NodeId>,
    pub params: NodeParams,
}

/// A validated DAG whose node storage is sorted by stable ID, not execution order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "GraphEnvelope", into = "GraphEnvelope")]
pub struct Graph {
    nodes: Vec<Node>,
    selected_output: NodeId,
    pub(crate) material: Option<crate::MaterialGraphDto>,
}

impl Graph {
    pub fn new(mut nodes: Vec<Node>, selected_output: NodeId) -> Result<Self, CoreError> {
        nodes.sort_unstable_by_key(|node| node.id);
        let graph = Self {
            nodes,
            selected_output,
            material: None,
        };
        validate(&graph)?;
        Ok(graph)
    }

    /// Legacy diagnostic nodes, empty for a material graph. Material graphs use
    /// `material_nodes` instead; the two node DTOs are deliberately not projected.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.index(id).map(|index| &self.nodes[index])
    }

    pub const fn selected_output(&self) -> NodeId {
        self.selected_output
    }

    pub(crate) fn from_material(dto: crate::MaterialGraphDto) -> Self {
        Self {
            nodes: Vec::new(),
            selected_output: dto.selected_output.node,
            material: Some(dto),
        }
    }

    pub(crate) fn set_parameter(
        &mut self,
        node: NodeId,
        parameter: ParameterId,
        value: crate::ParameterValue,
    ) -> Result<(), CoreError> {
        if let Some(material) = &mut self.material {
            let target = material
                .graph
                .iter_mut()
                .find(|target| target.id == node)
                .ok_or(CoreError::MissingNode(node))?;
            target.params = target.params.with_parameter(parameter, value)?;
            return Ok(());
        }
        let index = self.index(node).ok_or(CoreError::MissingNode(node))?;
        let crate::ParameterValue::Float(value) = value else {
            return Err(CoreError::InvalidMaterial(
                "legacy parameters require float values",
            ));
        };
        self.nodes[index].params = self.nodes[index].params.with_parameter(parameter, value)?;
        Ok(())
    }

    pub(crate) fn parameter_value(
        &self,
        node: NodeId,
        parameter: ParameterId,
    ) -> Result<crate::ParameterValue, CoreError> {
        if let Some(material) = &self.material {
            return material
                .node(node)
                .ok_or(CoreError::MissingNode(node))?
                .params
                .parameter(parameter);
        }
        Ok(crate::ParameterValue::Float(
            self.node(node)
                .ok_or(CoreError::MissingNode(node))?
                .params
                .parameter(parameter)?,
        ))
    }

    pub(crate) fn select_map(&mut self, map: MapKind) -> Result<bool, CoreError> {
        let material = self.material.as_mut().ok_or(CoreError::InvalidMaterial(
            "named map selection requires a material graph",
        ))?;
        let output = *material
            .outputs
            .get(&map)
            .ok_or(CoreError::InvalidMaterial(
                "material does not provide the selected map",
            ))?;
        let changed = material.selected_output != output;
        material.selected_output = output;
        self.selected_output = output.node;
        Ok(changed)
    }

    pub(crate) fn select_output(&mut self, output: NodeId) -> Result<(), CoreError> {
        if let Some(material) = &mut self.material {
            let node = material
                .node(output)
                .ok_or(CoreError::MissingNode(output))?;
            let port = if material.selected_output.node == output {
                material.selected_output.port
            } else {
                node.primary_output()
            };
            material.selected_output = crate::OutputRef { node: output, port };
            self.selected_output = output;
            return Ok(());
        }
        self.node(output).ok_or(CoreError::MissingNode(output))?;
        self.selected_output = output;
        Ok(())
    }

    fn index(&self, id: NodeId) -> Option<usize> {
        self.nodes.binary_search_by_key(&id, |node| node.id).ok()
    }

    /// Returns every legacy node once, inputs before consumers, including branches.
    /// Material graphs use `topological_material_nodes` instead.
    pub fn topological_nodes(&self) -> Vec<&Node> {
        // Live graph invariants guarantee that this traversal succeeds.
        self.topological_indices()
            .expect("validated graph")
            .into_iter()
            .map(|index| &self.nodes[index])
            .collect()
    }

    fn topological_indices(&self) -> Result<Vec<usize>, CoreError> {
        let mut state = vec![0u8; self.nodes.len()];
        let mut ordered = Vec::with_capacity(self.nodes.len());
        let mut path = Vec::new();
        for start in 0..self.nodes.len() {
            if state[start] == 2 {
                continue;
            }
            let mut next = Some(start);
            while let Some(index) = next {
                if state[index] == 2 {
                    break;
                }
                if state[index] == 1 {
                    return Err(CoreError::Cycle(self.nodes[index].id));
                }
                state[index] = 1;
                path.push(index);
                next = match self.nodes[index].input {
                    Some(input) => Some(self.index(input).ok_or(CoreError::MissingNode(input))?),
                    None => None,
                };
            }
            while let Some(index) = path.pop() {
                state[index] = 2;
                ordered.push(index);
            }
        }
        Ok(ordered)
    }
}

/// Untrusted, versioned diagnostic envelope; convert with `Graph::try_from`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphDto {
    pub schema_version: u32,
    pub graph: Vec<Node>,
    pub selected_output: NodeId,
}

// Both envelopes validate before becoming a Graph; serde never drops material ports.
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum GraphEnvelope {
    Legacy(GraphDto),
    Material(crate::MaterialGraphDto),
}

impl TryFrom<GraphEnvelope> for Graph {
    type Error = CoreError;

    fn try_from(value: GraphEnvelope) -> Result<Self, Self::Error> {
        match value {
            GraphEnvelope::Legacy(dto) => Self::try_from(dto),
            GraphEnvelope::Material(dto) => Self::try_from(dto),
        }
    }
}

impl From<Graph> for GraphEnvelope {
    fn from(graph: Graph) -> Self {
        match graph.material {
            Some(dto) => Self::Material(dto),
            None => Self::Legacy(GraphDto {
                schema_version: GRAPH_SCHEMA_VERSION,
                graph: graph.nodes,
                selected_output: graph.selected_output,
            }),
        }
    }
}

impl TryFrom<GraphDto> for Graph {
    type Error = CoreError;

    fn try_from(dto: GraphDto) -> Result<Self, Self::Error> {
        if dto.schema_version != GRAPH_SCHEMA_VERSION {
            return Err(CoreError::UnknownSchema(dto.schema_version));
        }
        Self::new(dto.graph, dto.selected_output)
    }
}

impl TryFrom<Graph> for GraphDto {
    type Error = CoreError;

    /// Returns an error for material graphs, which schema 1 cannot represent.
    fn try_from(graph: Graph) -> Result<Self, Self::Error> {
        if graph.material.is_some() {
            return Err(CoreError::InvalidMaterial(
                "schema-2 material cannot be represented by GraphDto",
            ));
        }
        Ok(Self {
            schema_version: GRAPH_SCHEMA_VERSION,
            graph: graph.nodes,
            selected_output: graph.selected_output,
        })
    }
}

/// Runtime snapshot. Its graph remains immutable through shared references.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationSnapshot {
    pub generation: u64,
    pub graph: Graph,
}

pub fn validate(graph: &Graph) -> Result<(), CoreError> {
    if let Some(material) = &graph.material {
        return material.validate();
    }
    for pair in graph.nodes.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(CoreError::DuplicateNode(pair[0].id));
        }
    }
    if graph.node(graph.selected_output).is_none() {
        return Err(CoreError::MissingNode(graph.selected_output));
    }
    for node in &graph.nodes {
        node.params.validate()?;
        match (node.params.kind(), node.input) {
            (NodeKind::Source, Some(_)) => return Err(CoreError::UnexpectedInput(node.id)),
            (NodeKind::Source, None) => {}
            (_, None) => return Err(CoreError::MissingInput(node.id)),
            (_, Some(input)) => {
                let parent = graph.node(input).ok_or(CoreError::MissingNode(input))?;
                if parent.params.output_kind() != MapKind::Height {
                    return Err(CoreError::WrongInputType {
                        node: node.id,
                        actual: parent.params.output_kind(),
                    });
                }
            }
        }
    }
    graph.topological_indices()?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoreError {
    InvalidExtent(Extent),
    InvalidPixelLength {
        expected: usize,
        actual: usize,
    },
    DuplicateNode(NodeId),
    MissingNode(NodeId),
    MissingInput(NodeId),
    UnexpectedInput(NodeId),
    Cycle(NodeId),
    WrongInputType {
        node: NodeId,
        actual: MapKind,
    },
    NonFiniteParameter(ParameterId),
    ParameterOutOfRange(ParameterId),
    InvalidLevelsInterval,
    WrongParameter {
        kind: NodeKind,
        parameter: ParameterId,
    },
    UnknownSchema(u32),
    InvalidInputKeyCount {
        expected: usize,
        actual: usize,
    },
    /// Invalid material DTO, parameter range, port contract, or metadata.
    InvalidMaterial(&'static str),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent(e) => write!(
                f,
                "extent {}x{} must have each dimension in 1..={MAX_DIMENSION}",
                e.width, e.height
            ),
            Self::InvalidPixelLength { expected, actual } => {
                write!(f, "expected {expected} RGBA8 bytes, got {actual}")
            }
            Self::DuplicateNode(id) => write!(f, "duplicate node {}", id.0),
            Self::MissingNode(id) => write!(f, "node {} does not exist", id.0),
            Self::MissingInput(id) => write!(f, "node {} requires a height input", id.0),
            Self::UnexpectedInput(id) => write!(f, "source node {} cannot have an input", id.0),
            Self::Cycle(id) => write!(f, "cycle at node {}", id.0),
            Self::WrongInputType { node, actual } => {
                write!(f, "node {} requires Height, got {actual:?}", node.0)
            }
            Self::NonFiniteParameter(id) => write!(f, "{id:?} must be finite"),
            Self::ParameterOutOfRange(id) => write!(f, "{id:?} is outside its permitted range"),
            Self::InvalidLevelsInterval => f.write_str("levels require white - black >= 1e-6"),
            Self::WrongParameter { kind, parameter } => {
                write!(f, "{parameter:?} is not a parameter of {kind:?}")
            }
            Self::UnknownSchema(version) => write!(f, "unsupported graph schema {version}"),
            Self::InvalidInputKeyCount { expected, actual } => {
                write!(f, "expected {expected} input keys, got {actual}")
            }
            Self::InvalidMaterial(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for CoreError {}
