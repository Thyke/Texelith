use crate::{
    CoreError, Extent, InputInterpretation, MapFormat, MapKind, Node, NodeKind, NodeParams,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub [u8; 32]);

/// Explicit algorithm configuration, independent of graph schema and generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheRevisions {
    pub algorithm: u32,
    pub color_policy: u32,
}

impl CacheRevisions {
    pub const CURRENT: Self = Self {
        algorithm: 1,
        color_policy: 1,
    };
}

impl Default for CacheRevisions {
    fn default() -> Self {
        Self::CURRENT
    }
}

pub fn node_cache_key(
    node: &Node,
    inputs: &[CacheKey],
    extent: Extent,
    source_extent: Extent,
) -> Result<CacheKey, CoreError> {
    node_cache_key_with_revisions(node, inputs, extent, source_extent, CacheRevisions::CURRENT)
}

pub fn node_cache_key_with_revisions(
    node: &Node,
    inputs: &[CacheKey],
    extent: Extent,
    source_extent: Extent,
    revisions: CacheRevisions,
) -> Result<CacheKey, CoreError> {
    let bytes = canonical_node_bytes(node, inputs, extent, source_extent, revisions)?;
    Ok(CacheKey(*blake3::hash(&bytes).as_bytes()))
}

/// Encodes cache schema 1 exactly as documented in the crate-level rustdoc.
/// Invalid node parameters, edge arity, key arity or dimensions are rejected.
pub fn canonical_node_bytes(
    node: &Node,
    inputs: &[CacheKey],
    extent: Extent,
    source_extent: Extent,
    revisions: CacheRevisions,
) -> Result<Vec<u8>, CoreError> {
    node.params.validate()?;
    extent.validate()?;
    source_extent.validate()?;
    let is_source = node.params.kind() == NodeKind::Source;
    match (is_source, node.input) {
        (true, Some(_)) => return Err(CoreError::UnexpectedInput(node.id)),
        (false, None) => return Err(CoreError::MissingInput(node.id)),
        _ => {}
    }
    let expected = usize::from(!is_source);
    if inputs.len() != expected {
        return Err(CoreError::InvalidInputKeyCount {
            expected,
            actual: inputs.len(),
        });
    }
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(b"texelith.node-cache\0");
    u32_bytes(&mut bytes, 1);
    u32_bytes(&mut bytes, revisions.algorithm);
    u32_bytes(&mut bytes, revisions.color_policy);
    bytes.push(match node.params.kind() {
        NodeKind::Source => 0,
        NodeKind::Levels => 1,
        NodeKind::Blur => 2,
        NodeKind::NormalFromHeight => 3,
    });
    bytes.push(match node.params.output_kind() {
        MapKind::Height => 0,
        MapKind::Normal => 1,
        _ => unreachable!("legacy nodes only produce height or normal"),
    });
    bytes.push(match node.params.output_kind().format() {
        MapFormat::R32Float => 0,
        MapFormat::Rgba16Float => 1,
    });
    match &node.params {
        NodeParams::Source(source) => {
            bytes.extend_from_slice(&source.content_hash);
            extent_bytes(&mut bytes, source.extent);
            bytes.push(match source.interpretation {
                InputInterpretation::SrgbPhoto => 0,
                InputInterpretation::LinearHeight => 1,
                InputInterpretation::LinearNormal => unreachable!("legacy source validated above"),
            });
        }
        NodeParams::Levels {
            black,
            white,
            gamma,
        } => {
            for value in [*black, *white, *gamma] {
                float_bytes(&mut bytes, value);
            }
        }
        NodeParams::Blur { sigma_source_px } => float_bytes(&mut bytes, *sigma_source_px),
        NodeParams::NormalFromHeight { strength } => float_bytes(&mut bytes, *strength),
    }
    u32_bytes(&mut bytes, inputs.len() as u32);
    for input in inputs {
        bytes.extend_from_slice(&input.0);
    }
    extent_bytes(&mut bytes, extent);
    extent_bytes(&mut bytes, source_extent);
    Ok(bytes)
}

pub(crate) fn u32_bytes(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
pub(crate) fn extent_bytes(bytes: &mut Vec<u8>, extent: Extent) {
    u32_bytes(bytes, extent.width);
    u32_bytes(bytes, extent.height);
}
pub(crate) fn float_bytes(bytes: &mut Vec<u8>, value: f32) {
    u32_bytes(bytes, if value == 0.0 { 0 } else { value.to_bits() });
}
