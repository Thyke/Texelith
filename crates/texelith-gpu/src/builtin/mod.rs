//! Source metadata for the built-in conversion nodes.
use texelith_core::{MaterialNode, MaterialNodeParams as P, SourceDescriptor};

pub(crate) fn source_descriptor(node: &MaterialNode) -> Option<&SourceDescriptor> {
    match &node.params {
        P::LegacySource(s) | P::ColorSource(s) | P::HeightSource(s) => Some(s),
        P::NormalSource { source, .. } => Some(source),
        _ => None,
    }
}

pub(crate) fn is_color_source(node: &MaterialNode) -> bool {
    matches!(node.params, P::ColorSource(_))
}
