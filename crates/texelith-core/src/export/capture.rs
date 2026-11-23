use super::{export_extent, validate_export_for_graph};
use crate::{
    CacheKey, CacheRevisions, CoreError, EvaluationSnapshot, ExportPlan, Graph, MaterialNodeParams,
    NodeParams, OutputPort, OutputRef, SourceDescriptor, material_node_cache_key, node_cache_key,
};
use serde::{Deserialize, Serialize};

/// Owned recipe for an accepted job, independent of subsequent Session mutations.
/// Graph parameters retain all seeds and source normal conventions. Consumers
/// must retain decoded sources by these content hashes until the job retires.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportCapture {
    pub graph: Graph,
    pub graph_hash: [u8; 32],
    pub sources: Vec<SourceDescriptor>,
    pub generation: u64,
    pub document_revision: u64,
}

impl ExportCapture {
    pub fn new(
        snapshot: &EvaluationSnapshot,
        document_revision: u64,
        plan: &ExportPlan,
    ) -> Result<Self, CoreError> {
        validate_export_for_graph(plan, &snapshot.graph)?;
        let graph = &snapshot.graph;
        let mut sources = Vec::new();
        let mut keys: Vec<(OutputRef, CacheKey)> = Vec::new();
        let mut hash = blake3::Hasher::new();
        hash.update(b"texelith.export-graph\0\x01");
        if graph.is_material() {
            for node in graph.topological_material_nodes() {
                match &node.params {
                    MaterialNodeParams::LegacySource(source)
                    | MaterialNodeParams::ColorSource(source)
                    | MaterialNodeParams::HeightSource(source)
                    | MaterialNodeParams::NormalSource { source, .. } => {
                        sources.push(source.clone())
                    }
                    _ => {}
                }
                let inputs: Vec<_> = node
                    .inputs
                    .iter()
                    .map(|edge| {
                        keys.iter()
                            .find(|(reference, _)| *reference == edge.source)
                            .map(|(_, key)| *key)
                            .expect("validated topological edge")
                    })
                    .collect();
                for port in [
                    OutputPort::Albedo,
                    OutputPort::Height,
                    OutputPort::Normal,
                    OutputPort::Ao,
                    OutputPort::Curvature,
                    OutputPort::Edge,
                    OutputPort::Cavity,
                    OutputPort::Roughness,
                    OutputPort::Metallic,
                    OutputPort::Mask,
                ] {
                    if !node.has_output(port) {
                        continue;
                    }
                    let reference = OutputRef {
                        node: node.id,
                        port,
                    };
                    let extent = export_extent(graph, reference)?;
                    let key = material_node_cache_key(
                        node,
                        port,
                        &inputs,
                        extent,
                        extent,
                        CacheRevisions::CURRENT,
                    )?;
                    hash.update(&node.id.0.to_le_bytes());
                    hash.update(&[port as u8]);
                    hash.update(&key.0);
                    keys.push((reference, key));
                }
            }
        } else {
            for node in graph.topological_nodes() {
                if let NodeParams::Source(source) = &node.params {
                    sources.push(source.clone());
                }
                let inputs: Vec<_> = node
                    .input
                    .map(|id| {
                        keys.iter()
                            .find(|(reference, _)| reference.node == id)
                            .map(|(_, key)| *key)
                            .expect("validated topological edge")
                    })
                    .into_iter()
                    .collect();
                let port = if node.params.output_kind() == crate::MapKind::Normal {
                    OutputPort::Normal
                } else {
                    OutputPort::Height
                };
                let reference = OutputRef {
                    node: node.id,
                    port,
                };
                let extent = export_extent(graph, reference)?;
                let key = node_cache_key(node, &inputs, extent, extent)?;
                hash.update(&node.id.0.to_le_bytes());
                hash.update(&key.0);
                keys.push((reference, key));
            }
        }
        Ok(Self {
            graph: graph.clone(),
            graph_hash: *hash.finalize().as_bytes(),
            sources,
            generation: snapshot.generation,
            document_revision,
        })
    }
}
