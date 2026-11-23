#![allow(dead_code)]
//! Tiny owned fixtures only; scheduling, hashing and accounting are production code.
use std::sync::Arc;
use texelith_core::{
    EvaluationSnapshot, Extent, Graph, InputInterpretation, Node, NodeId, NodeParams,
    SourceDescriptor,
};

pub const SOURCE: Extent = Extent {
    width: 64,
    height: 32,
};
pub const PROXY: Extent = Extent {
    width: 16,
    height: 8,
};

pub fn snapshot(generation: u64, source_hash: u8, strength: f32) -> Arc<EvaluationSnapshot> {
    let params = [
        NodeParams::Source(SourceDescriptor {
            content_hash: [source_hash; 32],
            extent: SOURCE,
            interpretation: InputInterpretation::LinearHeight,
        }),
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Blur {
            sigma_source_px: 1.25,
        },
        NodeParams::NormalFromHeight { strength },
    ];
    let nodes = params
        .into_iter()
        .enumerate()
        .map(|(index, params)| Node {
            id: NodeId(index as u64 + 1),
            input: (index != 0).then_some(NodeId(index as u64)),
            params,
        })
        .collect();
    Arc::new(EvaluationSnapshot {
        generation,
        graph: Graph::new(nodes, NodeId(4)).unwrap(),
    })
}
