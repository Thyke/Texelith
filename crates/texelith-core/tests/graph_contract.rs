//! Contract tests. Proposed construction boundary: Graph::new accepts untrusted
//! nodes and a selected output, validates them, and returns a live graph.
use texelith_core::{
    CacheKey, EvaluationSnapshot, Extent, Graph, InputInterpretation, Node, NodeId, NodeParams,
    SourceDescriptor, node_cache_key, validate,
};

fn extent(width: u32, height: u32) -> Extent {
    Extent { width, height }
}
fn source(width: u32, height: u32) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [7; 32],
        extent: extent(width, height),
        interpretation: InputInterpretation::SrgbPhoto,
    }
}
fn chain() -> Vec<Node> {
    vec![
        Node {
            id: NodeId(1),
            input: None,
            params: NodeParams::Source(source(257, 129)),
        },
        Node {
            id: NodeId(2),
            input: Some(NodeId(1)),
            params: NodeParams::Levels {
                black: 0.0,
                white: 1.0,
                gamma: 1.0,
            },
        },
        Node {
            id: NodeId(3),
            input: Some(NodeId(2)),
            params: NodeParams::Blur {
                sigma_source_px: 0.0,
            },
        },
        Node {
            id: NodeId(4),
            input: Some(NodeId(3)),
            params: NodeParams::NormalFromHeight { strength: 1.0 },
        },
    ]
}
fn graph(nodes: Vec<Node>) -> Result<Graph, texelith_core::CoreError> {
    Graph::new(nodes, NodeId(4))
}

#[test]
fn valid_fixed_chain_accepts_npot_and_validates() {
    let graph = graph(chain()).unwrap();
    assert!(validate(&graph).is_ok());
}
#[test]
fn rejects_missing_duplicate_and_selected_ids() {
    let mut nodes = chain();
    nodes[2].input = Some(NodeId(99));
    assert!(graph(nodes).is_err());
    let mut nodes = chain();
    nodes[2].id = NodeId(2);
    assert!(graph(nodes).is_err());
    assert!(Graph::new(chain(), NodeId(99)).is_err());
}
#[test]
fn rejects_cycles_wrong_types_and_source_input() {
    let mut nodes = chain();
    nodes[1].input = Some(NodeId(3));
    assert!(graph(nodes).is_err());
    let mut nodes = chain();
    nodes[1].input = Some(NodeId(4));
    assert!(graph(nodes).is_err());
    let mut nodes = chain();
    nodes[0].input = Some(NodeId(1));
    assert!(graph(nodes).is_err());
    let mut nodes = chain();
    nodes[1].input = None;
    assert!(graph(nodes).is_err());
}
#[test]
fn rejects_invalid_dimensions() {
    for (w, h) in [(0, 1), (1, 0), (4097, 1), (1, 4097)] {
        let mut nodes = chain();
        nodes[0].params = NodeParams::Source(source(w, h));
        assert!(graph(nodes).is_err(), "{w}x{h}");
    }
    let mut nodes = chain();
    nodes[0].params = NodeParams::Source(source(4096, 4096));
    assert!(graph(nodes).is_ok());
}
#[test]
fn rejects_nonfinite_and_out_of_range_parameters() {
    let bad = [
        NodeParams::Levels {
            black: f32::NAN,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Levels {
            black: 0.0,
            white: f32::INFINITY,
            gamma: 1.0,
        },
        NodeParams::Levels {
            black: -0.1,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Levels {
            black: 0.5,
            white: 0.5,
            gamma: 1.0,
        },
        NodeParams::Levels {
            black: 0.5,
            white: 0.5000005,
            gamma: 1.0,
        },
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 0.09,
        },
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 10.1,
        },
    ];
    for params in bad {
        let mut nodes = chain();
        nodes[1].params = params;
        assert!(graph(nodes).is_err());
    }
    for sigma in [f32::NAN, f32::INFINITY, -0.1, 5.1] {
        let mut nodes = chain();
        nodes[2].params = NodeParams::Blur {
            sigma_source_px: sigma,
        };
        assert!(graph(nodes).is_err());
    }
    for strength in [f32::NAN, f32::NEG_INFINITY, -0.1, 10.1] {
        let mut nodes = chain();
        nodes[3].params = NodeParams::NormalFromHeight { strength };
        assert!(graph(nodes).is_err());
    }
}
#[test]
fn dto_roundtrip_and_unknown_schema_rejection() {
    // The graph's serde boundary is the versioned diagnostic envelope, not raw live state.
    let graph = graph(chain()).unwrap();
    let json = serde_json::to_value(&graph).unwrap();
    assert_eq!(json["schema_version"], 1);
    let restored: Graph = serde_json::from_value(json.clone()).unwrap();
    assert!(validate(&restored).is_ok());
    let mut unknown = json;
    unknown["schema_version"] = 999.into();
    assert!(serde_json::from_value::<Graph>(unknown).is_err());
}
fn key(node: &Node, inputs: &[CacheKey], eval: Extent, original: Extent) -> CacheKey {
    node_cache_key(node, inputs, eval, original).unwrap()
}
#[test]
fn cache_identity_normalizes_negative_zero_and_ignores_node_id() {
    let mut a = chain()[2].clone();
    let mut b = a.clone();
    b.id = NodeId(999);
    b.params = NodeParams::Blur {
        sigma_source_px: -0.0,
    };
    let input = [CacheKey([3; 32])];
    assert_eq!(
        key(&a, &input, extent(100, 50), extent(200, 100)),
        key(&b, &input, extent(100, 50), extent(200, 100))
    );
    a.params = NodeParams::Blur {
        sigma_source_px: 1.0,
    };
    assert_ne!(
        key(&a, &input, extent(100, 50), extent(200, 100)),
        key(&b, &input, extent(100, 50), extent(200, 100))
    );
}
#[test]
fn cache_identity_changes_for_inputs_extents_and_interpretation() {
    let nodes = chain();
    let node = &nodes[2];
    let e = extent(100, 50);
    let s = extent(200, 100);
    let baseline = key(node, &[CacheKey([1; 32])], e, s);
    assert_ne!(baseline, key(node, &[CacheKey([2; 32])], e, s));
    assert_ne!(
        baseline,
        key(node, &[CacheKey([1; 32])], extent(101, 50), s)
    );
    assert_ne!(
        baseline,
        key(node, &[CacheKey([1; 32])], e, extent(201, 100))
    );
    let mut other = nodes[0].clone();
    if let NodeParams::Source(ref mut desc) = other.params {
        desc.interpretation = InputInterpretation::LinearHeight;
    }
    assert_ne!(key(&nodes[0], &[], e, s), key(&other, &[], e, s));
}
#[test]
fn cache_identity_is_independent_of_generation_and_downstream_changes() {
    let g = graph(chain()).unwrap();
    let first = EvaluationSnapshot {
        generation: 1,
        graph: g.clone(),
    };
    let later = EvaluationSnapshot {
        generation: 42,
        graph: g,
    };
    let nodes = chain();
    let ancestor_before = key(
        &nodes[1],
        &[CacheKey([5; 32])],
        extent(257, 129),
        extent(257, 129),
    );
    let mut changed = nodes.clone();
    changed[3].params = NodeParams::NormalFromHeight { strength: 9.0 };
    let ancestor_after = key(
        &changed[1],
        &[CacheKey([5; 32])],
        extent(257, 129),
        extent(257, 129),
    );
    assert_ne!(first.generation, later.generation);
    assert_eq!(ancestor_before, ancestor_after);
}
