use std::collections::BTreeMap;
use texelith_core::{
    CacheKey, CacheRevisions, CoreError, EvaluationSnapshot, Extent, Graph, InputInterpretation,
    Node, NodeId, NodeParams, SourceDescriptor, canonical_node_bytes, node_cache_key,
    node_cache_key_with_revisions,
};

const EVAL: Extent = Extent {
    width: 100,
    height: 50,
};
const ORIGINAL: Extent = Extent {
    width: 200,
    height: 100,
};

fn source() -> NodeParams {
    NodeParams::Source(SourceDescriptor {
        content_hash: [7; 32],
        extent: Extent {
            width: 257,
            height: 129,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    })
}

fn node(params: NodeParams) -> Node {
    let input = if matches!(params, NodeParams::Source(_)) {
        None
    } else {
        Some(NodeId(1))
    };
    Node {
        id: NodeId(2),
        input,
        params,
    }
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).unwrap())
        .collect()
}

#[test]
fn cache_identity_golden_bytes_and_blake3_hashes() {
    // Owned fixtures independently packed with Python struct '<I'/'<f' from
    // the documented layout. Hashes were calculated directly from these literal
    // bytes with BLAKE3, without calling the production encoder or key function.
    let fixtures = [
        (
            source(),
            "746578656c6974682e6e6f64652d6361636865000100000001000000010000000000000707070707070707070707070707070707070707070707070707070707070707010100008100000000000000006400000032000000c800000064000000",
            "0b62ff772c40695bfc57ca3ac2b2d3d7cb17545ea840632d95bf8392c7234f9e",
        ),
        (
            NodeParams::Levels {
                black: 0.25,
                white: 0.75,
                gamma: 2.0,
            },
            "746578656c6974682e6e6f64652d6361636865000100000001000000010000000100000000803e0000403f000000400100000003030303030303030303030303030303030303030303030303030303030303036400000032000000c800000064000000",
            "9d8614d1a8d8e0ae1b9ba7cfa925489052d8ef9efee076fef26be66971f03d3d",
        ),
        (
            NodeParams::Blur {
                sigma_source_px: 1.25,
            },
            "746578656c6974682e6e6f64652d6361636865000100000001000000010000000200000000a03f0100000003030303030303030303030303030303030303030303030303030303030303036400000032000000c800000064000000",
            "e3ce3f354a462658e9976297f633a70abb2102d5a74221b554941bef32412832",
        ),
        (
            NodeParams::NormalFromHeight { strength: 2.5 },
            "746578656c6974682e6e6f64652d636163686500010000000100000001000000030101000020400100000003030303030303030303030303030303030303030303030303030303030303036400000032000000c800000064000000",
            "18f5953a2485462da7d482a140891913cd015d908f2e1af3f7e7418e24f0ce61",
        ),
    ];
    for (params, bytes, hash) in fixtures {
        let node = node(params);
        let inputs = if node.input.is_none() {
            vec![]
        } else {
            vec![CacheKey([3; 32])]
        };
        assert_eq!(
            canonical_node_bytes(&node, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT).unwrap(),
            hex_bytes(bytes)
        );
        assert_eq!(
            node_cache_key(&node, &inputs, EVAL, ORIGINAL)
                .unwrap()
                .0
                .as_slice(),
            hex_bytes(hash)
        );
    }
}

#[test]
fn cache_identity_field_order_is_independent_of_json_and_rust_construction() {
    // Parse text directly, so serde_json's ordered Value map cannot pre-sort it.
    let first: Node = serde_json::from_str(
        r#"{"id":2,"input":1,"params":{"Levels":{"black":0.25,"white":0.75,"gamma":2.0}}}"#,
    )
    .unwrap();
    let second: Node = serde_json::from_str(
        r#"{"params":{"Levels":{"gamma":2.0,"white":0.75,"black":0.25}},"input":1,"id":2}"#,
    )
    .unwrap();
    let third = Node {
        params: NodeParams::Levels {
            gamma: 2.0,
            white: 0.75,
            black: 0.25,
        },
        input: Some(NodeId(1)),
        id: NodeId(2),
    };
    let inputs = [CacheKey([3; 32])];
    let bytes =
        canonical_node_bytes(&first, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT).unwrap();
    let key = node_cache_key(&first, &inputs, EVAL, ORIGINAL).unwrap();
    for other in [second, third] {
        assert_eq!(
            bytes,
            canonical_node_bytes(&other, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT).unwrap()
        );
        assert_eq!(
            key,
            node_cache_key(&other, &inputs, EVAL, ORIGINAL).unwrap()
        );
    }
}

#[test]
fn cache_identity_revisions_change_bytes_and_hashes() {
    for params in [
        source(),
        NodeParams::Blur {
            sigma_source_px: 1.0,
        },
    ] {
        let node = node(params);
        let inputs = if node.input.is_none() {
            vec![]
        } else {
            vec![CacheKey([3; 32])]
        };
        let current =
            canonical_node_bytes(&node, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT).unwrap();
        let key = node_cache_key(&node, &inputs, EVAL, ORIGINAL).unwrap();
        assert_eq!(
            key,
            node_cache_key_with_revisions(&node, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT)
                .unwrap()
        );
        for revision in [
            CacheRevisions {
                algorithm: 2,
                color_policy: 1,
            },
            CacheRevisions {
                algorithm: 1,
                color_policy: 2,
            },
        ] {
            assert_ne!(
                current,
                canonical_node_bytes(&node, &inputs, EVAL, ORIGINAL, revision).unwrap()
            );
            assert_ne!(
                key,
                node_cache_key_with_revisions(&node, &inputs, EVAL, ORIGINAL, revision).unwrap()
            );
        }
    }
}

fn graph(strength: f32, id_offset: u64) -> Graph {
    let params = [
        source(),
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Blur {
            sigma_source_px: 0.0,
        },
        NodeParams::NormalFromHeight { strength },
    ];
    let nodes = params
        .into_iter()
        .enumerate()
        .map(|(index, params)| Node {
            id: NodeId(index as u64 + 1 + id_offset),
            input: (index != 0).then_some(NodeId(index as u64 + id_offset)),
            params,
        })
        .collect();
    Graph::new(nodes, NodeId(4 + id_offset)).unwrap()
}

fn snapshot_keys(snapshot: &EvaluationSnapshot) -> Vec<CacheKey> {
    let mut keys = BTreeMap::new();
    let mut ordered = Vec::new();
    for node in snapshot.graph.topological_nodes() {
        let inputs: Vec<_> = node.input.map(|id| keys[&id]).into_iter().collect();
        let key = node_cache_key(node, &inputs, EVAL, ORIGINAL).unwrap();
        keys.insert(node.id, key);
        ordered.push(key);
    }
    ordered
}

#[test]
fn cache_identity_real_snapshots_ignore_generation_ids_and_downstream_edits() {
    let before = EvaluationSnapshot {
        generation: 1,
        graph: graph(1.0, 0),
    };
    let later = EvaluationSnapshot {
        generation: u64::MAX,
        graph: before.graph.clone(),
    };
    let renumbered = EvaluationSnapshot {
        generation: 50,
        graph: graph(1.0, 100),
    };
    let edited = EvaluationSnapshot {
        generation: 2,
        graph: graph(9.0, 0),
    };
    let keys = snapshot_keys(&before);
    assert_eq!(keys, snapshot_keys(&later));
    assert_eq!(keys, snapshot_keys(&renumbered));
    let changed_keys = snapshot_keys(&edited);
    assert_eq!(&keys[..3], &changed_keys[..3]);
    assert_ne!(keys[3], changed_keys[3]);
    let restored = EvaluationSnapshot {
        generation: 3,
        graph: graph(1.0, 0),
    };
    assert_eq!(keys, snapshot_keys(&restored));
}

#[test]
fn cache_identity_normalizes_all_zero_parameters_and_excludes_edge_ids() {
    for params in [
        NodeParams::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        },
        NodeParams::Blur {
            sigma_source_px: 0.0,
        },
        NodeParams::NormalFromHeight { strength: 0.0 },
    ] {
        let first = node(params);
        let mut second = first.clone();
        second.id = NodeId(999);
        second.input = Some(NodeId(888));
        match &mut second.params {
            NodeParams::Levels { black, .. } => *black = -0.0,
            NodeParams::Blur { sigma_source_px } => *sigma_source_px = -0.0,
            NodeParams::NormalFromHeight { strength } => *strength = -0.0,
            _ => unreachable!(),
        }
        let inputs = [CacheKey([3; 32])];
        assert_eq!(
            canonical_node_bytes(&first, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT).unwrap(),
            canonical_node_bytes(&second, &inputs, EVAL, ORIGINAL, CacheRevisions::CURRENT)
                .unwrap()
        );
        assert_eq!(
            node_cache_key(&first, &inputs, EVAL, ORIGINAL).unwrap(),
            node_cache_key(&second, &inputs, EVAL, ORIGINAL).unwrap()
        );
    }
}

#[test]
fn cache_identity_includes_source_hash_dimensions_and_both_extent_axes() {
    let first = node(source());
    let baseline = node_cache_key(&first, &[], EVAL, ORIGINAL).unwrap();
    for variant in 0..4 {
        let mut changed = first.clone();
        if let NodeParams::Source(ref mut source) = changed.params {
            match variant {
                0 => source.content_hash[0] ^= 1,
                1 => source.extent.width += 1,
                2 => source.extent.height += 1,
                _ => source.interpretation = InputInterpretation::LinearHeight,
            }
        }
        assert_ne!(
            baseline,
            node_cache_key(&changed, &[], EVAL, ORIGINAL).unwrap()
        );
    }
    for (eval, original) in [
        (Extent { width: 101, ..EVAL }, ORIGINAL),
        (Extent { height: 51, ..EVAL }, ORIGINAL),
        (
            EVAL,
            Extent {
                width: 201,
                ..ORIGINAL
            },
        ),
        (
            EVAL,
            Extent {
                height: 101,
                ..ORIGINAL
            },
        ),
    ] {
        assert_ne!(
            baseline,
            node_cache_key(&first, &[], eval, original).unwrap()
        );
    }
}

#[test]
fn cache_identity_rejects_invalid_parameters_extents_and_arity() {
    let blur = node(NodeParams::Blur {
        sigma_source_px: 1.0,
    });
    let inputs = [CacheKey([3; 32])];
    assert!(matches!(
        node_cache_key(&blur, &[], EVAL, ORIGINAL),
        Err(CoreError::InvalidInputKeyCount {
            expected: 1,
            actual: 0
        })
    ));
    assert!(node_cache_key(&blur, &[inputs[0], inputs[0]], EVAL, ORIGINAL).is_err());
    assert!(node_cache_key(&node(source()), &inputs, EVAL, ORIGINAL).is_err());
    assert!(
        node_cache_key(
            &node(NodeParams::Blur {
                sigma_source_px: f32::NAN
            }),
            &inputs,
            EVAL,
            ORIGINAL
        )
        .is_err()
    );
    for invalid in [
        Extent {
            width: 0,
            height: 1,
        },
        Extent {
            width: 1,
            height: 4097,
        },
    ] {
        assert!(node_cache_key(&blur, &inputs, invalid, ORIGINAL).is_err());
        assert!(node_cache_key(&blur, &inputs, EVAL, invalid).is_err());
    }
    let mut missing = blur.clone();
    missing.input = None;
    assert!(node_cache_key(&missing, &inputs, EVAL, ORIGINAL).is_err());
    let mut extra = node(source());
    extra.input = Some(NodeId(1));
    assert!(node_cache_key(&extra, &[], EVAL, ORIGINAL).is_err());
}
