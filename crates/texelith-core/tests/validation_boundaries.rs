use texelith_core::{
    CoreError, Extent, Graph, GraphDto, InputInterpretation, Node, NodeId, NodeKind, NodeParams,
    PARAMETER_DESCRIPTORS, ParameterId, SourceDescriptor,
};

fn source(id: u64) -> Node {
    Node {
        id: NodeId(id),
        input: None,
        params: NodeParams::Source(SourceDescriptor {
            content_hash: [0; 32],
            extent: Extent {
                width: 1,
                height: 4096,
            },
            interpretation: InputInterpretation::LinearHeight,
        }),
    }
}

#[test]
fn diagnostic_dto_cannot_bypass_graph_validation() {
    let nodes = vec![
        source(1),
        Node {
            id: NodeId(2),
            input: Some(NodeId(1)),
            params: NodeParams::Blur {
                sigma_source_px: 0.0,
            },
        },
    ];
    let graph = Graph::new(nodes.clone(), NodeId(2)).unwrap();
    let dto = GraphDto::try_from(graph.clone()).unwrap();
    assert_eq!(Graph::try_from(dto).unwrap(), graph);
    let json = serde_json::to_value(&graph).unwrap();
    assert_eq!(json.as_object().unwrap().len(), 3);
    for variant in 0..7 {
        let mut bad = json.clone();
        match variant {
            0 => bad["graph"][1]["id"] = 1.into(),
            1 => bad["graph"][1]["input"] = 99.into(),
            2 => bad["graph"][1]["input"] = 2.into(),
            3 => bad["graph"][0]["params"]["Source"]["extent"]["width"] = 0.into(),
            4 => bad["graph"][1]["params"]["Blur"]["sigma_source_px"] = (-1).into(),
            5 => bad["selected_output"] = 99.into(),
            _ => bad["schema_version"] = 999.into(),
        }
        assert!(
            serde_json::from_value::<Graph>(bad).is_err(),
            "variant {variant}"
        );
    }
    assert!(Graph::new(vec![], NodeId(1)).is_err());
    assert!(matches!(
        Graph::try_from(GraphDto {
            schema_version: 999,
            graph: nodes,
            selected_output: NodeId(2)
        }),
        Err(CoreError::UnknownSchema(999))
    ));
}

#[test]
fn sorted_storage_does_not_assume_id_execution_order_and_checks_disconnected_nodes() {
    let nodes = vec![
        Node {
            id: NodeId(1),
            input: Some(NodeId(100)),
            params: NodeParams::Blur {
                sigma_source_px: 0.0,
            },
        },
        source(100),
        Node {
            id: NodeId(2),
            input: Some(NodeId(100)),
            params: NodeParams::NormalFromHeight { strength: 1.0 },
        },
    ];
    let graph = Graph::new(nodes.clone(), NodeId(1)).unwrap();
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(|node| node.id.0)
            .collect::<Vec<_>>(),
        [1, 2, 100]
    );
    assert_eq!(
        graph
            .topological_nodes()
            .iter()
            .map(|node| node.id.0)
            .collect::<Vec<_>>(),
        [100, 1, 2]
    );
    assert_eq!(graph.selected_output(), NodeId(1));
    assert!(graph.node(NodeId(999)).is_none());
    let mut reversed = nodes.clone();
    reversed.reverse();
    assert_eq!(graph, Graph::new(reversed, NodeId(1)).unwrap());
    let mut bad = nodes;
    bad.push(Node {
        id: NodeId(3),
        input: Some(NodeId(3)),
        params: NodeParams::Blur {
            sigma_source_px: 0.0,
        },
    });
    assert!(matches!(
        Graph::new(bad, NodeId(1)),
        Err(CoreError::Cycle(NodeId(3)))
    ));
}

#[test]
fn long_graph_validation_and_traversal_are_iterative() {
    let mut nodes = vec![source(20_000)];
    for id in (1..20_000).rev() {
        nodes.push(Node {
            id: NodeId(id),
            input: Some(NodeId(id + 1)),
            params: NodeParams::Blur {
                sigma_source_px: 0.0,
            },
        });
    }
    let graph = Graph::new(nodes.clone(), NodeId(1)).unwrap();
    assert_eq!(graph.topological_nodes().len(), 20_000);
    nodes[0] = Node {
        id: NodeId(20_000),
        input: Some(NodeId(1)),
        params: NodeParams::Blur {
            sigma_source_px: 0.0,
        },
    };
    assert!(matches!(
        Graph::new(nodes, NodeId(1)),
        Err(CoreError::Cycle(_))
    ));
}

#[test]
fn parameter_ranges_defaults_and_cross_field_rules_are_authoritative() {
    let NodeParams::Source(desc) = source(1).params else {
        unreachable!()
    };
    for descriptor in PARAMETER_DESCRIPTORS {
        let defaults = NodeParams::default_for(descriptor.node_kind, desc.clone());
        assert_eq!(
            defaults.parameter(descriptor.id).unwrap(),
            descriptor.default
        );
        assert!(defaults.validate().is_ok());
        for value in [descriptor.min, descriptor.max] {
            assert!(descriptor.id.validate(value).is_ok());
        }
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            descriptor.min - 0.01,
            descriptor.max + 0.01,
        ] {
            assert!(descriptor.id.validate(value).is_err());
        }
    }
    let levels = NodeParams::default_for(NodeKind::Levels, desc);
    assert!(levels.with_parameter(ParameterId::Black, 1.0).is_err());
    assert!(levels.with_parameter(ParameterId::Strength, 1.0).is_err());
    assert_eq!(levels.parameter(ParameterId::Black).unwrap(), 0.0);
    assert_eq!(
        levels
            .with_parameter(ParameterId::Gamma, 2.0)
            .unwrap()
            .parameter(ParameterId::Gamma)
            .unwrap(),
        2.0
    );
    assert!(
        NodeParams::Levels {
            black: 0.0,
            white: 1e-6,
            gamma: 0.1
        }
        .validate()
        .is_ok()
    );
    assert!(
        NodeParams::Levels {
            black: 0.0,
            white: 0.999e-6,
            gamma: 0.1
        }
        .validate()
        .is_err()
    );
}

#[test]
fn decoded_rgba_source_hash_boundary_checks_shape_and_ignores_file_encoding() {
    let e = Extent::new(1, 1).unwrap();
    let pixels = [1, 2, 3, 255];
    let source = SourceDescriptor::from_rgba8(&pixels, e, InputInterpretation::SrgbPhoto).unwrap();
    assert_eq!(source.content_hash, *blake3::hash(&pixels).as_bytes());
    assert_eq!(source.extent, e);
    let mut changed_pixels = pixels;
    changed_pixels[0] = 2;
    assert_ne!(
        source.content_hash,
        SourceDescriptor::from_rgba8(&changed_pixels, e, InputInterpretation::SrgbPhoto)
            .unwrap()
            .content_hash
    );
    assert_eq!(
        source.content_hash,
        SourceDescriptor::from_rgba8(&pixels, e, InputInterpretation::LinearHeight)
            .unwrap()
            .content_hash
    );
    for pixels in [&[][..], &[1, 2, 3][..], &[0; 5][..]] {
        assert!(matches!(
            SourceDescriptor::from_rgba8(pixels, e, InputInterpretation::SrgbPhoto),
            Err(CoreError::InvalidPixelLength { .. })
        ));
    }
    assert!(
        SourceDescriptor::from_rgba8(
            &[],
            Extent {
                width: 0,
                height: 1
            },
            InputInterpretation::SrgbPhoto
        )
        .is_err()
    );
}
