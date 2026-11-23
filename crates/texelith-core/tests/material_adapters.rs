//! Adapter coverage beyond the sequential test-author contract.
use serde_json::json;
use texelith_core::{
    CacheKey, CacheRevisions, Extent, Graph, GraphDto, InputEdge, InputInterpretation, InputPort,
    MapKind, MaskInput, MaterialGraphDto, MaterialNode, MaterialNodeParams, NodeId,
    NormalConvention, OutputPort, OutputRef, SeamlessBase, SeamlessMethod, SourceDescriptor,
    canonical_material_node_bytes, material_from_source,
};

fn source(interpretation: InputInterpretation) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [7; 32],
        extent: Extent::new(17, 13).unwrap(),
        interpretation,
    }
}

#[test]
fn graph_serde_preserves_both_schemas_and_revalidates_materials() {
    let legacy: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    let legacy = Graph::try_from(legacy).unwrap();
    let value = serde_json::to_value(&legacy).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(serde_json::from_value::<Graph>(value).unwrap(), legacy);
    for graph in [
        material_from_source(source(InputInterpretation::SrgbPhoto)).unwrap(),
        Graph::try_from(MaterialGraphDto::from(legacy)).unwrap(),
    ] {
        assert!(graph.legacy_dto().is_err());
        let value = serde_json::to_value(&graph).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(
            serde_json::from_value::<Graph>(value.clone()).unwrap(),
            graph
        );
        let mut invalid = value;
        invalid["selected_output"]["node"] = json!(999);
        assert!(serde_json::from_value::<Graph>(invalid).is_err());
    }
}

#[test]
fn factory_branches_expose_topology_extents_and_normal_metadata() {
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
        InputInterpretation::LinearNormal,
    ] {
        let graph = material_from_source(source(interpretation)).unwrap();
        let nodes = graph.topological_material_nodes();
        assert_eq!(nodes.len(), graph.material_nodes().len());
        let mut visited = Vec::new();
        for node in nodes {
            for edge in &node.inputs {
                assert!(visited.contains(&edge.source.node));
                assert_eq!(
                    graph.material_output_extent(edge.source).unwrap(),
                    source(interpretation).extent
                );
                if edge.port == InputPort::Normal {
                    assert_eq!(
                        graph.output_normal_convention(edge.source).unwrap(),
                        Some(NormalConvention::OpenGl)
                    );
                }
            }
            visited.push(node.id);
        }
        let normal = graph.material_outputs().unwrap()[&MapKind::Normal];
        assert_eq!(graph.selected_output_ref(), normal);
        assert_eq!(
            graph.output_normal_convention(normal).unwrap(),
            Some(NormalConvention::OpenGl)
        );
        if interpretation == InputInterpretation::LinearNormal {
            assert!(
                !graph
                    .material_nodes()
                    .iter()
                    .any(|node| matches!(node.params, MaterialNodeParams::NormalFromHeight { .. }))
            );
        }
    }
}

#[test]
fn directx_metadata_is_resolved_from_the_referenced_producer() {
    let graph = material_from_source(source(InputInterpretation::LinearNormal)).unwrap();
    let mut dto = MaterialGraphDto::from(graph);
    if let MaterialNodeParams::NormalSource { convention, .. } = &mut dto.graph[0].params {
        *convention = NormalConvention::DirectX;
    } else {
        panic!("normal factory must preserve the imported source");
    }
    let normal = dto.outputs[&MapKind::Normal];
    let height = dto.outputs[&MapKind::Height];
    let graph = Graph::try_from(dto).unwrap();
    assert_eq!(
        graph.output_normal_convention(normal).unwrap(),
        Some(NormalConvention::DirectX)
    );
    assert_eq!(graph.output_normal_convention(height).unwrap(), None);
    assert!(
        graph
            .output_normal_convention(OutputRef {
                node: height.node,
                port: OutputPort::Normal
            })
            .is_err()
    );
}

#[test]
fn optional_mask_default_is_explicit_and_mask_edges_remain_ordered() {
    let graph = material_from_source(source(InputInterpretation::SrgbPhoto)).unwrap();
    let mut dto = MaterialGraphDto::from(graph);
    let roughness = dto
        .graph
        .iter_mut()
        .find(|node| matches!(node.params, MaterialNodeParams::Roughness { .. }))
        .unwrap();
    assert_eq!(
        roughness.mask_input().unwrap(),
        Some(MaskInput::ConstantOne)
    );
    let mask = OutputRef {
        node: NodeId(99),
        port: OutputPort::Mask,
    };
    roughness.inputs.push(InputEdge {
        port: InputPort::Mask,
        source: mask,
    });
    assert_eq!(
        roughness.mask_input().unwrap(),
        Some(MaskInput::Output(mask))
    );
    let roughness_id = roughness.id;
    dto.graph.push(MaterialNode {
        id: mask.node,
        inputs: vec![],
        params: MaterialNodeParams::PaintMask {
            boundary: texelith_core::Boundary::Repeat,
            extent: source(InputInterpretation::SrgbPhoto).extent,
            stroke_digest: [0; 32],
            strokes: vec![],
        },
    });
    let mut reordered = MaterialGraphDto::from(Graph::try_from(dto).unwrap());
    let node = reordered
        .graph
        .iter_mut()
        .find(|node| node.id == roughness_id)
        .unwrap();
    node.inputs.reverse();
    assert!(node.mask_input().is_err());
    assert!(Graph::try_from(reordered).is_err());
}

#[test]
fn histogram_base_parameters_are_validated_and_enter_content_identity() {
    let graph = material_from_source(source(InputInterpretation::LinearHeight)).unwrap();
    let mut dto = MaterialGraphDto::from(graph);
    let seamless = dto
        .graph
        .iter_mut()
        .find(|node| matches!(node.params, MaterialNodeParams::Seamless { .. }))
        .unwrap();
    let encode = |node: &MaterialNode| {
        canonical_material_node_bytes(
            node,
            OutputPort::Height,
            &[CacheKey([3; 32])],
            source(InputInterpretation::LinearHeight).extent,
            source(InputInterpretation::LinearHeight).extent,
            CacheRevisions::CURRENT,
        )
    };
    let mut encodings = Vec::new();
    for (seed, strength) in [(42, 0.5), (43, 0.5), (42, 1.0)] {
        if let MaterialNodeParams::Seamless { method, .. } = &mut seamless.params {
            *method = SeamlessMethod::Histogram {
                base: SeamlessBase::Splat {
                    seed,
                    grid: 4,
                    overlap: 0.25,
                },
                strength,
            };
        }
        encodings.push(encode(seamless).unwrap());
    }
    assert_ne!(encodings[0], encodings[1]);
    assert_ne!(encodings[0], encodings[2]);
    if let MaterialNodeParams::Seamless { method, .. } = &mut seamless.params {
        *method = SeamlessMethod::Histogram {
            base: SeamlessBase::OffsetBlend { overlap: f32::NAN },
            strength: 1.0,
        };
    }
    assert!(encode(seamless).is_err());
    assert!(Graph::try_from(dto).is_err());
}

#[test]
fn schema_two_source_golden_bytes_are_independent_of_the_encoder() {
    // Literal layout: domain/schema/revisions, ColorSource/Albedo/Rgba16,
    // decoded hash, descriptor dimensions/interpretation, zero edges, two extents.
    let hex = concat!(
        "746578656c6974682e6e6f64652d636163686500020000000100000001000000010001",
        "0707070707070707070707070707070707070707070707070707070707070707",
        "110000000d0000000000000000110000000d000000110000000d000000"
    );
    let golden: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let source = source(InputInterpretation::SrgbPhoto);
    let node = MaterialNode {
        id: NodeId(456),
        inputs: vec![],
        params: MaterialNodeParams::ColorSource(source.clone()),
    };
    assert_eq!(
        canonical_material_node_bytes(
            &node,
            OutputPort::Albedo,
            &[],
            source.extent,
            source.extent,
            CacheRevisions::CURRENT
        )
        .unwrap(),
        golden
    );
}
