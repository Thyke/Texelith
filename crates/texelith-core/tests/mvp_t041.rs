//! T-041 behavioral contract. Proposed API/wire details are recorded in
//! docs/evidence/T-041/test-contract.md. No test-only graph implementation.
use serde_json::{Value, json};
use texelith_core::{
    Boundary, CacheKey, CacheRevisions, Extent, Graph, GraphDto, InputEdge, InputInterpretation,
    InputPort, MapFormat, MapKind, MaterialGraphDto, MaterialNodeParams, MaterialOutputs, Node,
    NodeId, NodeParams, NormalConvention, OutputPort, OutputRef, SourceDescriptor,
    canonical_material_node_bytes, material_from_source, material_node_cache_key, validate,
};

const EXTENT: Extent = Extent {
    width: 17,
    height: 13,
};

fn source(interpretation: InputInterpretation) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [7; 32],
        extent: EXTENT,
        interpretation,
    }
}

#[test]
fn diagnostic_conversion_preserves_legacy_graphs() {
    let dto: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    let graph = Graph::try_from(dto.clone()).unwrap();
    assert_eq!(GraphDto::try_from(graph.clone()).unwrap(), dto);
    assert_eq!(graph.legacy_dto().unwrap(), dto);
}

#[test]
fn diagnostic_conversion_returns_errors_for_material_graphs() {
    let legacy: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    let migrated = Graph::try_from(MaterialGraphDto::try_from(legacy).unwrap()).unwrap();
    let mut graphs = vec![migrated];
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
        InputInterpretation::LinearNormal,
    ] {
        graphs.push(material_from_source(source(interpretation)).unwrap());
    }
    for graph in graphs {
        assert!(GraphDto::try_from(graph.clone()).is_err());
        assert!(graph.legacy_dto().is_err());
        let value = serde_json::to_value(&graph).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(serde_json::from_value::<Graph>(value).unwrap(), graph);
    }
}

fn output(node: u64, port: &str) -> Value {
    json!({"node": node, "port": port})
}

fn edge(port: &str, node: u64, source_port: &str) -> Value {
    json!({"port": port, "source": output(node, source_port)})
}

fn node(id: u64, inputs: Vec<Value>, params: Value) -> Value {
    json!({"id": id, "inputs": inputs, "params": params})
}

fn material() -> Value {
    json!({
        "schema_version": 2,
        "graph": [
            node(1, vec![], json!({"ColorSource": source(InputInterpretation::SrgbPhoto)})),
            node(2, vec![edge("Color", 1, "Albedo")], json!({"HeightFromAlbedo": {
                "band_weights": [1.0], "coarse_weight": 1.0, "remove_coarse": false,
                "boundary": "Repeat"
            }})),
            node(3, vec![edge("Height", 2, "Height")], json!({"NormalFromHeight": {
                "strength": 1.0, "filter": "Sobel", "scale_weights": [1.0],
                "boundary": "Repeat", "convention": "OpenGl"
            }})),
            node(4, vec![edge("Height", 2, "Height")], json!({"HorizonAo": {
                "directions": 8, "steps": 8, "radius_source_px": 32.0,
                "bias": 0.01, "boundary": "Repeat"
            }})),
            node(5, vec![edge("Normal", 3, "Normal")], json!({"Curvature": {
                "edge_scale": 1.0, "cavity_scale": 1.0, "boundary": "Repeat"
            }})),
            node(6, vec![edge("Color", 1, "Albedo")], json!({"Roughness": {
                "curve": [[0.0, 0.0], [1.0, 1.0]], "base": 0.5, "invert": false
            }})),
            node(7, vec![edge("Color", 1, "Albedo")], json!({"Metallic": {
                "picked_color": [0.5, 0.5, 0.5], "radius": 0.25,
                "threshold": 0.5, "feather": 0.1, "mask_mode": "Replace"
            }}))
        ],
        "outputs": {
            "Albedo": output(1, "Albedo"), "Height": output(2, "Height"),
            "Normal": output(3, "Normal"), "Ao": output(4, "Ao"),
            "Curvature": output(5, "Curvature"), "Edge": output(5, "Edge"),
            "Cavity": output(5, "Cavity"), "Roughness": output(6, "Roughness"),
            "Metallic": output(7, "Metallic")
        },
        "selected_output": output(3, "Normal")
    })
}

fn graph(value: Value) -> Graph {
    let dto: MaterialGraphDto = serde_json::from_value(value).expect("valid material DTO");
    Graph::try_from(dto).expect("valid material graph")
}

fn encoded(graph: &Graph) -> Value {
    let dto = MaterialGraphDto::from(graph.clone());
    let _: &MaterialOutputs = &dto.outputs;
    serde_json::to_value(dto).unwrap()
}

fn rejects(value: Value, reason: &str) {
    // Either the untrusted DTO boundary or semantic validation may reject it.
    let result = serde_json::from_value::<MaterialGraphDto>(value)
        .map_err(|error| error.to_string())
        .and_then(|dto| Graph::try_from(dto).map_err(|error| error.to_string()));
    assert!(result.is_err(), "accepted {reason}");
}

fn reference(id: u64, port: OutputPort) -> OutputRef {
    OutputRef {
        node: NodeId(id),
        port,
    }
}

fn cache_bytes(value: Value, id: u64, port: OutputPort, inputs: &[CacheKey]) -> Vec<u8> {
    // Validate the fixture before testing its per-node encoding.
    let live = graph(value);
    let dto = MaterialGraphDto::from(live);
    let node = dto.graph.iter().find(|node| node.id == NodeId(id)).unwrap();
    let bytes =
        canonical_material_node_bytes(node, port, inputs, EXTENT, EXTENT, CacheRevisions::CURRENT)
            .unwrap();
    assert_eq!(
        material_node_cache_key(node, port, inputs, EXTENT, EXTENT, CacheRevisions::CURRENT)
            .unwrap(),
        CacheKey(*blake3::hash(&bytes).as_bytes()),
        "the production key must hash the canonical encoding"
    );
    bytes
}

fn keys(count: usize) -> Vec<CacheKey> {
    (0..count).map(|i| CacheKey([i as u8 + 3; 32])).collect()
}

fn with_mask(mut value: Value) -> Value {
    value["graph"].as_array_mut().unwrap().push(node(
        8,
        vec![],
        json!({"PaintMask": {
            "extent": EXTENT, "stroke_digest": vec![0; 32]
        }}),
    ));
    for index in [5, 6] {
        value["graph"][index]["inputs"]
            .as_array_mut()
            .unwrap()
            .push(edge("Mask", 8, "Mask"));
    }
    value
}

#[test]
fn typed_references_and_enum_tags_roundtrip() {
    let input = InputEdge {
        port: InputPort::Height,
        source: reference(2, OutputPort::Height),
    };
    assert_eq!(
        serde_json::to_value(&input).unwrap(),
        edge("Height", 2, "Height")
    );
    let restored: InputEdge = serde_json::from_value(edge("Height", 2, "Height")).unwrap();
    assert_eq!(restored.port, input.port);
    assert_eq!(restored.source.node, input.source.node);
    assert_eq!(restored.source.port, input.source.port);
    for (boundary, tag) in [(Boundary::Clamp, "Clamp"), (Boundary::Repeat, "Repeat")] {
        assert_eq!(serde_json::to_value(boundary).unwrap(), tag);
    }
    for (convention, tag) in [
        (NormalConvention::OpenGl, "OpenGl"),
        (NormalConvention::DirectX, "DirectX"),
    ] {
        assert_eq!(serde_json::to_value(convention).unwrap(), tag);
    }
}

#[test]
fn material_map_formats_keep_scalar_precision_and_linear_float_normals() {
    for kind in [
        MapKind::Height,
        MapKind::Ao,
        MapKind::Curvature,
        MapKind::Edge,
        MapKind::Cavity,
        MapKind::Roughness,
        MapKind::Metallic,
        MapKind::Mask,
    ] {
        assert_eq!(
            kind.format(),
            MapFormat::R32Float,
            "precision changed for {kind:?}"
        );
    }
    assert_eq!(MapKind::Albedo.format(), MapFormat::Rgba16Float);
    assert_eq!(MapKind::Normal.format(), MapFormat::Rgba16Float);
}

#[test]
fn shared_ancestors_and_all_curvature_outputs_roundtrip() {
    let first = graph(material());
    assert!(validate(&first).is_ok());
    let value = encoded(&first);
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["outputs"], material()["outputs"]);
    assert_eq!(value["selected_output"], output(3, "Normal"));
    let second = graph(value.clone());
    assert_eq!(encoded(&second), value);
}

#[test]
fn factory_is_separate_from_legacy_and_builds_a_repeat_material() {
    let material = material_from_source(source(InputInterpretation::SrgbPhoto)).unwrap();
    assert!(validate(&material).is_ok());
    let value = encoded(&material);
    assert_eq!(value["schema_version"], 2);
    for name in [
        "Albedo",
        "Height",
        "Normal",
        "Ao",
        "Edge",
        "Cavity",
        "Roughness",
        "Metallic",
    ] {
        assert!(value["outputs"][name].is_object(), "factory omitted {name}");
    }
    let mut spatial = 0;
    for node in value["graph"].as_array().unwrap() {
        for params in node["params"].as_object().unwrap().values() {
            if let Some(boundary) = params.get("boundary") {
                spatial += 1;
                assert_eq!(boundary, "Repeat");
            }
        }
    }
    assert!(
        spatial > 0,
        "factory must expose explicit spatial boundary policy"
    );
}

#[test]
fn invalid_factory_source_fails_before_construction() {
    let mut invalid = source(InputInterpretation::SrgbPhoto);
    invalid.extent.width = 4097;
    assert!(material_from_source(invalid).is_err());
}

#[test]
fn source_semantics_do_not_apply_photo_transfer_to_scalar_or_normal_data() {
    graph(material());
    graph(scalar_chain(2));
    graph(normal_source("DirectX"));
    let mut color = material();
    color["graph"][0]["params"]["ColorSource"]["interpretation"] = json!("LinearHeight");
    rejects(color, "linear height interpreted as photo color");
    let mut height = scalar_chain(2);
    height["graph"][0]["params"]["HeightSource"]["interpretation"] = json!("SrgbPhoto");
    rejects(height, "authored height using photo transfer");
    let mut normal = normal_source("DirectX");
    normal["graph"][0]["params"]["NormalSource"]["source"]["interpretation"] = json!("SrgbPhoto");
    rejects(normal, "encoded normal using photo transfer");
}

#[test]
fn missing_duplicate_and_wrong_semantic_ports_are_rejected() {
    // Every negative control starts with the same independently valid graph.
    graph(material());
    for case in 0..8 {
        let mut bad = material();
        match case {
            0 => bad["graph"][2]["inputs"] = json!([]),
            1 => {
                bad["graph"][2]["inputs"] =
                    json!([edge("Height", 2, "Height"), edge("Height", 2, "Height")])
            }
            2 => bad["graph"][2]["inputs"][0] = edge("Color", 2, "Height"),
            3 => bad["graph"][2]["inputs"][0] = edge("Height", 1, "Albedo"),
            4 => bad["graph"][2]["inputs"][0] = edge("Height", 2, "Normal"),
            5 => bad["graph"][2]["inputs"][0] = edge("Height", 999, "Height"),
            6 => bad["graph"][0]["inputs"] = json!([edge("Color", 1, "Albedo")]),
            _ => bad["graph"][3]["inputs"][0] = edge("Height", 5, "Curvature"),
        }
        rejects(bad, &format!("invalid input case {case}"));
    }
}

#[test]
fn named_outputs_and_selection_validate_both_node_and_port() {
    graph(material());
    for case in 0..7 {
        let mut bad = material();
        match case {
            0 => bad["outputs"]["Height"] = output(999, "Height"),
            1 => bad["outputs"]["Height"] = output(3, "Normal"),
            2 => bad["outputs"]["Normal"] = output(2, "Normal"),
            3 => bad["outputs"]["Edge"] = output(5, "Cavity"),
            4 => bad["selected_output"] = output(999, "Normal"),
            5 => bad["selected_output"] = output(5, "Normal"),
            _ => bad["graph"][1]["id"] = json!(1),
        }
        rejects(bad, &format!("invalid output case {case}"));
    }
}

#[test]
fn semantic_type_correct_cycles_are_rejected_even_when_disconnected() {
    for self_cycle in [false, true] {
        let mut bad = material();
        let parent = if self_cycle { 8 } else { 9 };
        bad["graph"].as_array_mut().unwrap().extend([
            node(
                8,
                vec![edge("Height", parent, "Height")],
                json!({"Blur": {
                    "sigma_source_px": 1.0, "boundary": "Repeat"
                }}),
            ),
            node(
                9,
                vec![edge("Height", 8, "Height")],
                json!({"Levels": {
                    "black": 0.0, "white": 1.0, "gamma": 1.0, "boundary": "Repeat"
                }}),
            ),
        ]);
        rejects(bad, "disconnected scalar cycle");
    }
}

#[test]
fn height_normal_feedback_is_a_cycle_even_when_ports_match() {
    let mut bad = material();
    let imported = normal_source("OpenGl");
    let mut integration = imported["graph"][1].clone();
    integration["id"] = json!(8);
    integration["inputs"][0] = edge("Normal", 3, "Normal");
    bad["graph"].as_array_mut().unwrap().push(integration);
    bad["graph"][2]["inputs"][0] = edge("Height", 8, "Height");
    rejects(bad, "Height -> Normal -> Height feedback");
}

#[test]
fn optional_masks_have_a_valid_default_but_dangling_masks_never_do() {
    let no_mask = graph(material());
    assert!(
        validate(&no_mask).is_ok(),
        "omitted masks mean constant one"
    );
    let explicit_mask = graph(with_mask(material()));
    assert!(validate(&explicit_mask).is_ok());
    for index in [5, 6] {
        for source in [
            edge("Mask", 999, "Mask"),
            edge("Mask", 1, "Albedo"),
            edge("Mask", 8, "Height"),
        ] {
            let mut bad = with_mask(material());
            bad["graph"][index]["inputs"][1] = source;
            rejects(bad, "dangling or mistyped optional mask");
        }
    }
}

#[test]
fn multi_input_extents_must_match_without_implicit_resize() {
    graph(with_mask(material()));
    for (width, height) in [(18, 13), (17, 14)] {
        let mut bad = with_mask(material());
        bad["graph"][7]["params"]["PaintMask"]["extent"] =
            json!({"width": width, "height": height});
        rejects(bad, "mismatched secondary input extent");
    }
}

#[test]
fn extent_limits_accept_npot_and_singleton_axes() {
    for (width, height) in [(1, 1), (1, 4096), (4096, 1), (257, 193), (4096, 4096)] {
        let mut value = material();
        value["graph"][0]["params"]["ColorSource"]["extent"] =
            json!({"width": width, "height": height});
        assert!(validate(&graph(value)).is_ok());
    }
    for (width, height) in [(0, 1), (1, 0), (4097, 1), (1, 4097)] {
        let mut bad = material();
        bad["graph"][0]["params"]["ColorSource"]["extent"] =
            json!({"width": width, "height": height});
        rejects(bad, "out-of-bounds source extent");
    }
}

fn scalar_chain(count: u64) -> Value {
    let mut nodes = vec![node(
        1,
        vec![],
        json!({"HeightSource": source(InputInterpretation::LinearHeight)}),
    )];
    for id in 2..=count {
        nodes.push(node(
            id,
            vec![edge("Height", id - 1, "Height")],
            json!({"Blur": {
                "sigma_source_px": 0.0, "boundary": "Repeat"
            }}),
        ));
    }
    json!({"schema_version": 2, "graph": nodes,
        "outputs": {"Height": output(count, "Height")}, "selected_output": output(count, "Height")})
}

#[test]
fn node_limit_is_inclusive_and_counts_unselected_nodes() {
    assert!(validate(&graph(scalar_chain(256))).is_ok());
    let mut bad = scalar_chain(257);
    bad["outputs"]["Height"] = output(1, "Height");
    bad["selected_output"] = output(1, "Height");
    rejects(bad, "257 nodes including unselected nodes");
    rejects(
        json!({"schema_version": 2, "graph": [], "outputs": {}, "selected_output": output(1, "Height")}),
        "empty graph",
    );
}

#[test]
fn band_limits_are_inclusive_for_height_and_normal_parameters() {
    for (index, kind, field) in [
        (1, "HeightFromAlbedo", "band_weights"),
        (2, "NormalFromHeight", "scale_weights"),
    ] {
        for count in [1, 16] {
            let mut value = material();
            value["graph"][index]["params"][kind][field] = json!(vec![1.0; count]);
            graph(value);
        }
        for count in [0, 17] {
            let mut bad = material();
            bad["graph"][index]["params"][kind][field] = json!(vec![1.0; count]);
            rejects(bad, "invalid number of pyramid bands");
        }
    }
}

#[test]
fn normal_strength_and_legacy_map_parameter_ranges_are_validated() {
    for strength in [-0.01, 10.01] {
        let mut bad = material();
        bad["graph"][2]["params"]["NormalFromHeight"]["strength"] = json!(strength);
        rejects(bad, "normal strength outside the legacy range");
    }
    for strength in [0.0, 10.0] {
        let mut value = material();
        value["graph"][2]["params"]["NormalFromHeight"]["strength"] = json!(strength);
        graph(value);
    }
    let mut blur = scalar_chain(2);
    blur["graph"][1]["params"]["Blur"]["sigma_source_px"] = json!(5.01);
    rejects(blur, "material adapter weakening the legacy blur range");
}

#[test]
fn ao_quality_and_radius_bounds_are_inclusive() {
    for (field, lower, upper) in [
        ("directions", 4.0, 32.0),
        ("steps", 4.0, 32.0),
        ("radius_source_px", 1.0, 512.0),
    ] {
        for valid in [lower, upper] {
            let mut value = material();
            value["graph"][3]["params"]["HorizonAo"][field] = if field == "radius_source_px" {
                json!(valid)
            } else {
                json!(valid as u32)
            };
            graph(value);
        }
        for invalid in [lower - 1.0, upper + 1.0] {
            let mut bad = material();
            bad["graph"][3]["params"]["HorizonAo"][field] = if field == "radius_source_px" {
                json!(invalid)
            } else {
                json!(invalid as u32)
            };
            rejects(bad, "AO quality outside ADR bounds");
        }
    }
}

#[test]
fn roughness_curve_bounds_and_order_are_validated() {
    for count in [2, 16] {
        let mut value = material();
        value["graph"][5]["params"]["Roughness"]["curve"] = json!(
            (0..count)
                .map(|i| {
                    let x = i as f32 / (count - 1) as f32;
                    [x, 1.0 - x]
                })
                .collect::<Vec<_>>()
        );
        graph(value);
    }
    for curve in [
        json!([]),
        json!([[0.0, 0.0]]),
        json!([[0.5, 0.0], [0.5, 1.0]]),
        json!([[1.0, 0.0], [0.0, 1.0]]),
        json!([[-0.1, 0.0], [1.0, 1.0]]),
        json!([[0.0, 0.0], [1.0, 1.1]]),
        json!((0..17).map(|i| [i as f32 / 16.0, 0.5]).collect::<Vec<_>>()),
    ] {
        let mut bad = material();
        bad["graph"][5]["params"]["Roughness"]["curve"] = curve;
        rejects(bad, "invalid roughness curve");
    }
    for base in [-0.01, 1.01] {
        let mut bad = material();
        bad["graph"][5]["params"]["Roughness"]["base"] = json!(base);
        rejects(bad, "roughness outside [0,1]");
    }
}

#[test]
fn metallic_validates_ranges_and_preserves_zero_feather() {
    for (field, invalid) in [
        ("radius", -0.01),
        ("radius", 1.733),
        ("threshold", -0.01),
        ("threshold", 1.01),
        ("feather", -0.01),
    ] {
        let mut bad = material();
        bad["graph"][6]["params"]["Metallic"][field] = json!(invalid);
        rejects(bad, "invalid metallic selection range");
    }
    let mut exact_step = material();
    exact_step["graph"][6]["params"]["Metallic"]["radius"] = json!(3.0_f32.sqrt());
    exact_step["graph"][6]["params"]["Metallic"]["feather"] = json!(0.0);
    let restored = encoded(&graph(exact_step));
    assert_eq!(restored["graph"][6]["params"]["Metallic"]["feather"], 0.0);
}

#[test]
fn dto_conversion_rejects_nonfinite_rust_parameters() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [1, 2, 3, 5, 6] {
            let mut dto: MaterialGraphDto = serde_json::from_value(material()).unwrap();
            match &mut dto.graph[index].params {
                MaterialNodeParams::HeightFromAlbedo { band_weights, .. } => {
                    band_weights[0] = value
                }
                MaterialNodeParams::NormalFromHeight { strength, .. } => *strength = value,
                MaterialNodeParams::HorizonAo {
                    radius_source_px, ..
                } => *radius_source_px = value,
                MaterialNodeParams::Roughness { curve, .. } => curve[0][1] = value,
                MaterialNodeParams::Metallic { picked_color, .. } => picked_color[0] = value,
                _ => panic!("fixture no longer covers the intended parameter"),
            }
            assert!(
                Graph::try_from(dto).is_err(),
                "accepted {value} at node {index}"
            );
        }
    }
}

fn normal_source(convention: &str) -> Value {
    json!({"schema_version": 2, "graph": [
        node(1, vec![], json!({"NormalSource": {
            "source": {"content_hash": vec![7; 32], "extent": EXTENT, "interpretation": "LinearNormal"},
            "convention": convention
        }})),
        node(2, vec![edge("Normal", 1, "Normal")], json!({"HeightFromNormal": {
            "boundary": "Repeat", "v_cycles": 8, "pre_smooth": 3, "post_smooth": 3,
            "coarsest_iterations": 32, "relative_tolerance": 0.001, "absolute_tolerance": 0.000001
        }}))
    ], "outputs": {"Normal": output(1, "Normal"), "Height": output(2, "Height")},
        "selected_output": output(2, "Height")})
}

#[test]
fn imported_normal_branch_is_acyclic_and_keeps_convention_metadata() {
    for convention in ["OpenGl", "DirectX"] {
        let value = encoded(&graph(normal_source(convention)));
        assert_eq!(
            value["graph"][0]["params"]["NormalSource"]["convention"],
            convention
        );
        assert_eq!(value["graph"][1]["inputs"][0], edge("Normal", 1, "Normal"));
        assert_eq!(value["graph"].as_array().unwrap().len(), 2);
    }
}

#[test]
fn poisson_cycle_limit_and_positive_tolerances_are_validated() {
    for cycles in [1, 16] {
        let mut value = normal_source("OpenGl");
        value["graph"][1]["params"]["HeightFromNormal"]["v_cycles"] = json!(cycles);
        graph(value);
    }
    for cycles in [0, 17] {
        let mut bad = normal_source("OpenGl");
        bad["graph"][1]["params"]["HeightFromNormal"]["v_cycles"] = json!(cycles);
        rejects(bad, "invalid solver cycle budget");
    }
    for field in ["relative_tolerance", "absolute_tolerance"] {
        let mut bad = normal_source("OpenGl");
        bad["graph"][1]["params"]["HeightFromNormal"][field] = json!(-0.01);
        rejects(bad, "negative solver tolerance");
    }
}

#[test]
fn unknown_schemas_enum_tags_and_fields_fail_closed() {
    graph(material());
    for version in [0, 3, u32::MAX] {
        let mut bad = material();
        bad["schema_version"] = json!(version);
        rejects(bad, "unknown graph schema");
    }
    for pointer in [
        "/graph/2/params/NormalFromHeight/filter",
        "/graph/2/params/NormalFromHeight/convention",
        "/graph/2/params/NormalFromHeight/boundary",
        "/graph/2/inputs/0/port",
        "/graph/2/inputs/0/source/port",
        "/graph/6/params/Metallic/mask_mode",
    ] {
        let mut bad = material();
        *bad.pointer_mut(pointer).unwrap() = json!("FutureTag");
        rejects(bad, "unknown enum tag");
    }
    for pointer in [
        "",
        "/graph/2",
        "/graph/2/params/NormalFromHeight",
        "/graph/2/inputs/0/source",
    ] {
        let mut bad = material();
        bad.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("future_field".into(), json!(true));
        rejects(bad, "unknown field that could change numerical intent");
    }
}

#[test]
fn cache_schema_two_has_explicit_domain_and_version() {
    let bytes = cache_bytes(material(), 3, OutputPort::Normal, &keys(1));
    let domain = b"texelith.node-cache\0";
    assert!(bytes.starts_with(domain));
    assert_eq!(&bytes[domain.len()..domain.len() + 4], &2u32.to_le_bytes());
}

#[test]
fn cache_includes_boundary_filter_convention_and_scale_weights() {
    let baseline = cache_bytes(material(), 3, OutputPort::Normal, &keys(1));
    for (field, changed) in [
        ("boundary", json!("Clamp")),
        ("filter", json!("Central")),
        ("filter", json!("Scharr")),
        ("convention", json!("DirectX")),
        ("scale_weights", json!([0.5, 1.0])),
        ("strength", json!(2.0)),
    ] {
        let mut value = material();
        value["graph"][2]["params"]["NormalFromHeight"][field] = changed;
        assert_ne!(
            baseline,
            cache_bytes(value, 3, OutputPort::Normal, &keys(1)),
            "omitted {field}"
        );
    }
}

#[test]
fn cache_includes_every_map_parameter_exercised_by_the_material() {
    for (index, id, port, field, changed) in [
        (1, 2, OutputPort::Height, "band_weights", json!([0.5])),
        (1, 2, OutputPort::Height, "coarse_weight", json!(0.5)),
        (1, 2, OutputPort::Height, "remove_coarse", json!(true)),
        (3, 4, OutputPort::Ao, "directions", json!(16)),
        (3, 4, OutputPort::Ao, "steps", json!(16)),
        (3, 4, OutputPort::Ao, "radius_source_px", json!(64.0)),
        (3, 4, OutputPort::Ao, "bias", json!(0.02)),
        (4, 5, OutputPort::Edge, "edge_scale", json!(2.0)),
        (4, 5, OutputPort::Cavity, "cavity_scale", json!(2.0)),
        (
            5,
            6,
            OutputPort::Roughness,
            "curve",
            json!([[0.0, 0.2], [1.0, 0.8]]),
        ),
        (5, 6, OutputPort::Roughness, "base", json!(0.7)),
        (5, 6, OutputPort::Roughness, "invert", json!(true)),
        (
            6,
            7,
            OutputPort::Metallic,
            "picked_color",
            json!([0.2, 0.5, 0.5]),
        ),
        (6, 7, OutputPort::Metallic, "radius", json!(0.5)),
        (6, 7, OutputPort::Metallic, "threshold", json!(0.7)),
        (6, 7, OutputPort::Metallic, "feather", json!(0.2)),
        (6, 7, OutputPort::Metallic, "mask_mode", json!("Add")),
        (6, 7, OutputPort::Metallic, "mask_mode", json!("Subtract")),
    ] {
        let baseline = cache_bytes(material(), id, port, &keys(1));
        let mut value = material();
        let params = value["graph"][index]["params"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        params[field] = changed;
        assert_ne!(
            baseline,
            cache_bytes(value, id, port, &keys(1)),
            "omitted node {id} parameter {field}"
        );
    }
}

#[test]
fn cache_includes_solver_quality_and_tolerances() {
    let baseline = cache_bytes(normal_source("OpenGl"), 2, OutputPort::Height, &keys(1));
    for (field, changed) in [
        ("v_cycles", json!(16)),
        ("pre_smooth", json!(4)),
        ("post_smooth", json!(4)),
        ("coarsest_iterations", json!(64)),
        ("relative_tolerance", json!(0.0001)),
        ("absolute_tolerance", json!(0.0000001)),
        ("boundary", json!("Clamp")),
    ] {
        let mut value = normal_source("OpenGl");
        value["graph"][1]["params"]["HeightFromNormal"][field] = changed;
        assert_ne!(
            baseline,
            cache_bytes(value, 2, OutputPort::Height, &keys(1)),
            "omitted {field}"
        );
    }
}

#[test]
fn cache_source_tags_and_imported_normal_convention_are_distinct() {
    let color = cache_bytes(material(), 1, OutputPort::Albedo, &[]);
    let height = cache_bytes(scalar_chain(2), 1, OutputPort::Height, &[]);
    let gl = cache_bytes(normal_source("OpenGl"), 1, OutputPort::Normal, &[]);
    let dx = cache_bytes(normal_source("DirectX"), 1, OutputPort::Normal, &[]);
    for (a, b) in [(&color, &height), (&color, &gl), (&height, &gl), (&gl, &dx)] {
        assert_ne!(a, b, "source semantic/convention tag collision");
    }
}

#[test]
fn cache_distinguishes_output_ports_with_identical_scalar_formats() {
    let ports = [OutputPort::Curvature, OutputPort::Edge, OutputPort::Cavity];
    let bytes: Vec<_> = ports
        .into_iter()
        .map(|port| cache_bytes(material(), 5, port, &keys(1)))
        .collect();
    for a in 0..bytes.len() {
        for b in a + 1..bytes.len() {
            assert_ne!(bytes[a], bytes[b], "output tag collision");
        }
    }
}

#[test]
fn cache_input_keys_are_ordered_and_all_arity_errors_are_rejected() {
    let baseline = cache_bytes(with_mask(material()), 6, OutputPort::Roughness, &keys(2));
    let mut swapped = keys(2);
    swapped.reverse();
    assert_ne!(
        baseline,
        cache_bytes(with_mask(material()), 6, OutputPort::Roughness, &swapped)
    );
    let dto = MaterialGraphDto::from(graph(with_mask(material())));
    for count in [0, 1, 3] {
        assert!(
            material_node_cache_key(
                &dto.graph[5],
                OutputPort::Roughness,
                &keys(count),
                EXTENT,
                EXTENT,
                CacheRevisions::CURRENT
            )
            .is_err()
        );
    }
    assert!(
        material_node_cache_key(
            &dto.graph[0],
            OutputPort::Albedo,
            &keys(1),
            EXTENT,
            EXTENT,
            CacheRevisions::CURRENT
        )
        .is_err()
    );
    assert!(
        material_node_cache_key(
            &dto.graph[2],
            OutputPort::Height,
            &keys(1),
            EXTENT,
            EXTENT,
            CacheRevisions::CURRENT
        )
        .is_err()
    );
}

#[test]
fn optional_mask_presence_and_stroke_digest_change_identity() {
    assert_ne!(
        cache_bytes(material(), 6, OutputPort::Roughness, &keys(1)),
        cache_bytes(with_mask(material()), 6, OutputPort::Roughness, &keys(2))
    );
    let first = with_mask(material());
    let mut changed = first.clone();
    changed["graph"][7]["params"]["PaintMask"]["stroke_digest"][31] = json!(1);
    assert_ne!(
        cache_bytes(first, 8, OutputPort::Mask, &[]),
        cache_bytes(changed, 8, OutputPort::Mask, &[])
    );
}

fn seamless(seed: u32) -> Value {
    json!({"schema_version": 2, "graph": [
        node(1, vec![], json!({"ColorSource": source(InputInterpretation::SrgbPhoto)})),
        node(2, vec![edge("Color", 1, "Albedo")], json!({"Seamless": {
            "method": {"Splat": {"seed": seed, "grid": 4, "overlap": 0.25}}, "boundary": "Repeat"
        }}))
    ], "outputs": {"Albedo": output(2, "Albedo")}, "selected_output": output(2, "Albedo")})
}

#[test]
fn seamless_seed_and_method_parameters_are_content_identity() {
    let baseline = cache_bytes(seamless(42), 2, OutputPort::Albedo, &keys(1));
    assert_eq!(
        baseline,
        cache_bytes(seamless(42), 2, OutputPort::Albedo, &keys(1))
    );
    assert_ne!(
        baseline,
        cache_bytes(seamless(43), 2, OutputPort::Albedo, &keys(1))
    );
    for (field, changed) in [("grid", json!(8)), ("overlap", json!(0.5))] {
        let mut value = seamless(42);
        value["graph"][1]["params"]["Seamless"]["method"]["Splat"][field] = changed;
        assert_ne!(
            baseline,
            cache_bytes(value, 2, OutputPort::Albedo, &keys(1))
        );
    }
    let mut offset = seamless(42);
    offset["graph"][1]["params"]["Seamless"]["method"] = json!({"OffsetBlend": {"overlap": 0.25}});
    assert_ne!(
        baseline,
        cache_bytes(offset, 2, OutputPort::Albedo, &keys(1))
    );
}

#[test]
fn cache_canonicalizes_signed_zero_for_scalar_vector_and_curve_fields() {
    for pointer in [
        "/graph/2/params/NormalFromHeight/strength",
        "/graph/1/params/HeightFromAlbedo/band_weights/0",
        "/graph/5/params/Roughness/curve/0/1",
        "/graph/6/params/Metallic/picked_color/0",
    ] {
        let (id, port) = if pointer.contains("NormalFromHeight") {
            (3, OutputPort::Normal)
        } else if pointer.contains("HeightFromAlbedo") {
            (2, OutputPort::Height)
        } else if pointer.contains("Roughness") {
            (6, OutputPort::Roughness)
        } else {
            (7, OutputPort::Metallic)
        };
        let mut positive = material();
        *positive.pointer_mut(pointer).unwrap() = json!(0.0_f32);
        let mut negative = positive.clone();
        *negative.pointer_mut(pointer).unwrap() = json!(-0.0_f32);
        assert_eq!(
            cache_bytes(positive, id, port, &keys(1)),
            cache_bytes(negative, id, port, &keys(1)),
            "signed zero at {pointer}"
        );
    }
}

#[test]
fn cache_ignores_ids_storage_order_selected_output_and_generations() {
    let value = material();
    let baseline = cache_bytes(value.clone(), 3, OutputPort::Normal, &keys(1));
    let mut renumbered = value.clone();
    for node in renumbered["graph"].as_array_mut().unwrap() {
        node["id"] = json!(node["id"].as_u64().unwrap() + 100);
        for edge in node["inputs"].as_array_mut().unwrap() {
            edge["source"]["node"] = json!(edge["source"]["node"].as_u64().unwrap() + 100);
        }
    }
    for output in renumbered["outputs"].as_object_mut().unwrap().values_mut() {
        output["node"] = json!(output["node"].as_u64().unwrap() + 100);
    }
    renumbered["selected_output"] = output(104, "Ao");
    renumbered["graph"].as_array_mut().unwrap().reverse();
    assert_eq!(
        baseline,
        cache_bytes(renumbered, 103, OutputPort::Normal, &keys(1))
    );
    let first = texelith_core::EvaluationSnapshot {
        generation: 0,
        graph: graph(value),
    };
    let mut later = first.clone();
    later.generation = u64::MAX;
    assert_eq!(encoded(&first.graph), encoded(&later.graph));
}

#[test]
fn cache_encodes_inputs_source_identity_extents_and_revisions() {
    let dto = MaterialGraphDto::from(graph(material()));
    let normal = &dto.graph[2];
    let baseline = material_node_cache_key(
        normal,
        OutputPort::Normal,
        &keys(1),
        EXTENT,
        EXTENT,
        CacheRevisions::CURRENT,
    )
    .unwrap();
    assert_ne!(
        baseline,
        material_node_cache_key(
            normal,
            OutputPort::Normal,
            &[CacheKey([99; 32])],
            EXTENT,
            EXTENT,
            CacheRevisions::CURRENT
        )
        .unwrap()
    );
    for (eval, source_extent) in [
        (
            Extent {
                width: 18,
                ..EXTENT
            },
            EXTENT,
        ),
        (
            Extent {
                height: 14,
                ..EXTENT
            },
            EXTENT,
        ),
        (
            EXTENT,
            Extent {
                width: 18,
                ..EXTENT
            },
        ),
        (
            EXTENT,
            Extent {
                height: 14,
                ..EXTENT
            },
        ),
    ] {
        assert_ne!(
            baseline,
            material_node_cache_key(
                normal,
                OutputPort::Normal,
                &keys(1),
                eval,
                source_extent,
                CacheRevisions::CURRENT
            )
            .unwrap()
        );
    }
    for revisions in [
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
            baseline,
            material_node_cache_key(
                normal,
                OutputPort::Normal,
                &keys(1),
                EXTENT,
                EXTENT,
                revisions
            )
            .unwrap()
        );
    }
    let baseline = cache_bytes(material(), 1, OutputPort::Albedo, &[]);
    let mut changed = material();
    changed["graph"][0]["params"]["ColorSource"]["content_hash"][0] = json!(8);
    assert_ne!(baseline, cache_bytes(changed, 1, OutputPort::Albedo, &[]));
}

#[test]
fn cache_rejects_invalid_extents_and_nonfinite_dtos_directly() {
    let mut dto: MaterialGraphDto = serde_json::from_value(material()).unwrap();
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
        assert!(
            material_node_cache_key(
                &dto.graph[2],
                OutputPort::Normal,
                &keys(1),
                invalid,
                EXTENT,
                CacheRevisions::CURRENT
            )
            .is_err()
        );
        assert!(
            material_node_cache_key(
                &dto.graph[2],
                OutputPort::Normal,
                &keys(1),
                EXTENT,
                invalid,
                CacheRevisions::CURRENT
            )
            .is_err()
        );
    }
    if let MaterialNodeParams::NormalFromHeight { strength, .. } = &mut dto.graph[2].params {
        *strength = f32::NAN;
    }
    assert!(
        material_node_cache_key(
            &dto.graph[2],
            OutputPort::Normal,
            &keys(1),
            EXTENT,
            EXTENT,
            CacheRevisions::CURRENT
        )
        .is_err()
    );
}

#[test]
fn schema_one_migration_preserves_nondefault_parameters_and_numerical_policy() {
    let legacy: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    let old = Graph::try_from(legacy.clone()).unwrap();
    let migrated = MaterialGraphDto::try_from(legacy).unwrap();
    let live = Graph::try_from(migrated).unwrap();
    let value = encoded(&live);
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["graph"].as_array().unwrap().len(), 4);
    assert_eq!(
        value["graph"][0]["params"]["LegacySource"],
        json!(source(InputInterpretation::SrgbPhoto))
    );
    assert_eq!(
        value["graph"][1]["params"]["Levels"],
        json!({"black": 0.25, "white": 0.75, "gamma": 2.0, "boundary": "Clamp"})
    );
    assert_eq!(
        value["graph"][2]["params"]["Blur"],
        json!({"sigma_source_px": 1.25, "boundary": "Clamp"})
    );
    assert_eq!(
        value["graph"][3]["params"]["NormalFromHeight"],
        json!({
            "strength": 2.5, "filter": "Central", "scale_weights": [1.0], "boundary": "Clamp", "convention": "OpenGl"
        })
    );
    assert_eq!(value["selected_output"], output(4, "Normal"));
    assert_eq!(value["outputs"]["Height"], output(3, "Height"));
    assert_eq!(value["outputs"]["Normal"], output(4, "Normal"));
    assert!(
        value["outputs"].get("Albedo").is_none(),
        "legacy scalar source must not become linear RGB"
    );
    for (index, parent) in [(1, 1), (2, 2), (3, 3)] {
        assert_eq!(
            value["graph"][index]["inputs"],
            json!([edge("Height", parent, "Height")])
        );
    }
    assert_eq!(old.nodes().len(), 4, "legacy construction remains usable");
    assert_eq!(
        encoded(&graph(value.clone())),
        value,
        "migration is stable after a schema-2 roundtrip"
    );
}

#[test]
fn migration_validates_legacy_graphs_instead_of_repairing_them() {
    let fixture = include_str!("fixtures/t041/schema1-diagnostic.json");
    for case in 0..5 {
        let mut value: Value = serde_json::from_str(fixture).unwrap();
        match case {
            0 => value["graph"][1]["input"] = json!(999),
            1 => value["graph"][1]["input"] = json!(3),
            2 => value["graph"][2]["params"]["Blur"]["sigma_source_px"] = json!(-1.0),
            3 => value["selected_output"] = json!(999),
            _ => value["schema_version"] = json!(3),
        }
        let dto: GraphDto = serde_json::from_value(value).unwrap();
        assert!(
            MaterialGraphDto::try_from(dto).is_err(),
            "migration silently repaired case {case}"
        );
    }
}

#[test]
fn migration_keeps_legacy_linear_red_interpretation_and_selected_height() {
    let mut legacy: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    if let NodeParams::Source(source) = &mut legacy.graph[0].params {
        source.interpretation = InputInterpretation::LinearHeight;
    }
    legacy.selected_output = NodeId(3);
    let migrated = Graph::try_from(MaterialGraphDto::try_from(legacy).unwrap()).unwrap();
    let value = encoded(&migrated);
    assert_eq!(
        value["graph"][0]["params"]["LegacySource"]["interpretation"],
        "LinearHeight"
    );
    assert_eq!(value["selected_output"], output(3, "Height"));
    assert_eq!(value["graph"][2]["params"]["Blur"]["boundary"], "Clamp");
}

#[test]
fn legacy_struct_literal_constructors_and_schema_one_hash_golden_stay_usable() {
    // Literal schema-1 BLAKE3 from cache_identity.rs, independent of schema 2.
    let normal = Node {
        id: NodeId(4),
        input: Some(NodeId(3)),
        params: NodeParams::NormalFromHeight { strength: 2.5 },
    };
    let hash = texelith_core::node_cache_key(
        &normal,
        &[CacheKey([3; 32])],
        Extent {
            width: 100,
            height: 50,
        },
        Extent {
            width: 200,
            height: 100,
        },
    )
    .unwrap();
    let expected = "18f5953a2485462da7d482a140891913cd015d908f2e1af3f7e7418e24f0ce61";
    let actual: String = hash.0.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(actual, expected);
    let legacy: GraphDto =
        serde_json::from_str(include_str!("fixtures/t041/schema1-diagnostic.json")).unwrap();
    assert!(Graph::new(legacy.graph, legacy.selected_output).is_ok());
}
