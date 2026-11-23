//! Command wire format and discovery schema contracts. JSON is used only at
//! the production DTO boundary; all mutation is through Session.
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use texelith_core::*;

const EXTENT: Extent = Extent {
    width: 17,
    height: 13,
};

fn source(interpretation: InputInterpretation) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [44; 32],
        extent: EXTENT,
        interpretation,
    }
}

fn fixture() -> Graph {
    let mut dto = MaterialGraphDto::from(
        material_from_source(source(InputInterpretation::SrgbPhoto)).unwrap(),
    );
    let height = dto.outputs[&MapKind::Height];
    let edge = |port, source| InputEdge { port, source };
    // Separate imported-normal branch: a solver must never feed its own producer.
    dto.graph.extend([
        MaterialNode {
            id: NodeId(100),
            inputs: vec![],
            params: MaterialNodeParams::NormalSource {
                source: source(InputInterpretation::LinearNormal),
                convention: NormalConvention::OpenGl,
            },
        },
        MaterialNode {
            id: NodeId(101),
            inputs: vec![edge(
                InputPort::Normal,
                OutputRef {
                    node: NodeId(100),
                    port: OutputPort::Normal,
                },
            )],
            params: MaterialNodeParams::HeightFromNormal {
                boundary: Boundary::Repeat,
                v_cycles: 8,
                pre_smooth: 3,
                post_smooth: 3,
                coarsest_iterations: 32,
                relative_tolerance: 1e-3,
                absolute_tolerance: 1e-6,
            },
        },
        MaterialNode {
            id: NodeId(102),
            inputs: vec![edge(InputPort::Height, height)],
            params: MaterialNodeParams::Levels {
                black: 0.0,
                white: 1.0,
                gamma: 1.0,
                boundary: Boundary::Repeat,
            },
        },
        MaterialNode {
            id: NodeId(103),
            inputs: vec![edge(
                InputPort::Height,
                OutputRef {
                    node: NodeId(102),
                    port: OutputPort::Height,
                },
            )],
            params: MaterialNodeParams::Blur {
                sigma_source_px: 0.0,
                boundary: Boundary::Repeat,
            },
        },
        MaterialNode {
            id: NodeId(104),
            inputs: vec![],
            params: MaterialNodeParams::PaintMask {
                boundary: texelith_core::Boundary::Repeat,
                extent: EXTENT,
                stroke_digest: [0; 32],
                strokes: vec![],
            },
        },
    ]);
    dto.outputs.insert(
        MapKind::Mask,
        OutputRef {
            node: NodeId(104),
            port: OutputPort::Mask,
        },
    );
    Graph::try_from(dto).unwrap()
}

fn loaded() -> Session {
    Session::from_snapshot(EvaluationSnapshot {
        generation: 40,
        graph: fixture(),
    })
}
fn snapshot(session: &Session) -> EvaluationSnapshot {
    session.snapshot().unwrap()
}
fn node_id(graph: &Graph, kind: &str) -> NodeId {
    graph
        .material_nodes()
        .iter()
        .find(|node| {
            serde_json::to_value(&node.params)
                .unwrap()
                .get(kind)
                .is_some()
        })
        .unwrap()
        .id
}
fn field(graph: &Graph, node: NodeId, name: &str) -> Value {
    let params = serde_json::to_value(&graph.material_node(node).unwrap().params).unwrap();
    params.as_object().unwrap().values().next().unwrap()[name].clone()
}
fn request(
    node: NodeId,
    parameter: &str,
    value: Value,
    gesture: Option<GestureId>,
) -> CommandRequest {
    serde_json::from_value(json!({ "id": "node.set_parameter", "payload": {
        "address": {"node": node, "parameter": parameter}, "value": value, "gesture": gesture,
    }}))
    .unwrap_or_else(|error| {
        panic!("typed parameter {parameter} must route through CommandRequest: {error}")
    })
}
fn address(node: NodeId, parameter: &str) -> ParameterAddress {
    serde_json::from_value(json!({"node": node, "parameter": parameter})).unwrap()
}
fn set(
    session: &mut Session,
    node: NodeId,
    parameter: &str,
    value: Value,
    gesture: Option<GestureId>,
) -> CommandOutcome {
    session
        .dispatch(request(node, parameter, value, gesture))
        .unwrap()
}
fn observable(
    session: &Session,
) -> (
    Option<EvaluationSnapshot>,
    HistoryState,
    Vec<CommandAvailability>,
) {
    (
        session.snapshot(),
        session.history(),
        session.availability(),
    )
}
fn rejects(session: &mut Session, request: CommandRequest) {
    let before = observable(session);
    assert!(
        session.dispatch(request).is_err(),
        "invalid command accepted"
    );
    assert_eq!(observable(session), before, "failed command must be atomic");
}
fn keys(graph: &Graph) -> BTreeMap<NodeId, CacheKey> {
    let mut keys = BTreeMap::new();
    for node in graph.topological_material_nodes() {
        // These fixtures have only primary outputs as inputs. Named curvature
        // outputs are compared separately below using their actual output tags.
        let inputs: Vec<_> = node
            .inputs
            .iter()
            .map(|edge| keys[&edge.source.node])
            .collect();
        keys.insert(
            node.id,
            material_node_cache_key(
                node,
                node.primary_output(),
                &inputs,
                EXTENT,
                EXTENT,
                CacheRevisions::CURRENT,
            )
            .unwrap(),
        );
    }
    keys
}

fn assert_parameter_eq(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            if (a.is_u64() || a.is_i64()) && (b.is_u64() || b.is_i64()) {
                assert_eq!(a, b, "integer parameters must remain exact");
                return;
            }
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= f64::from(f32::EPSILON) * a.abs().max(b.abs()),
                "{a} != {b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_parameter_eq(a, b);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, value) in a {
                assert_parameter_eq(value, &b[key]);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

// Feed actual IEEE nonfinite values through production Deserialize. JSON cannot
// represent them, while Rust frontends can construct the same owned typed DTOs.
struct Tagged<D> {
    tag: &'static str,
    payload: D,
}
struct Payload<D>(D);
impl<'de, D: serde::Deserializer<'de, Error = serde::de::value::Error>> serde::de::EnumAccess<'de>
    for Tagged<D>
{
    type Error = serde::de::value::Error;
    type Variant = Payload<D>;
    fn variant_seed<V: serde::de::DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        use serde::de::IntoDeserializer;
        Ok((
            seed.deserialize(self.tag.into_deserializer())?,
            Payload(self.payload),
        ))
    }
}
impl<'de, D: serde::Deserializer<'de, Error = serde::de::value::Error>>
    serde::de::VariantAccess<'de> for Payload<D>
{
    type Error = serde::de::value::Error;
    fn newtype_variant_seed<T: serde::de::DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        seed.deserialize(self.0)
    }
    fn unit_variant(self) -> Result<(), Self::Error> {
        Err(serde::de::Error::custom(
            "fixture expects a newtype payload",
        ))
    }
    fn tuple_variant<V: serde::de::Visitor<'de>>(
        self,
        _: usize,
        _: V,
    ) -> Result<V::Value, Self::Error> {
        Err(serde::de::Error::custom(
            "fixture expects a newtype payload",
        ))
    }
    fn struct_variant<V: serde::de::Visitor<'de>>(
        self,
        _: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, Self::Error> {
        Err(serde::de::Error::custom(
            "fixture expects a newtype payload",
        ))
    }
}
struct FloatRow([f32; 2]);
impl<'de> serde::de::IntoDeserializer<'de, serde::de::value::Error> for FloatRow {
    type Deserializer =
        serde::de::value::SeqDeserializer<std::array::IntoIter<f32, 2>, serde::de::value::Error>;
    fn into_deserializer(self) -> Self::Deserializer {
        serde::de::value::SeqDeserializer::new(self.0.into_iter())
    }
}
fn typed_payload<'de, D: serde::Deserializer<'de, Error = serde::de::value::Error>>(
    tag: &'static str,
    payload: D,
) -> Result<ParameterValue, serde::de::value::Error> {
    ParameterValue::deserialize(serde::de::value::EnumAccessDeserializer::new(Tagged {
        tag,
        payload,
    }))
}

// Every control is independently exercised through serde -> typed request ->
// authoritative dispatcher -> immutable graph, then no-op, undo and redo.
fn edit_contract(kind: &str, parameter: &str, field_name: &str, typed: Value, expected: Value) {
    let mut session = loaded();
    let before = snapshot(&session);
    let node = node_id(&before.graph, kind);
    let before_keys = keys(&before.graph);
    let req = request(node, parameter, typed.clone(), None);
    assert_eq!(
        serde_json::from_value::<CommandRequest>(serde_json::to_value(&req).unwrap()).unwrap(),
        req
    );
    let outcome = session.dispatch(req).unwrap();
    let after = snapshot(&session);
    assert_eq!(after.generation, before.generation + 1);
    assert_parameter_eq(&field(&after.graph, node, field_name), &expected);
    let mut expected_dto = MaterialGraphDto::from(before.graph.clone());
    let expected_node = expected_dto
        .graph
        .iter_mut()
        .find(|n| n.id == node)
        .unwrap();
    let mut params = serde_json::to_value(&expected_node.params).unwrap();
    params.as_object_mut().unwrap().values_mut().next().unwrap()[field_name] = expected;
    expected_node.params = serde_json::from_value(params).unwrap();
    assert_eq!(
        after.graph,
        Graph::try_from(expected_dto).unwrap(),
        "an edit changes only its addressed field"
    );
    assert_eq!(outcome.snapshot, Some(after.clone()));
    assert!(outcome.effects.is_empty());
    assert_eq!(session.history().undo_entries, 1);
    assert_eq!(session.history().redo_entries, 0);
    assert_ne!(
        keys(&after.graph)[&node],
        before_keys[&node],
        "control must enter content identity"
    );
    assert_eq!(
        before.graph,
        fixture(),
        "previous snapshot remains immutable"
    );
    let stable = observable(&session);
    let noop = set(&mut session, node, parameter, typed, None);
    assert!(noop.snapshot.is_none());
    assert_eq!(
        observable(&session),
        stable,
        "no-op must preserve generation and history"
    );
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(snapshot(&session).graph, before.graph);
    assert_eq!(snapshot(&session).generation, after.generation + 1);
    assert_eq!(keys(&snapshot(&session).graph), before_keys);
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(snapshot(&session).graph, after.graph);
    assert_eq!(snapshot(&session).generation, after.generation + 2);
}

macro_rules! control {
    ($name:ident, $kind:literal, $id:literal, $field:literal, $tag:literal, $value:expr) => {
        #[test] fn $name() { edit_contract($kind, $id, $field, json!({$tag: $value}), json!($value)); }
    };
}
control!(
    height_bands,
    "HeightFromAlbedo",
    "BandWeights",
    "band_weights",
    "FloatVector",
    [0.25, -0.5, 2.0]
);
control!(
    height_coarse,
    "HeightFromAlbedo",
    "CoarseWeight",
    "coarse_weight",
    "Float",
    -0.5
);
control!(
    height_remove_coarse,
    "HeightFromAlbedo",
    "RemoveCoarse",
    "remove_coarse",
    "Bool",
    true
);
control!(
    normal_strength,
    "NormalFromHeight",
    "Strength",
    "strength",
    "Float",
    2.0
);
control!(
    normal_filter_central,
    "NormalFromHeight",
    "Filter",
    "filter",
    "Filter",
    "Central"
);
control!(
    normal_filter_scharr,
    "NormalFromHeight",
    "Filter",
    "filter",
    "Filter",
    "Scharr"
);
control!(
    normal_scales,
    "NormalFromHeight",
    "ScaleWeights",
    "scale_weights",
    "FloatVector",
    [0.25, 0.75]
);
control!(
    normal_directx,
    "NormalFromHeight",
    "Convention",
    "convention",
    "Convention",
    "DirectX"
);
control!(
    imported_normal_directx,
    "NormalSource",
    "Convention",
    "convention",
    "Convention",
    "DirectX"
);
control!(
    solver_cycles,
    "HeightFromNormal",
    "VCycles",
    "v_cycles",
    "UInt",
    16
);
control!(
    solver_pre_smooth,
    "HeightFromNormal",
    "PreSmooth",
    "pre_smooth",
    "UInt",
    4
);
control!(
    solver_post_smooth,
    "HeightFromNormal",
    "PostSmooth",
    "post_smooth",
    "UInt",
    4
);
control!(
    solver_coarsest,
    "HeightFromNormal",
    "CoarsestIterations",
    "coarsest_iterations",
    "UInt",
    64
);
control!(
    solver_relative_tolerance,
    "HeightFromNormal",
    "RelativeTolerance",
    "relative_tolerance",
    "Float",
    0.002
);
control!(
    solver_absolute_tolerance,
    "HeightFromNormal",
    "AbsoluteTolerance",
    "absolute_tolerance",
    "Float",
    0.000002
);
control!(
    ao_directions,
    "HorizonAo",
    "Directions",
    "directions",
    "UInt",
    16
);
control!(ao_steps, "HorizonAo", "Steps", "steps", "UInt", 16);
control!(
    ao_radius,
    "HorizonAo",
    "RadiusSourcePx",
    "radius_source_px",
    "Float",
    64.0
);
control!(ao_bias, "HorizonAo", "Bias", "bias", "Float", 0.02);
control!(
    curvature_edge,
    "Curvature",
    "EdgeScale",
    "edge_scale",
    "Float",
    2.0
);
control!(
    curvature_cavity,
    "Curvature",
    "CavityScale",
    "cavity_scale",
    "Float",
    2.0
);
control!(
    roughness_curve,
    "Roughness",
    "Curve",
    "curve",
    "Curve",
    [[0.0, 0.25], [0.5, 0.75], [1.0, 0.5]]
);
control!(roughness_base, "Roughness", "Base", "base", "Float", 0.75);
control!(
    roughness_invert,
    "Roughness",
    "Invert",
    "invert",
    "Bool",
    true
);
control!(
    metallic_color,
    "Metallic",
    "PickedColor",
    "picked_color",
    "Color",
    [0.25, 0.5, 0.75]
);
control!(
    metallic_radius,
    "Metallic",
    "Radius",
    "radius",
    "Float",
    0.5
);
control!(
    metallic_threshold,
    "Metallic",
    "Threshold",
    "threshold",
    "Float",
    0.75
);
control!(
    metallic_feather,
    "Metallic",
    "Feather",
    "feather",
    "Float",
    0.0
);
control!(
    metallic_mask_add,
    "Metallic",
    "MaskMode",
    "mask_mode",
    "MaskMode",
    "Add"
);
control!(
    metallic_mask_subtract,
    "Metallic",
    "MaskMode",
    "mask_mode",
    "MaskMode",
    "Subtract"
);
control!(
    material_levels_black,
    "Levels",
    "Black",
    "black",
    "Float",
    0.25
);
control!(
    material_levels_white,
    "Levels",
    "White",
    "white",
    "Float",
    0.75
);
control!(
    material_levels_gamma,
    "Levels",
    "Gamma",
    "gamma",
    "Float",
    2.0
);
control!(
    material_blur,
    "Blur",
    "SigmaSourcePx",
    "sigma_source_px",
    "Float",
    2.0
);
control!(
    seamless_splat,
    "Seamless",
    "SeamlessMethod",
    "method",
    "SeamlessMethod",
    json!({"Splat": {"seed": 4294967295u32, "grid": 8, "overlap": 0.5}})
);
control!(
    seamless_histogram_offset,
    "Seamless",
    "SeamlessMethod",
    "method",
    "SeamlessMethod",
    json!({"Histogram": {"base": {"OffsetBlend": {"overlap": 0.1}}, "strength": 0.5}})
);
control!(
    seamless_histogram_splat,
    "Seamless",
    "SeamlessMethod",
    "method",
    "SeamlessMethod",
    json!({"Histogram": {"base": {"Splat": {"seed": 42, "grid": 2, "overlap": 0.25}}, "strength": 1.0}})
);

#[test]
fn boundary_is_editable_on_every_spatial_node_and_roundtrips_history() {
    for kind in [
        "HeightFromAlbedo",
        "NormalFromHeight",
        "HeightFromNormal",
        "HorizonAo",
        "Curvature",
        "Levels",
        "Blur",
        "Seamless",
    ] {
        edit_contract(
            kind,
            "Boundary",
            "boundary",
            json!({"Boundary": "Clamp"}),
            json!("Clamp"),
        );
    }
}

fn range_contract(kind: &str, parameter: &str, tag: &str, valid: &[Value], invalid: &[Value]) {
    let mut session = loaded();
    let node = node_id(&snapshot(&session).graph, kind);
    for value in valid {
        set(&mut session, node, parameter, json!({tag: value}), None);
    }
    for value in invalid {
        rejects(
            &mut session,
            request(node, parameter, json!({tag: value}), None),
        );
    }
}

#[test]
fn scalar_and_integer_ranges_match_the_accepted_dto_boundaries() {
    for (kind, id, tag, good, bad) in [
        (
            "NormalFromHeight",
            "Strength",
            "Float",
            vec![json!(0.0), json!(10.0)],
            vec![json!(-0.001), json!(10.001)],
        ),
        (
            "HeightFromNormal",
            "VCycles",
            "UInt",
            vec![json!(1), json!(16)],
            vec![json!(0), json!(17)],
        ),
        (
            "HorizonAo",
            "Directions",
            "UInt",
            vec![json!(4), json!(32)],
            vec![json!(3), json!(33)],
        ),
        (
            "HorizonAo",
            "Steps",
            "UInt",
            vec![json!(4), json!(32)],
            vec![json!(3), json!(33)],
        ),
        (
            "HorizonAo",
            "RadiusSourcePx",
            "Float",
            vec![json!(1.0), json!(512.0)],
            vec![json!(0.99), json!(512.01)],
        ),
        (
            "Roughness",
            "Base",
            "Float",
            vec![json!(0.0), json!(1.0)],
            vec![json!(-0.01), json!(1.01)],
        ),
        (
            "Metallic",
            "Radius",
            "Float",
            vec![json!(0.0), json!(3.0_f32.sqrt())],
            vec![json!(-0.01), json!(1.74)],
        ),
        (
            "Metallic",
            "Threshold",
            "Float",
            vec![json!(0.0), json!(1.0)],
            vec![json!(-0.01), json!(1.01)],
        ),
        (
            "Levels",
            "Gamma",
            "Float",
            vec![json!(0.1), json!(10.0)],
            vec![json!(0.099), json!(10.01)],
        ),
        (
            "Blur",
            "SigmaSourcePx",
            "Float",
            vec![json!(0.0), json!(5.0)],
            vec![json!(-0.01), json!(5.01)],
        ),
    ] {
        range_contract(kind, id, tag, &good, &bad);
    }
    for id in ["PreSmooth", "PostSmooth", "CoarsestIterations"] {
        range_contract(
            "HeightFromNormal",
            id,
            "UInt",
            &[json!(1), json!(32)],
            &[json!(0)],
        );
    }
    for id in ["RelativeTolerance", "AbsoluteTolerance"] {
        range_contract(
            "HeightFromNormal",
            id,
            "Float",
            &[json!(0.0001), json!(0.1)],
            &[json!(0.0), json!(-0.1)],
        );
    }
    for (kind, id) in [
        ("HorizonAo", "Bias"),
        ("Curvature", "EdgeScale"),
        ("Curvature", "CavityScale"),
        ("Metallic", "Feather"),
    ] {
        range_contract(
            kind,
            id,
            "Float",
            &[json!(0.0), json!(100.0)],
            &[json!(-0.01)],
        );
    }
}

#[test]
fn vector_color_and_curve_bounds_are_validated_transactionally() {
    for (kind, id) in [
        ("HeightFromAlbedo", "BandWeights"),
        ("NormalFromHeight", "ScaleWeights"),
    ] {
        range_contract(
            kind,
            id,
            "FloatVector",
            &[json!([-1.0]), json!(vec![1.0; 16])],
            &[json!([]), json!(vec![1.0; 17])],
        );
    }
    range_contract(
        "Metallic",
        "PickedColor",
        "Color",
        &[json!([0.0, 0.5, 1.0])],
        &[json!([-0.01, 0.5, 1.0]), json!([0.0, 1.01, 0.5])],
    );
    let sixteen: Vec<_> = (0..16).map(|i| [i as f32 / 15.0, 0.5]).collect();
    let seventeen: Vec<_> = (0..17).map(|i| [i as f32 / 16.0, 0.5]).collect();
    range_contract(
        "Roughness",
        "Curve",
        "Curve",
        &[json!([[0.0, 1.0], [1.0, 0.0]]), json!(sixteen)],
        &[
            json!([]),
            json!([[0.0, 0.5]]),
            json!(seventeen),
            json!([[0.5, 0.0], [0.5, 1.0]]),
            json!([[1.0, 0.0], [0.0, 1.0]]),
            json!([[-0.1, 0.0], [1.0, 1.0]]),
            json!([[0.0, -0.1], [1.0, 1.0]]),
            json!([[0.0, 0.0], [1.1, 1.0]]),
            json!([[0.0, 0.0], [1.0, 1.1]]),
        ],
    );
}

#[test]
fn seamless_nested_ranges_and_all_modes_are_validated() {
    range_contract(
        "Seamless",
        "SeamlessMethod",
        "SeamlessMethod",
        &[
            json!({"OffsetBlend": {"overlap": 0.1}}),
            json!({"OffsetBlend": {"overlap": 0.5}}),
            json!({"Splat": {"seed": 0, "grid": 2, "overlap": 0.1}}),
            json!({"Splat": {"seed": u32::MAX, "grid": 8, "overlap": 0.5}}),
            json!({"Histogram": {"base": {"Splat": {"seed": 42, "grid": 4, "overlap": 0.25}}, "strength": 0.0}}),
        ],
        &[
            json!({"OffsetBlend": {"overlap": 0.099}}),
            json!({"OffsetBlend": {"overlap": 0.501}}),
            json!({"Splat": {"seed": 0, "grid": 1, "overlap": 0.25}}),
            json!({"Splat": {"seed": 0, "grid": 9, "overlap": 0.25}}),
            json!({"Histogram": {"base": {"OffsetBlend": {"overlap": 0.0}}, "strength": 0.5}}),
            json!({"Histogram": {"base": {"Splat": {"seed": 0, "grid": 1, "overlap": 0.25}}, "strength": 0.5}}),
            json!({"Histogram": {"base": {"OffsetBlend": {"overlap": 0.25}}, "strength": -0.01}}),
            json!({"Histogram": {"base": {"OffsetBlend": {"overlap": 0.25}}, "strength": 1.01}}),
        ],
    );
}

#[test]
fn levels_interval_and_wrong_node_or_value_type_do_not_mutate_state() {
    let mut session = loaded();
    let graph = snapshot(&session).graph;
    let levels = node_id(&graph, "Levels");
    set(&mut session, levels, "Black", json!({"Float": 0.5}), None);
    for (id, value) in [
        ("White", 0.5),
        ("White", 0.5000005),
        ("Black", -0.1),
        ("White", 1.1),
    ] {
        rejects(
            &mut session,
            request(levels, id, json!({"Float": value}), None),
        );
    }
    let normal = node_id(&graph, "NormalFromHeight");
    for (node, id, value) in [
        (normal, "Base", json!({"Float": 0.5})),
        (levels, "Gamma", json!({"Bool": true})),
        (normal, "Strength", json!({"UInt": 2})),
        (NodeId(999), "Strength", json!({"Float": 2.0})),
    ] {
        rejects(&mut session, request(node, id, value, None));
    }
}

#[test]
fn nonfinite_scalar_commands_are_rejected_for_each_material_float_control() {
    let mut session = loaded();
    for (kind, id) in [
        ("HeightFromAlbedo", "CoarseWeight"),
        ("NormalFromHeight", "Strength"),
        ("HeightFromNormal", "RelativeTolerance"),
        ("HeightFromNormal", "AbsoluteTolerance"),
        ("HorizonAo", "RadiusSourcePx"),
        ("HorizonAo", "Bias"),
        ("Curvature", "EdgeScale"),
        ("Curvature", "CavityScale"),
        ("Roughness", "Base"),
        ("Metallic", "Radius"),
        ("Metallic", "Threshold"),
        ("Metallic", "Feather"),
        ("Levels", "Black"),
        ("Levels", "White"),
        ("Levels", "Gamma"),
        ("Blur", "SigmaSourcePx"),
    ] {
        let node = node_id(&snapshot(&session).graph, kind);
        let parameter: ParameterId = serde_json::from_value(json!(id)).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            rejects(
                &mut session,
                CommandRequest::SetParameter {
                    address: ParameterAddress { node, parameter },
                    value: ParameterValue::Float(value),
                    gesture: None,
                },
            );
        }
    }
}

#[test]
fn nonfinite_vector_curve_color_and_nested_seamless_dtos_fail_validation() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for params in [
            MaterialNodeParams::HeightFromAlbedo {
                band_weights: vec![value],
                coarse_weight: 1.0,
                remove_coarse: false,
                boundary: Boundary::Repeat,
            },
            MaterialNodeParams::NormalFromHeight {
                strength: 1.0,
                filter: NormalFilter::Sobel,
                scale_weights: vec![value],
                boundary: Boundary::Repeat,
                convention: NormalConvention::OpenGl,
            },
            MaterialNodeParams::Roughness {
                curve: vec![[value, 0.0], [1.0, 1.0]],
                base: 0.5,
                invert: false,
            },
            MaterialNodeParams::Roughness {
                curve: vec![[0.0, value], [1.0, 1.0]],
                base: 0.5,
                invert: false,
            },
            MaterialNodeParams::Metallic {
                picked_color: [0.0, value, 1.0],
                radius: 0.25,
                threshold: 0.5,
                feather: 0.0,
                mask_mode: MaskMode::Replace,
            },
            MaterialNodeParams::Seamless {
                method: SeamlessMethod::Histogram {
                    base: SeamlessBase::Splat {
                        seed: 1,
                        grid: 4,
                        overlap: value,
                    },
                    strength: 1.0,
                },
                boundary: Boundary::Repeat,
            },
            MaterialNodeParams::Seamless {
                method: SeamlessMethod::Histogram {
                    base: SeamlessBase::OffsetBlend { overlap: 0.25 },
                    strength: value,
                },
                boundary: Boundary::Repeat,
            },
        ] {
            assert!(params.validate().is_err());
        }
    }
}

#[test]
fn nonfinite_owned_vector_curve_and_color_commands_are_rejected() {
    use serde::de::value::SeqDeserializer;
    let mut session = loaded();
    for (kind, id, value) in [
        (
            "HeightFromAlbedo",
            "BandWeights",
            json!({"FloatVector": [0.5, 1.0]}),
        ),
        (
            "NormalFromHeight",
            "ScaleWeights",
            json!({"FloatVector": [0.5, 1.0]}),
        ),
        (
            "Roughness",
            "Curve",
            json!({"Curve": [[0.0, 0.25], [1.0, 0.75]]}),
        ),
        (
            "Metallic",
            "PickedColor",
            json!({"Color": [0.25, 0.5, 0.75]}),
        ),
    ] {
        let node = node_id(&snapshot(&session).graph, kind);
        set(&mut session, node, id, value, None);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut cases = vec![
            (
                "HeightFromAlbedo",
                "BandWeights",
                typed_payload(
                    "FloatVector",
                    SeqDeserializer::new([value, 1.0].into_iter()),
                ),
            ),
            (
                "NormalFromHeight",
                "ScaleWeights",
                typed_payload(
                    "FloatVector",
                    SeqDeserializer::new([1.0, value].into_iter()),
                ),
            ),
            (
                "Roughness",
                "Curve",
                typed_payload(
                    "Curve",
                    SeqDeserializer::new(
                        [FloatRow([value, 0.0]), FloatRow([1.0, 1.0])].into_iter(),
                    ),
                ),
            ),
            (
                "Roughness",
                "Curve",
                typed_payload(
                    "Curve",
                    SeqDeserializer::new(
                        [FloatRow([0.0, value]), FloatRow([1.0, 1.0])].into_iter(),
                    ),
                ),
            ),
        ];
        for channel in 0..3 {
            let mut rgb = [0.5; 3];
            rgb[channel] = value;
            cases.push((
                "Metallic",
                "PickedColor",
                typed_payload("Color", SeqDeserializer::new(rgb.into_iter())),
            ));
        }
        for (kind, id, value) in cases {
            let before = observable(&session);
            if let Ok(value) = value {
                let node = node_id(&snapshot(&session).graph, kind);
                rejects(
                    &mut session,
                    CommandRequest::SetParameter {
                        address: address(node, id),
                        value,
                        gesture: None,
                    },
                );
            }
            assert_eq!(observable(&session), before);
        }
    }
}

#[test]
fn signed_zero_is_a_noop_in_owned_vectors_and_curves() {
    let mut session = loaded();
    let node = node_id(&snapshot(&session).graph, "HeightFromAlbedo");
    set(
        &mut session,
        node,
        "BandWeights",
        json!({"FloatVector": [0.0, 1.0]}),
        None,
    );
    let before = observable(&session);
    assert!(
        set(
            &mut session,
            node,
            "BandWeights",
            json!({"FloatVector": [-0.0, 1.0]}),
            None
        )
        .snapshot
        .is_none()
    );
    assert_eq!(observable(&session), before);
    let node = node_id(&snapshot(&session).graph, "Roughness");
    let before = observable(&session);
    assert!(
        set(
            &mut session,
            node,
            "Curve",
            json!({"Curve": [[-0.0, -0.0], [1.0, 1.0]]}),
            None
        )
        .snapshot
        .is_none()
    );
    assert_eq!(observable(&session), before);
}

#[test]
fn deterministic_seamless_commands_include_seed_in_hash_and_restore_it_with_undo() {
    let mut a = loaded();
    let mut b = loaded();
    let node = node_id(&snapshot(&a).graph, "Seamless");
    let value = json!({"SeamlessMethod": {"Splat": {"seed": 42, "grid": 4, "overlap": 0.25}}});
    set(&mut a, node, "SeamlessMethod", value.clone(), None);
    set(&mut b, node, "SeamlessMethod", value, None);
    let seed42 = keys(&snapshot(&a).graph);
    assert_eq!(seed42, keys(&snapshot(&b).graph));
    set(
        &mut b,
        node,
        "SeamlessMethod",
        json!({"SeamlessMethod": {"Splat": {"seed": 43, "grid": 4, "overlap": 0.25}}}),
        None,
    );
    assert_ne!(seed42[&node], keys(&snapshot(&b).graph)[&node]);
    b.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(seed42, keys(&snapshot(&b).graph));
}

#[test]
fn typed_undo_and_cancel_overflow_preserve_history_and_active_gesture() {
    let graph = fixture();
    let node = node_id(&graph, "Roughness");
    for gesture in [None, Some(GestureId(51))] {
        let mut session = Session::from_snapshot(EvaluationSnapshot {
            generation: u64::MAX - 1,
            graph: graph.clone(),
        });
        if let Some(id) = gesture {
            begin(&mut session, node, "Invert", id);
        }
        set(&mut session, node, "Invert", json!({"Bool": true}), gesture);
        assert_eq!(snapshot(&session).generation, u64::MAX);
        let request = gesture.map_or(CommandRequest::Undo, |gesture| {
            CommandRequest::CancelGesture { gesture }
        });
        let before = observable(&session);
        assert_eq!(
            session.dispatch(request),
            Err(CommandError::GenerationOverflow)
        );
        assert_eq!(observable(&session), before);
    }
}

fn begin(session: &mut Session, node: NodeId, id: &str, gesture: GestureId) {
    let outcome = session
        .dispatch(CommandRequest::BeginGesture {
            gesture,
            address: address(node, id),
        })
        .unwrap();
    assert!(outcome.snapshot.is_none());
}

#[test]
fn vector_gesture_coalesces_many_previews_into_one_owned_diff() {
    let mut session = loaded();
    let before = snapshot(&session);
    let node = node_id(&before.graph, "Roughness");
    let gesture = GestureId(44);
    begin(&mut session, node, "Curve", gesture);
    for index in 1..=100 {
        let y = index as f32 / 200.0;
        let outcome = set(
            &mut session,
            node,
            "Curve",
            json!({"Curve": [[0.0, y], [1.0, 1.0]]}),
            Some(gesture),
        );
        assert_eq!(
            outcome.snapshot.unwrap().generation,
            before.generation + index
        );
        assert_eq!(session.history().undo_entries, 0);
    }
    let after = snapshot(&session);
    let outcome = session
        .dispatch(CommandRequest::CommitGesture { gesture })
        .unwrap();
    assert!(outcome.snapshot.is_none());
    assert_eq!(session.history().undo_entries, 1);
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(snapshot(&session).graph, before.graph);
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(snapshot(&session).graph, after.graph);
}

#[test]
fn net_zero_and_cancelled_typed_gestures_preserve_redo() {
    for cancel in [false, true] {
        let mut session = loaded();
        let node = node_id(&snapshot(&session).graph, "Roughness");
        set(&mut session, node, "Invert", json!({"Bool": true}), None);
        session.dispatch(CommandRequest::Undo).unwrap();
        let baseline = snapshot(&session);
        let gesture = GestureId(45);
        begin(&mut session, node, "Curve", gesture);
        set(
            &mut session,
            node,
            "Curve",
            json!({"Curve": [[0.0, 0.25], [1.0, 0.75]]}),
            Some(gesture),
        );
        if !cancel {
            set(
                &mut session,
                node,
                "Curve",
                json!({"Curve": [[0.0, 0.0], [1.0, 1.0]]}),
                Some(gesture),
            );
        }
        let outcome = session
            .dispatch(if cancel {
                CommandRequest::CancelGesture { gesture }
            } else {
                CommandRequest::CommitGesture { gesture }
            })
            .unwrap();
        assert_eq!(snapshot(&session).graph, baseline.graph);
        assert_eq!(snapshot(&session).generation, baseline.generation + 2);
        assert_eq!(outcome.snapshot.is_some(), cancel);
        assert_eq!(session.history().undo_entries, 0);
        assert_eq!(session.history().redo_entries, 1);
        session.dispatch(CommandRequest::Redo).unwrap();
        assert_eq!(field(&snapshot(&session).graph, node, "invert"), true);
    }
}

#[test]
fn stale_and_invalid_requests_preserve_active_typed_gesture_and_redo() {
    let mut session = loaded();
    let node = node_id(&snapshot(&session).graph, "Roughness");
    set(&mut session, node, "Invert", json!({"Bool": true}), None);
    session.dispatch(CommandRequest::Undo).unwrap();
    let gesture = GestureId(46);
    begin(&mut session, node, "Curve", gesture);
    rejects(
        &mut session,
        request(
            node,
            "Curve",
            json!({"Curve": [[0.5, 0.0], [0.5, 1.0]]}),
            Some(gesture),
        ),
    );
    rejects(
        &mut session,
        request(
            node,
            "Curve",
            json!({"Curve": [[0.0, 0.5], [1.0, 1.0]]}),
            Some(GestureId(999)),
        ),
    );
    rejects(
        &mut session,
        request(node, "Invert", json!({"Bool": true}), Some(gesture)),
    );
    rejects(
        &mut session,
        CommandRequest::CommitGesture {
            gesture: GestureId(999),
        },
    );
    rejects(
        &mut session,
        CommandRequest::BeginGesture {
            gesture: GestureId(47),
            address: address(node, "Curve"),
        },
    );
    session
        .dispatch(CommandRequest::CancelGesture { gesture })
        .unwrap();
    assert_eq!(session.history().redo_entries, 1);
}

#[test]
fn undo_implicitly_commits_a_typed_gesture_and_new_edit_clears_redo_on_commit_only() {
    let mut session = loaded();
    let before = snapshot(&session);
    let node = node_id(&before.graph, "Roughness");
    let gesture = GestureId(48);
    begin(&mut session, node, "Curve", gesture);
    set(
        &mut session,
        node,
        "Curve",
        json!({"Curve": [[0.0, 0.5], [1.0, 1.0]]}),
        Some(gesture),
    );
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(snapshot(&session).graph, before.graph);
    assert_eq!(session.history().active_gesture, None);
    assert_eq!(session.history().redo_entries, 1);
    let gesture = GestureId(49);
    begin(&mut session, node, "Invert", gesture);
    set(
        &mut session,
        node,
        "Invert",
        json!({"Bool": true}),
        Some(gesture),
    );
    assert_eq!(session.history().redo_entries, 1);
    session
        .dispatch(CommandRequest::CommitGesture { gesture })
        .unwrap();
    assert_eq!(session.history().redo_entries, 0);
    assert_eq!(session.history().undo_entries, 1);
}

#[test]
fn typed_diff_byte_accounting_includes_owned_vectors_without_gesture_growth() {
    let mut small = loaded();
    let mut large = loaded();
    let node = node_id(&snapshot(&small).graph, "HeightFromAlbedo");
    let empty = small.history().byte_count;
    set(
        &mut small,
        node,
        "BandWeights",
        json!({"FloatVector": [0.5]}),
        None,
    );
    set(
        &mut large,
        node,
        "BandWeights",
        json!({"FloatVector": vec![0.5; 16]}),
        None,
    );
    assert!(small.history().byte_count > empty);
    assert!(
        large.history().byte_count >= small.history().byte_count + 15 * size_of::<f32>(),
        "owned vector payloads must be counted"
    );
    let retained = large.history().byte_count;
    large.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(
        large.history().byte_count,
        retained,
        "moving a diff between stacks retains its bytes"
    );
    large.dispatch(CommandRequest::Redo).unwrap();
    let gesture = GestureId(50);
    begin(&mut large, node, "BandWeights", gesture);
    set(
        &mut large,
        node,
        "BandWeights",
        json!({"FloatVector": vec![0.25; 16]}),
        Some(gesture),
    );
    let active_bytes = large.history().byte_count;
    for index in 1..=100 {
        set(
            &mut large,
            node,
            "BandWeights",
            json!({"FloatVector": vec![index as f32; 16]}),
            Some(gesture),
        );
    }
    assert_eq!(
        large.history().byte_count,
        active_bytes,
        "gesture must retain only first and latest values"
    );
}

#[test]
fn generation_overflow_is_atomic_and_a_noop_at_max_is_valid() {
    let graph = fixture();
    let node = node_id(&graph, "NormalFromHeight");
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: u64::MAX,
        graph,
    });
    let before = observable(&session);
    assert!(
        set(&mut session, node, "Strength", json!({"Float": 1.0}), None)
            .snapshot
            .is_none()
    );
    assert_eq!(observable(&session), before);
    rejects(
        &mut session,
        request(node, "Strength", json!({"Float": 2.0}), None),
    );
}

#[test]
fn edit_invalidates_only_changed_node_and_descendants_and_undo_reuses_content() {
    let mut session = loaded();
    let before = snapshot(&session);
    let node = node_id(&before.graph, "NormalFromHeight");
    let initial = keys(&before.graph);
    set(
        &mut session,
        node,
        "Convention",
        json!({"Convention": "DirectX"}),
        None,
    );
    let after = snapshot(&session);
    assert_eq!(
        after
            .graph
            .output_normal_convention(after.graph.material_outputs().unwrap()[&MapKind::Normal])
            .unwrap(),
        Some(NormalConvention::DirectX)
    );
    let changed = keys(&after.graph);
    let curvature = node_id(&before.graph, "Curvature");
    for id in initial.keys() {
        assert_eq!(
            initial[id] != changed[id],
            *id == node || *id == curvature,
            "wrong invalidation for {id:?}"
        );
    }
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(keys(&snapshot(&session).graph), initial);
    assert!(snapshot(&session).generation > before.generation);
}

fn select_map(map: MapKind) -> CommandRequest {
    serde_json::from_value(json!({"id": "material.select_map", "payload": {"map": map}}))
        .expect("named material map selection must be routed")
}

#[test]
fn every_named_map_including_curvature_suboutputs_selects_without_content_or_history_change() {
    let mut session = loaded();
    let before = snapshot(&session);
    let outputs = before.graph.material_outputs().unwrap().clone();
    let initial_keys = keys(&before.graph);
    for (map, output) in outputs {
        session.dispatch(select_map(map)).unwrap();
        let selected = snapshot(&session);
        assert_eq!(selected.graph.selected_output_ref(), output);
        assert_eq!(selected.generation, before.generation);
        assert_eq!(session.history(), HistoryState::default());
        assert_eq!(keys(&selected.graph), initial_keys);
        assert_acyclic(&selected.graph);
    }
}

#[test]
fn missing_material_map_selection_is_atomic_and_legacy_node_selection_remains_view_only() {
    let graph = material_from_source(source(InputInterpretation::LinearHeight)).unwrap();
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 5,
        graph,
    });
    rejects(&mut session, select_map(MapKind::Albedo));
    let before = snapshot(&session);
    let height = before.graph.material_outputs().unwrap()[&MapKind::Height];
    session
        .dispatch(CommandRequest::SelectOutput {
            output: height.node,
        })
        .unwrap();
    assert_eq!(snapshot(&session).generation, 5);
    assert_eq!(snapshot(&session).graph.selected_output_ref(), height);
    assert_eq!(session.history(), HistoryState::default());
}

fn assert_acyclic(graph: &Graph) {
    validate(graph).unwrap();
    let mut visited = Vec::new();
    for node in graph.topological_material_nodes() {
        for edge in &node.inputs {
            assert!(
                visited.contains(&edge.source.node),
                "consumer precedes parent or contains feedback"
            );
        }
        visited.push(node.id);
    }
    assert_eq!(visited.len(), graph.material_nodes().len());
}

#[test]
fn all_procedural_factory_branches_are_deterministic_acyclic_and_separate_imported_normals() {
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
        InputInterpretation::LinearNormal,
    ] {
        let graph = material_from_source(source(interpretation)).unwrap();
        assert_acyclic(&graph);
        assert_eq!(graph, material_from_source(source(interpretation)).unwrap());
        assert_eq!(
            keys(&graph),
            keys(&material_from_source(source(interpretation)).unwrap())
        );
        if interpretation == InputInterpretation::LinearNormal {
            assert!(
                !graph
                    .material_nodes()
                    .iter()
                    .any(|node| matches!(node.params, MaterialNodeParams::NormalFromHeight { .. }))
            );
            let height = graph
                .material_node(graph.material_outputs().unwrap()[&MapKind::Height].node)
                .unwrap();
            assert!(matches!(
                height.params,
                MaterialNodeParams::HeightFromNormal { .. }
            ));
            assert_eq!(
                height.inputs[0].source,
                graph.material_outputs().unwrap()[&MapKind::Normal]
            );
        }
    }
}

#[test]
fn stable_registry_describes_all_typed_controls_and_map_selection_once() {
    let descriptors = serde_json::to_value(command_descriptors()).unwrap();
    let commands = descriptors.as_array().unwrap();
    let parameter_commands: Vec<_> = commands
        .iter()
        .filter(|d| d["id"] == "node.set_parameter")
        .collect();
    assert_eq!(parameter_commands.len(), 1);
    let parameters = parameter_commands[0]["payload"]["Parameter"]["parameters"]
        .as_array()
        .unwrap();
    for id in [
        "Black",
        "White",
        "Gamma",
        "SigmaSourcePx",
        "Strength",
        "BandWeights",
        "CoarseWeight",
        "RemoveCoarse",
        "Boundary",
        "Filter",
        "ScaleWeights",
        "Convention",
        "VCycles",
        "PreSmooth",
        "PostSmooth",
        "CoarsestIterations",
        "RelativeTolerance",
        "AbsoluteTolerance",
        "Directions",
        "Steps",
        "RadiusSourcePx",
        "Bias",
        "EdgeScale",
        "CavityScale",
        "Curve",
        "Base",
        "Invert",
        "PickedColor",
        "Radius",
        "Threshold",
        "Feather",
        "MaskMode",
        "SeamlessMethod",
    ] {
        assert_eq!(
            parameters.iter().filter(|d| d["id"] == id).count(),
            1,
            "missing or duplicate discovery entry for {id}"
        );
    }
    for tag in [
        "Float",
        "UInt",
        "Bool",
        "FloatVector",
        "Curve",
        "Color",
        "Boundary",
        "Filter",
        "Convention",
        "MaskMode",
        "SeamlessMethod",
    ] {
        assert!(
            parameters.iter().any(|d| d["value_schema"]
                .as_object()
                .is_some_and(|s| s.contains_key(tag))),
            "discovery must describe {tag} payloads"
        );
    }
    assert_eq!(
        commands
            .iter()
            .filter(|d| d["id"] == "material.select_map")
            .count(),
        1
    );
    let availability = loaded().availability();
    assert_eq!(availability.len(), commands.len());
    for descriptor in command_descriptors() {
        assert_eq!(
            availability
                .iter()
                .filter(|a| a.id == descriptor.id)
                .count(),
            1
        );
    }
}

#[test]
fn malformed_or_unknown_typed_payloads_fail_closed_before_dispatch() {
    // Prove a valid new typed payload is routable before checking rejection.
    let graph = fixture();
    let node = node_id(&graph, "Roughness");
    let valid = serde_json::to_value(request(node, "Invert", json!({"Bool": true}), None)).unwrap();
    for replacement in [
        json!({"Bool": 1}),
        json!({"String": "true"}),
        json!({"Color": [0.0, 1.0]}),
        json!({"UInt": -1}),
        json!({"UInt": 1.5}),
        json!({"Boundary": "Mirror"}),
        json!({"Convention": "Automatic"}),
        json!({"Filter": "Unknown"}),
        json!({"MaskMode": "Multiply"}),
    ] {
        let mut invalid = valid.clone();
        invalid["payload"]["value"] = replacement;
        assert!(serde_json::from_value::<CommandRequest>(invalid).is_err());
    }
    let mut unknown = valid;
    unknown["payload"]["address"]["parameter"] = json!("FutureParameter");
    assert!(serde_json::from_value::<CommandRequest>(unknown).is_err());
}

#[test]
fn descriptor_ranges_lengths_and_enum_choices_are_discoverable_without_ui_rules() {
    let registry = serde_json::to_value(command_descriptors()).unwrap();
    let command = registry
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["id"] == "node.set_parameter")
        .unwrap();
    let descriptors = command["payload"]["Parameter"]["parameters"]
        .as_array()
        .unwrap();
    let schema = |id: &str, tag: &str| {
        descriptors
            .iter()
            .find(|d| d["id"] == id)
            .unwrap_or_else(|| panic!("missing descriptor {id}"))["value_schema"][tag]
            .clone()
    };
    for (id, tag, min, max) in [
        ("Strength", "Float", 0.0, 10.0),
        ("Gamma", "Float", 0.1, 10.0),
        ("Black", "Float", 0.0, 1.0),
        ("White", "Float", 0.0, 1.0),
        ("SigmaSourcePx", "Float", 0.0, 5.0),
        ("VCycles", "UInt", 1.0, 16.0),
        ("Directions", "UInt", 4.0, 32.0),
        ("Steps", "UInt", 4.0, 32.0),
        ("RadiusSourcePx", "Float", 1.0, 512.0),
        ("Base", "Float", 0.0, 1.0),
        ("Radius", "Float", 0.0, f64::from(3.0_f32.sqrt())),
        ("Threshold", "Float", 0.0, 1.0),
    ] {
        let s = schema(id, tag);
        assert_parameter_eq(&s["min"], &json!(min));
        assert_parameter_eq(&s["max"], &json!(max));
        assert!(s.get("default").is_some(), "default required for {id}");
    }
    for (id, tag, min, max) in [
        ("BandWeights", "FloatVector", 1, 16),
        ("ScaleWeights", "FloatVector", 1, 16),
        ("Curve", "Curve", 2, 16),
    ] {
        let s = schema(id, tag);
        assert_eq!(s["min_len"], min);
        assert_eq!(s["max_len"], max);
    }
    for (id, tag, expected) in [
        ("Boundary", "Boundary", vec!["Clamp", "Repeat"]),
        ("Filter", "Filter", vec!["Central", "Sobel", "Scharr"]),
        ("Convention", "Convention", vec!["OpenGl", "DirectX"]),
        ("MaskMode", "MaskMode", vec!["Replace", "Add", "Subtract"]),
    ] {
        let s = schema(id, tag);
        let mut choices: Vec<_> = s["choices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut expected = expected;
        choices.sort_unstable();
        expected.sort_unstable();
        assert_eq!(choices, expected);
    }
}

#[test]
fn unchanged_typed_edits_preserve_an_existing_redo_branch() {
    let mut session = loaded();
    let node = node_id(&snapshot(&session).graph, "Roughness");
    set(&mut session, node, "Invert", json!({"Bool": true}), None);
    session.dispatch(CommandRequest::Undo).unwrap();
    let before = observable(&session);
    for (id, value) in [
        ("Invert", json!({"Bool": false})),
        ("Curve", json!({"Curve": [[0.0, 0.0], [1.0, 1.0]]})),
    ] {
        assert!(set(&mut session, node, id, value, None).snapshot.is_none());
        assert_eq!(observable(&session), before);
    }
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(field(&snapshot(&session).graph, node, "invert"), true);
}

#[test]
fn descriptor_defaults_agree_with_the_procedural_dto_defaults() {
    let graph = fixture();
    let registry = serde_json::to_value(command_descriptors()).unwrap();
    let command = registry
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["id"] == "node.set_parameter")
        .unwrap();
    let descriptors = command["payload"]["Parameter"]["parameters"]
        .as_array()
        .unwrap();
    for (kind, id, name) in [
        ("HeightFromAlbedo", "BandWeights", "band_weights"),
        ("HeightFromAlbedo", "CoarseWeight", "coarse_weight"),
        ("HeightFromAlbedo", "RemoveCoarse", "remove_coarse"),
        ("HeightFromAlbedo", "Boundary", "boundary"),
        ("NormalFromHeight", "Strength", "strength"),
        ("NormalFromHeight", "Filter", "filter"),
        ("NormalFromHeight", "ScaleWeights", "scale_weights"),
        ("NormalFromHeight", "Convention", "convention"),
        ("HeightFromNormal", "VCycles", "v_cycles"),
        ("HeightFromNormal", "PreSmooth", "pre_smooth"),
        ("HeightFromNormal", "PostSmooth", "post_smooth"),
        (
            "HeightFromNormal",
            "CoarsestIterations",
            "coarsest_iterations",
        ),
        (
            "HeightFromNormal",
            "RelativeTolerance",
            "relative_tolerance",
        ),
        (
            "HeightFromNormal",
            "AbsoluteTolerance",
            "absolute_tolerance",
        ),
        ("HorizonAo", "Directions", "directions"),
        ("HorizonAo", "Steps", "steps"),
        ("HorizonAo", "RadiusSourcePx", "radius_source_px"),
        ("HorizonAo", "Bias", "bias"),
        ("Curvature", "EdgeScale", "edge_scale"),
        ("Curvature", "CavityScale", "cavity_scale"),
        ("Roughness", "Curve", "curve"),
        ("Roughness", "Base", "base"),
        ("Roughness", "Invert", "invert"),
        ("Metallic", "PickedColor", "picked_color"),
        ("Metallic", "Radius", "radius"),
        ("Metallic", "Threshold", "threshold"),
        ("Metallic", "Feather", "feather"),
        ("Metallic", "MaskMode", "mask_mode"),
        ("Levels", "Black", "black"),
        ("Levels", "White", "white"),
        ("Levels", "Gamma", "gamma"),
        ("Blur", "SigmaSourcePx", "sigma_source_px"),
        ("Seamless", "SeamlessMethod", "method"),
    ] {
        let descriptor = descriptors.iter().find(|d| d["id"] == id).unwrap();
        let schema = descriptor["value_schema"].as_object().unwrap();
        assert_eq!(schema.len(), 1, "one typed schema per parameter");
        let default = &schema.values().next().unwrap()["default"];
        assert_parameter_eq(default, &field(&graph, node_id(&graph, kind), name));
    }
}
