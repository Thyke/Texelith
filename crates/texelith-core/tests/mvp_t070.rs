//! Material template and preset contracts; all mutation goes through Session.
use serde_json::{Value, json};
use std::collections::HashSet;
use texelith_core::*;

fn session() -> Session {
    let extent = Extent {
        width: 17,
        height: 13,
    };
    let source = SourceDescriptor {
        content_hash: [70; 32],
        extent,
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let mut dto = MaterialGraphDto::from(material_from_source(source).unwrap());
    dto.graph.push(MaterialNode {
        id: NodeId(104),
        inputs: vec![],
        params: MaterialNodeParams::PaintMask {
            boundary: texelith_core::Boundary::Repeat,
            extent,
            stroke_digest: [0; 32],
            strokes: vec![],
        },
    });
    dto.outputs.insert(
        MapKind::Mask,
        OutputRef {
            node: NodeId(104),
            port: OutputPort::Mask,
        },
    );
    Session::from_snapshot(EvaluationSnapshot {
        generation: 40,
        graph: Graph::try_from(dto).unwrap(),
    })
}
fn request(id: &str, payload: Value) -> CommandRequest {
    serde_json::from_value(json!({"id": id, "payload": payload}))
        .unwrap_or_else(|e| panic!("{id}: {e}"))
}
fn dispatch(s: &mut Session, id: &str, payload: Value) -> CommandOutcome {
    s.dispatch(request(id, payload)).unwrap()
}
fn view(s: &Session) -> Value {
    serde_json::to_value(s.view_state()).unwrap()
}

#[test]
fn registry_has_unique_ids_and_every_t070_route() {
    let descriptors = command_descriptors();
    let ids: HashSet<_> = descriptors.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids.len(), descriptors.len());
    let availability = session().availability();
    assert_eq!(availability.len(), descriptors.len());
    for d in descriptors {
        assert_eq!(availability.iter().filter(|a| a.id == d.id).count(), 1);
        assert!(!d.label.is_empty());
    }
    for id in [
        "material.set_visible",
        "material.set_locked",
        "view.set_mode",
        "view.set_mesh",
        "view.set_light",
        "view.set_tiles",
        "view.set_pom",
        "view.set_camera",
        "view.capture_a",
        "view.capture_b",
        "view.set_split",
    ] {
        assert!(ids.contains(id), "missing descriptor {id}");
    }
}

#[test]
fn camera_and_map_selection_only_invalidate_preview() {
    let mut s = session();
    let before = s.snapshot().unwrap();
    let request_id = s.request_id();
    let revision = s.document_revision();
    let dirty = s.is_dirty();
    let outcome = dispatch(
        &mut s,
        "view.set_camera",
        json!({"camera": {"yaw": 0.4, "pitch": 0.2, "distance": 3.0, "target": [0.0, 0.0, 0.0]}}),
    );
    assert!(outcome.snapshot.is_none());
    assert_eq!(s.snapshot().unwrap(), before);
    assert!(s.request_id() > request_id);
    assert_eq!(s.document_revision(), revision);
    assert_eq!(s.is_dirty(), dirty);
    let request_id = s.request_id();
    dispatch(&mut s, "material.select_map", json!({"map": "Normal"}));
    assert!(s.request_id() > request_id);
    assert_eq!(s.snapshot().unwrap().generation, before.generation);
    assert_eq!(s.document_revision(), revision);
}

#[test]
fn visibility_and_lock_dirty_document_without_changing_graph_and_undo() {
    let mut s = session();
    let graph = s.snapshot().unwrap();
    let revision = s.document_revision();
    dispatch(
        &mut s,
        "material.set_visible",
        json!({"map":"Height", "visible":false}),
    );
    assert_eq!(s.snapshot().unwrap(), graph);
    assert!(s.document_revision() > revision);
    assert!(s.is_dirty());
    let revision = s.document_revision();
    dispatch(
        &mut s,
        "material.set_locked",
        json!({"map":"Height", "locked":true}),
    );
    assert_eq!(s.snapshot().unwrap(), graph);
    assert!(s.document_revision() > revision);
    let locked = view(&s);
    s.dispatch(CommandRequest::Undo).unwrap();
    assert_ne!(view(&s), locked);
    s.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(view(&s), locked);
}

#[test]
fn undo_map_change_with_untouched_gesture_invalidates_document_and_preview() {
    let mut s = session();
    let node = s
        .snapshot()
        .unwrap()
        .graph
        .material_nodes()
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::HeightFromAlbedo { .. }))
        .unwrap()
        .id;
    dispatch(
        &mut s,
        "material.set_visible",
        json!({"map":"Height", "visible":false}),
    );
    assert!(s.mark_saved(s.document_revision()));
    let revision = s.document_revision();
    let request_id = s.request_id();
    let graph = s.snapshot().unwrap();
    s.dispatch(CommandRequest::BeginGesture {
        gesture: GestureId(70),
        address: serde_json::from_value(json!({"node":node,"parameter":"CoarseWeight"})).unwrap(),
    })
    .unwrap();
    s.dispatch(CommandRequest::Undo).unwrap();
    assert!(s.view_state().maps[&MapKind::Height].visible);
    assert_eq!(s.snapshot().unwrap(), graph);
    assert!(s.document_revision() > revision);
    assert!(s.request_id() > request_id);
    assert!(s.is_dirty());
    assert_eq!(s.history().active_gesture, None);
}

#[test]
fn mesh_data_transfers_vertex_attributes_and_indices() {
    let mesh = MeshData {
        vertices: vec![Vertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.25, 0.75],
            tangent: [1.0, 0.0, 0.0, -1.0],
        }],
        indices: vec![0, 0, 0],
    };
    let encoded = serde_json::to_value(&mesh).unwrap();
    assert_eq!(
        encoded["vertices"][0]["tangent"],
        json!([1.0, 0.0, 0.0, -1.0])
    );
    assert_eq!(encoded["indices"], json!([0, 0, 0]));
    assert_eq!(serde_json::from_value::<MeshData>(encoded).unwrap(), mesh);
}

#[test]
fn lock_rejects_direct_parameter_and_stroke_edits_atomically() {
    let mut s = session();
    let height = s
        .snapshot()
        .unwrap()
        .graph
        .material_nodes()
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::HeightFromAlbedo { .. }))
        .unwrap()
        .id;
    dispatch(
        &mut s,
        "material.set_locked",
        json!({"map":"Height", "locked":true}),
    );
    let before = (
        s.snapshot(),
        s.history(),
        view(&s),
        s.document_revision(),
        s.request_id(),
    );
    let edit = request(
        "node.set_parameter",
        json!({"address":{"node":height,"parameter":"CoarseWeight"},"value":{"Float":0.5},"gesture":null}),
    );
    assert!(s.dispatch(edit).is_err());
    assert_eq!(
        (
            s.snapshot(),
            s.history(),
            view(&s),
            s.document_revision(),
            s.request_id()
        ),
        before
    );
    dispatch(
        &mut s,
        "material.set_locked",
        json!({"map":"Mask", "locked":true}),
    );
    let before = (
        s.snapshot(),
        s.history(),
        view(&s),
        s.document_revision(),
        s.request_id(),
    );
    let stroke = request(
        "mask.begin_stroke",
        json!({"node":104,"stroke_id":1,"brush":{"radius":0.2,"hardness":0.5,"opacity":0.75,"mode":"Add"},"sample":{"uv":[0.1,0.5],"pressure":1.0}}),
    );
    assert!(s.dispatch(stroke).is_err());
    assert_eq!(
        (
            s.snapshot(),
            s.history(),
            view(&s),
            s.document_revision(),
            s.request_id()
        ),
        before
    );
}

#[test]
fn captures_keep_coherent_independent_identities_and_split_is_clamped() {
    let mut s = session();
    dispatch(&mut s, "view.capture_a", json!({}));
    let a = view(&s)["capture_a"].clone();
    assert!(!a.is_null());
    let node = s
        .snapshot()
        .unwrap()
        .graph
        .material_nodes()
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::HeightFromAlbedo { .. }))
        .unwrap()
        .id;
    dispatch(
        &mut s,
        "node.set_parameter",
        json!({"address":{"node":node,"parameter":"CoarseWeight"},"value":{"Float":0.5},"gesture":null}),
    );
    dispatch(&mut s, "view.capture_b", json!({}));
    let state = view(&s);
    assert_eq!(state["capture_a"], a);
    assert!(!state["capture_b"].is_null());
    assert_ne!(state["capture_a"], state["capture_b"]);
    for (input, expected) in [(-2.0, 0.0), (0.35, 0.35), (4.0, 1.0)] {
        dispatch(&mut s, "view.set_split", json!({"fraction":input}));
        assert!((view(&s)["split_fraction"].as_f64().unwrap() - expected).abs() < 1e-6);
        assert_eq!(view(&s)["capture_a"], a);
        assert_eq!(view(&s)["capture_b"], state["capture_b"]);
    }
}
