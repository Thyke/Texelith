//! Project contract. Proposed integration names for APIs not yet implemented:
//! `adapt_diagnostic_graph`, `migrate_project`, `ProjectOpenToken`, and
//! `Session::{begin_project_open, complete_project_open, fail_project_open}`.
//! Completion takes a validated project whose sources have already been checked.
use serde_json::{Value, json};
use texelith_core::*;

fn source(hash: u8) -> SourceDescriptor {
    SourceDescriptor {
        content_hash: [hash; 32],
        extent: Extent {
            width: 16,
            height: 12,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

fn project_value() -> Value {
    let graph = material_from_source(source(84)).unwrap();
    let preset = BuiltinExportPresets.find("unreal-v1", 1).unwrap();
    let export = preset.bind(&graph, &[]).unwrap();
    let mut view = serde_json::to_value(ViewState::default()).unwrap();
    view.as_object_mut().unwrap().remove("capture_a");
    view.as_object_mut().unwrap().remove("capture_b");
    view["camera"]["yaw"] = json!(0.75);
    view["tiles"]["seed"] = json!(123456);
    json!({
        "format": "texelith.project", "schema_version": 1,
        "graph": MaterialGraphDto::from(graph),
        "assets": [{"id": 1, "relative_path": "../textures/source.png",
                    "decoded_hash": vec![84u8; 32], "interpretation": "SrgbPhoto"}],
        "export": export, "view": view
    })
}

fn project(value: Value) -> ProjectFile {
    serde_json::from_value(value).expect("valid v1 DTO")
}

#[test]
fn project_roundtrip_preserves_graph_export_view_and_excludes_runtime_state() {
    let input = project_value();
    let dto = project(input.clone());
    validate_project(dto.clone()).unwrap();
    let encoded = serde_json::to_value(&dto).unwrap();
    assert_eq!(encoded, input);
    assert_eq!(project(encoded.clone()), dto);
    let text = encoded.to_string();
    for forbidden in [
        "generation",
        "history",
        "pending_jobs",
        "capture_a",
        "capture_b",
    ] {
        assert!(
            !text.contains(forbidden),
            "runtime field {forbidden} persisted"
        );
    }
    assert_eq!(encoded["export"]["profile"]["id"], "unreal-v1");
    assert_eq!(encoded["export"]["profile"]["revision"], 1);
    assert_eq!(encoded["view"]["tiles"]["seed"], 123456);
}

#[test]
fn rejects_future_wrong_and_unrecognized_versions_and_fields() {
    for (field, value) in [
        ("format", json!("other")),
        ("schema_version", json!(0)),
        ("schema_version", json!(2)),
    ] {
        let mut v = project_value();
        v[field] = value;
        assert!(
            validate_project(project(v)).is_err(),
            "accepted invalid {field}"
        );
    }
    let mut v = project_value();
    v["graph"]["schema_version"] = json!(3);
    assert!(validate_project(project(v)).is_err());
    let mut v = project_value();
    v["unknown_semantic_setting"] = json!(true);
    assert!(serde_json::from_value::<ProjectFile>(v).is_err());
}

#[test]
fn portable_relative_paths_allow_external_links_but_never_implicit_network_or_absolute() {
    for path in ["source.png", "textures/brick.png", "../shared/source.png"] {
        let mut v = project_value();
        v["assets"][0]["relative_path"] = json!(path);
        assert!(validate_project(project(v)).is_ok(), "rejected {path:?}");
    }
    for path in [
        "",
        "/tmp/source.png",
        "C:/source.png",
        "C:\\source.png",
        "\\\\server\\share\\source.png",
        "//server/share/source.png",
        "https://example.invalid/image.png",
        "file:///tmp/image.png",
        "a\0b",
    ] {
        let mut v = project_value();
        v["assets"][0]["relative_path"] = json!(path);
        assert!(validate_project(project(v)).is_err(), "accepted {path:?}");
    }
}

#[test]
fn oversized_assets_nodes_and_total_brush_points_are_rejected() {
    let mut v = project_value();
    let asset = v["assets"][0].clone();
    v["assets"] = Value::Array(
        (0..65)
            .map(|id| {
                let mut a = asset.clone();
                a["id"] = json!(id);
                a
            })
            .collect(),
    );
    assert!(validate_project(project(v)).is_err());

    let mut v = project_value();
    let node = v["graph"]["graph"][0].clone();
    v["graph"]["graph"] = Value::Array(
        (0..257)
            .map(|id| {
                let mut n = node.clone();
                n["id"] = json!(id);
                n
            })
            .collect(),
    );
    assert!(validate_project(project(v)).is_err());

    let mut dto = project(project_value());
    let point = BrushSample {
        uv: [0.5, 0.5],
        pressure: 1.0,
    };
    let stroke = |id, count| BrushStroke {
        id: StrokeId(id),
        brush: Brush {
            radius: 0.1,
            hardness: 0.5,
            opacity: 1.0,
            mode: MaskMode::Add,
        },
        points: vec![point; count],
    };
    for (id, count) in [(1001, 50_001), (1002, 50_000)] {
        let strokes = vec![stroke(id, count)];
        dto.graph.graph.push(MaterialNode {
            id: NodeId(id),
            inputs: vec![],
            params: MaterialNodeParams::PaintMask {
                boundary: texelith_core::Boundary::Repeat,
                extent: source(84).extent,
                stroke_digest: stroke_digest(&strokes).unwrap(),
                strokes,
            },
        });
    }
    assert!(
        validate_project(dto).is_err(),
        "limit is across the whole project"
    );
}

#[test]
fn vector_strokes_roundtrip_with_the_graph() {
    let mut dto = project(project_value());
    let strokes = vec![BrushStroke {
        id: StrokeId(9),
        brush: Brush {
            radius: 0.2,
            hardness: 0.4,
            opacity: 0.7,
            mode: MaskMode::Subtract,
        },
        points: vec![
            BrushSample {
                uv: [0.2, 0.3],
                pressure: 0.8,
            },
            BrushSample {
                uv: [0.21, 0.3],
                pressure: 1.0,
            },
        ],
    }];
    dto.graph.graph.push(MaterialNode {
        id: NodeId(1001),
        inputs: vec![],
        params: MaterialNodeParams::PaintMask {
            boundary: texelith_core::Boundary::Repeat,
            extent: source(84).extent,
            stroke_digest: stroke_digest(&strokes).unwrap(),
            strokes,
        },
    });
    validate_project(dto.clone()).unwrap();
    let roundtrip: ProjectFile =
        serde_json::from_str(&serde_json::to_string(&dto).unwrap()).unwrap();
    assert_eq!(roundtrip, dto);
}

#[test]
fn diagnostic_import_is_explicit_and_project_migration_has_no_invented_v0() {
    let legacy = Graph::new(
        vec![Node {
            id: NodeId(1),
            input: None,
            params: NodeParams::Source(source(84)),
        }],
        NodeId(1),
    )
    .unwrap();
    let diagnostic = legacy.legacy_dto().unwrap();
    let migrated = adapt_diagnostic_graph(diagnostic.clone()).unwrap();
    assert_eq!(migrated.schema_version, 2);
    assert_eq!(migrated, MaterialGraphDto::try_from(diagnostic).unwrap());
    let dto = project(project_value());
    assert_eq!(migrate_project(dto.clone()).unwrap(), dto);
    let mut old = dto;
    old.schema_version = 0;
    assert!(migrate_project(old).is_err());
}

#[test]
fn project_open_is_atomic_tokenized_and_generation_is_monotonic() {
    let graph = material_from_source(source(42)).unwrap();
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 40,
        graph,
    });
    let before = session.snapshot().unwrap();
    let view = session.view_state().clone();
    let history = session.history();
    let a = ProjectOpenToken(1);
    let b = ProjectOpenToken(2);
    session.begin_project_open(a).unwrap();
    session.begin_project_open(b).unwrap();
    assert!(
        session
            .complete_project_open(a, validate_project(project(project_value())).unwrap())
            .is_err()
    );
    assert_eq!(session.snapshot().unwrap(), before);
    assert_eq!(session.view_state(), &view);
    assert_eq!(session.history(), history);
    session.fail_project_open(b).unwrap();
    assert_eq!(session.snapshot().unwrap(), before);
    let c = ProjectOpenToken(3);
    session.begin_project_open(c).unwrap();
    session
        .complete_project_open(c, validate_project(project(project_value())).unwrap())
        .unwrap();
    let after = session.snapshot().unwrap();
    assert!(after.generation > before.generation);
    assert_ne!(after.graph, before.graph);
    assert_eq!(session.history().undo_entries, 0);
    assert_eq!(session.history().redo_entries, 0);
}

#[test]
fn nested_project_semantics_reject_unknown_fields() {
    for pointer in [
        "/view/camera",
        "/view/light",
        "/view/tiles",
        "/view/pom",
        "/view/maps/Height",
        "/export/size/Resize",
        "/graph/graph/0/params/ColorSource",
        "/graph/graph/0/params/ColorSource/extent",
        "/assets/0",
        "/export/profile",
        "/export/outputs/0",
        "/export/displacement",
    ] {
        let mut value = project_value();
        value["view"]["maps"]["Height"] = json!({"visible": true, "locked": false});
        value["export"]["size"] = json!({"Resize": {"width": 16, "height": 12}});
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("future_semantics".into(), json!(true));
        assert!(
            serde_json::from_value::<ProjectFile>(value).is_err(),
            "accepted {pointer}"
        );
    }
}

#[test]
fn every_graph_source_requires_a_matching_asset_hash_and_interpretation() {
    for mutation in ["empty", "hash", "interpretation", "second_source"] {
        let mut dto = project(project_value());
        match mutation {
            "empty" => dto.assets.clear(),
            "hash" => dto.assets[0].decoded_hash = [0; 32],
            "interpretation" => dto.assets[0].interpretation = InputInterpretation::LinearHeight,
            _ => dto.graph.graph.push(MaterialNode {
                id: NodeId(900),
                inputs: vec![],
                params: MaterialNodeParams::ColorSource(source(85)),
            }),
        }
        assert!(validate_project(dto).is_err(), "accepted {mutation}");
    }
    let mut dto = project(project_value());
    dto.graph.graph.push(MaterialNode {
        id: NodeId(900),
        inputs: vec![],
        params: MaterialNodeParams::ColorSource(source(85)),
    });
    dto.assets.push(AssetRef {
        id: 2,
        relative_path: "extra.png".into(),
        decoded_hash: [85; 32],
        interpretation: InputInterpretation::SrgbPhoto,
    });
    assert!(validate_project(dto).is_ok());
}

#[test]
fn windows_devices_are_rejected_in_every_reference_component() {
    for path in [
        "NUL.png",
        "dir/COM1",
        "aux.exr",
        "CON",
        "dir/LpT9.png",
        "CONIN$",
        "CLOCK$.png",
        "nul .png",
        "dir/COM\u{00b9}.png",
    ] {
        let mut dto = project(project_value());
        dto.assets[0].relative_path = path.into();
        assert!(validate_project(dto).is_err(), "asset {path}");
        let mut dto = project(project_value());
        dto.view.mesh = PreviewMesh::Custom(OpaqueLocator(path.into()));
        assert!(validate_project(dto).is_err(), "mesh {path}");
        let mut dto = project(project_value());
        dto.view.light.environment = Some(OpaqueLocator(path.into()));
        assert!(validate_project(dto).is_err(), "environment {path}");
    }
    for path in ["../shared/normal.png", "dir/COM10.png", "console.png"] {
        let mut dto = project(project_value());
        dto.assets[0].relative_path = path.into();
        assert!(validate_project(dto).is_ok(), "rejected {path}");
    }
}

#[test]
fn validated_documents_serialize_in_canonical_node_and_asset_order() {
    let mut dto = project(project_value());
    let mut asset = dto.assets[0].clone();
    asset.id = 99;
    asset.relative_path = "another.png".into();
    dto.assets.push(asset);
    let canonical = validate_project(dto.clone()).unwrap().into_document();
    let golden = serde_json::to_string_pretty(&canonical).unwrap();
    dto.graph.graph.reverse();
    dto.assets.reverse();
    let validated = validate_project(dto).unwrap();
    assert_eq!(
        serde_json::to_string_pretty(validated.document()).unwrap(),
        golden
    );
    assert_eq!(validated.into_document(), canonical);
    assert!(canonical.graph.graph.windows(2).all(|n| n[0].id < n[1].id));
    assert_eq!(
        canonical.assets.iter().map(|a| a.id).collect::<Vec<_>>(),
        vec![1, 99]
    );
    let reloaded = validate_project(serde_json::from_str(&golden).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_string_pretty(reloaded.document()).unwrap(),
        golden
    );
}

fn opened_session() -> Session {
    let mut session = Session::new();
    session.begin_project_open(ProjectOpenToken(1)).unwrap();
    session
        .complete_project_open(
            ProjectOpenToken(1),
            validate_project(project(project_value())).unwrap(),
        )
        .unwrap();
    assert!(!session.is_dirty());
    session
}

#[test]
fn persisted_view_commands_keep_document_clean_and_reject_stale_save_completions() {
    for (id, payload) in [
        ("view.set_mode", json!({"mode": "ThreeD"})),
        ("view.set_mesh", json!({"mesh": "Cube"})),
        (
            "view.set_light",
            json!({"light": {"direction": [1.0, 0.0, 0.0], "intensity": 2.0, "environment": null}}),
        ),
        (
            "view.set_tiles",
            json!({"tiles": {"count": 2, "seed": 7, "random_rotation": true}}),
        ),
        (
            "view.set_pom",
            json!({"pom": {"enabled": true, "scale": 0.1}}),
        ),
        (
            "view.set_camera",
            json!({"camera": {"yaw": 1.0, "pitch": 0.0, "distance": 3.0, "target": [0.0, 0.0, 0.0]}}),
        ),
        ("view.set_split", json!({"fraction": 0.7})),
        (
            "material.set_visible",
            json!({"map": "Height", "visible": false}),
        ),
        (
            "material.set_locked",
            json!({"map": "Height", "locked": true}),
        ),
    ] {
        let mut session = opened_session();
        let snapshot = session.snapshot();
        let revision = session.document_revision();
        let view_revision = session.view_revision();
        let request: CommandRequest =
            serde_json::from_value(json!({"id": id, "payload": payload})).unwrap();
        session.dispatch(request.clone()).unwrap();
        assert_eq!(session.snapshot(), snapshot, "{id} changed graph");
        assert_eq!(
            session.is_dirty(),
            id.starts_with("material."),
            "{id} document dirty state"
        );
        assert!(session.view_revision() > view_revision || session.document_revision() > revision);
        assert!(!session.mark_saved(revision));
        assert!(!session.mark_project_saved(revision, view_revision));
        assert!(session.mark_project_saved(session.document_revision(), session.view_revision()));
        let clean_view_revision = session.view_revision();
        session.dispatch(request).unwrap();
        assert_eq!(session.view_revision(), clean_view_revision, "no-op {id}");
        assert!(!session.is_dirty());
    }
    let mut session = opened_session();
    session.dispatch(CommandRequest::CaptureA {}).unwrap();
    session.dispatch(CommandRequest::CaptureB {}).unwrap();
    assert!(!session.is_dirty(), "runtime captures are not persisted");
}

#[test]
fn camera_change_stays_clean_and_is_saved_and_restored_on_reopen() {
    let mut session = opened_session();
    let snapshot = session.snapshot().unwrap();
    let revision = session.document_revision();
    let camera = Camera {
        yaw: 1.25,
        pitch: 0.2,
        distance: 4.0,
        target: [0.1, 0.2, 0.3],
    };
    session
        .dispatch(CommandRequest::SetCamera { camera })
        .unwrap();
    assert!(!session.is_dirty());
    assert_eq!(session.document_revision(), revision);
    assert_eq!(session.snapshot().unwrap(), snapshot);

    let dto = ProjectFile {
        format: "texelith.project".into(),
        schema_version: 1,
        graph: MaterialGraphDto::from(snapshot.graph),
        assets: session.project_assets().to_vec(),
        export: session.export_settings().unwrap().clone(),
        view: SavedView::from(session.view_state().clone()),
    };
    let saved = serde_json::to_string(&dto).unwrap();
    let saved_project: ProjectFile = serde_json::from_str(&saved).unwrap();
    assert_eq!(saved_project.view.camera, camera);
    assert!(session.mark_project_saved(revision, session.view_revision()));
    assert!(!session.is_dirty());

    session
        .dispatch(CommandRequest::SetCamera {
            camera: Camera::default(),
        })
        .unwrap();
    assert_ne!(session.view_state().camera, camera);
    assert!(!session.is_dirty());
    let token = ProjectOpenToken(2);
    session.begin_project_open(token).unwrap();
    session
        .complete_project_open(token, validate_project(saved_project).unwrap())
        .unwrap();
    assert_eq!(session.view_state().camera, camera);
    assert!(!session.is_dirty());
}

#[test]
fn project_open_command_emits_worker_request_and_completion_is_transactional() {
    let mut session = opened_session();
    let before = session.snapshot();
    let request: CommandRequest = serde_json::from_value(
        json!({"id": "project.open", "payload": {"token": 2, "source": "project-file-token"}}),
    )
    .unwrap();
    assert_eq!(request.id().as_str(), "project.open");
    let outcome = session.dispatch(request).unwrap();
    assert_eq!(
        outcome.effects,
        vec![Effect::OpenProject {
            token: ProjectOpenToken(2),
            source: OpaqueLocator("project-file-token".into()),
        }]
    );
    assert_eq!(session.snapshot(), before);
    assert!(outcome.snapshot.is_none());
    session.fail_project_open(ProjectOpenToken(2)).unwrap();
    assert_eq!(session.snapshot(), before);
    assert!(!session.is_dirty());
    let outcome = Session::new()
        .dispatch(CommandRequest::OpenProject {
            token: ProjectOpenToken(3),
            source: OpaqueLocator("empty-session-project".into()),
        })
        .unwrap();
    assert_eq!(outcome.effects.len(), 1);
}
