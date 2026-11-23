use texelith_core::*;
use texelith_ui_egui::{command_adapter::UiAction, compare::*, controls::Controls};

#[test]
fn viewport_commands_preserve_graph_history_and_capture_recipes() {
    let mut c = Controls::new(true);
    let original = c.session().snapshot().unwrap();
    for request in [
        CommandRequest::SetMode {
            mode: ViewMode::ThreeD,
        },
        CommandRequest::SetMesh {
            mesh: PreviewMesh::Sphere,
        },
        CommandRequest::SetTiles {
            tiles: Tiles {
                count: 3,
                seed: 42,
                random_rotation: true,
            },
        },
        CommandRequest::SetPom {
            pom: Pom {
                enabled: true,
                scale: 0.05,
            },
        },
        CommandRequest::CaptureA {},
        CommandRequest::CaptureB {},
        CommandRequest::SetSplit { fraction: 0.3 },
    ] {
        c.action(UiAction::Request(request));
    }
    assert_eq!(c.session().view_state().mode, ViewMode::ThreeD);
    assert_eq!(c.session().view_state().mesh, PreviewMesh::Sphere);
    assert_eq!(c.session().view_state().tiles.count, 3);
    assert!(c.session().captured_a().is_some());
    assert!(c.session().captured_b().is_some());
    assert_eq!(
        c.session().snapshot().unwrap().generation,
        original.generation
    );
    assert_eq!(c.session().history().undo_entries, 0);
}

#[test]
fn split_never_publishes_a_stale_half_or_mismatched_camera() {
    let mut pair = PairCache::default();
    pair.begin(7);
    assert!(!pair.insert(Side::A, 6, 1));
    assert!(pair.insert(Side::A, 7, 2));
    assert!(pair.pair().is_none());
    pair.begin(8);
    assert!(!pair.insert(Side::B, 7, 3));
    assert!(pair.insert(Side::B, 8, 4));
    assert!(pair.pair().is_none());
    assert!(pair.insert(Side::A, 8, 5));
    assert_eq!(pair.pair(), Some((&5, &4)));
}

#[test]
fn framebuffer_and_orbit_are_bounded() {
    let extent = compare_extent(Extent::new(3840, 2160).unwrap());
    assert_eq!(extent, Extent::new(1024, 576).unwrap());
    let c = orbit(Camera::default(), [100., 10000.], -10000.);
    assert!(c.pitch.abs() <= 1.5);
    assert!((0.3..=20.).contains(&c.distance));
}

#[test]
fn coherent_material_request_obeys_proxy_size_visibility_and_current_channel() {
    let source = SourceDescriptor {
        content_hash: [69; 32],
        extent: Extent::new(4096, 2049).unwrap(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let snapshot = EvaluationSnapshot {
        graph: material_from_source(source).unwrap(),
        generation: 42,
    };
    let mut view = ViewState::default();
    let request = material_request(snapshot.clone(), MapKind::Normal, 69, true, &view).unwrap();
    assert_eq!(request.extent, Extent::new(1024, 512).unwrap());
    assert_eq!(request.outputs.len(), 6);
    assert!(request.outputs.contains_key(&MapKind::Albedo));
    assert!(request.outputs.contains_key(&MapKind::Height));
    assert_eq!(request.snapshot.generation, 42);
    assert!(texelith_gpu::MaterialPlan::new(&request, 1_073_741_824, Default::default()).is_ok());
    view.maps.insert(
        MapKind::Normal,
        MapView {
            visible: false,
            locked: false,
        },
    );
    let hidden = material_request(snapshot.clone(), MapKind::Normal, 70, true, &view).unwrap();
    assert!(!hidden.outputs.contains_key(&MapKind::Normal));
    let channel = material_request(snapshot, MapKind::Roughness, 71, false, &view).unwrap();
    assert_eq!(
        channel.outputs.keys().copied().collect::<Vec<_>>(),
        vec![MapKind::Roughness]
    );
}

#[test]
fn viewport_smoke_and_snapshot_options_are_validated() {
    use texelith_ui_egui::startup::Config;
    let parse = |s: &str| Config::parse(s.split_whitespace().map(str::to_string));
    assert!(
        parse("--mvp-smoke viewport --size 1024 --report viewport.json")
            .unwrap()
            .viewport
    );
    assert!(parse("--mvp-smoke unknown --report unknown.json").is_err());
    assert!(parse("--viewport-mesh unknown").is_err());
    assert_eq!(
        parse("--viewport-mesh cylinder").unwrap().viewport_mesh,
        "cylinder"
    );
}
