use texelith_core::*;
use texelith_ui_egui::{
    command_adapter::UiAction,
    controls::Controls,
    material_ui::{MAPS, options, request, supported},
};
fn loaded(interpretation: InputInterpretation) -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("load effect required")
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [3; 32],
            extent: Extent {
                width: 257,
                height: 193,
            },
            interpretation,
        },
    )
    .unwrap();
    c
}
#[test]
fn photo_import_provides_seven_selectable_maps_and_numeric_png_effects() {
    let mut c = loaded(InputInterpretation::SrgbPhoto);
    assert!(c.session().snapshot().unwrap().graph.is_material());
    for map in MAPS {
        let generation = c.session().snapshot().unwrap().generation;
        c.action(UiAction::Request(CommandRequest::SelectMap { map }));
        let s = c.session().snapshot().unwrap();
        assert_eq!(s.graph.selected_output_ref().port.map_kind(), map);
        assert_eq!(s.generation, generation);
        assert!(supported(&s, map));
        c.save_png(format!("{map:?}.png").into(), map);
        assert!(
            matches!(c.take_effects().as_slice(), [Effect::SavePng{map:m,generation:g,..}] if *m==map && *g==generation)
        );
    }
}
#[test]
fn seam_and_optional_mask_links_are_real_undoable_graph_edits() {
    let mut c = loaded(InputInterpretation::SrgbPhoto);
    let before = c.session().snapshot().unwrap();
    assert_eq!(options(&before.graph), (true, false));
    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
        seamless: false,
        painted_mask: true,
    }));
    let after = c.session().snapshot().unwrap();
    assert_ne!(before.graph, after.graph);
    assert_eq!(options(&after.graph), (false, true));
    for map in MAPS {
        assert!(supported(&after, map), "{map:?}");
    }
    c.action(UiAction::UndoShortcut);
    assert_eq!(c.session().snapshot().unwrap().graph, before.graph);
    c.action(UiAction::RedoShortcut);
    assert_eq!(c.session().snapshot().unwrap().graph, after.graph);
}
#[test]
fn imported_normal_is_separate_and_height_from_normal_is_available() {
    let c = loaded(InputInterpretation::LinearNormal);
    let s = c.session().snapshot().unwrap();
    assert!(
        s.graph
            .material_nodes()
            .iter()
            .any(|n| matches!(n.params, MaterialNodeParams::HeightFromNormal { .. }))
    );
    assert!(
        !s.graph
            .material_nodes()
            .iter()
            .any(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
    );
    assert!(supported(&s, MapKind::Normal));
    assert!(supported(&s, MapKind::Edge));
    // T-050 (height from normal) is merged, so Height is now supported.
    assert!(supported(&s, MapKind::Height));
}
#[test]
fn typed_parameters_coalesce_and_failed_load_preserves_material() {
    let mut c = loaded(InputInterpretation::SrgbPhoto);
    let node = c
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_nodes()
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
        .unwrap()
        .id;
    let address = ParameterAddress {
        node,
        parameter: ParameterId::Strength,
    };
    c.action(UiAction::DragBegin { address });
    for i in 1..10 {
        c.action(UiAction::Change {
            address,
            value: ParameterValue::Float(i as f32 / 10.0),
        });
    }
    c.action(UiAction::DragRelease);
    assert_eq!(c.session().history().undo_entries, 1);
    let before = c.session().snapshot().unwrap();
    c.action(UiAction::Change {
        address,
        value: ParameterValue::Float(f32::NAN),
    });
    assert_eq!(before, c.session().snapshot().unwrap());
    c.open_image("bad.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!()
    };
    c.fail_import(token);
    assert_eq!(before, c.session().snapshot().unwrap());
    let r = request(
        before.clone(),
        MapKind::Normal,
        42,
        texelith_gpu::Quality::Proxy,
    )
    .unwrap();
    assert_eq!(r.request_id, 42);
    assert_eq!(
        r.extent,
        texelith_gpu::proxy_extent(
            before
                .graph
                .material_output_extent(before.graph.selected_output_ref())
                .unwrap()
        )
        .unwrap()
    );
    c.action(UiAction::Request(CommandRequest::InspectSeams {
        enabled: true,
    }));
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::InspectSeams { enabled: true }]
    ));
    assert_eq!(before, c.session().snapshot().unwrap());
}

#[test]
fn maps_diagnostic_rejects_unknown_scenarios_and_incomplete_arguments() {
    use texelith_ui_egui::startup::Config;
    for args in [
        vec!["--mvp-smoke", "unknown"],
        vec!["--mvp-smoke", "maps"],
        vec![
            "--mvp-smoke",
            "maps",
            "--report",
            "report.json",
            "--size",
            "0",
        ],
    ] {
        assert!(Config::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    assert!(
        Config::parse(
            [
                "--mvp-smoke",
                "maps",
                "--report",
                "report.json",
                "--size",
                "1024"
            ]
            .into_iter()
            .map(str::to_owned)
        )
        .unwrap()
        .mvp_smoke
    );
}

#[test]
fn cancelled_png_publication_preserves_an_existing_destination() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target-evidence/T-061/tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("publication-{}.png", std::process::id()));
    let before = texelith_io::encode_height_png(1, 1, &[1234]).unwrap();
    let after = texelith_io::encode_height_png(1, 1, &[5678]).unwrap();
    texelith_io::publish_png(&path, &before).unwrap();
    assert!(texelith_io::publish_png_if_current(&path, &after, || false).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn linear_height_seam_toggle_discovery_and_history_remain_consistent() {
    let mut c = loaded(InputInterpretation::LinearHeight);
    assert_eq!(
        options(&c.session().snapshot().unwrap().graph),
        (true, false)
    );
    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
        seamless: false,
        painted_mask: false,
    }));
    assert_eq!(
        options(&c.session().snapshot().unwrap().graph),
        (false, false)
    );
    c.action(UiAction::UndoShortcut);
    assert_eq!(
        options(&c.session().snapshot().unwrap().graph),
        (true, false)
    );
}

#[test]
fn imported_height_and_normal_integration_do_not_apply_an_implicit_remap() {
    for interpretation in [
        InputInterpretation::LinearHeight,
        InputInterpretation::LinearNormal,
    ] {
        let c = loaded(interpretation);
        let snapshot = c.session().snapshot().unwrap();
        let mut stack = vec![snapshot.graph.material_outputs().unwrap()[&MapKind::Height].node];
        while let Some(id) = stack.pop() {
            let node = snapshot.graph.material_node(id).unwrap();
            assert!(
                !matches!(node.params, MaterialNodeParams::Levels { .. }),
                "remapping must be explicit"
            );
            stack.extend(node.inputs.iter().map(|edge| edge.source.node));
        }
    }
}

#[test]
fn map_shortcuts_use_commands_and_leave_document_content_and_history_unchanged() {
    let mut c = loaded(InputInterpretation::SrgbPhoto);
    let before = c.session().snapshot().unwrap();
    let history = c.session().history();
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        focused: true,
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        events: vec![egui::Event::Key {
            key: egui::Key::Num5,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let mut output = ctx.run_ui(input, |ui| c.show(ui));
    output.textures_delta.clear();
    let after = c.session().snapshot().unwrap();
    assert_eq!(
        after.graph.selected_output_ref().port.map_kind(),
        MapKind::Roughness
    );
    assert_eq!(after.generation, before.generation);
    assert_eq!(after.graph.material_nodes(), before.graph.material_nodes());
    assert_eq!(c.session().history(), history);
}
