use texelith_core::*;
use texelith_ui_egui::{
    command_adapter::UiAction,
    controls::Controls,
    export_jobs::{ExportOptions, FormatChoice, build_plan},
    startup::Config,
};

fn material() -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("import effect missing")
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [82; 32],
            extent: Extent::new(64, 64).unwrap(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c
}

#[test]
fn export_command_availability_requires_a_document_and_an_active_job() {
    let empty = Session::new();
    for id in [CommandId::RequestExport, CommandId::CancelExport] {
        assert!(
            !empty
                .availability()
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .enabled
        );
    }
    let mut c = material();
    assert!(
        !c.session()
            .availability()
            .iter()
            .find(|a| a.id == CommandId::CancelExport)
            .unwrap()
            .enabled
    );
    c.request_material_export().unwrap();
    // Capturing an effect alone has not admitted a worker job.
    assert!(
        !c.session()
            .availability()
            .iter()
            .find(|a| a.id == CommandId::CancelExport)
            .unwrap()
            .enabled
    );
    assert!(
        c.session()
            .availability()
            .iter()
            .find(|a| a.id == CommandId::RequestExport)
            .unwrap()
            .enabled
    );
}

#[test]
fn all_presets_bind_and_float_displacement_is_explicit() {
    let c = material();
    let graph = c.session().snapshot().unwrap().graph;
    for preset in BuiltinExportPresets.presets() {
        let mut options = ExportOptions {
            preset: preset.profile.id.clone(),
            ..Default::default()
        };
        let plan = build_plan(&graph, &options).unwrap();
        assert_eq!(plan, preset.bind(&graph, &[]).unwrap());
        options.format = FormatChoice::FloatDisplacement;
        options.size = SizePolicy::Resize {
            width: 32,
            height: 16,
        };
        let plan = build_plan(&graph, &options).unwrap();
        assert_eq!(plan.size, options.size);
        assert!(plan.outputs.iter().any(|o| o.format == ExportFormat::Exr32));
        validate_export_for_graph(&plan, &graph).unwrap();
    }
}

#[test]
fn button_and_palette_use_the_same_immutable_command_effect() {
    let mut c = material();
    c.request_material_export().unwrap();
    let Effect::Export { plan, capture } = c.take_effects().remove(0) else {
        panic!("real export effect missing")
    };
    let baseline = capture.clone();
    let normal = *capture
        .graph
        .material_outputs()
        .unwrap()
        .get(&MapKind::Normal)
        .unwrap();
    c.action(UiAction::Change {
        address: ParameterAddress {
            node: normal.node,
            parameter: ParameterId::Strength,
        },
        value: ParameterValue::Float(2.0),
    });
    c.action(UiAction::Invoke {
        origin: texelith_ui_egui::command_adapter::ActionOrigin::Palette,
        id: CommandId::RequestExport,
    });
    let Effect::Export {
        capture: next,
        plan: next_plan,
    } = c.take_effects().remove(0)
    else {
        panic!("registry export effect missing")
    };
    assert_eq!(plan, next_plan);
    assert_eq!(capture, baseline);
    assert_ne!(capture.graph_hash, next.graph_hash);
    assert_ne!(capture.generation, next.generation);
    let generation = c.session().snapshot().unwrap().generation;
    c.action(UiAction::Request(CommandRequest::CancelExport {
        job_id: 82,
    }));
    assert!(matches!(
        c.take_effects().as_slice(),
        [Effect::CancelExport { job_id: 82 }]
    ));
    assert_eq!(c.session().snapshot().unwrap().generation, generation);
}

#[test]
fn export_diagnostic_is_recognized_and_unknown_scenario_fails() {
    let config = Config::parse(
        [
            "--mvp-smoke",
            "export",
            "--size",
            "1024",
            "--report",
            "target/evidence/T-082/test.json",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    assert!(config.export_smoke && config.mvp_smoke);
    assert!(
        Config::parse(["--mvp-smoke", "unknown", "--report", "out.json"].map(str::to_owned))
            .is_err()
    );
}

#[test]
fn real_shared_gpu_export_contract() {
    let config = Config {
        size: 32,
        report: Some(
            std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
                .join("evidence/T-082/test-export.json"),
        ),
        ..Default::default()
    };
    // Missing hardware is a failure, never a passing skip.
    let report = texelith_ui_egui::export_jobs::smoke::exercise(&config).unwrap();
    assert_eq!(report["presets"].as_array().unwrap().len(), 6);
    for name in [
        "immutable_edits",
        "cancel_preserves_destination",
        "collision_preserves_destination",
        "one_job_bound",
        "float_displacement",
        "error_identity",
        "publication_only_completion",
        "cancel_availability",
    ] {
        assert_eq!(report[name], true, "{name}");
    }
}

#[test]
fn export_bar_click_captures_selected_options_and_keeps_legacy_save_available() {
    let mut c = material();
    c.export_options_mut().preset = "unity-hdrp-v1".into();
    c.export_options_mut().size = SizePolicy::Resize {
        width: 32,
        height: 16,
    };
    c.export_options_mut().format = FormatChoice::FloatDisplacement;
    let ctx = egui::Context::default();
    let frame = |c: &mut Controls, events| {
        let mut output = ctx.run_ui(
            egui::RawInput {
                focused: true,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                events,
                ..Default::default()
            },
            |ui| c.show(ui),
        );
        output.textures_delta.clear();
    };
    for _ in 0..3 {
        frame(&mut c, vec![]);
    }
    let rect = c.workspace_layout().export_material.unwrap();
    assert!(c.workspace_layout().export.unwrap().contains_rect(rect));
    assert!(c.workspace_layout().save_png.is_none());
    let options = c.workspace_layout().export_options.unwrap();
    for pressed in [true, false] {
        frame(
            &mut c,
            vec![
                egui::Event::PointerMoved(options.center()),
                egui::Event::PointerButton {
                    pos: options.center(),
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    for _ in 0..3 {
        frame(&mut c, vec![]);
    }
    assert!(c.workspace_layout().save_png.is_some());
    let rect = c.workspace_layout().export_material.unwrap();
    for pressed in [true, false] {
        frame(
            &mut c,
            vec![
                egui::Event::PointerMoved(rect.center()),
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    let effects = c.take_effects();
    let [Effect::Export { plan, .. }] = effects.as_slice() else {
        panic!("Export button emitted {effects:?}")
    };
    assert_eq!(plan.profile.id, "unity-hdrp-v1");
    assert_eq!(
        plan.size,
        SizePolicy::Resize {
            width: 32,
            height: 16
        }
    );
    assert!(plan.outputs.iter().any(|o| o.format == ExportFormat::Exr32));
}

#[test]
fn invalid_settings_do_not_emit_an_export_or_mutate_the_document() {
    let mut c = material();
    let revision = c.session().document_revision();
    c.export_options_mut().size = SizePolicy::Resize {
        width: 0,
        height: 16,
    };
    assert!(c.request_material_export().is_err());
    assert!(c.take_effects().is_empty());
    assert_eq!(c.session().document_revision(), revision);
    c.export_options_mut().size = SizePolicy::Keep;
    c.export_options_mut().preset = "missing-preset".into();
    assert!(c.request_material_export().is_err());
    assert!(c.take_effects().is_empty());
    assert_eq!(c.session().document_revision(), revision);
}
