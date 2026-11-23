use texelith_core::*;
use texelith_ui_egui::{controls::Controls, packing_editor::*, startup::Config};

fn material() -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!()
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [83; 32],
            extent: Extent::new(64, 64).unwrap(),
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c
}

#[test]
fn every_map_can_fill_every_destination_channel_through_export_commands() {
    let mut c = material();
    c.open_packing_editor().unwrap();
    let maps: Vec<_> = c
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_outputs()
        .unwrap()
        .keys()
        .copied()
        .collect();
    for map in maps {
        let components: &[Component] = match map {
            MapKind::Albedo => &[Component::R, Component::G, Component::B, Component::A],
            MapKind::Normal => &[Component::R, Component::G, Component::B],
            _ => &[Component::R],
        };
        for component in components {
            let output = &mut c.packing_preset_mut().unwrap().outputs[0];
            output.format = ExportFormat::Png8;
            output.channels = vec![
                PresetChannelSpec {
                    source: PresetChannelSource::Map {
                        map,
                        component: *component,
                        missing: MissingMap::Error
                    },
                    invert: true,
                    transfer: Transfer::Linear,
                };
                4
            ];
            output.import_hints = None;
            c.request_material_export().unwrap();
            let effects = c.take_effects();
            let [Effect::Export { plan, capture }] = effects.as_slice() else {
                panic!("{effects:?}")
            };
            for channel in &plan.outputs[0].channels {
                assert_eq!(
                    channel.source,
                    ChannelSource::Map(capture.graph.material_outputs().unwrap()[&map], *component)
                );
                assert!(channel.invert);
            }
        }
    }
}

#[test]
fn constants_transfer_warning_and_names_match_captured_plan() {
    let mut c = material();
    c.open_packing_editor().unwrap();
    let recipe = c.packing_preset_mut().unwrap();
    recipe.outputs.truncate(1);
    recipe.outputs[0].import_hints = None;
    recipe.outputs[0].channels = vec![
        PresetChannelSpec {
            source: PresetChannelSource::Constant(0.25),
            invert: true,
            transfer: Transfer::Linear,
        };
        4
    ];
    recipe.outputs[0].channels[1].transfer = Transfer::Srgb;
    recipe.outputs[0].channels[1].source = PresetChannelSource::Map {
        map: MapKind::Albedo,
        component: Component::R,
        missing: MissingMap::Error,
    };
    let plan = c.packing_plan().unwrap();
    assert_eq!(
        plan.outputs[0].channels[0].source,
        ChannelSource::Constant(0.25)
    );
    assert_eq!(
        plan.outputs[0].semantics().warnings,
        vec![ExportWarning::MixedTransfer]
    );
    assert_eq!(c.packing_names().unwrap(), vec!["T_Material_albedo.png"]);
    c.request_material_export().unwrap();
    let effects = c.take_effects();
    assert!(matches!(&effects[0], Effect::Export { plan: captured, .. } if *captured == plan));
}

#[test]
fn invalid_size_format_component_constant_or_name_never_exports() {
    let mut c = material();
    c.open_packing_editor().unwrap();
    let baseline = c.packing_preset_mut().unwrap().clone();
    for invalid in 0..7 {
        *c.packing_preset_mut().unwrap() = baseline.clone();
        c.export_options_mut().material_name = "Material".into();
        c.export_options_mut().size = SizePolicy::Keep;
        match invalid {
            0 => {
                c.export_options_mut().size = SizePolicy::Resize {
                    width: 0,
                    height: 64,
                }
            }
            1 => c.export_options_mut().material_name = "../escape".into(),
            2 => c.packing_preset_mut().unwrap().naming = "{unknown}".into(),
            3 => {
                c.packing_preset_mut().unwrap().outputs[0].channels[0].source =
                    PresetChannelSource::Constant(f32::NAN)
            }
            4 => c.packing_preset_mut().unwrap().outputs[0].format = ExportFormat::DdsBc5,
            5 => {
                c.packing_preset_mut().unwrap().outputs[0].channels[0].source =
                    PresetChannelSource::Map {
                        map: MapKind::Height,
                        component: Component::G,
                        missing: MissingMap::Error,
                    }
            }
            _ => c.packing_preset_mut().unwrap().outputs[0].name = "CON".into(),
        }
        let revision = c.session().document_revision();
        assert!(c.request_material_export().is_err(), "case {invalid}");
        assert!(c.take_effects().is_empty(), "case {invalid}");
        assert_eq!(c.session().document_revision(), revision);
    }
}

fn settle(c: &mut Controls) {
    let start = std::time::Instant::now();
    while c.preset_io_busy() {
        c.poll_preset_io();
        assert!(start.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

#[test]
fn versioned_custom_presets_survive_a_new_session_and_rebind_node_ids() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("evidence/T-083")
        .join(format!("presets-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("presets.json");
    let _ = std::fs::remove_file(&path);
    let mut c = material();
    c.start_preset_io(Some(path.clone())).unwrap();
    settle(&mut c);
    c.open_packing_editor().unwrap();
    c.packing_preset_mut().unwrap().outputs[0].channels[0].invert = true;
    c.export_options_mut().size = SizePolicy::Resize {
        width: 32,
        height: 16,
    };
    c.export_options_mut().format = texelith_ui_egui::export_jobs::FormatChoice::FloatDisplacement;
    let generation = c.session().snapshot().unwrap().generation;
    c.save_packing_preset("Stone").unwrap();
    settle(&mut c);
    assert!(
        c.preset_feedback().contains("Stone"),
        "{}",
        c.preset_feedback()
    );
    assert_eq!(c.session().snapshot().unwrap().generation, generation);
    assert!(
        c.take_effects().is_empty(),
        "IO must consume the real save effect"
    );
    let value = texelith_io::project_io::parse_json(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(value["version"], PRESET_LIBRARY_VERSION);
    assert!(!std::fs::read_to_string(&path).unwrap().contains("NodeId"));
    let mut reopened = material();
    reopened.start_preset_io(Some(path)).unwrap();
    settle(&mut reopened);
    assert_eq!(reopened.custom_presets().len(), 1);
    reopened.select_export_preset("custom-Stone").unwrap();
    reopened.request_material_export().unwrap();
    let effects = reopened.take_effects();
    let [Effect::Export { plan, .. }] = effects.as_slice() else {
        panic!()
    };
    assert_eq!(plan.profile.id, "custom-Stone");
    assert!(plan.outputs[0].channels[0].invert);
    assert_eq!(
        plan.size,
        SizePolicy::Resize {
            width: 32,
            height: 16
        }
    );
    assert!(plan.outputs.iter().any(|o| o.format == ExportFormat::Exr32));
    // A recipe resolves against a graph whose local IDs differ from its creator.
    let mut graph = MaterialGraphDto::from(reopened.session().snapshot().unwrap().graph);
    for node in &mut graph.graph {
        node.id.0 += 100;
        for edge in &mut node.inputs {
            edge.source.node.0 += 100;
        }
    }
    for reference in graph.outputs.values_mut() {
        reference.node.0 += 100;
    }
    graph.selected_output.node.0 += 100;
    let rebound = reopened.custom_presets()[0]
        .bind(&Graph::try_from(graph).unwrap(), &[])
        .unwrap();
    assert!(
        matches!(rebound.outputs[0].channels[0].source, ChannelSource::Map(reference, _) if reference.node.0 > 100)
    );
}

#[test]
fn preset_reader_rejects_future_duplicate_and_invalid_recipes() {
    let mut library = PresetLibrary::default();
    library.version += 1;
    assert!(library.validate().is_err());
    library = PresetLibrary::default();
    let mut preset = BuiltinExportPresets.presets()[0].clone();
    preset.profile.id = "custom-X".into();
    library.presets = vec![preset.clone(), preset];
    assert!(library.validate().is_err());
    library.presets.truncate(1);
    library.presets[0].outputs[0].channels[0].source = PresetChannelSource::Constant(2.0);
    assert!(library.validate().is_err());
    library.presets[0] = BuiltinExportPresets.presets()[0].clone();
    library.presets[0].profile.id = "custom-SmallPad".into();
    library.presets[0].size = SizePolicy::Pad {
        width: 8,
        height: 8,
    };
    assert!(
        library.validate().is_ok(),
        "small Pad recipes have no fixed source extent"
    );
}

#[test]
fn diagnostic_accepts_packing_only_and_rejects_unknown_scenarios() {
    let config = Config::parse(
        [
            "--mvp-smoke",
            "packing_ui",
            "--size",
            "1024",
            "--report",
            "out.json",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    assert!(config.packing_smoke && config.mvp_smoke);
    assert!(
        Config::parse(["--mvp-smoke", "unknown", "--report", "out.json"].map(str::to_owned))
            .is_err()
    );
}

#[test]
fn pointer_opens_editor_and_changes_inversion_without_mutating_graph() {
    let mut c = material();
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
    let click = |c: &mut Controls, rect: egui::Rect| {
        for pressed in [true, false] {
            frame(
                c,
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
    };
    for _ in 0..3 {
        frame(&mut c, vec![]);
    }
    assert!(c.packing_layout().open.is_none());
    let options = c.workspace_layout().export_options.unwrap();
    click(&mut c, options);
    for _ in 0..30 {
        frame(&mut c, vec![]);
    }
    let open = c.packing_layout().open.unwrap();
    assert!(
        c.workspace_layout()
            .export_options_area
            .unwrap()
            .contains_rect(open)
    );
    click(&mut c, open);
    for _ in 0..3 {
        frame(&mut c, vec![]);
    }
    let invert = c.packing_layout().inverts[0];
    let generation = c.session().snapshot().unwrap().generation;
    click(&mut c, invert);
    assert!(c.packing_preset_mut().unwrap().outputs[0].channels[0].invert);
    assert_eq!(c.session().snapshot().unwrap().generation, generation);
    c.request_material_export().unwrap();
    assert!(
        matches!(&c.take_effects()[0], Effect::Export { plan, .. } if plan.outputs[0].channels[0].invert)
    );
}

#[test]
fn malformed_storage_keeps_loaded_presets_and_can_be_reloaded_after_repair() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("evidence/T-083")
        .join(format!("reload-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("presets.json");
    let original = serde_json::json!({"version":PRESET_LIBRARY_VERSION, "presets":[]});
    std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut c = material();
    c.start_preset_io(Some(path.clone())).unwrap();
    settle(&mut c);
    c.open_packing_editor().unwrap();
    c.save_packing_preset("Keep").unwrap();
    settle(&mut c);
    let valid = std::fs::read(&path).unwrap();
    std::fs::write(&path, br#"{"version":1,"version":1,"presets":[]}"#).unwrap();
    c.reload_packing_presets().unwrap();
    settle(&mut c);
    assert_eq!(c.custom_presets().len(), 1);
    assert!(c.preset_feedback().contains("duplicate"));
    let bad = std::fs::read(&path).unwrap();
    assert!(c.save_packing_preset("Lost").is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bad);
    std::fs::write(&path, valid).unwrap();
    c.reload_packing_presets().unwrap();
    settle(&mut c);
    c.save_packing_preset("Recovered").unwrap();
    settle(&mut c);
    assert_eq!(c.custom_presets().len(), 2);
}
