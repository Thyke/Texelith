#[test]
fn workspace_routes_cards_and_fits_supported_sizes() {
    assert!(texelith_ui_egui::workspace::exercise().unwrap() >= 23);
}
#[test]
fn diagnostic_accepts_only_known_scenarios() {
    use texelith_ui_egui::startup::Config;
    assert!(
        Config::parse(["--mvp-smoke", "simple_layout", "--report", "out.json"].map(str::to_owned))
            .unwrap()
            .simple_layout_smoke
    );
    assert!(
        Config::parse(["--mvp-smoke", "unknown", "--report", "out.json"].map(str::to_owned))
            .is_err()
    );
}

use texelith_core::{
    CommandRequest, Effect, Extent, InputInterpretation, MapKind, SourceDescriptor,
};
use texelith_ui_egui::{command_adapter::UiAction, controls::Controls, material_ui::MAPS};

fn material(interpretation: InputInterpretation) -> Controls {
    let mut controls = Controls::new_material();
    controls.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = controls.take_effects().remove(0) else {
        panic!("import effect");
    };
    controls
        .complete_import(
            token,
            SourceDescriptor {
                content_hash: [3; 32],
                extent: Extent {
                    width: 64,
                    height: 32,
                },
                interpretation,
            },
        )
        .unwrap();
    controls
}

fn frame(
    controls: &mut Controls,
    ctx: &egui::Context,
    size: [f32; 2],
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
            focused: true,
            events,
            ..Default::default()
        },
        |ui| {
            controls.show(ui);
            controls.show_workspace_center(ui, |ui| {
                ui.label("Preview content supplied by the application");
            });
        },
    );
    output.textures_delta.clear();
    output
}

fn click(controls: &mut Controls, ctx: &egui::Context, size: [f32; 2], rect: egui::Rect) {
    for pressed in [true, false] {
        frame(
            controls,
            ctx,
            size,
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
}

fn context_menu(controls: &mut Controls, ctx: &egui::Context, size: [f32; 2], rect: egui::Rect) {
    for pressed in [true, false] {
        frame(
            controls,
            ctx,
            size,
            vec![
                egui::Event::PointerMoved(rect.center()),
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Secondary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    for _ in 0..3 {
        frame(controls, ctx, size, vec![]);
    }
}

#[test]
fn card_buttons_route_selection_visibility_lock_and_preserve_export() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    let size = [1600.0, 1000.0];
    for _ in 0..3 {
        frame(&mut controls, &ctx, size, vec![]);
    }
    let baseline = controls.session().snapshot().unwrap().generation;
    for map in MAPS {
        let card = controls.workspace_layout().cards[&map].clone();
        assert!(
            controls
                .workspace_layout()
                .maps
                .unwrap()
                .contains_rect(card.select),
            "{map:?} must be reachable"
        );
        click(&mut controls, &ctx, size, card.select);
        assert_eq!(
            controls
                .session()
                .snapshot()
                .unwrap()
                .graph
                .selected_output_ref()
                .port
                .map_kind(),
            map
        );
        context_menu(&mut controls, &ctx, size, card.select);
        let visibility = controls.workspace_layout().cards[&map].visibility;
        click(&mut controls, &ctx, size, visibility);
        assert!(
            !controls.session().view_state().maps[&map].visible,
            "visibility button for {map:?}"
        );
        frame(
            &mut controls,
            &ctx,
            size,
            vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        context_menu(&mut controls, &ctx, size, card.select);
        let lock = controls.workspace_layout().cards[&map].lock;
        click(&mut controls, &ctx, size, lock);
        frame(
            &mut controls,
            &ctx,
            size,
            vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(
            controls.session().view_state().maps[&map].locked,
            "lock button for {map:?}"
        );
        assert_eq!(controls.session().snapshot().unwrap().generation, baseline);
        controls.save_png("map.png".into(), map);
        assert!(
            matches!(controls.take_effects().as_slice(), [Effect::SavePng { map: actual, .. }] if *actual == map)
        );
    }
}

fn reach_export_control(
    controls: &mut Controls,
    ctx: &egui::Context,
    size: [f32; 2],
    field: impl Fn(&texelith_ui_egui::workspace::Layout) -> Option<egui::Rect>,
) -> egui::Rect {
    for _ in 0..12 {
        let layout = controls.workspace_layout();
        let rect = field(layout).expect("export control exists");
        let clip = layout
            .export_options_area
            .or(layout.inspector)
            .unwrap_or_else(|| layout.export.unwrap())
            .expand(0.5);
        if clip.contains_rect(rect) {
            return rect;
        }
        let delta = if rect.bottom() > clip.bottom() {
            -160.
        } else {
            160.
        };
        let pos = clip.center();
        frame(
            controls,
            ctx,
            size,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Unknown,
                    delta: egui::vec2(0., delta),
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        for _ in 0..12 {
            frame(controls, ctx, size, vec![]);
        }
    }
    panic!(
        "export control must be reachable by scrolling at {size:?}: {:?}",
        field(controls.workspace_layout())
    );
}

#[test]
fn export_and_center_stay_inside_the_window_at_all_required_scales() {
    for physical in [[960.0, 600.0], [1280.0, 720.0]] {
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let mut controls = material(InputInterpretation::SrgbPhoto);
            let ctx = egui::Context::default();
            ctx.set_pixels_per_point(scale);
            let size = [physical[0] / scale, physical[1] / scale];
            for _ in 0..3 {
                frame(&mut controls, &ctx, size, vec![]);
            }
            assert!(controls.workspace_layout().save_png.is_none());
            if controls.workspace_layout().inspector.is_none() {
                let button = controls.workspace_layout().export_material.unwrap();
                click(&mut controls, &ctx, size, button);
                for _ in 0..3 {
                    frame(&mut controls, &ctx, size, vec![]);
                }
            }
            let options = reach_export_control(&mut controls, &ctx, size, |l| l.export_options);
            click(&mut controls, &ctx, size, options);
            for _ in 0..30 {
                frame(&mut controls, &ctx, size, vec![]);
            }
            let save = reach_export_control(&mut controls, &ctx, size, |l| l.save_png);
            let bounds = egui::Rect::from_min_size(egui::Pos2::ZERO, size.into());
            let layout = controls.workspace_layout();
            let export = layout.export.unwrap();
            let center = layout.center.unwrap();
            // egui quantizes layout positions to 1/32 point; allow at most half a physical pixel.
            assert!(
                bounds.expand(0.5 / scale).contains_rect(export),
                "export: {physical:?}@{scale}: {export:?}, scalar {:?}, bounds {:?}",
                [export.min.x, export.min.y, export.max.x, export.max.y],
                [bounds.min.x, bounds.min.y, bounds.max.x, bounds.max.y]
            );
            assert!(
                bounds.contains_rect(save),
                "PNG save reachable at {physical:?}@{scale}"
            );
            assert!(
                center.width() > 0.0 && center.height() > 0.0,
                "center: {physical:?}@{scale}"
            );
            assert!(
                !center.intersects(export) || layout.inspector.is_none(),
                "{physical:?}@{scale} center={center:?} export={export:?}"
            );
            if size[0] < 1000.0 {
                assert!(
                    layout.inspector.is_none(),
                    "compact default leaves center visible"
                );
            }
        }
    }
}

#[test]
fn export_section_lives_in_the_right_layer_panel_at_1600_by_1000() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    for _ in 0..30 {
        frame(&mut controls, &ctx, [1600.0, 1000.0], vec![]);
    }
    let layout = controls.workspace_layout();
    assert!(
        layout
            .inspector
            .unwrap()
            .contains_rect(layout.export.unwrap())
    );
    assert!(!layout.center.unwrap().intersects(layout.export.unwrap()));
    assert!(layout.export_material.is_some());
    assert!(layout.export_options.is_some());
    assert!(layout.destination.is_some());
    assert!(layout.save_png.is_none());
}

#[test]
fn compact_panel_toggles_are_real_and_survive_resizing() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    let size = [960.0, 600.0];
    for _ in 0..3 {
        frame(&mut controls, &ctx, size, vec![]);
    }
    assert!(controls.workspace_layout().inspector.is_none());
    let toggle = controls.workspace_layout().inspector_toggle.unwrap();
    click(&mut controls, &ctx, size, toggle);
    frame(&mut controls, &ctx, size, vec![]);
    assert!(controls.workspace_layout().inspector.is_some());
    assert!(controls.workspace_layout().center.unwrap().width() > 100.0);
    let toggle = controls.workspace_layout().maps_toggle.unwrap();
    click(&mut controls, &ctx, size, toggle);
    frame(&mut controls, &ctx, size, vec![]);
    assert!(controls.workspace_layout().maps.is_none());
    for _ in 0..3 {
        frame(&mut controls, &ctx, [1440.0, 900.0], vec![]);
    }
    assert!(controls.workspace_layout().maps.is_none());
    assert!(controls.workspace_layout().inspector.is_some());
}

#[test]
fn thumbnail_cache_is_bounded_and_replacement_import_discards_old_maps() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let image = egui::ColorImage::filled([1024, 512], egui::Color32::from_rgb(21, 40, 60));
    let preview = texelith_ui_egui::workspace::thumbnail(&image);
    assert_eq!(preview.size, [128, 64]);
    assert!(preview.pixels.iter().all(|p| *p == image.pixels[0]));
    for map in MAPS {
        controls.cache_map_thumbnail(map, 1, &image);
    }
    assert!(
        MAPS.iter()
            .all(|map| controls.map_thumbnail_size(*map) == Some([128, 64]))
    );
    controls.open_image("replacement.png".into());
    let Effect::LoadImage { token, .. } = controls.take_effects().remove(0) else {
        panic!("import effect");
    };
    controls
        .complete_import(
            token,
            SourceDescriptor {
                content_hash: [4; 32],
                extent: Extent {
                    width: 32,
                    height: 64,
                },
                interpretation: InputInterpretation::SrgbPhoto,
            },
        )
        .unwrap();
    assert!(
        MAPS.iter()
            .all(|map| controls.map_thumbnail_size(*map).is_none())
    );
}

#[test]
fn source_badges_follow_imported_channel_and_do_not_advertise_ai() {
    for (interpretation, imported) in [
        (InputInterpretation::SrgbPhoto, MapKind::Albedo),
        (InputInterpretation::LinearHeight, MapKind::Height),
        (InputInterpretation::LinearNormal, MapKind::Normal),
    ] {
        let controls = material(interpretation);
        let snapshot = controls.session().snapshot().unwrap();
        for map in MAPS {
            assert_eq!(
                texelith_ui_egui::workspace::source_badge(&snapshot, map),
                if map == imported {
                    "Manual"
                } else {
                    "Procedural"
                }
            );
        }
    }
}

#[test]
fn hidden_preview_neutrals_are_semantic_and_set_export_is_explicitly_pending() {
    use texelith_ui_egui::workspace::neutral_preview;
    assert_eq!(
        neutral_preview(MapKind::Normal),
        egui::Color32::from_rgb(128, 128, 255)
    );
    assert_eq!(neutral_preview(MapKind::Ao), egui::Color32::WHITE);
    assert_eq!(neutral_preview(MapKind::Metallic), egui::Color32::BLACK);
    let mut controls = material(InputInterpretation::SrgbPhoto);
    controls.action(UiAction::Request(CommandRequest::SetVisible {
        map: MapKind::Normal,
        visible: false,
    }));
    controls.save_png("normal.png".into(), MapKind::Normal);
    assert!(matches!(
        controls.take_effects().as_slice(),
        [Effect::SavePng {
            map: MapKind::Normal,
            ..
        }]
    ));
}

#[test]
fn export_bar_routes_png_button_and_pending_set_export_has_no_effect() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    let size = [1600.0, 1400.0];
    for _ in 0..3 {
        frame(&mut controls, &ctx, size, vec![]);
    }
    let button = controls.workspace_layout().export_material.unwrap();
    click(&mut controls, &ctx, size, button);
    assert!(
        matches!(controls.take_effects().as_slice(), [Effect::Export { .. }]),
        "Export dispatches the material-set effect"
    );
    assert!(controls.workspace_layout().destination.is_some());
    let options = controls.workspace_layout().export_options.unwrap();
    click(&mut controls, &ctx, size, options);
    for _ in 0..30 {
        frame(&mut controls, &ctx, size, vec![]);
    }
    let destination = controls.workspace_layout().destination.unwrap();
    click(&mut controls, &ctx, size, destination);
    frame(
        &mut controls,
        &ctx,
        size,
        vec![egui::Event::Text("selected.png".into())],
    );
    let save = controls.workspace_layout().save_png.unwrap();
    click(&mut controls, &ctx, size, save);
    assert!(
        matches!(controls.take_effects().as_slice(), [Effect::SavePng { .. }]),
        "PNG button must dispatch the existing effect"
    );
}

fn visible_text(shapes: &[egui::epaint::ClippedShape]) -> String {
    fn append(shape: &egui::epaint::Shape, result: &mut String) {
        match shape {
            egui::epaint::Shape::Text(text) => {
                result.push_str(&text.galley.job.text);
                result.push('\n');
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    append(shape, result);
                }
            }
            _ => {}
        }
    }
    let mut text = String::new();
    for shape in shapes {
        append(&shape.shape, &mut text);
    }
    text
}

#[test]
fn inspector_exposes_key_controls_without_opening_advanced() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    controls.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Normal,
    }));
    let ctx = egui::Context::default();
    for _ in 0..3 {
        frame(&mut controls, &ctx, [1280.0, 800.0], vec![]);
    }
    let output = frame(&mut controls, &ctx, [1280.0, 800.0], vec![]);
    let text = visible_text(&output.shapes);
    assert!(text.contains("Normal Properties"), "{text}");
    assert!(
        text.contains("Strength"),
        "normal strength is visible outside Advanced"
    );
    assert!(text.contains("Advanced"));
    assert!(!text.contains("Normal filter"), "Advanced starts collapsed");
    assert!(!text.contains("Preset formats"));
    let options = controls.workspace_layout().export_options.unwrap();
    click(&mut controls, &ctx, [1280.0, 800.0], options);
    let output = frame(&mut controls, &ctx, [1280.0, 800.0], vec![]);
    assert!(visible_text(&output.shapes).contains("Preset formats"));
}

#[test]
fn application_can_attach_inspector_and_export_leaves_without_replacing_workspace() {
    let mut controls = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    for _ in 0..3 {
        let mut output = ctx.run_ui(
            egui::RawInput {
                focused: true,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                ..Default::default()
            },
            |ui| {
                controls.show_workspace_with(
                    ui,
                    |_, ui| {
                        ui.label("Attached inspector");
                    },
                    |_, ui| {
                        ui.label("Attached exporter");
                    },
                );
                controls.show_workspace_center(ui, |ui| {
                    ui.label("Attached viewport");
                });
            },
        );
        output.textures_delta.clear();
        let text = visible_text(&output.shapes);
        assert!(text.contains("Attached inspector"));
        assert!(text.contains("Attached exporter"));
        assert!(text.contains("Attached viewport"));
    }
    assert_eq!(controls.workspace_layout().cards.len(), 7);
    let center = controls.workspace_layout().center.unwrap();
    assert!(center.width() > 0.0 && center.height() > 0.0);
}

#[test]
fn horizontal_map_chips_scroll_and_export_checkboxes_change_captured_channels() {
    let mut c = material(InputInterpretation::SrgbPhoto);
    let ctx = egui::Context::default();
    let size = [1280., 800.];
    for _ in 0..5 {
        frame(&mut c, &ctx, size, vec![]);
    }
    assert!(
        c.workspace_layout().cards.values().all(|card| c
            .workspace_layout()
            .maps
            .unwrap()
            .contains_rect(card.select)),
        "all seven compact chips fit at 1280"
    );
    // Force a narrower canvas to exercise the last-resort horizontal scrollbar.
    let size = [960., 600.];
    frame(&mut c, &ctx, size, vec![]);
    let toggle = c.workspace_layout().inspector_toggle.unwrap();
    click(&mut c, &ctx, size, toggle);
    for _ in 0..5 {
        frame(&mut c, &ctx, size, vec![]);
    }
    let maps = c.workspace_layout().maps.unwrap();
    assert!(!maps.contains_rect(c.workspace_layout().cards[&MapKind::Edge].select));
    frame(
        &mut c,
        &ctx,
        size,
        vec![
            egui::Event::PointerMoved(maps.center()),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                delta: egui::vec2(-500., 0.),
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    for _ in 0..30 {
        frame(&mut c, &ctx, size, vec![]);
    }
    let edge = c.workspace_layout().cards[&MapKind::Edge].select;
    assert!(
        maps.contains_rect(edge),
        "last chip becomes reachable through horizontal scrolling"
    );
    click(&mut c, &ctx, size, edge);
    assert_eq!(
        c.session()
            .snapshot()
            .unwrap()
            .graph
            .selected_output_ref()
            .port
            .map_kind(),
        MapKind::Edge
    );
    // Use Normal's compact property group to keep all export checkboxes visible.
    c.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Normal,
    }));
    for _ in 0..5 {
        frame(&mut c, &ctx, [1600., 1000.], vec![]);
    }
    for map in [MapKind::Albedo, MapKind::Edge] {
        let checkbox = c.workspace_layout().export_channels[&map];
        click(&mut c, &ctx, [1600., 1000.], checkbox);
    }
    c.request_material_export().unwrap();
    let effects = c.take_effects();
    let Effect::Export { plan, .. } = &effects[0] else {
        panic!("export effect");
    };
    let sources: Vec<_> = plan
        .outputs
        .iter()
        .flat_map(|o| &o.channels)
        .filter_map(|channel| match channel.source {
            texelith_core::ChannelSource::Map(reference, _) => Some(reference.port.map_kind()),
            _ => None,
        })
        .collect();
    assert!(!sources.contains(&MapKind::Albedo));
    assert!(sources.contains(&MapKind::Edge));
}
