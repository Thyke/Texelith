use super::*;

fn frame(c: &mut Controls, ctx: &egui::Context, events: Vec<egui::Event>) -> egui::FullOutput {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            focused: true,
            events,
            ..Default::default()
        },
        |ui| {
            c.show(ui);
            c.show_workspace_center(ui, |_| {});
        },
    );
    output.textures_delta.clear();
    output
}

fn click(c: &mut Controls, ctx: &egui::Context, pos: egui::Pos2) {
    for pressed in [true, false] {
        frame(
            c,
            ctx,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Modifiers::NONE,
                },
            ],
        );
    }
}
fn find_text(shapes: &[egui::epaint::ClippedShape], wanted: &str) -> egui::Pos2 {
    fn find(shape: &egui::epaint::Shape, wanted: &str) -> Option<egui::Pos2> {
        match shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == wanted => {
                Some(text.pos + text.galley.size() * 0.5)
            }
            egui::epaint::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find(shape, wanted)),
            _ => None,
        }
    }
    shapes
        .iter()
        .find_map(|shape| find(&shape.shape, wanted))
        .unwrap_or_else(|| panic!("missing UI label: {wanted}"))
}

#[test]
fn map_card_palette_click_and_key_produce_identical_session_outcomes() {
    use texelith_ui_egui::input_locale::contract::key;
    for (keycode, map) in [
        Key::Num1,
        Key::Num2,
        Key::Num3,
        Key::Num4,
        Key::Num5,
        Key::Num6,
        Key::Num7,
    ]
    .into_iter()
    .zip(MAPS)
    {
        let mut outcomes = vec![];
        for route in ["card", "palette", "key"] {
            let mut c = material();
            let ctx = egui::Context::default();
            for _ in 0..3 {
                frame(&mut c, &ctx, vec![]);
            }
            match route {
                "card" => {
                    let pos = c.workspace_layout().cards[&map].select.center();
                    click(&mut c, &ctx, pos);
                }
                "palette" => {
                    frame(&mut c, &ctx, vec![key(Key::K, Modifiers::COMMAND)]);
                    frame(&mut c, &ctx, vec![]);
                    let output = frame(&mut c, &ctx, vec![]);
                    let label = format!(
                        "Select material map: {}",
                        texelith_ui_egui::material_ui::map_label(map)
                    );
                    let pos = find_text(&output.shapes, &label);
                    click(&mut c, &ctx, pos);
                }
                _ => {
                    frame(&mut c, &ctx, vec![key(keycode, Modifiers::NONE)]);
                }
            }
            assert_eq!(
                c.session()
                    .snapshot()
                    .unwrap()
                    .graph
                    .selected_output_ref()
                    .port
                    .map_kind(),
                map,
                "{route}"
            );
            let snapshot = c.session().snapshot().unwrap();
            outcomes.push((
                snapshot.graph.selected_output_ref(),
                snapshot.generation,
                c.session().document_revision(),
                c.session().history().undo_entries,
            ));
            assert!(c.take_effects().is_empty());
        }
        assert!(
            outcomes.windows(2).all(|pair| pair[0] == pair[1]),
            "{map:?}"
        );
    }
}

#[test]
fn modal_cancel_button_retains_dirty_material_and_escape_cancels_gesture_first() {
    use texelith_ui_egui::input_locale::contract::key;
    let mut c = material();
    let ctx = egui::Context::default();
    let revision = c.session().document_revision();
    c.drop_paths(vec!["replacement.png".into()]);
    for _ in 0..2 {
        frame(&mut c, &ctx, vec![]);
    }
    let output = frame(&mut c, &ctx, vec![]);
    let cancel = find_text(&output.shapes, "Cancel");
    click(&mut c, &ctx, cancel);
    assert!(c.pending_replacement().is_none());
    assert!(c.take_effects().is_empty());
    assert_eq!(c.session().document_revision(), revision);

    let snapshot = c.session().snapshot().unwrap();
    let node = snapshot
        .graph
        .material_nodes()
        .iter()
        .find(|n| {
            n.params
                .parameter(texelith_core::ParameterId::Strength)
                .is_ok()
        })
        .unwrap();
    let address = texelith_core::ParameterAddress {
        node: node.id,
        parameter: texelith_core::ParameterId::Strength,
    };
    c.action(UiAction::DragBegin { address });
    c.action(UiAction::Change {
        address,
        value: texelith_core::ParameterValue::Float(3.0),
    });
    frame(&mut c, &ctx, vec![key(Key::K, Modifiers::COMMAND)]);
    frame(&mut c, &ctx, vec![key(Key::Escape, Modifiers::NONE)]);
    assert!(c.session().history().active_gesture.is_none());
    assert_eq!(c.session().history().undo_entries, 0);
    let output = frame(&mut c, &ctx, vec![]);
    find_text(&output.shapes, "Search commands");
    frame(&mut c, &ctx, vec![key(Key::Escape, Modifiers::NONE)]);
}

#[test]
fn numeric_editor_keeps_digit_and_undo_shortcuts() {
    use texelith_ui_egui::input_locale::contract::key;
    let mut c = material();
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Normal,
        locked: true,
    }));
    let original = c.session().snapshot().unwrap();
    let history = c.session().history();
    let ctx = egui::Context::default();
    let mut number = 12.0_f32;
    for events in [
        vec![],
        vec![],
        vec![
            key(Key::Num7, Modifiers::NONE),
            egui::Event::Text("7".into()),
        ],
        vec![key(Key::Z, Modifiers::COMMAND)],
    ] {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1440.0, 900.0),
                )),
                focused: true,
                events,
                ..Default::default()
            },
            |ui| {
                egui::Window::new("Numeric focus fixture").show(ui, |ui| {
                    ui.add(egui::DragValue::new(&mut number)).request_focus();
                });
                c.show(ui);
                c.show_workspace_center(ui, |_| {});
            },
        );
        output.textures_delta.clear();
        assert!(ctx.egui_wants_keyboard_input());
        assert_eq!(c.session().snapshot().unwrap(), original);
        assert_eq!(c.session().history(), history);
    }
}

#[test]
fn platform_open_and_save_keys_show_explicit_project_ui_without_mutation() {
    use texelith_ui_egui::input_locale::contract::key;
    for modifiers in [
        Modifiers::CTRL,
        Modifiers {
            mac_cmd: true,
            command: true,
            ..Modifiers::NONE
        },
    ] {
        let mut c = material();
        let ctx = egui::Context::default();
        let before = c.session().snapshot().unwrap();
        frame(&mut c, &ctx, vec![key(Key::O, modifiers)]);
        let output = frame(&mut c, &ctx, vec![]);
        find_text(&output.shapes, "Project path (.texelith.json)");
        frame(&mut c, &ctx, vec![key(Key::Escape, Modifiers::NONE)]);
        for shift in [false, true] {
            frame(
                &mut c,
                &ctx,
                vec![key(Key::S, Modifiers { shift, ..modifiers })],
            );
            assert!(c.project_save_picker_open());
            assert_eq!(c.session().snapshot().unwrap(), before);
            assert!(c.session().is_dirty());
            assert!(c.take_effects().is_empty());
        }
    }
}
