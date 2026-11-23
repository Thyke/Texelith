use texelith_core::*;
use texelith_ui_egui::{command_adapter::UiAction, controls::Controls, inspectors::PaintPhase};

fn material() -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("load effect")
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [9; 32],
            extent: Extent {
                width: 64,
                height: 32,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c
}
fn address(c: &Controls, parameter: ParameterId) -> ParameterAddress {
    let snapshot = c.session().snapshot().unwrap();
    let node = snapshot
        .graph
        .material_nodes()
        .iter()
        .find(|n| n.params.parameter(parameter).is_ok())
        .unwrap()
        .id;
    ParameterAddress { node, parameter }
}
fn select(c: &mut Controls, map: MapKind) {
    c.action(UiAction::Request(CommandRequest::SelectMap { map }));
}
fn value(c: &Controls, a: ParameterAddress) -> ParameterValue {
    c.session()
        .snapshot()
        .unwrap()
        .graph
        .material_node(a.node)
        .unwrap()
        .params
        .parameter(a.parameter)
        .unwrap()
}

#[test]
fn typed_slider_and_curve_are_single_transactions_with_cancel_and_focus_loss() {
    let mut c = material();
    let a = address(&c, ParameterId::Strength);
    let original = value(&c, a);
    let before = c.session().history().undo_entries;
    c.action(UiAction::DragBegin { address: a });
    for v in [2.0, 3.0, 4.0] {
        c.action(UiAction::Change {
            address: a,
            value: ParameterValue::Float(v),
        });
    }
    c.action(UiAction::FocusLost);
    assert_eq!(c.session().history().undo_entries, before + 1);
    c.action(UiAction::UndoShortcut);
    assert_eq!(value(&c, a), original);
    let curve = address(&c, ParameterId::Curve);
    let original = value(&c, curve);
    c.action(UiAction::DragBegin { address: curve });
    c.action(UiAction::Change {
        address: curve,
        value: ParameterValue::Curve(vec![[0.0, 0.2], [0.4, 0.8], [1.0, 0.9]]),
    });
    c.action(UiAction::Escape);
    assert_eq!(value(&c, curve), original);
    assert!(c.session().history().active_gesture.is_none());
    let before = c.session().history().undo_entries;
    c.action(UiAction::Change {
        address: curve,
        value: ParameterValue::Curve(vec![[0.0, 0.1], [0.5, 0.7], [1.0, 0.8]]),
    });
    assert_eq!(c.session().history().undo_entries, before + 1);
    c.action(UiAction::UndoShortcut);
    assert_eq!(value(&c, curve), original);
}

#[test]
fn picker_rejects_out_of_order_edit_selection_import_and_lock_tokens() {
    let mut c = material();
    select(&mut c, MapKind::Metallic);
    let a = address(&c, ParameterId::PickedColor);
    let old = value(&c, a);
    let first = c.request_color_pick(a, [0.1, 0.2]).unwrap();
    let second = c.request_color_pick(a, [0.8, 0.9]).unwrap();
    assert!(!c.complete_color_pick(&first, [1.0, 0.0, 0.0]));
    assert_eq!(value(&c, a), old);
    assert!(c.complete_color_pick(&second, [0.25, 0.5, 0.75]));
    assert_eq!(value(&c, a), ParameterValue::Color([0.25, 0.5, 0.75]));
    let token = c.request_color_pick(a, [0.5, 0.5]).unwrap();
    c.action(UiAction::Change {
        address: a,
        value: ParameterValue::Color([0.1; 3]),
    });
    assert!(!c.complete_color_pick(&token, [1.0; 3]));
    let token = c.request_color_pick(a, [0.5, 0.5]).unwrap();
    select(&mut c, MapKind::Normal);
    assert!(!c.complete_color_pick(&token, [1.0; 3]));
    select(&mut c, MapKind::Metallic);
    let token = c.request_color_pick(a, [0.5, 0.5]).unwrap();
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Metallic,
        locked: true,
    }));
    assert!(!c.complete_color_pick(&token, [1.0; 3]));
    assert!(c.request_color_pick(a, [0.5, 0.5]).is_err());
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Metallic,
        locked: false,
    }));
    let token = c.request_color_pick(a, [0.5, 0.5]).unwrap();
    c.open_image("replacement.png".into());
    let Effect::LoadImage { token: import, .. } = c.take_effects().remove(0) else {
        panic!("import")
    };
    c.complete_import(
        import,
        SourceDescriptor {
            content_hash: [8; 32],
            extent: Extent {
                width: 64,
                height: 32,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    assert!(!c.complete_color_pick(&token, [1.0; 3]));
}

#[test]
fn pointer_stroke_has_one_history_entry_escape_cancels_and_focus_loss_commits() {
    let mut c = material();
    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
        seamless: true,
        painted_mask: true,
    }));
    select(&mut c, MapKind::Metallic);
    c.set_brush_enabled(true);
    let before = c.session().history().undo_entries;
    c.paint_at([0.1, 0.2], PaintPhase::Begin);
    for uv in [[0.2, 0.3], [0.5, 0.5], [0.9, 0.7]] {
        c.paint_at(uv, PaintPhase::Append);
    }
    assert!(c.session().history().active_stroke.is_some());
    c.paint_at([0.9, 0.7], PaintPhase::End);
    assert_eq!(c.session().history().undo_entries, before + 1);
    c.action(UiAction::UndoShortcut);
    assert_eq!(mask_len(&c), 0);
    c.action(UiAction::RedoShortcut);
    assert_eq!(mask_len(&c), 1);
    c.paint_at([0.3, 0.4], PaintPhase::Begin);
    c.action(UiAction::Escape);
    assert_eq!(mask_len(&c), 1);
    c.paint_at([0.2, 0.3], PaintPhase::Begin);
    c.action(UiAction::FocusLost);
    assert_eq!(mask_len(&c), 2);
    assert!(c.session().history().active_stroke.is_none());
    let before = c.session().history().undo_entries;
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Metallic,
        locked: true,
    }));
    c.paint_at([0.2, 0.3], PaintPhase::Begin);
    assert!(c.session().history().active_stroke.is_none());
    assert_eq!(mask_len(&c), 2);
    assert_eq!(c.session().history().undo_entries, before + 1);
}
fn mask_len(c: &Controls) -> usize {
    let s = c.session().snapshot().unwrap();
    let n = s.graph.material_outputs().unwrap()[&MapKind::Mask].node;
    let MaterialNodeParams::PaintMask { strokes, .. } = &s.graph.material_node(n).unwrap().params
    else {
        panic!("mask")
    };
    strokes.len()
}
#[test]
fn inspector_diagnostic_is_known_and_unknown_fails() {
    use texelith_ui_egui::startup::Config;
    assert!(
        Config::parse(["--mvp-smoke", "inspectors", "--report", "out.json"].map(str::to_owned))
            .is_ok()
    );
    assert!(
        Config::parse(["--mvp-smoke", "unknown", "--report", "out.json"].map(str::to_owned))
            .is_err()
    );
}

fn ui_frame(
    c: &mut Controls,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    focused: bool,
    brush: bool,
) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            focused,
            events,
            ..Default::default()
        },
        |ui| {
            c.show(ui);
            texelith_ui_egui::workspace::center_slot(ui, |ui| {
                if brush {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(400.0, 240.0), egui::Sense::hover());
                    c.viewport_gestures(ui, rect);
                }
            });
        },
    );
    output.textures_delta.clear();
}
fn pointer(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}
#[test]
fn native_event_shapes_drag_the_real_descriptor_slider_once() {
    let mut c = material();
    select(&mut c, MapKind::Normal);
    let ctx = egui::Context::default();
    for _ in 0..3 {
        ui_frame(&mut c, &ctx, vec![], true, false);
    }
    let a = address(&c, ParameterId::Strength);
    let original = value(&c, a);
    let rect = c.inspector_parameter_rects(a)[0];
    let start = egui::pos2(rect.left() + 15.0, rect.center().y);
    let end = egui::pos2(rect.left() + 90.0, rect.center().y);
    let before = c.session().history().undo_entries;
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(start), pointer(start, true)],
        true,
        false,
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(end)],
        true,
        false,
    );
    ui_frame(&mut c, &ctx, vec![pointer(end, false)], true, false);
    assert_ne!(value(&c, a), original);
    assert!(c.session().history().active_gesture.is_none());
    assert_eq!(c.session().history().undo_entries, before + 1);
}
#[test]
fn viewport_pointer_release_outside_image_commits_and_escape_does_not_restart_held_pointer() {
    let mut c = material();
    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
        seamless: true,
        painted_mask: true,
    }));
    select(&mut c, MapKind::Mask);
    c.set_brush_enabled(true);
    let ctx = egui::Context::default();
    for _ in 0..3 {
        ui_frame(&mut c, &ctx, vec![], true, true);
    }
    // The map chips now sit above the canvas; start inside its 400x240 fixture.
    let maps = c.workspace_layout().maps.unwrap();
    let start = maps.left_bottom() + egui::vec2(75., 65.);
    let end = start + egui::vec2(100.0, 40.0);
    let before = c.session().history().undo_entries;
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(start), pointer(start, true)],
        true,
        true,
    );
    assert!(
        c.session().history().active_stroke.is_some(),
        "press inside image begins stroke"
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(end)],
        true,
        true,
    );
    let outside = egui::pos2(1250.0, 800.0);
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(outside), pointer(outside, false)],
        true,
        true,
    );
    assert!(c.session().history().active_stroke.is_none());
    assert_eq!(c.session().history().undo_entries, before + 1);
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(start), pointer(start, true)],
        true,
        true,
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        true,
        true,
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(end)],
        true,
        true,
    );
    assert!(c.session().history().active_stroke.is_none());
    assert_eq!(c.session().history().undo_entries, before + 1);
    ui_frame(&mut c, &ctx, vec![pointer(end, false)], true, true);
}
#[test]
fn original_uv_ignores_letterbox_and_maps_each_tile_in_source_coordinates() {
    use texelith_ui_egui::inspectors::source_uv;
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 40.0), egui::vec2(400.0, 200.0));
    assert_eq!(source_uv(rect, egui::pos2(10.0, 40.0), 1), None);
    assert_eq!(
        source_uv(rect, egui::pos2(220.0, 140.0), 1),
        Some([0.5, 0.5])
    );
    assert_eq!(
        source_uv(rect, egui::pos2(320.0, 190.0), 2),
        Some([0.5, 0.5])
    );
    let mut c = material();
    select(&mut c, MapKind::Metallic);
    let a = address(&c, ParameterId::PickedColor);
    let token = c.request_color_pick(a, [0.5, 0.5]).unwrap();
    select(&mut c, MapKind::Normal);
    select(&mut c, MapKind::Metallic);
    assert!(!c.complete_color_pick(&token, [1.0; 3]));
}

#[test]
fn curve_numeric_widgets_preserve_order_and_coalesce_drag() {
    let mut c = material();
    select(&mut c, MapKind::Roughness);
    c.set_inspector_advanced(true);
    let ctx = egui::Context::default();
    for _ in 0..4 {
        ui_frame(&mut c, &ctx, vec![], true, false);
    }
    let a = address(&c, ParameterId::Curve);
    let rects = c.inspector_parameter_rects(a);
    assert!(rects.len() >= 4, "curve has editable knot coordinates");
    let y = rects[1].center();
    let before = c.session().history().undo_entries;
    let original = value(&c, a);
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(y), pointer(y, true)],
        true,
        false,
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![egui::Event::PointerMoved(y + egui::vec2(35.0, 0.0))],
        true,
        false,
    );
    ui_frame(
        &mut c,
        &ctx,
        vec![pointer(y + egui::vec2(35.0, 0.0), false)],
        true,
        false,
    );
    assert_ne!(value(&c, a), original);
    assert_eq!(c.session().history().undo_entries, before + 1);
    let ParameterValue::Curve(knots) = value(&c, a) else {
        panic!("curve")
    };
    assert!(knots.windows(2).all(|w| w[0][0] < w[1][0]));
    let valid = value(&c, a);
    c.action(UiAction::Change {
        address: a,
        value: ParameterValue::Curve(vec![[0.5, 0.0], [0.5, 1.0]]),
    });
    assert_eq!(
        value(&c, a),
        valid,
        "invalid unordered knots never mutate the material"
    );
}
