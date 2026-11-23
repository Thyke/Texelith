use super::*;
use crate::{command_adapter::UiAction, controls::Controls};
use texelith_core::*;

fn material(mode: ViewMode, split: bool) -> Controls {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        panic!("import");
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [4; 32],
            extent: Extent {
                width: 2048,
                height: 1024,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )
    .unwrap();
    c.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Normal,
    }));
    c.action(UiAction::Request(CommandRequest::SetMode { mode }));
    c.preview.split_view = split;
    c
}

fn frame(
    c: &mut Controls,
    ctx: &egui::Context,
    size: [f32; 2],
    events: Vec<egui::Event>,
) -> egui::Rect {
    let mut canvas = egui::Rect::NOTHING;
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
            focused: true,
            events,
            ..Default::default()
        },
        |ui| {
            c.show(ui);
            let center = crate::workspace::center_slot(ui, |ui| {
                c.viewport_controls(ui);
                // This is the same remaining space consumed by the native viewport.
                canvas = ui
                    .allocate_exact_size(ui.available_size(), egui::Sense::hover())
                    .0;
            });
            c.workspace.layout.center = Some(center.response.rect);
        },
    );
    output.textures_delta.clear();
    canvas
}

fn pointer(pos: egui::Pos2, pressed: bool) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            pressed,
            button: egui::PointerButton::Primary,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

fn strength(c: &Controls) -> (ParameterAddress, f32) {
    let (address, _) = c
        .inspectors
        .parameter_rects
        .iter()
        .find(|(a, _)| a.parameter == ParameterId::Strength)
        .unwrap();
    let ParameterValue::Float(value) = c
        .session
        .snapshot()
        .unwrap()
        .graph
        .material_node(address.node)
        .unwrap()
        .params
        .parameter(address.parameter)
        .unwrap()
    else {
        panic!("float strength");
    };
    (*address, value)
}

#[test]
fn inspector_slider_drag_keeps_canvas_and_surrounding_panels_identical_every_frame() {
    for size in [[1600., 1000.], [1280., 800.], [1100., 700.]] {
        for (mode, split) in [
            (ViewMode::TwoD, false),
            (ViewMode::ThreeD, false),
            (ViewMode::ThreeD, true),
        ] {
            for export_options in [false, true] {
                let mut c = material(mode, split);
                c.workspace.export_options_open = export_options;
                let ctx = egui::Context::default();
                theme(&ctx);
                for _ in 0..5 {
                    frame(&mut c, &ctx, size, vec![]);
                }
                let canvas = frame(&mut c, &ctx, size, vec![]);
                let before = c.workspace_layout().clone();
                assert!(
                    before.inspector.is_some(),
                    "inspector defaults open at {size:?}"
                );
                let (address, _) = strength(&c);
                let row = c
                    .inspectors
                    .parameter_rects
                    .iter()
                    .find(|(a, _)| *a == address)
                    .unwrap()
                    .1
                    .clone();
                let slider = row[0];
                let pos = egui::pos2(slider.right() - 2., slider.center().y);
                assert_eq!(frame(&mut c, &ctx, size, pointer(pos, true)), canvas);
                let (_, high) = strength(&c);
                let history = c.session.history().undo_entries;
                // Cross every slider position, including the change from two digits to one.
                for step in 0..=100 {
                    let pos = egui::pos2(
                        egui::lerp(slider.x_range(), 1. - step as f32 / 100.),
                        slider.center().y,
                    );
                    let actual = frame(&mut c, &ctx, size, vec![egui::Event::PointerMoved(pos)]);
                    assert_eq!(
                        actual, canvas,
                        "{size:?} {mode:?}, split={split}, options={export_options}, step={step}"
                    );
                    let layout = c.workspace_layout();
                    assert_eq!(layout.center, before.center);
                    assert_eq!(layout.maps, before.maps);
                    assert_eq!(layout.inspector, before.inspector);
                    assert_eq!(layout.export, before.export);
                    let current = &c
                        .inspectors
                        .parameter_rects
                        .iter()
                        .find(|(a, _)| *a == address)
                        .unwrap()
                        .1;
                    assert_eq!(current, &row, "slider and numeric editor must stay fixed");
                }
                let pos = egui::pos2(slider.left(), slider.center().y);
                assert_eq!(frame(&mut c, &ctx, size, pointer(pos, false)), canvas);
                let (_, low) = strength(&c);
                assert!(
                    high >= 0.074 && low <= 0.0001,
                    "real pointer drag must span the range: {low}..{high}"
                );
                assert_eq!(c.session.history().undo_entries, history + 1);
                assert!(c.session.history().active_gesture.is_none());
            }
        }
    }
}

#[test]
fn inspector_defaults_and_single_row_bars_fit_near_the_scaling_boundary() {
    for turkish in [false, true] {
        crate::material_ui::labels::set_turkish(turkish);
        for size in [
            [1600., 1000.],
            [1280., 800.],
            [1279., 800.],
            [1100., 700.],
            [1099., 700.],
        ] {
            let mut c = material(ViewMode::ThreeD, true);
            let ctx = egui::Context::default();
            theme(&ctx);
            ctx.set_pixels_per_point(1.25);
            for _ in 0..5 {
                frame(&mut c, &ctx, size, vec![]);
            }
            let layout = c.workspace_layout();
            assert_eq!(layout.inspector.is_some(), size[0] >= 1000.);
            let toolbar = layout.viewport_toolbar.unwrap();
            let button = layout.mode_buttons[0].unwrap();
            assert!(
                toolbar.height() <= button.height() + 0.5,
                "one toolbar row: {size:?}, tr={turkish}, {toolbar:?}"
            );
            assert!(
                layout.center.unwrap().expand(0.5).contains_rect(toolbar),
                "toolbar fits: {size:?}, tr={turkish}, {toolbar:?}"
            );
            let bar = layout.view_bar.unwrap();
            assert!(
                bar.top() <= layout.compare.unwrap().top()
                    && bar.bottom() >= layout.compare.unwrap().bottom(),
                "A/B stays in the view bar"
            );
            assert!(
                bar.height() <= 44.,
                "one view bar row: {size:?}, tr={turkish}, {bar:?}"
            );
            let pos = layout.inspector_toggle.unwrap().center();
            for pressed in [true, false] {
                frame(&mut c, &ctx, size, pointer(pos, pressed));
            }
            frame(&mut c, &ctx, size, vec![]);
            let expected = size[0] < 1000.;
            assert_eq!(
                c.workspace.inspector_open,
                Some(expected),
                "first click must toggle the default"
            );
            frame(&mut c, &ctx, [1600., 1000.], vec![]);
            assert_eq!(
                c.workspace_layout().inspector.is_some(),
                expected,
                "remember explicit toggle after resizing"
            );
        }
    }
    crate::material_ui::labels::set_turkish(false);
}

#[test]
fn side_panel_widths_only_follow_resize_handles_and_are_remembered() {
    let mut c = material(ViewMode::ThreeD, true);
    let ctx = egui::Context::default();
    theme(&ctx);
    let size = [1600., 1000.];
    for _ in 0..5 {
        frame(&mut c, &ctx, size, vec![]);
    }
    let rect = c.workspace_layout().inspector.unwrap();
    let maps = c.workspace_layout().maps.unwrap();
    let start = egui::pos2(rect.left(), rect.center().y);
    let end = start + egui::vec2(24., 0.);
    frame(&mut c, &ctx, size, pointer(start, true));
    frame(&mut c, &ctx, size, vec![egui::Event::PointerMoved(end)]);
    frame(&mut c, &ctx, size, pointer(end, false));
    frame(&mut c, &ctx, size, vec![]);
    let resized = c.workspace_layout().inspector.unwrap();
    assert!(
        (resized.width() - rect.width() + 24.).abs() < 1.,
        "user resize: {rect:?} -> {resized:?}"
    );
    for map in crate::material_ui::MAPS {
        c.action(UiAction::Request(CommandRequest::SelectMap { map }));
        for _ in 0..3 {
            frame(&mut c, &ctx, size, vec![]);
        }
        assert_eq!(
            c.workspace_layout().inspector.unwrap().width(),
            resized.width(),
            "map selection preserves width"
        );
        assert_eq!(
            c.workspace_layout().maps.unwrap().height(),
            maps.height(),
            "horizontal chips keep a fixed height"
        );
    }
    c.workspace.inspector_open = Some(false);
    frame(&mut c, &ctx, size, vec![]);
    c.workspace.inspector_open = Some(true);
    frame(&mut c, &ctx, size, vec![]);
    assert_eq!(
        c.workspace_layout().inspector.unwrap().width(),
        resized.width()
    );
    frame(&mut c, &ctx, [1280., 800.], vec![]);
    assert!(
        c.workspace_layout().inspector.unwrap().width() <= 334.,
        "right panel narrows with the window"
    );
}

#[test]
fn text_palette_is_near_white_and_secondary_text_meets_wcag_contrast() {
    fn luminance(color: Color32) -> f64 {
        let [r, g, b, _] = color.to_array();
        let linear = |v: u8| {
            let v = v as f64 / 255.;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    }
    let ctx = egui::Context::default();
    theme(&ctx);
    let visuals = &ctx.style_of(egui::Theme::Dark).visuals;
    let primary = visuals.text_color();
    assert!((237..=245).contains(&primary.r()));
    assert_eq!(primary.r(), primary.g());
    assert_eq!(primary.g(), primary.b());
    assert!(luminance(visuals.strong_text_color()) > luminance(primary));
    let surfaces = [
        visuals.panel_fill,
        visuals.window_fill,
        visuals.extreme_bg_color,
        visuals.faint_bg_color,
        visuals.widgets.inactive.bg_fill,
        visuals.widgets.hovered.bg_fill,
        visuals.widgets.active.bg_fill,
        visuals.widgets.open.bg_fill,
        visuals.selection.bg_fill,
    ];
    for background in surfaces {
        for foreground in [
            primary,
            visuals.strong_text_color(),
            visuals.weak_text_color(),
            visuals.widgets.inactive.text_color(),
            visuals.widgets.hovered.text_color(),
            visuals.widgets.active.text_color(),
            visuals.widgets.open.text_color(),
        ] {
            let contrast = (luminance(foreground) + 0.05) / (luminance(background) + 0.05);
            assert!(
                contrast >= 4.5,
                "{foreground:?} on {background:?}: {contrast:.2}:1"
            );
        }
    }
}

#[test]
fn review_controls_fit_at_native_150_percent_without_horizontal_scrolling() {
    for turkish in [false, true] {
        crate::material_ui::labels::set_turkish(turkish);
        for size in [[1600., 1000.], [1280., 800.], [1600. / 1.5, 1000. / 1.5]] {
            let mut c = material(ViewMode::ThreeD, true);
            let ctx = egui::Context::default();
            theme(&ctx);
            for _ in 0..5 {
                frame(&mut c, &ctx, size, vec![]);
            }
            let before = c.workspace_layout().clone();
            let maps = before.maps.unwrap();
            for card in before.cards.values() {
                assert!(
                    maps.contains_rect(card.select),
                    "{size:?}, tr={turkish}: {:?} outside {maps:?}",
                    card.select
                );
            }
            let toolbar = before.viewport_toolbar.unwrap().expand(0.5);
            for rect in [before.heatmap.unwrap(), before.heatmap_info.unwrap()] {
                assert!(
                    toolbar.contains_rect(rect),
                    "{size:?}, tr={turkish}: {rect:?} outside {toolbar:?}"
                );
            }
            let inspector = before.inspector.unwrap().shrink(1.);
            for rect in [
                before.export_material.unwrap(),
                before.destination.unwrap(),
                before.export_options.unwrap(),
            ] {
                assert!(
                    inspector.contains_rect(rect),
                    "{size:?}, tr={turkish}: {rect:?} outside {inspector:?}"
                );
            }
            c.inspectors.advanced = true;
            c.workspace.export_options_open = true;
            for _ in 0..5 {
                frame(&mut c, &ctx, size, vec![]);
            }
            assert_eq!(
                c.workspace_layout().export_material,
                before.export_material,
                "expanding options must preserve the primary action"
            );
            assert!(
                !c.workspace_layout()
                    .export_options_area
                    .unwrap()
                    .intersects(before.export_material.unwrap()),
                "advanced options must leave the pinned Export button unobstructed"
            );
            c.set_export_status(
                Some(crate::export_jobs::JobStatus {
                    job_id: 154,
                    stage: texelith_gpu::ExportStage::Completed,
                    output_name: "Normal".into(),
                    completed_items: 1,
                    total_items: 1,
                    destination: "completed-material".into(),
                    error: None,
                    cancel_requested: false,
                }),
                false,
            );
            frame(&mut c, &ctx, size, vec![]);
            assert!(
                inspector.contains_rect(c.workspace_layout().open_export_folder.unwrap()),
                "completion folder link remains visible even with advanced options open"
            );
            let pos = before.heatmap.unwrap().center();
            for pressed in [true, false] {
                frame(&mut c, &ctx, size, pointer(pos, pressed));
            }
            assert!(
                c.take_effects()
                    .iter()
                    .any(|effect| matches!(effect, Effect::InspectSeams { enabled: true })),
                "visible heatmap control must invoke seam inspection"
            );
        }
    }
    crate::material_ui::labels::set_turkish(false);
}
