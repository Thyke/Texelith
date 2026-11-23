//! Original Texelith line icons, distributed under the workspace MIT license.
//! Geometry uses stack arrays; no font, image decoding or per-frame icon allocation.
use egui::{Color32, Response, Sense, Stroke, Ui, Vec2, pos2, vec2};

pub const AMBER: Color32 = Color32::from_rgb(245, 165, 36);

pub(crate) fn logo(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(26., 30.), Sense::hover());
    let center = rect.center();
    let points = [
        [0., -12.],
        [11., -6.],
        [11., 6.],
        [0., 12.],
        [-11., 6.],
        [-11., -6.],
    ]
    .map(|p| center + Vec2::from(p));
    ui.painter().add(egui::Shape::convex_polygon(
        points.to_vec(),
        AMBER,
        Stroke::NONE,
    ));
    for point in [points[0], points[2], points[4]] {
        ui.painter()
            .line_segment([center, point], Stroke::new(2., ui.visuals().panel_fill));
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Shared original icon vocabulary; menus currently use text.
pub(crate) enum Icon {
    Image,
    Folder,
    Save,
    Undo,
    Redo,
    Search,
    Settings,
    Eye,
    Lock,
    Pan,
    Fit,
    Orbit,
    ZoomIn,
    ZoomOut,
    Light,
    Duplicate,
    Delete,
    More,
    Export,
    Batch,
    Brush,
    Clone,
    Eraser,
    Decal,
}

pub(crate) fn button(
    ui: &mut Ui,
    icon: Icon,
    label: &str,
    show_label: bool,
    active: bool,
) -> Response {
    sized_button(ui, icon, label, show_label, active, 30.)
}

pub(crate) fn small_button(ui: &mut Ui, icon: Icon, label: &str, active: bool) -> Response {
    sized_button(ui, icon, label, false, active, 22.)
}

pub(crate) fn tool_button(ui: &mut Ui, icon: Icon, label: &str) -> Response {
    sized_button(ui, icon, label, false, false, 44.)
}

fn sized_button(
    ui: &mut Ui,
    icon: Icon,
    label: &str,
    show_label: bool,
    active: bool,
    size: f32,
) -> Response {
    let galley = show_label.then(|| {
        ui.painter().layout_no_wrap(
            label.to_owned(),
            egui::FontId::proportional(13.0),
            ui.visuals().text_color(),
        )
    });
    let width = size + galley.as_ref().map_or(0.0, |g| g.size().x + 4.0);
    let (rect, response) = ui.allocate_exact_size(vec2(width, size), Sense::click());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, ui.is_enabled(), label));
    let visuals = ui.style().interact_selectable(&response, active);
    ui.painter().rect(
        rect,
        6,
        visuals.bg_fill,
        visuals.bg_stroke,
        egui::StrokeKind::Inside,
    );
    if response.has_focus() {
        ui.painter()
            .rect_stroke(rect, 6, Stroke::new(1.5, AMBER), egui::StrokeKind::Inside);
    }
    let scale = if size >= 44. { 1.5 } else { 1. };
    let origin = rect.min + Vec2::splat((size - 16. * scale) / 2.);
    let stroke = Stroke::new(
        1.5,
        if active {
            AMBER
        } else {
            visuals.fg_stroke.color
        },
    );
    let line = |a: [f32; 2], b: [f32; 2]| {
        ui.painter().line_segment(
            [
                origin + Vec2::from(a) * scale,
                origin + Vec2::from(b) * scale,
            ],
            stroke,
        );
    };
    let box_at = |a: [f32; 2], b: [f32; 2]| {
        ui.painter().rect_stroke(
            egui::Rect::from_min_max(
                origin + Vec2::from(a) * scale,
                origin + Vec2::from(b) * scale,
            ),
            2,
            stroke,
            egui::StrokeKind::Inside,
        );
    };
    match icon {
        Icon::Image | Icon::Decal => {
            box_at([0., 1.], [16., 15.]);
            line([1., 13.], [6., 8.]);
            line([6., 8.], [10., 12.]);
            line([10., 12.], [14., 7.]);
            ui.painter()
                .circle_filled(origin + vec2(5., 5.), 1.5, stroke.color);
        }
        Icon::Folder => {
            box_at([0., 4.], [16., 15.]);
            line([0., 4.], [0., 1.]);
            line([0., 1.], [6., 1.]);
            line([6., 1.], [9., 4.]);
        }
        Icon::Save => {
            box_at([1., 0.], [15., 16.]);
            box_at([4., 0.], [12., 6.]);
            box_at([4., 10.], [12., 16.]);
        }
        Icon::Undo | Icon::Redo => {
            let flip = matches!(icon, Icon::Redo);
            let x = |v| if flip { 16. - v } else { v };
            line([x(1.), 5.], [x(11.), 5.]);
            line([x(11.), 5.], [x(15.), 9.]);
            line([x(15.), 9.], [x(15.), 14.]);
            line([x(1.), 5.], [x(5.), 1.]);
            line([x(1.), 5.], [x(5.), 9.]);
        }
        Icon::ZoomIn | Icon::ZoomOut | Icon::Search => {
            ui.painter()
                .circle_stroke(origin + vec2(6., 6.), 5., stroke);
            line([10., 10.], [16., 16.]);
            if !matches!(icon, Icon::Search) {
                line([3., 6.], [9., 6.]);
            }
            if matches!(icon, Icon::ZoomIn) {
                line([6., 3.], [6., 9.]);
            }
        }
        Icon::Settings | Icon::Orbit => {
            ui.painter()
                .circle_stroke(origin + vec2(8., 8.), 6., stroke);
            if matches!(icon, Icon::Settings) {
                ui.painter()
                    .circle_stroke(origin + vec2(8., 8.), 2., stroke);
            } else {
                line([2., 2.], [2., 7.]);
                line([2., 2.], [7., 2.]);
            }
        }
        Icon::Eye => {
            line([0., 8.], [5., 3.]);
            line([5., 3.], [11., 3.]);
            line([11., 3.], [16., 8.]);
            line([16., 8.], [11., 13.]);
            line([11., 13.], [5., 13.]);
            line([5., 13.], [0., 8.]);
            ui.painter()
                .circle_stroke(origin + vec2(8., 8.), 2.5, stroke);
        }
        Icon::Lock => {
            box_at([2., 7.], [14., 16.]);
            box_at([5., 0.], [11., 10.]);
        }
        Icon::Pan => {
            for (a, b) in [
                ([8., 0.], [8., 16.]),
                ([0., 8.], [16., 8.]),
                ([8., 0.], [5., 3.]),
                ([8., 0.], [11., 3.]),
                ([0., 8.], [3., 5.]),
                ([16., 8.], [13., 11.]),
            ] {
                line(a, b);
            }
        }
        Icon::Fit => {
            for (a, b) in [
                ([0., 6.], [0., 0.]),
                ([0., 0.], [6., 0.]),
                ([10., 0.], [16., 0.]),
                ([16., 0.], [16., 6.]),
                ([16., 10.], [16., 16.]),
                ([16., 16.], [10., 16.]),
                ([6., 16.], [0., 16.]),
                ([0., 16.], [0., 10.]),
            ] {
                line(a, b);
            }
        }
        Icon::Batch => {
            box_at([0., 4.], [12., 16.]);
            line([4., 0.], [16., 0.]);
            line([16., 0.], [16., 12.]);
        }
        Icon::Export => {
            line([8., 12.], [8., 0.]);
            line([8., 0.], [3., 5.]);
            line([8., 0.], [13., 5.]);
            line([0., 8.], [0., 16.]);
            line([0., 16.], [16., 16.]);
            line([16., 16.], [16., 8.]);
        }
        Icon::Brush => {
            line([6., 10.], [14., 1.]);
            line([8., 12.], [16., 3.]);
            line([14., 1.], [16., 3.]);
            line([6., 10.], [2., 12.]);
            line([2., 12.], [0., 16.]);
            line([0., 16.], [6., 15.]);
            line([6., 15.], [8., 12.]);
        }
        Icon::Clone => {
            box_at([1., 12.], [15., 16.]);
            box_at([5., 1.], [11., 8.]);
            line([6., 8.], [4., 12.]);
            line([10., 8.], [12., 12.]);
        }
        Icon::Eraser => {
            for (a, b) in [
                ([1., 10.], [10., 1.]),
                ([10., 1.], [16., 7.]),
                ([16., 7.], [7., 16.]),
                ([7., 16.], [1., 10.]),
                ([5., 6.], [11., 12.]),
            ] {
                line(a, b);
            }
        }
        Icon::Light => {
            ui.painter()
                .circle_stroke(origin + vec2(8., 8.), 4., stroke);
            for (a, b) in [
                ([8., 0.], [8., 2.]),
                ([8., 14.], [8., 16.]),
                ([0., 8.], [2., 8.]),
                ([14., 8.], [16., 8.]),
                ([2., 2.], [3., 3.]),
                ([13., 13.], [14., 14.]),
                ([2., 14.], [3., 13.]),
                ([13., 3.], [14., 2.]),
            ] {
                line(a, b);
            }
        }
        Icon::Duplicate => {
            box_at([0., 4.], [12., 16.]);
            box_at([4., 0.], [16., 12.]);
        }
        Icon::Delete => {
            box_at([3., 4.], [13., 16.]);
            line([1., 3.], [15., 3.]);
            line([6., 0.], [10., 0.]);
            line([6., 7.], [6., 13.]);
            line([10., 7.], [10., 13.]);
        }
        Icon::More => {
            for y in [2., 8., 14.] {
                ui.painter()
                    .circle_filled(origin + vec2(8., y), 1.2, stroke.color);
            }
        }
    }
    if let Some(galley) = galley {
        ui.painter().galley(
            pos2(rect.left() + 28., rect.center().y - galley.size().y / 2.),
            galley,
            visuals.fg_stroke.color,
        );
    }
    response
        .on_hover_text(label)
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

pub(crate) fn theme(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.spacing.item_spacing = vec2(8., 8.);
        style.spacing.button_padding = vec2(10., 6.);
        style.spacing.interact_size.y = 28.;
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = Color32::from_rgb(31, 32, 33);
        style.visuals.window_fill = Color32::from_rgb(36, 37, 38);
        style.visuals.extreme_bg_color = Color32::from_rgb(22, 23, 24);
        style.visuals.faint_bg_color = Color32::from_rgb(42, 43, 44);
        style.visuals.selection.bg_fill = Color32::from_rgb(87, 64, 27);
        style.visuals.selection.stroke = Stroke::new(1., AMBER);
        style.visuals.weak_text_color = Some(Color32::from_gray(184));
        for widget in [
            &mut style.visuals.widgets.inactive,
            &mut style.visuals.widgets.hovered,
            &mut style.visuals.widgets.active,
            &mut style.visuals.widgets.noninteractive,
            &mut style.visuals.widgets.open,
        ] {
            widget.corner_radius = 6.into();
            widget.fg_stroke.color = Color32::from_gray(237);
        }
        style.visuals.widgets.hovered.fg_stroke.color = Color32::from_gray(245);
        style.visuals.widgets.active.fg_stroke.color = Color32::WHITE;
        // Keep pressed controls dark enough for near-white text. Amber remains
        // the selection stroke and the explicitly dark-text Export action.
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(87, 64, 27);
    });
}

#[cfg(test)]
#[path = "stability_tests.rs"]
mod stability_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{command_adapter::UiAction, controls::Controls};
    use texelith_core::*;

    fn material() -> Controls {
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
        c
    }

    fn frame(c: &mut Controls, ctx: &egui::Context, size: [f32; 2], events: Vec<egui::Event>) {
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
                    ui.allocate_space(ui.available_size());
                });
                c.workspace.layout.center = Some(center.response.rect);
            },
        );
        output.textures_delta.clear();
    }

    #[test]
    fn imports_open_in_split_and_export_selection_controls_captured_channels() {
        let mut c = material();
        assert!(c.preview.split_view);
        assert_eq!(c.session.view_state().mode, ViewMode::ThreeD);
        let initial = c.session.snapshot().unwrap();
        c.workspace.export_maps = Some(
            crate::material_ui::MAPS
                .into_iter()
                .filter(|m| *m != MapKind::Ao)
                .collect(),
        );
        let plan = c.packing_plan().unwrap();
        assert_eq!(c.session.snapshot().unwrap().generation, initial.generation);
        assert!(plan.outputs.iter().flat_map(|o| &o.channels).all(|channel| !matches!(channel.source, ChannelSource::Map(r, _) if r.port == OutputPort::Ao)));
        let orm = plan.outputs.iter().find(|o| o.channels.iter().any(|channel| matches!(channel.source, ChannelSource::Map(r, _) if r.port == OutputPort::Roughness))).unwrap();
        assert_eq!(orm.channels[0].source, ChannelSource::Constant(1.));
        assert!(plan.outputs.iter().any(|o| o.channels.iter().any(|channel| matches!(channel.source, ChannelSource::Map(r, _) if r.port == OutputPort::Edge))));
        c.request_material_export().unwrap();
        c.workspace.export_maps = Some(Default::default());
        let effects = c.take_effects();
        let Effect::Export { plan: captured, .. } = &effects[0] else {
            panic!("material export");
        };
        assert_eq!(captured, &plan);
        assert!(
            c.request_material_export().is_err(),
            "empty map selection must reject export"
        );
        assert!(c.take_effects().is_empty());
    }

    #[test]
    fn studio_geometry_fits_all_sizes_states_and_locales() {
        for turkish in [false, true] {
            crate::material_ui::labels::set_turkish(turkish);
            for size in [[1600., 1000.], [1280., 800.], [960., 600.]] {
                for state in 0..5 {
                    let mut c = if state == 0 {
                        Controls::new_material()
                    } else {
                        material()
                    };
                    if state != 0 {
                        c.action(UiAction::Request(CommandRequest::SelectMap {
                            map: MapKind::Normal,
                        }));
                    }
                    if state >= 2 {
                        c.action(UiAction::Request(CommandRequest::SetMode {
                            mode: ViewMode::ThreeD,
                        }));
                    }
                    c.preview.split_view = state >= 3;
                    c.workspace.export_options_open = state == 4;
                    let ctx = egui::Context::default();
                    theme(&ctx);
                    for _ in 0..4 {
                        frame(&mut c, &ctx, size, vec![]);
                    }
                    let layout = c.workspace_layout();
                    let bounds =
                        egui::Rect::from_min_size(egui::Pos2::ZERO, size.into()).expand(0.5);
                    for rect in [
                        layout.header_content,
                        layout.export,
                        layout.export_material,
                        layout.export_options,
                        layout.viewport_toolbar,
                        layout.view_bar,
                        layout.tool_rail,
                        layout.center,
                    ] {
                        assert!(
                            bounds.contains_rect(rect.unwrap()),
                            "{size:?}, state {state}, tr {turkish}: {rect:?}"
                        );
                    }
                    assert!(layout.center.unwrap().width() > 120.);
                    assert!(layout.center.unwrap().height() > 100.);
                    assert!(!layout.center.unwrap().intersects(layout.export.unwrap()));
                    if layout.inspector.is_some() {
                        assert!(
                            layout.export_material.unwrap().bottom()
                                <= layout.export_options.unwrap().top()
                        );
                    }
                    if let Some(inspector) = layout.inspector {
                        assert!(inspector.width() <= 402.);
                        assert!(inspector.left() >= layout.center.unwrap().right());
                        for (_, rects) in &c.inspectors.parameter_rects {
                            for rect in rects {
                                assert!(
                                    rect.left() >= inspector.left()
                                        && rect.right() <= inspector.right(),
                                    "slider row must fit: {rect:?} in {inspector:?}"
                                );
                            }
                        }
                    }
                    if let Some(maps) = layout.maps {
                        assert_eq!(layout.cards.len(), 7);
                        for card in layout.cards.values() {
                            assert!(
                                card.select.top() >= maps.top()
                                    && card.select.bottom() <= maps.bottom(),
                                "{size:?} state {state}: {:?} in {maps:?}",
                                layout.cards
                            );
                            assert!(card.select.contains_rect(card.thumbnail));
                        }
                    }
                }
            }
        }
        crate::material_ui::labels::set_turkish(false);
    }

    #[test]
    fn split_selection_preserves_camera_history_and_ab_captures() {
        let mut c = material();
        c.action(UiAction::Request(CommandRequest::CaptureA {}));
        let capture = c.session.view_state().capture_a;
        let camera = c.session.view_state().camera;
        assert_eq!(
            c.session.view_state().mesh,
            texelith_core::PreviewMesh::PreviewBall
        );
        assert_eq!(camera.yaw, 0.6);
        assert_eq!(camera.pitch, 0.3);
        assert_eq!(
            camera,
            crate::compare::reset_camera(&c.session.view_state().mesh)
        );
        let history = c.session.history().undo_entries;
        let generation = c.session.snapshot().unwrap().generation;
        let ctx = egui::Context::default();
        theme(&ctx);
        for _ in 0..3 {
            frame(&mut c, &ctx, [1280., 800.], vec![]);
        }
        for index in [2, 1, 0] {
            let pos = c.workspace.layout.mode_buttons[index].unwrap().center();
            for pressed in [true, false] {
                frame(
                    &mut c,
                    &ctx,
                    [1280., 800.],
                    vec![
                        egui::Event::PointerMoved(pos),
                        egui::Event::PointerButton {
                            pos,
                            pressed,
                            button: egui::PointerButton::Primary,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
            assert_eq!(c.preview.split_view, index == 2);
            assert_eq!(c.session.view_state().camera, camera);
            assert_eq!(c.session.view_state().capture_a, capture);
            assert_eq!(c.session.history().undo_entries, history);
            assert_eq!(c.session.snapshot().unwrap().generation, generation);
        }
    }

    #[test]
    fn wide_default_inspectors_and_compact_cards_fit_every_map() {
        for size in [[1600., 1000.], [1280., 800.]] {
            let mut c = material();
            let ctx = egui::Context::default();
            theme(&ctx);
            for map in crate::material_ui::MAPS {
                c.action(UiAction::Request(CommandRequest::SelectMap { map }));
                for _ in 0..4 {
                    frame(&mut c, &ctx, size, vec![]);
                }
                let inspector = c
                    .workspace
                    .layout
                    .inspector
                    .expect("wide inspector defaults open");
                let maps = c.workspace.layout.maps.unwrap();
                assert_eq!(c.workspace.layout.cards.len(), 7);
                for card in c.workspace.layout.cards.values() {
                    for rect in [card.thumbnail, card.select] {
                        assert!(
                            rect.top() >= maps.top() && rect.bottom() <= maps.bottom(),
                            "all seven cards must fit: {rect:?} in {maps:?}, cards {:?}",
                            c.workspace.layout.cards
                        );
                    }
                    assert!(card.select.contains_rect(card.thumbnail));
                }
                assert!(
                    !c.inspectors.parameter_rects.is_empty(),
                    "{map:?} key controls"
                );
                for (_, rects) in &c.inspectors.parameter_rects {
                    for rect in rects {
                        assert!(
                            inspector.contains_rect(*rect),
                            "{map:?}: {rect:?} outside {inspector:?}"
                        );
                    }
                }
            }
            let pos = c.workspace.layout.inspector_toggle.unwrap().center();
            for pressed in [true, false] {
                frame(
                    &mut c,
                    &ctx,
                    size,
                    vec![
                        egui::Event::PointerMoved(pos),
                        egui::Event::PointerButton {
                            pos,
                            pressed,
                            button: egui::PointerButton::Primary,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
            frame(&mut c, &ctx, [1600., 1000.], vec![]);
            assert_eq!(c.workspace.inspector_open, Some(false));
            assert!(
                c.workspace.layout.inspector.is_none(),
                "remember explicit close after resizing"
            );
        }
    }

    #[test]
    fn studio_snapshot_flags_require_a_destination_and_layers_are_opt_in() {
        use crate::startup::Config;
        assert!(!Config::default().debug_layers);
        assert!(Config::parse(["--studio-snapshots"].map(str::to_owned)).is_err());
        let config = Config::parse(
            [
                "--ui-snapshot",
                "target/evidence/T-147",
                "--studio-snapshots",
                "--debug-layers",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert!(config.studio_snapshots && config.debug_layers);
        let fixed = Config::parse(
            [
                "--ui-snapshot",
                "out",
                "--studio-snapshots",
                "--studio-snapshot-size",
                "1280x800",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(fixed.studio_snapshot_size, Some([1280, 800]));
        assert_eq!(
            fixed.native_options().unwrap().viewport.inner_size,
            Some(egui::vec2(1280., 800.))
        );
        assert!(Config::parse(["--studio-snapshot-size", "1280x800"].map(str::to_owned)).is_err());
        assert!(
            Config::parse(
                [
                    "--ui-snapshot",
                    "out",
                    "--studio-snapshots",
                    "--studio-snapshot-size",
                    "123x456"
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }
}
