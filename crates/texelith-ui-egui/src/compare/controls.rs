use super::*;
use crate::compare::labels::text;
use crate::compare::{assets::Load, orbit};
use texelith_core::{PreviewMesh, ViewMode};

impl Controls {
    pub(crate) fn viewport_header(&mut self, ui: &mut egui::Ui) {
        for (index, (mode, label, split)) in [
            (ViewMode::TwoD, "2D", false),
            (ViewMode::ThreeD, "3D", false),
            (ViewMode::ThreeD, "Split", true),
        ]
        .into_iter()
        .enumerate()
        {
            let response = ui.add_enabled(
                self.session.snapshot().is_some(),
                egui::Button::new(if split {
                    crate::workspace::labels::text("split")
                } else {
                    label.into()
                })
                .selected(
                    self.session.view_state().mode == mode
                        && (mode == ViewMode::TwoD || self.preview.split_view == split),
                ),
            );
            self.workspace.layout.mode_buttons[index] = Some(response.rect);
            if response.clicked() {
                self.preview.split_view = split;
                self.preview.fit_pending = mode == ViewMode::ThreeD;
                self.preview.revision += 1;
                self.action(UiAction::Request(CommandRequest::SetMode { mode }));
            }
        }
    }
    pub(crate) fn viewport_controls(&mut self, ui: &mut egui::Ui) {
        use crate::workspace::{
            labels::text as label,
            studio::{self, Icon},
        };
        let view_bar = egui::Panel::bottom("studio-view-bar")
            .exact_size(44.)
            .show(ui, |ui| {
                egui::ScrollArea::horizontal()
                    .id_salt("view-bar-scroll")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.;
                            if studio::button(
                                ui,
                                Icon::Pan,
                                &label("pan"),
                                false,
                                !self.inspectors.painting,
                            )
                            .clicked()
                            {
                                self.set_brush_enabled(false);
                            }
                            if studio::button(ui, Icon::Fit, &label("fit"), false, false).clicked()
                            {
                                self.fit_view();
                            }
                            let old_zoom = self.preview.zoom;
                            if studio::button(ui, Icon::ZoomIn, &label("zoom-in"), false, false)
                                .clicked()
                            {
                                self.preview.zoom = (self.preview.zoom * 1.25).min(400.);
                            }
                            if studio::button(ui, Icon::ZoomOut, &label("zoom-out"), false, false)
                                .clicked()
                            {
                                self.preview.zoom = (self.preview.zoom / 1.25).max(25.);
                            }
                            egui::ComboBox::from_id_salt("view-zoom")
                                .width(72.)
                                .selected_text(format!("{:.0}%", self.preview.zoom))
                                .show_ui(ui, |ui| {
                                    for zoom in [25., 50., 75., 100., 150., 200., 400.] {
                                        ui.selectable_value(
                                            &mut self.preview.zoom,
                                            zoom,
                                            format!("{zoom:.0}%"),
                                        );
                                    }
                                });
                            if old_zoom != self.preview.zoom
                                && self.session.view_state().mode == ViewMode::ThreeD
                                && !self.preview.split_view
                            {
                                let mut camera = self.session.view_state().camera;
                                camera.distance = (camera.distance * old_zoom / self.preview.zoom)
                                    .clamp(0.3, 20.);
                                self.action(UiAction::Request(CommandRequest::SetCamera {
                                    camera,
                                }));
                            }
                            ui.add_space((ui.available_width() - 390.).max(8.));
                            let orbit =
                                studio::button(ui, Icon::Orbit, &text("orbit"), false, false);
                            egui::Popup::menu(&orbit).show(|ui| {
                                if ui.button(text("reset-camera")).clicked() {
                                    self.reset_viewport_camera();
                                }
                                ui.checkbox(&mut self.preview.turntable, text("turntable"));
                            });
                            let move_view =
                                studio::button(ui, Icon::Pan, &label("move-view"), false, false);
                            egui::Popup::menu(&move_view).show(|ui| {
                                let mut camera = self.session.view_state().camera;
                                let mut changed = false;
                                for (axis, value) in
                                    ["X", "Y", "Z"].into_iter().zip(camera.target.iter_mut())
                                {
                                    changed |= ui
                                        .add(egui::DragValue::new(value).speed(0.01).prefix(axis))
                                        .changed();
                                }
                                if changed {
                                    self.action(UiAction::Request(CommandRequest::SetCamera {
                                        camera,
                                    }));
                                }
                            });
                            egui::ComboBox::from_id_salt("studio-environment")
                                .width(190.)
                                .selected_text(format!(
                                    "HDRI: {}",
                                    if self.preview.hdr_path.is_empty() {
                                        label("procedural-sky")
                                    } else {
                                        std::path::Path::new(&self.preview.hdr_path)
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .into_owned()
                                    }
                                ))
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            self.preview.hdr_path.is_empty(),
                                            label("procedural-sky"),
                                        )
                                        .clicked()
                                    {
                                        self.preview.pending = Some(Load::Sky);
                                        self.preview.hdr_path.clear();
                                    }
                                    if ui.button(text("load-hdr")).clicked() {
                                        self.browse_preview_asset(false);
                                    }
                                });
                            let light =
                                studio::button(ui, Icon::Light, &text("light"), false, false);
                            egui::Popup::menu(&light)
                                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                                .show(|ui| {
                                    let mut light = self.session.view_state().light.clone();
                                    let mut changed = ui
                                        .add(
                                            egui::Slider::new(&mut light.intensity, 0.0..=10.)
                                                .text(text("light")),
                                        )
                                        .changed();
                                    for (axis, value) in
                                        ["X", "Y", "Z"].into_iter().zip(light.direction.iter_mut())
                                    {
                                        changed |= ui
                                            .add(egui::Slider::new(value, -1.0..=1.).text(axis))
                                            .changed();
                                    }
                                    if changed {
                                        self.action(UiAction::Request(CommandRequest::SetLight {
                                            light,
                                        }));
                                    }
                                });
                            self.viewport_compare(ui);
                        });
                    });
            });
        self.workspace.layout.view_bar = Some(view_bar.response.rect);
        let toolbar = egui::ScrollArea::horizontal()
            .id_salt("canvas-toolbar-scroll")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let compact = ui.clip_rect().width() < 900.;
                    ui.spacing_mut().item_spacing.x = if compact { 3. } else { 6. };
                    ui.spacing_mut().button_padding.x = if compact { 4. } else { 7. };
                    if compact {
                        ui.style_mut().override_font_id = Some(egui::FontId::proportional(11.));
                    }
                    self.viewport_header(ui);
                    ui.separator();
                    if !compact {
                        ui.label(label("tiling"));
                    }
                    let mut tiles = self.session.view_state().tiles;
                    for (count, title) in [(1, "1x"), (2, "2x2"), (3, "3x3")] {
                        ui.selectable_value(&mut tiles.count, count, title);
                    }
                    if tiles != self.session.view_state().tiles {
                        self.action(UiAction::Request(CommandRequest::SetTiles { tiles }));
                    }
                    ui.separator();
                    if !compact {
                        ui.label(label("seamless-tiling"));
                    }
                    for (key, method) in [
                        (
                            "make-seamless",
                            texelith_core::SeamlessMethod::OffsetBlend { overlap: 0.25 },
                        ),
                        (
                            "fix-seams",
                            texelith_core::SeamlessMethod::Histogram {
                                base: texelith_core::SeamlessBase::OffsetBlend { overlap: 0.25 },
                                strength: 1.,
                            },
                        ),
                        (
                            "match-edges",
                            texelith_core::SeamlessMethod::Splat {
                                seed: 0,
                                grid: 4,
                                overlap: 0.25,
                            },
                        ),
                    ] {
                        if ui
                            .add_enabled(
                                self.session.snapshot().is_some(),
                                egui::Button::new(label(key)),
                            )
                            .clicked()
                        {
                            self.studio_seamless(method);
                        }
                    }
                    let mut heatmap = self.heatmap;
                    let heatmap_toggle = ui.toggle_value(
                        &mut heatmap,
                        crate::material_ui::labels::text("Seam heatmap"),
                    );
                    self.workspace.layout.heatmap = Some(heatmap_toggle.rect);
                    if heatmap_toggle.changed() {
                        self.action(UiAction::Request(CommandRequest::InspectSeams {
                            enabled: heatmap,
                        }));
                    }
                    let (info, response) =
                        ui.allocate_exact_size(egui::vec2(18., 18.), egui::Sense::hover());
                    ui.painter().circle_stroke(
                        info.center(),
                        7.,
                        egui::Stroke::new(1., ui.visuals().text_color()),
                    );
                    ui.painter().text(
                        info.center(),
                        egui::Align2::CENTER_CENTER,
                        "i",
                        egui::FontId::proportional(12.),
                        ui.visuals().text_color(),
                    );
                    self.workspace.layout.heatmap_info = Some(info);
                    response.on_hover_text(label("heatmap-help"));
                })
                .response
                .rect
            });
        self.workspace.layout.viewport_toolbar = Some(toolbar.inner.intersect(ui.clip_rect()));
        ui.separator();
    }

    fn studio_seamless(&mut self, method: texelith_core::SeamlessMethod) {
        if let Some(snapshot) = self.session.snapshot() {
            let (_, painted_mask) = crate::material_ui::options(&snapshot.graph);
            self.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                seamless: true,
                painted_mask,
            }));
        }
        if let Some(snapshot) = self.session.snapshot() {
            let nodes: Vec<_> = snapshot
                .graph
                .material_nodes()
                .iter()
                .filter(|n| matches!(n.params, texelith_core::MaterialNodeParams::Seamless { .. }))
                .map(|n| n.id)
                .collect();
            for node in nodes {
                self.action(UiAction::Request(CommandRequest::SetParameter {
                    gesture: None,
                    address: texelith_core::ParameterAddress {
                        node,
                        parameter: texelith_core::ParameterId::SeamlessMethod,
                    },
                    value: texelith_core::ParameterValue::SeamlessMethod(method.clone()),
                }));
            }
        }
    }
    fn viewport_compare(&mut self, ui: &mut egui::Ui) {
        let compare = ui.button("A / B").on_hover_text(text("compare"));
        self.workspace.layout.compare = Some(compare.rect);
        egui::Popup::menu(&compare)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                if ui.button(text("capture-a")).clicked() {
                    self.action(UiAction::Request(CommandRequest::CaptureA {}));
                }
                if ui.button(text("capture-b")).clicked() {
                    self.action(UiAction::Request(CommandRequest::CaptureB {}));
                }
                ui.add_enabled_ui(self.session.view_state().capture_a.is_some(), |ui| {
                    if ui
                        .checkbox(&mut self.preview.compare_enabled, text("compare"))
                        .changed()
                    {
                        self.preview.split_view = false;
                        self.preview.revision += 1;
                    }
                });
                let mut fraction = self.session.view_state().split_fraction;
                if ui
                    .add(egui::Slider::new(&mut fraction, 0.0..=1.0))
                    .changed()
                {
                    self.action(UiAction::Request(CommandRequest::SetSplit { fraction }));
                }
            });
    }
    fn fit_view(&mut self) {
        self.preview.zoom = 100.;
        self.preview.pan = egui::Vec2::ZERO;
        if self.session.view_state().mode == ViewMode::ThreeD {
            self.reset_viewport_camera();
        }
    }
    pub(crate) fn viewport_settings(&mut self, ui: &mut egui::Ui) {
        if self.session.snapshot().is_none() {
            return;
        }
        let view = self.session.view_state().clone();
        ui.horizontal_wrapped(|ui| {
            if view.mode == ViewMode::ThreeD {
                let mut mesh = view.mesh.clone();
                egui::ComboBox::from_id_salt("preview-mesh")
                    .selected_text(text(match mesh {
                        PreviewMesh::PreviewBall => "preview-ball",
                        PreviewMesh::RoundedCube => "rounded-cube",
                        PreviewMesh::Plane => "plane",
                        PreviewMesh::Cube => "cube",
                        PreviewMesh::Sphere => "sphere",
                        PreviewMesh::Cylinder => "cylinder",
                        _ => "custom",
                    }))
                    .show_ui(ui, |ui| {
                        for (m, label) in [
                            (PreviewMesh::PreviewBall, "preview-ball"),
                            (PreviewMesh::RoundedCube, "rounded-cube"),
                            (PreviewMesh::Plane, "plane"),
                            (PreviewMesh::Cube, "cube"),
                            (PreviewMesh::Sphere, "sphere"),
                            (PreviewMesh::Cylinder, "cylinder"),
                        ] {
                            ui.selectable_value(&mut mesh, m, text(label));
                        }
                        if let Some(imported) = &self.preview.imported_mesh {
                            ui.selectable_value(&mut mesh, imported.clone(), text("custom"));
                        }
                    });
                if mesh != view.mesh {
                    self.action(UiAction::Request(CommandRequest::SetMesh { mesh }));
                    self.reset_viewport_camera();
                }
                let mut pom = view.pom;
                let changed = ui.checkbox(&mut pom.enabled, "POM").changed();
                ui.label(text("depth"));
                let scale = ui
                    .add(egui::Slider::new(&mut pom.scale, 0.0..=0.2))
                    .changed();
                if changed || scale {
                    self.action(UiAction::Request(CommandRequest::SetPom { pom }));
                }
            }
        });
        if view.mode == ViewMode::ThreeD {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().slider_width = 100.;
                ui.label(text("exposure"));
                if ui
                    .add(egui::Slider::new(&mut self.preview.exposure, -5.0..=5.0))
                    .changed()
                {
                    self.preview.revision += 1;
                }
                if ui.button(text("reset-camera")).clicked() {
                    self.reset_viewport_camera();
                }
                ui.checkbox(&mut self.preview.timings, text("timings"));
                ui.checkbox(&mut self.preview.turntable, text("turntable"));
            });
        }
        ui.horizontal_wrapped(|ui| {
            let mut tiles = view.tiles;
            ui.checkbox(&mut tiles.random_rotation, text("rotation"))
                .on_hover_text(text("rotation-warning"));
            if tiles.random_rotation {
                ui.add(egui::DragValue::new(&mut tiles.seed));
            }
            if tiles != view.tiles {
                self.action(UiAction::Request(CommandRequest::SetTiles { tiles }));
            }
        });
        if view.mode == ViewMode::ThreeD {
            egui::CollapsingHeader::new(text("assets")).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.preview.mesh_path)
                            .hint_text(text("mesh-path"))
                            .desired_width(220.),
                    );
                    if ui
                        .button(crate::input_locale::labels::text("native-browse"))
                        .clicked()
                    {
                        self.browse_preview_asset(true);
                    }
                    if ui.button(text("load-mesh")).clicked() {
                        self.preview.pending =
                            Some(Load::Mesh(self.preview.mesh_path.clone().into()));
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.preview.hdr_path)
                            .hint_text(text("hdr-path"))
                            .desired_width(220.),
                    );
                    if ui
                        .button(crate::input_locale::labels::text("native-browse"))
                        .clicked()
                    {
                        self.browse_preview_asset(false);
                    }
                    if ui.button(text("load-hdr")).clicked() {
                        self.preview.pending =
                            Some(Load::Hdr(self.preview.hdr_path.clone().into()));
                    }
                });
            });
            egui::CollapsingHeader::new(text("environment")).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.preview.environment_enabled, text("ibl-enabled"));
                    ui.checkbox(&mut self.preview.background_visible, text("sky-background"));
                    if ui.button(text("sky")).clicked() {
                        self.preview.pending = Some(Load::Sky);
                    }
                });
                let mut light = view.light.clone();
                let mut light_changed = false;
                egui::Grid::new("viewport-lighting").show(ui, |ui| {
                    for (key, value, range) in [
                        (
                            "environment-rotation",
                            &mut self.preview.environment_rotation,
                            0.0..=360.0,
                        ),
                        (
                            "environment-intensity",
                            &mut self.preview.environment_intensity,
                            0.0..=8.0,
                        ),
                        (
                            "background-blur",
                            &mut self.preview.background_blur,
                            0.0..=1.0,
                        ),
                    ] {
                        ui.label(text(key));
                        ui.add(egui::Slider::new(value, range));
                        ui.end_row();
                    }
                    ui.label(text("light"));
                    light_changed |= ui
                        .add(egui::Slider::new(&mut light.intensity, 0.0..=10.0))
                        .changed();
                    ui.end_row();
                    // Direction is the vector towards the light, in world space.
                    for (axis, value) in ["X", "Y", "Z"].into_iter().zip(&mut light.direction) {
                        ui.label(format!("{} {axis}", text("light-direction")));
                        light_changed |= ui.add(egui::Slider::new(value, -1.0..=1.0)).changed();
                        ui.end_row();
                    }
                });
                if light_changed {
                    self.action(UiAction::Request(CommandRequest::SetLight { light }));
                }
                ui.small(text("orbit"));
            });
        }
    }
    pub(crate) fn orbit_viewport(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if self.session.view_state().mode != ViewMode::ThreeD {
            return;
        }
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        if response.double_clicked()
            || ((response.has_focus()
                || (response.hovered() && !ui.ctx().egui_wants_keyboard_input()))
                && ui.input(|i| i.key_pressed(egui::Key::F)))
        {
            self.reset_viewport_camera();
            return;
        }
        if response.drag_started() {
            self.preview.turntable = false;
        }
        if self.preview.turntable && ui.input(|i| i.focused) {
            let seconds = ui.input(|i| i.stable_dt).clamp(0., 0.05);
            let mut camera = self.session.view_state().camera;
            camera.yaw = (camera.yaw + seconds * 0.15).rem_euclid(std::f32::consts::TAU);
            self.action(UiAction::Request(CommandRequest::SetCamera { camera }));
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
        }
        let delta = if response.dragged() {
            ui.input(|i| i.pointer.delta())
        } else {
            egui::Vec2::ZERO
        };
        let scroll = if response.hovered() {
            ui.input(|i| i.smooth_scroll_delta.y)
        } else {
            0.
        };
        if delta != egui::Vec2::ZERO || scroll != 0. {
            let camera = orbit(self.session.view_state().camera, [delta.x, delta.y], scroll);
            if scroll != 0. {
                self.preview.zoom = (self.preview.zoom * (scroll * 0.002).exp()).clamp(25., 400.);
            }
            self.action(UiAction::Request(CommandRequest::SetCamera { camera }));
        }
    }
    fn reset_viewport_camera(&mut self) {
        let camera =
            crate::compare::frame_camera(&self.session.view_state().mesh, self.preview.aspect);
        self.action(UiAction::Request(CommandRequest::SetCamera { camera }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(c: &mut Controls, ctx: &egui::Context, events: Vec<egui::Event>, focus: bool) {
        let mut output = ctx.run_ui(
            egui::RawInput {
                focused: true,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(500., 400.),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                let (_, response) =
                    ui.allocate_exact_size(egui::vec2(400., 300.), egui::Sense::click_and_drag());
                if focus {
                    response.request_focus();
                }
                c.orbit_viewport(ui, &response);
            },
        );
        output.textures_delta.clear();
    }
    fn controls() -> Controls {
        let mut c = Controls::new(true);
        c.action(UiAction::Request(CommandRequest::SetMode {
            mode: ViewMode::ThreeD,
        }));
        c
    }
    #[test]
    fn turntable_is_opt_in_and_only_animates_the_3d_camera() {
        let mut c = controls();
        let ctx = egui::Context::default();
        let original = c.session().view_state().camera;
        frame(&mut c, &ctx, vec![], false);
        assert_eq!(c.session().view_state().camera, original);
        c.preview.turntable = true;
        let generation = c.session().snapshot().unwrap().generation;
        frame(&mut c, &ctx, vec![], false);
        let animated = c.session().view_state().camera;
        assert!(animated.yaw > original.yaw && animated.yaw - original.yaw < 0.01);
        assert_eq!(animated.pitch, original.pitch);
        assert_eq!(animated.distance, original.distance);
        assert_eq!(c.session().snapshot().unwrap().generation, generation);
        assert_eq!(c.session().history().undo_entries, 0);
        c.action(UiAction::Request(CommandRequest::SetMode {
            mode: ViewMode::TwoD,
        }));
        frame(&mut c, &ctx, vec![], false);
        assert_eq!(c.session().view_state().camera, animated);
    }
    #[test]
    fn focused_viewport_f_resets_camera_without_changing_material_history() {
        let mut c = controls();
        let ctx = egui::Context::default();
        c.action(UiAction::Request(CommandRequest::SetCamera {
            camera: texelith_core::Camera {
                yaw: 1.,
                pitch: -0.4,
                distance: 8.,
                ..Default::default()
            },
        }));
        let generation = c.session().snapshot().unwrap().generation;
        frame(&mut c, &ctx, vec![], true);
        frame(
            &mut c,
            &ctx,
            vec![egui::Event::Key {
                key: egui::Key::F,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            false,
        );
        assert_eq!(
            c.session().view_state().camera,
            crate::compare::reset_camera(&c.session().view_state().mesh)
        );
        assert_eq!(c.session().snapshot().unwrap().generation, generation);
        assert_eq!(c.session().history().undo_entries, 0);
    }
    #[test]
    fn pointer_drag_changes_yaw_and_pitch_and_wheel_changes_distance() {
        let mut c = controls();
        let ctx = egui::Context::default();
        frame(&mut c, &ctx, vec![], false);
        let start = egui::pos2(100., 100.);
        frame(
            &mut c,
            &ctx,
            vec![
                egui::Event::PointerMoved(start),
                egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            false,
        );
        let end = egui::pos2(150., 150.);
        frame(&mut c, &ctx, vec![egui::Event::PointerMoved(end)], false);
        let orbit = c.session().view_state().camera;
        assert!(orbit.yaw > 0. && orbit.pitch > 0.);
        frame(
            &mut c,
            &ctx,
            vec![
                egui::Event::PointerButton {
                    pos: end,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Unknown,
                    delta: egui::vec2(0., 120.),
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            false,
        );
        assert!(c.session().view_state().camera.distance < orbit.distance);
    }
}
