use crate::command_adapter::{ActionOrigin, CommandAdapter, UiAction};
use texelith_core::{
    CommandId, CommandRequest, Effect, Extent, ImportToken, InputInterpretation, MapKind, NodeKind,
    OpaqueLocator, ParameterAddress, ParameterId, ParameterValue, Session, SourceDescriptor,
};

#[path = "input_locale/runtime.rs"]
mod global_input;
#[path = "inspectors/gestures.rs"]
mod inspector_gestures;
#[path = "workspace/panels.rs"]
mod material_panels;
#[path = "inspectors/panel.rs"]
mod procedural_panels;

#[path = "compare/controls.rs"]
mod viewport_controls;

#[path = "export_jobs/panel.rs"]
mod export_controls;

#[path = "native_files.rs"]
mod native_files;
pub(crate) use native_files::default_export_folder;

pub struct Controls {
    pub(crate) native_files: native_files::State,
    pub(crate) projects: crate::project_batch::State,
    pub(crate) batch_submission: Option<crate::batch_jobs::Submission>,
    pub(crate) batch_status: crate::batch_jobs::Status,
    pub(crate) batch_dirty: bool,
    pub(crate) batch_cancel_requested: bool,
    pub(crate) batch_interactive_import: Option<ImportToken>,
    pub(crate) packing: crate::packing_editor::State,
    pub(crate) export_options: crate::export_jobs::ExportOptions,
    export_status: Option<crate::export_jobs::JobStatus>,
    export_busy: bool,
    export_feedback: String,
    export_native_options: std::collections::VecDeque<crate::export_jobs::ExportOptions>,
    pub(crate) preview: crate::compare::UiState,
    pub(crate) input: crate::input_locale::InputState,
    pub(crate) inspectors: crate::inspectors::InspectorState,
    pub(crate) workspace: crate::workspace::Workspace,
    pub(crate) material_mode: bool,
    pub(crate) import_kind: crate::material_ui::ImportKind,
    pub(crate) normal_convention: texelith_core::NormalConvention,
    pub(crate) heatmap: bool,
    pub(crate) session: Session,
    pub(crate) adapter: CommandAdapter,
    palette: bool,
    search: String,
    pub(crate) path: String,
    destination: String,
    pub(crate) next_import: u64,
    pub(crate) message: String,
    focus: Option<CommandId>,
    pub(crate) effects: Vec<Effect>,
    diagnostic: bool,
    suppressed_slider: Option<ParameterAddress>,
    import_interrupted_drag: bool,
    paths: std::collections::HashMap<String, std::path::PathBuf>,
    next_path: u64,
    linear_height: bool,
    scripted: bool,
    pub(crate) render_revision: u64,
    pub(crate) render_quality: texelith_gpu::Quality,
    #[cfg(test)]
    slider_rect: Option<egui::Rect>,
}

impl Controls {
    pub fn new(diagnostic: bool) -> Self {
        let mut s = Self {
            native_files: Default::default(),
            projects: Default::default(),
            batch_submission: None,
            batch_status: Default::default(),
            batch_dirty: false,
            batch_cancel_requested: false,
            batch_interactive_import: None,
            packing: Default::default(),
            export_options: Default::default(),
            export_status: None,
            export_busy: false,
            export_feedback: String::new(),
            export_native_options: Default::default(),
            preview: Default::default(),
            input: Default::default(),
            inspectors: Default::default(),
            workspace: Default::default(),
            material_mode: false,
            import_kind: crate::material_ui::ImportKind::Photo,
            normal_convention: texelith_core::NormalConvention::OpenGl,
            heatmap: false,
            session: Session::new(),
            adapter: CommandAdapter::default(),
            palette: false,
            search: String::new(),
            path: String::new(),
            destination: String::new(),
            next_import: 0,
            message: String::new(),
            focus: None,
            effects: Vec::new(),
            diagnostic,
            suppressed_slider: None,
            import_interrupted_drag: false,
            paths: Default::default(),
            next_path: 0,
            linear_height: false,
            scripted: false,
            render_revision: 0,
            render_quality: texelith_gpu::Quality::Full,
            #[cfg(test)]
            slider_rect: None,
        };
        if diagnostic {
            let token = ImportToken(1);
            s.session
                .dispatch(CommandRequest::NewFromImage {
                    token,
                    source: OpaqueLocator("diagnostic fixture".into()),
                })
                .unwrap();
            s.session
                .complete_import(
                    token,
                    SourceDescriptor {
                        content_hash: [1; 32],
                        extent: Extent {
                            width: 4,
                            height: 4,
                        },
                        interpretation: InputInterpretation::LinearHeight,
                    },
                )
                .unwrap();
            s.next_import = 1;
        }
        s
    }
    pub fn new_material() -> Self {
        let mut controls = Self::new(false);
        controls.material_mode = true;
        controls.preview.split_view = true;
        controls.action(UiAction::Request(CommandRequest::SetCamera {
            camera: crate::compare::reset_camera(&controls.session.view_state().mesh),
        }));
        controls
    }
    pub(crate) fn close_palette(&mut self) {
        self.palette = false;
    }
    pub(crate) fn open_palette(&mut self) {
        self.palette = true;
        self.input.palette_focus = true;
    }
    pub(crate) fn use_scripted_input(&mut self) {
        self.scripted = true;
    }
    pub fn take_effects(&mut self) -> Vec<Effect> {
        self.export_native_options.clear();
        std::mem::take(&mut self.effects)
    }
    pub fn session(&self) -> &Session {
        &self.session
    }
    pub fn action(&mut self, action: UiAction) {
        if matches!(action, UiAction::Escape) {
            if self.session.history().active_gesture.is_none()
                && self.session.history().active_stroke.is_none()
                && self.inspectors.armed.is_none()
            {
                self.packing.open = false;
            }
            self.inspectors.cancel_transients();
        }
        let previous = self.adapter.clone();
        let active_address = previous.active_address();
        for request in self.adapter.handle(action) {
            if matches!(request, CommandRequest::BatchCancel {}) && self.batch_status.busy {
                self.batch_cancel_requested = true;
                continue;
            }
            let batch_command = matches!(
                request,
                CommandRequest::BatchStart {}
                    | CommandRequest::BatchCancel {}
                    | CommandRequest::BatchRetryFailed {}
            );
            let render = matches!(
                request,
                CommandRequest::BeginGesture { .. }
                    | CommandRequest::SetParameter { .. }
                    | CommandRequest::CommitGesture { .. }
                    | CommandRequest::CancelGesture { .. }
                    | CommandRequest::Undo
                    | CommandRequest::Redo
                    | CommandRequest::SelectOutput { .. }
                    | CommandRequest::SelectMap { .. }
                    | CommandRequest::ConfigureMaterial { .. }
                    | CommandRequest::BeginStroke { .. }
                    | CommandRequest::AppendStroke { .. }
                    | CommandRequest::CommitStroke { .. }
                    | CommandRequest::CancelStroke { .. }
            );
            let capture_a = matches!(request, CommandRequest::CaptureA {});
            let fit_camera = matches!(request, CommandRequest::SetMesh { .. })
                || (matches!(
                    request,
                    CommandRequest::SetMode {
                        mode: texelith_core::ViewMode::ThreeD
                    }
                ) && self.session.view_state().mode != texelith_core::ViewMode::ThreeD);
            match self.session.dispatch(request) {
                Ok(outcome) => {
                    self.preview.fit_pending |= fit_camera;
                    self.batch_dirty |= batch_command;
                    if capture_a {
                        self.preview.compare_enabled = true;
                    }
                    for effect in outcome.effects {
                        if let Effect::LoadImage { token, .. } = &effect {
                            self.batch_interactive_import = Some(*token);
                        }
                        if matches!(effect, Effect::Export { .. }) {
                            self.export_native_options
                                .push_back(self.export_options.clone());
                        }
                        if let Some(effect) = self.route_project_effect(effect) {
                            self.effects.push(effect);
                        }
                    }
                    self.message.clear();
                    self.adapter.sync(&self.session);
                    if render {
                        self.render_revision = self
                            .render_revision
                            .checked_add(1)
                            .expect("render revision exhausted");
                        self.render_quality = if self.session.history().active_gesture.is_some() {
                            texelith_gpu::Quality::Proxy
                        } else {
                            texelith_gpu::Quality::Full
                        };
                    }
                    if active_address.is_some() && self.adapter.active_address().is_none() {
                        // A command can end the transaction while egui still owns
                        // the pointer. Ignore that slider through its release frame.
                        self.suppressed_slider = active_address;
                    }
                }
                Err(error) => {
                    self.message = error.to_string();
                    self.adapter = previous.clone();
                }
            }
        }
    }
    pub(crate) fn locator(&mut self, path: std::path::PathBuf) -> OpaqueLocator {
        self.next_path = self.next_path.checked_add(1).expect("path IDs exhausted");
        let key = format!("native-{}", self.next_path);
        self.paths.insert(key.clone(), path);
        OpaqueLocator(key)
    }
    pub(crate) fn take_path(&mut self, locator: OpaqueLocator) -> Option<std::path::PathBuf> {
        self.paths.remove(&locator.0)
    }
    pub fn open_image(&mut self, path: std::path::PathBuf) {
        self.path = path.to_string_lossy().into_owned();
        self.next_import = self
            .next_import
            .checked_add(1)
            .expect("import IDs exhausted");
        self.remember_project_source(ImportToken(self.next_import), path.clone());
        let source = self.locator(path);
        self.action(UiAction::Request(CommandRequest::NewFromImage {
            token: ImportToken(self.next_import),
            source,
        }));
    }
    pub fn save_png(&mut self, path: std::path::PathBuf, map: MapKind) {
        if self.native_files.enabled
            && path.exists()
            && rfd::MessageDialog::new()
                .set_title(crate::input_locale::labels::text("native-overwrite-title"))
                .set_description(format!(
                    "{}\n\n{}",
                    crate::input_locale::labels::text("native-overwrite-png"),
                    path.display()
                ))
                .set_level(rfd::MessageLevel::Warning)
                .set_buttons(rfd::MessageButtons::YesNo)
                .show()
                != rfd::MessageDialogResult::Yes
        {
            return;
        }
        let destination = self.locator(path);
        self.action(UiAction::Request(CommandRequest::SavePng {
            destination: destination.clone(),
            map,
        }));
        // A rejected command emitted no effect, so its native locator is unused.
        if !self
            .effects
            .iter()
            .any(|e| matches!(e, Effect::SavePng { destination: d, .. } if d == &destination))
        {
            self.paths.remove(&destination.0);
        }
    }
    pub(crate) fn interpretation(&self) -> InputInterpretation {
        if self.linear_height {
            InputInterpretation::LinearHeight
        } else {
            InputInterpretation::SrgbPhoto
        }
    }
    pub fn complete_import(
        &mut self,
        token: ImportToken,
        source: SourceDescriptor,
    ) -> Result<(), String> {
        self.complete_typed_import(token, source, self.normal_convention)
    }
    pub(crate) fn complete_typed_import(
        &mut self,
        token: ImportToken,
        source: SourceDescriptor,
        convention: texelith_core::NormalConvention,
    ) -> Result<(), String> {
        let active_address = self.adapter.active_address();
        let preferred_mode = if self.preview.split_view {
            texelith_core::ViewMode::ThreeD
        } else {
            self.session.view_state().mode
        };
        if self.material_mode {
            self.session
                .complete_material_import_with_convention(token, source, convention)
        } else {
            self.session.complete_import(token, source)
        }
        .map_err(|e| e.to_string())?;
        if self.material_mode {
            self.preview.compare_enabled = false;
            self.action(UiAction::Request(CommandRequest::SetMode {
                mode: preferred_mode,
            }));
            self.action(UiAction::Request(CommandRequest::SetCamera {
                camera: crate::compare::frame_camera(
                    &self.session.view_state().mesh,
                    self.preview.aspect,
                ),
            }));
        }
        self.commit_project_source(token);
        if self.batch_interactive_import == Some(token) {
            self.batch_interactive_import = None;
        }
        self.workspace.clear_thumbnails();
        self.inspectors.cancel_transients();
        self.inspectors.painting = false;
        self.adapter.sync(&self.session);
        // Decode can finish while a pointer still owns an old-document slider.
        // Keep that pointer from creating fresh edits in the replacement history.
        self.suppressed_slider = active_address.or(self.suppressed_slider);
        self.import_interrupted_drag |= active_address.is_some();
        self.render_revision = self
            .render_revision
            .checked_add(1)
            .expect("render revision exhausted");
        self.render_quality = texelith_gpu::Quality::Full;
        self.diagnostic = false;
        Ok(())
    }
    pub fn fail_import(&mut self, token: ImportToken) {
        let _ = self.session.fail_import(token);
        if self.batch_interactive_import == Some(token) {
            self.batch_interactive_import = None;
        }
    }
    fn invoke(&mut self, id: CommandId, origin: ActionOrigin) {
        if self.material_mode {
            match id {
                CommandId::ConfigureMaterial => {
                    if let Some(snapshot) = self.session.snapshot() {
                        let (seamless, painted_mask) = crate::material_ui::options(&snapshot.graph);
                        self.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                            seamless: !seamless,
                            painted_mask,
                        }));
                    }
                    return;
                }
                CommandId::InspectSeams => {
                    self.action(UiAction::Request(CommandRequest::InspectSeams {
                        enabled: !self.heatmap,
                    }));
                    return;
                }
                CommandId::SelectMap => {
                    if let Some(snapshot) = self.session.snapshot() {
                        let selected = snapshot.graph.selected_output_ref().port.map_kind();
                        let index = crate::material_ui::MAPS
                            .iter()
                            .position(|map| *map == selected)
                            .unwrap_or(0);
                        if let Some(map) = (1..=7)
                            .map(|offset| crate::material_ui::MAPS[(index + offset) % 7])
                            .find(|map| crate::material_ui::supported(&snapshot, *map))
                        {
                            self.action(UiAction::Request(CommandRequest::SelectMap { map }));
                        }
                    }
                    return;
                }
                _ => {}
            }
        }
        match id {
            CommandId::SaveExportPreset => {
                if let Err(error) = self.open_packing_editor() {
                    self.packing.feedback = error;
                }
                self.packing.focus_name = true;
            }
            CommandId::RequestExport => {
                self.confirm_material_export();
            }
            CommandId::CancelExport => self.cancel_material_export(),
            CommandId::OpenProject => {
                self.input_action(crate::input_locale::InputAction::OpenProject)
            }
            CommandId::SaveProject => self.project_save_intent(false),
            CommandId::SaveProjectAs => self.project_save_intent(true),
            CommandId::NewFromImage if self.native_files.enabled => self.browse_image(),
            CommandId::NewFromImage
            | CommandId::SetParameter
            | CommandId::BeginGesture
            | CommandId::SelectOutput
            | CommandId::SavePng => self.focus = Some(id),
            _ => self.action(UiAction::Invoke { origin, id }),
        }
    }
    pub fn show(&mut self, root: &mut egui::Ui) {
        if self.material_mode {
            self.show_material(root);
            return;
        }
        let ctx = root.ctx().clone();
        self.global_input(&ctx);
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let focused = self.scripted || ctx.input(|i| i.focused);
        egui::Panel::right("controls")
            .default_size(300.0)
            .show(root, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if self.scripted {
                        ui.disable();
                    }
                    ui.heading(crate::input_locale::labels::text(
                        "legacy-material-controls",
                    ));
                    if self.diagnostic {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "Diagnostic fixture — no image loaded",
                        );
                    } else if self.session.snapshot().is_none() {
                        ui.label(crate::input_locale::labels::text("legacy-no-image-loaded"));
                    }
                    ui.horizontal(|ui| {
                        ui.label(crate::input_locale::labels::text("legacy-image-path"));
                        let r = ui.text_edit_singleline(&mut self.path);
                        if self.focus == Some(CommandId::NewFromImage) {
                            r.request_focus();
                            self.focus = None;
                        }
                    });
                    if ui
                        .button(crate::input_locale::labels::text("legacy-new-from-image"))
                        .clicked()
                        && !self.path.is_empty()
                    {
                        self.request_image(self.path.clone().into());
                    }
                    ui.checkbox(
                        &mut self.linear_height,
                        crate::input_locale::labels::text(
                            "legacy-import-as-linear-height-red-channel",
                        ),
                    );
                    ui.small(crate::input_locale::labels::text(
                        "legacy-otherwise-srgb-photo-luminance-applies-to-the-next-import",
                    ));
                    ui.small(crate::input_locale::labels::text(
                        "legacy-a-successful-load-clears-history-alpha-icc-and-exif-are-ignored",
                    ));
                    let snapshot = self.session.snapshot();
                    if let Some(snapshot) = snapshot {
                        for (kind, parameter, label) in [
                            (NodeKind::Levels, ParameterId::Black, "Black"),
                            (NodeKind::Levels, ParameterId::White, "White"),
                            (NodeKind::Levels, ParameterId::Gamma, "Gamma"),
                            (
                                NodeKind::Blur,
                                ParameterId::SigmaSourcePx,
                                "Blur (source px)",
                            ),
                            (
                                NodeKind::NormalFromHeight,
                                ParameterId::Strength,
                                "Normal strength",
                            ),
                        ] {
                            if let Some(node) = snapshot
                                .graph
                                .nodes()
                                .iter()
                                .find(|n| n.params.kind() == kind)
                            {
                                let address = ParameterAddress {
                                    node: node.id,
                                    parameter,
                                };
                                let mut value = node.params.parameter(parameter).unwrap();
                                let d = parameter.descriptor();
                                let r = ui.add(
                                    egui::Slider::new(&mut value, d.min..=d.max)
                                        .text(crate::input_locale::labels::legacy(label)),
                                );
                                #[cfg(test)]
                                if parameter == ParameterId::Gamma {
                                    self.slider_rect = Some(r.rect);
                                }
                                if matches!(
                                    self.focus,
                                    Some(CommandId::SetParameter | CommandId::BeginGesture)
                                ) {
                                    r.request_focus();
                                    self.focus = None;
                                }
                                if !escape
                                    && focused
                                    && !self.import_interrupted_drag
                                    && self.suppressed_slider != Some(address)
                                {
                                    if r.drag_started() {
                                        self.action(UiAction::DragBegin { address });
                                    }
                                    if r.changed() {
                                        self.action(UiAction::Change {
                                            address,
                                            value: ParameterValue::Float(value),
                                        });
                                    }
                                    if r.drag_stopped() {
                                        self.action(UiAction::DragRelease);
                                    }
                                }
                            }
                        }
                        ui.separator();
                        ui.label(crate::input_locale::labels::text("legacy-selected-output"));
                        for node in snapshot.graph.nodes().iter().filter(|n| {
                            matches!(
                                n.params.kind(),
                                NodeKind::Levels | NodeKind::Blur | NodeKind::NormalFromHeight
                            )
                        }) {
                            let label = format!("{:?} ({})", node.params.kind(), node.id.0);
                            let r = ui.selectable_label(
                                snapshot.graph.selected_output() == node.id,
                                label,
                            );
                            if self.focus == Some(CommandId::SelectOutput) {
                                r.request_focus();
                                self.focus = None;
                            }
                            if r.clicked() {
                                self.action(UiAction::Request(CommandRequest::SelectOutput {
                                    output: node.id,
                                }));
                            }
                        }
                    } else {
                        ui.add_enabled(
                            false,
                            egui::Slider::new(&mut 0.0, 0.0..=1.0).text(
                                crate::input_locale::labels::text(
                                    "legacy-parameters-require-an-image",
                                ),
                            ),
                        );
                    }
                    ui.separator();
                    let history = self.session.history();
                    ui.label(format!(
                        "Undo: {} | Redo: {} | History: {} bytes",
                        history.undo_entries, history.redo_entries, history.byte_count
                    ));
                    let entries = self.adapter.palette(&self.session);
                    ui.horizontal(|ui| {
                        for id in [CommandId::Undo, CommandId::Redo] {
                            let e = entries.iter().find(|e| e.id == id).unwrap();
                            if ui
                                .add_enabled(e.enabled, egui::Button::new(&e.label))
                                .clicked()
                            {
                                self.invoke(id, ActionOrigin::Menu);
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(crate::input_locale::labels::text("legacy-png-path"));
                        let r = ui.text_edit_singleline(&mut self.destination);
                        if self.focus == Some(CommandId::SavePng) {
                            r.request_focus();
                            self.focus = None;
                        }
                    });
                    let can_save = entries
                        .iter()
                        .any(|e| e.id == CommandId::SavePng && e.enabled);
                    if ui
                        .add_enabled(
                            can_save && !self.destination.is_empty(),
                            egui::Button::new(crate::input_locale::labels::text("legacy-save-png")),
                        )
                        .clicked()
                    {
                        let map = self
                            .session
                            .snapshot()
                            .and_then(|s| {
                                s.graph.node(s.graph.selected_output()).map(|n| {
                                    if n.params.kind() == NodeKind::NormalFromHeight {
                                        MapKind::Normal
                                    } else {
                                        MapKind::Height
                                    }
                                })
                            })
                            .unwrap();
                        self.save_png(self.destination.clone().into(), map);
                    }
                    if !self.effects.is_empty() {
                        ui.label(format!(
                            "{} effect(s) pending application I/O",
                            self.effects.len()
                        ));
                    }
                    if !self.message.is_empty() {
                        ui.colored_label(egui::Color32::LIGHT_RED, &self.message);
                    }
                    if ui
                        .button(crate::input_locale::labels::text("commands"))
                        .clicked()
                    {
                        self.open_palette();
                    }
                });
            });
        if self.palette {
            egui::Window::new(crate::input_locale::labels::text("commands"))
                .collapsible(false)
                .show(root, |ui| {
                    self.palette_search(ui);
                    for e in self.adapter.palette(&self.session) {
                        if format!("{} {}", e.id.as_str(), e.label)
                            .to_lowercase()
                            .contains(&self.search.to_lowercase())
                            && ui
                                .add_enabled(
                                    e.enabled,
                                    egui::Button::new(format!("{} — {}", e.id.as_str(), e.label)),
                                )
                                .clicked()
                        {
                            self.invoke(e.id, ActionOrigin::Palette);
                            self.palette = false;
                        }
                    }
                });
        }
        self.input_dialogs(root);
        if ctx.input(|i| !i.pointer.any_down()) {
            self.suppressed_slider = None;
            self.import_interrupted_drag = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_smoke_input_is_independent_of_native_focus_loss() {
        for scripted in [false, true] {
            let mut controls = Controls::new(true);
            if scripted {
                controls.use_scripted_input();
            }
            let address = ParameterAddress {
                node: texelith_core::NodeId(4),
                parameter: ParameterId::Strength,
            };
            controls.action(UiAction::DragBegin { address });
            controls.action(UiAction::Change {
                address,
                value: ParameterValue::Float(3.0),
            });
            let ctx = egui::Context::default();
            let input = egui::RawInput {
                focused: false,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| controls.show(ui));
            output.textures_delta.clear();
            assert_eq!(
                controls.session.history().active_gesture.is_some(),
                scripted
            );
            assert_eq!(
                controls.session.history().undo_entries,
                usize::from(!scripted)
            );
        }
    }

    #[test]
    fn release_without_a_change_and_view_selection_still_request_new_renders() {
        let mut controls = Controls::new(true);
        let address = ParameterAddress {
            node: texelith_core::NodeId(4),
            parameter: ParameterId::Strength,
        };
        let generation = controls.session.snapshot().unwrap().generation;
        controls.action(UiAction::DragBegin { address });
        let dragging = controls.render_revision;
        assert_eq!(controls.render_quality, texelith_gpu::Quality::Proxy);
        controls.action(UiAction::DragRelease);
        assert!(controls.render_revision > dragging);
        assert_eq!(controls.render_quality, texelith_gpu::Quality::Full);
        assert_eq!(controls.session.snapshot().unwrap().generation, generation);
        assert_eq!(controls.session.history().undo_entries, 0);
        let released = controls.render_revision;
        controls.action(UiAction::Request(CommandRequest::SelectOutput {
            output: texelith_core::NodeId(3),
        }));
        assert!(controls.render_revision > released);
        assert_eq!(controls.session.snapshot().unwrap().generation, generation);
    }

    #[test]
    fn command_adapter_palette_begin_focuses_parameter() {
        let mut controls = Controls::new(true);
        controls.invoke(CommandId::BeginGesture, ActionOrigin::Palette);
        assert_eq!(controls.focus, Some(CommandId::BeginGesture));
        assert!(controls.session.history().active_gesture.is_none());
    }

    #[test]
    fn command_adapter_escape_during_egui_drag_restores_and_preserves_redo() {
        let mut controls = Controls::new(true);
        let address = ParameterAddress {
            node: texelith_core::NodeId(2),
            parameter: ParameterId::Gamma,
        };
        controls.action(UiAction::Change {
            address,
            value: ParameterValue::Float(2.0),
        });
        controls.action(UiAction::UndoShortcut);
        let ctx = egui::Context::default();
        let frame = |controls: &mut Controls, events| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                events,
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| controls.show(ui));
            output.textures_delta.clear();
        };
        frame(&mut controls, vec![]);
        let rect = controls.slider_rect.unwrap();
        let start = egui::pos2(rect.left() + 30.0, rect.center().y);
        let end = egui::pos2(rect.left() + 90.0, rect.center().y);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        frame(
            &mut controls,
            vec![egui::Event::PointerMoved(start), button(start, true)],
        );
        frame(&mut controls, vec![egui::Event::PointerMoved(end)]);
        assert!(controls.session.history().active_gesture.is_some());
        frame(
            &mut controls,
            vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        frame(&mut controls, vec![button(end, false)]);
        let value = controls
            .session
            .snapshot()
            .unwrap()
            .graph
            .node(address.node)
            .unwrap()
            .params
            .parameter(address.parameter)
            .unwrap();
        assert_eq!(value, 1.0);
        assert_eq!(controls.session.history().undo_entries, 0);
        assert_eq!(controls.session.history().redo_entries, 1);
        assert!(controls.session.history().active_gesture.is_none());
    }

    struct DragHarness {
        controls: Controls,
        ctx: egui::Context,
        pointer: egui::Pos2,
    }

    impl DragHarness {
        fn new() -> Self {
            let mut harness = Self {
                controls: Controls::new(true),
                ctx: egui::Context::default(),
                pointer: egui::Pos2::ZERO,
            };
            harness.frame(vec![], egui::Modifiers::NONE);
            harness
        }

        fn address(&self) -> ParameterAddress {
            ParameterAddress {
                node: texelith_core::NodeId(2),
                parameter: ParameterId::Gamma,
            }
        }

        fn value(&self) -> f32 {
            let address = self.address();
            self.controls
                .session
                .snapshot()
                .unwrap()
                .graph
                .node(address.node)
                .unwrap()
                .params
                .parameter(address.parameter)
                .unwrap()
        }

        fn frame(&mut self, mut events: Vec<egui::Event>, modifiers: egui::Modifiers) {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                events,
                ..Default::default()
            };
            let mut output = self.ctx.run_ui(input, |ui| self.controls.show(ui));
            output.textures_delta.clear();
        }

        fn button(&mut self, pressed: bool) {
            self.frame(
                vec![egui::Event::PointerButton {
                    pos: self.pointer,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }],
                egui::Modifiers::NONE,
            );
        }

        fn begin(&mut self) {
            let rect = self.controls.slider_rect.unwrap();
            self.pointer = egui::pos2(rect.left() + 30.0, rect.center().y);
            self.frame(
                vec![egui::Event::PointerMoved(self.pointer)],
                egui::Modifiers::NONE,
            );
            self.button(true);
            self.move_by(40.0);
            assert!(self.controls.session.history().active_gesture.is_some());
        }

        fn move_by(&mut self, dx: f32) {
            self.pointer.x += dx;
            self.frame(
                vec![egui::Event::PointerMoved(self.pointer)],
                egui::Modifiers::NONE,
            );
        }

        fn key(&mut self, key: egui::Key, command: bool) {
            let modifiers = egui::Modifiers {
                ctrl: command,
                command,
                ..Default::default()
            };
            self.frame(
                vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers,
                }],
                modifiers,
            );
        }

        fn assert_interruption_survives_release(&mut self, value: f32, undo: usize, redo: usize) {
            let generation = self.controls.session.snapshot().unwrap().generation;
            for step in 0..5 {
                assert_eq!(self.value(), value, "step {step}");
                let history = self.controls.session.history();
                assert_eq!(history.undo_entries, undo, "step {step}");
                assert_eq!(history.redo_entries, redo, "step {step}");
                assert!(history.active_gesture.is_none(), "step {step}");
                assert_eq!(
                    self.controls.session.snapshot().unwrap().generation,
                    generation
                );
                match step {
                    0 | 3 => self.frame(vec![], egui::Modifiers::NONE),
                    1 => self.move_by(15.0),
                    2 => self.button(false),
                    _ => {}
                }
            }
            // Suppression ends with this interaction; a fresh drag still commits once.
            self.begin();
            self.move_by(10.0);
            self.button(false);
            assert_eq!(self.controls.session.history().undo_entries, undo + 1);
            assert_eq!(self.controls.session.history().redo_entries, 0);
            assert_ne!(self.value(), value);
            assert!(self.controls.session.history().active_gesture.is_none());
        }
    }

    #[test]
    fn command_adapter_undo_mid_drag_ignores_held_pointer_until_release() {
        let mut harness = DragHarness::new();
        harness.begin();
        assert_ne!(harness.value(), 1.0);
        harness.key(egui::Key::Z, true);
        harness.assert_interruption_survives_release(1.0, 0, 1);
    }

    #[test]
    fn completed_import_mid_drag_suppresses_old_pointer_in_the_new_document() {
        let mut harness = DragHarness::new();
        harness.begin();
        harness.controls.open_image("replacement.png".into());
        let Effect::LoadImage { token, .. } = harness.controls.take_effects().remove(0) else {
            panic!("expected load")
        };
        harness
            .controls
            .complete_import(
                token,
                SourceDescriptor {
                    content_hash: [2; 32],
                    extent: Extent {
                        width: 4,
                        height: 4,
                    },
                    interpretation: InputInterpretation::LinearHeight,
                },
            )
            .unwrap();
        harness.assert_interruption_survives_release(1.0, 0, 0);
    }

    #[test]
    fn command_adapter_redo_mid_drag_ignores_held_pointer_until_release() {
        let mut harness = DragHarness::new();
        harness.begin();
        let dragged = harness.value();
        harness.key(egui::Key::Z, true);
        // Redo is available after Undo terminates the gesture, while the same
        // physical drag is still held. It must not be overwritten by that drag.
        harness.key(egui::Key::Y, true);
        harness.assert_interruption_survives_release(dragged, 1, 0);
    }

    #[test]
    fn command_adapter_redo_ends_net_zero_drag_without_reapplying_pointer_value() {
        let mut harness = DragHarness::new();
        harness.begin();
        let dragged = harness.value();
        harness.button(false);
        harness.controls.action(UiAction::Change {
            address: harness.address(),
            value: ParameterValue::Float(2.0),
        });
        harness.controls.action(UiAction::UndoShortcut);
        harness.begin();
        assert_eq!(harness.value(), dragged);
        assert_eq!(harness.controls.session.history().redo_entries, 1);
        // Core permits redo during an active net-zero gesture.
        harness.key(egui::Key::Y, true);
        harness.assert_interruption_survives_release(2.0, 2, 0);
    }

    #[test]
    fn command_adapter_escape_mid_drag_ignores_held_pointer_until_release() {
        let mut harness = DragHarness::new();
        harness.controls.action(UiAction::Change {
            address: harness.address(),
            value: ParameterValue::Float(2.0),
        });
        harness.controls.action(UiAction::UndoShortcut);
        harness.begin();
        harness.key(egui::Key::Escape, false);
        harness.assert_interruption_survives_release(1.0, 0, 1);
    }

    #[test]
    fn command_adapter_palette_cancel_mid_drag_ignores_held_pointer_until_release() {
        let mut harness = DragHarness::new();
        harness.begin();
        harness
            .controls
            .invoke(CommandId::CancelGesture, ActionOrigin::Palette);
        harness.assert_interruption_survives_release(1.0, 0, 0);
    }
}
