use super::*;
use crate::input_locale::{
    DirtyChoice, DropIntent, InputAction, classify_drop, labels::text, shortcut,
};

impl Controls {
    pub(crate) fn palette_search(&mut self, ui: &mut egui::Ui) {
        let response =
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text(text("palette-search")));
        if self.input.palette_focus {
            response.request_focus();
            self.input.palette_focus = false;
        }
    }

    pub(crate) fn input_palette_entries(&mut self, ui: &mut egui::Ui) {
        for map in crate::material_ui::MAPS {
            let label = format!(
                "{}: {}",
                text("command-material-select-map"),
                crate::material_ui::labels::text(crate::material_ui::map_label(map))
            );
            if !label.to_lowercase().contains(&self.search.to_lowercase()) {
                continue;
            }
            let enabled = self
                .session
                .snapshot()
                .is_some_and(|s| crate::material_ui::supported(&s, map));
            if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                self.input_action(InputAction::Request(CommandRequest::SelectMap { map }));
                self.palette = false;
            }
        }
        for (key, action) in [
            ("save", InputAction::SaveProject { save_as: false }),
            ("save-as", InputAction::SaveProject { save_as: true }),
        ] {
            let label = text(key);
            if label.to_lowercase().contains(&self.search.to_lowercase())
                && ui.button(label).clicked()
            {
                self.input_action(action);
                self.palette = false;
            }
        }
    }

    pub fn input_action(&mut self, action: InputAction) {
        match action {
            InputAction::Request(request) => self.action(UiAction::Request(request)),
            InputAction::Palette => {
                self.palette = true;
                self.input.palette_focus = true;
            }
            InputAction::OpenProject => self.input.project_picker = true,
            InputAction::SaveProject { save_as } => self.project_save_intent(save_as),
        }
    }

    pub fn pending_replacement(&self) -> Option<&DropIntent> {
        self.input.pending.as_ref().map(|pending| &pending.intent)
    }
    pub fn input_feedback_key(&self) -> Option<&'static str> {
        self.input.feedback
    }

    pub fn drop_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        match classify_drop(&paths) {
            Ok(intent) => self.request_intent(intent),
            Err(key) => self.input.feedback = Some(key),
        }
    }

    /// Interactive image replacement; startup/diagnostics can still use open_image.
    pub fn request_image(&mut self, path: std::path::PathBuf) {
        self.request_captured(crate::input_locale::CapturedIntent {
            intent: DropIntent::Image(path),
            import_kind: self.import_kind,
            convention: self.normal_convention,
            linear_height: self.linear_height,
        });
    }

    pub fn request_intent(&mut self, intent: DropIntent) {
        self.request_captured(crate::input_locale::CapturedIntent {
            intent,
            import_kind: crate::material_ui::ImportKind::Photo,
            convention: self.normal_convention,
            linear_height: false,
        });
    }

    fn request_captured(&mut self, captured: crate::input_locale::CapturedIntent) {
        if self.input.pending.is_some() {
            self.input.feedback = Some("drop-busy");
            return;
        }
        // The legacy four-node diagnostic has no project workflow; preserve its
        // immediate import contract. The shipped Simple workspace guards images.
        if matches!(captured.intent, DropIntent::Project(_))
            || (self.material_mode && matches!(captured.intent, DropIntent::Image(_)))
        {
            // Commit the focused gesture before assessing dirty state.
            self.action(UiAction::FocusLost);
            if self.session.has_unsaved_project() {
                self.input.pending = Some(captured);
                self.input.feedback = None;
                return;
            }
        }
        self.execute_intent(captured);
    }

    pub fn resolve_dirty(&mut self, choice: DirtyChoice) {
        match choice {
            DirtyChoice::Cancel => {
                self.input.pending = None;
                self.input.feedback = None;
            }
            DirtyChoice::Save if self.input.pending.is_some() => {
                self.project_save_intent(false);
            }
            DirtyChoice::Discard => {
                if let Some(intent) = self.input.pending.take() {
                    self.execute_intent(intent);
                }
            }
            DirtyChoice::Save => {}
        }
    }

    fn execute_intent(&mut self, captured: crate::input_locale::CapturedIntent) {
        self.input.feedback = None;
        match captured.intent {
            DropIntent::Image(path) => {
                self.import_kind = captured.import_kind;
                self.normal_convention = captured.convention;
                self.linear_height = captured.linear_height;
                self.open_image(path);
            }
            DropIntent::Project(path) => {
                self.next_import = self
                    .next_import
                    .checked_add(1)
                    .expect("import IDs exhausted");
                let source = self.locator(path);
                self.action(UiAction::Request(CommandRequest::OpenProject {
                    token: texelith_core::ProjectOpenToken(self.next_import),
                    source,
                }));
                self.input.feedback = Some("project-open-pending");
            }
            DropIntent::Batch(paths) => {
                self.input.batch = paths;
                self.input.feedback = Some("batch-offer");
            }
            DropIntent::Environment(_) => self.input.feedback = Some("environment-pending"),
            DropIntent::Mesh(_) => self.input.feedback = Some("mesh-pending"),
        }
    }

    /// Called before panel layout, so pointer location never gates a drop.
    pub(crate) fn global_input(&mut self, ctx: &egui::Context) {
        if self.scripted || self.input.last_frame == Some(ctx.cumulative_frame_nr()) {
            return;
        }
        self.input.last_frame = Some(ctx.cumulative_frame_nr());
        let editing = ctx.egui_wants_keyboard_input();
        if !ctx.input(|i| i.focused) {
            self.action(UiAction::FocusLost);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            let active = self.session.history().active_gesture.is_some()
                || self.session.history().active_stroke.is_some()
                || self.inspectors.armed.is_some();
            self.action(UiAction::Escape);
            if !active {
                if self.projects.save_picker {
                    self.projects.save_picker = false;
                } else if !self.project_asset_issues().is_empty() {
                    self.cancel_project_open();
                } else if self.input.pending.is_some() {
                    self.resolve_dirty(DirtyChoice::Cancel);
                } else if self.projects.batch_open {
                    self.projects.batch_open = false;
                } else {
                    self.palette = false;
                    self.input.project_picker = false;
                    self.input.feedback = None;
                }
            }
        }
        let events = ctx.input(|i| i.events.clone());
        for event in events {
            if let egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                modifiers,
                ..
            } = event
                && self.input.pending.is_none()
                && !self.input.project_picker
                && !self.projects.save_picker
                && self.project_asset_issues().is_empty()
                && let Some(action) = shortcut(key, modifiers, editing, self.palette)
            {
                ctx.input_mut(|i| {
                    i.consume_key(modifiers, key);
                });
                // The native application uses Ctrl+O for its primary image flow;
                // embedded/headless hosts retain the existing project action.
                if key == egui::Key::O && self.native_files.enabled {
                    self.browse_image();
                } else {
                    self.input_action(action);
                }
            }
        }
        let paths = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect::<Vec<_>>()
        });
        if !paths.is_empty() {
            self.drop_paths(paths);
        }
    }

    pub(crate) fn input_dialogs(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        self.project_batch_dialogs(&ctx);
        if self.projects.save_picker || !self.project_asset_issues().is_empty() {
            return;
        }
        if self.input.pending.is_some() {
            egui::Modal::new(egui::Id::unique("dirty-replacement")).show(&ctx, |ui| {
                ui.set_max_width(420.0);
                ui.heading(text("dirty-title"));
                ui.label(text("dirty-message"));
                if let Some(key) = self.input.feedback {
                    ui.label(text(key));
                }
                ui.horizontal(|ui| {
                    for (key, choice) in [
                        ("save", DirtyChoice::Save),
                        ("discard", DirtyChoice::Discard),
                        ("cancel", DirtyChoice::Cancel),
                    ] {
                        if ui.button(text(key)).clicked() {
                            self.resolve_dirty(choice);
                        }
                    }
                });
            });
        } else if self.input.project_picker {
            egui::Modal::new(egui::Id::unique("project-picker")).show(&ctx, |ui| {
                ui.heading(text("project-open"));
                ui.label(text("project-path"));
                ui.text_edit_singleline(&mut self.input.project_path);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !self.input.project_path.is_empty(),
                            egui::Button::new(text("open")),
                        )
                        .clicked()
                    {
                        let path = self.input.project_path.clone().into();
                        if matches!(classify_drop(&[path]), Ok(DropIntent::Project(_))) {
                            self.input.project_picker = false;
                            self.drop_paths(vec![self.input.project_path.clone().into()]);
                        } else {
                            self.input.feedback = Some("drop-unsupported");
                        }
                    }
                    if ui.button(text("cancel")).clicked() {
                        self.input.project_picker = false;
                    }
                });
                if let Some(key) = self.input.feedback {
                    ui.label(text(key));
                }
            });
        } else if let Some(key) = self.input.feedback {
            egui::Window::new(text("commands"))
                .id(egui::Id::unique("input-feedback"))
                .collapsible(false)
                .show(root, |ui| {
                    ui.set_max_width(420.0);
                    ui.label(text(key));
                    if key == "batch-offer" && ui.button(text("batch-add")).clicked() {
                        self.projects.batch_open = true;
                        self.input.feedback = None;
                    }
                    if ui.button(text("cancel")).clicked() {
                        self.input.feedback = None;
                        self.input.batch.clear();
                    }
                });
        }
        let hovered = ctx.input(|i| {
            i.raw
                .hovered_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect::<Vec<_>>()
        });
        if !hovered.is_empty() {
            let key = match classify_drop(&hovered) {
                Ok(DropIntent::Image(_)) => "drop-image",
                Ok(DropIntent::Project(_)) => "drop-project",
                Ok(DropIntent::Batch(_)) => "drop-batch",
                Ok(DropIntent::Environment(_)) => "drop-environment",
                Ok(DropIntent::Mesh(_)) => "drop-mesh",
                Err(key) => key,
            };
            egui::Area::new(egui::Id::unique("global-drop-feedback"))
                .order(egui::Order::Tooltip)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .interactable(false)
                .show(&ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.label(text(key));
                    });
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dirty_intent_captures_typed_import_and_global_drop_uses_photo() {
        let mut c = crate::input_locale::contract::fixture().unwrap();
        c.import_kind = crate::material_ui::ImportKind::Normal;
        c.normal_convention = texelith_core::NormalConvention::DirectX;
        c.request_image("normal.png".into());
        c.import_kind = crate::material_ui::ImportKind::Height;
        c.normal_convention = texelith_core::NormalConvention::OpenGl;
        c.resolve_dirty(DirtyChoice::Discard);
        assert_eq!(c.import_kind, crate::material_ui::ImportKind::Normal);
        assert_eq!(
            c.normal_convention,
            texelith_core::NormalConvention::DirectX
        );
        assert!(matches!(
            c.take_effects().as_slice(),
            [Effect::LoadImage { .. }]
        ));
        c.drop_paths(vec!["photo.jpg".into()]);
        c.resolve_dirty(DirtyChoice::Discard);
        assert_eq!(c.import_kind, crate::material_ui::ImportKind::Photo);
        assert!(!c.linear_height);
        assert!(matches!(
            c.take_effects().as_slice(),
            [Effect::LoadImage { .. }]
        ));
    }
}
