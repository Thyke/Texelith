use super::*;

impl Controls {
    pub(crate) fn project_batch_header(&mut self, root: &mut egui::Ui) {
        use crate::input_locale::InputAction;
        use crate::workspace::{labels::text as label, studio};
        egui::Panel::top("project-batch-header").show(root, |ui| {
            let header = ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 18.;
                studio::logo(ui);
                ui.label(egui::RichText::new("Texelith").size(24.).strong());
                if ui.available_width() > 700. {
                    ui.separator();
                    ui.label(label("studio-subtitle"));
                }
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button(label("menu-file"), |ui| {
                        if ui
                            .button(crate::input_locale::labels::text("native-open-image"))
                            .clicked()
                        {
                            self.browse_image();
                            ui.close();
                        }
                        if ui.button(text("project-open")).clicked() {
                            self.input_action(InputAction::OpenProject);
                            ui.close();
                        }
                        for (key, save_as) in [("save", false), ("save-as", true)] {
                            if ui
                                .add_enabled(
                                    self.session.snapshot().is_some(),
                                    egui::Button::new(text(key)),
                                )
                                .clicked()
                            {
                                self.input_action(InputAction::SaveProject { save_as });
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                self.session.snapshot().is_some() && !self.export_busy(),
                                egui::Button::new(label("export-action")),
                            )
                            .clicked()
                        {
                            self.confirm_material_export();
                            ui.close();
                        }
                        if ui.button(text("batch-title")).clicked() {
                            self.projects.batch_open = true;
                            ui.close();
                        }
                    });
                    ui.menu_button(label("menu-edit"), |ui| {
                        for (key, id) in [("undo", CommandId::Undo), ("redo", CommandId::Redo)] {
                            let enabled = self
                                .session
                                .availability()
                                .iter()
                                .any(|e| e.id == id && e.enabled);
                            if ui
                                .add_enabled(enabled, egui::Button::new(label(key)))
                                .clicked()
                            {
                                self.action(UiAction::Invoke {
                                    id,
                                    origin: crate::command_adapter::ActionOrigin::Menu,
                                });
                                ui.close();
                            }
                        }
                        if ui.button(label("search")).clicked() {
                            self.open_palette();
                            ui.close();
                        }
                    });
                    ui.menu_button(label("menu-layer"), |ui| {
                        for key in ["add-layer", "duplicate-layer", "delete-layer"] {
                            ui.add_enabled(false, egui::Button::new(label(key)))
                                .on_disabled_hover_text(label("faz2-tooltip"));
                        }
                    });
                    ui.menu_button(label("menu-texture"), |ui| {
                        ui.label(crate::input_locale::labels::text("native-open-image"));
                        egui::ComboBox::from_id_salt("studio-import-kind")
                            .selected_text(crate::input_locale::labels::import_kind(
                                self.import_kind,
                            ))
                            .show_ui(ui, |ui| {
                                for kind in [
                                    crate::material_ui::ImportKind::Photo,
                                    crate::material_ui::ImportKind::Height,
                                    crate::material_ui::ImportKind::Normal,
                                ] {
                                    ui.selectable_value(
                                        &mut self.import_kind,
                                        kind,
                                        crate::input_locale::labels::import_kind(kind),
                                    );
                                }
                            });
                        if self.import_kind == crate::material_ui::ImportKind::Normal {
                            for convention in [
                                texelith_core::NormalConvention::OpenGl,
                                texelith_core::NormalConvention::DirectX,
                            ] {
                                ui.selectable_value(
                                    &mut self.normal_convention,
                                    convention,
                                    format!("{convention:?}"),
                                );
                            }
                        }
                        if let Some(snapshot) = self.session.snapshot() {
                            self.map_options(ui, &snapshot);
                        }
                    });
                    ui.menu_button(label("menu-view"), |ui| {
                        let mut maps = self.workspace.maps_open.unwrap_or(true);
                        self.workspace.layout.maps_toggle =
                            Some(ui.checkbox(&mut maps, label("map-channel")).rect);
                        self.workspace.maps_open = Some(maps);
                        let mut inspector = self
                            .workspace
                            .inspector_is_open(ui.ctx().content_rect().width());
                        let response = ui.checkbox(&mut inspector, label("layers"));
                        self.workspace.layout.inspector_toggle = Some(response.rect);
                        if response.changed() {
                            self.workspace.inspector_open = Some(inspector);
                        }
                        self.viewport_header(ui);
                        if ui.button(label("view-settings")).clicked() {
                            self.workspace.about = false;
                            self.workspace.view_settings = true;
                            ui.close();
                        }
                    });
                    ui.menu_button(label("menu-help"), |ui| {
                        if ui.button(label("about")).clicked() {
                            self.workspace.about = true;
                            ui.close();
                        }
                        let mut turkish = crate::material_ui::labels::is_turkish();
                        if ui.checkbox(&mut turkish, "T\u{fc}rk\u{e7}e").changed() {
                            crate::material_ui::labels::set_turkish(turkish);
                        }
                    });
                });
                if self.project_busy() {
                    ui.spinner();
                }
            });
            self.workspace.layout.header_content = Some(header.response.rect);
        });
        egui::Panel::top("document-tabs")
            .exact_size(42.)
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    let name = self
                        .projects
                        .path
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .map(|p| p.to_string_lossy().into_owned())
                        .or_else(|| {
                            std::path::Path::new(&self.path)
                                .file_name()
                                .map(|p| p.to_string_lossy().into_owned())
                        })
                        .unwrap_or_else(|| label("untitled"));
                    let tab = egui::Frame::group(ui.style())
                        .fill(ui.visuals().faint_bg_color)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{}{}",
                                        name,
                                        if self.session.has_unsaved_project() {
                                            " *"
                                        } else {
                                            ""
                                        }
                                    ))
                                    .strong(),
                                );
                                if ui
                                    .add_enabled(
                                        self.session.snapshot().is_some()
                                            && !self.project_busy()
                                            && !self.export_busy()
                                            && !self.workspace.busy,
                                        egui::Button::new("\u{d7}").frame(false),
                                    )
                                    .on_hover_text(label("close-document"))
                                    .clicked()
                                {
                                    self.workspace.close_document = true;
                                }
                            });
                        });
                    self.workspace.layout.document_tab = Some(tab.response.rect);
                    ui.painter().hline(
                        tab.response.rect.x_range(),
                        tab.response.rect.bottom(),
                        egui::Stroke::new(2., studio::AMBER),
                    );
                    if ui
                        .button("+")
                        .on_hover_text(crate::input_locale::labels::text("native-open-image"))
                        .clicked()
                    {
                        self.browse_image();
                    }
                });
            });
        if self.workspace.about {
            egui::Window::new(label("about"))
                .open(&mut self.workspace.about)
                .show(root, |ui| {
                    ui.heading("Texelith");
                    ui.label(label("studio-subtitle"));
                    ui.label("MIT");
                });
        }
        if self.workspace.view_settings {
            let mut open = true;
            egui::Window::new(label("view-settings"))
                .open(&mut open)
                .show(root, |ui| self.viewport_settings(ui));
            self.workspace.view_settings = open;
        }
        if self.workspace.close_document {
            if self.session.has_unsaved_project() {
                egui::Modal::new(egui::Id::unique("close-document")).show(root.ctx(), |ui| {
                    ui.heading(crate::input_locale::labels::text("dirty-title"));
                    ui.label(crate::input_locale::labels::text("dirty-message"));
                    ui.horizontal(|ui| {
                        if ui.button(text("save")).clicked() {
                            self.input_action(InputAction::SaveProject { save_as: false });
                        }
                        if ui
                            .button(crate::input_locale::labels::text("discard"))
                            .clicked()
                        {
                            self.close_studio_document();
                        }
                        if ui.button(text("cancel")).clicked() {
                            self.workspace.close_document = false;
                        }
                    });
                });
            } else {
                self.close_studio_document();
            }
        }
    }

    fn close_studio_document(&mut self) {
        self.action(UiAction::FocusLost);
        self.session = Session::new();
        self.adapter.sync(&self.session);
        self.workspace.clear_thumbnails();
        self.workspace.close_document = false;
        self.projects.path = None;
        self.path.clear();
        self.preview.pan = egui::Vec2::ZERO;
        self.preview.zoom = 100.;
        self.render_revision += 1;
    }
    pub(crate) fn project_batch_dialogs(&mut self, ctx: &egui::Context) {
        if self.projects.save_picker {
            egui::Modal::new(egui::Id::unique("project-save-as")).show(ctx, |ui| {
                ui.set_max_width(500.0);
                ui.heading(text("save-as"));
                ui.label(text("project-path"));
                ui.text_edit_singleline(&mut self.projects.save_path);
                if !self.projects.feedback.is_empty() {
                    ui.label(&self.projects.feedback);
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !self.projects.save_path.is_empty() && !self.projects.saving,
                            egui::Button::new(text("save")),
                        )
                        .clicked()
                    {
                        match self.save_project_to(self.projects.save_path.clone().into(), true) {
                            Ok(()) => self.projects.save_picker = false,
                            Err(e) => self.projects.feedback = e,
                        }
                    }
                    if ui.button(text("cancel")).clicked() {
                        self.projects.save_picker = false;
                    }
                });
            });
            return;
        }
        if let Some(issue) = self.project_asset_issues().first().cloned() {
            egui::Modal::new(egui::Id::unique("project-relink")).show(ctx, |ui| {
                ui.set_max_width(500.0);
                ui.heading(text(if issue.changed {
                    "project-changed-asset"
                } else {
                    "project-missing-asset"
                }));
                ui.label(issue.path.to_string_lossy());
                ui.label(text("project-relink-explanation"));
                ui.text_edit_singleline(&mut self.projects.relink_path);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !self.projects.relink_path.is_empty(),
                            egui::Button::new(text("project-relink")),
                        )
                        .clicked()
                        && let Err(e) = self.resolve_project_asset(AssetChoice::Relink(
                            self.projects.relink_path.clone().into(),
                        ))
                    {
                        self.projects.feedback = e;
                    }
                    if issue.changed
                        && ui.button(text("project-use-changed")).clicked()
                        && let Err(e) = self.resolve_project_asset(AssetChoice::UseChanged)
                    {
                        self.projects.feedback = e;
                    }
                    if ui.button(text("cancel")).clicked() {
                        self.cancel_project_open();
                    }
                });
            });
            return;
        }
        if self.projects.batch_open {
            let mut open = true;
            egui::Window::new(text("batch-title"))
                .id(egui::Id::unique("project-batch-panel"))
                .open(&mut open)
                .default_width(560.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(440.0)
                        .show(ui, |ui| {
                            if !self.input.batch.is_empty() {
                                ui.label(format!(
                                    "{}: {}",
                                    text("batch-files"),
                                    self.input.batch.len()
                                ));
                            }
                            ui.label(text("batch-folder"));
                            ui.text_edit_singleline(&mut self.projects.batch_folder);
                            ui.label(text("export-destination"));
                            ui.text_edit_singleline(&mut self.projects.batch_destination);
                            ui.label(text("batch-manifest"));
                            ui.text_edit_singleline(&mut self.projects.batch_manifest);
                            ui.checkbox(&mut self.projects.batch_continue, text("batch-continue"));
                            let busy = self.batch_status.busy;
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .add_enabled(
                                        !busy
                                            && !self.projects.batch_destination.is_empty()
                                            && (!self.input.batch.is_empty()
                                                || !self.projects.batch_folder.is_empty()),
                                        egui::Button::new(text("batch-add")),
                                    )
                                    .clicked()
                                {
                                    let selection = if self.input.batch.is_empty() {
                                        crate::batch_jobs::InputSelection::Folder {
                                            path: self.projects.batch_folder.clone().into(),
                                            recursive: false,
                                        }
                                    } else {
                                        crate::batch_jobs::InputSelection::Files(
                                            self.input.batch.clone(),
                                        )
                                    };
                                    let destination =
                                        PathBuf::from(&self.projects.batch_destination);
                                    let manifest = if self.projects.batch_manifest.is_empty() {
                                        destination.join("batch.texelith-job.json")
                                    } else {
                                        self.projects.batch_manifest.clone().into()
                                    };
                                    if let Err(e) = self.queue_batch(
                                        selection,
                                        destination,
                                        manifest,
                                        self.projects.batch_continue,
                                    ) {
                                        self.projects.feedback = e;
                                    } else {
                                        self.input.batch.clear();
                                    }
                                }
                                for (id, key) in [
                                    (CommandId::BatchStart, "batch-start"),
                                    (CommandId::BatchCancel, "batch-cancel"),
                                    (CommandId::BatchRetryFailed, "batch-retry"),
                                ] {
                                    let enabled = if id == CommandId::BatchCancel {
                                        busy || self
                                            .session
                                            .batch()
                                            .is_some_and(BatchJob::can_cancel)
                                    } else {
                                        !busy
                                            && self
                                                .session
                                                .availability()
                                                .iter()
                                                .any(|a| a.id == id && a.enabled)
                                    };
                                    if ui
                                        .add_enabled(enabled, egui::Button::new(text(key)))
                                        .clicked()
                                    {
                                        self.action(UiAction::Invoke {
                                            origin: crate::command_adapter::ActionOrigin::Menu,
                                            id,
                                        });
                                    }
                                }
                                if ui
                                    .add_enabled(
                                        !busy && !self.projects.batch_manifest.is_empty(),
                                        egui::Button::new(text("batch-recover")),
                                    )
                                    .clicked()
                                {
                                    self.projects.batch_recover =
                                        Some(self.projects.batch_manifest.clone().into());
                                }
                            });
                            if self.projects.restart_offer && !busy {
                                ui.label(text("batch-restart-question"));
                                ui.horizontal(|ui| {
                                    if ui.button(text("batch-resume")).clicked() {
                                        match self
                                            .resolve_batch_restart(RestartChoice::RetryIncomplete)
                                        {
                                            Ok(()) => {
                                                self.projects.restart_offer = false;
                                            }
                                            Err(e) => self.projects.feedback = e,
                                        }
                                    }
                                    if ui.button(text("cancel")).clicked()
                                        && let Err(e) =
                                            self.resolve_batch_restart(RestartChoice::KeepStopped)
                                    {
                                        self.projects.feedback = e;
                                    }
                                });
                            }
                            if let Some(error) = &self.batch_status.error {
                                ui.label(error);
                            }
                            if let Some(job) = self.session.batch() {
                                let completed = job
                                    .items()
                                    .iter()
                                    .filter(|i| i.state == BatchState::Committed)
                                    .count();
                                ui.add(
                                    egui::ProgressBar::new(
                                        completed as f32 / job.items().len().max(1) as f32,
                                    )
                                    .text(format!("{completed} / {}", job.items().len())),
                                );
                                for (index, item) in job.items().iter().enumerate() {
                                    let key = match item.state {
                                        BatchState::Queued => "batch-queued",
                                        BatchState::Loading => "batch-loading",
                                        BatchState::Evaluating => "batch-evaluating",
                                        BatchState::Encoding => "batch-encoding",
                                        BatchState::Committed => "batch-committed",
                                        BatchState::Failed => "batch-failed",
                                        BatchState::Cancelled => "batch-cancelled",
                                    };
                                    ui.label(format!(
                                        "{} — {}",
                                        std::path::Path::new(&item.source.0)
                                            .file_name()
                                            .map(|p| p.to_string_lossy().into_owned())
                                            .unwrap_or_else(|| format!(
                                                "{} {}",
                                                text("batch-item"),
                                                index + 1
                                            )),
                                        text(key)
                                    ));
                                    if let Some(error) = &item.error {
                                        ui.label(error);
                                    }
                                }
                            }
                        });
                });
            self.projects.batch_open = open;
        }
    }
}
