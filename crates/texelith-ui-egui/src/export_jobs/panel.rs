use super::*;
use crate::{export_jobs::*, input_locale::labels::text};
use texelith_core::{BuiltinExportPresets, ChannelSource, ExportPresetTable, SizePolicy};

impl Controls {
    pub fn export_options(&self) -> &ExportOptions {
        &self.export_options
    }
    pub fn export_options_mut(&mut self) -> &mut ExportOptions {
        &mut self.export_options
    }
    pub fn export_status(&self) -> Option<&JobStatus> {
        self.export_status.as_ref()
    }
    pub fn export_busy(&self) -> bool {
        self.export_busy
    }
    pub(crate) fn export_message(&mut self, error: String) {
        self.native_files.export_detail = format!(
            "Export: {error}\nDestination: {}\nName: {}\nPreset: {}\nSize: {:?}",
            self.export_options.destination.display(),
            self.export_options.material_name,
            self.export_options.preset,
            self.export_options.size,
        );
        self.export_feedback = error.lines().next().unwrap_or(&error).to_owned();
    }
    pub(crate) fn set_export_status(&mut self, status: Option<JobStatus>, busy: bool) {
        self.session.set_export_active(busy);
        if let Some(failed) = &status
            && failed.stage == texelith_gpu::ExportStage::Failed
            && self
                .export_status
                .as_ref()
                .is_none_or(|old| old.job_id != failed.job_id || old.stage != failed.stage)
            && let Some(error) = &failed.error
        {
            self.export_feedback = error.lines().next().unwrap_or(error).to_owned();
            self.native_files.export_detail = format!(
                "Export job {}: {error}\nCaptured destination: {}\nOutput: {}",
                failed.job_id,
                failed.destination.display(),
                failed.output_name,
            );
        }
        if let Some(completed) = &status
            && completed.stage == texelith_gpu::ExportStage::Completed
            && self
                .export_status
                .as_ref()
                .is_none_or(|old| old.job_id != completed.job_id || old.stage != completed.stage)
        {
            self.native_files.toast = Some((
                format!(
                    "{}: {}",
                    text("native-exported"),
                    completed.destination.display()
                ),
                std::time::Instant::now(),
            ));
        }
        self.export_status = status;
        self.export_busy = busy;
    }
    pub(crate) fn take_export_effects(&mut self) -> Vec<(Effect, ExportOptions)> {
        let mut exports = Vec::new();
        let mut retained = Vec::new();
        for effect in std::mem::take(&mut self.effects) {
            match effect {
                Effect::Export { .. } => exports.push((
                    effect,
                    self.export_native_options
                        .pop_front()
                        .unwrap_or_else(|| self.export_options.clone()),
                )),
                Effect::CancelExport { .. } => exports.push((effect, self.export_options.clone())),
                other => retained.push(other),
            }
        }
        self.effects = retained;
        exports
    }
    pub fn request_material_export(&mut self) -> Result<(), String> {
        let result = self.capture_material_export();
        if let Err(error) = &result {
            self.export_message(error.clone());
        }
        result
    }
    fn capture_material_export(&mut self) -> Result<(), String> {
        if self.export_busy
            || self
                .effects
                .iter()
                .any(|e| matches!(e, Effect::Export { .. }))
        {
            return Err(text("export-busy"));
        }
        self.action(UiAction::FocusLost);
        let plan = self.packing_plan()?;
        self.export_options.destination =
            super::native_files::absolute_destination(&self.export_options.destination)?;
        self.action(UiAction::Request(CommandRequest::SetExportSettings {
            settings: plan,
        }));
        if !self.message.is_empty() {
            return Err(self.message.clone());
        }
        self.action(UiAction::Request(CommandRequest::RequestExport {}));
        if !self.message.is_empty() {
            return Err(self.message.clone());
        }
        self.export_feedback.clear();
        self.native_files.export_detail.clear();
        Ok(())
    }
    pub fn cancel_material_export(&mut self) {
        if let Some(status) = &self.export_status {
            self.action(UiAction::Request(CommandRequest::CancelExport {
                job_id: status.job_id,
            }));
        }
    }
    pub(crate) fn export_palette_entries(&mut self, ui: &mut egui::Ui) {
        for (id, enabled) in [
            (
                CommandId::RequestExport,
                self.session.snapshot().is_some() && !self.export_busy,
            ),
            (CommandId::CancelExport, self.export_busy),
            (
                CommandId::SaveExportPreset,
                self.session.snapshot().is_some(),
            ),
        ] {
            let label = crate::input_locale::labels::command(id);
            if label.to_lowercase().contains(&self.search.to_lowercase())
                && ui.add_enabled(enabled, egui::Button::new(label)).clicked()
            {
                self.invoke(id, ActionOrigin::Palette);
                self.palette = false;
            }
        }
    }
    pub(crate) fn show_material_export(&mut self, ui: &mut egui::Ui) {
        use crate::workspace::labels::text as label;
        let has_material = self.session.snapshot().is_some();
        ui.style_mut().animation_time = 0.;
        let previous_preset = self.export_options.preset.clone();
        egui::Grid::new("export-fields")
            .num_columns(2)
            .spacing([8., 6.])
            .show(ui, |ui| {
                ui.label(label("engine-preset"));
                egui::ComboBox::from_id_salt("export-preset")
                    .width((ui.available_width() - 8.).max(100.))
                    .selected_text(preset_name(&self.export_options.preset))
                    .show_ui(ui, |ui| {
                        for preset in BuiltinExportPresets.presets() {
                            ui.selectable_value(
                                &mut self.export_options.preset,
                                preset.profile.id.clone(),
                                preset_name(&preset.profile.id),
                            );
                        }
                        for preset in &self.packing.library.presets {
                            ui.selectable_value(
                                &mut self.export_options.preset,
                                preset.profile.id.clone(),
                                &preset.profile.id,
                            );
                        }
                    });
                ui.end_row();
                ui.label(label("output-folder"));
                ui.horizontal(|ui| {
                    let mut path = self
                        .export_options
                        .destination
                        .to_string_lossy()
                        .into_owned();
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut path)
                            .id_salt("material-export-destination")
                            .desired_width((ui.available_width() - 38.).max(60.)),
                    );
                    self.workspace.layout.destination = Some(field.rect);
                    if field.changed() {
                        self.export_options.destination = path.into();
                        self.native_files.folder_base = None;
                    }
                    if ui
                        .button("\u{2026}")
                        .on_hover_text(text("native-folder"))
                        .clicked()
                    {
                        self.browse_export_folder();
                    }
                });
                ui.end_row();
                ui.label(label("file-name"));
                let old_name = self.export_options.material_name.clone();
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut self.export_options.material_name)
                            .desired_width(ui.available_width()),
                    )
                    .changed()
                {
                    let base = self.native_files.folder_base.clone().unwrap_or_else(|| {
                        super::default_export_folder(&old_name)
                            .parent()
                            .unwrap()
                            .to_owned()
                    });
                    if self.export_options.destination == base.join(&old_name) {
                        self.export_options.destination =
                            base.join(&self.export_options.material_name);
                    }
                }
                ui.end_row();
            });
        if previous_preset != self.export_options.preset {
            let selected = self.export_options.preset.clone();
            if let Err(error) = self.select_export_preset(&selected) {
                self.export_feedback = error;
            }
            self.workspace.export_maps = None;
        }
        ui.add_space(2.);
        ui.label(label("maps-to-export"));
        let mut maps = self.selected_export_maps();
        let mut changed = false;
        egui::Grid::new("export-map-selection")
            .num_columns(3)
            .spacing([8., 3.])
            .show(ui, |ui| {
                for (i, map) in crate::material_ui::MAPS.into_iter().enumerate() {
                    let mut checked = maps.contains(&map);
                    let checkbox = ui.checkbox(
                        &mut checked,
                        crate::material_ui::labels::text(crate::material_ui::map_label(map)),
                    );
                    self.workspace
                        .layout
                        .export_channels
                        .insert(map, checkbox.rect);
                    if checkbox.changed() {
                        changed = true;
                        if checked {
                            maps.insert(map);
                        } else {
                            maps.remove(&map);
                        }
                    }
                    if i % 3 == 2 {
                        ui.end_row();
                    }
                }
            });
        if changed {
            self.workspace.export_maps = Some(maps);
        }
        ui.add_space(4.);
        let export = ui.add_enabled(
            has_material && !self.export_busy,
            egui::Button::new((
                egui::Atom::grow(),
                egui::RichText::new(label("export-action"))
                    .size(20.)
                    .strong()
                    .color(egui::Color32::BLACK),
                egui::Atom::grow(),
            ))
            .fill(crate::workspace::studio::AMBER)
            .min_size(egui::vec2(ui.available_width(), 42.)),
        );
        if has_material && !self.export_busy {
            let title_width = ui
                .painter()
                .layout_no_wrap(
                    label("export-action"),
                    egui::FontId::proportional(20.),
                    egui::Color32::BLACK,
                )
                .size()
                .x;
            let p = export.rect.center() - egui::vec2(title_width * 0.5 + 20., 0.);
            let stroke = egui::Stroke::new(1.6, egui::Color32::BLACK);
            for (a, b) in [
                (egui::vec2(0., 7.), egui::vec2(0., -8.)),
                (egui::vec2(0., -8.), egui::vec2(-5., -3.)),
                (egui::vec2(0., -8.), egui::vec2(5., -3.)),
                (egui::vec2(-8., 3.), egui::vec2(-8., 10.)),
                (egui::vec2(-8., 10.), egui::vec2(8., 10.)),
                (egui::vec2(8., 10.), egui::vec2(8., 3.)),
            ] {
                ui.painter().line_segment([p + a, p + b], stroke);
            }
        }
        self.workspace.layout.export_material = Some(export.rect);
        if export.clicked() {
            self.confirm_material_export();
        }
        let mut options_open =
            self.workspace.export_options_open || self.focus == Some(CommandId::SavePng);
        let options = ui
            .horizontal(|ui| {
                let options = ui.selectable_label(
                    options_open,
                    format!(
                        "{} {}",
                        if options_open { "-" } else { "\u{203a}" },
                        label("more-options")
                    ),
                );
                if self.export_busy && ui.button(text("export-cancel")).clicked() {
                    self.cancel_material_export();
                }
                if let Some(status) = self.export_status.clone()
                    && status.stage == texelith_gpu::ExportStage::Completed
                {
                    let folder = ui.link(text("native-open-folder"));
                    self.workspace.layout.open_export_folder = Some(folder.rect);
                    if folder.clicked() {
                        self.open_export_folder(&status.destination);
                    }
                }
                options
            })
            .inner;
        self.workspace.layout.export_options = Some(options.rect);
        if options.clicked() {
            options_open = !options_open;
        }
        if let Some(popup) = egui::Popup::from_response(&options)
            .align(egui::RectAlign::LEFT_END)
            .open_bool(&mut options_open)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .width(ui.available_width().max(240.))
            .show(|ui| {
                egui::ScrollArea::vertical()
                    .id_salt("export-options-scroll")
                    .max_height((ui.ctx().content_rect().height() - 80.).clamp(120., 340.))
                    .show(ui, |ui| {
                        self.show_export_options(ui);
                        self.show_png_export(ui);
                    })
                    .inner_rect
            })
        {
            self.workspace.layout.export_options_area = Some(popup.inner);
        }
        self.workspace.export_options_open = options_open;
        if let Some(status) = self.export_status.clone() {
            ui.label(text(if status.cancel_requested && !status.terminal() {
                "export-cancelling"
            } else {
                stage_key(status.stage)
            }));
            if !status.terminal() {
                ui.label(format!(
                    "{}/{} {}",
                    status.completed_items, status.total_items, status.output_name
                ));
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(30));
            }
        }
        if !self.export_feedback.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.export_feedback)
                .on_hover_text(&self.native_files.export_detail);
        }
        self.show_packing_editor(ui.ctx());
    }

    fn selected_export_maps(&self) -> std::collections::BTreeSet<MapKind> {
        if let Some(maps) = &self.workspace.export_maps {
            return maps.clone();
        }
        self.packing_plan()
            .map(|plan| {
                plan.outputs
                    .iter()
                    .flat_map(|o| o.channels.iter())
                    .filter_map(|c| match c.source {
                        ChannelSource::Map(r, _) => Some(r.port.map_kind()),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_else(|_| {
                crate::material_ui::MAPS
                    .into_iter()
                    .filter(|m| *m != MapKind::Edge)
                    .collect()
            })
    }
    fn show_export_options(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            self.packing_button(ui);
            let size_label = match self.export_options.size {
                SizePolicy::Keep => text("export-source-size"),
                SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
                    format!("{width} × {height}")
                }
            };
            egui::ComboBox::from_id_salt("export-size")
                .width(100.0)
                .selected_text(size_label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.export_options.size,
                        SizePolicy::Keep,
                        text("export-source-size"),
                    );
                    for size in [512, 1024, 2048, 4096] {
                        ui.selectable_value(
                            &mut self.export_options.size,
                            SizePolicy::Resize {
                                width: size,
                                height: size,
                            },
                            format!("{size} × {size}"),
                        );
                    }
                });
            egui::ComboBox::from_id_salt("export-format")
                .width(110.0)
                .selected_text(text(match self.export_options.format {
                    FormatChoice::Preset => "export-preset-formats",
                    FormatChoice::FloatDisplacement => "export-float-displacement",
                }))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.export_options.format,
                        FormatChoice::Preset,
                        text("export-preset-formats"),
                    );
                    ui.selectable_value(
                        &mut self.export_options.format,
                        FormatChoice::FloatDisplacement,
                        text("export-float-displacement"),
                    );
                });
            ui.label(text("packing-template"));
            let mut template = self
                .native_files
                .naming
                .clone()
                .or_else(|| self.packing.draft.as_ref().map(|p| p.naming.clone()))
                .or_else(|| {
                    self.custom_presets()
                        .iter()
                        .find(|p| p.profile.id == self.export_options.preset)
                        .map(|p| p.naming.clone())
                })
                .unwrap_or_else(|| "T_{name}_{map}".into());
            if ui
                .add(egui::TextEdit::singleline(&mut template).desired_width(150.0))
                .changed()
            {
                self.native_files.naming = Some(template);
            }
        });
        let has_material = self.session.snapshot().is_some();
        if has_material {
            match self.packing_names() {
                Ok(mut names) => {
                    names.push("assignment.json".into());
                    ui.horizontal_wrapped(|ui| {
                        ui.small(text("packing-preview"));
                        ui.small(names.join(" · "));
                    });
                }
                Err(error) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            }
        }
        ui.small(text("export-no-overwrite"));
    }
}
fn preset_name(id: &str) -> String {
    if id == "custom-draft" {
        return text("packing-draft");
    }
    (match id {
        "unreal-v1" => "Unreal",
        "unity-urp-v1" => "Unity URP",
        "unity-hdrp-v1" => "Unity HDRP",
        "godot-v1" => "Godot",
        "blender-v1" => "Blender",
        "gltf2-v1" => "glTF 2.0",
        _ => id.strip_prefix("custom-").unwrap_or(id),
    })
    .to_owned()
}
fn stage_key(stage: texelith_gpu::ExportStage) -> &'static str {
    use texelith_gpu::ExportStage::*;
    match stage {
        Evaluating => "export-evaluating",
        Readback => "export-readback",
        Encoding => "export-encoding",
        ReadyToPublish | Publishing => "export-publishing",
        Completed => "export-completed",
        Cancelled => "export-cancelled",
        Failed => "export-failed",
    }
}
