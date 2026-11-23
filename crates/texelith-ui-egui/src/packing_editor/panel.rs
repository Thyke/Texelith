use super::*;

fn map_name(map: MapKind) -> String {
    crate::material_ui::labels::text(crate::material_ui::map_label(map))
}
fn source_name(source: PresetChannelSource) -> String {
    match source {
        PresetChannelSource::Map { map, .. } => map_name(map),
        PresetChannelSource::Constant(_) => text("packing-constant"),
    }
}
fn format_name(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Png8 => "PNG 8",
        ExportFormat::Png16 => "PNG 16",
        ExportFormat::Exr32 => "EXR 32",
        ExportFormat::DdsBc5 => "DDS BC5",
    }
}
fn constant() -> PresetChannelSpec {
    PresetChannelSpec {
        source: PresetChannelSource::Constant(1.0),
        invert: false,
        transfer: Transfer::Linear,
    }
}
impl Controls {
    pub(crate) fn packing_button(&mut self, ui: &mut egui::Ui) {
        let response = ui.add_enabled(
            self.session.snapshot().is_some(),
            egui::Button::new(text("packing-title")),
        );
        self.packing.layout.open = Some(response.rect);
        if response.clicked()
            && let Err(error) = self.open_packing_editor()
        {
            self.packing.feedback = error;
        }
        if self.packing.open
            && self
                .packing
                .draft
                .as_ref()
                .is_some_and(|p| p.profile.id != self.export_options.preset)
            && let Err(error) = self.open_packing_editor()
        {
            self.packing.feedback = error;
        }
    }
    pub(crate) fn show_packing_editor(&mut self, ctx: &egui::Context) {
        if !self.packing.open {
            return;
        }
        let mut open = true;
        let mut save = false;
        let mut reload = false;
        self.packing.layout.channel_sources.clear();
        self.packing.layout.inverts.clear();
        egui::Window::new(text("packing-title")).id(egui::Id::unique("packing-editor"))
            .open(&mut open).default_width(680.0).resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height((ctx.content_rect().height() - 110.0).max(180.0)).show(ui, |ui| {
                    let Some(recipe) = self.packing.draft.as_mut() else { return };
                    ui.horizontal_wrapped(|ui| {
                        ui.label(text("packing-output"));
                        self.packing.selected_output = self.packing.selected_output.min(recipe.outputs.len().saturating_sub(1));
                        egui::ComboBox::from_id_salt("packing-output").selected_text(&recipe.outputs[self.packing.selected_output].name).show_ui(ui, |ui| {
                            for (index, output) in recipe.outputs.iter().enumerate() {
                                ui.selectable_value(&mut self.packing.selected_output, index, &output.name);
                            }
                        });
                        if ui.add_enabled(recipe.outputs.len() < 32, egui::Button::new(text("packing-add"))).clicked() {
                            recipe.outputs.push(PresetOutput { name: format!("packed{}", recipe.outputs.len() + 1), format: ExportFormat::Png8,
                                channels: vec![constant(); 4], quantization: Some(Default::default()), import_hints: None });
                            self.packing.selected_output = recipe.outputs.len() - 1;
                        }
                        if ui.add_enabled(recipe.outputs.len() > 1, egui::Button::new(text("packing-remove"))).clicked() {
                            recipe.outputs.remove(self.packing.selected_output);
                            self.packing.selected_output = 0;
                        }
                    });
                    let output = &mut recipe.outputs[self.packing.selected_output];
                    let before = output.clone();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(text("export-name"));
                        ui.add(egui::TextEdit::singleline(&mut output.name).desired_width(130.0));
                        egui::ComboBox::from_id_salt("packing-format").selected_text(format_name(output.format)).show_ui(ui, |ui| {
                            for format in [ExportFormat::Png8, ExportFormat::Png16, ExportFormat::Exr32, ExportFormat::DdsBc5] {
                                ui.selectable_value(&mut output.format, format, format_name(format));
                            }
                        });
                        let mut count = output.channels.len();
                        egui::ComboBox::from_id_salt("packing-count").selected_text(format!("{count} {}", text("packing-channels"))).show_ui(ui, |ui| {
                            for n in 1..=4 { ui.selectable_value(&mut count, n, n.to_string()); }
                        });
                        output.channels.resize(count, constant());
                    });
                    ui.separator();
                    ui.small(text("packing-channel-help"));
                    for (index, channel) in output.channels.iter_mut().enumerate() {
                        ui.push_id(index, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.strong(["R", "G", "B", "A"][index]);
                                let response = egui::ComboBox::from_id_salt("source").width(115.0).selected_text(source_name(channel.source)).show_ui(ui, |ui| {
                                    let maps = [MapKind::Albedo, MapKind::Height, MapKind::Normal, MapKind::Ao, MapKind::Roughness, MapKind::Metallic, MapKind::Edge, MapKind::Curvature, MapKind::Cavity, MapKind::Mask];
                                    for map in maps {
                                        if ui.selectable_label(matches!(channel.source, PresetChannelSource::Map { map: selected, .. } if selected == map), map_name(map)).clicked() {
                                            channel.source = PresetChannelSource::Map { map, component: Component::R, missing: MissingMap::Error };
                                        }
                                    }
                                    if ui.selectable_label(matches!(channel.source, PresetChannelSource::Constant(_)), text("packing-constant")).clicked() { channel.source = PresetChannelSource::Constant(1.0); }
                                }).response;
                                self.packing.layout.channel_sources.push(response.rect);
                                match &mut channel.source {
                                    PresetChannelSource::Constant(value) => { ui.add(egui::DragValue::new(value).range(0.0..=1.0).speed(0.01)); }
                                    PresetChannelSource::Map { map, component, .. } => {
                                        egui::ComboBox::from_id_salt("component").width(45.0).selected_text(format!("{component:?}")).show_ui(ui, |ui| {
                                            for c in [Component::R, Component::G, Component::B, Component::A] {
                                                if *map == MapKind::Albedo || (*map == MapKind::Normal && c != Component::A) || c == Component::R {
                                                    ui.selectable_value(component, c, format!("{c:?}"));
                                                }
                                            }
                                        });
                                    }
                                }
                                let response = ui.checkbox(&mut channel.invert, text("packing-invert"));
                                self.packing.layout.inverts.push(response.rect);
                                egui::ComboBox::from_id_salt("transfer").width(70.0).selected_text(match channel.transfer { Transfer::Linear => text("packing-linear"), Transfer::Srgb => "sRGB".into() }).show_ui(ui, |ui| {
                                    ui.selectable_value(&mut channel.transfer, Transfer::Linear, text("packing-linear"));
                                    ui.selectable_value(&mut channel.transfer, Transfer::Srgb, "sRGB");
                                });
                            });
                        });
                    }
                    if *output != before {
                        output.import_hints = None;
                        output.quantization = matches!(output.format, ExportFormat::Png8 | ExportFormat::Png16).then_some(Default::default());
                    }
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(text("packing-template"));
                        if ui.add(egui::TextEdit::singleline(&mut recipe.naming).desired_width(240.0)).changed() { self.native_files.naming = None; }
                    });
                    ui.small(text("packing-tokens"));
                    ui.horizontal_wrapped(|ui| {
                        ui.label(text("export-size"));
                        let mut mode = match self.export_options.size { SizePolicy::Keep => 0, SizePolicy::Resize { .. } => 1, SizePolicy::Pad { .. } => 2 };
                        egui::ComboBox::from_id_salt("packing-size-policy").selected_text(text(["export-source-size", "packing-resize", "packing-pad"][mode])).show_ui(ui, |ui| {
                            for (i, key) in ["export-source-size", "packing-resize", "packing-pad"].into_iter().enumerate() { ui.selectable_value(&mut mode, i, text(key)); }
                        });
                        let (mut width, mut height) = match self.export_options.size { SizePolicy::Keep => (1024, 1024), SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => (width, height) };
                        if mode != 0 { ui.add(egui::DragValue::new(&mut width).range(0..=8192)); ui.label("×"); ui.add(egui::DragValue::new(&mut height).range(0..=8192)); }
                        self.export_options.size = match mode { 1 => SizePolicy::Resize { width, height }, 2 => SizePolicy::Pad { width, height }, _ => SizePolicy::Keep };
                    });
                    let validation = self.packing_plan();
                    match &validation {
                        Ok(plan) => {
                            if plan.outputs.iter().any(|o| !o.semantics().warnings.is_empty()) { ui.colored_label(egui::Color32::YELLOW, text("packing-mixed-transfer")); }
                            ui.strong(text("packing-preview"));
                            if let Ok(names) = names(plan, &self.export_options.material_name) { for name in names { ui.monospace(name); } }
                        }
                        Err(error) => { ui.colored_label(egui::Color32::LIGHT_RED, format!("{}: {error}", text("packing-invalid"))); }
                    }
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(text("packing-preset-name"));
                        let name = ui.add(egui::TextEdit::singleline(&mut self.packing.save_name).desired_width(150.0));
                        if self.packing.focus_name { name.request_focus(); self.packing.focus_name = false; }
                        let response = ui.add_enabled(validation.is_ok() && !self.packing.busy && !self.packing.load_failed, egui::Button::new(text("packing-save")));
                        self.packing.layout.save = Some(response.rect);
                        save = response.clicked();
                        reload = ui.add_enabled(!self.packing.busy, egui::Button::new(text("packing-reload"))).clicked();
                        if self.packing.busy { ui.spinner(); }
                    });
                    if !self.packing.feedback.is_empty() { ui.label(&self.packing.feedback); }
                });
            });
        self.packing.open = open;
        if save && let Err(error) = self.save_packing_preset(&self.packing.save_name.clone()) {
            self.packing.feedback = error;
        }
        if reload && let Err(error) = self.reload_packing_presets() {
            self.packing.feedback = error;
        }
    }
}
