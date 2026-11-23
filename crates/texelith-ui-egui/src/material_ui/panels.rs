use super::*;
use crate::material_ui::labels::text;
use crate::material_ui::{self, ImportKind, MAPS};
use texelith_core::{MaterialNodeParams, ParameterSchema};
impl Controls {
    pub(super) fn show_material(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        let editing = ctx.egui_wants_keyboard_input();
        if !self.scripted {
            let (undo, redo, escape, focused, palette) = ctx.input(|i| {
                (
                    i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                    i.modifiers.command
                        && (i.key_pressed(egui::Key::Y)
                            || (i.modifiers.shift && i.key_pressed(egui::Key::Z))),
                    i.key_pressed(egui::Key::Escape),
                    i.focused,
                    i.modifiers.command && i.key_pressed(egui::Key::K),
                )
            });
            if undo && !editing {
                self.action(UiAction::UndoShortcut);
            }
            if redo && !editing {
                self.action(UiAction::RedoShortcut);
            }
            if escape {
                self.action(UiAction::Escape);
                self.palette = false;
            }
            if !focused {
                self.action(UiAction::FocusLost);
            }
            if palette {
                self.palette = !self.palette;
            }
            if !editing && !self.palette {
                for (key, map) in [
                    egui::Key::Num1,
                    egui::Key::Num2,
                    egui::Key::Num3,
                    egui::Key::Num4,
                    egui::Key::Num5,
                    egui::Key::Num6,
                    egui::Key::Num7,
                ]
                .into_iter()
                .zip(MAPS)
                {
                    if ctx.input(|i| i.key_pressed(key))
                        && self
                            .session
                            .snapshot()
                            .is_some_and(|s| material_ui::supported(&s, map))
                    {
                        self.action(UiAction::Request(CommandRequest::SelectMap { map }));
                    }
                }
            }
            if let Some(path) =
                ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()))
            {
                self.open_image(path);
            }
        }
        egui::Panel::bottom("material_export").show(root, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(text("PNG destination"));
                let destination =
                    ui.add(egui::TextEdit::singleline(&mut self.destination).desired_width(280.0));
                if self.focus == Some(CommandId::SavePng) {
                    destination.request_focus();
                    self.focus = None;
                }
                let selected = self
                    .session
                    .snapshot()
                    .map(|s| s.graph.selected_output_ref().port.map_kind());
                if ui
                    .add_enabled(
                        selected.is_some() && !self.destination.is_empty(),
                        egui::Button::new(text("Save selected PNG")),
                    )
                    .clicked()
                {
                    self.save_png(self.destination.clone().into(), selected.unwrap());
                }
                ui.small(text("Height: PNG16 | other maps: PNG8 | set export in M3"));
            });
        });
        egui::Panel::left("material_maps")
            .default_size(155.0)
            .show(root, |ui| {
                ui.heading(text("Maps"));
                ui.small(text("1-7 to select"));
                if let Some(s) = self.session.snapshot() {
                    for map in MAPS {
                        let enabled = material_ui::supported(&s, map);
                        let selected = s.graph.selected_output_ref().port.map_kind() == map;
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::new(text(material_ui::map_label(map)))
                                    .selected(selected)
                                    .min_size(egui::vec2(125.0, 34.0)),
                            )
                            .on_disabled_hover_text(
                                "Unavailable for this source; normal integration is not merged yet",
                            )
                            .clicked()
                        {
                            self.action(UiAction::Request(CommandRequest::SelectMap { map }));
                        }
                    }
                    ui.separator();
                    ui.small(text("Procedural material"));
                    let mut tiles = self.session.view_state().tiles;
                    egui::ComboBox::from_id_salt("tile-count")
                        .selected_text(format!("{} x {} tiles", tiles.count, tiles.count))
                        .show_ui(ui, |ui| {
                            for count in [1, 2, 3] {
                                if ui
                                    .selectable_value(
                                        &mut tiles.count,
                                        count,
                                        format!("{count} x {count}"),
                                    )
                                    .changed()
                                {
                                    self.action(UiAction::Request(CommandRequest::SetTiles {
                                        tiles,
                                    }));
                                }
                            }
                        });
                    let mut heatmap = self.heatmap;
                    if ui.checkbox(&mut heatmap, text("Seam heatmap")).changed() {
                        self.action(UiAction::Request(CommandRequest::InspectSeams {
                            enabled: heatmap,
                        }));
                    }
                    let (mut seamless, mut mask) = material_ui::options(&s.graph);
                    let changed = ui.checkbox(&mut seamless, text("Seamless")).changed()
                        | ui.checkbox(&mut mask, text("Use painted mask")).changed();
                    if changed {
                        self.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                            seamless,
                            painted_mask: mask,
                        }));
                    }
                    ui.small(text("Mask applies to roughness and metallic"));
                    for map in [MapKind::Mask, MapKind::Curvature, MapKind::Cavity] {
                        if material_ui::supported(&s, map)
                            && ui.small_button(text(material_ui::map_label(map))).clicked()
                        {
                            self.action(UiAction::Request(CommandRequest::SelectMap { map }));
                        }
                    }
                }
            });
        egui::Panel::right("material_inspector")
            .default_size(320.0)
            .show(root, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.material_inspector(ui));
            });
        if self.palette {
            egui::Window::new("Commands")
                .collapsible(false)
                .show(root, |ui| {
                    ui.text_edit_singleline(&mut self.search);
                    for e in self
                        .adapter
                        .palette(&self.session)
                        .into_iter()
                        .filter(|entry| {
                            matches!(
                                entry.id,
                                CommandId::NewFromImage
                                    | CommandId::SetParameter
                                    | CommandId::BeginGesture
                                    | CommandId::CommitGesture
                                    | CommandId::CancelGesture
                                    | CommandId::Undo
                                    | CommandId::Redo
                                    | CommandId::SavePng
                                    | CommandId::SelectMap
                                    | CommandId::ConfigureMaterial
                                    | CommandId::InspectSeams
                            )
                        })
                    {
                        if format!("{} {}", e.id.as_str(), e.label)
                            .to_lowercase()
                            .contains(&self.search.to_lowercase())
                            && ui
                                .add_enabled(e.enabled, egui::Button::new(e.label))
                                .clicked()
                        {
                            self.invoke(e.id, ActionOrigin::Palette);
                            self.palette = false;
                        }
                    }
                });
        }
        if ctx.input(|i| !i.pointer.any_down()) {
            self.suppressed_slider = None;
            self.import_interrupted_drag = false;
        }
    }
    fn material_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading(text("Material inspector"));
        let mut turkish = crate::material_ui::labels::is_turkish();
        if ui.checkbox(&mut turkish, "Türkçe").changed() {
            crate::material_ui::labels::set_turkish(turkish);
        }
        ui.label(text("Local image path"));
        let path_response = ui.text_edit_singleline(&mut self.path);
        if self.focus == Some(CommandId::NewFromImage) {
            path_response.request_focus();
            self.focus = None;
        }
        egui::ComboBox::from_id_salt("import-kind")
            .selected_text(format!("{:?}", self.import_kind))
            .show_ui(ui, |ui| {
                for value in [ImportKind::Photo, ImportKind::Height, ImportKind::Normal] {
                    ui.selectable_value(&mut self.import_kind, value, format!("{value:?}"));
                }
            });
        if self.import_kind == ImportKind::Normal {
            egui::ComboBox::from_id_salt("import-convention")
                .selected_text(format!("{:?}", self.normal_convention))
                .show_ui(ui, |ui| {
                    for value in [
                        texelith_core::NormalConvention::OpenGl,
                        texelith_core::NormalConvention::DirectX,
                    ] {
                        ui.selectable_value(
                            &mut self.normal_convention,
                            value,
                            format!("{value:?}"),
                        );
                    }
                });
        }
        if ui.button(text("New material from image")).clicked() && !self.path.is_empty() {
            self.open_image(self.path.clone().into());
        }
        ui.small(text(
            "Photo: PNG/JPEG, sRGB -> linear. Height: PNG16 / EXR32. Normal: RGB8 PNG.",
        ));
        ui.small(text(
            "Alpha, ICC and EXIF ignored. Successful import clears history.",
        ));
        if let Some(snapshot) = self.session.snapshot() {
            let selected = snapshot.graph.selected_output_ref();
            ui.separator();
            ui.heading(text(material_ui::map_label(selected.port.map_kind())));
            if let Some(node) = snapshot.graph.material_nodes().iter().find(|node| {
                matches!(
                    node.params,
                    MaterialNodeParams::ColorSource(_)
                        | MaterialNodeParams::HeightSource(_)
                        | MaterialNodeParams::NormalSource { .. }
                )
            }) {
                let source_label = match node.params {
                    MaterialNodeParams::ColorSource(_) => "Source: sRGB photo",
                    MaterialNodeParams::HeightSource(_) => "Source: linear height",
                    _ => "Source: encoded normal",
                };
                ui.small(text(source_label));
                if let MaterialNodeParams::NormalSource { convention, .. } = node.params {
                    ui.small(format!("Source normal: {convention:?}"));
                    ui.colored_label(egui::Color32::YELLOW,
                        "Height integration is not merged. Normal / Edge previews remain available; no convergence result is claimed.");
                }
            }
            if selected.port == texelith_core::OutputPort::Normal {
                ui.small(text("Normal preview and legacy PNG: OpenGL +Y"));
            }
            let mut needed = std::collections::BTreeSet::new();
            let mut stack = vec![selected.node];
            while let Some(id) = stack.pop() {
                if needed.insert(id)
                    && let Some(node) = snapshot.graph.material_node(id)
                {
                    stack.extend(node.inputs.iter().map(|edge| edge.source.node));
                }
            }
            // The selected producer appears first; dependencies remain inspectable.
            let mut nodes: Vec<_> = snapshot
                .graph
                .material_nodes()
                .iter()
                .filter(|node| needed.contains(&node.id))
                .collect();
            nodes.sort_by_key(|node| (node.id != selected.node, std::cmp::Reverse(node.id)));
            for node in nodes {
                let title = match node.params {
                    MaterialNodeParams::HeightFromAlbedo { .. } => "Height bands",
                    MaterialNodeParams::NormalFromHeight { .. } => "Normal filter",
                    MaterialNodeParams::HorizonAo { .. } => "Ambient occlusion",
                    MaterialNodeParams::Curvature { .. } => "Edge / cavity",
                    MaterialNodeParams::Roughness { .. } => "Roughness curve",
                    MaterialNodeParams::Metallic { .. } => "Metallic selection",
                    MaterialNodeParams::Levels { .. } => "Height remap",
                    MaterialNodeParams::Seamless { .. } => "Seamless method",
                    _ => continue,
                };
                egui::CollapsingHeader::new(text(title))
                    .id_salt(node.id.0)
                    .default_open(node.id == selected.node)
                    .open((node.id == selected.node).then_some(true))
                    .show(ui, |ui| {
                        for descriptor in &texelith_core::MATERIAL_PARAMETER_DESCRIPTORS {
                            if descriptor.id == ParameterId::Boundary
                                && matches!(node.params, MaterialNodeParams::Levels { .. })
                            {
                                continue;
                            }
                            if let Ok(value) = node.params.parameter(descriptor.id) {
                                let address = ParameterAddress {
                                    node: node.id,
                                    parameter: descriptor.id,
                                };
                                self.material_parameter(
                                    ui,
                                    address,
                                    value,
                                    &descriptor.value_schema,
                                );
                            }
                        }
                    });
            }
            if let Some(mask) = snapshot
                .graph
                .material_outputs()
                .and_then(|m| m.get(&MapKind::Mask))
                && ui.button(text("Paint center dab (undoable)")).clicked()
            {
                let count = snapshot
                    .graph
                    .material_node(mask.node)
                    .and_then(|node| match &node.params {
                        MaterialNodeParams::PaintMask { strokes, .. } => Some(strokes.len()),
                        _ => None,
                    })
                    .unwrap_or(0);
                let stroke_id = texelith_core::StrokeId(count as u64 + 1);
                self.action(UiAction::Request(CommandRequest::BeginStroke {
                    node: mask.node,
                    stroke_id,
                    brush: texelith_core::Brush {
                        radius: 0.2,
                        hardness: 0.5,
                        opacity: 1.0,
                        mode: texelith_core::MaskMode::Replace,
                    },
                    sample: texelith_core::BrushSample {
                        uv: [0.5, 0.5],
                        pressure: 1.0,
                    },
                }));
                self.action(UiAction::Request(CommandRequest::CommitStroke {
                    stroke_id,
                }));
            }
        }
        ui.separator();
        let entries = self.adapter.palette(&self.session);
        ui.horizontal(|ui| {
            for id in [CommandId::Undo, CommandId::Redo] {
                let entry = entries.iter().find(|entry| entry.id == id).unwrap();
                if ui
                    .add_enabled(entry.enabled, egui::Button::new(entry.label))
                    .clicked()
                {
                    self.invoke(id, ActionOrigin::Menu);
                }
            }
        });
        if !self.message.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.message);
        }
        if ui.button(text("Commands (Ctrl+K)")).clicked() {
            self.palette = true;
        }
    }

    fn material_parameter(
        &mut self,
        ui: &mut egui::Ui,
        address: ParameterAddress,
        mut value: ParameterValue,
        schema: &ParameterSchema,
    ) {
        let label = text(&format!("{:?}", address.parameter));
        let mut responses = Vec::new();
        ui.push_id((address.node.0, format!("{:?}", address.parameter)), |ui| {
            ui.label(&label);
            match (&mut value, schema) {
                (ParameterValue::Float(v), ParameterSchema::Float { min, max, .. }) => {
                    responses.push(if max.abs() > 10000.0 || min.abs() > 10000.0 {
                        ui.add(egui::DragValue::new(v).speed(0.01).range(*min..=*max))
                    } else {
                        ui.add(egui::Slider::new(v, *min..=*max))
                    });
                }
                (ParameterValue::UInt(v), ParameterSchema::UInt { min, max, .. }) => {
                    responses.push(ui.add(egui::DragValue::new(v).range(*min..=*max)));
                }
                (ParameterValue::Bool(v), _) => {
                    responses.push(ui.checkbox(v, label));
                }
                (ParameterValue::Color(v), _) => {
                    responses.push(ui.color_edit_button_rgb(v));
                }
                (
                    ParameterValue::FloatVector(v),
                    ParameterSchema::FloatVector {
                        min,
                        max,
                        min_len,
                        max_len,
                        ..
                    },
                ) => {
                    ui.horizontal(|ui| {
                        let mut add =
                            ui.add_enabled(v.len() < *max_len, egui::Button::new(text("Add band")));
                        if add.clicked() {
                            v.push(1.0);
                            add.mark_changed();
                        }
                        responses.push(add);
                        let mut remove = ui.add_enabled(
                            v.len() > *min_len,
                            egui::Button::new(text("Remove band")),
                        );
                        if remove.clicked() {
                            v.pop();
                            remove.mark_changed();
                        }
                        responses.push(remove);
                    });
                    for (i, x) in v.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Band {}", i + 1));
                            responses.push(
                                ui.add(egui::DragValue::new(x).speed(0.01).range(*min..=*max)),
                            );
                        });
                    }
                }
                (ParameterValue::Curve(v), ParameterSchema::Curve { min, max, .. }) => {
                    for point in v {
                        ui.horizontal(|ui| {
                            for x in point {
                                responses.push(
                                    ui.add(egui::DragValue::new(x).speed(0.01).range(*min..=*max)),
                                );
                            }
                        });
                    }
                }
                (ParameterValue::Boundary(v), ParameterSchema::Boundary { choices, .. }) => {
                    for choice in *choices {
                        responses.push(ui.selectable_value(v, *choice, format!("{choice:?}")));
                    }
                }
                (ParameterValue::Filter(v), ParameterSchema::Filter { choices, .. }) => {
                    for choice in *choices {
                        responses.push(ui.selectable_value(v, *choice, format!("{choice:?}")));
                    }
                }
                (ParameterValue::Convention(v), ParameterSchema::Convention { choices, .. }) => {
                    for choice in *choices {
                        responses.push(ui.selectable_value(v, *choice, format!("{choice:?}")));
                    }
                }
                (ParameterValue::MaskMode(v), ParameterSchema::MaskMode { choices, .. }) => {
                    for choice in *choices {
                        responses.push(ui.selectable_value(v, *choice, format!("{choice:?}")));
                    }
                }
                (
                    ParameterValue::SeamlessMethod(v),
                    ParameterSchema::SeamlessMethod {
                        overlap,
                        grid,
                        seed,
                        ..
                    },
                ) => {
                    let mut offset = ui.selectable_label(
                        matches!(v, texelith_core::SeamlessMethod::OffsetBlend { .. }),
                        text("Offset blend"),
                    );
                    if offset.clicked() {
                        *v = texelith_core::SeamlessMethod::OffsetBlend { overlap: 0.25 };
                        offset.mark_changed();
                    }
                    responses.push(offset);
                    let mut splat = ui.selectable_label(
                        matches!(v, texelith_core::SeamlessMethod::Splat { .. }),
                        text("Periodic splats"),
                    );
                    if splat.clicked() {
                        *v = texelith_core::SeamlessMethod::Splat {
                            seed: 0,
                            grid: 4,
                            overlap: 0.25,
                        };
                        splat.mark_changed();
                    }
                    responses.push(splat);
                    match v {
                        texelith_core::SeamlessMethod::OffsetBlend { overlap: v } => responses
                            .push(ui.add(
                                egui::Slider::new(v, overlap[0]..=overlap[1]).text(text("Overlap")),
                            )),
                        texelith_core::SeamlessMethod::Splat {
                            seed: s,
                            grid: g,
                            overlap: v,
                        } => {
                            responses
                                .push(ui.add(egui::DragValue::new(s).range(seed[0]..=seed[1])));
                            responses.push(
                                ui.add(egui::Slider::new(g, grid[0]..=grid[1]).text(text("Grid"))),
                            );
                            responses.push(ui.add(
                                egui::Slider::new(v, overlap[0]..=overlap[1]).text(text("Overlap")),
                            ));
                        }
                        _ => {
                            ui.label(text("Histogram blend is not merged"));
                        }
                    }
                }
                _ => {
                    ui.label(text("Read-only parameter"));
                }
            }
        });
        if matches!(
            self.focus,
            Some(CommandId::SetParameter | CommandId::BeginGesture)
        ) && let Some(response) = responses.first()
        {
            response.request_focus();
            self.focus = None;
        }
        let allowed = !self.import_interrupted_drag
            && self.suppressed_slider != Some(address)
            && (self.scripted
                || ui
                    .ctx()
                    .input(|i| i.focused && !i.key_pressed(egui::Key::Escape)));
        if !allowed {
            return;
        }
        if responses.iter().any(|r| r.drag_started()) {
            self.action(UiAction::DragBegin { address });
        }
        if responses.iter().any(|r| r.changed()) {
            self.action(UiAction::Change { address, value });
        }
        if responses.iter().any(|r| r.drag_stopped()) {
            self.action(UiAction::DragRelease);
        }
    }
}
