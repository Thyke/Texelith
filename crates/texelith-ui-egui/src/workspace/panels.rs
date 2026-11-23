use super::*;
use crate::material_ui::labels::text;
use crate::material_ui::{self, ImportKind, MAPS};
use texelith_core::{MaterialNodeParams, ParameterSchema};
#[path = "cards.rs"]
mod cards;
impl Controls {
    pub(super) fn show_material(&mut self, root: &mut egui::Ui) {
        self.show_workspace_with(
            root,
            Self::show_procedural_inspector,
            Self::show_simple_export,
        );
    }

    /// Compose inspector and export leaves while retaining the workspace's input,
    /// command routes, responsive panel sizing and persistent export placement.
    pub fn show_workspace_with(
        &mut self,
        root: &mut egui::Ui,
        inspector: impl FnOnce(&mut Self, &mut egui::Ui),
        export: impl FnOnce(&mut Self, &mut egui::Ui),
    ) {
        let ctx = root.ctx().clone();
        self.global_input(&ctx);
        self.workspace.layout = crate::workspace::Layout::default();
        self.project_batch_header(root);
        self.workspace_panels(root, inspector, export);
        if self.palette {
            egui::Window::new(crate::input_locale::labels::text("commands"))
                .collapsible(false)
                .show(root, |ui| {
                    self.palette_search(ui);
                    self.input_palette_entries(ui);
                    self.export_palette_entries(ui);
                    for e in self
                        .adapter
                        .palette(&self.session)
                        .into_iter()
                        .filter(|entry| {
                            matches!(
                                entry.id,
                                CommandId::NewFromImage
                                    | CommandId::OpenProject
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
        self.input_dialogs(root);
        self.show_export_toast(&ctx);
        if ctx.input(|i| !i.pointer.any_down()) {
            self.suppressed_slider = None;
            self.import_interrupted_drag = false;
        }
    }
    pub fn show_simple_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading(text("Material inspector"));
        egui::CollapsingHeader::new(crate::workspace::labels::text("import"))
            .default_open(self.session.snapshot().is_none())
            .show(ui, |ui| {
                ui.label(text("Local image path"));
                let path_response = ui.text_edit_singleline(&mut self.path);
                if self.focus == Some(CommandId::NewFromImage) {
                    path_response.request_focus();
                    self.focus = None;
                }
                egui::ComboBox::from_id_salt("import-kind")
                    .selected_text(crate::input_locale::labels::import_kind(self.import_kind))
                    .show_ui(ui, |ui| {
                        for value in [ImportKind::Photo, ImportKind::Height, ImportKind::Normal] {
                            ui.selectable_value(
                                &mut self.import_kind,
                                value,
                                crate::input_locale::labels::import_kind(value),
                            );
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
                    self.request_image(self.path.clone().into());
                }
                ui.small(text(
                    "Photo: PNG/JPEG, sRGB -> linear. Height: PNG16 / EXR32. Normal: RGB8 PNG.",
                ));
                ui.small(text(
                    "Alpha, ICC and EXIF ignored. Successful import clears history.",
                ));
            });
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
                    ui.small(crate::inspectors::labels::text("poisson-status"));
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
            ui.separator();
            ui.strong(crate::workspace::labels::text("frequent"));
            let mut frequent = Vec::new();
            for node in &nodes {
                for parameter in [
                    ParameterId::Strength,
                    ParameterId::Black,
                    ParameterId::White,
                    ParameterId::Gamma,
                    ParameterId::RadiusSourcePx,
                    ParameterId::Bias,
                    ParameterId::EdgeScale,
                    ParameterId::CavityScale,
                    ParameterId::Base,
                    ParameterId::Radius,
                    ParameterId::PickedColor,
                    ParameterId::Invert,
                    ParameterId::Threshold,
                    ParameterId::Feather,
                    ParameterId::CoarseWeight,
                    ParameterId::BandWeights,
                    ParameterId::ScaleWeights,
                    ParameterId::RemoveCoarse,
                    ParameterId::Directions,
                    ParameterId::Steps,
                ] {
                    if frequent.len() == 6 {
                        break;
                    }
                    if let Ok(value) = node.params.parameter(parameter) {
                        let descriptor = texelith_core::MATERIAL_PARAMETER_DESCRIPTORS
                            .iter()
                            .find(|d| d.id == parameter)
                            .expect("registered material parameter");
                        let address = ParameterAddress {
                            node: node.id,
                            parameter,
                        };
                        frequent.push(address);
                        self.material_parameter(ui, address, value, &descriptor.value_schema);
                    }
                }
                if frequent.len() == 6 {
                    break;
                }
            }
            if frequent.is_empty() {
                ui.small(crate::workspace::labels::text("no-controls"));
            }
            egui::CollapsingHeader::new(crate::workspace::labels::text("advanced")).show(
                ui,
                |ui| {
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
                                        if frequent.contains(&ParameterAddress {
                                            node: node.id,
                                            parameter: descriptor.id,
                                        }) {
                                            continue;
                                        }
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
                },
            );
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
                    .add_enabled(entry.enabled, egui::Button::new(&entry.label))
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
            self.open_palette();
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
        let locked = self.session.snapshot().is_some_and(|s| {
            self.session
                .view_state()
                .maps
                .get(&s.graph.selected_output_ref().port.map_kind())
                .is_some_and(|state| state.locked)
        });
        ui.add_enabled_ui(!locked, |ui| {
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
                            let mut add = ui.add_enabled(
                                v.len() < *max_len,
                                egui::Button::new(text("Add band")),
                            );
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
                                    responses.push(ui.add(
                                        egui::DragValue::new(x).speed(0.01).range(*min..=*max),
                                    ));
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
                    (
                        ParameterValue::Convention(v),
                        ParameterSchema::Convention { choices, .. },
                    ) => {
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
                            strength,
                            ..
                        },
                    ) => {
                        crate::inspectors::seamless::show(
                            ui,
                            v,
                            overlap,
                            grid,
                            seed,
                            strength,
                            &mut responses,
                        );
                    }
                    _ => {
                        ui.label(text("Read-only parameter"));
                    }
                }
            });
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
