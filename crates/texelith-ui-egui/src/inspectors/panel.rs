use super::*;
use crate::inspectors::labels::text as label;
use crate::material_ui;
use crate::material_ui::labels::text;
use texelith_core::{MaterialNodeParams, ParameterSchema};
impl Controls {
    pub fn show_procedural_inspector(&mut self, ui: &mut egui::Ui) {
        self.inspectors.parameter_rects.clear();
        if let Some(snapshot) = self.session.snapshot() {
            let selected = snapshot.graph.selected_output_ref();
            if selected.port == texelith_core::OutputPort::Edge
                || selected.port == texelith_core::OutputPort::Cavity
                || selected.port == texelith_core::OutputPort::Curvature
            {
                ui.horizontal(|ui| {
                    for map in [MapKind::Edge, MapKind::Cavity, MapKind::Curvature] {
                        if ui
                            .selectable_label(
                                selected.port.map_kind() == map,
                                text(material_ui::map_label(map)),
                            )
                            .clicked()
                        {
                            self.action(UiAction::Request(CommandRequest::SelectMap { map }));
                        }
                    }
                });
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
            let mut frequent = Vec::new();
            for (node, parameter) in
                crate::inspectors::key_controls::parameters(selected.port.map_kind(), &nodes)
            {
                let value = node
                    .params
                    .parameter(parameter)
                    .expect("applicable key control");
                let address = ParameterAddress {
                    node: node.id,
                    parameter,
                };
                frequent.push(address);
                self.procedural_parameter(
                    ui,
                    address,
                    value,
                    &parameter.material_descriptor().value_schema,
                    true,
                );
            }
            if frequent.is_empty() {
                ui.small(crate::workspace::labels::text("no-controls"));
            }
            egui::CollapsingHeader::new(crate::workspace::labels::text("advanced"))
                .default_open(self.inspectors.advanced)
                .show_unindented(ui, |ui| {
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
                            ui.small(label("poisson-status"));
                            if !self.inspectors.quality_status.is_empty() {
                                ui.small(&self.inspectors.quality_status);
                            }
                        }
                    }
                    if selected.port == texelith_core::OutputPort::Normal {
                        ui.small(text("Normal preview and legacy PNG: OpenGL +Y"));
                    }

                    self.paint_tools(ui);
                    for node in nodes {
                        let title = match node.params {
                            MaterialNodeParams::HeightFromAlbedo { .. } => "Height bands",
                            MaterialNodeParams::HeightFromNormal { .. } => "Poisson quality",
                            MaterialNodeParams::NormalFromHeight { .. }
                            | MaterialNodeParams::NormalSource { .. } => "Normal filter",
                            MaterialNodeParams::HorizonAo { .. } => "Ambient occlusion",
                            MaterialNodeParams::Curvature { .. } => "Edge / cavity",
                            MaterialNodeParams::Roughness { .. } => "Roughness curve",
                            MaterialNodeParams::Metallic { .. } => "Metallic selection",
                            MaterialNodeParams::Levels { .. } => "Height remap",
                            MaterialNodeParams::Seamless { .. } => "Seamless method",
                            MaterialNodeParams::PaintMask { .. } => "Paint boundary",
                            MaterialNodeParams::Blur { .. } => "Blur",
                            _ => continue,
                        };
                        egui::CollapsingHeader::new(match title {
                            "Poisson quality" => label("poisson-quality"),
                            "Paint boundary" => label("paint-boundary"),
                            _ => text(title),
                        })
                        .id_salt(node.id.0)
                        .default_open(node.id == selected.node)
                        .open((node.id == selected.node).then_some(true))
                        .show_unindented(ui, |ui| {
                            for descriptor in &texelith_core::MATERIAL_PARAMETER_DESCRIPTORS {
                                if let Ok(value) = node.params.parameter(descriptor.id) {
                                    if !matches!(
                                        descriptor.id,
                                        ParameterId::ScaleWeights
                                            | ParameterId::BandWeights
                                            | ParameterId::Curve
                                    ) && frequent.contains(&ParameterAddress {
                                        node: node.id,
                                        parameter: descriptor.id,
                                    }) {
                                        continue;
                                    }
                                    let address = ParameterAddress {
                                        node: node.id,
                                        parameter: descriptor.id,
                                    };
                                    self.procedural_parameter(
                                        ui,
                                        address,
                                        value,
                                        &descriptor.value_schema,
                                        false,
                                    );
                                }
                            }
                        });
                    }
                });
        }
        if !self.message.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.message);
        }
    }

    fn procedural_parameter(
        &mut self,
        ui: &mut egui::Ui,
        address: ParameterAddress,
        mut value: ParameterValue,
        schema: &ParameterSchema,
        key_control: bool,
    ) {
        let parameter_label = if key_control {
            label(crate::inspectors::key_controls::label(address.parameter))
        } else {
            text(&format!("{:?}", address.parameter))
        };
        let mut responses = Vec::new();
        let locked = self.session.snapshot().is_some_and(|s| {
            self.session
                .view_state()
                .maps
                .get(&s.graph.selected_output_ref().port.map_kind())
                .is_some_and(|state| state.locked)
        });
        ui.add_enabled_ui(!locked, |ui| {
            let id = (
                address.node.0,
                format!("{:?}", address.parameter),
                key_control,
            );
            ui.push_id(id, |ui| {
                if !matches!(
                    value,
                    ParameterValue::Float(_)
                        | ParameterValue::UInt(_)
                        | ParameterValue::Bool(_)
                        | ParameterValue::Convention(_)
                ) && !(key_control && matches!(value, ParameterValue::FloatVector(_)))
                {
                    ui.label(&parameter_label);
                }
                match (&mut value, schema) {
                    (ParameterValue::Float(v), ParameterSchema::Float { min, max, .. }) => {
                        let normal_strength = key_control
                            && address.parameter == ParameterId::Strength
                            && self.session.snapshot().is_some_and(|s| {
                                s.graph.material_node(address.node).is_some_and(|n| {
                                    matches!(n.params, MaterialNodeParams::NormalFromHeight { .. })
                                })
                            });
                        if normal_strength {
                            let unit = if self.session.snapshot().is_some_and(|s| {
                                s.graph
                                    .material_nodes()
                                    .iter()
                                    .any(|n| matches!(n.params, MaterialNodeParams::ColorSource(_)))
                            }) {
                                0.015
                            } else {
                                1.
                            };
                            crate::inspectors::rows::normal_strength(
                                ui,
                                &parameter_label,
                                v,
                                unit,
                                &mut responses,
                            );
                        } else {
                            let working = key_control
                                .then(|| match address.parameter {
                                    ParameterId::Bias => Some(0.0..=0.1),
                                    ParameterId::Gamma => Some(0.1..=3.0),
                                    ParameterId::EdgeScale | ParameterId::CavityScale => {
                                        Some(0.0..=3.0)
                                    }
                                    ParameterId::CoarseWeight => Some(-2.0..=2.0),
                                    ParameterId::RadiusSourcePx => Some(1.0..=128.0),
                                    _ => None,
                                })
                                .flatten();
                            if let Some(working) = working {
                                crate::inspectors::rows::float_range(
                                    ui,
                                    &parameter_label,
                                    v,
                                    *min..=*max,
                                    working,
                                    &mut responses,
                                );
                            } else {
                                crate::inspectors::rows::float(
                                    ui,
                                    &parameter_label,
                                    v,
                                    *min..=*max,
                                    &mut responses,
                                );
                            }
                        }
                    }
                    (ParameterValue::UInt(v), ParameterSchema::UInt { min, max, .. }) => {
                        crate::inspectors::rows::uint(
                            ui,
                            &parameter_label,
                            v,
                            *min..=*max,
                            &mut responses,
                        );
                    }
                    (ParameterValue::Bool(v), _) => {
                        responses.push(ui.checkbox(v, parameter_label));
                    }
                    (ParameterValue::Color(v), ParameterSchema::Color { min, max, .. }) => {
                        ui.small(label("linear-rgb"));
                        ui.horizontal(|ui| {
                            for (component, x) in ["R", "G", "B"].into_iter().zip(v.iter_mut()) {
                                responses.push(
                                    ui.add(
                                        egui::DragValue::new(x)
                                            .speed(0.001)
                                            .range(*min..=*max)
                                            .prefix(format!("{component} ")),
                                    ),
                                );
                            }
                        });
                        if ui.button(label("pick-source")).clicked() {
                            self.set_brush_enabled(false);
                            self.inspectors.armed = Some(address);
                        }
                        if self.inspectors.armed == Some(address) {
                            ui.small(label("pick-instruction"));
                        }
                        if self.inspectors.pending.is_some() {
                            ui.small(label("pick-pending"));
                        }
                    }
                    (ParameterValue::FloatVector(v), _) if key_control => {
                        let mut scale = crate::inspectors::key_controls::detail_scale(v);
                        crate::inspectors::rows::float(
                            ui,
                            &parameter_label,
                            &mut scale,
                            1.0..=8.0,
                            &mut responses,
                        );
                        if responses.iter().any(|r| r.changed()) {
                            *v = crate::inspectors::key_controls::scale_weights(scale);
                        }
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
                            ui.push_id(i, |ui| {
                                ui.label(format!("Band {}", i + 1));
                                responses.push(
                                    ui.add(egui::DragValue::new(x).speed(0.01).range(*min..=*max)),
                                );
                            });
                        }
                    }
                    (ParameterValue::Curve(v), _) if key_control => {
                        for (name, curve) in [
                            ("response-photo", vec![[0., 0.8], [0.25, 0.6], [1., 0.45]]),
                            ("response-linear", vec![[0., 0.], [1., 1.]]),
                            ("response-soft", vec![[0., 0.25], [1., 0.75]]),
                        ] {
                            responses.push(ui.selectable_value(v, curve, label(name)));
                        }
                    }
                    (
                        ParameterValue::Curve(v),
                        ParameterSchema::Curve {
                            min,
                            max,
                            min_len,
                            max_len,
                            ..
                        },
                    ) => {
                        ui.small(label("curve-xy"));
                        let mut remove = None;
                        let mut insert = None;
                        let length = v.len();
                        for i in 0..length {
                            ui.horizontal(|ui| {
                                // Neighbour bounds preserve strict ordering; Session remains authoritative.
                                let low = if i == 0 { *min } else { v[i - 1][0].next_up() };
                                let high = if i + 1 == length {
                                    *max
                                } else {
                                    v[i + 1][0].next_down()
                                };
                                ui.push_id(i, |ui| {
                                    responses.push(
                                        ui.add(
                                            egui::DragValue::new(&mut v[i][0])
                                                .speed(0.005)
                                                .range(low..=high)
                                                .prefix("x "),
                                        ),
                                    );
                                    responses.push(
                                        ui.add(
                                            egui::DragValue::new(&mut v[i][1])
                                                .speed(0.005)
                                                .range(*min..=*max)
                                                .prefix("y "),
                                        ),
                                    );
                                    let mut minus = ui
                                        .add_enabled(length > *min_len, egui::Button::new("?"))
                                        .on_hover_text(label("remove-knot"));
                                    if minus.clicked() {
                                        remove = Some(i);
                                        minus.mark_changed();
                                    }
                                    responses.push(minus);
                                    if i + 1 < length {
                                        let middle = (v[i][0] + v[i + 1][0]) * 0.5;
                                        let mut plus = ui
                                            .add_enabled(
                                                length < *max_len
                                                    && middle > v[i][0]
                                                    && middle < v[i + 1][0],
                                                egui::Button::new("+"),
                                            )
                                            .on_hover_text(label("add-knot"));
                                        if plus.clicked() {
                                            insert = Some((
                                                i + 1,
                                                [middle, (v[i][1] + v[i + 1][1]) * 0.5],
                                            ));
                                            plus.mark_changed();
                                        }
                                        responses.push(plus);
                                    }
                                });
                            });
                        }
                        if let Some(i) = remove {
                            v.remove(i);
                        }
                        if let Some((i, point)) = insert {
                            v.insert(i, point);
                        }
                    }
                    (ParameterValue::Boundary(v), ParameterSchema::Boundary { choices, .. }) => {
                        for choice in *choices {
                            responses.push(ui.selectable_value(v, *choice, format!("{choice:?}")));
                        }
                    }
                    (ParameterValue::Filter(v), ParameterSchema::Filter { choices, .. }) => {
                        ui.horizontal(|ui| {
                            for choice in *choices {
                                let name = match choice {
                                    texelith_core::NormalFilter::Central => "smooth-none",
                                    texelith_core::NormalFilter::Sobel => "smooth-soft",
                                    texelith_core::NormalFilter::Scharr => "smooth-balanced",
                                };
                                responses.push(ui.selectable_value(
                                    v,
                                    *choice,
                                    if key_control {
                                        label(name)
                                    } else {
                                        format!("{choice:?}")
                                    },
                                ));
                            }
                        });
                    }
                    (ParameterValue::Convention(v), ParameterSchema::Convention { .. }) => {
                        let mut invert = *v == texelith_core::NormalConvention::DirectX;
                        let response = ui.checkbox(
                            &mut invert,
                            if key_control {
                                label("invert-y")
                            } else {
                                text("Convention")
                            },
                        );
                        if response.changed() {
                            *v = if invert {
                                texelith_core::NormalConvention::DirectX
                            } else {
                                texelith_core::NormalConvention::OpenGl
                            };
                        }
                        responses.push(response);
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
        self.inspectors
            .parameter_rects
            .push((address, responses.iter().map(|r| r.rect).collect()));
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
        let numeric_edit = ui.ctx().egui_wants_keyboard_input()
            && responses.iter().any(|r| r.has_focus() && r.changed());
        if responses.iter().any(|r| r.drag_started()) || numeric_edit {
            if self.adapter.active_address().is_some_and(|a| a != address) {
                self.action(UiAction::DragRelease);
            }
            self.action(UiAction::DragBegin { address });
        }
        if responses.iter().any(|r| r.changed()) {
            self.action(UiAction::Change { address, value });
        }
        if self.adapter.active_address() == Some(address)
            && (responses.iter().any(|r| r.drag_stopped() || r.lost_focus())
                || ui.ctx().input(|i| i.key_pressed(egui::Key::Enter)))
        {
            self.action(UiAction::DragRelease);
        }
    }
}
