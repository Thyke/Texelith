use super::*;
use crate::inspectors::{PaintPhase, PickRequest, source_uv};
use texelith_core::{Brush, BrushSample, MaterialNodeParams, StrokeId};
impl Controls {
    pub fn set_brush_enabled(&mut self, enabled: bool) {
        if !enabled && let Some(stroke_id) = self.session.history().active_stroke {
            self.action(UiAction::Request(CommandRequest::CommitStroke {
                stroke_id,
            }));
        }
        if enabled && self.session.view_state().tiles.count > 1 {
            let mut tiles = self.session.view_state().tiles;
            tiles.count = 1;
            self.action(UiAction::Request(CommandRequest::SetTiles { tiles }));
        }
        self.inspectors.painting = enabled;
        self.inspectors.armed = None;
    }
    pub fn set_brush(&mut self, brush: Brush) -> Result<(), String> {
        brush.validate().map_err(|e| e.to_string())?;
        self.inspectors.brush = brush;
        Ok(())
    }
    fn selected_locked(&self) -> bool {
        self.session.snapshot().is_some_and(|s| {
            self.session
                .view_state()
                .maps
                .get(&s.graph.selected_output_ref().port.map_kind())
                .is_some_and(|m| m.locked)
        })
    }
    pub fn paint_at(&mut self, uv: [f32; 2], phase: PaintPhase) {
        if phase == PaintPhase::End {
            if let Some(stroke_id) = self.session.history().active_stroke {
                self.action(UiAction::Request(CommandRequest::CommitStroke {
                    stroke_id,
                }));
            }
            self.inspectors.last_uv = None;
            return;
        }
        if !self.inspectors.painting || self.selected_locked() {
            return;
        }
        let sample = BrushSample { uv, pressure: 1.0 };
        if let Err(e) = sample.validate() {
            self.message = e.to_string();
            return;
        }
        let Some(snapshot) = self.session.snapshot() else {
            return;
        };
        let Some(mask) = snapshot
            .graph
            .material_outputs()
            .and_then(|m| m.get(&MapKind::Mask))
        else {
            return;
        };
        match phase {
            PaintPhase::Begin => {
                let existing = snapshot
                    .graph
                    .material_node(mask.node)
                    .and_then(|n| match &n.params {
                        MaterialNodeParams::PaintMask { strokes, .. } => {
                            strokes.iter().map(|s| s.id.0).max()
                        }
                        _ => None,
                    })
                    .unwrap_or(0);
                self.inspectors.next_stroke = self
                    .inspectors
                    .next_stroke
                    .max(existing)
                    .checked_add(1)
                    .expect("stroke IDs exhausted");
                self.action(UiAction::Request(CommandRequest::BeginStroke {
                    node: mask.node,
                    stroke_id: StrokeId(self.inspectors.next_stroke),
                    brush: self.inspectors.brush,
                    sample,
                }));
            }
            PaintPhase::Append => {
                if self.inspectors.last_uv == Some(uv) {
                    return;
                }
                if let Some(stroke_id) = self.session.history().active_stroke {
                    self.action(UiAction::Request(CommandRequest::AppendStroke {
                        stroke_id,
                        samples: vec![sample],
                    }));
                    if !self.message.is_empty() {
                        // Preserve the bounded prefix and require a new pointer press.
                        let message = self.message.clone();
                        self.action(UiAction::Request(CommandRequest::CommitStroke {
                            stroke_id,
                        }));
                        self.message = message;
                    }
                }
            }
            PaintPhase::End => unreachable!(),
        }
        self.inspectors.last_uv = Some(uv);
    }
    pub fn request_color_pick(
        &mut self,
        address: ParameterAddress,
        uv: [f32; 2],
    ) -> Result<PickRequest, String> {
        let snapshot = self.session.snapshot().ok_or("No material")?;
        if self.selected_locked() {
            return Err("Map is locked".into());
        }
        BrushSample { uv, pressure: 1.0 }
            .validate()
            .map_err(|e| e.to_string())?;
        if !snapshot
            .graph
            .material_nodes()
            .iter()
            .any(|n| matches!(n.params, MaterialNodeParams::ColorSource(_)))
        {
            return Err("A photo source is required".into());
        }
        if !matches!(
            snapshot
                .graph
                .material_node(address.node)
                .and_then(|n| n.params.parameter(address.parameter).ok()),
            Some(ParameterValue::Color(_))
        ) {
            return Err("Not a color parameter".into());
        }
        self.inspectors.next_pick = self
            .inspectors
            .next_pick
            .checked_add(1)
            .expect("picker IDs exhausted");
        let request = PickRequest {
            token: self.inspectors.next_pick,
            generation: snapshot.generation,
            request_id: self.session.request_id(),
            document_revision: self.session.document_revision(),
            output: snapshot.graph.selected_output_ref(),
            address,
            uv,
        };
        self.inspectors.pending = Some(request.clone());
        self.inspectors.submitted = false;
        Ok(request)
    }
    pub fn complete_color_pick(&mut self, request: &PickRequest, color: [f32; 3]) -> bool {
        if self.inspectors.pending.as_ref() != Some(request) {
            return false;
        }
        self.inspectors.pending = None;
        self.inspectors.submitted = false;
        if self.selected_locked()
            || self.session.request_id() != request.request_id
            || self.session.document_revision() != request.document_revision
            || !self.session.snapshot().is_some_and(|s| {
                s.generation == request.generation
                    && s.graph.selected_output_ref() == request.output
            })
        {
            return false;
        }
        let before = self.session.snapshot().unwrap().generation;
        self.action(UiAction::Change {
            address: request.address,
            value: ParameterValue::Color(color),
        });
        self.message.is_empty() && (self.session.snapshot().unwrap().generation >= before)
    }
    pub fn viewport_gestures(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        self.inspectors.viewport = Some(rect);
        let response = ui.interact(
            rect.intersect(ui.clip_rect()),
            ui.scope_id().with("source-brush"),
            egui::Sense::click_and_drag(),
        );
        if self.scripted {
            return;
        }
        let (focused, escape, position, pressed, down, released) = ui.ctx().input(|i| {
            (
                i.focused,
                i.key_pressed(egui::Key::Escape),
                i.pointer.interact_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.primary_released(),
            )
        });
        if escape || !focused {
            return;
        }
        if !self.inspectors.painting && self.inspectors.armed.is_none() {
            if response.dragged() {
                self.preview.pan += ui.input(|i| i.pointer.delta());
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.preview.zoom = (self.preview.zoom * (scroll * 0.002).exp()).clamp(25., 400.);
            }
        }
        let tiles = self.session.view_state().tiles.count;
        let uv = position.and_then(|p| {
            let uv = source_uv(rect, p, tiles)?;
            let settings = self.session.view_state().tiles;
            if settings.random_rotation {
                let position = [
                    (p.x - rect.left()) / rect.width(),
                    (p.y - rect.top()) / rect.height(),
                ];
                Some(texelith_gpu::compare::tile_uv(position, settings).0)
            } else {
                Some(uv)
            }
        });
        if response.hovered() && self.inspectors.armed.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            if pressed
                && let (Some(address), Some(uv)) = (self.inspectors.armed.take(), uv)
                && let Err(e) = self.request_color_pick(address, uv)
            {
                self.message = e;
            }
            return;
        }
        if self.inspectors.painting {
            if let (Some(uv), Some(pos)) = (uv, position)
                && (response.hovered() || self.session.history().active_stroke.is_some())
            {
                ui.ctx().set_cursor_icon(egui::CursorIcon::None);
                let radius = self.inspectors.brush.radius;
                let radii = rect.size() * radius / tiles.max(1) as f32;
                let painter = ui.painter().with_clip_rect(rect);
                painter.add(egui::epaint::EllipseShape::stroke(
                    pos,
                    radii,
                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                ));
                painter.add(egui::epaint::EllipseShape::stroke(
                    pos,
                    radii * self.inspectors.brush.hardness,
                    egui::Stroke::new(1.0, egui::Color32::BLACK),
                ));
                if pressed && response.hovered() {
                    self.paint_at(uv, PaintPhase::Begin);
                } else if down && self.session.history().active_stroke.is_some() {
                    self.paint_at(uv, PaintPhase::Append);
                }
            }
            if released {
                self.paint_at(uv.unwrap_or([0.0, 0.0]), PaintPhase::End);
            }
        }
    }
    pub fn inspector_parameter_rects(&self, address: ParameterAddress) -> &[egui::Rect] {
        self.inspectors
            .parameter_rects
            .iter()
            // Prefer the complete editor when Advanced also exposes a simplified key control.
            .rev()
            .find(|(a, _)| *a == address)
            .map_or(&[], |(_, rects)| rects.as_slice())
    }
    pub fn set_inspector_advanced(&mut self, open: bool) {
        self.inspectors.advanced = open;
    }
    pub(crate) fn note_inspector_evaluation(&mut self, success: bool) {
        let Some(snapshot) = self.session.snapshot() else {
            return;
        };
        let mut stack = vec![snapshot.graph.selected_output_ref().node];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(node) = snapshot.graph.material_node(id) {
                if matches!(node.params, MaterialNodeParams::HeightFromNormal { .. }) {
                    self.inspectors.quality_status = crate::inspectors::labels::text(if success {
                        "poisson-converged"
                    } else {
                        "poisson-failed"
                    });
                    return;
                }
                stack.extend(node.inputs.iter().map(|e| e.source.node));
            }
        }
        self.inspectors.quality_status = crate::inspectors::labels::text("poisson-not-run");
    }
    pub(crate) fn set_inspector_error(&mut self, error: String) {
        self.message = error;
    }
    pub fn thumbnail_generation(&self, map: MapKind) -> Option<u64> {
        self.workspace.thumbnails.get(&map).map(|t| t.generation)
    }
}
impl Controls {
    pub(super) fn paint_tools(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = self.session.snapshot() else {
            return;
        };
        let selected = snapshot.graph.selected_output_ref().port.map_kind();
        if !matches!(
            selected,
            MapKind::Metallic | MapKind::Roughness | MapKind::Mask
        ) {
            return;
        }
        let has_mask = snapshot
            .graph
            .material_outputs()
            .is_some_and(|m| m.contains_key(&MapKind::Mask));
        egui::CollapsingHeader::new(crate::inspectors::labels::text("paint-mask"))
            .default_open(selected == MapKind::Mask)
            .open((selected == MapKind::Mask).then_some(true))
            .show_unindented(ui, |ui| {
                if !has_mask {
                    if ui
                        .add_enabled(
                            !self.selected_locked(),
                            egui::Button::new(crate::inspectors::labels::text("enable-mask")),
                        )
                        .clicked()
                    {
                        let (seamless, _) = crate::material_ui::options(&snapshot.graph);
                        self.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                            seamless,
                            painted_mask: true,
                        }));
                    }
                    return;
                }
                ui.add_enabled_ui(!self.selected_locked(), |ui| {
                    let mut enabled = self.inspectors.painting;
                    if ui
                        .checkbox(
                            &mut enabled,
                            crate::inspectors::labels::text("paint-enabled"),
                        )
                        .changed()
                    {
                        self.set_brush_enabled(enabled);
                    }
                    let mut responses = Vec::new();
                    for (key, value, range) in [
                        (
                            "brush-radius",
                            &mut self.inspectors.brush.radius,
                            0.001..=1.0,
                        ),
                        (
                            "brush-hardness",
                            &mut self.inspectors.brush.hardness,
                            0.0..=1.0,
                        ),
                        (
                            "brush-opacity",
                            &mut self.inspectors.brush.opacity,
                            0.0..=1.0,
                        ),
                    ] {
                        crate::inspectors::rows::float(
                            ui,
                            &crate::inspectors::labels::text(key),
                            value,
                            range,
                            &mut responses,
                        );
                    }
                    for (mode, key) in [
                        (texelith_core::MaskMode::Replace, "replace"),
                        (texelith_core::MaskMode::Add, "add"),
                        (texelith_core::MaskMode::Subtract, "subtract"),
                    ] {
                        ui.selectable_value(
                            &mut self.inspectors.brush.mode,
                            mode,
                            crate::inspectors::labels::text(key),
                        );
                    }
                });
                ui.small(crate::inspectors::labels::text("paint-help"));
            });
    }
}
