use super::*;
use crate::workspace::{self, Card, labels::text as label};

impl Controls {
    pub(super) fn workspace_panels(
        &mut self,
        root: &mut egui::Ui,
        inspector: impl FnOnce(&mut Self, &mut egui::Ui),
        export: impl FnOnce(&mut Self, &mut egui::Ui),
    ) {
        use crate::workspace::studio::{self, Icon};
        egui::Panel::bottom("studio-status")
            .exact_size(30.)
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    let busy = self.export_busy() || self.workspace.busy;
                    if busy {
                        ui.spinner();
                    } else {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(18., 18.), egui::Sense::hover());
                        ui.painter().circle_filled(
                            rect.center(),
                            8.,
                            egui::Color32::from_rgb(131, 190, 42),
                        );
                        let stroke = egui::Stroke::new(1.6, egui::Color32::from_gray(28));
                        ui.painter().line_segment(
                            [
                                rect.center() + egui::vec2(-4., 0.),
                                rect.center() + egui::vec2(-1., 3.),
                            ],
                            stroke,
                        );
                        ui.painter().line_segment(
                            [
                                rect.center() + egui::vec2(-1., 3.),
                                rect.center() + egui::vec2(5., -4.),
                            ],
                            stroke,
                        );
                    }
                    ui.label(if self.export_busy() {
                        crate::input_locale::labels::text("export-busy")
                    } else if busy {
                        label("processing")
                    } else {
                        label("ready")
                    })
                    .on_hover_text(format!(
                        "{}\n{}",
                        self.workspace.status, self.native_files.export_detail
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label("sRGB / Linear (Auto)")
                            .on_hover_text(&self.workspace.gpu);
                        ui.separator();
                        ui.label("PBR (8 / 16-bit)")
                            .on_hover_text(&self.workspace.gpu);
                        if let Some(snapshot) = self.session.snapshot()
                            && let Ok(extent) = snapshot
                                .graph
                                .material_output_extent(snapshot.graph.selected_output_ref())
                        {
                            ui.separator();
                            ui.label(format!("{} \u{d7} {}", extent.width, extent.height))
                                .on_hover_text(&self.workspace.gpu);
                        }
                    });
                });
            });
        let rail = egui::Panel::left("studio-tool-rail")
            .exact_size(80.)
            .resizable(false)
            .show(root, |ui| {
                for (icon, key) in [
                    (Icon::Brush, "tool-brush"),
                    (Icon::Clone, "tool-clone"),
                    (Icon::Eraser, "tool-eraser"),
                    (Icon::Decal, "tool-decal"),
                ] {
                    ui.add_enabled_ui(false, |ui| {
                        let response = egui::Frame::new().inner_margin(6.).show(ui, |ui| {
                            ui.set_min_size(egui::vec2(54., 65.));
                            ui.vertical_centered(|ui| {
                                studio::tool_button(ui, icon, &label(key));
                                ui.small(label(key));
                            });
                        });
                        response
                            .response
                            .on_disabled_hover_text(label("faz2-tooltip"));
                    });
                    ui.separator();
                }
            });
        self.workspace.layout.tool_rail = Some(rail.response.rect);
        let width = root.ctx().content_rect().width();
        let inspector_open = self.workspace.inspector_is_open(width);
        let panel_width = if inspector_open {
            (width * 0.26).clamp(280., 400.)
        } else {
            42.
        };
        if self.workspace.last_width != Some(width) {
            let id = root.scope_id().with("material_inspector");
            root.ctx().data_mut(|data| {
                data.insert_persisted(
                    id,
                    egui::containers::panel::PanelState {
                        outer_rect: egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2((width * 0.26).clamp(280., 400.), 1.),
                        ),
                    },
                )
            });
            self.workspace.last_width = Some(width);
        }
        let panel = egui::Panel::right(if inspector_open {
            "material_inspector"
        } else {
            "material_inspector_closed"
        })
        .default_size(panel_width)
        .min_size(if inspector_open { 240. } else { 42. })
        .max_size(panel_width)
        .resizable(inspector_open)
        .show(root, |ui| {
            workspace::bounded_panel_contents(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.;
                if !inspector_open {
                    let toggle = studio::small_button(ui, Icon::Batch, &label("layers"), false);
                    self.workspace.layout.inspector_toggle = Some(toggle.rect);
                    if toggle.clicked() {
                        self.workspace.inspector_open = Some(true);
                    }
                    let button =
                        studio::small_button(ui, Icon::Export, &label("export-action"), false);
                    self.workspace.layout.export_material = Some(button.rect);
                    self.workspace.layout.export_options = Some(button.rect);
                    self.workspace.layout.export = Some(button.rect);
                    egui::Popup::menu(&button)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                            ui.set_width(300.);
                            let scroll = egui::ScrollArea::vertical()
                                .max_height(380.)
                                .show(ui, |ui| export(self, ui));
                            self.workspace.layout.export = Some(scroll.inner_rect);
                        });
                    return;
                }
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.workspace.paint_tab, false, label("layer-tab"));
                    ui.selectable_value(&mut self.workspace.paint_tab, true, label("paint-tab"));
                    {
                        let toggle = ui.small_button("\u{bb}").on_hover_text(label("inspector"));
                        self.workspace.layout.inspector_toggle = Some(toggle.rect);
                        if toggle.clicked() {
                            self.workspace.inspector_open = Some(false);
                        }
                    }
                });
                ui.separator();
                if self.workspace.paint_tab {
                    ui.add_enabled(false, egui::Label::new(label("paint-placeholder")));
                    return;
                }
                // Reserve the export action before laying out the scrolling inspector.
                // Expanding properties must never push the primary action off screen.
                egui::Panel::bottom("studio-pinned-export")
                    .exact_size(350.)
                    .resizable(false)
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let section = egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            egui::CollapsingHeader::new(label("export-action"))
                                .default_open(true)
                                .show_unindented(ui, |ui| export(self, ui));
                        });
                        self.workspace.layout.export =
                            Some(section.response.rect.intersect(ui.clip_rect()));
                    });
                egui::ScrollArea::vertical()
                    .id_salt("layer-panel-scroll")
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add_enabled_ui(false, |ui| {
                                egui::ComboBox::from_id_salt("add-layer")
                                    .width(90.)
                                    .selected_text(format!("+ {}", label("add-layer")))
                                    .show_ui(ui, |_| {})
                                    .response
                            })
                            .inner
                            .on_disabled_hover_text(label("faz2-tooltip"));
                            for (icon, key) in [
                                (Icon::Duplicate, "duplicate-layer"),
                                (Icon::Folder, "folder-layer"),
                                (Icon::Delete, "delete-layer"),
                                (Icon::More, "more-layer"),
                            ] {
                                ui.add_enabled_ui(false, |ui| {
                                    studio::small_button(ui, icon, &label(key), false)
                                })
                                .inner
                                .on_disabled_hover_text(label("faz2-tooltip"));
                            }
                        });
                        egui::Frame::group(ui.style())
                            .stroke(egui::Stroke::new(1., studio::AMBER))
                            .fill(ui.visuals().selection.bg_fill.gamma_multiply(0.35))
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    if let Some(snapshot) = self.session.snapshot() {
                                        let map =
                                            snapshot.graph.selected_output_ref().port.map_kind();
                                        let state = self
                                            .session
                                            .view_state()
                                            .maps
                                            .get(&map)
                                            .copied()
                                            .unwrap_or_default();
                                        if studio::small_button(
                                            ui,
                                            Icon::Eye,
                                            &label("visible"),
                                            state.visible,
                                        )
                                        .clicked()
                                        {
                                            self.action(UiAction::Request(
                                                CommandRequest::SetVisible {
                                                    map,
                                                    visible: !state.visible,
                                                },
                                            ));
                                        }
                                    }
                                    ui.label(label("base-layer"));
                                    ui.add_enabled(false, egui::Button::new("100%"));
                                });
                            });
                        ui.add_space(6.);
                        let map = self
                            .session
                            .snapshot()
                            .map(|s| {
                                text(material_ui::map_label(
                                    s.graph.selected_output_ref().port.map_kind(),
                                ))
                            })
                            .unwrap_or_default();
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            egui::CollapsingHeader::new(format!("{} {}", map, label("properties")))
                                .default_open(true)
                                .show_unindented(ui, |ui| inspector(self, ui));
                        });
                    });
            });
        });
        if inspector_open {
            self.workspace.layout.inspector = Some(panel.response.rect);
        }
        if self.workspace.maps_open.unwrap_or(true) {
            let strip = egui::Panel::top("map-channel-strip")
                .exact_size(58.)
                .show(root, |ui| {
                    egui::ScrollArea::horizontal()
                        .id_salt("map-channel-scroll")
                        .show(ui, |ui| {
                            ui.horizontal_top(|ui| {
                                ui.spacing_mut().item_spacing.x = 3.;
                                ui.style_mut().override_font_id =
                                    Some(egui::FontId::proportional(if width < 1200. {
                                        11.
                                    } else {
                                        13.
                                    }));
                                let toggle =
                                    ui.add(egui::Button::new(label("map-channel")).frame(false));
                                self.workspace.layout.maps_toggle = Some(toggle.rect);
                                if toggle.clicked() {
                                    self.workspace.maps_open = Some(false);
                                }
                                let snapshot = self.session.snapshot();
                                for map in MAPS {
                                    self.map_card(ui, snapshot.as_ref(), map);
                                }
                            });
                        });
                });
            self.workspace.layout.maps = Some(strip.response.rect);
        }
    }

    /// Default export content; the application can replace this leaf via `show_workspace_with`.
    pub fn show_simple_export(&mut self, ui: &mut egui::Ui) {
        self.show_material_export(ui);
    }

    pub(crate) fn show_png_export(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(label("destination"));
            let response =
                ui.add(egui::TextEdit::singleline(&mut self.destination).desired_width(200.0));
            self.workspace.layout.destination = Some(response.rect);
            if ui
                .button(crate::input_locale::labels::text("native-browse"))
                .clicked()
            {
                self.browse_png();
            }
            if self.focus == Some(CommandId::SavePng) {
                response.request_focus();
                self.focus = None;
            }
            let selected = self
                .session
                .snapshot()
                .map(|s| s.graph.selected_output_ref().port.map_kind());
            let save = ui.add_enabled(
                selected.is_some() && !self.destination.is_empty(),
                egui::Button::new(label("save-png")),
            );
            self.workspace.layout.save_png = Some(save.rect);
            if save.clicked() {
                self.save_png(self.destination.clone().into(), selected.unwrap());
            }
            ui.small(label("png-depth"));
        });
    }

    fn map_card(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: Option<&texelith_core::EvaluationSnapshot>,
        map: MapKind,
    ) {
        let enabled = snapshot.is_some_and(|s| material_ui::supported(s, map));
        let selected =
            snapshot.is_some_and(|s| s.graph.selected_output_ref().port.map_kind() == map);
        let state = self
            .session
            .view_state()
            .maps
            .get(&map)
            .copied()
            .unwrap_or_default();
        let badge = snapshot.map_or("Procedural", |s| workspace::source_badge(s, map));
        let mut frame = egui::Frame::group(ui.style()).inner_margin(3.);
        if selected {
            frame.fill = ui.visuals().selection.bg_fill.gamma_multiply(0.35);
            frame.stroke = ui.visuals().selection.stroke;
        }
        let mut thumb = egui::Rect::NOTHING;
        let chip = frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                thumb = ui
                    .allocate_exact_size(egui::vec2(32., 32.), egui::Sense::hover())
                    .0;
                if let Some(preview) = self.workspace.thumbnails.get_mut(&map) {
                    let texture = preview.texture.get_or_insert_with(|| {
                        ui.ctx().load_texture(
                            format!("map-card-{map:?}"),
                            preview.image.clone(),
                            egui::TextureOptions::LINEAR,
                        )
                    });
                    ui.painter().image(
                        texture.id(),
                        thumb,
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                        egui::Color32::WHITE,
                    );
                } else {
                    ui.painter()
                        .rect_filled(thumb, 3., crate::workspace::neutral_preview(map));
                }
                ui.add(
                    egui::Label::new(text(material_ui::map_label(map)))
                        .wrap_mode(egui::TextWrapMode::Extend)
                        .selectable(false),
                );
            });
        });
        let response = ui.interact(
            chip.response.rect,
            ui.scope_id().with(("map-chip", map)),
            egui::Sense::click(),
        );
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::Role::Button, enabled, material_ui::map_label(map))
        });
        if response.clicked() && enabled {
            self.action(UiAction::Request(CommandRequest::SelectMap { map }));
        }
        let mut visibility = egui::Rect::NOTHING;
        let mut lock = egui::Rect::NOTHING;
        response.context_menu(|ui| {
            ui.add_enabled_ui(enabled, |ui| {
                let mut visible = state.visible;
                let r = ui
                    .checkbox(&mut visible, label("visible"))
                    .on_hover_text(label("visibility-help"));
                visibility = r.rect;
                if r.changed() {
                    self.action(UiAction::Request(CommandRequest::SetVisible {
                        map,
                        visible,
                    }));
                }
                let mut locked = state.locked;
                let r = ui
                    .checkbox(&mut locked, label("locked"))
                    .on_hover_text(label("lock-help"));
                lock = r.rect;
                if r.changed() {
                    self.action(UiAction::Request(CommandRequest::SetLocked { map, locked }));
                }
            });
            ui.small(label(if badge == "Manual" {
                "manual"
            } else {
                "procedural"
            }));
        });
        self.workspace.layout.cards.insert(
            map,
            Card {
                select: chip.response.rect,
                thumbnail: thumb,
                visibility,
                lock,
                badge,
            },
        );
    }

    pub(crate) fn map_options(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: &texelith_core::EvaluationSnapshot,
    ) {
        ui.separator();
        let mut tiles = self.session.view_state().tiles;
        egui::ComboBox::from_id_salt("tile-count")
            .selected_text(format!("{} x {} tiles", tiles.count, tiles.count))
            .show_ui(ui, |ui| {
                for count in [1, 2, 3] {
                    if ui
                        .selectable_value(&mut tiles.count, count, format!("{count} x {count}"))
                        .changed()
                    {
                        self.action(UiAction::Request(CommandRequest::SetTiles { tiles }));
                    }
                }
            });
        let mut heatmap = self.heatmap;
        if ui.checkbox(&mut heatmap, text("Seam heatmap")).changed() {
            self.action(UiAction::Request(CommandRequest::InspectSeams {
                enabled: heatmap,
            }));
        }
        let (mut seamless, mut mask) = material_ui::options(&snapshot.graph);
        if ui.checkbox(&mut seamless, text("Seamless")).changed()
            | ui.checkbox(&mut mask, text("Use painted mask")).changed()
        {
            self.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                seamless,
                painted_mask: mask,
            }));
        }
        for map in [MapKind::Mask, MapKind::Curvature, MapKind::Cavity] {
            if material_ui::supported(snapshot, map)
                && ui.small_button(text(material_ui::map_label(map))).clicked()
            {
                self.action(UiAction::Request(CommandRequest::SelectMap { map }));
            }
        }
    }
}
