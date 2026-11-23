//! Native Layer Studio matrix. Run once empty and once with --input.
use crate::{command_adapter::UiAction, controls::Controls, material_ui::snapshots::Captures};
use std::time::{Duration, Instant};
use texelith_core::*;

#[derive(Default)]
pub(crate) struct Driver {
    stage: usize,
    started: Option<Instant>,
    requested: bool,
}

impl Driver {
    pub fn tick(
        &mut self,
        ctx: &egui::Context,
        c: &mut Controls,
        settled: bool,
        config: &crate::startup::Config,
        captures: &Captures,
    ) -> bool {
        let material = config.input.is_some();
        let requested_size = config.studio_snapshot_size;
        let states = if material {
            4 + crate::material_ui::MAPS.len() + if config.readme_snapshots { 4 } else { 0 }
        } else {
            1
        };
        let size_count = if requested_size.is_some() { 1 } else { 3 };
        if self.stage == size_count * states {
            return true;
        }
        if material && c.session().snapshot().is_none() {
            return false;
        }
        let [width, height] =
            requested_size.unwrap_or([[1600, 1000], [1280, 800], [960, 600]][self.stage / states]);
        let state = self.stage % states;
        // Fixed-size review captures specify physical pixels, including at 150% DPI.
        let scale = if requested_size.is_some() {
            ctx.pixels_per_point()
        } else {
            1.
        };
        let desired_size = egui::vec2(width as f32, height as f32) / scale;
        if self.started.is_none()
            && (ctx.content_rect().size() - desired_size).abs().max_elem() > 1.
        {
            // Resize only between settled states, before requesting new GPU work.
            // Reissuing InnerSize for every map also reconfigured unchanged surfaces.
            if !material || settled {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(desired_size));
            }
            return false;
        }
        let name = format!(
            "studio-{}-{width}x{height}",
            if material {
                match state {
                    0..=3 => ["2d", "3d", "split", "export-options"][state],
                    11 => "seams-before",
                    12 => "seams-after",
                    13 => "tiled-before",
                    14 => "tiled-after",
                    _ => crate::material_ui::map_label(crate::material_ui::MAPS[state - 4]),
                }
            } else {
                "empty"
            }
        );
        if self.started.is_none() {
            if config.readme_snapshots && self.stage == 0 {
                c.export_options.destination = r"C:\Textures\Texelith".into();
                c.export_options.material_name = match config
                    .input
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .and_then(|s| s.to_str())
                {
                    Some("rock_wall_08_diff_2k") => "Rock_Wall_PBR",
                    Some("wood_planks_diff_2k") => "Wood_Planks_PBR",
                    Some("rusty_metal_02_diff_2k") => "Rusty_Metal_PBR",
                    _ => "Red_Brick_PBR",
                }
                .into();
                // Bare metal is selected by color; the rusty orange stays dielectric.
                if config
                    .input
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .is_some_and(|s| s == "rusty_metal_02_diff_2k")
                {
                    // An empty painted mask is zero. Subtracting it preserves
                    // the color selection instead of the photo default of zero metal.
                    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                        seamless: false,
                        painted_mask: true,
                    }));
                    let snapshot = c.session().snapshot().unwrap();
                    let node = snapshot.graph.material_outputs().unwrap()[&MapKind::Metallic].node;
                    let normal = snapshot.graph.material_outputs().unwrap()[&MapKind::Normal].node;
                    for (parameter, value) in [
                        (
                            ParameterId::PickedColor,
                            ParameterValue::Color([0.4, 0.4, 0.4]),
                        ),
                        (ParameterId::Radius, ParameterValue::Float(0.12)),
                        (ParameterId::Threshold, ParameterValue::Float(0.45)),
                    ] {
                        c.action(UiAction::Request(CommandRequest::SetParameter {
                            address: ParameterAddress { node, parameter },
                            value,
                            gesture: None,
                        }));
                    }
                    c.action(UiAction::Request(CommandRequest::SetParameter {
                        address: ParameterAddress {
                            node: normal,
                            parameter: ParameterId::Strength,
                        },
                        value: ParameterValue::Float(0.3),
                        gesture: None,
                    }));
                }
            }
            c.close_palette();
            c.preview.split_view = matches!(state, 2 | 3);
            c.preview.revision += 1;
            c.workspace.export_options_open = state == 3;
            c.workspace.inspector_open = None;
            if material {
                if state >= 11 {
                    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                        seamless: matches!(state, 12 | 14),
                        painted_mask: false,
                    }));
                    c.action(UiAction::Request(CommandRequest::SetTiles {
                        tiles: Tiles {
                            count: 3,
                            ..Default::default()
                        },
                    }));
                    c.action(UiAction::Request(CommandRequest::InspectSeams {
                        enabled: state <= 12,
                    }));
                }
                c.action(UiAction::Request(CommandRequest::SelectMap {
                    map: if state >= 11 {
                        MapKind::Albedo
                    } else if state < 4 {
                        texelith_core::MapKind::Normal
                    } else {
                        crate::material_ui::MAPS[state - 4]
                    },
                }));
                c.action(UiAction::Request(CommandRequest::SetMode {
                    mode: if matches!(state, 1..=3) {
                        ViewMode::ThreeD
                    } else {
                        ViewMode::TwoD
                    },
                }));
            }
            self.started = Some(Instant::now());
        } else if self.requested {
            if captures.lock().unwrap().contains_key(&name) {
                self.stage += 1;
                self.started = None;
                self.requested = false;
            }
        } else if (!material || (settled && c.map_thumbnails_current()))
            && self.started.unwrap().elapsed() > Duration::from_millis(800)
        {
            let captures = captures.clone();
            ctx.request_screenshot(move |image| {
                captures.lock().unwrap().insert(name, image);
            });
            self.requested = true;
        }
        false
    }
}
