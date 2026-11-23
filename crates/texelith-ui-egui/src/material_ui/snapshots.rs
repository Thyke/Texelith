use crate::{command_adapter::UiAction, controls::Controls};
use std::sync::{Arc, Mutex};
use texelith_core::*;
pub(crate) type Captures = Arc<Mutex<std::collections::BTreeMap<String, Arc<egui::ColorImage>>>>;
#[derive(Default)]
pub(crate) struct Driver {
    stage: usize,
    baseline: u64,
    requested: bool,
    started: bool,
    ready_since: Option<std::time::Instant>,
}
impl Driver {
    pub fn tick(
        &mut self,
        ctx: &egui::Context,
        c: &mut Controls,
        last_id: u64,
        settled: bool,
        captures: &Captures,
    ) -> bool {
        let names = [
            "albedo",
            "height",
            "normal",
            "ao",
            "roughness",
            "metallic",
            "edge",
            "heatmap",
            "original",
            "painted-mask",
        ];
        if self.stage >= names.len() {
            return true;
        }
        if !self.started {
            c.close_palette();
            let request = match self.stage {
                0..=6 => CommandRequest::SelectMap {
                    map: super::MAPS[self.stage],
                },
                7 => CommandRequest::InspectSeams { enabled: true },
                8 => {
                    c.action(UiAction::Request(CommandRequest::InspectSeams {
                        enabled: false,
                    }));
                    CommandRequest::ConfigureMaterial {
                        seamless: false,
                        painted_mask: false,
                    }
                }
                _ => {
                    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
                        seamless: true,
                        painted_mask: true,
                    }));
                    let snapshot = c.session().snapshot().unwrap();
                    let node = snapshot.graph.material_outputs().unwrap()[&MapKind::Mask].node;
                    let stroke_id = StrokeId(999);
                    c.action(UiAction::Request(CommandRequest::BeginStroke {
                        node,
                        stroke_id,
                        brush: Brush {
                            radius: 0.2,
                            hardness: 0.5,
                            opacity: 1.0,
                            mode: MaskMode::Replace,
                        },
                        sample: BrushSample {
                            uv: [0.5, 0.5],
                            pressure: 1.0,
                        },
                    }));
                    c.action(UiAction::Request(CommandRequest::CommitStroke {
                        stroke_id,
                    }));
                    CommandRequest::SelectMap { map: MapKind::Mask }
                }
            };
            c.action(UiAction::Request(request));
            self.baseline = last_id;
            self.started = true;
        } else if settled && last_id > self.baseline && !self.requested {
            let ready_since = self.ready_since.get_or_insert_with(std::time::Instant::now);
            if ready_since.elapsed() < std::time::Duration::from_millis(300) {
                return false;
            }
            let name = names[self.stage].to_string();
            let captures = captures.clone();
            ctx.request_screenshot(move |image| {
                captures.lock().unwrap().insert(name, image);
            });
            self.requested = true;
        } else if self.requested && captures.lock().unwrap().contains_key(names[self.stage]) {
            self.ready_since = None;
            self.stage += 1;
            self.started = false;
            self.requested = false;
        }
        false
    }
}
