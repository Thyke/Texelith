use super::*;
use std::time::{Duration, Instant};

#[derive(Default)]
pub(crate) struct Driver {
    stage: usize,
    ready: Option<Instant>,
    requested: bool,
}
impl Driver {
    pub fn tick(
        &mut self,
        ctx: &egui::Context,
        c: &mut Controls,
        captures: &crate::material_ui::snapshots::Captures,
    ) -> bool {
        let names = ["packing-editor", "packing-mixed", "packing-invalid"];
        if self.stage == names.len() {
            return true;
        }
        if self.ready.is_none() {
            c.close_palette();
            if self.stage == 0 {
                if c.open_packing_editor().is_err() {
                    return false;
                }
                let preset = c.packing_preset_mut().unwrap();
                preset.outputs.truncate(1);
                preset.outputs[0].channels.resize(
                    4,
                    PresetChannelSpec {
                        source: PresetChannelSource::Constant(1.0),
                        invert: false,
                        transfer: Transfer::Linear,
                    },
                );
                preset.outputs[0].import_hints = None;
                c.packing.save_name = "Stone".into();
            } else if self.stage == 1 {
                let output = &mut c.packing_preset_mut().unwrap().outputs[0];
                output.channels[1] = PresetChannelSpec {
                    source: PresetChannelSource::Map {
                        map: MapKind::Roughness,
                        component: Component::R,
                        missing: MissingMap::Error,
                    },
                    invert: true,
                    transfer: Transfer::Linear,
                };
                output.channels[2].transfer = Transfer::Linear;
            } else {
                c.export_options_mut().material_name = "../Stone".into();
            }
            self.ready = Some(Instant::now());
        } else if !self.requested && self.ready.unwrap().elapsed() > Duration::from_millis(350) {
            let captures = captures.clone();
            let name = names[self.stage].to_owned();
            ctx.request_screenshot(move |image| {
                captures.lock().unwrap().insert(name, image);
            });
            self.requested = true;
        } else if self.requested && captures.lock().unwrap().contains_key(names[self.stage]) {
            self.stage += 1;
            self.ready = None;
            self.requested = false;
        }
        false
    }
}
