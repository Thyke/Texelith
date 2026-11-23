use crate::{controls::Controls, input_locale::DirtyChoice};
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
        let names = [
            "input-dirty",
            "input-save-pending",
            "input-batch",
            "input-environment",
            "input-mesh",
            "input-unsupported",
        ];
        if self.stage >= names.len() {
            return true;
        }
        if self.ready.is_none() {
            c.close_palette();
            match self.stage {
                0 => c.drop_paths(vec!["replacement.png".into()]),
                1 => c.resolve_dirty(DirtyChoice::Save),
                2 => {
                    c.resolve_dirty(DirtyChoice::Cancel);
                    c.drop_paths(vec!["a.png".into(), "b.jpg".into()]);
                }
                3 => c.drop_paths(vec!["studio.hdr".into()]),
                4 => c.drop_paths(vec!["preview.glb".into()]),
                _ => c.drop_paths(vec!["unsupported.txt".into()]),
            }
            self.ready = Some(Instant::now());
        } else if !self.requested && self.ready.unwrap().elapsed() >= Duration::from_millis(350) {
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
