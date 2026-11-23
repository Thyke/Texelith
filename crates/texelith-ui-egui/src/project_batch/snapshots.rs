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
        let names = [
            "projects-header",
            "projects-save-as",
            "projects-batch",
            "projects-missing-source",
            "projects-changed-source",
        ];
        if self.stage == names.len() {
            return true;
        }
        if self.ready.is_none() {
            c.close_palette();
            c.projects.batch_open = false;
            c.projects.save_picker = false;
            if self.stage == 1 {
                c.project_save_intent(true);
            }
            if self.stage == 2 {
                c.projects.batch_open = true;
                c.projects.batch_folder = "Textures/Stone".into();
                c.projects.batch_destination = "Exports/Stone".into();
                c.projects.batch_manifest = "Exports/Stone/batch.texelith-job.json".into();
            }
            if self.stage == 3 {
                let Some(source) = c.projects.assets.values().next().cloned() else {
                    return false;
                };
                c.open_project_path(source.parent().unwrap().join("material.texelith.json"));
            }
            if self.stage == 4 {
                let Some(issue) = c.project_asset_issues().first() else {
                    return false;
                };
                let relink = issue.path.parent().unwrap().join("relinked.png");
                c.projects.relink_path = relink.to_string_lossy().into_owned();
                if c.resolve_project_asset(AssetChoice::Relink(relink))
                    .is_err()
                {
                    return false;
                }
            }
            self.ready = Some(Instant::now());
        } else if !self.requested
            && self.ready.unwrap().elapsed() > Duration::from_millis(350)
            && (self.stage < 3 || !c.project_asset_issues().is_empty())
        {
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
