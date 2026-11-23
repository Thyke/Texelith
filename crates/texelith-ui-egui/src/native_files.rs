use super::*;
use crate::input_locale::labels::text;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub(crate) struct State {
    pub enabled: bool,
    pub naming: Option<String>,
    pub folder_base: Option<PathBuf>,
    pub toast: Option<(String, std::time::Instant)>,
    pub export_detail: String,
}

pub(crate) fn default_export_folder(name: &str) -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(std::env::temp_dir);
    let documents = std::env::var_os("XDG_DOCUMENTS_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home.join("Documents"));
    documents.join("Texelith").join(name)
}

pub(crate) fn absolute_destination(path: &Path) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err(text("native-folder-required"));
    }
    let path = std::path::absolute(path).map_err(|e| format!("Folder path is invalid: {e}"))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("Choose a named material folder inside the drive or share root")?;
    texelith_io::export_transactions::validate_file_names(&[name])
        .map_err(|e| format!("Material folder: {e}"))?;
    Ok(path)
}

impl Controls {
    /// Native hosts opt in; embedded/headless hosts never open system windows.
    pub fn enable_native_dialogs(&mut self) {
        self.native_files.enabled = true;
    }

    pub(crate) fn browse_image(&mut self) {
        if !self.native_files.enabled {
            return;
        }
        let dialog = rfd::FileDialog::new().set_title(text("native-open-image"));
        let dialog = match self.import_kind {
            crate::material_ui::ImportKind::Photo => {
                dialog.add_filter("PNG / JPEG", &["png", "jpg", "jpeg"])
            }
            crate::material_ui::ImportKind::Height => {
                dialog.add_filter("PNG / EXR", &["png", "exr"])
            }
            crate::material_ui::ImportKind::Normal => dialog.add_filter("PNG", &["png"]),
        };
        if let Some(path) = dialog.pick_file() {
            self.path = path.to_string_lossy().into_owned();
            self.request_image(path);
        }
    }

    pub(crate) fn browse_export_folder(&mut self) {
        if !self.native_files.enabled {
            return;
        }
        // Pick a parent, then publish a named set inside it. Never replace the
        // entire Documents folder just because it was selected in the dialog.
        if let Some(base) = rfd::FileDialog::new()
            .set_title(text("native-export-parent"))
            .pick_folder()
        {
            self.select_export_folder(base);
        }
    }

    pub(crate) fn select_export_folder(&mut self, base: PathBuf) {
        self.export_options.destination = base.join(&self.export_options.material_name);
        self.native_files.folder_base = Some(base);
        self.export_feedback.clear();
        self.native_files.export_detail.clear();
    }

    pub(crate) fn browse_png(&mut self) {
        if self.native_files.enabled
            && let Some(path) = rfd::FileDialog::new()
                .set_title(text("native-png"))
                .add_filter("PNG", &["png"])
                .set_file_name("texture.png")
                .save_file()
        {
            self.destination = path.to_string_lossy().into_owned();
        }
    }

    pub(crate) fn browse_preview_asset(&mut self, mesh: bool) {
        if !self.native_files.enabled {
            return;
        }
        let dialog = rfd::FileDialog::new();
        let dialog = if mesh {
            dialog
                .set_title(text("native-mesh"))
                .add_filter("OBJ / glTF", &["obj", "gltf", "glb"])
        } else {
            dialog
                .set_title(text("native-hdri"))
                .add_filter("HDR / EXR", &["hdr", "exr"])
        };
        if let Some(path) = dialog.pick_file() {
            let label = path.to_string_lossy().into_owned();
            self.preview.pending = Some(if mesh {
                self.preview.mesh_path = label;
                crate::compare::assets::Load::Mesh(path)
            } else {
                self.preview.hdr_path = label;
                crate::compare::assets::Load::Hdr(path)
            });
        }
    }

    pub(crate) fn confirm_material_export(&mut self) {
        let result = (|| {
            self.packing_plan()?;
            let path = absolute_destination(&self.export_options.destination)?;
            self.export_options.destination = path.clone();
            let replace = if path.exists() {
                if !self.native_files.enabled
                    || !rfd::MessageDialog::new()
                        .set_title(text("native-overwrite-title"))
                        .set_description(format!(
                            "{}\n\n{}",
                            text("native-overwrite-message"),
                            path.display()
                        ))
                        .set_level(rfd::MessageLevel::Warning)
                        .set_buttons(rfd::MessageButtons::YesNo)
                        .show()
                        .eq(&rfd::MessageDialogResult::Yes)
                {
                    return Ok(());
                }
                true
            } else {
                false
            };
            // Permission belongs to one immutable request, never a later export.
            self.export_options.replace_existing = replace;
            let result = self.request_material_export();
            self.export_options.replace_existing = false;
            result
        })();
        if let Err(error) = result {
            self.export_message(error);
        }
    }

    pub(crate) fn open_export_folder(&mut self, path: &Path) {
        let result = open_folder(path);
        if let Err(error) = result {
            self.export_feedback = error.to_string();
        }
    }

    pub(crate) fn show_export_toast(&mut self, ctx: &egui::Context) {
        if let Some((message, started)) = &self.native_files.toast
            && started.elapsed() < std::time::Duration::from_secs(8)
        {
            egui::Area::new(egui::Id::unique("export-success-toast"))
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, 48.0])
                .order(egui::Order::Tooltip)
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_max_width(420.0);
                        ui.label(message);
                    });
                });
            ctx.request_repaint_after(std::time::Duration::from_secs(1));
        }
    }
}

fn open_folder(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = std::process::Command::new("explorer.exe");
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let mut command = std::process::Command::new("xdg-open");
    command.arg(path).spawn().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_folder_is_absolute_and_scoped_to_named_material() {
        let folder = default_export_folder("Rock");
        assert!(folder.is_absolute());
        assert!(folder.ends_with(Path::new("Texelith").join("Rock")));
        assert_eq!(absolute_destination(&folder).unwrap(), folder);
        assert!(absolute_destination(Path::new("")).is_err());
        assert!(
            absolute_destination(Path::new("relative-export"))
                .unwrap()
                .is_absolute()
        );
    }

    #[test]
    fn naming_preview_matches_captured_export_and_rejects_invalid_templates() {
        let mut c = crate::input_locale::contract::fixture().unwrap();
        c.native_files.naming = Some("{name}_{preset}_{index}_{map}".into());
        c.export_options.material_name = "Rock".into();
        let names = c.packing_names().unwrap();
        assert!(names[0].starts_with("Rock_unreal-v1_0000_"));
        c.request_material_export().unwrap();
        let Effect::Export { plan, .. } = c.take_effects().remove(0) else {
            panic!("missing export");
        };
        assert_eq!(crate::packing_editor::names(&plan, "Rock").unwrap(), names);
        for invalid in ["../{name}", "{unknown}", "same-file"] {
            c.native_files.naming = Some(invalid.into());
            assert!(c.request_material_export().is_err());
            assert!(c.take_effects().is_empty());
        }
    }

    #[test]
    fn unconfirmed_existing_destination_never_emits_an_export() {
        let mut c = crate::input_locale::contract::fixture().unwrap();
        c.export_options.destination = std::env::current_dir().unwrap();
        c.confirm_material_export();
        assert!(c.take_effects().is_empty());
        assert!(!c.export_options.replace_existing);
    }

    #[test]
    fn completion_toast_uses_captured_destination_and_does_not_restart() {
        let mut c = Controls::new_material();
        let destination = default_export_folder("Captured");
        let status = crate::export_jobs::JobStatus {
            job_id: 1,
            stage: texelith_gpu::ExportStage::Completed,
            output_name: String::new(),
            completed_items: 1,
            total_items: 1,
            destination: destination.clone(),
            error: None,
            cancel_requested: false,
        };
        c.set_export_status(Some(status.clone()), false);
        let (message, started) = c.native_files.toast.clone().unwrap();
        assert!(message.contains(destination.to_str().unwrap()));
        c.export_options.destination = default_export_folder("Later-edit");
        c.set_export_status(Some(status), false);
        assert_eq!(c.native_files.toast.unwrap(), (message, started));
    }

    fn path_frame(
        c: &mut Controls,
        ctx: &egui::Context,
        field: usize,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1600.0, 1400.0),
                )),
                focused: true,
                events,
                ..Default::default()
            },
            |ui| {
                c.global_input(ui.ctx());
                match field {
                    // The text-path editor remains in the legacy leaf; Layer Studio uses Open.
                    0 => c.show_simple_inspector(ui),
                    1 | 2 => c.show_simple_export(ui),
                    3 | 4 => c.viewport_settings(ui),
                    _ => c.input_dialogs(ui),
                }
            },
        );
        output.textures_delta.clear();
        output
    }

    fn text_position(shape: &egui::epaint::Shape, wanted: &str) -> Option<egui::Pos2> {
        match shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == wanted => {
                Some(text.pos + text.galley.size() * 0.5)
            }
            egui::epaint::Shape::Vec(shapes) => {
                shapes.iter().find_map(|shape| text_position(shape, wanted))
            }
            _ => None,
        }
    }

    #[test]
    fn path_rows_support_select_all_copy_and_paste_without_global_shortcut_interference() {
        use crate::input_locale::contract::key;
        let initial = "before_path_142";
        let pasted = "C:\\Textures\\Kaya y\u{fc}zeyi.png";
        for field in 0..7 {
            let mut c = crate::input_locale::contract::fixture().unwrap();
            c.action(UiAction::Request(CommandRequest::SetMode {
                mode: texelith_core::ViewMode::ThreeD,
            }));
            c.take_effects();
            match field {
                0 => c.path = initial.into(),
                1 => c.destination = initial.into(),
                2 => c.export_options.destination = initial.into(),
                3 => c.preview.mesh_path = initial.into(),
                4 => c.preview.hdr_path = initial.into(),
                5 => {
                    c.input.project_picker = true;
                    c.input.project_path = initial.into();
                }
                _ => {
                    c.projects.save_picker = true;
                    c.projects.save_path = initial.into();
                }
            }
            let ctx = egui::Context::default();
            ctx.all_styles_mut(|style| style.animation_time = 0.0);
            let first = path_frame(&mut c, &ctx, field, vec![]);
            let header = match field {
                0 => Some(crate::workspace::labels::text("import")),
                1 => Some(format!(
                    "\u{203a} {}",
                    crate::workspace::labels::text("more-options")
                )),
                2 => None,
                3 | 4 => Some(crate::compare::labels::text("assets")),
                _ => None,
            };
            if let Some(header) = header {
                let pos = first
                    .shapes
                    .iter()
                    .find_map(|s| text_position(&s.shape, &header))
                    .expect("path section rendered");
                for pressed in [true, false] {
                    path_frame(
                        &mut c,
                        &ctx,
                        field,
                        vec![
                            egui::Event::PointerMoved(pos),
                            egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Primary,
                                pressed,
                                modifiers: egui::Modifiers::NONE,
                            },
                        ],
                    );
                }
            }
            let frame = path_frame(&mut c, &ctx, field, vec![]);
            let pos = frame
                .shapes
                .iter()
                .find_map(|s| text_position(&s.shape, initial))
                .unwrap_or_else(|| panic!("path field {field} rendered"));
            for pressed in [true, false] {
                path_frame(
                    &mut c,
                    &ctx,
                    field,
                    vec![
                        egui::Event::PointerMoved(pos),
                        egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
            let modifiers = egui::Modifiers {
                ctrl: true,
                command: true,
                ..egui::Modifiers::NONE
            };
            path_frame(&mut c, &ctx, field, vec![key(egui::Key::A, modifiers)]);
            // winit turns Ctrl+C / Ctrl+V into these clipboard events. The
            // native clipboard feature is checked separately in the host gate.
            let copied = path_frame(&mut c, &ctx, field, vec![egui::Event::Copy]);
            assert!(copied.platform_output.commands.iter().any(|command| matches!(command, egui::OutputCommand::CopyText(text) if text == initial)), "copy failed for path row {field}: {:?}, focus {:?}, at {pos:?}", copied.platform_output.commands, ctx.memory(|m| m.focused()));
            path_frame(&mut c, &ctx, field, vec![egui::Event::Paste(pasted.into())]);
            let actual = match field {
                0 => c.path,
                1 => c.destination,
                2 => c.export_options.destination.to_string_lossy().into_owned(),
                3 => c.preview.mesh_path,
                4 => c.preview.hdr_path,
                5 => c.input.project_path,
                _ => c.projects.save_path,
            };
            assert_eq!(actual, pasted, "paste failed for path row {field}");
        }
    }
}
