//! Global input is translated to the same typed commands used by the panels.
#[doc(hidden)]
pub mod contract;
pub mod labels;
pub mod smoke;
pub(crate) mod snapshots;

use std::path::PathBuf;
use texelith_core::CommandRequest;

#[derive(Clone, Debug, PartialEq)]
pub enum InputAction {
    Request(CommandRequest),
    Palette,
    OpenProject,
    SaveProject { save_as: bool },
}

pub fn shortcut(
    key: egui::Key,
    modifiers: egui::Modifiers,
    editing: bool,
    palette: bool,
) -> Option<InputAction> {
    use egui::Key;
    if modifiers.alt {
        return None;
    }
    let command = modifiers.command || modifiers.ctrl || modifiers.mac_cmd;
    if command {
        return match key {
            Key::K if !modifiers.shift => Some(InputAction::Palette),
            Key::O if !modifiers.shift && !palette => Some(InputAction::OpenProject),
            Key::S if !palette => Some(InputAction::SaveProject {
                save_as: modifiers.shift,
            }),
            Key::Z if !editing && !palette => Some(InputAction::Request(if modifiers.shift {
                CommandRequest::Redo
            } else {
                CommandRequest::Undo
            })),
            Key::Y if !editing && !palette && !modifiers.shift => {
                Some(InputAction::Request(CommandRequest::Redo))
            }
            _ => None,
        };
    }
    if editing || palette || modifiers.shift {
        return None;
    }
    [
        Key::Num1,
        Key::Num2,
        Key::Num3,
        Key::Num4,
        Key::Num5,
        Key::Num6,
        Key::Num7,
    ]
    .into_iter()
    .zip(crate::material_ui::MAPS)
    .find(|(candidate, _)| *candidate == key)
    .map(|(_, map)| InputAction::Request(CommandRequest::SelectMap { map }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropIntent {
    Image(PathBuf),
    Project(PathBuf),
    Batch(Vec<PathBuf>),
    Environment(PathBuf),
    Mesh(PathBuf),
}

/// Classify names only; decoding and filesystem access stay on the IO worker.
/// Mixed lists are rejected as a whole instead of silently using the first file.
pub fn classify_drop(paths: &[PathBuf]) -> Result<DropIntent, &'static str> {
    let image = |path: &std::path::Path| {
        path.extension().and_then(|s| s.to_str()).is_some_and(|s| {
            ["png", "jpg", "jpeg"]
                .iter()
                .any(|ext| s.eq_ignore_ascii_case(ext))
        })
    };
    if paths.len() > 1 && paths.iter().all(|path| image(path)) {
        return Ok(DropIntent::Batch(paths.to_vec()));
    }
    let [path] = paths else {
        return Err("drop-unsupported");
    };
    if path
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.to_ascii_lowercase().ends_with(".texelith.json"))
    {
        return Ok(DropIntent::Project(path.clone()));
    }
    if image(path) {
        return Ok(DropIntent::Image(path.clone()));
    }
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "hdr" | "exr" => Ok(DropIntent::Environment(path.clone())),
        "obj" | "gltf" | "glb" => Ok(DropIntent::Mesh(path.clone())),
        _ => Err("drop-unsupported"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirtyChoice {
    Save,
    Discard,
    Cancel,
}

#[derive(Default)]
pub(crate) struct InputState {
    pub pending: Option<CapturedIntent>,
    pub batch: Vec<PathBuf>,
    pub feedback: Option<&'static str>,
    pub project_picker: bool,
    pub project_path: String,
    pub last_frame: Option<u64>,
    pub palette_focus: bool,
}

pub(crate) struct CapturedIntent {
    pub intent: DropIntent,
    pub import_kind: crate::material_ui::ImportKind,
    pub convention: texelith_core::NormalConvention,
    pub linear_height: bool,
}
