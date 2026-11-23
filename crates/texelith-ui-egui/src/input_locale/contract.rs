use crate::{command_adapter::UiAction, controls::Controls};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use texelith_core::*;

#[derive(Debug)]
pub struct PathDrop(pub PathBuf);
impl egui::DroppedFile for PathDrop {
    fn path(&self) -> &Path {
        &self.0
    }
    fn bytes(&self) -> Result<Vec<u8>, String> {
        panic!("drop bytes must not be read by the UI")
    }
}

pub fn frame(
    c: &mut Controls,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    drops: &[PathBuf],
    edit: bool,
) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            focused: true,
            events,
            dropped_files: drops
                .iter()
                .map(|p| Arc::new(PathDrop(p.clone())) as egui::DroppedFileHandle)
                .collect(),
            ..Default::default()
        },
        |ui| {
            if edit {
                egui::Window::new("Focus fixture").show(ui, |ui| {
                    ui.text_edit_singleline(&mut String::new()).request_focus();
                });
            }
            c.show(ui);
            c.show_workspace_center(ui, |_| {});
        },
    );
    output.textures_delta.clear();
}
pub fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

pub fn fixture() -> Result<Controls, String> {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        return Err("missing fixture load effect".into());
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [3; 32],
            extent: Extent {
                width: 64,
                height: 32,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )?;
    Ok(c)
}

/// Headless event-shape coverage, also run by the hardware diagnostic.
/// This deliberately makes no claim about native focus or window responsiveness.
pub fn exercise() -> Result<Vec<String>, String> {
    let mut checks = vec![];
    for (key, map) in [
        egui::Key::Num1,
        egui::Key::Num2,
        egui::Key::Num3,
        egui::Key::Num4,
        egui::Key::Num5,
        egui::Key::Num6,
        egui::Key::Num7,
    ]
    .into_iter()
    .zip(crate::material_ui::MAPS)
    {
        let mut c = fixture()?;
        let ctx = egui::Context::default();
        frame(
            &mut c,
            &ctx,
            vec![self::key(key, egui::Modifiers::NONE)],
            &[],
            false,
        );
        if c.session()
            .snapshot()
            .unwrap()
            .graph
            .selected_output_ref()
            .port
            .map_kind()
            != map
        {
            return Err(format!("key {key:?} did not select {map:?}"));
        }
        checks.push(format!("key-{key:?}-selects-{map:?}"));
    }
    for panel in ["maps", "inspector", "export", "center"] {
        let mut c = fixture()?;
        let ctx = egui::Context::default();
        for _ in 0..3 {
            frame(&mut c, &ctx, vec![], &[], false);
        }
        let layout = c.workspace_layout();
        let rect = match panel {
            "maps" => layout.maps,
            "inspector" => layout.inspector,
            "export" => layout.export,
            _ => layout.center,
        }
        .ok_or("missing panel")?;
        let old = c.session().snapshot().unwrap().generation;
        frame(
            &mut c,
            &ctx,
            vec![egui::Event::PointerMoved(rect.center())],
            &["dropped.png".into()],
            false,
        );
        if c.pending_replacement().is_none() || !c.take_effects().is_empty() {
            return Err(format!(
                "drop on {panel} did not guard the unsaved material"
            ));
        }
        c.resolve_dirty(super::DirtyChoice::Discard);
        if !matches!(c.take_effects().as_slice(), [Effect::LoadImage { .. }])
            || c.session().snapshot().unwrap().generation != old
        {
            return Err(format!("drop on {panel} bypassed effect routing"));
        }
        frame(&mut c, &ctx, vec![], &[], false);
        if !c.take_effects().is_empty() {
            return Err("drop was dispatched twice".into());
        }
        checks.push(format!("drop-{panel}-one-effect-no-frame-thread-read"));
    }
    let mut c = fixture()?;
    let ctx = egui::Context::default();
    frame(&mut c, &ctx, vec![], &[], true);
    let selected = c.session().snapshot().unwrap().graph.selected_output_ref();
    frame(
        &mut c,
        &ctx,
        vec![key(egui::Key::Num7, egui::Modifiers::NONE)],
        &[],
        true,
    );
    if c.session().snapshot().unwrap().graph.selected_output_ref() != selected {
        return Err("map shortcut stole text input".into());
    }
    checks.push("text-focus-guards-map-shortcut".into());
    c.input_action(super::InputAction::Palette);
    frame(
        &mut c,
        &ctx,
        vec![key(egui::Key::Num7, egui::Modifiers::NONE)],
        &[],
        false,
    );
    if c.session().snapshot().unwrap().graph.selected_output_ref() != selected {
        return Err("map shortcut stole palette input".into());
    }
    checks.push("palette-guards-map-shortcut".into());
    c.action(UiAction::Request(CommandRequest::SetLocked {
        map: MapKind::Normal,
        locked: true,
    }));
    let revision = c.session().document_revision();
    c.drop_paths(vec!["replacement.png".into()]);
    c.resolve_dirty(super::DirtyChoice::Cancel);
    if c.pending_replacement().is_some()
        || !c.take_effects().is_empty()
        || c.session().document_revision() != revision
    {
        return Err("dirty cancel changed the document".into());
    }
    checks.push("dirty-cancel-retains-document".into());
    c.drop_paths(vec!["replacement.png".into()]);
    c.resolve_dirty(super::DirtyChoice::Save);
    if c.pending_replacement().is_none() || !c.take_effects().is_empty() || !c.session().is_dirty()
    {
        return Err("pending save falsely succeeded".into());
    }
    checks.push("save-unavailable-retains-dirty-intent".into());
    c.resolve_dirty(super::DirtyChoice::Discard);
    if !matches!(c.take_effects().as_slice(), [Effect::LoadImage { .. }]) {
        return Err("discard did not emit import".into());
    }
    checks.push("discard-routes-import-effect".into());
    super::labels::validate_catalogs()?;
    checks.push("all-four-en-US-tr-TR-catalogs-covered".into());
    Ok(checks)
}
