//! Simple workspace diagnostics.
pub(crate) mod labels;
mod layout;
pub(crate) mod snapshots;
pub(crate) mod studio;
pub(crate) use layout::Workspace;
pub(crate) use layout::bounded_panel_contents;
pub use layout::{Card, Layout, center_slot, export_slot, inspector_slot, source_badge, thumbnail};

use crate::{command_adapter::UiAction, controls::Controls, material_ui::MAPS, startup::Config};
use texelith_core::{CommandRequest, Effect, Extent, InputInterpretation, SourceDescriptor};

impl Controls {
    pub(crate) fn map_canvas_rect(&self, bounds: egui::Rect) -> egui::Rect {
        egui::Rect::from_center_size(
            bounds.center() + self.preview.pan,
            bounds.size() * (self.preview.zoom / 100.),
        )
    }
    pub fn workspace_layout(&self) -> &Layout {
        &self.workspace.layout
    }

    /// Cache a bounded preview from presentation or the background thumbnail worker.
    pub fn cache_map_thumbnail(
        &mut self,
        map: texelith_core::MapKind,
        generation: u64,
        image: &egui::ColorImage,
    ) {
        self.workspace.thumbnails.insert(
            map,
            layout::Thumbnail {
                generation,
                image: thumbnail(image),
                texture: None,
            },
        );
    }

    pub fn map_thumbnail_size(&self, map: texelith_core::MapKind) -> Option<[usize; 2]> {
        self.workspace.thumbnails.get(&map).map(|t| t.image.size)
    }

    /// Whether every available map card has finished refreshing for this document.
    pub fn map_thumbnails_current(&self) -> bool {
        self.session().snapshot().is_some_and(|snapshot| {
            MAPS.into_iter().all(|map| {
                !crate::material_ui::supported(&snapshot, map)
                    || self.thumbnail_generation(map) == Some(snapshot.generation)
            })
        })
    }

    /// Compose the viewport slot independently from inspector/export leaf content.
    pub fn show_workspace_center<R>(
        &mut self,
        root: &mut egui::Ui,
        content: impl FnOnce(&mut egui::Ui) -> R,
    ) -> R {
        let response = center_slot(root, content);
        self.workspace.layout.center = Some(response.response.rect);
        response.inner
    }
}

pub fn neutral_preview(map: texelith_core::MapKind) -> egui::Color32 {
    use texelith_core::MapKind;
    match map {
        MapKind::Normal => egui::Color32::from_rgb(128, 128, 255),
        MapKind::Ao => egui::Color32::WHITE,
        MapKind::Albedo | MapKind::Height | MapKind::Roughness => egui::Color32::from_gray(128),
        _ => egui::Color32::BLACK,
    }
}

pub fn exercise() -> Result<usize, String> {
    let mut c = Controls::new_material();
    c.open_image("fixture.png".into());
    let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
        return Err("missing import effect".into());
    };
    c.complete_import(
        token,
        SourceDescriptor {
            content_hash: [7; 32],
            extent: Extent {
                width: 64,
                height: 64,
            },
            interpretation: InputInterpretation::SrgbPhoto,
        },
    )?;
    let mut checks = 0;
    for map in MAPS {
        c.action(UiAction::Request(CommandRequest::SelectMap { map }));
        if c.session()
            .snapshot()
            .unwrap()
            .graph
            .selected_output_ref()
            .port
            .map_kind()
            != map
        {
            return Err("selection failed".into());
        }
        c.action(UiAction::Request(CommandRequest::SetVisible {
            map,
            visible: false,
        }));
        c.action(UiAction::Request(CommandRequest::SetLocked {
            map,
            locked: true,
        }));
        let state = c
            .session()
            .view_state()
            .maps
            .get(&map)
            .ok_or("missing map state")?;
        if state.visible || !state.locked {
            return Err("map state failed".into());
        }
        c.save_png("out.png".into(), map);
        if !matches!(c.take_effects().as_slice(), [Effect::SavePng { map: saved, .. }] if *saved == map)
        {
            return Err("save effect failed".into());
        }
        checks += 3;
    }
    let image = egui::ColorImage::filled([256, 128], egui::Color32::from_rgb(20, 40, 80));
    for map in MAPS {
        c.cache_map_thumbnail(map, c.session().snapshot().unwrap().generation, &image);
        if c.map_thumbnail_size(map) != Some([128, 64]) {
            return Err("thumbnail dimensions exceed bounded preview contract".into());
        }
        checks += 1;
    }
    for (size, scale) in [[960.0, 600.0], [1280.0, 720.0]]
        .into_iter()
        .flat_map(|size| {
            [1.0, 1.25, 1.5, 2.0]
                .into_iter()
                .map(move |scale| (size, scale))
        })
    {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(scale);
        let size = egui::vec2(size[0] / scale, size[1] / scale);
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            focused: true,
            ..Default::default()
        };
        // Let egui settle intrinsic panel sizes before checking the final geometry.
        for _ in 0..3 {
            let mut output = ctx.run_ui(input.clone(), |ui| {
                c.show(ui);
                c.show_workspace_center(ui, |_| {});
            });
            output.textures_delta.clear();
        }
        let layout = c.workspace_layout();
        let center = layout.center.ok_or("missing viewport slot")?;
        let export = layout.export.ok_or("missing export slot")?;
        if center.width() <= 0.0
            || center.height() <= 0.0
            || center.intersects(export)
            || !egui::Rect::from_min_size(egui::Pos2::ZERO, size)
                .expand(0.5 / scale)
                .contains_rect(export)
            || layout.cards.len() != 7
        {
            return Err(format!("panels do not fit at {size:?} / {scale}"));
        }
        checks += 1;
    }
    Ok(checks)
}

pub fn smoke(config: &Config) -> Result<(), String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = config.backends()?;
    let instance = wgpu::Instance::new(descriptor);
    let adapters = crate::material_ui::block_on(instance.enumerate_adapters(config.backends()?));
    let adapter = adapters
        .into_iter()
        .find(|a| {
            config.adapter_name.as_ref().is_none_or(|name| {
                a.get_info()
                    .name
                    .to_lowercase()
                    .contains(&name.to_lowercase())
            })
        })
        .ok_or("Requested native GPU adapter unavailable")?;
    crate::startup::validate_adapter(&adapter)?;
    let info = adapter.get_info();
    let checks = exercise()?;
    let report = config.report.as_ref().ok_or("missing report")?;
    if let Some(parent) = report.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let value = serde_json::json!({"task":"T-071", "scenario":"simple_layout", "passed":true, "assertions":checks, "adapter":info.name, "backend":format!("{:?}", info.backend), "requested_size":config.size, "fixture_extent":[64,64], "coverage":"Session commands/effects, thumbnail bounds and egui panel geometry; no GPU evaluation", "native_input":"not measured", "set_export":"explicitly disabled until T-082", "thumbnails":"lazy cache of existing presented previews; no additional readback"});
    std::fs::write(
        report,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
