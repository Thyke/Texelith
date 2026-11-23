//! Panel slots and bounded background previews shared by the Simple shell.
use std::collections::BTreeMap;
use texelith_core::{EvaluationSnapshot, MapKind, MaterialNodeParams};

#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub header_content: Option<egui::Rect>,
    pub viewport_toolbar: Option<egui::Rect>,
    pub view_bar: Option<egui::Rect>,
    pub heatmap: Option<egui::Rect>,
    pub heatmap_info: Option<egui::Rect>,
    pub compare: Option<egui::Rect>,
    pub tool_rail: Option<egui::Rect>,
    pub mode_buttons: [Option<egui::Rect>; 3],
    pub maps: Option<egui::Rect>,
    pub inspector: Option<egui::Rect>,
    pub export: Option<egui::Rect>,
    pub center: Option<egui::Rect>,
    pub cards: BTreeMap<MapKind, Card>,
    pub maps_toggle: Option<egui::Rect>,
    pub inspector_toggle: Option<egui::Rect>,
    pub save_png: Option<egui::Rect>,
    pub destination: Option<egui::Rect>,
    pub export_material: Option<egui::Rect>,
    pub export_options: Option<egui::Rect>,
    pub export_options_area: Option<egui::Rect>,
    pub open_export_folder: Option<egui::Rect>,
    pub document_tab: Option<egui::Rect>,
    pub export_channels: BTreeMap<MapKind, egui::Rect>,
}

#[derive(Clone, Debug)]
pub struct Card {
    pub select: egui::Rect,
    pub visibility: egui::Rect,
    pub lock: egui::Rect,
    pub thumbnail: egui::Rect,
    pub badge: &'static str,
}

pub(crate) struct Thumbnail {
    pub generation: u64,
    pub image: egui::ColorImage,
    pub texture: Option<egui::TextureHandle>,
}

#[derive(Default)]
pub(crate) struct Workspace {
    pub layout: Layout,
    pub maps_open: Option<bool>,
    pub inspector_open: Option<bool>,
    pub thumbnails: BTreeMap<MapKind, Thumbnail>,
    pub status: String,
    pub busy: bool,
    pub gpu: String,
    pub debug_layers: bool,
    pub export_options_open: bool,
    pub paint_tab: bool,
    pub close_document: bool,
    pub about: bool,
    pub view_settings: bool,
    pub last_width: Option<f32>,
    pub export_maps: Option<std::collections::BTreeSet<MapKind>>,
}

impl Workspace {
    pub fn inspector_is_open(&self, logical_width: f32) -> bool {
        self.inspector_open.unwrap_or(logical_width >= 1000.)
    }

    pub fn clear_thumbnails(&mut self) {
        self.thumbnails.clear();
    }
}

/// Leaf panels may compose these slots without depending on private workspace state.
pub fn export_slot<R>(
    root: &mut egui::Ui,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    egui::Panel::bottom("material_export")
        .default_size(60.0)
        .min_size(0.0)
        .resizable(false)
        .show(root, content)
}

pub fn inspector_slot<R>(
    root: &mut egui::Ui,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    egui::Panel::right("material_inspector")
        .default_size(400.0)
        .min_size(240.0)
        .max_size(400.0)
        .resizable(true)
        .show(root, |ui| {
            bounded_panel_contents(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, content).inner
            })
        })
}

/// Keep the panel's remembered/dragged size independent of its children's minimum sizes.
/// egui's panel frame otherwise grows to fit even a single overflowing numeric editor.
pub(crate) fn bounded_panel_contents<R>(
    ui: &mut egui::Ui,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let rect = ui.available_rect_before_wrap();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    child.shrink_clip_rect(rect);
    let result = content(&mut child);
    ui.allocate_rect(rect, egui::Sense::hover());
    result
}

pub fn center_slot<R>(
    root: &mut egui::Ui,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    egui::CentralPanel::default().show(root, content)
}

pub fn source_badge(snapshot: &EvaluationSnapshot, map: MapKind) -> &'static str {
    let manual = snapshot.graph.material_nodes().iter().any(|node| {
        matches!(
            (map, &node.params),
            (MapKind::Albedo, MaterialNodeParams::ColorSource(_))
                | (MapKind::Height, MaterialNodeParams::HeightSource(_))
                | (MapKind::Normal, MaterialNodeParams::NormalSource { .. })
                | (MapKind::Mask, MaterialNodeParams::PaintMask { .. })
        )
    });
    if manual { "Manual" } else { "Procedural" }
}

/// Samples an existing presented image; never triggers a GPU readback or retains
/// the full-resolution image. Seven RGBA previews consume at most 448 KiB on GPU.
pub fn thumbnail(image: &egui::ColorImage) -> egui::ColorImage {
    let [width, height] = image.size;
    let factor = (128.0 / width.max(height) as f32).min(1.0);
    let size = [
        (width as f32 * factor).round().max(1.0) as usize,
        (height as f32 * factor).round().max(1.0) as usize,
    ];
    let mut pixels = Vec::with_capacity(size[0] * size[1]);
    for y in 0..size[1] {
        for x in 0..size[0] {
            let sx = ((2 * x + 1) * width / (2 * size[0])).min(width - 1);
            let sy = ((2 * y + 1) * height / (2 * size[1])).min(height - 1);
            pixels.push(image.pixels[sy * width + sx]);
        }
    }
    egui::ColorImage::new(size, pixels)
}
