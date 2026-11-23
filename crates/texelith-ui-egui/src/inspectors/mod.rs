//! Procedural inspector state; all document changes go through Controls::action.
pub(crate) mod key_controls;
pub mod labels;
pub(crate) mod rows;
pub(crate) mod seamless;
pub mod smoke;
use texelith_core::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintPhase {
    Begin,
    Append,
    End,
}

/// Captures the document, selected output and edit target before asynchronous readback.
#[derive(Clone, Debug, PartialEq)]
pub struct PickRequest {
    pub token: u64,
    pub generation: u64,
    pub request_id: u64,
    pub document_revision: u64,
    pub output: OutputRef,
    pub address: ParameterAddress,
    pub uv: [f32; 2],
}

pub(crate) struct InspectorState {
    pub painting: bool,
    pub brush: Brush,
    pub next_stroke: u64,
    pub last_uv: Option<[f32; 2]>,
    pub armed: Option<ParameterAddress>,
    pub pending: Option<PickRequest>,
    pub submitted: bool,
    pub next_pick: u64,
    pub viewport: Option<egui::Rect>,
    pub parameter_rects: Vec<(ParameterAddress, Vec<egui::Rect>)>,
    pub quality_status: String,
    pub advanced: bool,
}
impl Default for InspectorState {
    fn default() -> Self {
        Self {
            painting: false,
            brush: Brush {
                radius: 0.04,
                hardness: 0.5,
                opacity: 1.0,
                mode: MaskMode::Replace,
            },
            next_stroke: 0,
            last_uv: None,
            armed: None,
            pending: None,
            submitted: false,
            next_pick: 0,
            viewport: None,
            parameter_rects: Default::default(),
            quality_status: String::new(),
            advanced: false,
        }
    }
}
impl InspectorState {
    pub fn cancel_transients(&mut self) {
        self.armed = None;
        self.pending = None;
        self.submitted = false;
        self.last_uv = None;
    }
}

/// Inclusive image UV; letterbox margins never yield samples.
pub fn source_uv(rect: egui::Rect, position: egui::Pos2, tiles: u32) -> Option<[f32; 2]> {
    if rect.width() <= 0.0 || rect.height() <= 0.0 || !rect.contains(position) {
        return None;
    }
    let local = (position - rect.min) / rect.size();
    let tile = tiles.max(1) as f32;
    Some([local.x, local.y].map(|v| {
        let v = v.clamp(0.0, 1.0) * tile;
        if v == tile { 1.0 } else { v.fract() }
    }))
}
