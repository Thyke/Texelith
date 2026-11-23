//! Static registry metadata and frontend-independent command transport values.

use crate::{
    Boundary, Brush, BrushSample, CoreError, EvaluationSnapshot, ExportCapture, ExportPlan,
    ExportPreset, MATERIAL_PARAMETER_DESCRIPTORS, MapKind, MaskMode, MaterialParameterDescriptor,
    NodeId, NormalConvention, NormalFilter, ParameterId, SeamlessMethod, StrokeId,
};
use serde::{Deserialize, Serialize};

#[path = "view/mod.rs"]
mod view;
pub use view::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandId {
    #[serde(rename = "material.configure")]
    ConfigureMaterial,
    #[serde(rename = "view.inspect_seams")]
    InspectSeams,
    #[serde(rename = "document.new_from_image")]
    NewFromImage,
    #[serde(rename = "node.set_parameter")]
    SetParameter,
    #[serde(rename = "edit.begin_gesture")]
    BeginGesture,
    #[serde(rename = "edit.commit_gesture")]
    CommitGesture,
    #[serde(rename = "edit.cancel_gesture")]
    CancelGesture,
    #[serde(rename = "mask.begin_stroke")]
    BeginStroke,
    #[serde(rename = "mask.append_stroke")]
    AppendStroke,
    #[serde(rename = "mask.commit_stroke")]
    CommitStroke,
    #[serde(rename = "mask.cancel_stroke")]
    CancelStroke,
    #[serde(rename = "edit.undo")]
    Undo,
    #[serde(rename = "edit.redo")]
    Redo,
    #[serde(rename = "view.select_output")]
    SelectOutput,
    #[serde(rename = "material.select_map")]
    SelectMap,
    #[serde(rename = "material.set_visible")]
    SetVisible,
    #[serde(rename = "material.set_locked")]
    SetLocked,
    #[serde(rename = "view.set_mode")]
    SetMode,
    #[serde(rename = "view.set_mesh")]
    SetMesh,
    #[serde(rename = "view.set_light")]
    SetLight,
    #[serde(rename = "view.set_tiles")]
    SetTiles,
    #[serde(rename = "view.set_pom")]
    SetPom,
    #[serde(rename = "view.set_camera")]
    SetCamera,
    #[serde(rename = "view.capture_a")]
    CaptureA,
    #[serde(rename = "view.capture_b")]
    CaptureB,
    #[serde(rename = "view.set_split")]
    SetSplit,
    #[serde(rename = "export.set_settings")]
    SetExportSettings,
    #[serde(rename = "export.request")]
    RequestExport,
    #[serde(rename = "export.cancel")]
    CancelExport,
    #[serde(rename = "export.save_preset")]
    SaveExportPreset,
    #[serde(rename = "batch.start")]
    BatchStart,
    #[serde(rename = "batch.cancel")]
    BatchCancel,
    #[serde(rename = "batch.retry_failed")]
    BatchRetryFailed,
    #[serde(rename = "image.save_png")]
    SavePng,
    #[serde(rename = "project.open")]
    OpenProject,
    #[serde(rename = "project.save")]
    SaveProject,
    #[serde(rename = "project.save_as")]
    SaveProjectAs,
}

impl CommandId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConfigureMaterial => "material.configure",
            Self::InspectSeams => "view.inspect_seams",
            Self::NewFromImage => "document.new_from_image",
            Self::SetParameter => "node.set_parameter",
            Self::BeginGesture => "edit.begin_gesture",
            Self::CommitGesture => "edit.commit_gesture",
            Self::CancelGesture => "edit.cancel_gesture",
            Self::BeginStroke => "mask.begin_stroke",
            Self::AppendStroke => "mask.append_stroke",
            Self::CommitStroke => "mask.commit_stroke",
            Self::CancelStroke => "mask.cancel_stroke",
            Self::Undo => "edit.undo",
            Self::Redo => "edit.redo",
            Self::SelectOutput => "view.select_output",
            Self::SelectMap => "material.select_map",
            Self::SetVisible => "material.set_visible",
            Self::SetLocked => "material.set_locked",
            Self::SetMode => "view.set_mode",
            Self::SetMesh => "view.set_mesh",
            Self::SetLight => "view.set_light",
            Self::SetTiles => "view.set_tiles",
            Self::SetPom => "view.set_pom",
            Self::SetCamera => "view.set_camera",
            Self::CaptureA => "view.capture_a",
            Self::CaptureB => "view.capture_b",
            Self::SetSplit => "view.set_split",
            Self::SetExportSettings => "export.set_settings",
            Self::RequestExport => "export.request",
            Self::CancelExport => "export.cancel",
            Self::SaveExportPreset => "export.save_preset",
            Self::BatchStart => "batch.start",
            Self::BatchCancel => "batch.cancel",
            Self::BatchRetryFailed => "batch.retry_failed",
            Self::SavePng => "image.save_png",
            Self::OpenProject => "project.open",
            Self::SaveProject => "project.save",
            Self::SaveProjectAs => "project.save_as",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportToken(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GestureId(pub u64);

/// Uninterpreted frontend token; never resolve it as an OS path inside core.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueLocator(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterAddress {
    pub node: NodeId,
    pub parameter: ParameterId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    UInt(u32),
    Bool(bool),
    FloatVector(Vec<f32>),
    Curve(Vec<[f32; 2]>),
    Color([f32; 3]),
    Boundary(Boundary),
    Filter(NormalFilter),
    Convention(NormalConvention),
    MaskMode(MaskMode),
    SeamlessMethod(SeamlessMethod),
}

impl ParameterValue {
    /// Legacy scalar accessor; typed callers should match the value variant.
    ///
    /// # Panics
    /// Panics for a non-float value.
    pub const fn as_f32(&self) -> f32 {
        match self {
            Self::Float(value) => *value,
            _ => panic!("parameter value is not a float"),
        }
    }

    pub(crate) fn owned_bytes(&self) -> usize {
        match self {
            Self::FloatVector(values) => values.len() * size_of::<f32>(),
            Self::Curve(values) => values.len() * size_of::<[f32; 2]>(),
            _ => 0,
        }
    }
}

/// The tagged ID and payload are inseparable, so mismatched/unknown payloads
/// fail serde routing. Contextual validation belongs to Session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "id", content = "payload", deny_unknown_fields)]
pub enum CommandRequest {
    #[serde(rename = "project.save")]
    SaveProject { destination: OpaqueLocator },
    #[serde(rename = "project.save_as")]
    SaveProjectAs { destination: OpaqueLocator },
    #[serde(rename = "material.configure")]
    ConfigureMaterial { seamless: bool, painted_mask: bool },
    #[serde(rename = "view.inspect_seams")]
    InspectSeams { enabled: bool },
    #[serde(rename = "project.open")]
    OpenProject {
        token: crate::ProjectOpenToken,
        source: OpaqueLocator,
    },
    #[serde(rename = "document.new_from_image")]
    NewFromImage {
        token: ImportToken,
        source: OpaqueLocator,
    },
    #[serde(rename = "node.set_parameter")]
    SetParameter {
        address: ParameterAddress,
        value: ParameterValue,
        gesture: Option<GestureId>,
    },
    #[serde(rename = "edit.begin_gesture")]
    BeginGesture {
        gesture: GestureId,
        address: ParameterAddress,
    },
    #[serde(rename = "edit.commit_gesture")]
    CommitGesture { gesture: GestureId },
    #[serde(rename = "edit.cancel_gesture")]
    CancelGesture { gesture: GestureId },
    #[serde(rename = "mask.begin_stroke")]
    BeginStroke {
        node: NodeId,
        stroke_id: StrokeId,
        brush: Brush,
        sample: BrushSample,
    },
    #[serde(rename = "mask.append_stroke")]
    AppendStroke {
        stroke_id: StrokeId,
        samples: Vec<BrushSample>,
    },
    #[serde(rename = "mask.commit_stroke")]
    CommitStroke { stroke_id: StrokeId },
    #[serde(rename = "mask.cancel_stroke")]
    CancelStroke { stroke_id: StrokeId },
    #[serde(rename = "edit.undo")]
    Undo,
    #[serde(rename = "edit.redo")]
    Redo,
    #[serde(rename = "view.select_output")]
    SelectOutput { output: NodeId },
    #[serde(rename = "material.select_map")]
    SelectMap { map: MapKind },
    #[serde(rename = "material.set_visible")]
    SetVisible { map: MapKind, visible: bool },
    #[serde(rename = "material.set_locked")]
    SetLocked { map: MapKind, locked: bool },
    #[serde(rename = "view.set_mode")]
    SetMode { mode: ViewMode },
    #[serde(rename = "view.set_mesh")]
    SetMesh { mesh: PreviewMesh },
    #[serde(rename = "view.set_light")]
    SetLight { light: Light },
    #[serde(rename = "view.set_tiles")]
    SetTiles { tiles: Tiles },
    #[serde(rename = "view.set_pom")]
    SetPom { pom: Pom },
    #[serde(rename = "view.set_camera")]
    SetCamera { camera: Camera },
    #[serde(rename = "view.capture_a")]
    CaptureA {},
    #[serde(rename = "view.capture_b")]
    CaptureB {},
    #[serde(rename = "view.set_split")]
    SetSplit { fraction: f32 },
    #[serde(rename = "export.set_settings")]
    SetExportSettings { settings: ExportPlan },
    #[serde(rename = "export.request")]
    RequestExport {},
    #[serde(rename = "export.cancel")]
    CancelExport { job_id: u64 },
    #[serde(rename = "export.save_preset")]
    SaveExportPreset {
        destination: OpaqueLocator,
        preset: ExportPreset,
    },
    #[serde(rename = "batch.start")]
    BatchStart {},
    #[serde(rename = "batch.cancel")]
    BatchCancel {},
    #[serde(rename = "batch.retry_failed")]
    BatchRetryFailed {},
    #[serde(rename = "image.save_png")]
    SavePng {
        destination: OpaqueLocator,
        map: MapKind,
    },
}

impl CommandRequest {
    pub const fn id(&self) -> CommandId {
        match self {
            Self::OpenProject { .. } => CommandId::OpenProject,
            Self::SaveProject { .. } => CommandId::SaveProject,
            Self::SaveProjectAs { .. } => CommandId::SaveProjectAs,
            Self::ConfigureMaterial { .. } => CommandId::ConfigureMaterial,
            Self::InspectSeams { .. } => CommandId::InspectSeams,
            Self::NewFromImage { .. } => CommandId::NewFromImage,
            Self::SetParameter { .. } => CommandId::SetParameter,
            Self::BeginGesture { .. } => CommandId::BeginGesture,
            Self::CommitGesture { .. } => CommandId::CommitGesture,
            Self::CancelGesture { .. } => CommandId::CancelGesture,
            Self::BeginStroke { .. } => CommandId::BeginStroke,
            Self::AppendStroke { .. } => CommandId::AppendStroke,
            Self::CommitStroke { .. } => CommandId::CommitStroke,
            Self::CancelStroke { .. } => CommandId::CancelStroke,
            Self::Undo => CommandId::Undo,
            Self::Redo => CommandId::Redo,
            Self::SelectOutput { .. } => CommandId::SelectOutput,
            Self::SelectMap { .. } => CommandId::SelectMap,
            Self::SetVisible { .. } => CommandId::SetVisible,
            Self::SetLocked { .. } => CommandId::SetLocked,
            Self::SetMode { .. } => CommandId::SetMode,
            Self::SetMesh { .. } => CommandId::SetMesh,
            Self::SetLight { .. } => CommandId::SetLight,
            Self::SetTiles { .. } => CommandId::SetTiles,
            Self::SetPom { .. } => CommandId::SetPom,
            Self::SetCamera { .. } => CommandId::SetCamera,
            Self::CaptureA { .. } => CommandId::CaptureA,
            Self::CaptureB { .. } => CommandId::CaptureB,
            Self::SetSplit { .. } => CommandId::SetSplit,
            Self::SetExportSettings { .. } => CommandId::SetExportSettings,
            Self::RequestExport { .. } => CommandId::RequestExport,
            Self::CancelExport { .. } => CommandId::CancelExport,
            Self::SaveExportPreset { .. } => CommandId::SaveExportPreset,
            Self::BatchStart { .. } => CommandId::BatchStart,
            Self::BatchCancel { .. } => CommandId::BatchCancel,
            Self::BatchRetryFailed { .. } => CommandId::BatchRetryFailed,
            Self::SavePng { .. } => CommandId::SavePng,
        }
    }
}

/// Typed payload schemas, including parameter ranges, lengths and choices.
/// Coupled constraints such as the levels interval are validated by node DTOs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum PayloadSchema {
    MaterialOptions,
    Toggle,
    Import,
    ProjectOpen,
    ProjectSave,
    Parameter {
        parameters: &'static [MaterialParameterDescriptor],
    },
    BeginGesture,
    Gesture,
    BeginStroke,
    AppendStroke,
    Stroke,
    Empty,
    Output,
    Map,
    Export,
    MapVisible,
    MapLocked,
    ViewMode,
    ViewMesh,
    ViewLight,
    ViewTiles,
    ViewPom,
    ViewCamera,
    ViewSplit,
    ExportSettings,
    ExportJob,
    ExportPreset,
}

/// Static prerequisites, not a claim that a command is currently enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityRequirement {
    Always,
    Document,
    NoActiveGesture,
    ActiveGesture,
    PaintMask,
    ActiveStroke,
    UndoHistory,
    RedoHistory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UndoPolicy {
    NewDocument,
    ParameterDiff,
    Gesture,
    History,
    None,
}

/// Static metadata is serialized for discovery, never deserialized into a registry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CommandDescriptor {
    pub id: CommandId,
    pub label: &'static str,
    pub payload: PayloadSchema,
    pub availability: AvailabilityRequirement,
    pub undo_policy: UndoPolicy,
}

pub fn command_descriptors() -> &'static [CommandDescriptor] {
    use AvailabilityRequirement as A;
    use CommandId as C;
    use PayloadSchema as P;
    use UndoPolicy as U;
    static DESCRIPTORS: [CommandDescriptor; 37] = [
        CommandDescriptor {
            id: C::ConfigureMaterial,
            label: "material.configure",
            payload: P::MaterialOptions,
            availability: A::Document,
            undo_policy: U::ParameterDiff,
        },
        CommandDescriptor {
            id: C::InspectSeams,
            label: "view.inspect_seams",
            payload: P::Toggle,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::NewFromImage,
            label: "New from image (clears history after successful load)",
            payload: P::Import,
            availability: A::Always,
            undo_policy: U::NewDocument,
        },
        CommandDescriptor {
            id: C::SetParameter,
            label: "Set parameter",
            payload: P::Parameter {
                parameters: &MATERIAL_PARAMETER_DESCRIPTORS,
            },
            availability: A::Document,
            undo_policy: U::ParameterDiff,
        },
        CommandDescriptor {
            id: C::BeginGesture,
            label: "Begin parameter gesture",
            payload: P::BeginGesture,
            availability: A::NoActiveGesture,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::CommitGesture,
            label: "Commit parameter gesture",
            payload: P::Gesture,
            availability: A::ActiveGesture,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::CancelGesture,
            label: "Cancel parameter gesture",
            payload: P::Gesture,
            availability: A::ActiveGesture,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::Undo,
            label: "Undo",
            payload: P::Empty,
            availability: A::UndoHistory,
            undo_policy: U::History,
        },
        CommandDescriptor {
            id: C::BeginStroke,
            label: "Begin UV brush stroke",
            payload: P::BeginStroke,
            availability: A::PaintMask,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::AppendStroke,
            label: "Append UV brush samples (100000 point limit; finish and restart)",
            payload: P::AppendStroke,
            availability: A::ActiveStroke,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::CommitStroke,
            label: "Commit brush stroke (also on focus loss)",
            payload: P::Stroke,
            availability: A::ActiveStroke,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::CancelStroke,
            label: "Cancel brush stroke (Escape)",
            payload: P::Stroke,
            availability: A::ActiveStroke,
            undo_policy: U::Gesture,
        },
        CommandDescriptor {
            id: C::Redo,
            label: "Redo",
            payload: P::Empty,
            availability: A::RedoHistory,
            undo_policy: U::History,
        },
        CommandDescriptor {
            id: C::SelectOutput,
            label: "Select output",
            payload: P::Output,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SelectMap,
            label: "Select material map",
            payload: P::Map,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SavePng,
            label: "Save PNG",
            payload: P::Export,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetVisible,
            label: "material.set_visible",
            payload: P::MapVisible,
            availability: A::Document,
            undo_policy: U::ParameterDiff,
        },
        CommandDescriptor {
            id: C::SetLocked,
            label: "material.set_locked",
            payload: P::MapLocked,
            availability: A::Document,
            undo_policy: U::ParameterDiff,
        },
        CommandDescriptor {
            id: C::SetMode,
            label: "view.set_mode",
            payload: P::ViewMode,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetMesh,
            label: "view.set_mesh",
            payload: P::ViewMesh,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetLight,
            label: "view.set_light",
            payload: P::ViewLight,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetTiles,
            label: "view.set_tiles",
            payload: P::ViewTiles,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetPom,
            label: "view.set_pom",
            payload: P::ViewPom,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetCamera,
            label: "view.set_camera",
            payload: P::ViewCamera,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::CaptureA,
            label: "view.capture_a",
            payload: P::Empty,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::CaptureB,
            label: "view.capture_b",
            payload: P::Empty,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetSplit,
            label: "view.set_split",
            payload: P::ViewSplit,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SetExportSettings,
            label: "Set export settings",
            payload: P::ExportSettings,
            availability: A::Document,
            undo_policy: U::ParameterDiff,
        },
        CommandDescriptor {
            id: C::RequestExport,
            label: "Request export",
            payload: P::Empty,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::BatchStart,
            label: "Start batch",
            payload: P::Empty,
            availability: A::Always,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::BatchCancel,
            label: "Cancel batch",
            payload: P::Empty,
            availability: A::Always,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::BatchRetryFailed,
            label: "Retry failed batch items",
            payload: P::Empty,
            availability: A::Always,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::OpenProject,
            label: "Open project",
            payload: P::ProjectOpen,
            availability: A::Always,
            undo_policy: U::NewDocument,
        },
        CommandDescriptor {
            id: C::CancelExport,
            label: "Cancel export",
            payload: P::ExportJob,
            availability: A::Always,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SaveExportPreset,
            label: "Save export preset",
            payload: P::ExportPreset,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SaveProject,
            label: "Save project",
            payload: P::ProjectSave,
            availability: A::Document,
            undo_policy: U::None,
        },
        CommandDescriptor {
            id: C::SaveProjectAs,
            label: "Save project as",
            payload: P::ProjectSave,
            availability: A::Document,
            undo_policy: U::None,
        },
    ];
    &DESCRIPTORS
}

/// Native work requests. Material exports own their graph/source recipe; the
/// consumer never reads the live Session to reconstruct an accepted job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    SaveProject {
        destination: OpaqueLocator,
        save_as: bool,
    },
    SaveExportPreset {
        destination: OpaqueLocator,
        preset: ExportPreset,
    },
    CancelExport {
        job_id: u64,
    },
    InspectSeams {
        enabled: bool,
    },
    OpenProject {
        token: crate::ProjectOpenToken,
        source: OpaqueLocator,
    },
    Export {
        plan: ExportPlan,
        capture: Box<ExportCapture>,
    },
    LoadImage {
        token: ImportToken,
        source: OpaqueLocator,
    },
    SavePng {
        destination: OpaqueLocator,
        map: MapKind,
        output: NodeId,
        generation: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterDiff {
    pub address: ParameterAddress,
    pub before: ParameterValue,
    pub after: ParameterValue,
}

impl ParameterDiff {
    pub(crate) fn byte_count(&self) -> usize {
        size_of::<Self>() + self.before.owned_bytes() + self.after.owned_bytes()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryState {
    pub undo_entries: usize,
    pub redo_entries: usize,
    /// Retained diff payload bytes in both stacks and the active gesture.
    /// Excludes allocator capacity and the fixed Session/Vec bookkeeping.
    pub byte_count: usize,
    pub active_gesture: Option<GestureId>,
    pub active_stroke: Option<StrokeId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandAvailability {
    pub id: CommandId,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommandOutcome {
    pub effects: Vec<Effect>,
    pub snapshot: Option<EvaluationSnapshot>,
    pub history: HistoryState,
    pub availability: Vec<CommandAvailability>,
}

/// Transactional dispatcher errors. JSON unknown IDs/payloads are reported by serde.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandError {
    Core(CoreError),
    Batch(crate::BatchError),
    Disabled(CommandId),
    NoDocument,
    GestureAlreadyActive,
    StaleGesture(GestureId),
    StaleStroke(StrokeId),
    StaleImport(ImportToken),
    GenerationOverflow,
}

impl From<CoreError> for CommandError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(error) => error.fmt(f),
            Self::Batch(error) => write!(f, "batch error: {error:?}"),
            Self::Disabled(id) => write!(f, "command {} is disabled", id.as_str()),
            Self::NoDocument => f.write_str("no document is loaded"),
            Self::GestureAlreadyActive => f.write_str("a gesture is already active"),
            Self::StaleGesture(id) => write!(f, "stale gesture {}", id.0),
            Self::StaleStroke(id) => write!(f, "stale stroke {}", id.0),
            Self::StaleImport(token) => write!(f, "stale import {}", token.0),
            Self::GenerationOverflow => f.write_str("generation counter exhausted"),
        }
    }
}

impl std::error::Error for CommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}
