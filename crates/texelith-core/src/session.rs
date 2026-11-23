//! Synchronous document commands. Frontends execute effects and schedule snapshots.

use crate::stroke::StrokeBuilder;
use crate::{
    Brush, BrushSample, BrushStroke, CaptureIdentity, CommandAvailability, CommandError, CommandId,
    CommandOutcome, CommandRequest, CoreError, Effect, EvaluationSnapshot, ExportCapture,
    ExportPlan, GestureId, Graph, HistoryState, ImportToken, MapKind, MapView, MaterialNodeParams,
    Node, NodeId, NodeKind, NodeParams, ParameterAddress, ParameterDiff, ParameterValue,
    SourceDescriptor, StrokeId, ViewState, command_descriptors, stroke_digest,
    validate_export_for_graph,
};

#[derive(Clone, Debug)]
struct StrokeDiff {
    node: NodeId,
    index: usize,
    stroke: BrushStroke,
    before_digest: [u8; 32],
    after_digest: [u8; 32],
}

#[derive(Clone, Debug)]
enum HistoryDiff {
    MaterialOptions {
        before: (bool, bool),
        after: (bool, bool),
    },
    ExportSettings {
        before: Option<ExportPlan>,
        after: Option<ExportPlan>,
    },
    Parameter(ParameterDiff),
    Stroke(StrokeDiff),
    Map {
        map: MapKind,
        before: MapView,
        after: MapView,
    },
}

impl From<ParameterDiff> for HistoryDiff {
    fn from(diff: ParameterDiff) -> Self {
        Self::Parameter(diff)
    }
}

impl HistoryDiff {
    fn byte_count(&self) -> usize {
        size_of::<Self>()
            + match self {
                Self::MaterialOptions { .. } => 0,
                Self::Parameter(diff) => diff.byte_count() - size_of::<ParameterDiff>(),
                Self::Stroke(diff) => diff.stroke.points.len() * size_of::<BrushSample>(),
                Self::Map { .. } => 0,
                Self::ExportSettings { before, after } => before
                    .iter()
                    .chain(after)
                    .map(ExportPlan::owned_bytes)
                    .sum(),
            }
    }
}

#[derive(Debug)]
struct StrokeGesture {
    node: NodeId,
    index: usize,
    before_digest: [u8; 32],
    builder: StrokeBuilder,
}

#[derive(Debug)]
struct Gesture {
    id: GestureId,
    diff: ParameterDiff,
}

impl Gesture {
    fn changed(&self) -> bool {
        self.diff.before != self.diff.after
    }
}

/// Authoritative document state, with no native I/O, pixels, or execution resources.
///
/// Frontends supply unique request/gesture tokens and dispatch serially. A failed
/// request leaves the document, pending import, gesture, and history untouched.
#[derive(Debug, Default)]
pub struct Session {
    batch: Option<crate::BatchJob>,
    last_batch_id: Option<u64>,
    document: Option<EvaluationSnapshot>,
    view: ViewState,
    export_settings: Option<ExportPlan>,
    export_active: bool,
    capture_a_snapshot: Option<EvaluationSnapshot>,
    capture_b_snapshot: Option<EvaluationSnapshot>,
    request_id: u64,
    document_revision: u64,
    saved_revision: u64,
    view_revision: u64,
    saved_view_revision: u64,
    pending_import: Option<ImportToken>,
    pending_project_open: Option<crate::ProjectOpenToken>,
    project_assets: Vec<crate::AssetRef>,
    undo: Vec<HistoryDiff>,
    redo: Vec<HistoryDiff>,
    // Keep history discovery independent of the number of retained edits.
    undo_bytes: usize,
    redo_bytes: usize,
    gesture: Option<Gesture>,
    stroke: Option<StrokeGesture>,
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initializes from an owned, already validated graph and runtime generation.
    /// This is not a project-loading or graph-replacement command.
    pub fn from_snapshot(snapshot: EvaluationSnapshot) -> Self {
        Self {
            document: Some(snapshot),
            ..Self::default()
        }
    }

    pub fn batch(&self) -> Option<&crate::BatchJob> {
        self.batch.as_ref()
    }

    /// Complete worker phases without changing the visible document/history.
    pub fn batch_transition(
        &mut self,
        work: crate::BatchWorkId,
        state: crate::BatchState,
    ) -> Result<(), crate::BatchError> {
        self.batch
            .as_mut()
            .ok_or(crate::BatchError::StaleCompletion { received: work })?
            .transition(work, state)
    }

    pub fn batch_fail(
        &mut self,
        work: crate::BatchWorkId,
        error: String,
    ) -> Result<(), crate::BatchError> {
        self.batch
            .as_mut()
            .ok_or(crate::BatchError::StaleCompletion { received: work })?
            .fail(work, error)
    }

    pub fn retire_cancelled_batch(
        &mut self,
        work: crate::BatchWorkId,
    ) -> Result<(), crate::BatchError> {
        self.batch
            .as_mut()
            .ok_or(crate::BatchError::StaleCompletion { received: work })?
            .retire_cancelled(work)
    }

    /// Admit an already captured, preflighted job. IDs increase for the Session's
    /// lifetime, and submitted work must retire before its owner can be replaced.
    pub fn set_batch(&mut self, job: crate::BatchJob) -> Result<(), crate::BatchError> {
        if self
            .batch
            .as_ref()
            .is_some_and(|current| current.pending_work().is_some())
        {
            return Err(crate::BatchError::ActiveItem);
        }
        if self.last_batch_id.is_some_and(|id| job.id() <= id) {
            return Err(crate::BatchError::DuplicateJobId { id: job.id() });
        }
        self.last_batch_id = Some(job.id());
        self.batch = Some(job);
        Ok(())
    }

    pub fn export_settings(&self) -> Option<&ExportPlan> {
        self.export_settings.as_ref()
    }

    /// Report worker ownership through completion or cancellation cleanup.
    /// This runtime state does not change the document or its history.
    pub fn set_export_active(&mut self, active: bool) {
        self.export_active = active;
    }

    pub fn view_state(&self) -> &ViewState {
        &self.view
    }
    pub fn captured_a(&self) -> Option<&EvaluationSnapshot> {
        self.capture_a_snapshot.as_ref()
    }
    pub fn captured_b(&self) -> Option<&EvaluationSnapshot> {
        self.capture_b_snapshot.as_ref()
    }
    pub fn request_id(&self) -> u64 {
        self.request_id
    }
    pub fn document_revision(&self) -> u64 {
        self.document_revision
    }
    /// A frontend accepted new linked-asset identities after project validation.
    pub fn mark_project_modified(&mut self) {
        self.document_revision = self.document_revision.saturating_add(1);
    }
    /// Document edits require saving; preview-only changes persist on save but
    /// do not mark the document dirty.
    pub fn is_dirty(&self) -> bool {
        self.document_revision != self.saved_revision
    }
    /// Includes persisted preview changes without changing graph dirty semantics.
    pub fn has_unsaved_project(&self) -> bool {
        self.is_dirty() || self.view_revision != self.saved_view_revision
    }

    /// Legacy graph/export save completion. Persisted preview changes require
    /// `mark_project_saved` with the view revision captured by the save worker.
    pub fn mark_saved(&mut self, revision: u64) -> bool {
        if self.view_revision != self.saved_view_revision {
            return false;
        }
        self.mark_project_saved(revision, self.view_revision)
    }

    /// Persisted view revision, independent of graph/cache generations.
    pub fn view_revision(&self) -> u64 {
        self.view_revision
    }

    /// Complete a save of both graph/export state and persisted view state.
    pub fn mark_project_saved(&mut self, revision: u64, view_revision: u64) -> bool {
        if self.document.is_some()
            && revision == self.document_revision
            && view_revision == self.view_revision
        {
            self.saved_revision = revision;
            self.saved_view_revision = view_revision;
            true
        } else {
            false
        }
    }

    pub fn snapshot(&self) -> Option<EvaluationSnapshot> {
        self.document.clone()
    }

    pub fn history(&self) -> HistoryState {
        HistoryState {
            undo_entries: self.undo.len(),
            redo_entries: self.redo.len(),
            byte_count: self.undo_bytes
                + self.redo_bytes
                + self
                    .gesture
                    .as_ref()
                    .map_or(0, |gesture| gesture.diff.byte_count())
                + self.stroke.as_ref().map_or(0, |stroke| {
                    size_of::<StrokeGesture>() - size_of::<StrokeBuilder>()
                        + stroke.builder.byte_count()
                }),
            active_gesture: self.gesture.as_ref().map(|gesture| gesture.id),
            active_stroke: self.stroke.as_ref().map(|stroke| stroke.builder.stroke.id),
        }
    }

    /// Runtime prerequisites; payload validation and generation checks still run
    /// during dispatch. Redo considers the branch after an implicit gesture commit.
    pub fn availability(&self) -> Vec<CommandAvailability> {
        command_descriptors()
            .iter()
            .map(|descriptor| CommandAvailability {
                id: descriptor.id,
                enabled: self.enabled(descriptor.id),
            })
            .collect()
    }

    fn enabled(&self, id: CommandId) -> bool {
        let changed = self.gesture.as_ref().is_some_and(Gesture::changed) || self.stroke.is_some();
        match id {
            CommandId::BatchStart => self.batch.as_ref().is_some_and(crate::BatchJob::can_start),
            CommandId::BatchCancel => self.batch.as_ref().is_some_and(crate::BatchJob::can_cancel),
            CommandId::BatchRetryFailed => self.batch.as_ref().is_some_and(crate::BatchJob::can_retry_failed) && self.last_batch_id != Some(u64::MAX),
            CommandId::NewFromImage | CommandId::OpenProject => true,
            CommandId::CancelExport => self.export_active,
            CommandId::SetExportSettings | CommandId::SaveExportPreset | CommandId::SaveProject | CommandId::SaveProjectAs => self.document.is_some(),
            CommandId::RequestExport => self.document.is_some() && self.export_settings.is_some(),
            CommandId::SetVisible | CommandId::SetLocked | CommandId::SetMode | CommandId::SetMesh | CommandId::SetLight | CommandId::SetTiles | CommandId::SetPom | CommandId::SetCamera | CommandId::CaptureA | CommandId::CaptureB | CommandId::SetSplit => self.document.is_some(),
            CommandId::SetParameter => self.document.is_some() && self.stroke.is_none(),
            CommandId::SelectOutput => self.document.is_some(),
            CommandId::ConfigureMaterial | CommandId::InspectSeams | CommandId::SelectMap => self
                .document
                .as_ref()
                .is_some_and(|snapshot| snapshot.graph.is_material()),
            CommandId::SavePng => self.document.as_ref().is_some_and(|snapshot| {
                if snapshot.graph.is_material() { return true; }
                snapshot
                    .graph
                    .nodes()
                    .iter()
                    .any(|node| node.params.kind() != NodeKind::Source)
            }),
            CommandId::BeginGesture => self.document.is_some() && self.gesture.is_none() && self.stroke.is_none(),
            CommandId::CommitGesture | CommandId::CancelGesture => self.gesture.is_some(),
            CommandId::BeginStroke => self.gesture.is_none() && self.stroke.is_none() && self.document.as_ref().is_some_and(|snapshot| snapshot.graph.material_nodes().iter().any(|node| matches!(&node.params, MaterialNodeParams::PaintMask { stroke_digest, strokes, .. } if !strokes.is_empty() || *stroke_digest == [0; 32]))),
            CommandId::AppendStroke | CommandId::CommitStroke | CommandId::CancelStroke => self.stroke.is_some(),
            CommandId::Undo => changed || !self.undo.is_empty(),
            CommandId::Redo => !changed && !self.redo.is_empty(),
        }
    }

    pub fn dispatch(&mut self, request: CommandRequest) -> Result<CommandOutcome, CommandError> {
        let persisted_view_before = crate::SavedView::from(self.view.clone());
        let mut effects = Vec::new();
        let previous_generation = self.document.as_ref().map(|d| d.generation);
        let mut preview_changed = false;
        let mut document_changed = false;
        let changed = match request {
            CommandRequest::SaveProject { ref destination }
            | CommandRequest::SaveProjectAs { ref destination } => {
                self.document.as_ref().ok_or(CommandError::NoDocument)?;
                effects.push(Effect::SaveProject {
                    destination: destination.clone(),
                    save_as: matches!(request, CommandRequest::SaveProjectAs { .. }),
                });
                false
            }
            CommandRequest::OpenProject { token, source } => {
                self.begin_project_open(token)?;
                effects.push(Effect::OpenProject { token, source });
                false
            }
            CommandRequest::NewFromImage { token, source } => {
                self.pending_import = Some(token);
                self.pending_project_open = None;
                effects.push(Effect::LoadImage { token, source });
                false
            }
            CommandRequest::SetParameter {
                address,
                value,
                gesture,
            } => {
                if self.stroke.is_some() {
                    return Err(CommandError::GestureAlreadyActive);
                }
                self.ensure_unlocked(address.node)?;
                match (&self.gesture, gesture) {
                    (Some(active), Some(id)) if active.id != id => {
                        return Err(CommandError::StaleGesture(id));
                    }
                    (None, Some(id)) => return Err(CommandError::StaleGesture(id)),
                    (Some(active), Some(_)) if active.diff.address != address => {
                        return Err(CommandError::GestureAlreadyActive);
                    }
                    (Some(_), None) => return Err(CommandError::GestureAlreadyActive),
                    _ => {}
                }
                let before = self.parameter(address)?;
                let changed = self.change_parameter(address, value.clone())?;
                if changed {
                    if let Some(active) = &mut self.gesture {
                        active.diff.after = value;
                    } else {
                        self.commit_diff(ParameterDiff {
                            address,
                            before,
                            after: value,
                        });
                    }
                }
                changed
            }
            CommandRequest::BeginGesture { gesture, address } => {
                if self.gesture.is_some() || self.stroke.is_some() {
                    return Err(CommandError::GestureAlreadyActive);
                }
                self.ensure_unlocked(address.node)?;
                let value = self.parameter(address)?;
                self.gesture = Some(Gesture {
                    id: gesture,
                    diff: ParameterDiff {
                        address,
                        before: value.clone(),
                        after: value,
                    },
                });
                false
            }
            CommandRequest::CommitGesture { gesture } => {
                self.match_gesture(gesture)?;
                self.commit_gesture();
                false
            }
            CommandRequest::CancelGesture { gesture } => {
                let diff = &self.match_gesture(gesture)?.diff;
                let changed = self.change_parameter(diff.address, diff.before.clone())?;
                self.gesture = None;
                changed
            }
            CommandRequest::BeginStroke {
                node,
                stroke_id,
                brush,
                sample,
            } => self.begin_stroke(node, stroke_id, brush, sample)?,
            CommandRequest::AppendStroke { stroke_id, samples } => {
                self.append_stroke(stroke_id, &samples)?
            }
            CommandRequest::CommitStroke { stroke_id } => {
                self.match_stroke(stroke_id)?;
                self.commit_stroke();
                false
            }
            CommandRequest::CancelStroke { stroke_id } => {
                let diff = self.stroke_diff(self.match_stroke(stroke_id)?);
                let changed = self.change_diff(&HistoryDiff::Stroke(diff), false)?;
                self.stroke = None;
                changed
            }
            CommandRequest::Undo => {
                let (result, is_map, is_export) = self.replay(false)?;
                document_changed = (is_map || is_export) && result;
                preview_changed = is_map && result;
                result && !is_map && !is_export
            }
            CommandRequest::Redo => {
                let (result, is_map, is_export) = self.replay(true)?;
                document_changed = (is_map || is_export) && result;
                preview_changed = is_map && result;
                result && !is_map && !is_export
            }
            CommandRequest::SelectOutput { output } => {
                let document = self.document.as_mut().ok_or(CommandError::NoDocument)?;
                let changed = document.graph.selected_output() != output;
                document.graph.select_output(output)?;
                preview_changed = changed;
                changed
            }
            CommandRequest::InspectSeams { enabled } => {
                if !self
                    .document
                    .as_ref()
                    .is_some_and(|d| d.graph.is_material())
                {
                    return Err(CommandError::Disabled(CommandId::InspectSeams));
                }
                effects.push(Effect::InspectSeams { enabled });
                preview_changed = true;
                false
            }
            CommandRequest::ConfigureMaterial {
                seamless,
                painted_mask,
            } => {
                if self.gesture.is_some() || self.stroke.is_some() {
                    return Err(CommandError::GestureAlreadyActive);
                }
                let before = self
                    .document
                    .as_ref()
                    .ok_or(CommandError::NoDocument)?
                    .graph
                    .clone();
                let after = crate::configure_material(&before, seamless, painted_mask)?;
                if before != after {
                    let generation = self.next_generation()?;
                    self.document = Some(EvaluationSnapshot {
                        generation,
                        graph: after.clone(),
                    });
                    self.commit_diff(HistoryDiff::MaterialOptions {
                        before: crate::material_options(&before),
                        after: (seamless, painted_mask),
                    });
                    true
                } else {
                    false
                }
            }
            CommandRequest::SelectMap { map } => {
                let document = self.document.as_mut().ok_or(CommandError::NoDocument)?;
                let changed = document.graph.select_map(map)?;
                preview_changed = true;
                changed
            }
            CommandRequest::SetVisible { map, visible } => {
                let mut state = self.map_view(map)?;
                state.visible = visible;
                document_changed = self.set_map(map, state)?;
                preview_changed = document_changed;
                false
            }
            CommandRequest::SetLocked { map, locked } => {
                let mut state = self.map_view(map)?;
                state.locked = locked;
                document_changed = self.set_map(map, state)?;
                preview_changed = document_changed;
                false
            }
            CommandRequest::SetMode { mode } => {
                self.require_document()?;
                preview_changed = self.view.mode != mode;
                self.view.mode = mode;
                false
            }
            CommandRequest::SetMesh { mesh } => {
                self.require_document()?;
                preview_changed = self.view.mesh != mesh;
                self.view.mesh = mesh;
                false
            }
            CommandRequest::SetLight { light } => {
                self.require_document()?;
                if !light.intensity.is_finite()
                    || light.intensity < 0.0
                    || !light.direction.iter().all(|v| v.is_finite())
                    || light.direction.iter().all(|v| *v == 0.0)
                {
                    return Err(CoreError::InvalidMaterial("invalid light").into());
                }
                preview_changed = self.view.light != light;
                self.view.light = light;
                false
            }
            CommandRequest::SetTiles { tiles } => {
                self.require_document()?;
                if !(1..=3).contains(&tiles.count) {
                    return Err(CoreError::InvalidMaterial("invalid tile count").into());
                }
                preview_changed = self.view.tiles != tiles;
                self.view.tiles = tiles;
                false
            }
            CommandRequest::SetPom { pom } => {
                self.require_document()?;
                if !pom.scale.is_finite() || pom.scale < 0.0 {
                    return Err(CoreError::InvalidMaterial("invalid POM scale").into());
                }
                preview_changed = self.view.pom != pom;
                self.view.pom = pom;
                false
            }
            CommandRequest::SetCamera { camera } => {
                self.require_document()?;
                if !camera.yaw.is_finite()
                    || !camera.pitch.is_finite()
                    || !camera.distance.is_finite()
                    || camera.distance <= 0.0
                    || !camera.target.iter().all(|v| v.is_finite())
                {
                    return Err(CoreError::InvalidMaterial("invalid camera").into());
                }
                preview_changed = self.view.camera != camera;
                self.view.camera = camera;
                false
            }
            CommandRequest::CaptureA {} => {
                let capture = self.capture()?;
                self.capture_a_snapshot = self.snapshot();
                self.view.capture_a = Some(capture);
                preview_changed = true;
                false
            }
            CommandRequest::CaptureB {} => {
                let capture = self.capture()?;
                self.capture_b_snapshot = self.snapshot();
                self.view.capture_b = Some(capture);
                preview_changed = true;
                false
            }
            CommandRequest::SetSplit { fraction } => {
                self.require_document()?;
                if !fraction.is_finite() {
                    return Err(CoreError::InvalidMaterial("invalid split").into());
                }
                let fraction = fraction.clamp(0.0, 1.0);
                preview_changed = self.view.split_fraction != fraction;
                self.view.split_fraction = fraction;
                false
            }
            CommandRequest::SetExportSettings { settings } => {
                self.require_document()?;
                if self.gesture.is_some() || self.stroke.is_some() {
                    return Err(CommandError::GestureAlreadyActive);
                }
                validate_export_for_graph(&settings, &self.require_document()?.graph)?;
                if self.export_settings.as_ref() != Some(&settings) {
                    let before = self.export_settings.replace(settings.clone());
                    self.commit_diff(HistoryDiff::ExportSettings {
                        before,
                        after: Some(settings),
                    });
                    document_changed = true;
                }
                false
            }
            CommandRequest::CancelExport { job_id } => {
                effects.push(Effect::CancelExport { job_id });
                false
            }
            CommandRequest::SaveExportPreset {
                destination,
                preset,
            } => {
                preset.bind(&self.require_document()?.graph, &[])?;
                effects.push(Effect::SaveExportPreset {
                    destination,
                    preset,
                });
                false
            }
            CommandRequest::RequestExport {} => {
                self.require_document()?;
                let plan = self
                    .export_settings
                    .clone()
                    .ok_or(CommandError::Disabled(CommandId::RequestExport))?;
                let capture =
                    ExportCapture::new(self.require_document()?, self.document_revision, &plan)?;
                effects.push(Effect::Export {
                    plan,
                    capture: Box::new(capture),
                });
                false
            }
            CommandRequest::BatchStart {} => {
                let job = self
                    .batch
                    .as_mut()
                    .ok_or(CommandError::Disabled(CommandId::BatchStart))?;
                let index = job
                    .items()
                    .iter()
                    .find(|item| item.state == crate::BatchState::Queued)
                    .map(|item| item.index)
                    .ok_or(CommandError::Disabled(CommandId::BatchStart))?;
                let work = job.work_id(index).map_err(CommandError::Batch)?;
                job.transition(work, crate::BatchState::Loading)
                    .map_err(CommandError::Batch)?;
                false
            }
            CommandRequest::BatchCancel {} => {
                self.batch
                    .as_mut()
                    .ok_or(CommandError::Disabled(CommandId::BatchCancel))?
                    .cancel()
                    .map_err(CommandError::Batch)?;
                false
            }
            CommandRequest::BatchRetryFailed {} => {
                let job = self
                    .batch
                    .as_ref()
                    .ok_or(CommandError::Disabled(CommandId::BatchRetryFailed))?;
                let id = job
                    .id()
                    .checked_add(1)
                    .ok_or(CommandError::GenerationOverflow)?;
                let retry = job.retry_failed(id).map_err(CommandError::Batch)?;
                self.set_batch(retry).map_err(CommandError::Batch)?;
                false
            }
            CommandRequest::SavePng { destination, map } => {
                let document = self.document.as_ref().ok_or(CommandError::NoDocument)?;
                // Prefer the selected output when it matches. Otherwise take the
                // last matching compute node on its ancestry, then a stable DAG fallback.
                let output = self.export_output(map)?;
                effects.push(Effect::SavePng {
                    destination,
                    map,
                    output,
                    generation: document.generation,
                });
                false
            }
        };
        if self.document.as_ref().map(|d| d.generation) != previous_generation {
            document_changed = true;
            preview_changed = true;
        }
        // Visibility/lock edits already participate in the document revision.
        if !document_changed && crate::SavedView::from(self.view.clone()) != persisted_view_before {
            self.view_revision = self.view_revision.saturating_add(1);
        }
        if document_changed {
            self.document_revision = self.document_revision.saturating_add(1);
        }
        if preview_changed {
            self.request_id = self.request_id.saturating_add(1);
        }
        Ok(self.outcome(effects, changed))
    }

    pub fn project_assets(&self) -> &[crate::AssetRef] {
        &self.project_assets
    }

    pub fn begin_project_open(
        &mut self,
        token: crate::ProjectOpenToken,
    ) -> Result<(), CommandError> {
        self.pending_project_open = Some(token);
        self.pending_import = None;
        Ok(())
    }

    pub fn fail_project_open(
        &mut self,
        token: crate::ProjectOpenToken,
    ) -> Result<(), CommandError> {
        self.match_project_open(token)?;
        self.pending_project_open = None;
        Ok(())
    }

    fn match_project_open(&self, token: crate::ProjectOpenToken) -> Result<(), CommandError> {
        if self.pending_project_open == Some(token) {
            Ok(())
        } else {
            Err(CoreError::InvalidMaterial("stale project open token").into())
        }
    }

    pub fn complete_project_open(
        &mut self,
        token: crate::ProjectOpenToken,
        project: crate::ValidatedProject,
    ) -> Result<CommandOutcome, CommandError> {
        self.match_project_open(token)?;
        let generation = self.next_generation()?;
        self.document = Some(EvaluationSnapshot {
            generation,
            graph: project.graph,
        });
        self.view = project.dto.view.into();
        self.export_settings = Some(project.dto.export);
        self.project_assets = project.dto.assets;
        self.capture_a_snapshot = None;
        self.capture_b_snapshot = None;
        self.document_revision = self.document_revision.saturating_add(1);
        self.saved_revision = self.document_revision;
        self.view_revision = self.view_revision.saturating_add(1);
        self.saved_view_revision = self.view_revision;
        self.request_id = self.request_id.saturating_add(1);
        self.pending_project_open = None;
        self.undo.clear();
        self.redo.clear();
        self.undo_bytes = 0;
        self.redo_bytes = 0;
        self.gesture = None;
        self.stroke = None;
        Ok(self.outcome(Vec::new(), true))
    }

    /// Installs the default material graph only after successful decode/validation.
    /// The caller installs matching decoded pixels before scheduling this snapshot.
    pub fn complete_import(
        &mut self,
        token: ImportToken,
        source: SourceDescriptor,
    ) -> Result<CommandOutcome, CommandError> {
        self.match_import(token)?;
        let nodes = [
            NodeKind::Source,
            NodeKind::Levels,
            NodeKind::Blur,
            NodeKind::NormalFromHeight,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, kind)| Node {
            id: NodeId(index as u64 + 1),
            input: (index > 0).then_some(NodeId(index as u64)),
            params: NodeParams::default_for(kind, source.clone()),
        })
        .collect();
        let graph = Graph::new(nodes, NodeId(4))?;
        self.install_import(graph)
    }

    /// Install a typed material through the same token-checked import transaction.
    pub fn complete_material_import(
        &mut self,
        token: ImportToken,
        source: SourceDescriptor,
    ) -> Result<CommandOutcome, CommandError> {
        self.complete_material_import_with_convention(
            token,
            source,
            crate::NormalConvention::OpenGl,
        )
    }

    /// The captured import convention is part of the new document, not an undoable edit.
    pub fn complete_material_import_with_convention(
        &mut self,
        token: ImportToken,
        source: SourceDescriptor,
        convention: crate::NormalConvention,
    ) -> Result<CommandOutcome, CommandError> {
        self.match_import(token)?;
        let mut dto = crate::MaterialGraphDto::from(crate::demo_material_from_source(source)?);
        for node in &mut dto.graph {
            if let MaterialNodeParams::NormalSource {
                convention: value, ..
            } = &mut node.params
            {
                *value = convention;
            }
        }
        self.install_import(Graph::try_from(dto)?)
    }

    fn install_import(&mut self, graph: Graph) -> Result<CommandOutcome, CommandError> {
        let generation = self.next_generation()?;
        self.document = Some(EvaluationSnapshot { generation, graph });
        self.view = ViewState::default();
        self.export_settings = None;
        self.capture_a_snapshot = None;
        self.capture_b_snapshot = None;
        self.document_revision = self.document_revision.saturating_add(1);
        self.request_id = self.request_id.saturating_add(1);
        self.pending_import = None;
        self.project_assets.clear();
        self.undo.clear();
        self.redo.clear();
        self.undo_bytes = 0;
        self.redo_bytes = 0;
        self.gesture = None;
        self.stroke = None;
        Ok(self.outcome(Vec::new(), true))
    }

    /// Consumes only the latest decode request, preserving the current document.
    pub fn fail_import(&mut self, token: ImportToken) -> Result<CommandOutcome, CommandError> {
        self.match_import(token)?;
        self.pending_import = None;
        Ok(self.outcome(Vec::new(), false))
    }

    fn match_import(&self, token: ImportToken) -> Result<(), CommandError> {
        if self.pending_import == Some(token) {
            Ok(())
        } else {
            Err(CommandError::StaleImport(token))
        }
    }

    fn match_gesture(&self, id: GestureId) -> Result<&Gesture, CommandError> {
        self.gesture
            .as_ref()
            .filter(|gesture| gesture.id == id)
            .ok_or(CommandError::StaleGesture(id))
    }

    fn parameter(&self, address: ParameterAddress) -> Result<ParameterValue, CommandError> {
        let document = self.document.as_ref().ok_or(CommandError::NoDocument)?;
        Ok(document
            .graph
            .parameter_value(address.node, address.parameter)?)
    }

    fn next_generation(&self) -> Result<u64, CommandError> {
        self.document
            .as_ref()
            .map_or(0, |document| document.generation)
            .checked_add(1)
            .ok_or(CommandError::GenerationOverflow)
    }

    // All fallible work precedes mutation. Callers can then update history without
    // cloning growing stacks on each slider value or undo operation.
    fn change_parameter(
        &mut self,
        address: ParameterAddress,
        value: ParameterValue,
    ) -> Result<bool, CommandError> {
        let before = self.parameter(address)?;
        if before == value {
            return Ok(false);
        }
        let generation = self.next_generation()?;
        let document = self.document.as_mut().ok_or(CommandError::NoDocument)?;
        document
            .graph
            .set_parameter(address.node, address.parameter, value)?;
        document.generation = generation;
        Ok(true)
    }

    fn commit_diff(&mut self, diff: impl Into<HistoryDiff>) {
        let diff = diff.into();
        if !matches!(&diff, HistoryDiff::Parameter(diff) if diff.before == diff.after) {
            self.undo_bytes += diff.byte_count();
            self.undo.push(diff);
            self.redo.clear();
            self.redo_bytes = 0;
        }
    }

    fn commit_gesture(&mut self) {
        if let Some(gesture) = self.gesture.take() {
            self.commit_diff(gesture.diff);
        }
    }

    fn replay(&mut self, redo: bool) -> Result<(bool, bool, bool), CommandError> {
        let id = if redo {
            CommandId::Redo
        } else {
            CommandId::Undo
        };
        if !self.enabled(id) {
            return Err(CommandError::Disabled(id));
        }
        // Determine the post-commit target without consuming the gesture. A redo
        // after a nonempty gesture is disabled because commit clears its branch.
        let diff = if redo {
            self.redo.last().cloned()
        } else {
            self.gesture
                .as_ref()
                .filter(|gesture| gesture.changed())
                .map(|gesture| HistoryDiff::Parameter(gesture.diff.clone()))
                .or_else(|| {
                    self.stroke
                        .as_ref()
                        .map(|stroke| HistoryDiff::Stroke(self.stroke_diff(stroke)))
                })
                .or_else(|| self.undo.last().cloned())
        }
        .expect("enabled history command");
        let is_map = matches!(diff, HistoryDiff::Map { .. });
        let is_export = matches!(diff, HistoryDiff::ExportSettings { .. });
        let changed = self.change_diff(&diff, redo)?;
        self.commit_gesture();
        // The graph may already have been undone; use the captured diff instead
        // of reading its after value back from the now changed document.
        if self.stroke.take().is_some() {
            self.commit_diff(diff.clone());
        }
        let bytes = diff.byte_count();
        if redo {
            self.redo.pop();
            self.redo_bytes -= bytes;
            self.undo.push(diff);
            self.undo_bytes += bytes;
        } else {
            self.undo.pop();
            self.undo_bytes -= bytes;
            self.redo.push(diff);
            self.redo_bytes += bytes;
        }
        Ok((changed, is_map, is_export))
    }

    fn match_stroke(&self, id: StrokeId) -> Result<&StrokeGesture, CommandError> {
        self.stroke
            .as_ref()
            .filter(|stroke| stroke.builder.stroke.id == id)
            .ok_or(CommandError::StaleStroke(id))
    }

    fn stroke_diff(&self, stroke: &StrokeGesture) -> StrokeDiff {
        let (strokes, after_digest) = self.mask(stroke.node).expect("active stroke node");
        StrokeDiff {
            node: stroke.node,
            index: stroke.index,
            stroke: strokes[stroke.index].clone(),
            before_digest: stroke.before_digest,
            after_digest,
        }
    }

    fn commit_stroke(&mut self) {
        if let Some(stroke) = &self.stroke {
            let diff = self.stroke_diff(stroke);
            self.stroke = None;
            self.commit_diff(HistoryDiff::Stroke(diff));
        }
    }

    fn mask(&self, node: NodeId) -> Result<(&[BrushStroke], [u8; 32]), CommandError> {
        let document = self.document.as_ref().ok_or(CommandError::NoDocument)?;
        let target = document
            .graph
            .material_node(node)
            .ok_or(CoreError::MissingNode(node))?;
        if let MaterialNodeParams::PaintMask {
            strokes,
            stroke_digest,
            ..
        } = &target.params
        {
            Ok((strokes, *stroke_digest))
        } else {
            Err(CoreError::InvalidMaterial("stroke target must be a PaintMask").into())
        }
    }

    fn begin_stroke(
        &mut self,
        node: NodeId,
        id: StrokeId,
        brush: Brush,
        sample: BrushSample,
    ) -> Result<bool, CommandError> {
        if self.gesture.is_some() || self.stroke.is_some() {
            return Err(CommandError::GestureAlreadyActive);
        }
        self.ensure_unlocked(node)?;
        let builder = StrokeBuilder::new(id, brush, sample)?;
        let (strokes, before_digest) = self.mask(node)?;
        if strokes.is_empty() && before_digest != [0; 32] {
            return Err(CoreError::InvalidMaterial(
                "digest-only legacy mask has no replay data; create a vector mask before painting",
            )
            .into());
        }
        if strokes.iter().any(|stroke| stroke.id == id) {
            return Err(CoreError::InvalidMaterial("stroke ID already exists").into());
        }
        let index = strokes.len();
        let mut strokes = strokes.to_vec();
        strokes.push(builder.preview()?);
        let digest = stroke_digest(&strokes)?;
        let changed = self.change_mask(node, strokes, digest)?;
        self.stroke = Some(StrokeGesture {
            node,
            index,
            before_digest,
            builder,
        });
        Ok(changed)
    }

    fn append_stroke(
        &mut self,
        id: StrokeId,
        samples: &[BrushSample],
    ) -> Result<bool, CommandError> {
        let active = self.match_stroke(id)?;
        let mut builder = active.builder.clone();
        builder.append(samples)?;
        let node = active.node;
        let mut strokes = self.mask(node)?.0.to_vec();
        strokes[active.index] = builder.preview()?;
        let digest = stroke_digest(&strokes)?;
        let changed = self.change_mask(node, strokes, digest)?;
        self.stroke.as_mut().expect("matching stroke").builder = builder;
        Ok(changed)
    }

    fn change_diff(&mut self, diff: &HistoryDiff, redo: bool) -> Result<bool, CommandError> {
        match diff {
            HistoryDiff::MaterialOptions { before, after } => {
                let (seamless, painted_mask) = if redo { *after } else { *before };
                let graph = crate::configure_material(
                    &self.require_document()?.graph,
                    seamless,
                    painted_mask,
                )?;
                let generation = self.next_generation()?;
                self.document = Some(EvaluationSnapshot { generation, graph });
                Ok(true)
            }
            HistoryDiff::ExportSettings { before, after } => {
                self.export_settings = if redo { after.clone() } else { before.clone() };
                Ok(true)
            }
            HistoryDiff::Map { map, before, after } => {
                self.view
                    .maps
                    .insert(*map, if redo { *after } else { *before });
                Ok(true)
            }
            HistoryDiff::Parameter(diff) => self.change_parameter(
                diff.address,
                if redo {
                    diff.after.clone()
                } else {
                    diff.before.clone()
                },
            ),
            HistoryDiff::Stroke(diff) => {
                let mut strokes = self.mask(diff.node)?.0.to_vec();
                if redo {
                    strokes.insert(diff.index, diff.stroke.clone());
                } else {
                    strokes.remove(diff.index);
                }
                self.change_mask(
                    diff.node,
                    strokes,
                    if redo {
                        diff.after_digest
                    } else {
                        diff.before_digest
                    },
                )
            }
        }
    }

    fn change_mask(
        &mut self,
        node: NodeId,
        strokes: Vec<BrushStroke>,
        digest: [u8; 32],
    ) -> Result<bool, CommandError> {
        let (before, before_digest) = self.mask(node)?;
        if before == strokes && before_digest == digest {
            return Ok(false);
        }
        let generation = self.next_generation()?;
        let document = self.document.as_mut().ok_or(CommandError::NoDocument)?;
        let material = document.graph.material.as_mut().expect("material mask");
        let target = material
            .graph
            .iter()
            .position(|target| target.id == node)
            .expect("mask exists");
        let MaterialNodeParams::PaintMask {
            extent, boundary, ..
        } = material.graph[target].params
        else {
            unreachable!("mask checked")
        };
        let params = MaterialNodeParams::PaintMask {
            boundary,
            extent,
            stroke_digest: digest,
            strokes,
        };
        params.validate()?;
        let points: usize = material
            .graph
            .iter()
            .filter(|target| target.id != node)
            .map(|target| match &target.params {
                MaterialNodeParams::PaintMask { strokes, .. } => {
                    strokes.iter().map(|stroke| stroke.points.len()).sum()
                }
                _ => 0,
            })
            .sum();
        let MaterialNodeParams::PaintMask { strokes, .. } = &params else {
            unreachable!()
        };
        if points
            + strokes
                .iter()
                .map(|stroke| stroke.points.len())
                .sum::<usize>()
            > crate::MAX_BRUSH_POINTS
        {
            return Err(CoreError::InvalidMaterial(
                "graph brush point limit exceeded; finish and restart",
            )
            .into());
        }
        material.graph[target].params = params;
        document.generation = generation;
        Ok(true)
    }

    fn require_document(&self) -> Result<&EvaluationSnapshot, CommandError> {
        self.document.as_ref().ok_or(CommandError::NoDocument)
    }
    fn map_view(&self, map: MapKind) -> Result<MapView, CommandError> {
        let graph = &self.require_document()?.graph;
        if !graph
            .material_outputs()
            .is_some_and(|outputs| outputs.contains_key(&map))
        {
            return Err(CoreError::InvalidMaterial("map is not present").into());
        }
        Ok(self.view.maps.get(&map).copied().unwrap_or_default())
    }
    fn set_map(&mut self, map: MapKind, after: MapView) -> Result<bool, CommandError> {
        if self.gesture.is_some() || self.stroke.is_some() {
            return Err(CommandError::GestureAlreadyActive);
        }
        let before = self.map_view(map)?;
        if before == after {
            return Ok(false);
        }
        self.view.maps.insert(map, after);
        self.commit_diff(HistoryDiff::Map { map, before, after });
        Ok(true)
    }
    fn ensure_unlocked(&self, node: NodeId) -> Result<(), CommandError> {
        let graph = &self.require_document()?.graph;
        if graph.material_outputs().is_some_and(|outputs| {
            outputs.iter().any(|(map, output)| {
                output.node == node && self.view.maps.get(map).is_some_and(|state| state.locked)
            })
        }) {
            return Err(CoreError::InvalidMaterial("map is locked").into());
        }
        Ok(())
    }
    fn capture(&self) -> Result<CaptureIdentity, CommandError> {
        Ok(CaptureIdentity {
            generation: self.require_document()?.generation,
            document_revision: self.document_revision,
            request_id: self.request_id,
        })
    }

    fn export_output(&self, map: MapKind) -> Result<NodeId, CommandError> {
        let graph = &self
            .document
            .as_ref()
            .ok_or(CommandError::NoDocument)?
            .graph;
        if let Some(outputs) = graph.material_outputs() {
            return outputs
                .get(&map)
                .map(|output| output.node)
                .ok_or(CommandError::Disabled(CommandId::SavePng));
        }
        let matches = |node: &&Node| {
            node.params.kind() != NodeKind::Source && node.params.output_kind() == map
        };
        let mut current = graph.node(graph.selected_output());
        while let Some(node) = current {
            if matches(&node) {
                return Ok(node.id);
            }
            current = node.input.and_then(|id| graph.node(id));
        }
        graph
            .topological_nodes()
            .into_iter()
            .rev()
            .find(matches)
            .map(|node| node.id)
            .ok_or(CommandError::Disabled(CommandId::SavePng))
    }

    fn outcome(&self, effects: Vec<Effect>, changed: bool) -> CommandOutcome {
        CommandOutcome {
            effects,
            snapshot: if changed { self.snapshot() } else { None },
            history: self.history(),
            availability: self.availability(),
        }
    }
}
