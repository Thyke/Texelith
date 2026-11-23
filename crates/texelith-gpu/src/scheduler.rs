//! Immediate scheduling transitions, shared by the worker and fake-fence tests.
use crate::{JobId, MaterialRequest, PreviewRequest, Quality, RenderRequest};
use std::time::Instant;
use texelith_core::{Extent, NodeId, NodeParams, SourceDescriptor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedulerError {
    Busy,
    Closed,
    DeviceLost,
    Stale,
    InvalidRequest,
    ExhaustedIds,
}
impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SchedulerError {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkKind {
    Interactive,
    Save,
}
#[derive(Clone, Debug)]
pub struct Work {
    pub id: u64,
    pub request: RenderRequest,
    pub kind: WorkKind,
    pub requested_at: Instant,
    pub material: Option<MaterialRequest>,
    pub preview: Option<PreviewRequest>,
    pub(crate) export_job: Option<JobId>,
    intent: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Published,
    Discarded,
    Saved,
    Unknown,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveEvent {
    Completed {
        save_id: u64,
        request_id: u64,
        generation: u64,
        extent: Extent,
        output: NodeId,
    },
    Cancelled {
        save_id: u64,
    },
    Failed {
        save_id: u64,
        message: String,
    },
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerDiagnostics {
    pub in_flight: u64,
    pub pending: u64,
    pub peak_in_flight: u64,
    pub peak_pending: u64,
    pub submitted: u64,
    pub coalesced: u64,
    pub stale_publications: u64,
    pub cancellation_requests: u64,
}
#[derive(Default)]
pub struct Scheduler {
    export_job: Option<JobId>,
    pending_preview: Option<PreviewRequest>,
    latest_preview: Option<(JobId, u64)>,
    latest_material: Option<MaterialRequest>,
    published_material: Option<MaterialRequest>,
    intent: u64,
    latest: Option<RenderRequest>,
    latest_source: Option<SourceDescriptor>,
    pending: Option<(RenderRequest, Instant)>,
    active: Option<Work>,
    published: Option<RenderRequest>,
    save: Option<(u64, RenderRequest, Instant)>,
    save_source: Option<SourceDescriptor>,
    save_started: bool,
    save_event: Option<SaveEvent>,
    next_save: u64,
    next_work: u64,
    closed: bool,
    error: Option<SchedulerError>,
    stats: SchedulerDiagnostics,
    intent_valid: bool,
}

pub(crate) fn source(request: &RenderRequest) -> Result<&SourceDescriptor, SchedulerError> {
    let mut node = request
        .snapshot
        .graph
        .node(request.output)
        .ok_or(SchedulerError::InvalidRequest)?;
    while let Some(input) = node.input {
        node = request
            .snapshot
            .graph
            .node(input)
            .ok_or(SchedulerError::InvalidRequest)?;
    }
    match &node.params {
        NodeParams::Source(source) => Ok(source),
        _ => Err(SchedulerError::InvalidRequest),
    }
}

impl Scheduler {
    pub(crate) fn request_preview(
        &mut self,
        request: PreviewRequest,
    ) -> Result<(), SchedulerError> {
        request
            .validate()
            .map_err(|_| SchedulerError::InvalidRequest)?;
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        let material = self.latest_material.as_ref().ok_or(SchedulerError::Stale)?;
        if material.job_id != request.job_id
            || material.snapshot.generation != request.generation
            || material.request_id != request.material.request_id
            || self
                .latest_preview
                .is_some_and(|(job, id)| job == request.job_id && id >= request.request_id)
        {
            return Err(SchedulerError::Stale);
        }
        self.intent = self
            .intent
            .checked_add(1)
            .ok_or(SchedulerError::ExhaustedIds)?;
        self.stats.cancellation_requests += u64::from(self.active.is_some());
        self.cancel_save();
        self.latest_preview = Some((request.job_id, request.request_id));
        self.pending_preview = Some(request);
        if self
            .pending
            .replace((
                self.latest.as_ref().expect("material intent").clone(),
                Instant::now(),
            ))
            .is_some()
        {
            self.stats.coalesced += 1;
        }
        self.stats.peak_pending = 1;
        self.intent_valid = true;
        Ok(())
    }
    pub(crate) fn request_retained_preview(
        &mut self,
        request: PreviewRequest,
        material: MaterialRequest,
    ) -> Result<(), SchedulerError> {
        // The worker authenticates this explicit snapshot before entering here.
        // Re-select its scheduling metadata without queueing material evaluation.
        request
            .validate()
            .map_err(|_| SchedulerError::InvalidRequest)?;
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        if self
            .latest_preview
            .is_some_and(|(job, id)| job == request.job_id && id >= request.request_id)
        {
            return Err(SchedulerError::Stale);
        }
        self.latest_material = None;
        self.request_material(material)?;
        self.request_preview(request)
    }
    pub fn request_material(&mut self, request: MaterialRequest) -> Result<(), SchedulerError> {
        request
            .validate()
            .map_err(|_| SchedulerError::InvalidRequest)?;
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        if self.latest_material.as_ref().is_some_and(|r| {
            r.job_id == request.job_id
                && (request.request_id <= r.request_id
                    || request.snapshot.generation < r.snapshot.generation)
        }) {
            return Err(SchedulerError::Stale);
        }
        let intent = self
            .intent
            .checked_add(1)
            .ok_or(SchedulerError::ExhaustedIds)?;
        let placeholder = RenderRequest {
            request_id: request.request_id,
            snapshot: request.snapshot.clone(),
            output: request
                .outputs
                .values()
                .next()
                .expect("validated nonempty")
                .node,
            extent: request.extent,
            quality: request.quality,
        };
        self.intent = intent;
        self.pending_preview = None;
        if self
            .latest_preview
            .is_some_and(|(job, _)| job != request.job_id)
        {
            self.latest_preview = None;
        }
        self.stats.cancellation_requests += u64::from(self.active.is_some());
        self.cancel_save();
        self.published = None;
        self.published_material = None;
        self.latest = Some(placeholder.clone());
        self.latest_source = None;
        self.latest_material = Some(request);
        self.intent_valid = true;
        if self
            .pending
            .replace((placeholder, Instant::now()))
            .is_some()
        {
            self.stats.coalesced += 1;
        }
        self.stats.peak_pending = 1;
        Ok(())
    }
    pub fn try_latest_material(&mut self) -> Option<MaterialRequest> {
        self.published_material.take()
    }
    /// Current interactive/save documents retain their content-addressed source
    /// while independent batch inputs pass through the resident registry.
    pub(crate) fn retains_source(&self, source: &SourceDescriptor) -> bool {
        self.latest_source.as_ref() == Some(source)
            || self.save_source.as_ref() == Some(source)
            || self.latest_material.as_ref().is_some_and(|request| {
                request
                    .snapshot
                    .graph
                    .material_nodes()
                    .iter()
                    .any(|node| crate::builtin::source_descriptor(node) == Some(source))
            })
    }
    pub fn should_continue(&self, work_id: u64) -> bool {
        self.active
            .as_ref()
            .is_some_and(|w| w.id == work_id && self.is_current(w))
    }
    pub fn request(&mut self, request: RenderRequest) -> Result<(), SchedulerError> {
        request
            .extent
            .validate()
            .map_err(|_| SchedulerError::InvalidRequest)?;
        let source = source(&request)?.clone();
        self.request_validated(request, source)
    }
    pub(crate) fn request_validated(
        &mut self,
        request: RenderRequest,
        source: SourceDescriptor,
    ) -> Result<(), SchedulerError> {
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        if let Some(latest) = &self.latest
            && self.latest_material.is_none()
            && (request.request_id <= latest.request_id
                || request.snapshot.generation < latest.snapshot.generation)
        {
            return Err(SchedulerError::Stale);
        }
        if self.save.as_ref().is_some_and(|(_, save, _)| {
            save.snapshot.generation != request.snapshot.generation
                || self.save_source.as_ref() != Some(&source)
        }) {
            self.cancel_save();
        }
        self.intent = self
            .intent
            .checked_add(1)
            .ok_or(SchedulerError::ExhaustedIds)?;
        self.stats.cancellation_requests += u64::from(self.active.is_some());
        self.latest_material = None;
        self.pending_preview = None;
        self.latest_preview = None;
        self.published_material = None;
        self.published = None;
        self.intent_valid = true;
        self.latest = Some(request.clone());
        self.latest_source = Some(source);
        if self.pending.replace((request, Instant::now())).is_some() {
            self.stats.coalesced += 1;
        }
        self.stats.peak_pending = 1;
        Ok(())
    }
    pub fn request_save(&mut self, request: RenderRequest) -> Result<u64, SchedulerError> {
        let source = source(&request)?.clone();
        self.request_save_validated(request, source)
    }
    pub(crate) fn request_save_validated(
        &mut self,
        request: RenderRequest,
        source: SourceDescriptor,
    ) -> Result<u64, SchedulerError> {
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        if self.export_job.is_some()
            || self.save.is_some()
            || self.save_event.is_some()
            || self
                .active
                .as_ref()
                .is_some_and(|w| w.kind == WorkKind::Save)
        {
            return Err(SchedulerError::Busy);
        }
        if request.quality != Quality::Full
            || request.extent != source.extent
            || !self.latest.as_ref().is_some_and(|latest| {
                latest.snapshot.generation == request.snapshot.generation
                    && self.latest_source.as_ref() == Some(&source)
            })
            || !self.intent_valid
        {
            return Err(SchedulerError::InvalidRequest);
        }
        self.next_save = self
            .next_save
            .checked_add(1)
            .ok_or(SchedulerError::ExhaustedIds)?;
        self.save = Some((self.next_save, request, Instant::now()));
        self.save_source = Some(source);
        self.save_started = false;
        Ok(self.next_save)
    }
    pub fn start_next(&mut self) -> Option<Work> {
        if self.closed || self.active.is_some() {
            return None;
        }
        let (request, requested_at, kind) = if let Some((r, t)) = self.pending.take() {
            (r, t, WorkKind::Interactive)
        } else if !self.save_started && self.save_event.is_none() {
            let (_, r, t) = self.save.as_ref()?;
            self.save_started = true;
            (r.clone(), *t, WorkKind::Save)
        } else {
            return None;
        };
        let Some(id) = self.next_work.checked_add(1) else {
            self.fail(SchedulerError::ExhaustedIds);
            return None;
        };
        self.next_work = id;
        let preview = if kind == WorkKind::Interactive {
            self.pending_preview.take()
        } else {
            None
        };
        let work = Work {
            id,
            request,
            kind,
            requested_at,
            material: if kind == WorkKind::Interactive && preview.is_none() {
                self.latest_material.clone()
            } else {
                None
            },
            preview,
            export_job: None,
            intent: self.intent,
        };
        self.active = Some(work.clone());
        self.stats.submitted += 1;
        self.stats.peak_in_flight = 1;
        Some(work)
    }
    pub(crate) fn is_current(&self, work: &Work) -> bool {
        if let Some(job) = work.export_job {
            // Live edits yield to interactive work but never invalidate the
            // captured job. The worker resumes the item after that work retires.
            return !self.closed && self.export_job == Some(job) && self.pending.is_none();
        }
        !self.closed
            && self.intent_valid
            && match work.kind {
                WorkKind::Interactive => self.latest.as_ref().is_some_and(|r| {
                    work.intent == self.intent
                        && r.request_id == work.request.request_id
                        && r.snapshot.generation == work.request.snapshot.generation
                }),
                WorkKind::Save => self.save.is_some() && self.save_event.is_none(),
            }
    }
    pub fn complete(&mut self, id: u64) -> Completion {
        if !self.active.as_ref().is_some_and(|w| w.id == id) {
            return Completion::Unknown;
        }
        let work = self.active.take().expect("checked active");
        if !self.is_current(&work) {
            return Completion::Discarded;
        }
        match work.kind {
            WorkKind::Interactive => {
                if work.preview.is_some() {
                    // The worker publishes the fenced preview, not a map frame.
                } else if let Some(material) = work.material {
                    self.published_material = Some(material);
                } else {
                    self.published = Some(work.request);
                }
                Completion::Published
            }
            WorkKind::Save => {
                let save_id = self.save.as_ref().expect("current save").0;
                self.save_event = Some(SaveEvent::Completed {
                    save_id,
                    request_id: work.request.request_id,
                    generation: work.request.snapshot.generation,
                    extent: work.request.extent,
                    output: work.request.output,
                });
                Completion::Saved
            }
        }
    }
    pub fn try_latest(&mut self) -> Option<RenderRequest> {
        self.published.take()
    }
    pub fn try_save_event(&mut self) -> Option<SaveEvent> {
        let event = self.save_event.take()?;
        self.save = None;
        self.save_source = None;
        self.save_started = false;
        Some(event)
    }
    pub fn close(&mut self) {
        self.export_job = None;
        self.closed = true;
        self.pending = None;
        self.pending_preview = None;
        self.published = None;
        self.published_material = None;
        self.cancel_save();
    }
    pub fn fail(&mut self, error: SchedulerError) {
        self.close();
        self.error = Some(error);
    }
    pub fn try_error(&mut self) -> Option<SchedulerError> {
        self.error.take()
    }
    pub fn diagnostics(&self) -> SchedulerDiagnostics {
        SchedulerDiagnostics {
            pending: u64::from(self.pending.is_some()),
            in_flight: u64::from(self.active.is_some()),
            ..self.stats
        }
    }
    pub(crate) fn closed(&self) -> bool {
        self.closed
    }
    pub(crate) fn save_id(&self) -> Option<u64> {
        self.save.as_ref().map(|s| s.0)
    }
    pub(crate) fn cancel_active(&mut self, id: u64) {
        if self.active.as_ref().is_some_and(|w| w.id == id) {
            self.active = None;
        }
    }
    pub(crate) fn fail_active(&mut self, id: u64, message: String) {
        if self.active.as_ref().is_some_and(|work| {
            work.id == id && work.kind == WorkKind::Save && self.is_current(work)
        }) {
            self.save_event = Some(SaveEvent::Failed {
                save_id: self.save_id().expect("current save"),
                message,
            });
        }
        self.cancel_active(id);
    }
    fn cancel_save(&mut self) {
        if let Some((id, _, _)) = &self.save {
            self.save_event = Some(SaveEvent::Cancelled { save_id: *id });
        }
    }

    pub(crate) fn admit_export(&mut self, job: JobId) -> Result<(), SchedulerError> {
        if self.closed {
            return Err(SchedulerError::Closed);
        }
        if self.export_job.is_some()
            || self.save.is_some()
            || self.save_event.is_some()
            || self
                .active
                .as_ref()
                .is_some_and(|w| w.kind == WorkKind::Save)
        {
            return Err(SchedulerError::Busy);
        }
        self.export_job = Some(job);
        Ok(())
    }
    pub(crate) fn cancel_export(&mut self) {
        self.export_job = None;
    }
    pub(crate) fn start_export(&mut self, request: RenderRequest, job: JobId) -> Option<Work> {
        if self.closed
            || self.export_job != Some(job)
            || self.active.is_some()
            || self.pending.is_some()
        {
            return None;
        }
        self.next_work = self.next_work.checked_add(1)?;
        let work = Work {
            id: self.next_work,
            request,
            kind: WorkKind::Save,
            requested_at: Instant::now(),
            material: None,
            preview: None,
            export_job: Some(job),
            intent: self.intent,
        };
        self.active = Some(work.clone());
        self.stats.submitted += 1;
        self.stats.peak_in_flight = 1;
        Some(work)
    }
}
