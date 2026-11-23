//! Shared-device boundary. UI calls only change bounded mailboxes; GPU work,
//! hashing, allocation, polling and mapping are confined to the named worker.
use crate::{
    CompletedMaterial, CompletedPreview, GpuError, MaterialRequest, PreviewRequest,
    ResourceReservations, TypedSourcePixels, TypedSourceUpload,
    pool::ResourcePool,
    scheduler::{SaveEvent, Scheduler, SchedulerError},
};
use crossbeam_channel::{Sender, bounded};
use std::{
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Instant,
};
use texelith_core::{EvaluationSnapshot, Extent, MapKind, NodeId, SourceDescriptor};

#[path = "worker_runtime.rs"]
mod runtime;
pub use runtime::DisplayLease;
pub(crate) use runtime::node_registration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Proxy,
    Full,
}
#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub request_id: u64,
    pub snapshot: Arc<EvaluationSnapshot>,
    pub output: NodeId,
    pub extent: Extent,
    pub quality: Quality,
}
#[derive(Clone)]
pub struct SourceUpload {
    pub descriptor: SourceDescriptor,
    pub pixels: Arc<[u8]>,
}
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub budget_bytes: u64,
    pub max_cache_entries: usize,
}
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            budget_bytes: 1024 * 1024 * 1024,
            max_cache_entries: 128,
        }
    }
}
pub struct CompletedFrame {
    pub request_id: u64,
    pub generation: u64,
    pub quality: Quality,
    pub extent: Extent,
    pub output: NodeId,
    pub lease: DisplayLease,
    pub requested_at: Instant,
    pub submitted_at: Instant,
    pub completed_at: Instant,
}
#[derive(Debug)]
pub enum WorkerEvent {
    SourceReady {
        content_hash: [u8; 32],
    },
    BudgetExceeded {
        budget_bytes: u64,
        required_bytes: u64,
    },
    DeviceLost,
    Error(GpuError),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveError {
    Busy,
    Cancelled,
    Closed,
    InvalidRequest,
    Failed(String),
}
impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SaveError {}
impl From<SchedulerError> for SaveError {
    fn from(e: SchedulerError) -> Self {
        match e {
            SchedulerError::Busy => Self::Busy,
            SchedulerError::Closed => Self::Closed,
            _ => Self::InvalidRequest,
        }
    }
}

/// Tightly packed, quantized map bytes for the I/O worker. Height is big-endian
/// 16-bit grayscale; normals are 8-bit RGB, OpenGL +Y. No display transfer or
/// row padding remains. The PNG encoder must omit sRGB tags for these data maps.
#[derive(Debug)]
pub struct CompletedReadback {
    pub save_id: u64,
    pub request_id: u64,
    pub generation: u64,
    pub output: NodeId,
    pub extent: Extent,
    pub kind: MapKind,
    pub pixels: Vec<u8>,
    pub requested_at: Instant,
    pub submitted_at: Instant,
    pub completed_at: Instant,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerDiagnostics {
    pub in_flight: u64,
    pub pending: u64,
    pub peak_in_flight: u64,
    pub peak_pending: u64,
    pub peak_bytes: u64,
    pub allocated_bytes: u64,
    pub retired_bytes: u64,
    pub submitted: u64,
    pub coalesced: u64,
    pub stale_publications: u64,
    pub validation_errors: u64,
    pub cache_hits: u64,
    pub pyramid_cache_hits: u64,
    pub cache_misses: u64,
    pub scheduled: u64,
    pub cancellation_requests: u64,
    pub cancelled_pass_batches: u64,
    pub cache_evictions: u64,
    /// Active reservation shares; retiring shares remain charged in retired_bytes.
    pub viewport_reserved_bytes: u64,
    pub readback_reserved_bytes: u64,
    pub queue_submit_ns: u64,
    /// T-067: last completed cold convolution, excluding viewport rendering.
    pub ibl_convolution_ns: u64,
}

#[derive(Default)]
struct Mailboxes {
    clear_results: Option<Sender<()>>,
    export: crate::export_readback::Mailbox,
    scheduler: Scheduler,
    frame: Option<CompletedFrame>,
    material: Option<CompletedMaterial>,
    preview: Option<CompletedPreview>,
    material_identity: Option<runtime::MaterialIdentity>,
    retained_materials: std::collections::VecDeque<runtime::MaterialIdentity>,
    reservations: ResourceReservations,
    upload: Option<TypedSourceUpload>,
    uploading: bool,
    source_serial: u64,
    event: Option<WorkerEvent>,
    ready: Option<[u8; 32]>,
    readback: Option<Result<CompletedReadback, SaveError>>,
    validation_errors: u64,
    cache_hits: u64,
    pyramid_cache_hits: u64,
    cache_misses: u64,
    compute_submissions: u64,
    device_lost: bool,
    cancelled_pass_batches: u64,
    cache_evictions: u64,
    viewport_reserved_bytes: u64,
    readback_reserved_bytes: u64,
    queue_submit_ns: u64,
    ibl_convolution_ns: u64, // T-067
}

pub struct GpuWorker {
    shared: Arc<Mutex<Mailboxes>>,
    pool: Arc<Mutex<ResourcePool>>,
    wake: Sender<()>,
    thread: Mutex<Option<JoinHandle<()>>>,
}
impl GpuWorker {
    /// Accept one immutable material set. IDs must increase for this worker,
    /// so delayed IO acknowledgement cannot target a later job with the same ID.
    pub fn request_export(&self, request: crate::ExportRequest) -> Result<(), SaveError> {
        let mut state = self.shared.lock().unwrap();
        if state.scheduler.closed() {
            return Err(SaveError::Closed);
        }
        if state.export.request.is_some()
            || state.export.retiring
            || state.export.progress.is_some()
            || state.export.charge.upgrade().is_some()
        {
            return Err(SaveError::Busy);
        }
        if state
            .export
            .last_job
            .is_some_and(|id| id.0 >= request.job_id.0)
        {
            return Err(SaveError::InvalidRequest);
        }
        state.scheduler.admit_export(request.job_id)?;
        state.export.last_job = Some(request.job_id);
        state.export.next_item = 0;
        state.export.outstanding = false;
        state.export.delivered = false;
        state.export.request = Some(request);
        state.export.retiring = true;
        state.export.progress(crate::ExportStage::Evaluating, None);
        drop(state);
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn try_export_progress(&self) -> Option<crate::ExportProgress> {
        self.shared.lock().unwrap().export.progress.take()
    }
    pub fn try_export_plane(&self) -> Option<crate::ExportPlane> {
        let mut state = self.shared.lock().unwrap();
        let plane = state.export.plane.take()?;
        state.export.delivered = true;
        Some(plane)
    }
    /// Call after the bounded IO encode finishes. Consuming the plane releases
    /// its byte reservation before the next output can be scheduled.
    pub fn acknowledge_export(
        &self,
        plane: crate::ExportPlane,
        result: Result<(), String>,
    ) -> Result<(), SaveError> {
        let mut state = self.shared.lock().unwrap();
        if !state
            .export
            .request
            .as_ref()
            .is_some_and(|r| r.job_id == plane.charge.job)
            || state.export.next_item != plane.charge.item
            || !state.export.delivered
            || !state.export.charge.ptr_eq(&Arc::downgrade(&plane.charge))
        {
            return Err(SaveError::Cancelled);
        }
        drop(plane);
        state.export.outstanding = false;
        state.export.delivered = false;
        if let Err(error) = result {
            state
                .export
                .progress(crate::ExportStage::Failed, Some(error));
            state.export.request = None;
            state.scheduler.cancel_export();
        } else {
            state.export.next_item += 1;
            let done =
                state.export.next_item == state.export.request.as_ref().unwrap().plan.outputs.len();
            state.export.progress(
                if done {
                    crate::ExportStage::ReadyToPublish
                } else {
                    crate::ExportStage::Evaluating
                },
                None,
            );
        }
        drop(state);
        let _ = self.wake.try_send(());
        Ok(())
    }
    /// IO must call this before publishing its completed staging transaction.
    pub fn begin_export_publication(&self, job: crate::JobId) -> Result<(), SaveError> {
        let mut state = self.shared.lock().unwrap();
        if !state
            .export
            .request
            .as_ref()
            .is_some_and(|r| r.job_id == job)
            || state.export.stage != Some(crate::ExportStage::ReadyToPublish)
        {
            return Err(SaveError::InvalidRequest);
        }
        state.export.progress(crate::ExportStage::Publishing, None);
        Ok(())
    }
    pub fn finish_export(
        &self,
        job: crate::JobId,
        result: Result<(), String>,
    ) -> Result<(), SaveError> {
        let mut state = self.shared.lock().unwrap();
        if !state
            .export
            .request
            .as_ref()
            .is_some_and(|r| r.job_id == job)
            || state.export.stage != Some(crate::ExportStage::Publishing)
        {
            return Err(SaveError::InvalidRequest);
        }
        match result {
            Ok(()) => state.export.progress(crate::ExportStage::Completed, None),
            Err(error) => state
                .export
                .progress(crate::ExportStage::Failed, Some(error)),
        }
        state.export.request = None;
        state.scheduler.cancel_export();
        drop(state);
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn cancel_export(&self, job: crate::JobId) -> Result<(), SaveError> {
        let mut state = self.shared.lock().unwrap();
        if !state
            .export
            .request
            .as_ref()
            .is_some_and(|r| r.job_id == job)
        {
            return Err(SaveError::InvalidRequest);
        }
        state.export.cancel();
        state.scheduler.cancel_export();
        drop(state);
        let _ = self.wake.try_send(());
        Ok(())
    }
    /// Replace the single interactive intent; validation never touches the GPU.
    pub fn request_preview(&self, request: PreviewRequest) -> Result<(), GpuError> {
        request.validate()?;
        let old = {
            let mut state = self.shared.lock().unwrap();
            if !state
                .material_identity
                .as_ref()
                .is_some_and(|identity| identity.matches(&request.material))
            {
                return Err(GpuError(
                    "preview requires the current coherent material from this worker".into(),
                ));
            }
            state
                .scheduler
                .request_preview(request)
                .map_err(|e| GpuError(e.to_string()))?;
            state.preview.take()
        };
        drop(old);
        let _ = self.wake.try_send(());
        Ok(())
    }
    /// Render an explicitly retained viewport/A-B snapshot without evaluating its
    /// graph again. Only unmodified, still-live bundles issued by this worker
    /// qualify; the bounded provenance registry owns no material textures.
    pub fn request_retained_preview(&self, request: PreviewRequest) -> Result<(), GpuError> {
        request.validate()?;
        let old = {
            let mut state = self.shared.lock().unwrap();
            if state.device_lost {
                return Err(GpuError("device lost".into()));
            }
            let material = state
                .retained_materials
                .iter()
                .find(|identity| identity.matches(&request.material))
                .ok_or_else(|| {
                    GpuError("retained preview requires an issued coherent material".into())
                })?
                .recipe
                .clone();
            state
                .scheduler
                .request_retained_preview(request, material)
                .map_err(|e| GpuError(e.to_string()))?;
            state.preview.take()
        };
        drop(old);
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn try_latest_preview(&self) -> Option<CompletedPreview> {
        self.shared.lock().unwrap().preview.take()
    }
    pub fn request_material(&self, request: MaterialRequest) -> Result<(), GpuError> {
        request.validate()?;
        let old = {
            let mut state = self.shared.lock().unwrap();
            state
                .scheduler
                .request_material(request)
                .map_err(|e| GpuError(e.to_string()))?;
            state.material_identity = None;
            (
                state.material.take(),
                state.frame.take(),
                state.preview.take(),
            )
        };
        drop(old);
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn try_latest_material(&self) -> Option<CompletedMaterial> {
        self.shared.lock().unwrap().material.take()
    }
    /// Reservations are admitted with the next material's actual live set. Call
    /// only between requests; they occupy the same ledger as textures and fences.
    pub fn set_resource_reservations(
        &self,
        reservations: ResourceReservations,
    ) -> Result<(), GpuError> {
        let mut state = self.shared.lock().unwrap();
        let stats = state.scheduler.diagnostics();
        if stats.pending != 0 || stats.in_flight != 0 || state.export.retiring {
            return Err(GpuError("reservation change Busy".into()));
        }
        reservations
            .viewport_bytes
            .checked_add(reservations.readback_bytes)
            .ok_or_else(|| GpuError("reservation overflow".into()))?;
        state.reservations = reservations;
        Ok(())
    }
    /// Uses the presentation device. Pipeline initialization happens on the worker;
    /// asynchronous initialization failures are returned through `try_event`.
    pub fn spawn(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: GpuConfig,
    ) -> Result<Self, GpuError> {
        if config.budget_bytes == 0 || config.max_cache_entries == 0 {
            return Err(GpuError(
                "nonzero budget and cache capacity required".into(),
            ));
        }
        let shared = Arc::new(Mutex::new(Mailboxes::default()));
        let pool = Arc::new(Mutex::new(ResourcePool::new(config.budget_bytes)));
        let (wake, receive) = bounded(1);
        let weak = Arc::downgrade(&shared);
        let notify = wake.clone();
        device.set_device_lost_callback(move |_, _| {
            if let Some(shared) = weak.upgrade() {
                let mut state = shared.lock().unwrap();
                state.scheduler.fail(SchedulerError::DeviceLost);
                state
                    .export
                    .progress(crate::ExportStage::Failed, Some("device lost".into()));
                state.export.request = None;
                state.export.plane = None;
                state.device_lost = true;
                state.event = Some(WorkerEvent::DeviceLost);
                state.material_identity = None;
                state.retained_materials.clear();
                let old = (
                    state.frame.take(),
                    state.material.take(),
                    state.preview.take(),
                );
                let upload = state.upload.take();
                drop(state);
                drop(old);
                drop(upload);
                let _ = notify.try_send(());
            }
        });
        let worker_shared = shared.clone();
        let worker_pool = pool.clone();
        let thread = std::thread::Builder::new()
            .name("texelith-gpu".into())
            .spawn(move || {
                runtime::run(device, queue, config, worker_shared, worker_pool, receive);
            })
            .map_err(|e| GpuError(e.to_string()))?;
        Ok(Self {
            shared,
            pool,
            wake,
            thread: Mutex::new(Some(thread)),
        })
    }
    pub fn request(&self, request: RenderRequest) -> Result<(), GpuError> {
        // Validation is CPU-only and does not hash or touch pixel data.
        let source = crate::scheduler::source(&request)
            .map_err(|e| GpuError(e.to_string()))?
            .clone();
        let expected = match request.quality {
            Quality::Full => source.extent,
            Quality::Proxy => crate::proxy_extent(source.extent)?,
        };
        if request.extent != expected {
            return Err(GpuError("request extent does not match quality".into()));
        }
        let old = {
            let mut state = self.shared.lock().unwrap();
            state
                .scheduler
                .request_validated(request, source)
                .map_err(|e| GpuError(e.to_string()))?;
            state.material_identity = None;
            state.retained_materials.clear();
            (
                state.frame.take(),
                state.material.take(),
                state.preview.take(),
            )
        };
        drop(old);
        let _ = self.wake.try_send(());
        Ok(())
    }
    /// A single decoded upload (at most 64 MiB) can be outstanding. Hashing and
    /// verification happen on the worker, never in this call.
    pub fn install_source(&self, upload: SourceUpload) -> Result<(), GpuError> {
        self.install_typed_source(TypedSourceUpload {
            pixels: TypedSourcePixels::Rgba8 {
                pixels: upload.pixels,
                interpretation: upload.descriptor.interpretation,
            },
            descriptor: upload.descriptor,
        })
    }
    pub fn install_typed_source(&self, upload: TypedSourceUpload) -> Result<(), GpuError> {
        upload.pixels.validate_length(upload.descriptor.extent)?;
        {
            let mut state = self.shared.lock().unwrap();
            if state.scheduler.closed() {
                return Err(GpuError("worker closed".into()));
            }
            if state.uploading {
                return Err(GpuError("source upload Busy".into()));
            }
            state.source_serial = state
                .source_serial
                .checked_add(1)
                .ok_or_else(|| GpuError("source serial exhausted".into()))?;
            state.ready = None;
            state.upload = Some(upload);
            state.uploading = true;
        };
        let _ = self.wake.try_send(());
        Ok(())
    }
    pub fn try_latest(&self) -> Option<CompletedFrame> {
        self.shared.lock().unwrap().frame.take()
    }
    pub fn try_event(&self) -> Option<WorkerEvent> {
        let mut state = self.shared.lock().unwrap();
        state.event.take().or_else(|| {
            state
                .ready
                .take()
                .map(|content_hash| WorkerEvent::SourceReady { content_hash })
        })
    }
    pub fn request_save(&self, request: RenderRequest) -> Result<u64, SaveError> {
        let source = crate::scheduler::source(&request)?.clone();
        let mut state = self.shared.lock().unwrap();
        if state.export.retiring || state.export.charge.upgrade().is_some() {
            return Err(SaveError::Busy);
        }
        let id = state.scheduler.request_save_validated(request, source)?;
        drop(state);
        let _ = self.wake.try_send(());
        Ok(id)
    }
    pub fn try_save(&self) -> Option<Result<CompletedReadback, SaveError>> {
        let (event, result) = {
            let mut state = self.shared.lock().unwrap();
            let event = state.scheduler.try_save_event()?;
            (event, state.readback.take())
        };
        match event {
            SaveEvent::Cancelled { .. } => Some(Err(SaveError::Cancelled)),
            SaveEvent::Failed { message, .. } => Some(Err(SaveError::Failed(message))),
            SaveEvent::Completed { .. } => {
                Some(result.unwrap_or_else(|| Err(SaveError::Failed("missing readback".into()))))
            }
        }
    }
    pub fn diagnostics(&self) -> WorkerDiagnostics {
        let (scheduler, validation_errors, cache_hits, cache_misses, submissions, extra) = {
            let state = self.shared.lock().unwrap();
            (
                state.scheduler.diagnostics(),
                state.validation_errors,
                state.cache_hits,
                state.cache_misses,
                state.compute_submissions,
                (
                    state.cancelled_pass_batches,
                    state.cache_evictions,
                    state.viewport_reserved_bytes,
                    state.readback_reserved_bytes,
                    state.queue_submit_ns,
                    state.pyramid_cache_hits,
                    state.ibl_convolution_ns, // T-067
                ),
            )
        };
        let pool = self.pool.lock().unwrap().diagnostics();
        WorkerDiagnostics {
            in_flight: scheduler.in_flight,
            pending: scheduler.pending,
            peak_in_flight: scheduler.peak_in_flight,
            peak_pending: scheduler.peak_pending,
            peak_bytes: pool.peak_bytes,
            allocated_bytes: pool.allocated_bytes,
            retired_bytes: pool.retired_bytes,
            submitted: submissions,
            scheduled: scheduler.submitted,
            coalesced: scheduler.coalesced,
            stale_publications: scheduler.stale_publications,
            validation_errors,
            cancellation_requests: scheduler.cancellation_requests,
            cancelled_pass_batches: extra.0,
            cache_evictions: extra.1,
            viewport_reserved_bytes: extra.2,
            readback_reserved_bytes: extra.3,
            queue_submit_ns: extra.4,
            pyramid_cache_hits: extra.5,
            ibl_convolution_ns: extra.6, // T-067
            cache_hits,
            cache_misses,
        }
    }
    /// Diagnostic barrier: invalidate result leases while retaining resident sources.
    /// The acknowledgement is emitted on the GPU worker before any subsequent work.
    pub fn invalidate_result_caches(&self) -> Result<crossbeam_channel::Receiver<()>, GpuError> {
        let (send, receive) = bounded(1);
        let mut state = self.shared.lock().unwrap();
        let busy = state.scheduler.diagnostics();
        if state.scheduler.closed()
            || busy.in_flight != 0
            || busy.pending != 0
            || state.export.request.is_some()
            || state.clear_results.is_some()
        {
            return Err(GpuError(
                "cache invalidation requires an idle worker".into(),
            ));
        }
        state.clear_results = Some(send);
        let _ = self.wake.try_send(());
        Ok(receive)
    }
    /// Closes mailboxes without waiting for a fence or joining a thread.
    pub fn close(&self) {
        let (frame, upload, material, preview) = {
            let mut state = self.shared.lock().unwrap();
            state.scheduler.close();
            state.export.cancel();
            state.material_identity = None;
            state.retained_materials.clear();
            (
                state.frame.take(),
                state.upload.take(),
                state.material.take(),
                state.preview.take(),
            )
        };
        drop(frame);
        drop(upload);
        drop(material);
        drop(preview);
        let _ = self.wake.try_send(());
    }
    /// Call only outside the UI update loop, after `close`.
    pub fn join(&self) -> Result<(), GpuError> {
        self.close();
        if let Some(thread) = self.thread.lock().unwrap().take() {
            thread
                .join()
                .map_err(|_| GpuError("GPU worker panicked".into()))?;
        }
        Ok(())
    }
}
impl Drop for GpuWorker {
    fn drop(&mut self) {
        self.close();
    }
}
