//! Bounded batch adapter. All filesystem work is serialized off the frame thread;
//! the existing export coordinator remains the sole consumer of GPU export mailboxes.
mod files;
pub(crate) mod memory;
pub(crate) mod presentation;
#[cfg(test)]
pub(crate) mod regressions;
pub mod smoke;
use crate::{
    command_adapter::UiAction,
    controls::Controls,
    export_jobs::{ExportJobs, ExportOptions},
};
pub use files::{
    BatchSource, InputSelection, Manifest, NativeItem, Prepared, enumerate, recover_manifest,
    save_manifest,
};
pub use memory::presentation_reservation;
use std::{path::PathBuf, sync::Arc};
use texelith_core::*;
use texelith_gpu::{ExportStage, GpuWorker, TypedSourceUpload};

pub const CPU_BUDGET: u64 = 512 * 1024 * 1024;
pub const INTERACTIVE_RESERVATION: u64 = 256 * 1024 * 1024;
/// PNG8 encoding scratch: encoded Vec capacity plus bounded codec workspace.
/// The input plane is in retained headroom; GPU readback has already retired.
pub fn export_reservation(width: u32, height: u32, bytes_per_pixel: u64) -> Result<u64, String> {
    admit_codec(png_scratch(width, height, bytes_per_pixel, 0)?)
}

pub(crate) fn export_phase_reservation(
    width: u32,
    height: u32,
    channels: u64,
    format: ExportFormat,
) -> Result<u64, String> {
    // PNG16 holds native samples, native bytes and the encoder's endian copy.
    // EXR holds one converted f32 plane. PNG8 borrows the existing plane.
    let bpp = channels
        .checked_mul(u64::from(format.bit_depth()))
        .ok_or("Codec size overflow")?
        / 8;
    let bytes = match format {
        ExportFormat::Png16 => png_scratch(width, height, bpp, 3)?,
        ExportFormat::Png8 => png_scratch(width, height, bpp, 0)?,
        ExportFormat::Exr32 | ExportFormat::DdsBc5 => u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(bpp))
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(64 * 1024 * 1024))
            .ok_or("Codec size overflow")?,
    };
    admit_codec(bytes)
}

/// png 0.18 retains zlib bytes while copying to the final Vec, or while
/// retrying incompressible input with stored blocks. Bound both Vec capacities
/// using stored-block size and fdeflate's maximum 12-bit literal (versus 8-bit
/// input), plus 8 MiB for codec/rows. Incompressible input can hold both buffers.
pub(crate) fn png_scratch(width: u32, height: u32, bpp: u64, copies: u64) -> Result<u64, String> {
    let raw = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(bpp))
        .ok_or("Batch CPU reservation overflow")?;
    let filtered = raw
        .checked_add(u64::from(height))
        .ok_or("Codec size overflow")?;
    if bpp == 4 && copies == 0 {
        // RGBA8 uses a fixed destination and one level-1 zlib Vec. Charge its
        // doubled capacity plus workspace; no fdeflate fallback buffer exists.
        let bound = filtered
            .checked_add(filtered.div_ceil(1000))
            .and_then(|n| n.checked_add(65536))
            .ok_or("Codec size overflow")?;
        return bound
            .checked_add(
                bound
                    .checked_next_power_of_two()
                    .ok_or("Codec size overflow")?,
            )
            .and_then(|n| n.checked_add(8 * 1024 * 1024))
            .ok_or("Codec size overflow".into());
    }
    let capacity = filtered
        .checked_add(
            filtered
                .div_ceil(65535)
                .checked_mul(5)
                .ok_or("Codec size overflow")?,
        )
        .and_then(|n| n.checked_add(65536))
        .and_then(u64::checked_next_power_of_two)
        .ok_or("Codec size overflow")?;
    let fast_capacity = filtered
        .checked_mul(3)
        .map(|n| n.div_ceil(2))
        .and_then(|n| n.checked_add(65536))
        .and_then(u64::checked_next_power_of_two)
        .ok_or("Codec size overflow")?;
    raw.checked_mul(copies)
        .and_then(|n| n.checked_add(capacity.checked_add(fast_capacity.max(capacity))?))
        .and_then(|n| n.checked_add(8 * 1024 * 1024))
        .ok_or("Codec size overflow".into())
}
fn admit_codec(bytes: u64) -> Result<u64, String> {
    if bytes > CPU_BUDGET - INTERACTIVE_RESERVATION {
        return Err("Batch codec reservation exceeds shared CPU budget".into());
    }
    Ok(bytes)
}
fn err(e: impl ToString) -> String {
    e.to_string()
}
fn err_debug(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
pub(crate) struct Submission {
    id: u64,
    selection: InputSelection,
    destination: PathBuf,
    manifest: PathBuf,
    capture: ExportCapture,
    plan: ExportPlan,
    continue_on_error: bool,
}
enum Task {
    Prepare(Box<Submission>),
    Recover(PathBuf),
    Save(PathBuf, Box<BatchJob>, Vec<NativeItem>),
    CreateDerived(PathBuf, Box<BatchJob>, Vec<NativeItem>),
    Load(BatchWorkId, PathBuf, Box<ExportCapture>),
}
enum Reply {
    Prepared(Result<Box<Prepared>, String>),
    Saved(Result<(), String>),
    Created(Result<PathBuf, String>),
    Loaded(BatchWorkId, Result<TypedSourceUpload, String>),
}
struct Worker {
    send: crossbeam_channel::Sender<Task>,
    receive: crossbeam_channel::Receiver<Reply>,
    stop: crossbeam_channel::Sender<()>,
}
impl Worker {
    fn spawn() -> Result<Self, String> {
        let (send, receive) = crossbeam_channel::bounded(1);
        let (result, results) = crossbeam_channel::bounded(1);
        let (stop, stopped) = crossbeam_channel::bounded(1);
        std::thread::Builder::new().name("texelith-batch-io".into()).spawn(move || loop {
            let task = crossbeam_channel::select! { recv(stopped) -> _ => break, recv(receive) -> r => match r { Ok(t) => t, Err(_) => break } };
            let reply = match task {
                    Task::Prepare(s) => Reply::Prepared(files::prepare(*s).map(Box::new)),
                    Task::Recover(p) => Reply::Prepared(recover_manifest(&p).map(Box::new)),
                Task::Save(p,j,n) => Reply::Saved(save_manifest(&p, &j, &n)),
                Task::CreateDerived(p,j,n) => Reply::Created(files::create_derived_manifest(&p, &j, &n)),
                Task::Load(w,p,c) => Reply::Loaded(w, files::decode(&p, &c)),
            };
            crossbeam_channel::select! { recv(stopped) -> _ => break, send(result, reply) -> r => if r.is_err() { break } }
        }).map_err(err)?;
        Ok(Self {
            send,
            receive: results,
            stop,
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
    }
}

#[derive(Clone, Debug, Default)]
pub struct Status {
    pub busy: bool,
    pub error: Option<String>,
    pub peak_active: usize,
    pub peak_prefetch: usize,
    pub cpu_reserved_bytes: u64,
}
pub struct BatchJobs {
    io: Worker,
    gpu: Arc<GpuWorker>,
    native: Vec<NativeItem>,
    manifest: Option<PathBuf>,
    manifest_base: Option<PathBuf>,
    pending_io: bool,
    dirty: bool,
    upload: Option<(BatchWorkId, TypedSourceUpload)>,
    export: Option<(BatchWorkId, u64)>,
    seen_job: Option<u64>,
    cancel_requested: bool,
    pub status: Status,
}
impl BatchJobs {
    pub fn spawn(gpu: Arc<GpuWorker>) -> Result<Self, String> {
        Ok(Self {
            io: Worker::spawn()?,
            gpu,
            native: Vec::new(),
            manifest: None,
            manifest_base: None,
            pending_io: false,
            dirty: false,
            upload: None,
            export: None,
            seen_job: None,
            cancel_requested: false,
            status: Status::default(),
        })
    }
    pub fn recover(&mut self, path: PathBuf) -> Result<(), String> {
        if self.status.busy || self.pending_io {
            return Err("Batch coordinator busy".into());
        }
        self.io.send.try_send(Task::Recover(path)).map_err(err)?;
        self.pending_io = true;
        self.status.busy = true;
        self.status.cpu_reserved_bytes = CPU_BUDGET;
        Ok(())
    }
    pub fn manifest_path(&self) -> Option<&std::path::Path> {
        self.manifest.as_deref()
    }
    fn send(&mut self, task: Task) -> Result<(), String> {
        self.io.send.try_send(task).map_err(err)?;
        self.pending_io = true;
        self.status.busy = true;
        Ok(())
    }
    pub fn tick(&mut self, c: &mut Controls, exports: &mut ExportJobs) {
        if let Err(error) = self.advance(c, exports) {
            self.status.error = Some(error);
            if let Some((_, id)) = self.export {
                exports.cancel(id);
            }
            if self.status.busy {
                self.cancel_requested = true;
            }
        }
        self.status.cpu_reserved_bytes = memory::peak_bytes();
        c.batch_status = self.status.clone();
    }
    fn advance(&mut self, c: &mut Controls, exports: &mut ExportJobs) -> Result<(), String> {
        self.dirty |= std::mem::take(&mut c.batch_dirty);
        self.cancel_requested |= std::mem::take(&mut c.batch_cancel_requested);
        if !self.status.busy
            && let Some(submission) = c.batch_submission.take()
        {
            self.cancel_requested = false;
            self.status = Status {
                busy: true,
                cpu_reserved_bytes: CPU_BUDGET,
                ..Default::default()
            };
            self.send(Task::Prepare(Box::new(submission)))?;
        }
        if let Ok(reply) = self.io.receive.try_recv() {
            self.pending_io = false;
            match reply {
                Reply::Prepared(result) => {
                    let prepared = match result {
                        Ok(p) => p,
                        Err(e) => {
                            self.status.busy = false;
                            return Err(e);
                        }
                    };
                    self.native = prepared.native;
                    self.manifest_base = Some(prepared.manifest.clone());
                    self.manifest = Some(prepared.manifest);
                    self.seen_job = Some(prepared.job.id());
                    c.session.set_batch(prepared.job).map_err(err_debug)?;
                    // Recovery only offers explicit resume; it never starts work.
                    self.status.busy = c.session.batch().is_some_and(BatchJob::can_start);
                }
                Reply::Saved(result) => result?,
                Reply::Created(result) => match result {
                    Ok(path) => self.manifest = Some(path),
                    Err(error) => {
                        self.status.busy = false;
                        self.dirty = false;
                        return Err(error);
                    }
                },
                Reply::Loaded(work, result) => {
                    let job = c.session.batch().ok_or("Missing batch")?;
                    let cancelled = job
                        .items()
                        .iter()
                        .any(|i| i.id == work.item && i.state == BatchState::Cancelled);
                    if cancelled {
                        c.session.retire_cancelled_batch(work).map_err(err_debug)?;
                    } else {
                        match result {
                            Ok(upload) => {
                                let item = job
                                    .items()
                                    .iter()
                                    .find(|i| i.id == work.item)
                                    .ok_or("Missing item")?;
                                if item.source_digest != [0; 32]
                                    && upload.descriptor.content_hash != item.source_digest
                                {
                                    c.session
                                        .batch_fail(work, "Source changed since preflight".into())
                                        .map_err(err_debug)?;
                                } else {
                                    self.native
                                        .iter_mut()
                                        .find(|n| n.index == work.item.index)
                                        .ok_or("Missing native item")?
                                        .decoded_hash = Some(upload.descriptor.content_hash);
                                    c.session
                                        .batch_transition(work, BatchState::Evaluating)
                                        .map_err(err_debug)?;
                                    let next = c.session.batch().unwrap().pending_work().unwrap();
                                    self.upload = Some((next, upload));
                                }
                            }
                            Err(error) => c.session.batch_fail(work, error).map_err(err_debug)?,
                        }
                    }
                    self.dirty = true;
                }
            }
        }
        let Some(job) = c.session.batch() else {
            return Ok(());
        };
        if self.seen_job != Some(job.id()) {
            if self.export.is_some() || self.pending_io {
                return Err("Retry before batch retirement".into());
            }
            self.seen_job = Some(job.id());
            let base = self.manifest_base.as_ref().ok_or("Missing manifest base")?;
            let base = base.clone();
            self.manifest = None;
            self.native
                .retain(|n| job.items().iter().any(|i| i.index == n.index));
            self.status.busy = true;
            self.status.error = None;
            self.dirty = false;
            return self.send(Task::CreateDerived(
                base,
                Box::new(job.clone()),
                self.native.clone(),
            ));
        }
        // A failed reservation leaves the replacement job quiescent. A new
        // submission/recovery is required before any compute or save can start.
        if self.manifest.is_none() {
            return Ok(());
        }
        let active = job
            .items()
            .iter()
            .filter(|i| {
                matches!(
                    i.state,
                    BatchState::Loading | BatchState::Evaluating | BatchState::Encoding
                )
            })
            .count();
        self.status.peak_active = self.status.peak_active.max(active);
        if let Some((work, id)) = self.export {
            let cancelled = job
                .items()
                .iter()
                .any(|i| i.id == work.item && i.state == BatchState::Cancelled);
            if cancelled || self.cancel_requested {
                exports.cancel(id);
            }
            if let Some(status) = exports.status().filter(|s| s.job_id == id) {
                if status.terminal() {
                    self.export = None;
                    let current = c
                        .session
                        .batch()
                        .unwrap()
                        .pending_work()
                        .ok_or("Missing export phase")?;
                    if cancelled {
                        c.session
                            .retire_cancelled_batch(current)
                            .map_err(err_debug)?;
                    } else if status.stage == ExportStage::Completed {
                        if c.session
                            .batch()
                            .unwrap()
                            .items()
                            .iter()
                            .any(|i| i.id == work.item && i.state == BatchState::Evaluating)
                        {
                            c.session
                                .batch_transition(current, BatchState::Encoding)
                                .map_err(err_debug)?;
                        }
                        let current = c.session.batch().unwrap().pending_work().unwrap();
                        c.session
                            .batch_transition(current, BatchState::Committed)
                            .map_err(err_debug)?;
                    } else if self.cancel_requested && status.stage == ExportStage::Cancelled {
                        c.session
                            .dispatch(CommandRequest::BatchCancel {})
                            .map_err(err)?;
                        c.session
                            .retire_cancelled_batch(current)
                            .map_err(err_debug)?;
                    } else {
                        c.session
                            .batch_fail(
                                current,
                                status.error.unwrap_or("Batch export failed".into()),
                            )
                            .map_err(err_debug)?;
                    }
                    self.dirty = true;
                } else if status.stage == ExportStage::Encoding
                    && !cancelled
                    && job
                        .items()
                        .iter()
                        .any(|i| i.id == work.item && i.state == BatchState::Evaluating)
                {
                    c.session
                        .batch_transition(work, BatchState::Encoding)
                        .map_err(err_debug)?;
                    self.dirty = true;
                }
            }
        }
        if self.pending_io {
            return Ok(());
        }
        if self.cancel_requested && self.export.is_none() {
            self.cancel_requested = false;
            if c.session.batch().is_some_and(BatchJob::can_cancel) {
                c.session
                    .dispatch(CommandRequest::BatchCancel {})
                    .map_err(err)?;
                self.dirty = true;
            }
        }
        let job = c.session.batch().unwrap();
        // Cancelled decode/upload tokens retire only after their worker result is consumed.
        if job.items().iter().any(|i| i.state == BatchState::Cancelled) {
            if let Some((work, _)) = self.upload.take() {
                c.session.retire_cancelled_batch(work).map_err(err_debug)?;
                self.dirty = true;
            } else if self.export.is_none()
                && let Some(work) = job.pending_work()
            {
                c.session.retire_cancelled_batch(work).map_err(err_debug)?;
                self.dirty = true;
            }
        }
        if self.dirty {
            self.dirty = false;
            return self.send(Task::Save(
                self.manifest.clone().ok_or("Missing manifest")?,
                Box::new(c.session.batch().unwrap().clone()),
                self.native.clone(),
            ));
        }
        if self.export.is_some() {
            return Ok(());
        }
        let job = c.session.batch().unwrap();
        if let Some((work, upload)) = self.upload.take() {
            // Do not replace the GPU's source-ready acknowledgement while an
            // interactive import still owns it. Its decoder has priority too.
            if exports.busy() || c.batch_interactive_import.is_some() {
                self.upload = Some((work, upload));
                return Ok(());
            }
            let instantiated = (|| -> Result<_, String> {
                let graph = instantiate_recipe(&job.recipe().graph, upload.descriptor.clone())
                    .map_err(err)?;
                let mut plan = job.export_settings().clone();
                // Source extent describes this input; the captured size policy is unchanged.
                plan.source_extent = upload.descriptor.extent;
                let capture = ExportCapture::new(
                    &EvaluationSnapshot {
                        generation: work.generation,
                        graph,
                    },
                    job.recipe().document_revision,
                    &plan,
                )
                .map_err(err)?;
                Ok((plan, capture))
            })();
            let (plan, capture) = match instantiated {
                Ok(value) => value,
                Err(error) => {
                    c.session.batch_fail(work, error).map_err(err_debug)?;
                    self.dirty = true;
                    return Ok(());
                }
            };
            for output in &plan.outputs {
                let size = match plan.size {
                    SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
                        Extent { width, height }
                    }
                    _ => upload.descriptor.extent,
                };
                if let Err(error) = export_phase_reservation(
                    size.width,
                    size.height,
                    output.channels.len() as u64,
                    output.format,
                ) {
                    c.session.batch_fail(work, error).map_err(err_debug)?;
                    self.dirty = true;
                    return Ok(());
                }
            }
            if let Err(error) = self.gpu.install_typed_source(upload.clone()) {
                if error.to_string().contains("Busy") {
                    self.upload = Some((work, upload));
                } else {
                    c.session
                        .batch_fail(work, error.to_string())
                        .map_err(err_debug)?;
                    self.dirty = true;
                }
                return Ok(());
            }
            let native = self
                .native
                .iter()
                .find(|n| n.index == work.item.index)
                .ok_or("Missing native source")?;
            let id = exports.start(
                Effect::Export {
                    plan,
                    capture: Box::new(capture),
                },
                ExportOptions {
                    destination: native.destination.clone(),
                    material_name: native.material_name.clone(),
                    ..Default::default()
                },
            )?;
            self.export = Some((work, id));
            return Ok(());
        }
        if job.pending_work().is_none() && job.can_start() {
            c.action(UiAction::Request(CommandRequest::BatchStart {}));
            self.dirty = true;
            return Ok(());
        }
        let job = c.session.batch().unwrap();
        if let Some(work) = job.pending_work() {
            if c.batch_interactive_import.is_some() {
                return Ok(());
            }
            if job
                .items()
                .iter()
                .any(|i| i.id == work.item && i.state == BatchState::Loading)
            {
                let native = self
                    .native
                    .iter()
                    .find(|n| n.index == work.item.index)
                    .ok_or("Missing native source")?;
                return self.send(Task::Load(
                    work,
                    native.source.path.clone(),
                    Box::new(job.recipe().clone()),
                ));
            }
        } else {
            self.status.busy = false;
        }
        Ok(())
    }
}

impl Controls {
    /// Capture through the same command/effect route as a single material export.
    /// Enumeration, hashes and collision checks happen later on the bounded IO worker.
    pub fn queue_batch(
        &mut self,
        selection: InputSelection,
        destination: PathBuf,
        manifest: PathBuf,
        continue_on_error: bool,
    ) -> Result<(), String> {
        if self.batch_status.busy || self.batch_submission.is_some() || self.export_busy() {
            return Err("Batch coordinator busy".into());
        }
        self.action(UiAction::FocusLost);
        let plan = self.packing_plan()?;
        self.session
            .dispatch(CommandRequest::SetExportSettings { settings: plan })
            .map_err(err)?;
        let effect = self
            .session
            .dispatch(CommandRequest::RequestExport {})
            .map_err(err)?
            .effects
            .into_iter()
            .next()
            .ok_or("Missing captured export")?;
        let Effect::Export { plan, capture } = effect else {
            return Err("Expected export capture".into());
        };
        let id = self
            .session
            .batch()
            .map_or(Some(1), |j| j.id().checked_add(1))
            .ok_or("Batch IDs exhausted")?;
        self.batch_submission = Some(Submission {
            id,
            selection,
            destination,
            manifest,
            capture: *capture,
            plan,
            continue_on_error,
        });
        self.batch_status.busy = true;
        Ok(())
    }
    pub fn batch_status(&self) -> &Status {
        &self.batch_status
    }
    pub fn resume_batch(&mut self) -> Result<(), String> {
        let job = self.session.batch().ok_or("Missing recovered batch")?;
        let resumed = job
            .resume_incomplete(job.id().checked_add(1).ok_or("Batch IDs exhausted")?)
            .map_err(err_debug)?;
        self.session.set_batch(resumed).map_err(err_debug)
    }
}
