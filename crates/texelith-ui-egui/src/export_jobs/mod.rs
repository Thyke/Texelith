//! Material-set orchestration. Frame callbacks only enqueue intents and read progress.
mod encode;
#[cfg(test)]
mod regressions;
pub mod smoke;
mod worker;

use crossbeam_channel::{Sender, bounded};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use texelith_core::*;
use texelith_gpu::{ExportStage, GpuWorker, JobId};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FormatChoice {
    #[default]
    Preset,
    FloatDisplacement,
}

#[derive(Clone, Debug)]
pub struct ExportOptions {
    pub preset: String,
    pub size: SizePolicy,
    pub format: FormatChoice,
    pub destination: PathBuf,
    pub material_name: String,
    /// Set only by an explicit confirmation for this captured request.
    pub replace_existing: bool,
}
impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            preset: "unreal-v1".into(),
            size: SizePolicy::Keep,
            format: FormatChoice::Preset,
            destination: crate::controls::default_export_folder("Material"),
            material_name: "Material".into(),
            replace_existing: false,
        }
    }
}

pub fn build_plan(graph: &Graph, options: &ExportOptions) -> Result<ExportPlan, String> {
    let preset = BuiltinExportPresets
        .find(&options.preset, 1)
        .ok_or("Unknown export preset")?;
    // Binding validates DDS block dimensions; apply the explicit size policy first.
    let mut preset = preset.clone();
    preset.size = options.size;
    let mut plan = preset.bind(graph, &[]).map_err(|e| e.to_string())?;
    if options.format == FormatChoice::FloatDisplacement {
        let reference = *graph
            .material_outputs()
            .and_then(|m| m.get(&MapKind::Height))
            .ok_or("Height unavailable")?;
        plan.outputs.retain(|o| {
            !o.channels.iter().any(
                |c| matches!(c.source, ChannelSource::Map(r, _) if r.port == OutputPort::Height),
            )
        });
        plan.outputs.push(ExportOutput {
            name: "Displacement".into(),
            format: ExportFormat::Exr32,
            channels: vec![ChannelSpec {
                source: ChannelSource::Map(reference, Component::R),
                invert: false,
                transfer: Transfer::Linear,
            }],
            normal_has_negative_z: false,
            quantization: None,
            import_hints: None,
        });
    }
    validate_export_for_graph(&plan, graph).map_err(|e| e.to_string())?;
    Ok(plan)
}

#[derive(Clone, Debug)]
pub struct JobStatus {
    pub job_id: u64,
    pub stage: ExportStage,
    pub output_name: String,
    pub completed_items: usize,
    pub total_items: usize,
    pub destination: PathBuf,
    pub error: Option<String>,
    pub cancel_requested: bool,
}
impl JobStatus {
    pub fn terminal(&self) -> bool {
        matches!(
            self.stage,
            ExportStage::Completed | ExportStage::Cancelled | ExportStage::Failed
        )
    }
}

#[derive(Default)]
struct Shared {
    status: Option<JobStatus>,
    busy: bool,
    cancel: Arc<AtomicBool>,
}
struct Job {
    request: texelith_gpu::ExportRequest,
    options: ExportOptions,
    cancel: Arc<AtomicBool>,
}

/// One bounded IO worker, one accepted set, one charged readback plane at a time.
pub struct ExportJobs {
    send: Sender<Job>,
    shared: Arc<Mutex<Shared>>,
    stop: Arc<AtomicBool>,
    next: u64,
}
impl ExportJobs {
    pub fn spawn(gpu: Arc<GpuWorker>) -> Result<Self, String> {
        let (send, receive) = bounded::<Job>(1);
        let shared = Arc::new(Mutex::new(Shared::default()));
        let state = shared.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        std::thread::Builder::new()
            .name("texelith-material-export-io".into())
            .spawn(move || {
                while !stopped.load(Ordering::Acquire) {
                    match receive.recv_timeout(std::time::Duration::from_millis(20)) {
                        Ok(job) => worker::run(&gpu, &state, &stopped, job),
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            send,
            shared,
            stop,
            next: 0,
        })
    }
    pub fn busy(&self) -> bool {
        self.shared.lock().unwrap().busy
    }
    pub fn status(&self) -> Option<JobStatus> {
        self.shared.lock().unwrap().status.clone()
    }

    pub fn start(&mut self, effect: Effect, options: ExportOptions) -> Result<u64, String> {
        let Effect::Export { plan, capture } = effect else {
            return Err("Expected export effect".into());
        };
        let extent = match plan.size {
            SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
                Extent { width, height }
            }
            _ => plan.source_extent,
        };
        for output in &plan.outputs {
            crate::batch_jobs::export_phase_reservation(
                extent.width,
                extent.height,
                output.channels.len() as u64,
                output.format,
            )?;
        }
        let mut state = self.shared.lock().unwrap();
        if state.busy {
            return Err("An export is already running".into());
        }
        self.next = self.next.checked_add(1).ok_or("Export IDs exhausted")?;
        let request = texelith_gpu::ExportRequest::new(JobId(self.next), *capture, plan)
            .map_err(|e| e.to_string())?;
        let cancel = Arc::new(AtomicBool::new(false));
        let status = JobStatus {
            job_id: self.next,
            stage: ExportStage::Evaluating,
            output_name: request.plan().outputs[0].name.clone(),
            completed_items: 0,
            total_items: request.plan().outputs.len(),
            destination: options.destination.clone(),
            error: None,
            cancel_requested: false,
        };
        crate::export_perf::mark(crate::export_perf::Event::Accepted);
        self.send
            .try_send(Job {
                request,
                options,
                cancel: cancel.clone(),
            })
            .map_err(|e| e.to_string())?;
        state.busy = true;
        state.cancel = cancel;
        state.status = Some(status);
        Ok(self.next)
    }
    pub fn cancel(&self, job_id: u64) {
        let mut state = self.shared.lock().unwrap();
        if state.busy && state.status.as_ref().is_some_and(|s| s.job_id == job_id) {
            state.cancel.store(true, Ordering::Release);
            if let Some(status) = &mut state.status {
                status.cancel_requested = true;
            }
        }
    }
    pub fn tick(&mut self, controls: &mut crate::controls::Controls) {
        for (effect, options) in controls.take_export_effects() {
            match effect {
                Effect::CancelExport { job_id } => self.cancel(job_id),
                other => {
                    if let Err(error) = self.start(other, options) {
                        controls.export_message(error);
                    }
                }
            }
        }
        let state = self.shared.lock().unwrap();
        controls.set_export_status(state.status.clone(), state.busy);
    }
}
impl Drop for ExportJobs {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.shared
            .lock()
            .unwrap()
            .cancel
            .store(true, Ordering::Release);
    }
}
