//! Opt-in bounded recorder and independent input producer; no timed file I/O.
use crate::{command_adapter::UiAction, controls::Controls, perf_metrics::*};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{NodeId, ParameterAddress, ParameterId, ParameterValue};
use texelith_gpu::{CompletedFrame, RenderRequest, WorkerDiagnostics};

const CAP: usize = 20_000;
pub(crate) type Shared = Arc<Mutex<Recorder>>;
pub(crate) struct Recorder {
    pub measurement: Option<crate::measurement::Shared>,
    pub origin: Instant,
    pub trace: RunTrace,
    pub current_event: u64,
    pub frame_index: Option<usize>,
    last_frame: Option<u64>,
    pub overhead_ns: Vec<u64>,
    pub diagnostics: Vec<serde_json::Value>,
    pub observed: Vec<(u64, u64)>,
    pub first_usable_ns: Option<u64>,
}
impl Recorder {
    pub fn new(origin: Instant) -> Shared {
        let mut trace = RunTrace {
            frames: Vec::with_capacity(CAP),
            inputs: Vec::with_capacity(CAP),
            requests: Vec::with_capacity(CAP),
            ..Default::default()
        };
        trace.inputs.push(InputSample {
            event_id: 0,
            scheduled_ns: Some(0),
            occurred_ns: Some(0),
            handled_ns: Some(0),
        });
        Arc::new(Mutex::new(Self {
            measurement: None,
            origin,
            trace,
            current_event: 0,
            frame_index: None,
            last_frame: None,
            overhead_ns: Vec::with_capacity(CAP),
            diagnostics: Vec::with_capacity(CAP),
            observed: Vec::with_capacity(CAP),
            first_usable_ns: None,
        }))
    }
    pub fn ns(&self, at: Instant) -> u64 {
        at.duration_since(self.origin).as_nanos() as u64
    }
    pub fn now(&self) -> u64 {
        self.ns(Instant::now())
    }
    pub fn overhead(&mut self, start: Instant) {
        if self.overhead_ns.len() < CAP {
            self.overhead_ns.push(start.elapsed().as_nanos() as u64);
        } else {
            self.trace.dropped_samples += 1;
        }
    }
    pub fn begin_frame(&mut self, nr: u64, at: Instant) {
        if self.last_frame == Some(nr) {
            return;
        }
        self.last_frame = Some(nr);
        if self.trace.frames.len() == CAP {
            self.trace.dropped_samples += 1;
            self.frame_index = None;
            return;
        }
        self.frame_index = Some(self.trace.frames.len());
        self.trace.frames.push(FrameSample {
            update_start_ns: Some(self.ns(at)),
            update_end_ns: None,
            rendered_ns: None,
            request_id: None,
        });
    }
    pub fn end_frame(&mut self) {
        let now = self.now();
        if let Some(i) = self.frame_index {
            self.trace.frames[i].update_end_ns = Some(now);
        }
    }
    pub fn request(&mut self, r: &RenderRequest) {
        if self.trace.requests.len() == CAP {
            self.trace.dropped_samples += 1;
            return;
        }
        let now = self.now();
        self.trace.requests.push(RequestSample {
            request_id: r.request_id,
            generation: r.snapshot.generation,
            event_id: self.current_event,
            quality: r.quality,
            extent: [r.extent.width, r.extent.height],
            requested_ns: Some(now),
            submitted_ns: None,
            completed_ns: None,
        });
        if self.trace.release_ns.is_some() && r.quality == texelith_gpu::Quality::Full {
            self.trace.full_request_id = Some(r.request_id);
        }
    }
    pub fn publication(&mut self, f: &CompletedFrame) {
        let requested = self.ns(f.requested_at);
        let submitted = self.ns(f.submitted_at);
        let completed = self.ns(f.completed_at);
        if let Some(r) = self
            .trace
            .requests
            .iter_mut()
            .rev()
            .find(|r| r.request_id == f.request_id)
        {
            r.requested_ns = Some(requested);
            r.submitted_ns = Some(submitted);
            r.completed_ns = Some(completed);
        }
        if self.observed.len() < CAP {
            self.observed.push((f.request_id, self.now()));
        } else {
            self.trace.dropped_samples += 1;
        }
    }
    pub fn complete_frame(&mut self, index: usize, id: Option<u64>, at: Instant) {
        if let Some(monitor) = &self.measurement {
            monitor.phase("first_render_completed");
            if id.is_some() {
                monitor.phase("first_usable_frame");
            }
        }
        let ns = self.ns(at);
        if self.trace.frames[index].rendered_ns.is_none() {
            self.trace.frames[index].rendered_ns = Some(ns);
            self.trace.frames[index].request_id = id;
            if id.is_some() && self.first_usable_ns.is_none() {
                self.first_usable_ns = Some(ns);
                self.trace.cold_start_ns = cold_start_ns();
            }
        }
    }
    pub fn sample_diagnostics(&mut self, d: WorkerDiagnostics) {
        self.trace.counters.peak_app_bytes = self.trace.counters.peak_app_bytes.max(d.peak_bytes);
        self.trace.counters.peak_in_flight = d.peak_in_flight;
        self.trace.counters.peak_pending = d.peak_pending;
        self.trace.counters.stale_publications = d.stale_publications;
        self.trace.counters.validation_errors = d.validation_errors;
        if self.diagnostics.len() < CAP {
            self.diagnostics.push(serde_json::json!({"at_ns":self.now(),"in_flight":d.in_flight,"pending":d.pending,
                "allocated_bytes":d.allocated_bytes,"retired_bytes":d.retired_bytes,"peak_bytes":d.peak_bytes,
                "submitted":d.submitted,"scheduled":d.scheduled,"coalesced":d.coalesced,
                "cache_hits":d.cache_hits,"cache_misses":d.cache_misses,"stale_publications":d.stale_publications,"validation_errors":d.validation_errors}));
        } else {
            self.trace.dropped_samples += 1;
        }
    }
}

#[cfg(windows)]
fn cold_start_ns() -> Option<u64> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn QueryPerformanceCounter(value: *mut i64) -> i32;
        fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    }
    let launch: i64 = std::env::var("TEXELITH_PERF_LAUNCH_QPC")
        .ok()?
        .parse()
        .ok()?;
    let (mut now, mut freq) = (0, 0);
    // QPC and .NET Stopwatch share the system monotonic counter on Windows.
    unsafe {
        if QueryPerformanceCounter(&mut now) == 0 || QueryPerformanceFrequency(&mut freq) == 0 {
            return None;
        }
    }
    if now < launch || freq <= 0 {
        return None;
    }
    Some(((now - launch) as u128 * 1_000_000_000 / freq as u128) as u64)
}
#[cfg(not(windows))]
fn cold_start_ns() -> Option<u64> {
    None
}

struct Event {
    id: u64,
    value: Option<f32>,
}
pub(crate) struct Driver {
    phase: u8,
    seconds: u64,
    receiver: Option<crossbeam_channel::Receiver<Event>>,
    producer: Option<std::thread::JoinHandle<()>>,
    pub done: bool,
}
impl Driver {
    pub fn new(seconds: u64) -> Self {
        Self {
            phase: 0,
            seconds,
            receiver: None,
            producer: None,
            done: false,
        }
    }
    pub fn tick(
        &mut self,
        controls: &mut Controls,
        settled: bool,
        recorder: &Shared,
        ctx: &egui::Context,
    ) {
        let address = ParameterAddress {
            node: NodeId(4),
            parameter: ParameterId::Strength,
        };
        if self.phase == 0 && settled {
            controls.action(UiAction::DragBegin { address });
            controls.action(UiAction::Change {
                address,
                value: ParameterValue::Float(0.0),
            });
            self.phase = 1;
        } else if self.phase == 1 && settled {
            let (tx, rx) = crossbeam_channel::bounded(4096);
            self.receiver = Some(rx);
            let recorder = recorder.clone();
            let ctx = ctx.clone();
            let seconds = self.seconds;
            self.producer = Some(
                std::thread::Builder::new()
                    .name("texelith-perf-input".into())
                    .spawn(move || {
                        let start = Instant::now() + Duration::from_millis(100);
                        {
                            let mut r = recorder.lock().unwrap();
                            r.trace.drag_start_ns = Some(r.ns(start));
                            r.trace.drag_end_ns = Some(r.ns(start + Duration::from_secs(seconds)));
                        }
                        let count = seconds * 60;
                        for index in 0..=count {
                            let deadline = start + Duration::from_nanos(index * 1_000_000_000 / 60);
                            while let Some(left) = deadline.checked_duration_since(Instant::now()) {
                                std::thread::sleep(left);
                            }
                            let occurred = Instant::now();
                            let id = index + 1;
                            {
                                let mut r = recorder.lock().unwrap();
                                let input = InputSample {
                                    event_id: id,
                                    scheduled_ns: Some(r.ns(deadline)),
                                    occurred_ns: Some(r.ns(occurred)),
                                    handled_ns: None,
                                };
                                if index == count {
                                    r.trace.release_ns = input.occurred_ns;
                                }
                                if r.trace.inputs.len() < CAP {
                                    r.trace.inputs.push(input);
                                } else {
                                    r.trace.dropped_samples += 1;
                                }
                            }
                            let event = Event {
                                id,
                                value: (index < count)
                                    .then_some(index as f32 * 10.0 / (count - 1) as f32),
                            };
                            if tx.try_send(event).is_err() {
                                recorder.lock().unwrap().trace.dropped_samples += 1;
                                break;
                            }
                            ctx.request_repaint();
                        }
                    })
                    .expect("perf input thread"),
            );
            self.phase = 2;
        }
        if let Some(rx) = &self.receiver {
            for event in rx.try_iter() {
                {
                    let mut r = recorder.lock().unwrap();
                    let now = r.now();
                    if let Some(i) = r
                        .trace
                        .inputs
                        .iter_mut()
                        .rev()
                        .find(|i| i.event_id == event.id)
                    {
                        i.handled_ns = Some(now);
                    }
                    r.current_event = event.id;
                }
                if let Some(value) = event.value {
                    controls.action(UiAction::Change {
                        address,
                        value: ParameterValue::Float(value),
                    });
                } else {
                    controls.action(UiAction::DragRelease);
                    self.phase = 3;
                }
            }
        }
        if self.phase == 3 && settled {
            let mut r = recorder.lock().unwrap();
            if r.trace.full_request_id.is_some_and(|id| {
                r.trace
                    .frames
                    .iter()
                    .any(|f| f.request_id == Some(id) && f.rendered_ns.is_some())
            }) {
                r.trace.completed = true;
                self.done = true;
            }
        }
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        self.receiver.take();
        if let Some(p) = self.producer.take() {
            let _ = p.join();
        }
    }
}

/// Encoder completion is later than paint and submission. No compositor/scanout claim.
pub(crate) struct FrameFence {
    pub recorder: Shared,
    pub index: usize,
    pub id: Option<u64>,
    pub painted: Arc<std::sync::atomic::AtomicBool>,
}
impl egui_wgpu::CallbackTrait for FrameFence {
    fn prepare(
        &self,
        _: &wgpu::Device,
        _: &wgpu::Queue,
        _: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let recorder = self.recorder.clone();
        let painted = self.painted.clone();
        let (index, id) = (self.index, self.id);
        encoder.on_submitted_work_done(move || {
            let at = Instant::now();
            if painted.load(std::sync::atomic::Ordering::Acquire) {
                let mut r = recorder.lock().unwrap();
                r.complete_frame(index, id, at);
                r.overhead(at);
            }
        });
        Vec::new()
    }
    fn paint(
        &self,
        _: egui::PaintCallbackInfo,
        _: &mut wgpu::RenderPass<'static>,
        _: &egui_wgpu::CallbackResources,
    ) {
        self.painted
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

pub(crate) fn finish(
    recorder: &Shared,
    benchmark_error: Option<String>,
    adapter: Option<&serde_json::Value>,
    config: &crate::startup::Config,
    errors: &[String],
) -> Result<(), String> {
    let mut r = recorder.lock().unwrap();
    r.trace.counters.validation_errors += errors.len() as u64;
    if benchmark_error.is_some() {
        r.trace.completed = false;
    }
    let gate = evaluate(&r.trace);
    let lifecycle = r.measurement.as_ref().map(|m| m.snapshot());
    let environment_reasons = lifecycle
        .as_ref()
        .map(|l| l.final_reasons())
        .unwrap_or_default();
    let environment_valid = environment_reasons.is_empty();
    let summary = |samples: &[u64]| serde_json::json!({"count":samples.len(),"p50_ns":nearest_rank(samples,50),"p95_ns":nearest_rank(samples,95),"p99_ns":nearest_rank(samples,99),"max_ns":samples.iter().max()});
    let active = |at: Option<u64>| {
        at.zip(r.trace.drag_start_ns.zip(r.trace.drag_end_ns))
            .is_some_and(|(at, (start, end))| at >= start && at < end)
    };
    let observed: Vec<_> = r
        .observed
        .iter()
        .filter_map(|(id, at)| {
            if !active(Some(*at)) {
                return None;
            }
            let request = r.trace.requests.iter().find(|x| x.request_id == *id)?;
            if request.quality != texelith_gpu::Quality::Proxy || request.event_id == 0 {
                return None;
            }
            let input = r
                .trace
                .inputs
                .iter()
                .find(|x| x.event_id == request.event_id)?;
            at.checked_sub(input.occurred_ns?)
        })
        .collect();
    let requested: Vec<_> = r
        .trace
        .requests
        .iter()
        .filter(|x| active(x.requested_ns) && x.quality == texelith_gpu::Quality::Proxy)
        .filter_map(|x| x.completed_ns?.checked_sub(x.requested_ns?))
        .collect();
    let submitted: Vec<_> = r
        .trace
        .requests
        .iter()
        .filter(|x| active(x.requested_ns) && x.quality == texelith_gpu::Quality::Proxy)
        .filter_map(|x| x.completed_ns?.checked_sub(x.submitted_ns?))
        .collect();
    let lateness: Vec<_> = r
        .trace
        .inputs
        .iter()
        .filter(|x| x.event_id != 0)
        .filter_map(|x| x.occurred_ns?.checked_sub(x.scheduled_ns?))
        .collect();
    let outcomes: Vec<_> = r.trace.requests.iter().map(|x| {
        let rendered = r.trace.frames.iter().any(|f| f.request_id == Some(x.request_id) && f.rendered_ns.is_some());
        serde_json::json!({"request_id":x.request_id,"outcome":if rendered { "rendered" } else if x.completed_ns.is_some() { "observed_not_rendered" } else { "superseded_or_unfinished_worker_reason_unavailable" }})
    }).collect();
    let report = serde_json::json!({"schema":2,"scenario":"normal-drag","passed":gate.passed() && environment_valid,"gate":gate,"trace":r.trace,
        "status":if !environment_valid { "ENVIRONMENT_INVALID" } else if gate.passed() { "PASS" } else { "FAIL" },
        "environment_reasons":environment_reasons,"lifecycle":lifecycle,
        "adapter":adapter,"errors":errors,"benchmark_error":benchmark_error,"profile":if cfg!(debug_assertions) {"debug"} else {"release"},
        "worktree":std::env::current_dir().ok(),"size":[4096,4096],"target_input_hz":60,"target_presentation_hz":60,"present_mode":"Fifo",
        "monitor_refresh_hz":"see script environment.json; standalone not measured","physical_scanout":"not measured",
        "repaint_target":"request_repaint_after 1/60 second plus independent 60 Hz input wakes; actual completed cadence is recorded, extra reactive frames retained",
        "frame_boundary":"GPU completion callback attached to the same eframe command encoder that draws the texture; conservative render/submission completion, excludes compositor and physical scanout",
        "first_usable_since_app_entry_ns":r.first_usable_ns,"cold_start_boundary":"script QPC immediately before process Start to first texture draw GPU completion; missing external QPC means NotMeasured",
        "warmup":"initial full source/levels/blur/normal/display completion, then proxy strength=0 completion before producer starts",
        "normal_benchmark":"same presentation device, after window and interactive pool shutdown; production normal kernel with reusable zero-initialized R32 height and Rgba16 normal; 10 warmups + every one of 100 measured dispatches",
        "app_allocation_scope":"GPU pool staging/scratch/cached/retired resources counted once; separate normal-only resources do not overlap; excludes eframe/surfaces and CPU decoder/recorder allocations",
        "recorder_sample_cap_per_stream":CAP,"instrumentation_overhead":summary(&r.overhead_ns),"raw_overhead_ns":r.overhead_ns,
        "overhead_scope":"recorder frame start, request, publication, diagnostic sampling and frame completion callbacks; driver/egui paint callback costs remain included in CPU update or frame cadence",
        "diagnostics":r.diagnostics,"raw_observations":r.observed,"input_to_observed":summary(&observed),"request_to_completion":summary(&requested),"submission_to_completion":summary(&submitted),
        "worker_latency_scope":"active-window proxy observations only; superseded work has no fabricated completion sample",
        "producer_lateness":summary(&lateness),"request_outcomes":outcomes,
        "outcome_limit":"worker API lacks per-request cancellation events; coalesced is only pending-slot replacement, not a cancellation total",
        "manual_checks":"not measured; T-021 visual/manual follow-up required"});
    let path = config.report.as_ref().ok_or("missing report path")?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "Performance gate: {:?}\nMetrics: {:?}\nReport: {}",
        gate.violations,
        gate.metrics,
        path.display()
    );
    if !environment_valid {
        Err(format!("ENVIRONMENT_INVALID: {environment_reasons:?}"))
    } else if gate.passed() {
        Ok(())
    } else {
        Err(format!(
            "ADR 0003 slider gate failed: {:?}",
            gate.violations
        ))
    }
}
