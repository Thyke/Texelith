use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    display::{AllocationStats, DISPLAY_BUDGET, DisplayBridge, DisplayLease, SharedGpu},
    startup::Config,
};

#[derive(Default)]
struct Progress {
    material_captures: crate::material_ui::snapshots::Captures,
    rendered: AtomicU64,
    completed: AtomicU64,
    errors: Mutex<Vec<String>>,
    screenshot: Mutex<Option<Arc<egui::ColorImage>>>,
    resized_screenshot: Mutex<Option<Arc<egui::ColorImage>>>,
    integration: Mutex<Option<serde_json::Value>>,
}

impl Progress {
    fn error(&self, error: impl ToString) {
        let message = error.to_string();
        eprintln!("GPU diagnostic error: {message}");
        let mut errors = self.errors.lock().unwrap();
        if errors.len() < 64 && !errors.contains(&message) {
            errors.push(message);
        }
    }
}

/// Runs only nonblocking polls on a named worker; never polls from App::ui.
struct PollWorker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl PollWorker {
    fn spawn(gpu: SharedGpu, progress: Arc<Progress>) -> Result<Self, std::io::Error> {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("texelith-gpu-completion".into())
            .spawn(move || {
                while !worker_stop.load(Ordering::Acquire) {
                    if let Err(error) = gpu.device.poll(wgpu::PollType::Poll) {
                        progress.error(error);
                        break;
                    }
                    thread::sleep(Duration::from_millis(2));
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for PollWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct FrameCompletion {
    lease: Arc<DisplayLease>,
    progress: Arc<Progress>,
    painted: Arc<AtomicBool>,
    target: u64,
}

impl egui_wgpu::CallbackTrait for FrameCompletion {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let lease = Arc::clone(&self.lease);
        let progress = Arc::clone(&self.progress);
        let painted = Arc::clone(&self.painted);
        // wgpu 30 installs this action AFTER submitting the encoder's command buffer.
        // The same encoder contains egui's texture draw, so this is a real last-use fence.
        encoder.on_submitted_work_done(move || {
            if painted.load(Ordering::Acquire) {
                progress.completed.fetch_add(1, Ordering::AcqRel);
            }
            drop(lease);
        });
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        // This single root viewport is painted only by eframe's UI thread.
        if self.progress.rendered.load(Ordering::Acquire) < self.target {
            self.progress.rendered.fetch_add(1, Ordering::AcqRel);
            self.painted.store(true, Ordering::Release);
        }
    }
}

struct Shell {
    studio_snapshots: crate::workspace::snapshots::Driver,
    export_close_prompt: bool,
    export_close_after: bool,
    input_snapshots: crate::input_locale::snapshots::Driver,
    packing_snapshots: crate::packing_editor::snapshots::Driver,
    projects_snapshots: crate::project_batch::snapshots::Driver,
    viewport_snapshots: crate::compare::snapshots::Driver,
    material_snapshots: crate::material_ui::snapshots::Driver,
    perf: Option<crate::perf_runtime::Shared>,
    perf_driver: Option<crate::perf_runtime::Driver>,
    integration: Option<crate::integration::Integration>,
    controls: crate::controls::Controls,
    bridge: DisplayBridge,
    config: Config,
    progress: Arc<Progress>,
    label: String,
    started: Instant,
    last_frame: Option<u64>,
    screenshot_requested: bool,
    resize_requested: bool,
    resized_screenshot_requested: bool,
    snapshot_palette_started: Option<Instant>,
    viewport_initialized: bool,
    snapshot_scale_ready_at: Option<Instant>,
    _worker: PollWorker,
}

impl eframe::App for Shell {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(integration) = &mut self.integration {
            integration.poll_exports(&mut self.controls);
        }
        if ctx.input(|i| i.viewport().close_requested()) && self.controls.export_busy() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.export_close_prompt = !self.export_close_after;
        }
        if self.export_close_after && !self.controls.export_busy() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if self.controls.export_busy() {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
        if let Some(perf) = &self.perf {
            let r = perf.lock().unwrap();
            if let Some(monitor) = &r.measurement {
                monitor.logic_tick();
                if monitor.closing()
                    || self.perf_driver.as_ref().is_some_and(|d| d.done)
                    || self.integration.as_ref().is_some_and(|i| i.error.is_some())
                {
                    monitor.phase("shutdown_started");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            // eframe can omit ui() for hidden windows. Logic and watchdog wakes
            // must keep close handling alive independently of painting.
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.export_close_prompt {
            use crate::input_locale::labels::text;
            egui::Window::new(text("export-close-title"))
                .collapsible(false)
                .show(ui, |ui| {
                    ui.label(text("export-close-body"));
                    if ui.button(text("export-cancel-close")).clicked() {
                        self.controls.cancel_material_export();
                        self.export_close_after = true;
                        self.export_close_prompt = false;
                    }
                    if ui.button(text("export-finish-close")).clicked() {
                        self.export_close_after = true;
                        self.export_close_prompt = false;
                    }
                    if ui.button(text("export-keep-working")).clicked() {
                        self.export_close_prompt = false;
                    }
                });
        }
        let update_started = self.perf.as_ref().map(|_| Instant::now());
        let ctx = ui.ctx().clone();
        if let Some(perf) = &self.perf {
            if let Some(monitor) = &perf.lock().unwrap().measurement {
                monitor.phase("first_ui_tick");
            }
            let timed_out =
                self.started.elapsed() > Duration::from_secs(self.config.duration_seconds + 90);
            if timed_out
                || self.perf_driver.as_ref().is_some_and(|d| d.done)
                || self.integration.as_ref().is_some_and(|i| i.error.is_some())
            {
                if timed_out {
                    self.progress.error("performance scenario timed out");
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            let start = Instant::now();
            let mut r = perf.lock().unwrap();
            r.begin_frame(ctx.cumulative_frame_nr(), update_started.unwrap());
            r.overhead(start);
        }
        if self.config.ui_snapshot.is_some() {
            if self.snapshot_scale_ready_at.is_none() {
                ctx.set_pixels_per_point(self.config.scale.parse().unwrap());
                // Surface reconfiguration must finish before the shared compute
                // worker starts submitting. This is snapshot setup, not measured startup.
                self.snapshot_scale_ready_at = Some(Instant::now() + Duration::from_millis(200));
            }
            if self.started.elapsed() > Duration::from_secs(60) {
                self.progress.error("UI snapshot timed out");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            if self.config.studio_snapshots {
                let settled = self.integration.as_ref().is_none_or(|i| {
                    i.material.as_ref().is_none_or(|m| {
                        if crate::compare::active(&self.controls) {
                            m.viewport.settled() && !m.viewport.needs_frame(&self.controls)
                        } else {
                            m.full_visible()
                        }
                    })
                });
                if self.studio_snapshots.tick(
                    &ctx,
                    &mut self.controls,
                    self.config.input.is_some(),
                    settled,
                    self.config.studio_snapshot_size,
                    &self.progress.material_captures,
                ) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
            } else {
                if self.progress.screenshot.lock().unwrap().is_some()
                    && self.snapshot_palette_started.is_none()
                {
                    self.controls.open_palette();
                    self.snapshot_palette_started = Some(Instant::now());
                }
                if self.progress.resized_screenshot.lock().unwrap().is_some() {
                    let done = if self.config.viewport {
                        if let Some(material) =
                            self.integration.as_ref().and_then(|i| i.material.as_ref())
                        {
                            self.viewport_snapshots.tick(
                                &ctx,
                                &mut self.controls,
                                material.viewport.publications,
                                material.viewport.settled(),
                                &self.progress.material_captures,
                            )
                        } else {
                            false
                        }
                    } else if self.config.projects_snapshots {
                        self.projects_snapshots.tick(
                            &ctx,
                            &mut self.controls,
                            &self.progress.material_captures,
                        )
                    } else if self.config.packing_snapshots {
                        self.packing_snapshots.tick(
                            &ctx,
                            &mut self.controls,
                            &self.progress.material_captures,
                        )
                    } else if self.config.input_snapshots {
                        self.input_snapshots.tick(
                            &ctx,
                            &mut self.controls,
                            &self.progress.material_captures,
                        )
                    } else if !self.config.viewport
                        && self.config.input.is_some()
                        && let Some(material) =
                            self.integration.as_ref().and_then(|i| i.material.as_ref())
                    {
                        self.material_snapshots.tick(
                            &ctx,
                            &mut self.controls,
                            material.last_id,
                            material.full_visible(),
                            &self.progress.material_captures,
                        )
                    } else {
                        true
                    };
                    if done {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        return;
                    }
                }
                if !self.screenshot_requested
                    && ctx.cumulative_frame_nr() >= 10
                    && (self.config.input.is_none()
                        || self.integration.as_ref().is_none_or(|i| i.full_visible()))
                    && self
                        .integration
                        .as_ref()
                        .and_then(|i| i.material.as_ref())
                        .is_none_or(|_| self.controls.map_thumbnails_current())
                {
                    self.screenshot_requested = true;
                    let progress = Arc::clone(&self.progress);
                    ctx.request_screenshot(move |image| {
                        *progress.screenshot.lock().unwrap() = Some(image);
                    });
                }
                if !self.resized_screenshot_requested
                    && self
                        .snapshot_palette_started
                        .is_some_and(|started| started.elapsed() >= Duration::from_secs(1))
                {
                    self.resized_screenshot_requested = true;
                    let progress = Arc::clone(&self.progress);
                    ctx.request_screenshot(move |image| {
                        *progress.resized_screenshot.lock().unwrap() = Some(image);
                    });
                }
            }
        }
        let rendered = self.progress.rendered.load(Ordering::Acquire);
        let completed = self.progress.completed.load(Ordering::Acquire);
        if self.config.render_smoke && self.started.elapsed() > Duration::from_secs(60) {
            self.progress
                .error("smoke timed out before all requested frames completed");
        }
        let error = self.progress.errors.lock().unwrap().first().cloned();
        if self.config.render_smoke && (completed >= self.config.frames || error.is_some()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        self.controls.workspace.gpu.clone_from(&self.label);
        let frame_nr = ctx.cumulative_frame_nr();
        if self.config.render_smoke
            && !self.config.replace_texture_every_frame
            && self.config.frames >= 60
            && rendered >= self.config.frames / 2
        {
            if !self.resize_requested {
                self.resize_requested = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(800.0, 560.0)));
            }
            if self.config.report.is_some()
                && !self.resized_screenshot_requested
                && rendered >= self.config.frames / 2 + 10
            {
                self.resized_screenshot_requested = true;
                let progress = Arc::clone(&self.progress);
                ctx.request_screenshot(move |image| {
                    *progress.resized_screenshot.lock().unwrap() = Some(image);
                });
            }
        }
        if self.config.render_smoke
            && self.config.report.is_some()
            && !self.screenshot_requested
            && rendered >= self.config.frames.min(10).saturating_sub(1)
        {
            self.screenshot_requested = true;
            let progress = Arc::clone(&self.progress);
            ctx.request_screenshot(move |image| {
                *progress.screenshot.lock().unwrap() = Some(image);
            });
        }
        let needs_texture = self.integration.is_none()
            && (self.bridge.texture_id().is_none()
                || (self.config.replace_texture_every_frame && rendered < self.config.frames));
        // egui may repeat a UI pass; never unregister a texture from an earlier pass
        // of the same frame or allocate more than once for that frame.
        if self.last_frame != Some(frame_nr) && needs_texture {
            self.last_frame = Some(frame_nr);
            if let Err(error) = self.bridge.replace_diagnostic() {
                self.progress.error(error);
            }
        }

        // Consume a completed result against the still-current Session before either
        // input path advances it. GpuWorker::request clears its completed mailbox;
        // waiting until after the next input/request can starve the viewport forever.
        // The once-per-frame guard also prevents unregistering an earlier egui pass.
        if self.last_frame != Some(frame_nr)
            && let Some(integration) = &mut self.integration
        {
            integration.receive_completed(&self.controls);
        }
        if self.config.viewport && !self.viewport_initialized {
            self.viewport_initialized =
                crate::compare::snapshots::initialize(&self.config, &mut self.controls);
        }
        if !self.config.render_smoke {
            if self.controls.material_mode {
                self.controls.show_workspace_with(
                    ui,
                    crate::controls::Controls::show_procedural_inspector,
                    crate::controls::Controls::show_simple_export,
                );
            } else {
                self.controls.show(ui);
            }
        }
        if let Some(integration) = &mut self.integration {
            // Native registration happens once before this frame's image shapes.
            if self.last_frame != Some(frame_nr) {
                self.last_frame = Some(frame_nr);
                if self.config.integration_smoke {
                    integration.smoke(
                        &mut self.controls,
                        self.config.save_normal.as_ref().unwrap(),
                    );
                }
                if let (Some(driver), Some(perf)) = (&mut self.perf_driver, &self.perf) {
                    driver.tick(&mut self.controls, integration.settled(), perf, &ctx);
                }
                if self
                    .snapshot_scale_ready_at
                    .is_none_or(|ready| Instant::now() >= ready)
                {
                    integration.tick(&mut self.controls);
                }
                if let Some(perf) = &self.perf {
                    let start = Instant::now();
                    let d = integration.gpu.diagnostics();
                    let mut r = perf.lock().unwrap();
                    r.sample_diagnostics(d);
                    r.overhead(start);
                }
            }
            if let Some(error) = &integration.error
                && (self.config.integration_smoke || self.config.ui_snapshot.is_some())
            {
                self.progress.error(error);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            if self.config.integration_smoke && integration.smoke_done {
                if !self.screenshot_requested {
                    self.screenshot_requested = true;
                    let progress = self.progress.clone();
                    ctx.request_screenshot(move |image| {
                        *progress.screenshot.lock().unwrap() = Some(image)
                    });
                } else if self.progress.screenshot.lock().unwrap().is_some() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        let center = crate::workspace::center_slot(ui, |ui| {
            if self.controls.material_mode {
                self.controls.viewport_controls(ui);
            }
            if let Some(integration) = &mut self.integration {
                integration.show(ui, &mut self.controls);
                return;
            }
            ui.label(&self.label);
            ui.label("Shared eframe device + queue | Rgba8Unorm | linear sampler | FIFO");
            ui.separator();
            if let Some(error) = error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
                return;
            }
            ui.label("Grayscale codes: black to white (no extra transfer)");
            ui.label("Normals below: -X to +Z to +X; green +Y top / -Y bottom");
            let available = ui.available_size() - egui::vec2(0.0, 65.0);
            let width = available.x.min(available.y.max(1.0) * 2.0).max(1.0);
            if let (Some(id), Some(lease)) = (self.bridge.texture_id(), self.bridge.lease()) {
                let response = ui.image((id, egui::vec2(width, width / 2.0)));
                let target = if self.config.render_smoke {
                    self.config.frames
                } else {
                    u64::MAX
                };
                if rendered < target {
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                        response.rect,
                        FrameCompletion {
                            lease,
                            progress: Arc::clone(&self.progress),
                            painted: Arc::new(AtomicBool::new(false)),
                            target,
                        },
                    ));
                }
            }
            let stats = self.bridge.accounting().lock().unwrap().clone();
            ui.label(format!(
                "Rendered {rendered} | GPU completed {completed}{}",
                if self.config.render_smoke {
                    format!(" / {}", self.config.frames)
                } else {
                    String::new()
                }
            ));
            ui.label(format!(
                "Immutable display allocations: {} live / {} peak | {:.1} MiB / 32 MiB budget",
                stats.live,
                stats.peak_live,
                stats.live_bytes as f64 / 1048576.0
            ));
        });
        self.controls.workspace.layout.center = Some(center.response.rect);
        if self.config.perf_scenario {
            ctx.request_repaint_after(Duration::from_nanos(1_000_000_000 / 60));
        } else if self.config.render_smoke || self.config.ui_snapshot.is_some() {
            ctx.request_repaint();
        } else if self.integration.as_ref().is_some_and(|integration| {
            !integration
                .material
                .as_ref()
                .is_some_and(|material| material.viewport_can_sleep(&self.controls))
        }) {
            ctx.request_repaint_after(Duration::from_millis(8));
        }
        if let Some(perf) = &self.perf {
            perf.lock().unwrap().end_frame();
        }
    }
}

impl Drop for Shell {
    fn drop(&mut self) {
        if let Some(perf) = &self.perf
            && let Some(monitor) = &perf.lock().unwrap().measurement
        {
            monitor.phase("shutdown_started");
            monitor.phase("shell_drop_started");
        }
        if let Some(integration) = &self.integration {
            *self.progress.integration.lock().unwrap() = Some(integration.report());
            if let Some(error) = &integration.error {
                self.progress.error(error);
            }
        }
    }
}

#[derive(Default)]
struct Runtime {
    gpu: Option<SharedGpu>,
    accounting: Option<Arc<Mutex<AllocationStats>>>,
    adapter: Option<serde_json::Value>,
    compute: Option<Arc<texelith_gpu::GpuWorker>>,
    io: Option<Arc<crate::io_worker::IoWorker>>,
}

pub fn run(config: Config) -> Result<(), String> {
    if config.mvp_smoke {
        if config.projects_smoke || config.workflow_smoke {
            return crate::project_batch::smoke::run(&config);
        }
        if config.batch_smoke {
            return crate::batch_jobs::smoke::run(&config);
        }
        if config.packing_smoke {
            return crate::packing_editor::smoke::run(&config);
        }
        if config.export_smoke {
            return crate::export_jobs::smoke::run(&config);
        }
        if config.viewport {
            return crate::compare::smoke::run(&config);
        }
        if config.input_smoke {
            return crate::input_locale::smoke::run(&config);
        }
        if config.inspectors_smoke {
            return crate::inspectors::smoke::run(&config);
        }
        if config.simple_layout_smoke {
            return crate::workspace::smoke(&config);
        }
        return crate::material_ui::smoke::run(&config);
    }
    if config.perf_scenario {
        let watchdog = crate::measurement::Watchdog::start(
            config
                .report
                .as_deref()
                .ok_or("missing measurement report")?,
            config.duration_seconds,
        )?;
        // Keep supervision alive through all run_inner locals' destructors,
        // including eframe/device destruction and worker joins.
        let result = run_inner(config, Some(watchdog.monitor.clone()));
        let lifecycle = watchdog.monitor.snapshot();
        watchdog.finish()?;
        if !lifecycle.final_reasons().is_empty() {
            return Err(format!(
                "ENVIRONMENT_INVALID: {:?}",
                lifecycle.final_reasons()
            ));
        }
        if let Some(reason) = lifecycle.stop_reason {
            return Err(reason);
        }
        result
    } else {
        run_inner(config, None)
    }
}

fn run_inner(
    config: Config,
    measurement: Option<crate::measurement::Shared>,
) -> Result<(), String> {
    crate::material_ui::labels::set_turkish(config.locale == "tr-TR");
    let started = Instant::now();
    let perf = config
        .perf_scenario
        .then(|| crate::perf_runtime::Recorder::new(started));
    if let Some(perf) = &perf {
        perf.lock().unwrap().measurement = measurement.clone();
    }
    let creator_perf = perf.clone();
    let mut options = config.native_options_measured(measurement.clone())?;
    if config.ui_snapshot.is_some() {
        // Preserve the same logical workspace when capturing at different scales.
        let scale: f32 = config.scale.parse().map_err(|_| "invalid snapshot scale")?;
        options.viewport = options
            .viewport
            .with_inner_size([1280.0 * scale, 800.0 * scale]);
    }
    let progress = Arc::new(Progress::default());
    let runtime = Arc::new(Mutex::new(Runtime::default()));
    let creator_progress = Arc::clone(&progress);
    let creator_runtime = Arc::clone(&runtime);
    let app_config = config.clone();
    let creator_measurement = measurement.clone();
    if let Some(monitor) = &measurement {
        monitor.phase("native_run_started");
    }
    let result = eframe::run_native(
        "Texelith",
        options,
        Box::new(move |cc| {
            if let Some(monitor) = &creator_measurement {
                monitor.phase("app_created");
                monitor.attach(&cc.egui_ctx)?;
                // Request painting even during eframe's initially hidden bootstrap.
                cc.egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::Visible(true));
                cc.egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
                cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            let state = cc
                .wgpu_render_state
                .clone()
                .ok_or("eframe did not provide a wgpu render state")?;
            crate::startup::validate_adapter(&state.adapter)?;
            let info = state.adapter.get_info();
            let label = format!(
                "{} | {:?} | {} {}",
                info.name, info.backend, info.driver, info.driver_info
            );
            eprintln!(
                "Actual adapter: {label}\nAdapter limits: {:?}\nDevice limits: {:?}",
                state.adapter.limits(),
                state.device.limits()
            );
            let bridge = DisplayBridge::new(state.clone());
            let gpu = bridge.shared_gpu();
            let errors = Arc::clone(&creator_progress);
            gpu.device
                .on_uncaptured_error(Arc::new(move |error| errors.error(error)));
            let errors = Arc::clone(&creator_progress);
            gpu.device.set_device_lost_callback(move |reason, message| {
                if reason != wgpu::DeviceLostReason::Destroyed {
                    errors.error(format!("device lost: {reason:?}: {message}"));
                }
            });
            crate::workspace::studio::theme(&cc.egui_ctx);
            let worker = PollWorker::spawn(gpu.clone(), Arc::clone(&creator_progress))?;
            let integrated = !app_config.render_smoke && !app_config.command_smoke;
            let compute = if integrated {
                Some(Arc::new(texelith_gpu::GpuWorker::spawn(
                    gpu.device.clone(),
                    gpu.queue.clone(),
                    Default::default(),
                )?))
            } else {
                None
            };
            let io = if integrated {
                Some(Arc::new(
                    if app_config.perf_scenario && app_config.input.is_none() {
                        crate::io_worker::IoWorker::spawn_with_fixture(true)?
                    } else {
                        crate::io_worker::IoWorker::spawn()?
                    },
                ))
            } else {
                None
            };
            if let Some(monitor) = &creator_measurement {
                monitor.phase("workers_spawned");
            }
            let mut integration = compute.as_ref().zip(io.as_ref()).map(|(compute, io)| {
                crate::integration::Integration::new(
                    state.clone(),
                    compute.clone(),
                    io.clone(),
                    app_config.preview_height,
                )
            });
            if let Some(i) = &mut integration {
                i.perf = creator_perf.clone();
            }
            let material_mode =
                integrated && !app_config.integration_smoke && !app_config.perf_scenario;
            if material_mode && let Some(i) = &mut integration {
                i.material = Some(crate::material_ui::MaterialIntegration::new(
                    i.gpu.clone(),
                    state.device.clone().into(),
                    state.queue.clone().into(),
                )?);
            }
            let mut controls = crate::controls::Controls::new(
                !integrated && (app_config.command_smoke || app_config.ui_snapshot.is_some()),
            );
            if material_mode {
                controls = crate::controls::Controls::new_material();
            }
            controls.workspace.debug_layers = app_config.debug_layers;
            controls.enable_native_dialogs();
            if app_config.ui_snapshot.is_some() && !app_config.studio_snapshots {
                controls.set_inspector_advanced(true);
            }
            if let Some(path) = &app_config.input {
                controls.open_image(path.clone());
            } else if app_config.perf_scenario {
                controls.open_image("generated-faz0-fixture".into());
            }
            if app_config.integration_smoke || app_config.perf_scenario {
                controls.use_scripted_input();
            }
            *creator_runtime.lock().unwrap() = Runtime {
                gpu: Some(gpu),
                compute,
                io,
                accounting: Some(bridge.accounting()),
                adapter: Some(serde_json::json!({
                    "name": info.name, "backend": format!("{:?}", info.backend),
                    "driver": info.driver, "driver_info": info.driver_info,
                    "vendor": info.vendor, "device": info.device,
                    "device_type": format!("{:?}", info.device_type),
                    "adapter_limits": format!("{:?}", state.adapter.limits()),
                    "device_limits": format!("{:?}", state.device.limits()),
                    "adapter_features": format!("{:?}", state.adapter.features()),
                    "enabled_features": format!("{:?}", state.device.features()),
                    "timestamp_queries_supported": state.adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                    "display_format": "Rgba8Unorm", "surface_format": format!("{:?}", state.target_format),
                    "r32float_support": format!("{:?}", state.adapter.get_texture_format_features(wgpu::TextureFormat::R32Float)),
                    "rgba16float_support": format!("{:?}", state.adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float)),
                })),
            };
            Ok(Box::new(Shell {
                studio_snapshots: Default::default(),
                export_close_prompt: false,
                export_close_after: false,
                input_snapshots: Default::default(),
                packing_snapshots: Default::default(),
                projects_snapshots: Default::default(),
                viewport_snapshots: Default::default(),
                material_snapshots: Default::default(),
                perf: creator_perf,
                perf_driver: app_config
                    .perf_scenario
                    .then(|| crate::perf_runtime::Driver::new(app_config.duration_seconds)),
                controls,
                integration,
                bridge,
                config: app_config,
                progress: creator_progress,
                label,
                started,
                last_frame: None,
                screenshot_requested: false,
                resize_requested: false,
                resized_screenshot_requested: false,
                snapshot_palette_started: None,
                viewport_initialized: false,
                snapshot_scale_ready_at: None,
                _worker: worker,
            }))
        }),
    );

    // Shutdown only, outside App::ui: drain final presentation callbacks before reporting.
    if let Some(monitor) = &measurement {
        monitor.phase("shutdown_started");
        monitor.phase("native_run_returned");
    }
    let runtime = runtime.lock().unwrap();
    if let Some(monitor) = &measurement {
        monitor.phase("worker_joins_started");
    }
    if let Some(worker) = &runtime.compute
        && let Err(error) = worker.join()
    {
        progress.error(error);
    }
    if let Some(worker) = &runtime.io
        && let Err(error) = worker.join()
    {
        progress.error(error);
    }
    if let Some(monitor) = &measurement {
        monitor.phase("worker_joins_completed");
    }
    if let Some(gpu) = &runtime.gpu
        && let Err(error) = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
    {
        progress.error(error);
    }
    if let Err(error) = result {
        progress.error(error);
    }
    if let Some(worker) = &runtime.compute {
        // Completion callbacks can themselves retire the last display owner.
        // Drain those retirement callbacks outside the UI loop too.
        if let Some(gpu) = &runtime.gpu {
            for _ in 0..3 {
                if let Err(error) = gpu.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                }) {
                    progress.error(error);
                    break;
                }
            }
        }
        let diagnostics = worker.diagnostics();
        if diagnostics.validation_errors != 0 || diagnostics.allocated_bytes != 0 {
            progress.error(format!(
                "shutdown: {} validation errors, {} GPU bytes retained",
                diagnostics.validation_errors, diagnostics.allocated_bytes
            ));
        }
    }
    if let Some(perf) = &perf {
        // Error callbacks must remain able to acquire Progress while measuring.
        let invalid = measurement
            .as_ref()
            .is_some_and(|m| !m.snapshot().final_reasons().is_empty() || m.closing());
        let benchmark_error = if invalid {
            Some("measurement stopped; normal benchmark not run".into())
        } else {
            if let Some(monitor) = &measurement {
                monitor.phase("normal_benchmark_started");
            }
            runtime.gpu.as_ref().map_or_else(
                || Some("shared device unavailable".to_owned()),
                |gpu| crate::perf_normal::measure(gpu, perf).err(),
            )
        };
        if let Some(monitor) = &measurement {
            monitor.phase("report_started");
        }
        let errors = progress.errors.lock().unwrap().clone();
        return crate::perf_runtime::finish(
            perf,
            benchmark_error,
            runtime.adapter.as_ref(),
            &config,
            &errors,
        );
    }
    if config.integration_smoke
        && !progress
            .integration
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|r| r["smoke_done"] == true)
    {
        progress.error("integration smoke did not complete full preview / PNG save");
    }
    let rendered = progress.rendered.load(Ordering::Acquire);
    let completed = progress.completed.load(Ordering::Acquire);
    if config.render_smoke && (rendered != config.frames || completed != config.frames) {
        progress.error(format!(
            "expected {} rendered/completed diagnostic frames, got {rendered}/{completed}",
            config.frames
        ));
    }
    let accounting = runtime
        .accounting
        .as_ref()
        .map(|stats| stats.lock().unwrap().clone())
        .unwrap_or_default();
    if accounting.live != 0 {
        progress.error(format!(
            "{} display allocations remain at shutdown",
            accounting.live
        ));
    }
    let screenshot_path = config
        .report
        .as_ref()
        .map(|path| path.with_extension(if config.input.is_some() { "png" } else { "bmp" }));
    if let (Some(path), Some(image)) = (
        &screenshot_path,
        progress.screenshot.lock().unwrap().as_ref(),
    ) {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        if let Err(error) = write_screenshot(path, image) {
            progress.error(error);
        }
    }
    let resized_screenshot_path = config.report.as_ref().map(|path| {
        path.with_extension(if config.input.is_some() {
            "resized.png"
        } else {
            "resized.bmp"
        })
    });
    if let (Some(path), Some(image)) = (
        &resized_screenshot_path,
        progress.resized_screenshot.lock().unwrap().as_ref(),
    ) && let Err(error) = write_screenshot(path, image)
    {
        progress.error(error);
    }
    if let Some(dir) = &config.ui_snapshot {
        std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
        for (name, captured) in [
            ("controls", &progress.screenshot),
            ("palette", &progress.resized_screenshot),
        ] {
            match captured.lock().unwrap().as_ref() {
                Some(image) => {
                    let extension = "png";
                    write_evidence_screenshot(
                        &dir.join(format!("{name}-{}.{extension}", config.scale)),
                        image,
                    )
                    .map_err(|error| error.to_string())?
                }
                None if config.studio_snapshots => {}
                None => progress.error(format!("missing {name} UI snapshot")),
            }
        }
    }
    if let Some(dir) = &config.ui_snapshot {
        for (name, image) in progress.material_captures.lock().unwrap().iter() {
            write_evidence_screenshot(&dir.join(format!("{name}-{}.png", config.scale)), image)
                .map_err(|e| e.to_string())?;
        }
    }
    let errors = progress.errors.lock().unwrap().clone();
    let report = serde_json::json!({
        "task": if config.studio_snapshots { "T-147" } else if config.projects_snapshots { "T-088" } else if config.packing_snapshots { "T-083" } else if config.viewport { "T-069" } else if config.input_snapshots { "T-073" } else if config.ui_snapshot.is_some() { "T-072" } else if config.input.is_some() && !config.integration_smoke { "T-061" } else if config.input.is_some() || config.integration_smoke { "T-019" } else { "T-013" }, "passed": errors.is_empty(),
        "integration": progress.integration.lock().unwrap().clone(),
        "gpu_allocated_bytes_after_shutdown": runtime.compute.as_ref().map(|w| w.diagnostics().allocated_bytes),
        "requested_backend": config.backend, "requested_adapter": config.adapter_name,
        "requested_frames": config.frames, "render_smoke": config.render_smoke,
        "replace_texture_every_frame": config.replace_texture_every_frame,
        "rendered_frames": rendered, "gpu_completed_frames": completed,
        "frame_boundary": "egui render-pass callback, acknowledged by the same command encoder's GPU completion; excludes compositor/scanout",
        "elapsed_seconds": started.elapsed().as_secs_f64(), "adapter": runtime.adapter,
        "allocations": accounting, "display_budget_bytes": DISPLAY_BUDGET,
        "surface_and_eframe_allocations": "outside application display accounting",
        "retirement": if runtime.compute.is_some() {
            "T-017 immutable owning display leases; unregister native ID before future frames; each encoder completion retains a clone through its draw; last owner retires through queue completion; pool ledger checked at shutdown"
        } else {
            "immutable diagnostic allocations; unregister old native ID before future frames; encoder completion retains lease through last referencing submission; no pooling or frame-delay reuse"
        },
        "validation_errors": errors,
        "screenshot": screenshot_path.filter(|_| progress.screenshot.lock().unwrap().is_some()),
        "resized_screenshot": resized_screenshot_path.filter(|_| progress.resized_screenshot.lock().unwrap().is_some()),
    });
    if let Some(path) = &config.report {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        eprintln!("Report: {}", path.display());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

// Capture the composited surface as PNG (imports) or the existing diagnostic BMP.
// Encoding and filesystem I/O run after the window closes, outside UI/GPU callbacks.
fn write_screenshot(path: &std::path::Path, image: &egui::ColorImage) -> std::io::Result<()> {
    use std::io::Write as _;
    if path.extension().is_some_and(|extension| extension == "png") {
        let pixels: Vec<u8> = image
            .pixels
            .iter()
            .flat_map(|pixel| [pixel.r(), pixel.g(), pixel.b()])
            .collect();
        return texelith_io::write_normal_png(
            path,
            image.size[0] as u32,
            image.size[1] as u32,
            &pixels,
            true,
        )
        .map_err(std::io::Error::other);
    }
    let width = image.size[0] as u32;
    let height = image.size[1] as u32;
    let row_bytes = (width * 3).next_multiple_of(4);
    let pixel_bytes = row_bytes * height;
    let mut bytes = Vec::with_capacity((54 + pixel_bytes) as usize);
    bytes.extend_from_slice(b"BM");
    bytes.extend_from_slice(&(54 + pixel_bytes).to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&54_u32.to_le_bytes());
    bytes.extend_from_slice(&40_u32.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&24_u16.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&pixel_bytes.to_le_bytes());
    bytes.extend_from_slice(&[0; 16]);
    for y in (0..height as usize).rev() {
        for x in 0..width as usize {
            let pixel = image.pixels[y * width as usize + x];
            bytes.extend_from_slice(&[pixel.b(), pixel.g(), pixel.r()]);
        }
        bytes.resize(bytes.len() + (row_bytes - width * 3) as usize, 0);
    }
    std::fs::File::create(path)?.write_all(&bytes)
}

fn write_evidence_screenshot(
    path: &std::path::Path,
    image: &egui::ColorImage,
) -> Result<(), String> {
    // Preserve the full native capture as an ignored artifact. Shared evidence stays small.
    let task = path
        .ancestors()
        .filter_map(|p| p.file_name())
        .find(|s| s.to_str().is_some_and(|s| s.starts_with("T-")))
        .unwrap_or_else(|| std::ffi::OsStr::new("T-072"));
    let raw = std::path::Path::new("target/evidence")
        .join(task)
        .join("native-captures")
        .join(
            path.parent()
                .and_then(|p| p.file_name())
                .unwrap_or_default(),
        )
        .join(path.file_name().ok_or("Screenshot filename missing")?);
    if let Some(parent) = raw.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_screenshot(&raw, image).map_err(|e| e.to_string())?;
    let width = image.size[0].min(1152);
    let height = (image.size[1] * width / image.size[0]).max(1);
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(
                image.pixels
                    [(y * image.size[1] / height) * image.size[0] + x * image.size[0] / width],
            );
        }
    }
    write_screenshot(path, &egui::ColorImage::new([width, height], pixels))
        .map_err(|e| e.to_string())
}
