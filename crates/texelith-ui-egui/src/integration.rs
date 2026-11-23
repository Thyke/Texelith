use crate::{
    command_adapter::UiAction,
    controls::Controls,
    io_worker::{IoRequest, IoResult, IoWorker},
    routing::RenderRouting,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use texelith_core::{
    CommandRequest, Effect, ImportToken, MapKind, NodeId, ParameterAddress, ParameterId,
    ParameterValue,
};
use texelith_gpu::{CompletedFrame, GpuWorker, Quality, RenderRequest, SourceUpload, WorkerEvent};

struct Registered {
    id: egui::TextureId,
    frame: CompletedFrame,
}

struct DisplayCompletion {
    lease: texelith_gpu::DisplayLease,
    request_id: u64,
    painted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
}
impl egui_wgpu::CallbackTrait for DisplayCompletion {
    fn prepare(
        &self,
        _: &wgpu::Device,
        _: &wgpu::Queue,
        _: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let lease = self.lease.clone();
        let completed = self.completed.clone();
        let id = self.request_id;
        // Deferred action runs after submission of the encoder containing egui's
        // texture draw. Its lease cannot return to the pool before that fence.
        encoder.on_submitted_work_done(move || {
            completed.fetch_max(id, Ordering::AcqRel);
            lease.retire_after_submission();
        });
        Vec::new()
    }
    fn paint(
        &self,
        _: egui::PaintCallbackInfo,
        _: &mut wgpu::RenderPass<'static>,
        _: &egui_wgpu::CallbackResources,
    ) {
        self.painted.store(self.request_id, Ordering::Release);
    }
}

struct PendingSave {
    id: u64,
    request: RenderRequest,
    path: PathBuf,
    map: MapKind,
}

pub(crate) struct Integration {
    batch: Option<crate::batch_jobs::BatchJobs>,
    exports: Option<crate::export_jobs::ExportJobs>,
    pub(crate) material: Option<crate::material_ui::MaterialIntegration>,
    pub perf: Option<crate::perf_runtime::Shared>,
    pub gpu: Arc<GpuWorker>,
    pub io: Arc<IoWorker>,
    state: egui_wgpu::RenderState,
    routing: RenderRouting,
    current: Option<Registered>,
    seen_revision: u64,
    latest_import: Option<ImportToken>,
    pending_load: Option<IoRequest>,
    decoded: Option<(ImportToken, SourceUpload)>,
    // Keep the immutable source allocation shared with upload requests.
    source: Option<SourceUpload>,
    source_ready: bool,
    uploading: bool,
    save: Option<PendingSave>,
    pub status: String,
    load_status: String,
    pub save_status: String,
    pub error: Option<String>,
    pub painted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    stale_discarded: u64,
    publications: Vec<serde_json::Value>,
    smoke_stage: u8,
    smoke_started: Instant,
    pub smoke_done: bool,
    saved: Option<serde_json::Value>,
    preview_height: bool,
}
impl Integration {
    pub fn new(
        state: egui_wgpu::RenderState,
        gpu: Arc<GpuWorker>,
        io: Arc<IoWorker>,
        preview_height: bool,
    ) -> Self {
        let exports = crate::export_jobs::ExportJobs::spawn(gpu.clone());
        let export_error = exports.as_ref().err().cloned();
        let batch = crate::batch_jobs::BatchJobs::spawn(gpu.clone()).ok();
        Self {
            batch,
            exports: exports.ok(),
            material: None,
            perf: None,
            gpu,
            io,
            state,
            routing: RenderRouting::default(),
            current: None,
            seen_revision: 0,
            latest_import: None,
            pending_load: None,
            decoded: None,
            source: None,
            source_ready: false,
            uploading: false,
            save: None,
            status: "Drop a PNG/JPEG or enter its path to open an image".into(),
            load_status: String::new(),
            save_status: String::new(),
            error: export_error,
            painted: Arc::default(),
            completed: Arc::default(),
            stale_discarded: 0,
            publications: Vec::new(),
            smoke_stage: 0,
            smoke_started: Instant::now(),
            smoke_done: false,
            saved: None,
            preview_height,
        }
    }
    pub fn poll_exports(&mut self, controls: &mut Controls) {
        if controls.packing.worker.is_none()
            && let Err(error) = controls.start_preset_io(None)
        {
            controls.packing.feedback = error;
        }
        controls.poll_preset_io();
        if let Some(exports) = &mut self.exports {
            if let Some(batch) = &mut self.batch {
                batch.tick(controls, exports);
            }
            exports.tick(controls);
        }
    }
    pub fn tick(&mut self, controls: &mut Controls) {
        controls.poll_project_io();
        if let Some(upload) = controls.take_project_upload()
            && let Err(error) = self.gpu.install_typed_source(upload)
        {
            controls.fail_project_upload(error.to_string());
        }
        if let Some(path) = controls.projects.batch_recover.take()
            && let Some(batch) = &mut self.batch
        {
            if let Err(error) = batch.recover(path) {
                controls.projects.feedback = error;
            }
            controls.projects.restart_offer = true;
        }
        self.poll_exports(controls);
        if let Some(material) = &mut self.material {
            material.tick(controls);
            self.error = material.error.clone();
            return;
        }
        if self.error.is_some() {
            return;
        }
        let generation = controls.session().snapshot().map_or(0, |s| s.generation);
        self.io.generation.store(generation, Ordering::Release);
        let mut saves = Vec::new();
        for effect in controls.take_effects() {
            match effect {
                Effect::Export { .. }
                | Effect::SaveProject { .. }
                | Effect::CancelExport { .. }
                | Effect::SaveExportPreset { .. } => {}
                Effect::InspectSeams { .. } => {}
                Effect::OpenProject { .. } => {}
                Effect::LoadImage { token, source } => {
                    if let Some(path) = controls.take_path(source) {
                        self.latest_import = Some(token);
                        self.decoded = None;
                        self.pending_load = Some(IoRequest::Load {
                            token,
                            path,
                            interpretation: controls.interpretation(),
                        });
                        self.load_status = "Loading image… previous result retained".into();
                    }
                }
                Effect::SavePng {
                    destination,
                    map,
                    output,
                    generation,
                } => {
                    if let Some(path) = controls.take_path(destination) {
                        saves.push((path, map, output, generation));
                    }
                }
            }
        }
        while let Ok(result) = self.io.results.try_recv() {
            match result {
                IoResult::Loaded { token, result } if Some(token) == self.latest_import => {
                    match result {
                        Ok(upload) => self.decoded = Some((token, upload)),
                        Err(error) => {
                            controls.fail_import(token);
                            self.latest_import = None;
                            self.load_status =
                                format!("Load failed: {error}. Previous material retained.");
                        }
                    }
                }
                IoResult::Loaded { .. } => {}
                IoResult::Saved {
                    generation,
                    extent,
                    kind,
                    result,
                } => {
                    self.save = None;
                    match result {
                        Ok(()) => {
                            self.saved = Some(
                                serde_json::json!({ "generation": generation, "width": extent.width, "height": extent.height, "kind": format!("{kind:?}") }),
                            );
                            self.save_status =
                                format!("Saved {} × {} {kind:?} PNG", extent.width, extent.height);
                        }
                        Err(error) => self.save_status = error,
                    }
                }
            }
        }
        if let Some(request) = self.pending_load.take() {
            match self.io.requests.try_send(request) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(request)) => {
                    self.pending_load = Some(request)
                }
                Err(_) => self.error = Some("I/O worker closed".into()),
            }
        }
        while let Some(event) = self.gpu.try_event() {
            match event {
                WorkerEvent::SourceReady { content_hash } => {
                    self.uploading = false;
                    self.source_ready = self
                        .source
                        .as_ref()
                        .is_some_and(|s| s.descriptor.content_hash == content_hash);
                    if self.source_ready && self.load_status.starts_with("Preparing") {
                        self.load_status.clear();
                    }
                }
                other => self.error = Some(format!("GPU: {other:?}")),
            }
        }
        if self.error.is_some() {
            self.gpu.close();
            return;
        }
        if !self.uploading
            && let Some((token, upload)) = self.decoded.take()
        {
            match self.gpu.install_source(upload.clone()) {
                Ok(()) => match controls.complete_import(token, upload.descriptor.clone()) {
                    Ok(()) => {
                        if self.preview_height {
                            controls.action(UiAction::Request(CommandRequest::SelectOutput {
                                output: NodeId(3),
                            }));
                        }
                        self.source = Some(upload);
                        self.source_ready = false;
                        self.uploading = true;
                        self.latest_import = None;
                        self.load_status =
                            "Preparing source on GPU… previous result retained".into();
                        self.io.generation.store(
                            controls.session().snapshot().unwrap().generation,
                            Ordering::Release,
                        );
                    }
                    Err(error) => self.error = Some(error),
                },
                Err(error) => self.error = Some(error.to_string()),
            }
        }
        if self.source_ready
            && self.seen_revision != controls.render_revision
            && let Some(snapshot) = controls.session().snapshot()
        {
            let output = snapshot.graph.selected_output();
            match self
                .routing
                .issue(snapshot, output, controls.render_quality)
            {
                Ok(request) => {
                    if let Some(perf) = &self.perf {
                        let start = Instant::now();
                        let mut r = perf.lock().unwrap();
                        r.request(&request);
                        r.overhead(start);
                    }
                    match self.gpu.request(request) {
                        Ok(()) => {
                            self.seen_revision = controls.render_revision;
                            self.status = format!(
                                "Computing {:?}… previous result retained",
                                controls.render_quality
                            );
                        }
                        Err(error) => self.error = Some(error.to_string()),
                    }
                }
                Err(error) => self.error = Some(error),
            }
        }
        for (path, map, output, generation) in saves {
            if self.save.is_some() {
                self.save_status = "Save busy: wait for the current PNG".into();
                continue;
            }
            let Some(snapshot) = controls.session().snapshot() else {
                continue;
            };
            if !self.source_ready || snapshot.generation != generation {
                self.save_status = "Save cancelled: source is loading or document changed".into();
                continue;
            }
            let Some(source) = &self.source else { continue };
            let request = RenderRequest {
                request_id: self.routing.latest.as_ref().map_or(0, |r| r.request_id),
                snapshot: Arc::new(snapshot),
                output,
                extent: source.descriptor.extent,
                quality: Quality::Full,
            };
            match self.gpu.request_save(request.clone()) {
                Ok(id) => {
                    self.save = Some(PendingSave {
                        id,
                        request,
                        path,
                        map,
                    });
                    self.save_status = "Saving full-resolution numeric map…".into();
                }
                Err(error) => self.save_status = format!("Save: {error}"),
            }
        }
        if let Some(result) = self.gpu.try_save() {
            match result {
                Ok(readback) => {
                    let valid = self.save.as_ref().is_some_and(|s| {
                        s.id == readback.save_id
                            && s.request.request_id == readback.request_id
                            && s.request.output == readback.output
                            && s.request.extent == readback.extent
                            && s.map == readback.kind
                            && s.request.snapshot.generation == readback.generation
                    }) && self.io.generation.load(Ordering::Acquire)
                        == readback.generation;
                    if valid {
                        let save = self.save.as_mut().unwrap();
                        match self.io.requests.try_send(IoRequest::Save {
                            path: save.path.clone(),
                            readback,
                        }) {
                            Ok(()) => {}
                            Err(_) => {
                                self.save = None;
                                self.save_status = "Save busy: I/O queue occupied; retry".into();
                            }
                        }
                    } else {
                        self.save = None;
                        self.save_status = "Save cancelled: document changed".into();
                    }
                }
                Err(error) => {
                    self.save = None;
                    self.save_status = format!("Save: {error}");
                }
            }
        }
        self.receive_completed(controls);
    }
    pub(crate) fn receive_completed(&mut self, controls: &Controls) {
        if self.material.is_some() || self.error.is_some() {
            return;
        }
        if let Some(frame) = self.gpu.try_latest() {
            if let Some(perf) = &self.perf {
                let start = Instant::now();
                let mut r = perf.lock().unwrap();
                r.publication(&frame);
                r.overhead(start);
            }
            // Repeat the entire freshness check at the native registration boundary.
            let fresh = self.source_ready
                && self.routing.accepts(&frame)
                && self.seen_revision == controls.render_revision
                && controls.session().snapshot().is_some_and(|s| {
                    s.generation == frame.generation && s.graph.selected_output() == frame.output
                });
            if fresh {
                let id = self.state.renderer.write().register_native_texture(
                    &self.state.device,
                    frame.lease.view(),
                    wgpu::FilterMode::Linear,
                );
                self.status = format!(
                    "{:?} · {} × {} · request {} · generation {}",
                    frame.quality,
                    frame.extent.width,
                    frame.extent.height,
                    frame.request_id,
                    frame.generation
                );
                self.publications.push(serde_json::json!({ "request_id": frame.request_id, "generation": frame.generation,
                    "output": frame.output.0, "quality": format!("{:?}", frame.quality), "width": frame.extent.width, "height": frame.extent.height }));
                // Keep diagnostic history bounded during long interactive sessions.
                if self.publications.len() > 64 {
                    self.publications.remove(0);
                }
                if let Some(old) = self.current.replace(Registered { id, frame }) {
                    self.state.renderer.write().free_texture(&old.id);
                    // Every previously submitted draw also owns a callback lease.
                    old.frame.lease.retire_after_submission();
                }
            } else {
                self.stale_discarded += 1;
            }
        }
    }
    pub fn show(&mut self, ui: &mut egui::Ui, controls: &mut Controls) {
        if let Some(material) = &mut self.material {
            if crate::compare::active(controls) {
                material.viewport.show(ui, controls, &self.state);
            } else {
                material.show(ui, controls);
            }
            return;
        }
        ui.label(&self.status);
        if !self.load_status.is_empty() {
            ui.label(&self.load_status);
        }
        if !self.save_status.is_empty() {
            ui.label(&self.save_status);
        }
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        if let Some(display) = &self.current {
            let extent = display.frame.extent;
            let available = ui.available_size().max(egui::vec2(1.0, 1.0));
            let scale = (available.x / extent.width as f32).min(available.y / extent.height as f32);
            let size = egui::vec2(extent.width as f32 * scale, extent.height as f32 * scale);
            let response = ui.image((display.id, size));
            self.perf_fence(ui, response.rect, Some(display.frame.request_id));
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                response.rect,
                DisplayCompletion {
                    lease: display.frame.lease.clone(),
                    request_id: display.frame.request_id,
                    painted: self.painted.clone(),
                    completed: self.completed.clone(),
                },
            ));
        } else {
            self.perf_fence(ui, ui.min_rect(), None);
        }
    }
    fn perf_fence(&self, ui: &egui::Ui, rect: egui::Rect, id: Option<u64>) {
        if let Some(recorder) = &self.perf {
            let index = recorder.lock().unwrap().frame_index;
            if let Some(index) = index {
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    crate::perf_runtime::FrameFence {
                        recorder: recorder.clone(),
                        index,
                        id,
                        painted: Arc::default(),
                    },
                ));
            }
        }
    }
    pub fn full_visible(&self) -> bool {
        if let Some(material) = &self.material {
            return if material.viewport.engaged() {
                material.viewport.settled()
            } else {
                material.full_visible()
            };
        }
        self.current.as_ref().is_some_and(|d| {
            d.frame.quality == Quality::Full
                && self.routing.accepts(&d.frame)
                && self.painted.load(Ordering::Acquire) == d.frame.request_id
        })
    }
    pub(crate) fn settled(&self) -> bool {
        self.current.as_ref().is_some_and(|d| {
            self.routing.accepts(&d.frame)
                && self.painted.load(Ordering::Acquire) == d.frame.request_id
                && self.completed.load(Ordering::Acquire) >= d.frame.request_id
        })
    }
    pub fn smoke(&mut self, controls: &mut Controls, destination: &std::path::Path) {
        if self.smoke_started.elapsed() > Duration::from_secs(90) {
            self.error = Some("Integration smoke timed out".into());
            return;
        }
        if !self.settled() || self.seen_revision != controls.render_revision {
            return;
        }
        let address = ParameterAddress {
            node: NodeId(4),
            parameter: ParameterId::Strength,
        };
        let change = |value| UiAction::Change {
            address,
            value: ParameterValue::Float(value),
        };
        match self.smoke_stage {
            0 => {
                controls.action(UiAction::DragBegin { address });
                controls.action(change(3.0));
            }
            1 => {
                if self.current.as_ref().unwrap().frame.quality != Quality::Proxy {
                    self.error = Some("missing proxy publication".into());
                    return;
                }
                // Release must advance request_id even with no further parameter change.
                controls.action(UiAction::DragRelease);
            }
            2 => {
                if !self.full_visible() || controls.session().history().undo_entries != 1 {
                    self.error = Some("full release/history contract failed".into());
                    return;
                }
                controls.action(UiAction::UndoShortcut);
            }
            3 => {
                if controls
                    .session()
                    .snapshot()
                    .unwrap()
                    .graph
                    .node(NodeId(4))
                    .unwrap()
                    .params
                    .parameter(ParameterId::Strength)
                    != Ok(1.0)
                {
                    self.error = Some("undo did not restore strength".into());
                    return;
                }
                controls.action(UiAction::RedoShortcut);
            }
            4 => {
                if !self.full_visible()
                    || controls.session().history().undo_entries != 1
                    || controls.session().history().redo_entries != 0
                    || controls
                        .session()
                        .snapshot()
                        .unwrap()
                        .graph
                        .node(NodeId(4))
                        .unwrap()
                        .params
                        .parameter(ParameterId::Strength)
                        != Ok(3.0)
                {
                    self.error = Some("redo/full publication failed".into());
                    return;
                }
                controls.save_png(destination.to_path_buf(), MapKind::Normal);
            }
            5 => {
                if self.saved.is_none() {
                    if self.save.is_none() && !self.save_status.is_empty() {
                        self.error = Some(self.save_status.clone());
                    }
                    return;
                }
                let saved = self.saved.as_ref().unwrap();
                let source = self.source.as_ref().unwrap().descriptor.extent;
                if saved["width"] != source.width
                    || saved["height"] != source.height
                    || saved["kind"] != "Normal"
                    || saved["generation"] != controls.session().snapshot().unwrap().generation
                    || !self.full_visible()
                {
                    self.error =
                        Some("saved PNG is stale, incomplete, or not full resolution".into());
                    return;
                }
                self.smoke_done = true;
                return;
            }
            _ => return,
        }
        self.smoke_stage += 1;
    }
    pub fn report(&self) -> serde_json::Value {
        if let Some(material) = &self.material {
            return material.report();
        }
        let d = self.gpu.diagnostics();
        serde_json::json!({ "smoke_done": self.smoke_done, "smoke_stage": self.smoke_stage,
            "publications": self.publications, "stale_discarded_before_registration": self.stale_discarded,
            "painted_request": self.painted.load(Ordering::Acquire), "completed_request": self.completed.load(Ordering::Acquire),
            "saved": self.saved, "status": self.status, "load_status": self.load_status, "save_status": self.save_status, "error": self.error,
            "source": self.source.as_ref().map(|s| &s.descriptor),
            "peak_bytes": d.peak_bytes, "peak_in_flight": d.peak_in_flight, "peak_pending": d.peak_pending,
            "validation_errors": d.validation_errors, "cache_hits": d.cache_hits, "cache_misses": d.cache_misses })
    }
}
impl Drop for Integration {
    fn drop(&mut self) {
        if let Some(display) = self.current.take() {
            self.state.renderer.write().free_texture(&display.id);
            display.frame.lease.retire_after_submission();
        }
        self.io.generation.store(u64::MAX, Ordering::Release);
        self.gpu.close();
        self.io.close();
    }
}
