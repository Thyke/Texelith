use super::{ImportKind, presenter};
use crate::controls::Controls;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use texelith_core::*;
use texelith_gpu::*;

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;

#[path = "../inspectors/runtime.rs"]
mod inspector_runtime;

// Includes all seven cached 128-square RGBA workspace thumbnails.
pub(crate) const PRESENTATION_RESERVATION: u64 = 224 * 1024 * 1024 + 7 * 128 * 128 * 4;
struct PendingUpload {
    token: ImportToken,
    source: SourceDescriptor,
    convention: NormalConvention,
}
enum Work {
    Load {
        token: ImportToken,
        path: PathBuf,
        kind: ImportKind,
        convention: NormalConvention,
    },
    Present {
        bundle: CompletedMaterial,
        map: MapKind,
        heatmap: bool,
        save: Option<PathBuf>,
    },
}
enum Event {
    Loaded {
        token: ImportToken,
        result: Result<TypedSourceUpload, String>,
    },
    Presented {
        id: u64,
        generation: u64,
        quality: Quality,
        extent: Extent,
        map: MapKind,
        result: Result<presenter::Presentation, String>,
    },
}
struct Worker {
    send: Sender<Work>,
    recv: Receiver<Event>,
    stop: Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        generation: Arc<AtomicU64>,
    ) -> Result<Self, String> {
        let (send, receive) = bounded(1);
        let (events, recv) = bounded(1);
        let (stop, stopped) = bounded(1);
        let thread = std::thread::Builder::new()
            .name("texelith-material-presentation".into())
            .spawn(move || {
                loop {
                    let work = crossbeam_channel::select! {
                        recv(stopped) -> _ => break,
                        recv(receive) -> result => match result { Ok(work) => work, Err(_) => break },
                    };
                    let event = match work {
                        Work::Load { token, path, kind, convention } => {
                            let _cpu = crate::batch_jobs::memory::reserve();
                            Event::Loaded {
                                token,
                                result: presenter::decode(&path, kind, convention),
                            }
                        },
                        Work::Present { bundle, map, heatmap, save } => {
                            let id = bundle.request_id;
                            let captured = bundle.generation;
                            let quality = bundle.quality;
                            let extent = bundle.extent;
                            let result = if save.is_some() && generation.load(Ordering::Acquire) != captured {
                                Err("Save cancelled: document changed".into())
                            } else {
                                presenter::present(device.clone(), queue.clone(), bundle, map, heatmap, save.as_deref(), Some(&generation))
                            };
                            Event::Presented { id, generation: captured, quality, extent, map, result }
                        }
                    };
                    crossbeam_channel::select! {
                        recv(stopped) -> _ => break,
                        send(events, event) -> result => if result.is_err() { break },
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self {
            send,
            recv,
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub(crate) struct MaterialIntegration {
    pub(crate) viewport: crate::compare::Viewport,
    inspectors: inspector_runtime::Auxiliary,
    gpu: Arc<GpuWorker>,
    worker: Worker,
    generation: Arc<AtomicU64>,
    next: u64,
    seen: u64,
    seen_gesture: Option<GestureId>,
    seen_output: Option<OutputRef>,
    active: Option<(u64, u64, MapKind)>,
    ready: bool,
    source_hash: Option<[u8; 32]>,
    pending_upload: Option<PendingUpload>,
    latest_import: Option<ImportToken>,
    pending_load: Option<Work>,
    convention: NormalConvention,
    heatmap: bool,
    seen_heatmap: bool,
    presenting: bool,
    save: Option<(PathBuf, MapKind, u64)>,
    saving: bool,
    texture: Option<egui::TextureHandle>,
    image: Option<egui::ColorImage>,
    map: Option<MapKind>,
    pub last_nonconstant: bool,
    pub status: String,
    pub save_status: String,
    pub error: Option<String>,
    pub publications: u64,
    pub last_id: u64,
    displayed: Option<(u64, Quality, Extent)>,
    metrics: Option<SeamMetrics>,
    started: Instant,
}
impl MaterialIntegration {
    pub fn new(
        gpu: Arc<GpuWorker>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        // The separate presenter is serialized with material work. Its staging buffer,
        // heatmap, scratch and managed display are withheld from the SAME 1 GiB ledger.
        gpu.set_resource_reservations(ResourceReservations {
            viewport_bytes: PRESENTATION_RESERVATION,
            readback_bytes: 0,
        })
        .map_err(|e| e.to_string())?;
        let generation = Arc::new(AtomicU64::new(0));
        let inspectors =
            inspector_runtime::Auxiliary::new(device.clone(), queue.clone(), generation.clone())?;
        let worker = Worker::new(device, queue, generation.clone())?;
        Ok(Self {
            viewport: crate::compare::Viewport::new(gpu.clone()),
            inspectors,
            gpu,
            worker,
            generation,
            next: 0,
            seen: 0,
            seen_gesture: None,
            seen_output: None,
            active: None,
            ready: false,
            source_hash: None,
            pending_upload: None,
            latest_import: None,
            pending_load: None,
            convention: NormalConvention::OpenGl,
            heatmap: false,
            seen_heatmap: false,
            presenting: false,
            save: None,
            saving: false,
            texture: None,
            image: None,
            map: None,
            last_nonconstant: false,
            status: String::new(),
            save_status: String::new(),
            error: None,
            publications: 0,
            last_id: 0,
            displayed: None,
            metrics: None,
            started: Instant::now(),
        })
    }
    pub fn tick(&mut self, c: &mut Controls) {
        self.viewport.tick_assets(c);
        self.inspectors.poll(c);
        self.generation.store(
            c.session().snapshot().map_or(0, |s| s.generation),
            Ordering::Release,
        );
        if let Some(id) = self
            .inspectors
            .cancel_stale(self.generation.load(Ordering::Acquire))
            && self.active.is_some_and(|(active, _, _)| active == id)
        {
            self.active = None;
        }
        for effect in c.take_effects() {
            match effect {
                Effect::LoadImage { token, source } => {
                    if let Some(path) = c.take_path(source) {
                        self.latest_import = Some(token);
                        self.pending_load = Some(Work::Load {
                            token,
                            path,
                            kind: c.import_kind,
                            convention: c.normal_convention,
                        });
                        self.convention = c.normal_convention;
                        self.error = None;
                        self.status = "Loading... previous material retained".into();
                    }
                }
                Effect::SavePng {
                    destination,
                    map,
                    generation,
                    ..
                } => {
                    if let Some(path) = c.take_path(destination) {
                        if self.save.is_none() && !self.saving {
                            self.save = Some((path, map, generation));
                            self.save_status = "Saving full-resolution PNG...".into();
                        } else {
                            self.save_status = "A PNG save is already running".into();
                        }
                    }
                }
                Effect::InspectSeams { enabled } => {
                    self.heatmap = enabled;
                    c.heatmap = enabled;
                }
                Effect::OpenProject { .. } => {}
                Effect::CancelExport { .. }
                | Effect::SaveExportPreset { .. }
                | Effect::SaveProject { .. } => {}
                Effect::Export { .. } => {
                    self.status =
                        "Material export requires the application export coordinator.".into()
                }
            }
        }
        while let Ok(event) = self.worker.recv.try_recv() {
            match event {
                Event::Loaded { token, result } => {
                    self.loaded(c, token, result);
                }
                Event::Presented {
                    id,
                    generation,
                    quality,
                    extent,
                    map,
                    result,
                } => {
                    self.presenting = false;
                    self.active = None;
                    match result {
                        Ok(p) => {
                            let was_save = p.saved.is_some();
                            if let Some(path) = p.saved {
                                self.save_status = format!("Saved {}", path.display());
                            }
                            if !was_save
                                && c.session().snapshot().is_some_and(|s| {
                                    s.graph.selected_output_ref().port.map_kind() == map
                                        && self.seen_output == Some(s.graph.selected_output_ref())
                                        && ((s.generation == generation
                                            && self.seen == c.render_revision)
                                            // Finish samples of the same held gesture. Requiring
                                            // the newest generation starves a continuously moving slider.
                                            || (self.seen_gesture.is_some()
                                                && self.seen_gesture == c.session().history().active_gesture))
                                })
                                && self.latest_import.is_none()
                                && !c.project_busy()
                                && !crate::compare::active(c)
                                && self.seen_heatmap == self.heatmap
                            {
                                self.last_nonconstant = p.nonconstant;
                                if !self.heatmap {
                                    c.cache_map_thumbnail(map, generation, &p.image);
                                }
                                self.image = Some(p.image);
                                self.texture = None;
                                self.metrics = p.metrics;
                                self.map = Some(map);
                                self.publications += 1;
                                self.last_id = id;
                                self.displayed = Some((generation, quality, extent));
                                c.note_inspector_evaluation(true);
                                self.status = format!("{} | ready", super::map_label(map));
                            } else if !was_save {
                                // A transient import/project/view change can invalidate a
                                // completion without changing the parameter revision.
                                self.seen = 0;
                            }
                            if was_save {
                                self.seen = 0;
                            }
                        }
                        Err(e) => {
                            self.status = e.clone();
                            self.error = Some(e);
                        }
                    }
                    self.saving = false;
                }
            }
        }
        while let Some(event) = self.gpu.try_event() {
            match event {
                WorkerEvent::SourceReady { content_hash } if c.ack_project_upload(content_hash) => {
                    self.source_hash = Some(content_hash);
                    self.ready = true;
                    self.latest_import = None;
                    self.pending_upload = None;
                    self.active = None;
                    self.seen = 0;
                    self.error = None;
                }
                WorkerEvent::SourceReady { content_hash }
                    if self
                        .pending_upload
                        .as_ref()
                        .is_some_and(|p| p.source.content_hash == content_hash) =>
                {
                    let pending = self.pending_upload.take().unwrap();
                    if self.latest_import == Some(pending.token) {
                        match c.complete_typed_import(
                            pending.token,
                            pending.source,
                            pending.convention,
                        ) {
                            Ok(()) => {
                                self.source_hash = Some(content_hash);
                                self.ready = true;
                                self.latest_import = None;
                                self.active = None;
                                self.seen = 0;
                                self.error = None;
                                self.generation.store(
                                    c.session().snapshot().unwrap().generation,
                                    Ordering::Release,
                                );
                                self.status = "Preparing material...".into();
                            }
                            Err(e) => self.fail_import(c, pending.token, e),
                        }
                    }
                }
                WorkerEvent::SourceReady { content_hash }
                    if self.source_hash == Some(content_hash) =>
                {
                    self.ready = true;
                    self.active = None;
                    self.seen = 0;
                }
                WorkerEvent::SourceReady { .. } => {}
                other if c.project_upload_pending() => {
                    c.fail_project_upload(format!("{other:?}"));
                }
                other if self.pending_upload.is_some() => {
                    let pending = self.pending_upload.take().unwrap();
                    self.fail_import(c, pending.token, format!("{other:?}"));
                }
                other => {
                    if self.viewport.busy() {
                        self.viewport.fail(format!("{other:?}"));
                    }
                    self.inspectors.failed(c);
                    c.note_inspector_evaluation(false);
                    self.status = format!(
                        "Material evaluation failed: {other:?}. Previous preview retained."
                    );
                    self.error = Some(self.status.clone());
                    self.active = None;
                    self.saving = false;
                }
            }
        }
        if (!crate::compare::active(c) || self.active.is_some())
            && let Some(bundle) = self.gpu.try_latest_material()
        {
            let bundle = self.inspectors.receive(bundle);
            if let Some(bundle) = bundle {
                let id = bundle.request_id;
                if let Some((expected, _, map)) = self.active
                    && id == expected
                {
                    let save = if self.saving {
                        self.save.take().map(|s| s.0)
                    } else {
                        None
                    };
                    match self.worker.send.try_send(Work::Present {
                        bundle,
                        map,
                        heatmap: self.heatmap && !self.saving,
                        save,
                    }) {
                        Ok(()) => self.presenting = true,
                        Err(_) => {
                            self.active = None;
                            self.status = "Presentation busy; retrying".into();
                            self.seen = 0;
                        }
                    }
                } else {
                    drop(bundle);
                    self.active = None;
                }
            } else {
                self.active = None;
            }
        }
        if !self.presenting
            && self.active.is_none()
            && self.pending_upload.is_none()
            && let Some(load) = self.pending_load.take()
        {
            match self.worker.send.try_send(load) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(w)) => self.pending_load = Some(w),
                Err(_) => {
                    if let Some(token) = self.latest_import {
                        self.fail_import(c, token, "Import worker closed".into());
                    }
                }
            }
        }
        if !self.ready
            || c.project_busy()
            || self.pending_upload.is_some()
            || self.presenting
            || self.active.is_some()
            || self.latest_import.is_some()
            || self.inspectors.busy()
        {
            return;
        }
        if crate::compare::active(c) && self.save.is_none() {
            // Update foreground intent first, then use idle compute time for
            // picker/thumbnail work. Map selection is never needed to fill cards.
            self.viewport.tick(c);
            if !self.viewport.busy() {
                self.active = self.inspectors.request(c, &self.gpu, &mut self.next);
                if self.active.is_some() {
                    return;
                }
            }
            return;
        }
        if self.viewport.busy() {
            self.seen = 0;
        }
        self.viewport.suspended();
        let Some(snapshot) = c.session().snapshot() else {
            return;
        };
        let selected = snapshot.graph.selected_output_ref().port.map_kind();
        let saving = self.save.is_some();
        if !saving && self.seen == c.render_revision && self.seen_heatmap == self.heatmap {
            self.active = self.inspectors.request(c, &self.gpu, &mut self.next);
            return;
        }
        let map = self.save.as_ref().map_or(selected, |s| s.1);
        if self
            .save
            .as_ref()
            .is_some_and(|s| s.2 != snapshot.generation)
        {
            self.save = None;
            self.save_status = "Save cancelled: document changed".into();
            return;
        }
        self.next += 1;
        let quality = if saving {
            Quality::Full
        } else {
            c.render_quality
        };
        match super::request(snapshot, map, self.next, quality)
            .and_then(|r| self.gpu.request_material(r).map_err(|e| e.to_string()))
        {
            Ok(()) => {
                self.active = Some((self.next, c.session().snapshot().unwrap().generation, map));
                self.saving = saving;
                self.seen = c.render_revision;
                self.seen_gesture = c.session().history().active_gesture;
                self.seen_output =
                    Some(c.session().snapshot().unwrap().graph.selected_output_ref());
                self.seen_heatmap = self.heatmap;
                self.error = None;
                self.status = format!("Computing {}...", super::map_label(map));
            }
            Err(e) => {
                self.status = e;
                self.seen = c.render_revision;
                self.seen_heatmap = self.heatmap;
                self.save = None;
            }
        }
    }
    fn loaded(
        &mut self,
        c: &mut Controls,
        token: ImportToken,
        result: Result<TypedSourceUpload, String>,
    ) {
        if self.latest_import != Some(token) {
            return;
        }
        match result {
            Ok(upload) => match self.gpu.install_typed_source(upload.clone()) {
                Ok(()) => {
                    // Admission only queues work. Keep the document/history
                    // and previous source until the matching GPU acknowledgement.
                    self.pending_upload = Some(PendingUpload {
                        token,
                        source: upload.descriptor,
                        convention: self.convention,
                    });
                    self.status = "Uploading... previous material retained".into();
                }
                Err(e) => self.fail_import(c, token, e.to_string()),
            },
            Err(e) => self.fail_import(c, token, e),
        }
    }
    fn fail_import(&mut self, c: &mut Controls, token: ImportToken, error: String) {
        // A newer queued import supersedes this result without consuming its token.
        if self.latest_import == Some(token) {
            c.fail_import(token);
            self.latest_import = None;
            self.status = format!("Import failed: {error}. Previous material retained.");
        }
    }
    pub fn show(&mut self, ui: &mut egui::Ui, c: &mut Controls) {
        c.workspace.status.clone_from(&self.status);
        c.workspace.busy = self.active.is_some() || self.presenting || self.latest_import.is_some();
        if !self.save_status.is_empty() {
            c.workspace.status.clone_from(&self.save_status);
        }
        if c.session().snapshot().is_none() {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                if ui
                    .add(
                        egui::Button::new(crate::input_locale::labels::text("native-open-image"))
                            .min_size(egui::vec2(220.0, 56.0)),
                    )
                    .clicked()
                {
                    c.browse_image();
                }
                ui.label(crate::input_locale::labels::text("native-drop-hint"));
            });
            return;
        }
        let selected = c
            .session()
            .snapshot()
            .map(|s| s.graph.selected_output_ref().port.map_kind());
        if let Some(map) = selected
            && c.session()
                .view_state()
                .maps
                .get(&map)
                .is_some_and(|state| !state.visible)
        {
            ui.label(crate::workspace::labels::text("map-hidden"));
            let size = ui.available_size().max(egui::vec2(1.0, 1.0));
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 0.0, crate::workspace::neutral_preview(map));
            return;
        }
        if let Some(m) = self.metrics {
            c.workspace.status = format!(
                "Seam RMS {:.5} | interior {:.5} | ratio {:.3}",
                m.boundary_rms, m.interior_rms, m.ratio
            );
        }
        if self.texture.is_none()
            && let Some(image) = self.image.take()
        {
            self.texture = Some(ui.ctx().load_texture(
                "material-2d",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        if let Some(texture) = &self.texture {
            let tiles = c.session().view_state().tiles.count as usize;
            let available = ui.available_size().max(egui::vec2(1.0, 1.0));
            let size = texture.size_vec2();
            let scale = (available.x / size.x).min(available.y / size.y);
            let fitted = size * scale;
            let canvas = egui::Rect::from_min_size(ui.cursor().min, available);
            let bounds = c.map_canvas_rect(egui::Rect::from_center_size(canvas.center(), fitted));
            let full = bounds.size();
            let top = bounds.min;
            let painter = ui.painter().with_clip_rect(canvas);
            for y in 0..tiles {
                for x in 0..tiles {
                    let cell = full / tiles as f32;
                    let rect = egui::Rect::from_min_size(
                        top + egui::vec2(x as f32 * cell.x, y as f32 * cell.y),
                        cell,
                    );
                    painter.image(
                        texture.id(),
                        rect,
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            }
            ui.allocate_space(available);
            c.viewport_gestures(ui, egui::Rect::from_min_size(top, full));
        }
    }
    pub(crate) fn viewport_can_sleep(&self, c: &Controls) -> bool {
        crate::compare::active(c)
            && !self.viewport.needs_frame(c)
            && self.active.is_none()
            && !self.presenting
            && !self.inspectors.busy()
            && self.latest_import.is_none()
            && self.pending_load.is_none()
            && self.pending_upload.is_none()
            && self.save.is_none()
            && !c.project_busy()
            && !c.batch_status.busy
            && !c.export_busy()
    }
    pub fn full_visible(&self) -> bool {
        self.texture.is_some()
            && self
                .displayed
                .is_some_and(|(_, quality, _)| quality == Quality::Full)
            && self.active.is_none()
            && !self.presenting
            && self.ready
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"material_demo":true,"publications":self.publications,"last_request_id":self.last_id,"displayed":self.displayed.map(|(generation, quality, extent)| serde_json::json!({"generation":generation,"quality":format!("{quality:?}"),"extent":extent})),"status":self.status,"error":self.error,"presentation_reservation_bytes":PRESENTATION_RESERVATION,"elapsed_seconds":self.started.elapsed().as_secs_f64()})
    }
}
