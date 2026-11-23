use super::{assets::*, *};
use crate::{command_adapter::UiAction, controls::Controls, display::NativePreview};
use std::{sync::Arc, time::Instant};
use texelith_core::*;
use texelith_gpu::{viewport_shared::PreviewContent, *};

pub fn material_request(
    snapshot: EvaluationSnapshot,
    map: MapKind,
    id: u64,
    lit: bool,
    view: &ViewState,
) -> Result<MaterialRequest, String> {
    let all = snapshot
        .graph
        .material_outputs()
        .ok_or("Material graph required")?;
    let outputs = all
        .iter()
        .filter(|(kind, _)| {
            (if lit {
                viewport_shared::MATERIAL_MAPS.contains(kind)
            } else {
                **kind == map
            }) && view.maps.get(kind).is_none_or(|m| m.visible)
        })
        .map(|(k, v)| (*k, *v))
        .collect::<MaterialOutputs>();
    if outputs.is_empty() {
        return Err("Selected map is hidden or unavailable".into());
    }
    let extent = snapshot
        .graph
        .material_output_extent(*outputs.values().next().unwrap())
        .map_err(|e| e.to_string())?;
    Ok(MaterialRequest {
        job_id: JobId(id),
        request_id: id,
        snapshot: Arc::new(snapshot),
        outputs,
        extent: compare_extent(extent),
        quality: Quality::Proxy,
    })
}

enum Pending {
    Material { id: u64, side: Side },
    Preview { id: u64, side: Side },
}
pub(crate) struct Viewport {
    gpu: Arc<GpuWorker>,
    assets: Assets,
    asset_id: u64,
    waiting_asset: bool,
    custom: Option<Arc<MeshData>>,
    environment: Arc<ibl::Environment>,
    environment_name: String,
    repaint: Option<egui::Context>,
    environment_enabled: bool,
    environment_exposure: f32,
    environment_rotation: f32,
    background_visible: bool,
    background_blur: f32,
    key: Vec<u8>,
    material_key: Vec<u8>,
    materials: Vec<(Vec<u8>, Arc<CompletedMaterial>)>,
    started: Option<Instant>,
    frame_ms: std::collections::VecDeque<f32>,
    pub material_requests: u64,
    identity: u64,
    next: u64,
    pending: Option<Pending>,
    candidates: PairCache<CompletedPreview>,
    ready: Option<(CompletedPreview, Option<CompletedPreview>)>,
    shown: Option<(NativePreview, Option<NativePreview>)>,
    recipes: Vec<(Side, EvaluationSnapshot)>,
    view: ViewState,
    map: MapKind,
    split_view: bool,
    quality: Quality,
    extent: Extent,
    exposure: f32,
    pub status: String,
    asset_status: String,
    pub error: Option<String>,
    pub publications: u64,
}
impl Viewport {
    pub fn new(gpu: Arc<GpuWorker>) -> Self {
        Self {
            gpu,
            assets: Assets::new(),
            asset_id: 0,
            waiting_asset: false,
            custom: None,
            environment: Arc::new(ibl::Environment::procedural()),
            environment_name: super::labels::text("sky"),
            repaint: None,
            environment_enabled: true,
            environment_exposure: 0.,
            environment_rotation: 0.,
            background_visible: true,
            background_blur: 0.,
            key: Vec::new(),
            material_key: Vec::new(),
            materials: Vec::new(),
            started: None,
            frame_ms: std::collections::VecDeque::new(),
            material_requests: 0,
            identity: 0,
            next: 1_000_000,
            pending: None,
            candidates: PairCache::default(),
            ready: None,
            shown: None,
            recipes: Vec::new(),
            view: ViewState::default(),
            map: MapKind::Normal,
            split_view: false,
            quality: Quality::Proxy,
            extent: Extent {
                width: 1024,
                height: 768,
            },
            exposure: 0.,
            status: String::new(),
            asset_status: String::new(),
            error: None,
            publications: 0,
        }
    }
    pub fn busy(&self) -> bool {
        self.pending.is_some() || !self.recipes.is_empty()
    }
    pub fn engaged(&self) -> bool {
        !self.key.is_empty()
    }
    pub fn settled(&self) -> bool {
        self.shown.is_some() && !self.busy() && !self.waiting_asset && self.error.is_none()
    }
    pub fn suspended(&mut self) {
        self.key.clear();
        self.materials.clear();
        self.material_key.clear();
        self.pending = None;
        self.recipes.clear();
        self.ready = None;
        self.candidates.begin(self.identity);
    }
    pub fn tick_assets(&mut self, c: &mut Controls) {
        if let Some(load) = c.preview.pending.take() {
            let id = self.asset_id + 1;
            match self.assets.request(id, load) {
                Ok(()) => {
                    self.asset_id = id;
                    self.waiting_asset = true;
                    self.status = "Loading viewport asset; previous frame retained".into();
                }
                Err(load) => c.preview.pending = Some(load),
            }
        }
        while let Ok((id, result)) = self.assets.recv.try_recv() {
            if id != self.asset_id {
                continue;
            }
            self.waiting_asset = false;
            match result {
                Ok(Loaded::Mesh(mesh)) => {
                    self.custom = Some(mesh);
                    let mesh = PreviewMesh::Custom(OpaqueLocator(format!("mesh-{id}")));
                    c.preview.imported_mesh = Some(mesh.clone());
                    c.action(UiAction::Request(CommandRequest::SetMesh { mesh }));
                    self.error = None;
                    self.asset_status.clear();
                }
                Ok(Loaded::Hdr(env, name)) => {
                    self.environment = env;
                    self.environment_name = name;
                    c.preview.environment_enabled = true;
                    c.preview.background_visible = true;
                    let mut light = c.session().view_state().light.clone();
                    light.environment = Some(OpaqueLocator(format!("environment-{id}")));
                    c.action(UiAction::Request(CommandRequest::SetLight { light }));
                    self.error = None;
                    self.asset_status.clear();
                }
                Err(e) => {
                    self.asset_status = format!("Asset load failed: {e}. Previous frame retained.");
                }
            }
            c.preview.revision += 1;
        }
    }
    pub fn tick(&mut self, c: &Controls) {
        if self.waiting_asset {
            return;
        }
        let Some(snapshot) = c.session().snapshot() else {
            return;
        };
        let view = requested_view(c);
        let key = self.intent_key(c, &view, snapshot.generation);
        // Finish the in-flight frame before adopting the latest input. Replacing
        // every request during a drag can cancel all frames before publication.
        if key != self.key && !self.busy() {
            self.key = key;
            self.identity += 1;
            self.pending = None;
            self.ready = None;
            self.candidates.begin(self.identity);
            self.recipes.clear();
            self.error = None;
            self.map = snapshot.graph.selected_output_ref().port.map_kind();
            self.view = view;
            self.split_view = c.preview.split_view && self.view.mode == ViewMode::ThreeD;
            // Stored A/B comparisons retain ADR 0008's bounded proxy recipes.
            // Live 3D and the studio map/lit split settle after a gesture.
            self.quality = if self.view.capture_a.is_some() && !self.split_view {
                Quality::Proxy
            } else {
                c.render_quality
            };
            self.exposure = c.preview.exposure;
            self.environment_enabled =
                c.preview.environment_enabled && c.preview.environment_intensity > 0.;
            self.environment_exposure = c.preview.environment_intensity.max(0.000001).log2();
            self.environment_rotation = c.preview.environment_rotation.to_radians();
            self.background_visible = c.preview.background_visible;
            self.background_blur = c.preview.background_blur;
            if self.split_view {
                self.recipes.push((Side::B, snapshot.clone()));
                self.recipes.push((Side::A, snapshot));
            } else if let Some(a) = c
                .session()
                .captured_a()
                .filter(|_| c.preview.compare_enabled)
            {
                self.recipes.push((
                    Side::B,
                    c.session()
                        .captured_b()
                        .cloned()
                        .unwrap_or_else(|| snapshot.clone()),
                ));
                self.recipes.push((Side::A, a.clone()));
            } else {
                self.recipes.push((Side::B, snapshot));
            }
        }
        if let Some(bundle) = self.gpu.try_latest_material()
            && let Some(Pending::Material { id, side }) = self.pending
            && bundle.request_id == id
        {
            let bundle = Arc::new(bundle);
            if self.materials.len() == 2 {
                self.materials.remove(0);
            }
            self.materials
                .push((self.material_key.clone(), bundle.clone()));
            self.request_frame(bundle, id, side);
        }
        if let Some(frame) = self.gpu.try_latest_preview()
            && let Some(Pending::Preview { id, side }) = self.pending
            && frame.request_id == id
        {
            self.pending = None;
            if let Some(started) = self.started.take() {
                if self.frame_ms.len() == 240 {
                    self.frame_ms.pop_front();
                }
                self.frame_ms
                    .push_back(started.elapsed().as_secs_f32() * 1000.);
            }
            if self.view.capture_a.is_some() || self.split_view {
                self.candidates.insert(side, self.identity, frame);
                if let Some((a, b)) = self.candidates.take() {
                    self.ready = Some((a, Some(b)));
                }
            } else {
                self.ready = Some((frame, None));
            }
            if self.ready.is_some() {
                self.publications += 1;
                self.status = format!("Ready | shared GPU | material {:?}", self.quality);
                if let Some(ctx) = &self.repaint {
                    ctx.request_repaint();
                }
            }
        }
        if self.pending.is_none()
            && let Some((side, snapshot)) = self.recipes.pop()
        {
            self.next += 1;
            let map = self.map;
            let mut request_view = self.view.clone();
            if self.split_view && side == Side::A {
                // A hidden channel still has a neutral 2D pane; the lit pane
                // independently omits its contribution via the original view.
                request_view.maps.entry(map).or_default().visible = true;
            }
            let result = material_request(
                snapshot,
                map,
                self.next,
                self.view.mode == ViewMode::ThreeD && !(self.split_view && side == Side::A),
                &request_view,
            );
            match result {
                Ok(mut request) => {
                    if self.quality == Quality::Full {
                        request.extent = request
                            .snapshot
                            .graph
                            .material_output_extent(*request.outputs.values().next().unwrap())
                            .expect("validated material output");
                        request.quality = Quality::Full;
                    }
                    let key = serde_json::to_vec(&(
                        &request.snapshot.graph,
                        request.snapshot.generation,
                        &request.outputs,
                        request.extent,
                        request.quality == Quality::Full,
                    ))
                    .expect("material key");
                    if let Some(bundle) = self
                        .materials
                        .iter()
                        .find(|(cached, _)| *cached == key)
                        .map(|(_, bundle)| bundle.clone())
                    {
                        self.request_frame(bundle, self.next, side);
                    } else {
                        // Keep at most the other A/B side across recipe changes.
                        if self.view.capture_a.is_none() && !self.split_view {
                            self.materials.clear();
                        } else if self.materials.len() == 2 {
                            self.materials.remove(0);
                        }
                        self.material_key = key;
                        match self.gpu.request_material(request) {
                            Ok(()) => {
                                self.material_requests += 1;
                                self.pending = Some(Pending::Material {
                                    id: self.next,
                                    side,
                                });
                                self.status = "Preparing material; previous frame retained".into();
                            }
                            Err(e) => self.fail(e.to_string()),
                        }
                    }
                }
                Err(e) => self.fail(e),
            }
        }
    }
    fn request_frame(&mut self, bundle: Arc<CompletedMaterial>, id: u64, side: Side) {
        let settings = pbr::PbrSettings {
            exposure: self.exposure,
            environment_rotation: self.environment_rotation,
            background_visible: self.background_visible,
            background_blur: self.background_blur,
            ..Default::default()
        };
        let content = if self.view.mode == ViewMode::TwoD || (self.split_view && side == Side::A) {
            PreviewContent::Map {
                map: self.map,
                tiles: self.view.tiles,
                convention: bundle.normal_convention,
            }
        } else if matches!(self.view.mesh, PreviewMesh::Custom(_)) {
            let Some(mesh) = self.custom.clone() else {
                self.fail("Custom mesh is not loaded".into());
                return;
            };
            PreviewContent::Mesh {
                settings,
                mesh,
                environment: self.environment_enabled.then(|| self.environment.clone()),
                exposure: self.environment_exposure,
            }
        } else if self.environment_enabled {
            PreviewContent::Ibl {
                settings,
                environment: self.environment.clone(),
                exposure: self.environment_exposure,
            }
        } else {
            PreviewContent::Pbr(settings)
        };
        let mut view = self.view.clone();
        if self.split_view && side == Side::A {
            view.mode = ViewMode::TwoD;
        }
        let request = PreviewRequest {
            job_id: bundle.job_id,
            generation: bundle.generation,
            request_id: id,
            material: bundle,
            extent: if self.view.capture_a.is_some() || self.split_view {
                compare_extent(self.extent)
            } else {
                self.extent
            },
            view,
            content,
        };
        match self.gpu.request_retained_preview(request) {
            Ok(()) => {
                self.started = Some(Instant::now());
                self.pending = Some(Pending::Preview { id, side });
                self.status = "Rendering; previous frame retained".into();
            }
            Err(e) => self.fail(e.to_string()),
        }
    }
    fn intent_key(&self, c: &Controls, view: &ViewState, generation: u64) -> Vec<u8> {
        serde_json::to_vec(&(
            &view,
            generation,
            c.render_revision,
            c.preview.revision,
            c.preview.split_view,
            self.extent,
            c.preview.exposure,
            (
                c.preview.environment_enabled,
                c.preview.environment_rotation,
                c.preview.environment_intensity,
                c.preview.background_visible,
                c.preview.background_blur,
            ),
        ))
        .expect("finite view")
    }
    pub(crate) fn needs_frame(&self, c: &Controls) -> bool {
        self.busy()
            || self.ready.is_some()
            || self.waiting_asset
            || c.preview.pending.is_some()
            || c.session().snapshot().is_some_and(|snapshot| {
                self.key != self.intent_key(c, &requested_view(c), snapshot.generation)
            })
    }
    pub(crate) fn set_extent(&mut self, extent: Extent) {
        self.extent = extent;
    }
    pub(crate) fn p95_ms(&self) -> f32 {
        let mut samples: Vec<_> = self.frame_ms.iter().copied().collect();
        samples.sort_by(f32::total_cmp);
        samples
            .get((samples.len() * 95).div_ceil(100).saturating_sub(1))
            .copied()
            .unwrap_or(0.)
    }
    pub fn fail(&mut self, e: String) {
        self.status = e.clone();
        self.error = Some(e);
        self.pending = None;
        self.recipes.clear();
    }
    pub fn show(&mut self, ui: &mut egui::Ui, c: &mut Controls, state: &egui_wgpu::RenderState) {
        self.repaint = Some(ui.ctx().clone());
        if let Some((a, b)) = self.ready.take() {
            self.shown = Some((
                NativePreview::new(state.clone(), a),
                b.map(|b| NativePreview::new(state.clone(), b)),
            ));
            ui.ctx().request_repaint();
        }
        c.workspace.status.clone_from(&self.status);
        c.workspace.busy = self.busy() || self.waiting_asset;
        if !self.asset_status.is_empty() {
            c.workspace.status.clone_from(&self.asset_status);
        }
        if c.preview.timings {
            c.workspace
                .status
                .push_str(&format!(" | p95 {:.2} ms", self.p95_ms()));
        }
        let mut available = ui.available_size().max(egui::vec2(1., 1.));
        if c.session().view_state().mode == ViewMode::TwoD
            && let Some(snapshot) = c.session().snapshot()
            && let Ok(source) = snapshot
                .graph
                .material_output_extent(snapshot.graph.selected_output_ref())
        {
            let size = egui::vec2(source.width as f32, source.height as f32);
            available = size * (available.x / size.x).min(available.y / size.y);
        }
        if c.preview.split_view && c.session().view_state().mode == ViewMode::ThreeD {
            available.x = ((available.x - 8.) * 0.5).max(1.);
        }
        let pixels = available * ui.ctx().pixels_per_point();
        c.preview.aspect = pixels.x / pixels.y.max(1.);
        if c.preview.fit_pending && c.session().view_state().mode == ViewMode::ThreeD {
            c.preview.fit_pending = false;
            c.action(UiAction::Request(CommandRequest::SetCamera {
                camera: super::frame_camera(&c.session().view_state().mesh, c.preview.aspect),
            }));
        }
        let extent = Extent {
            width: pixels.x.max(1.) as u32,
            height: pixels.y.max(1.) as u32,
        };
        if extent != self.extent {
            self.extent = extent;
        }
        let split = c.preview.split_view && c.session().view_state().mode == ViewMode::ThreeD;
        let total = if split {
            egui::vec2(available.x * 2. + 8., available.y)
        } else {
            available
        };
        let (rect, response) = ui.allocate_exact_size(
            total,
            if split {
                egui::Sense::hover()
            } else {
                egui::Sense::click_and_drag()
            },
        );
        ui.painter()
            .rect_filled(rect, 0., egui::Color32::from_gray(26));
        if let Some((a, b)) = &self.shown {
            let uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.));
            if split {
                let left = egui::Rect::from_min_size(rect.min, available);
                let right = egui::Rect::from_min_size(
                    rect.min + egui::vec2(available.x + 8., 0.),
                    available,
                );
                if let Some(b) = b {
                    let source = c.session().snapshot().and_then(|s| {
                        s.graph
                            .material_output_extent(s.graph.selected_output_ref())
                            .ok()
                    });
                    let size =
                        source.map_or(left.size(), |e| egui::vec2(e.width as f32, e.height as f32));
                    let fitted = size * (left.width() / size.x).min(left.height() / size.y);
                    let map_rect =
                        c.map_canvas_rect(egui::Rect::from_center_size(left.center(), fitted));
                    if c.session()
                        .view_state()
                        .maps
                        .get(&self.map)
                        .is_some_and(|m| !m.visible)
                    {
                        ui.painter().rect_filled(
                            left,
                            0.,
                            crate::workspace::neutral_preview(self.map),
                        );
                    } else {
                        a.paint(ui, map_rect, uv, left);
                    }
                    let clip = ui.clip_rect();
                    ui.set_clip_rect(left.intersect(clip));
                    c.viewport_gestures(ui, map_rect);
                    ui.set_clip_rect(clip);
                    b.paint(ui, right, uv, right);
                } else {
                    a.paint(ui, right, uv, right);
                }
            } else if let Some(b) = b {
                let x = egui::lerp(rect.x_range(), c.session().view_state().split_fraction);
                a.paint(
                    ui,
                    rect,
                    uv,
                    egui::Rect::from_min_max(rect.min, egui::pos2(x, rect.max.y)),
                );
                b.paint(
                    ui,
                    rect,
                    uv,
                    egui::Rect::from_min_max(egui::pos2(x, rect.min.y), rect.max),
                );
                ui.painter().line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    (1., egui::Color32::WHITE),
                );
                ui.painter().text(
                    rect.min + egui::vec2(8., 8.),
                    egui::Align2::LEFT_TOP,
                    "A",
                    egui::FontId::proportional(16.),
                    egui::Color32::WHITE,
                );
                ui.painter().text(
                    rect.right_top() + egui::vec2(-8., 8.),
                    egui::Align2::RIGHT_TOP,
                    "B",
                    egui::FontId::proportional(16.),
                    egui::Color32::WHITE,
                );
            } else {
                let draw_rect = if c.session().view_state().mode == ViewMode::TwoD {
                    c.map_canvas_rect(rect)
                } else {
                    rect
                };
                a.paint(ui, draw_rect, uv, rect);
            }
        }
        if c.session().view_state().mode == ViewMode::TwoD {
            c.viewport_gestures(ui, c.map_canvas_rect(rect));
        } else if split {
            let right =
                egui::Rect::from_min_size(rect.min + egui::vec2(available.x + 8., 0.), available);
            let response = ui.interact(
                right,
                ui.scope_id().with("split-orbit"),
                egui::Sense::click_and_drag(),
            );
            c.orbit_viewport(ui, &response);
        } else {
            c.orbit_viewport(ui, &response);
        }
        if self.needs_frame(c) {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(1));
        }
    }
    pub fn take_ready(&mut self) -> Option<(CompletedPreview, Option<CompletedPreview>)> {
        self.ready.take()
    }
    #[cfg(test)]
    pub(crate) fn cached_materials(&self) -> impl Iterator<Item = &CompletedMaterial> {
        self.materials.iter().map(|(_, material)| material.as_ref())
    }
}

fn requested_view(c: &Controls) -> ViewState {
    let mut view = c.session().view_state().clone();
    if !c.preview.compare_enabled || c.preview.split_view {
        view.capture_a = None;
        view.capture_b = None;
    }
    view.split_fraction = 0.5;
    view
}
