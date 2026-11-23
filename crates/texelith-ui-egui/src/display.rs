use std::sync::{Arc, Mutex};

/// Handles cloned from eframe's actual render state; cloning does not create a device.
#[derive(Clone)]
pub struct SharedGpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl SharedGpu {
    pub fn from_render_state(state: &egui_wgpu::RenderState) -> Self {
        Self {
            device: Arc::new(state.device.clone()),
            queue: Arc::new(state.queue.clone()),
        }
    }
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub(crate) struct AllocationStats {
    pub created: u64,
    pub released: u64,
    pub live: u64,
    pub peak_live: u64,
    pub live_bytes: u64,
    pub peak_bytes: u64,
}

impl AllocationStats {
    fn allocate(&mut self, bytes: u64) -> Result<(), String> {
        if self.live_bytes + bytes > DISPLAY_BUDGET {
            return Err(
                "immutable display retirement exceeded the 32 MiB diagnostic budget".into(),
            );
        }
        self.created += 1;
        self.live += 1;
        self.live_bytes += bytes;
        self.peak_live = self.peak_live.max(self.live);
        self.peak_bytes = self.peak_bytes.max(self.live_bytes);
        Ok(())
    }

    fn release(&mut self, bytes: u64) {
        self.released += 1;
        self.live -= 1;
        self.live_bytes -= bytes;
    }
}

pub(crate) const DISPLAY_BUDGET: u64 = 32 * 1024 * 1024;
pub(crate) const WIDTH: u32 = 512;
pub(crate) const HEIGHT: u32 = 256;
const BYTES: u64 = WIDTH as u64 * HEIGHT as u64 * 4;

/// Owns one immutable, opaque, gamma-code Rgba8Unorm display allocation.
/// This diagnostic bridge deliberately never returns an allocation to a reuse pool.
pub struct DisplayLease {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    accounting: Arc<Mutex<AllocationStats>>,
}

impl Drop for DisplayLease {
    fn drop(&mut self) {
        self.accounting.lock().unwrap().release(BYTES);
    }
}

struct RegisteredDisplay {
    id: egui::TextureId,
    lease: Arc<DisplayLease>,
}

/// UI-thread native texture registration on eframe's renderer.
pub struct DisplayBridge {
    state: egui_wgpu::RenderState,
    gpu: SharedGpu,
    current: Option<RegisteredDisplay>,
    accounting: Arc<Mutex<AllocationStats>>,
    pixels: Vec<u8>,
}

impl DisplayBridge {
    pub fn new(state: egui_wgpu::RenderState) -> Self {
        Self {
            gpu: SharedGpu::from_render_state(&state),
            state,
            current: None,
            accounting: Arc::default(),
            pixels: diagnostic_ramps(),
        }
    }

    pub fn shared_gpu(&self) -> SharedGpu {
        self.gpu.clone()
    }

    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.current.as_ref().map(|display| display.id)
    }

    pub fn lease(&self) -> Option<Arc<DisplayLease>> {
        self.current
            .as_ref()
            .map(|display| Arc::clone(&display.lease))
    }

    /// Diagnostic allocation/upload. Production compute will supply a completed lease.
    pub fn replace_diagnostic(&mut self) -> Result<(), String> {
        self.accounting.lock().unwrap().allocate(BYTES)?;
        let texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texelith immutable diagnostic ramps"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.gpu.queue.write_texture(
            texture.as_image_copy(),
            &self.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: Some(HEIGHT),
            },
            texture.size(),
        );
        let view = texture.create_view(&Default::default());
        let lease = Arc::new(DisplayLease {
            texture,
            view,
            accounting: Arc::clone(&self.accounting),
        });
        let id = self.state.renderer.write().register_native_texture(
            &self.gpu.device,
            &lease.view,
            wgpu::FilterMode::Linear,
        );
        if let Some(old) = self.current.replace(RegisteredDisplay { id, lease }) {
            // Earlier command buffers retain the old lease through FrameCompletion.
            // No future frame receives this ID. Never call Texture::destroy here.
            self.state.renderer.write().free_texture(&old.id);
        }
        Ok(())
    }

    pub(crate) fn accounting(&self) -> Arc<Mutex<AllocationStats>> {
        Arc::clone(&self.accounting)
    }
}

impl Drop for DisplayBridge {
    fn drop(&mut self) {
        if let Some(display) = self.current.take() {
            self.state.renderer.write().free_texture(&display.id);
        }
    }
}

fn diagnostic_ramps() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(BYTES as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let t = x as f32 / (WIDTH - 1) as f32;
            let rgba = if y < HEIGHT / 2 {
                let value = (t * 255.0).round() as u8;
                [value, value, value, 255]
            } else {
                // A hemisphere of unit normals: left=-X, center=+Z, right=+X.
                // Green varies from +Y at the top to -Y at the bottom (image rows down).
                let nx = (t * 2.0 - 1.0) * 0.8;
                let ny = (1.0 - (y - HEIGHT / 2) as f32 / (HEIGHT / 2 - 1) as f32 * 2.0) * 0.5;
                let nz = (1.0 - nx * nx - ny * ny).sqrt();
                let code = |n: f32| ((n * 0.5 + 0.5) * 255.0).round() as u8;
                [code(nx), code(ny), code(nz), 255]
            };
            pixels.extend_from_slice(&rgba);
        }
    }
    pixels
}

/// Registration of a worker-completed viewport on eframe's existing device.
/// The callback holds the immutable lease through submission of its egui draw.
pub(crate) struct NativePreview {
    state: egui_wgpu::RenderState,
    id: egui::TextureId,
    frame: texelith_gpu::CompletedPreview,
}
impl NativePreview {
    pub fn new(state: egui_wgpu::RenderState, frame: texelith_gpu::CompletedPreview) -> Self {
        let id = state.renderer.write().register_native_texture(
            &state.device,
            frame.lease.view(),
            wgpu::FilterMode::Linear,
        );
        Self { state, id, frame }
    }
    pub fn paint(&self, ui: &egui::Ui, rect: egui::Rect, uv: egui::Rect, clip: egui::Rect) {
        ui.painter()
            .with_clip_rect(ui.clip_rect().intersect(clip))
            .image(self.id, rect, uv, egui::Color32::WHITE);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            PreviewFence(self.frame.lease.clone()),
        ));
    }
}
struct PreviewFence(texelith_gpu::DisplayLease);
impl egui_wgpu::CallbackTrait for PreviewFence {
    fn paint(
        &self,
        _: egui::PaintCallbackInfo,
        _: &mut wgpu::RenderPass<'static>,
        _: &egui_wgpu::CallbackResources,
    ) {
    }
    fn prepare(
        &self,
        _: &wgpu::Device,
        _: &wgpu::Queue,
        _: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let lease = self.0.clone();
        encoder.on_submitted_work_done(move || lease.retire_after_submission());
        Vec::new()
    }
}
impl Drop for NativePreview {
    fn drop(&mut self) {
        self.state.renderer.write().free_texture(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_budget_is_bounded_and_balances_releases() {
        let mut stats = AllocationStats::default();
        for _ in 0..DISPLAY_BUDGET / BYTES {
            stats.allocate(BYTES).unwrap();
        }
        assert!(stats.allocate(BYTES).is_err());
        assert_eq!(stats.created, DISPLAY_BUDGET / BYTES);
        stats.release(BYTES);
        stats.allocate(BYTES).unwrap();
        assert_eq!(stats.peak_bytes, DISPLAY_BUDGET);
        while stats.live > 0 {
            stats.release(BYTES);
        }
        assert_eq!(stats.created, stats.released);
        assert_eq!(stats.live_bytes, 0);
    }

    #[test]
    fn ramps_have_exact_endpoints_and_normal_orientation() {
        let pixels = diagnostic_ramps();
        let at = |x: u32, y: u32| {
            &pixels[((y * WIDTH + x) * 4) as usize..((y * WIDTH + x) * 4 + 4) as usize]
        };
        assert_eq!(pixels.len(), BYTES as usize);
        assert_eq!(at(0, 0), [0, 0, 0, 255]);
        assert_eq!(at(WIDTH - 1, 0), [255, 255, 255, 255]);
        assert!(at(0, HEIGHT / 2)[0] < 128);
        assert!(at(WIDTH - 1, HEIGHT / 2)[0] > 128);
        assert!(at(WIDTH / 2, HEIGHT / 2)[1] > at(WIDTH / 2, HEIGHT - 1)[1]);
        assert!(at(WIDTH / 2, HEIGHT * 3 / 4)[2] >= 254);
    }
}
