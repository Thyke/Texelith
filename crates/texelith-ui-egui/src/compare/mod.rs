//! Simple viewport intent, bounded comparison publication and asset IO.
pub(crate) mod assets;
pub mod labels;
mod runtime;
pub mod smoke;
pub(crate) mod snapshots;
pub(crate) use runtime::Viewport;
pub use runtime::material_request;
use texelith_core::{Camera, Extent};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

/// A maximum of one candidate image per side; only an identical intent publishes.
pub struct PairCache<T> {
    identity: u64,
    a: Option<T>,
    b: Option<T>,
}
impl<T> Default for PairCache<T> {
    fn default() -> Self {
        Self {
            identity: 0,
            a: None,
            b: None,
        }
    }
}
impl<T> PairCache<T> {
    pub fn begin(&mut self, identity: u64) {
        self.identity = identity;
        self.a = None;
        self.b = None;
    }
    pub fn insert(&mut self, side: Side, identity: u64, image: T) -> bool {
        if identity != self.identity {
            return false;
        }
        *match side {
            Side::A => &mut self.a,
            Side::B => &mut self.b,
        } = Some(image);
        true
    }
    pub fn pair(&self) -> Option<(&T, &T)> {
        Some((self.a.as_ref()?, self.b.as_ref()?))
    }
    pub(crate) fn take(&mut self) -> Option<(T, T)> {
        self.pair()?;
        Some((self.a.take()?, self.b.take()?))
    }
}
pub fn compare_extent(extent: Extent) -> Extent {
    texelith_gpu::proxy_extent(extent).expect("valid framebuffer")
}
pub fn orbit(mut camera: Camera, delta: [f32; 2], scroll: f32) -> Camera {
    camera.yaw += delta[0] * 0.008;
    camera.pitch = (camera.pitch + delta[1] * 0.008).clamp(-1.5, 1.5);
    camera.distance = (camera.distance * (-scroll * 0.002).exp()).clamp(0.3, 20.);
    camera
}

pub fn reset_camera(mesh: &texelith_core::PreviewMesh) -> Camera {
    frame_camera(mesh, 1.0)
}

pub(crate) fn frame_camera(mesh: &texelith_core::PreviewMesh, aspect: f32) -> Camera {
    let mut camera = Camera {
        yaw: if matches!(mesh, texelith_core::PreviewMesh::Plane) {
            0.45
        } else {
            0.6
        },
        pitch: if matches!(mesh, texelith_core::PreviewMesh::Plane) {
            0.85
        } else {
            0.3
        },
        ..Default::default()
    };
    let back = [
        camera.yaw.sin() * camera.pitch.cos(),
        camera.pitch.sin(),
        camera.yaw.cos() * camera.pitch.cos(),
    ];
    let right = [camera.yaw.cos(), 0., -camera.yaw.sin()];
    let up = [
        -camera.yaw.sin() * camera.pitch.sin(),
        camera.pitch.cos(),
        -camera.yaw.cos() * camera.pitch.sin(),
    ];
    let dot = |a: [f32; 3], b: [f32; 3]| a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if let Ok(mesh) = texelith_gpu::pbr::primitive(mesh) {
        // Fit the actual projected bounds to 70% of each framebuffer axis.
        camera.distance = mesh
            .vertices
            .iter()
            .map(|v| {
                dot(v.position, back)
                    + (dot(v.position, right).abs() / aspect.max(0.1))
                        .max(dot(v.position, up).abs())
                        * 2.4142137
                        / 0.7
            })
            .fold(0.0, f32::max);
    } else {
        camera.distance = 3.6 / aspect.clamp(0.1, 1.0);
    }
    camera
}

pub(crate) struct UiState {
    pub split_view: bool,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub mesh_path: String,
    pub imported_mesh: Option<texelith_core::PreviewMesh>,
    pub hdr_path: String,
    pub exposure: f32,
    pub revision: u64,
    pub compare_enabled: bool,
    pub timings: bool,
    pub pending: Option<assets::Load>,
    pub environment_enabled: bool,
    pub environment_rotation: f32,
    pub environment_intensity: f32,
    pub background_visible: bool,
    pub background_blur: f32,
    pub aspect: f32,
    pub turntable: bool,
    pub fit_pending: bool,
}
impl Default for UiState {
    fn default() -> Self {
        Self {
            split_view: false,
            zoom: 100.,
            pan: egui::Vec2::ZERO,
            mesh_path: String::new(),
            imported_mesh: None,
            hdr_path: String::new(),
            exposure: 0.,
            revision: 0,
            compare_enabled: false,
            timings: false,
            pending: None,
            environment_enabled: true,
            environment_rotation: 0.,
            environment_intensity: 1.,
            background_visible: true,
            background_blur: 0.,
            aspect: 1.,
            turntable: false,
            fit_pending: true,
        }
    }
}

pub(crate) fn active(c: &crate::controls::Controls) -> bool {
    let v = c.session().view_state();
    v.mode == texelith_core::ViewMode::ThreeD
        || (c.preview.compare_enabled && v.capture_a.is_some())
        || v.tiles.random_rotation
        // The CPU-presented map already supports tiled diagnostic overlays;
        // the GPU viewport's map renderer does not apply the seam heatmap.
        || (v.tiles.count != 1 && !c.heatmap)
}

#[cfg(test)]
mod framing_tests {
    use super::*;

    #[test]
    fn tiled_heatmap_uses_diagnostic_presenter_but_3d_keeps_gpu_viewport() {
        use crate::command_adapter::UiAction;
        use texelith_core::*;
        let mut c = crate::controls::Controls::new(true);
        c.action(UiAction::Request(CommandRequest::SetTiles {
            tiles: Tiles {
                count: 3,
                ..Default::default()
            },
        }));
        assert_eq!(c.session().view_state().tiles.count, 3);
        assert!(active(&c));
        c.heatmap = true;
        assert!(!active(&c));
        c.action(UiAction::Request(CommandRequest::SetMode {
            mode: ViewMode::ThreeD,
        }));
        assert!(active(&c));
    }

    #[test]
    fn reset_fits_seventy_percent_at_wide_and_tall_aspects() {
        use texelith_core::PreviewMesh::*;
        for mesh in [PreviewBall, RoundedCube, Plane, Sphere, Cube, Cylinder] {
            for aspect in [0.6, 1., 16. / 9.] {
                let c = frame_camera(&mesh, aspect);
                let back = [
                    c.yaw.sin() * c.pitch.cos(),
                    c.pitch.sin(),
                    c.yaw.cos() * c.pitch.cos(),
                ];
                let right = [c.yaw.cos(), 0., -c.yaw.sin()];
                let up = [
                    -c.yaw.sin() * c.pitch.sin(),
                    c.pitch.cos(),
                    -c.yaw.cos() * c.pitch.sin(),
                ];
                let dot = |a: [f32; 3], b: [f32; 3]| {
                    a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
                };
                let mut bound = 0.0f32;
                for v in texelith_gpu::pbr::primitive(&mesh).unwrap().vertices {
                    let z = c.distance - dot(v.position, back);
                    assert!(z > 0.01);
                    bound = bound.max(
                        (dot(v.position, right).abs() / aspect).max(dot(v.position, up).abs())
                            * 2.4142137
                            / z,
                    );
                }
                assert!(
                    (bound - 0.7).abs() < 1e-5,
                    "{mesh:?} aspect={aspect} fills {bound}"
                );
            }
        }
    }
}
