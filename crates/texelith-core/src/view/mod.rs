use crate::{MapKind, OpaqueLocator};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// CPU-side triangle mesh data for transfer to the preview renderer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Vertex attributes, including tangent handedness in the fourth component.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewMode {
    #[default]
    TwoD,
    ThreeD,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewMesh {
    #[default]
    PreviewBall,
    RoundedCube,
    Plane,
    Cube,
    Sphere,
    Cylinder,
    Custom(OpaqueLocator),
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            distance: 3.0,
            target: [0.0; 3],
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Light {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub environment: Option<OpaqueLocator>,
}
impl Default for Light {
    fn default() -> Self {
        Self {
            direction: [-0.4, 0.6, 1.0],
            intensity: 2.0,
            environment: None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tiles {
    pub count: u32,
    pub seed: u32,
    pub random_rotation: bool,
}
impl Default for Tiles {
    fn default() -> Self {
        Self {
            count: 1,
            seed: 0,
            random_rotation: false,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pom {
    pub enabled: bool,
    pub scale: f32,
}
impl Default for Pom {
    fn default() -> Self {
        Self {
            enabled: false,
            scale: 0.0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapView {
    pub visible: bool,
    pub locked: bool,
}
impl Default for MapView {
    fn default() -> Self {
        Self {
            visible: true,
            locked: false,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureIdentity {
    pub generation: u64,
    pub document_revision: u64,
    pub request_id: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewState {
    pub mode: ViewMode,
    pub mesh: PreviewMesh,
    pub light: Light,
    pub tiles: Tiles,
    pub pom: Pom,
    pub camera: Camera,
    pub maps: BTreeMap<MapKind, MapView>,
    pub capture_a: Option<CaptureIdentity>,
    pub capture_b: Option<CaptureIdentity>,
    pub split_fraction: f32,
}
impl Default for ViewState {
    fn default() -> Self {
        Self {
            mode: ViewMode::default(),
            mesh: PreviewMesh::default(),
            light: Light::default(),
            tiles: Tiles::default(),
            pom: Pom::default(),
            camera: Camera::default(),
            maps: BTreeMap::new(),
            capture_a: None,
            capture_b: None,
            split_fraction: 0.5,
        }
    }
}
