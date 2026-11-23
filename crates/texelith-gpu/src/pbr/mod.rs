//! Direct, neutral metallic/roughness inspection shading (ADR 0008).
//! All GPU payloads are supplied by the shared worker's admission ledger.
mod mesh;
mod renderer;
pub use mesh::primitive;
pub(crate) use renderer::pipeline_with_source; // T-067 shared vertex/material ABI
pub(crate) use renderer::{display_pipeline, pipeline};
use texelith_core::{Camera, Extent, Light, NormalConvention};

/// Material multipliers/overrides for inspection. Maps remain authoritative.
#[derive(Clone, Copy, Debug)]
pub struct PbrSettings {
    /// Stops, applied to material radiance before the fixed tone curve.
    pub exposure: f32,
    pub surface: crate::pom::SurfaceSettings,
    pub base_color: [f32; 3],
    pub roughness: Option<f32>,
    pub metallic: Option<f32>,
    /// Environment yaw in radians; rotates background and indirect lighting together.
    pub environment_rotation: f32,
    /// Show the environment instead of the neutral solid background.
    pub background_visible: bool,
    /// Background roughness, independent of the material's specular roughness.
    pub background_blur: f32,
}
impl Default for PbrSettings {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            surface: Default::default(),
            base_color: [1.0; 3],
            roughness: None,
            metallic: None,
            environment_rotation: 0.0,
            background_visible: false,
            background_blur: 0.0,
        }
    }
}
impl PbrSettings {
    pub fn validate(self) -> Result<(), crate::GpuError> {
        self.surface.validate()?;
        if !self.exposure.is_finite()
            || !self.environment_rotation.is_finite()
            || !self.background_blur.is_finite()
            || !(0.0..=1.0).contains(&self.background_blur)
            || !(-20.0..=20.0).contains(&self.exposure)
            || self
                .base_color
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || [self.roughness, self.metallic]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            return Err(crate::GpuError(
                "invalid PBR exposure/material settings".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn scene(self, camera: Camera, light: &Light, extent: Extent) -> [[f32; 4]; 4] {
        // Scale before normalizing: finite, very large light directions cannot overflow.
        let s = light
            .direction
            .into_iter()
            .map(f32::abs)
            .fold(0.0, f32::max);
        let mut l = if s > 0.0 {
            light.direction.map(|v| v / s)
        } else {
            [0.0, 0.0, 1.0]
        };
        let len = l.into_iter().map(|v| v * v).sum::<f32>().sqrt();
        l = l.map(|v| v / len);
        [
            [
                camera.yaw,
                camera.pitch.clamp(-1.56, 1.56),
                camera.distance.clamp(0.01, 1e6),
                extent.width as f32 / extent.height as f32,
            ],
            [
                camera.target[0].clamp(-1e6, 1e6),
                camera.target[1].clamp(-1e6, 1e6),
                camera.target[2].clamp(-1e6, 1e6),
                self.exposure,
            ],
            [l[0], l[1], l[2], light.intensity.min(65504.0)],
            [
                0.0,
                self.environment_rotation.rem_euclid(std::f32::consts::TAU),
                f32::from(self.background_visible),
                self.background_blur,
            ],
        ]
    }
    pub(crate) fn material(self, convention: Option<NormalConvention>) -> [[f32; 4]; 4] {
        [
            [
                self.base_color[0],
                self.base_color[1],
                self.base_color[2],
                self.roughness.unwrap_or(-1.0),
            ],
            [
                self.metallic.unwrap_or(-1.0),
                if convention == Some(NormalConvention::DirectX) {
                    -1.0
                } else {
                    1.0
                },
                0.0,
                0.0,
            ],
            [0.0; 4],
            [0.0; 4],
        ]
    }
}

/// Original ACES-style rational fit: x(2.51x+.03)/(x(2.43x+.59)+.14).
/// Scaling by x for x>1 avoids overflow at high finite HDR inputs.
pub fn tone_map(x: f32) -> f32 {
    let x = x.max(0.0);
    if x > 1.0 {
        ((2.51 + 0.03 / x) / (2.43 + 0.59 / x + 0.14 / x / x)).clamp(0.0, 1.0)
    } else {
        (x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    }
}
pub fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// GGX BRDF times N.L, per unit white directional radiance. Cosines refer to
/// normalized shading vectors. AO deliberately has no place in direct lighting.
pub fn direct_brdf(
    albedo: [f32; 3],
    roughness: f32,
    metallic: f32,
    nv: f32,
    nl: f32,
    nh: f32,
    vh: f32,
) -> [f32; 3] {
    let (nv, nl, nh, vh) = (
        nv.clamp(0.0, 1.0),
        nl.clamp(0.0, 1.0),
        nh.clamp(0.0, 1.0),
        vh.clamp(0.0, 1.0),
    );
    if nv == 0.0 || nl == 0.0 {
        return [0.0; 3];
    }
    let alpha = roughness.clamp(0.0, 1.0).powi(2).max(0.0025);
    let a2 = alpha * alpha;
    let d = a2 / (std::f32::consts::PI * ((1.0 - nh * nh) + nh * nh * a2).powi(2)).max(1e-12);
    let visibility = 0.5
        / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt())
            .max(1e-8);
    let m = metallic.clamp(0.0, 1.0);
    albedo.map(|c| {
        let c = c.clamp(0.0, 1.0);
        let f0 = 0.04 * (1.0 - m) + c * m;
        let f = f0 + (1.0 - f0) * (1.0 - vh).powi(5);
        ((1.0 - f) * (1.0 - m) * c / std::f32::consts::PI + d * visibility * f) * nl
    })
}

pub(crate) struct PackedMesh {
    pub vertices: Vec<[f32; 12]>,
    pub indices: Vec<u32>,
}
pub(crate) struct Meshes(pub [PackedMesh; 6]);
impl Default for Meshes {
    fn default() -> Self {
        use texelith_core::PreviewMesh::*;
        Self(
            [Plane, Cube, Sphere, Cylinder, PreviewBall, RoundedCube].map(|kind| {
                let mesh = primitive(&kind).expect("built-in primitive");
                PackedMesh {
                    vertices: mesh
                        .vertices
                        .into_iter()
                        .map(|v| {
                            [
                                v.position[0],
                                v.position[1],
                                v.position[2],
                                v.normal[0],
                                v.normal[1],
                                v.normal[2],
                                v.uv[0],
                                v.uv[1],
                                v.tangent[0],
                                v.tangent[1],
                                v.tangent[2],
                                v.tangent[3],
                            ]
                        })
                        .collect(),
                    indices: mesh.indices,
                }
            }),
        )
    }
}
pub(crate) fn mesh_index(mesh: &texelith_core::PreviewMesh) -> Result<usize, crate::GpuError> {
    use texelith_core::PreviewMesh::*;
    match mesh {
        Plane => Ok(0),
        Cube => Ok(1),
        Sphere => Ok(2),
        Cylinder => Ok(3),
        PreviewBall => Ok(4),
        RoundedCube => Ok(5),
        Custom(_) => Err(crate::GpuError(
            "custom mesh bridge is not installed (T-069)".into(),
        )),
    }
}
