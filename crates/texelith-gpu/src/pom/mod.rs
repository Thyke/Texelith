//! Bounded fragment-space height inspection. No silhouette or self-shadowing.
use texelith_core::{Boundary, Pom};

/// Material-space interpretation shared by the direct and IBL preview paths.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSettings {
    pub midlevel: f32,
    pub boundary: Boundary,
}
impl Default for SurfaceSettings {
    fn default() -> Self {
        Self {
            midlevel: 0.5,
            boundary: Boundary::Clamp,
        }
    }
}
impl SurfaceSettings {
    pub fn validate(self) -> Result<(), crate::GpuError> {
        if !self.midlevel.is_finite() || !(0.0..=1.0).contains(&self.midlevel) {
            return Err(crate::GpuError("invalid POM midlevel".into()));
        }
        Ok(())
    }
    /// Reserved material row 2: scale, midlevel, repeat flag, reserved.
    /// Extremely large finite user scales saturate before any GPU arithmetic.
    pub(crate) fn uniform(self, pom: Pom, has_height: bool) -> [f32; 4] {
        [
            if pom.enabled && has_height {
                pom.scale.clamp(0.0, 1.0)
            } else {
                0.0
            },
            self.midlevel,
            f32::from(self.boundary == Boundary::Repeat),
            0.0,
        ]
    }
}

/// Shared production source, also exercised directly by numerical GPU probes.
pub fn shader_source() -> &'static str {
    include_str!("../../shaders/pom/displacement.wgsl")
}
