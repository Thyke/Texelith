//! Checked framebuffer planning and the worker's fenced viewport adapter.
use crate::{CompletedMaterial, DisplayLease, GpuError, JobId, pool::AllocationDesc};
use std::sync::Arc;
use texelith_core::{Extent, ViewState};

#[derive(Clone)]
pub struct PreviewRequest {
    pub job_id: JobId,
    pub generation: u64,
    /// Monotonic view intent within a job, independent of material request IDs.
    pub request_id: u64,
    pub material: Arc<CompletedMaterial>,
    /// Actual framebuffer pixels, before the 2048 long-axis cap.
    pub extent: Extent,
    pub view: ViewState,
    pub content: crate::viewport_shared::PreviewContent,
}
impl std::fmt::Debug for PreviewRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreviewRequest")
            .field("job_id", &self.job_id)
            .field("generation", &self.generation)
            .field("request_id", &self.request_id)
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}
impl PreviewRequest {
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        ViewportTargetPlan::new(self.extent.width, self.extent.height)?;
        let view = &self.view;
        let camera = &view.camera;
        if self.job_id != self.material.job_id
            || self.generation != self.material.generation
            || ![
                camera.yaw,
                camera.pitch,
                camera.distance,
                view.light.intensity,
                view.pom.scale,
                view.split_fraction,
            ]
            .into_iter()
            .all(f32::is_finite)
            || !camera.target.into_iter().all(f32::is_finite)
            || !view.light.direction.into_iter().all(f32::is_finite)
            || camera.distance <= 0.0
            || view.light.intensity < 0.0
            || view.pom.scale < 0.0
            || !(0.0..=1.0).contains(&view.split_fraction)
            || !(1..=3).contains(&view.tiles.count)
        {
            return Err(GpuError("invalid preview identity or view".into()));
        }
        if let crate::viewport_shared::PreviewContent::Clear(color) = self.content
            && !color.into_iter().all(f32::is_finite)
        {
            return Err(GpuError("nonfinite preview clear".into()));
        }
        if let crate::viewport_shared::PreviewContent::Pbr(settings)
        | crate::viewport_shared::PreviewContent::Ibl { settings, .. }
        | crate::viewport_shared::PreviewContent::Mesh { settings, .. } = &self.content
        {
            settings.validate()?;
            if let crate::viewport_shared::PreviewContent::Mesh { mesh, .. } = &self.content {
                crate::compare::validate_mesh(mesh)?;
            } else {
                crate::pbr::mesh_index(&view.mesh)?;
            }
        }
        if let crate::viewport_shared::PreviewContent::Ibl { exposure, .. }
        | crate::viewport_shared::PreviewContent::Mesh { exposure, .. } = &self.content
            && (!exposure.is_finite() || !(-20.0..=20.0).contains(exposure))
        {
            return Err(GpuError("invalid environment exposure".into()));
        }
        Ok(())
    }
}

pub struct CompletedPreview {
    pub job_id: JobId,
    pub generation: u64,
    pub request_id: u64,
    pub extent: Extent,
    pub scale: f32,
    pub lease: DisplayLease,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportTargetPlan {
    pub extent: Extent,
    pub scale: f32,
}
impl ViewportTargetPlan {
    pub fn new(width: u32, height: u32) -> Result<Self, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError("empty framebuffer".into()));
        }
        let long = width.max(height);
        let cap = long.min(2048);
        let axis = |n: u32| ((u64::from(n) * u64::from(cap) / u64::from(long)) as u32).max(1);
        Ok(Self {
            extent: Extent::new(axis(width), axis(height))?,
            scale: cap as f32 / long as f32,
        })
    }
    /// HDR, depth, immutable display, in that order. All are single sampled.
    pub fn allocations(&self) -> [AllocationDesc; 3] {
        use wgpu::{TextureFormat as F, TextureUsages as U};
        [
            (F::Rgba16Float, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING),
            (F::Depth32Float, U::RENDER_ATTACHMENT),
            (
                F::Rgba8Unorm,
                U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC,
            ),
        ]
        .map(|(format, usage)| AllocationDesc::Texture {
            extent: self.extent,
            format,
            usage,
        })
    }
}
