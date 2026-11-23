use crate::{Axis, GpuError};
use texelith_core::{Boundary, CacheKey, Extent, SourceDescriptor};

/// Scalar shared primitives. Each pass writes a distinct caller-owned R32 texture.
#[derive(Clone, Copy, Debug)]
pub enum PyramidPass {
    Gaussian(Axis),
    AreaDownsample,
    /// Vertical five-tap filtering followed by exact area reduction. Fusing the
    /// two preserves the R32 arithmetic and saves a full-size scratch texture.
    GaussianAreaDownsample,
    BilinearUpsample,
    LaplacianBand,
    WeightedReconstruct {
        band_weight: f32,
        coarse_weight: f32,
    },
}

pub fn pyramid_extents(source: Extent) -> Result<Vec<Extent>, GpuError> {
    source.validate()?;
    let mut levels = vec![source];
    let mut extent = source;
    while extent.width != 1 || extent.height != 1 {
        extent = Extent::new(extent.width.div_ceil(2), extent.height.div_ceil(2))?;
        levels.push(extent);
    }
    Ok(levels)
}

/// Consumer-independent identity, including the canonical source interpretation.
pub fn pyramid_cache_key(
    source: &SourceDescriptor,
    boundary: Boundary,
    filter_revision: u32,
) -> Result<CacheKey, GpuError> {
    source.extent.validate()?;
    let mut bytes = Vec::from(&b"texelith.pyramid.v1\0\0"[..]);
    bytes.extend_from_slice(&source.content_hash);
    let interpretation: u32 = match source.interpretation {
        texelith_core::InputInterpretation::SrgbPhoto => 0,
        texelith_core::InputInterpretation::LinearHeight => 1,
        texelith_core::InputInterpretation::LinearNormal => 2,
    };
    bytes.extend_from_slice(&interpretation.to_le_bytes());
    for value in [
        source.extent.width,
        source.extent.height,
        filter_revision,
        u32::from(boundary == Boundary::Repeat),
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    // Reuse the core BLAKE3 boundary without changing its frozen legacy encoding.
    Ok(CacheKey(
        SourceDescriptor::from_rgba8(
            &bytes,
            Extent::new((bytes.len() / 4) as u32, 1)?,
            source.interpretation,
        )?
        .content_hash,
    ))
}

use crate::{
    GpuExecutor, PreparedPass,
    pool::{AllocationDesc, AllocationId, FenceId, ResourcePool},
};
use std::sync::{Arc, Mutex};

/// Persistent Gaussian chain. The caller drives submissions and acknowledges real
/// queue fences; it may check its worker token between any two encoded passes.
/// Source texture residency must already be charged to the same ledger.
pub struct PreparedPyramid {
    pub key: CacheKey,
    levels: Vec<wgpu::Texture>,
    passes: Vec<PreparedPass>,
    pub(crate) _scratch: Vec<wgpu::Texture>,
    reservations: PyramidReservations,
    scratch_bytes: u64,
    submitted: Option<FenceId>,
    ready: bool,
}
struct PyramidReservations {
    pool: Arc<Mutex<ResourcePool>>,
    ids: Vec<AllocationId>,
}
impl Drop for PyramidReservations {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            let _ = pool.release(*id);
        }
    }
}
impl PreparedPyramid {
    pub fn new(
        executor: &GpuExecutor,
        source: &SourceDescriptor,
        input: &wgpu::Texture,
        boundary: Boundary,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        let extents = pyramid_extents(source.extent)?;
        if input.width() != source.extent.width
            || input.height() != source.extent.height
            || input.format() != wgpu::TextureFormat::R32Float
            || input.dimension() != wgpu::TextureDimension::D2
            || input.depth_or_array_layers() != 1
            || input.mip_level_count() != 1
            || input.sample_count() != 1
            || !input.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(GpuError(
                "pyramid source must be prepared scalar data".into(),
            ));
        }
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let desc = |extent| AllocationDesc::Texture {
            extent,
            format: wgpu::TextureFormat::R32Float,
            usage,
        };
        let mut plan: Vec<_> = extents[1..].iter().map(|e| desc(*e)).collect();
        let mut scratch_bytes = plan.iter().try_fold(0u64, |sum, d| {
            sum.checked_add(d.byte_len().map_err(|e| GpuError(e.to_string()))?)
                .ok_or_else(|| GpuError("pyramid byte count overflow".into()))
        })?;
        for e in &extents[..extents.len() - 1] {
            let d = desc(*e);
            let bytes = d
                .byte_len()
                .map_err(|e| GpuError(e.to_string()))?
                .checked_add(4 * 64)
                .ok_or_else(|| GpuError("pyramid byte count overflow".into()))?;
            scratch_bytes = scratch_bytes
                .checked_add(bytes)
                .ok_or_else(|| GpuError("pyramid byte count overflow".into()))?;
            plan.push(d);
            plan.extend([AllocationDesc::Buffer { bytes: 64 }; 2]);
            plan.push(AllocationDesc::Staging { bytes: 2 * 64 });
        }
        let resources = pool
            .lock()
            .unwrap()
            .allocate_backed(executor.device(), &plan)
            .map_err(|e| GpuError(e.to_string()))?;
        let reservations = PyramidReservations {
            pool,
            ids: resources.iter().map(|(id, _)| *id).collect(),
        };
        let mut levels = vec![input.clone()];
        levels.extend(
            resources[..extents.len() - 1]
                .iter()
                .map(|(_, r)| r.texture().clone()),
        );
        let mut scratch = Vec::new();
        let mut passes = Vec::new();
        for i in 0..levels.len() - 1 {
            let offset = extents.len() - 1 + 4 * i;
            let x = resources[offset].1.texture().clone();
            passes.push(executor.prepare_pyramid_with_buffer(
                PyramidPass::Gaussian(Axis::X),
                &levels[i],
                None,
                &x,
                boundary,
                Some(resources[offset + 1].1.buffer()),
            )?);
            passes.push(executor.prepare_pyramid_with_buffer(
                PyramidPass::GaussianAreaDownsample,
                &x,
                None,
                &levels[i + 1],
                boundary,
                Some(resources[offset + 2].1.buffer()),
            )?);
            scratch.push(x);
        }
        Ok(Self {
            key: pyramid_cache_key(source, boundary, 1)?,
            levels,
            passes,
            _scratch: scratch,
            reservations,
            scratch_bytes,
            submitted: None,
            ready: false,
        })
    }
    pub fn levels(&self) -> &[wgpu::Texture] {
        &self.levels
    }
    pub fn passes(&self) -> &[PreparedPass] {
        &self.passes
    }
    /// All pyramid reservations: shared coarse levels, scratch, uniforms and upload staging.
    /// Only the caller's prepared source is excluded.
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
    /// Marks every retained allocation as used by this queue submission.
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        let mut pool = self.reservations.pool.lock().unwrap();
        for id in &self.reservations.ids {
            pool.mark_submitted(*id, fence)
                .map_err(|e| GpuError(e.to_string()))?;
        }
        self.submitted = Some(fence);
        self.ready = false;
        Ok(())
    }
    /// Call only after the matching queue fence. Cancelled/partial chains must
    /// pass `completed_chain = false`, and cannot be published as a cache hit.
    pub fn complete(&mut self, fence: FenceId, completed_chain: bool) {
        self.reservations.pool.lock().unwrap().complete(fence);
        if self.submitted == Some(fence) {
            self.ready = completed_chain;
        }
    }
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}
