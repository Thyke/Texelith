//! Shared-device compute primitives for ADR 0003.
//!
//! Construct on the render worker, prepare bindings when resources change, then
//! reuse `PreparedPass::encode`. Encoding never submits, polls, maps, or waits.
//! `GpuWorker` owns bounded scheduling, allocation budgets and queue retirement.
//! `GpuExecutor` remains available for independently scheduled kernel checks.
//! Source input is owned decoded RGBA8, numeric height, or encoded RGB normals.
//! Alpha is ignored; ICC/EXIF handling and file decoding remain outside this boundary.

pub mod cache;
mod executor;
mod params;
pub mod pool;
pub mod scheduler;
mod worker;
pub use worker::*;
mod material;
mod pyramid;
mod resources;
mod sources;
pub use material::{
    CompletedMaterial, JobId, MaterialError, MaterialPlan, MaterialRequest, ResourceReservations,
};
pub use pyramid::{PreparedPyramid, PyramidPass, pyramid_cache_key, pyramid_extents};
pub use sources::{TypedSourcePixels, TypedSourceUpload};

pub use executor::{GpuExecutor, PreparedPass};
pub use params::{Axis, BlurWeights, NodeUniform, Pass, proxy_extent, workgroups};

#[derive(Clone, Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

impl From<texelith_core::CoreError> for GpuError {
    fn from(error: texelith_core::CoreError) -> Self {
        Self(error.to_string())
    }
}

mod node_registry;
macro_rules! declare_nodes {
    ($( $visibility:vis $module:ident { $($export:ident),* } [$($state:ident, $runtime:literal)?] [$($suite:literal, $runner:literal, [$($validator:ident, $validation:literal)?])?]; )*) => {
        $($visibility mod $module;
        $(pub use $module::$export;)*)*
    };
}
gpu_nodes!(declare_nodes);

// FAZ1_GPU_MODULES: T-063 begin
pub mod viewport_resources;
pub mod viewport_shared;
pub use viewport_resources::{CompletedPreview, PreviewRequest};
// FAZ1_GPU_MODULES: T-063 end
// FAZ1_GPU_MODULES: T-065 begin
pub mod pbr;
// FAZ1_GPU_MODULES: T-065 end

// FAZ1_GPU_MODULES: T-079 begin
pub mod export_readback;
pub mod export_shared;
pub use export_readback::{ExportPlane, ExportProgress, ExportRequest, ExportStage};
// FAZ1_GPU_MODULES: T-079 end
// FAZ1_GPU_MODULES: T-067 begin
pub mod ibl;
// FAZ1_GPU_MODULES: T-067 end

// FAZ1_GPU_MODULES: T-081 begin
pub mod bc5;
// FAZ1_GPU_MODULES: T-081 end
// FAZ1_GPU_MODULES: T-068 begin
pub mod pom;
// FAZ1_GPU_MODULES: T-068 end

// FAZ1_GPU_MODULES: T-069
pub mod compare;
