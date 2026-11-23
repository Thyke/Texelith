//! Bounded material-set readback. Each plane is owned by one IO consumer until
//! acknowledgement; taking it from the mailbox does not admit another plane.
use crate::{
    ExportPackingMetadata, GpuError, JobId,
    pool::{AllocationId, ResourcePool},
};
use std::sync::{Arc, Mutex};
use texelith_core::{ExportCapture, ExportPlan, Extent, validate_export_for_graph};

#[derive(Clone, Debug)]
pub struct ExportRequest {
    pub(crate) job_id: JobId,
    pub(crate) capture: Arc<ExportCapture>,
    pub(crate) plan: Arc<ExportPlan>,
}
impl ExportRequest {
    pub fn new(job_id: JobId, capture: ExportCapture, plan: ExportPlan) -> Result<Self, GpuError> {
        validate_export_for_graph(&plan, &capture.graph)?;
        if !capture.graph.is_material() {
            return Err(GpuError(
                "material export requires a material graph; use request_save for legacy graphs"
                    .into(),
            ));
        }
        for output in &plan.outputs {
            crate::export_shared::validate_format(output.format)?;
        }
        Ok(Self {
            job_id,
            capture: Arc::new(capture),
            plan: Arc::new(plan),
        })
    }
    pub fn job_id(&self) -> JobId {
        self.job_id
    }
    pub fn capture(&self) -> &ExportCapture {
        &self.capture
    }
    pub fn plan(&self) -> &ExportPlan {
        &self.plan
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportStage {
    Evaluating,
    Readback,
    Encoding,
    ReadyToPublish,
    Publishing,
    Completed,
    Cancelled,
    Failed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportProgress {
    pub job_id: JobId,
    pub item_index: usize,
    pub output_name: String,
    pub total_items: usize,
    pub stage: ExportStage,
    pub error: Option<String>,
}

/// PNG16 bytes are big-endian; EXR32 bytes are little-endian IEEE f32. PNG8
/// bytes are UNORM8. File encoding/publication belongs to IO, not this worker.
pub struct ExportPlane {
    pub job_id: JobId,
    pub item_index: usize,
    pub output_name: String,
    pub generation: u64,
    pub graph_hash: [u8; 32],
    pub document_revision: u64,
    pub format: texelith_core::ExportFormat,
    pub channels: u8,
    pub payload: crate::export_shared::ExportPayload,
    pub metadata: ExportPackingMetadata,
    pub pixels: Vec<u8>,
    pub(crate) charge: Arc<PlaneCharge>,
}
pub(crate) struct PlaneCharge {
    pub pool: Arc<Mutex<ResourcePool>>,
    pub id: AllocationId,
    pub job: JobId,
    pub item: usize,
}
impl Drop for PlaneCharge {
    fn drop(&mut self) {
        let _ = self.pool.lock().unwrap().release_transient(self.id);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlaneLayout {
    pub extent: Extent,
    pub channels: u8,
    pub bit_depth: u8,
    pub padded_bytes_per_row: u32,
    pub staging_bytes: u64,
    pub packed_bytes: u64,
}
impl PlaneLayout {
    pub fn new(extent: Extent, channels: u8, bit_depth: u8) -> Result<Self, GpuError> {
        extent.validate()?;
        if !matches!(channels, 1 | 3 | 4) || !matches!(bit_depth, 8 | 16 | 32) {
            return Err(GpuError("unsupported numeric plane layout".into()));
        }
        let padded_bytes_per_row = (extent.width * 16).div_ceil(256) * 256;
        Ok(Self {
            extent,
            channels,
            bit_depth,
            padded_bytes_per_row,
            staging_bytes: u64::from(padded_bytes_per_row) * u64::from(extent.height),
            packed_bytes: u64::from(extent.width)
                * u64::from(extent.height)
                * u64::from(channels)
                * u64::from(bit_depth / 8),
        })
    }
}
/// Strip the RGBA32Float carrier pitch and unused channels without applying
/// transfer, displacement, dither or normal conversion a second time.
pub fn strip_carrier(mapped: &[u8], layout: PlaneLayout) -> Result<Vec<u8>, GpuError> {
    let checked = PlaneLayout::new(layout.extent, layout.channels, layout.bit_depth)?;
    if checked.staging_bytes != layout.staging_bytes
        || checked.padded_bytes_per_row != layout.padded_bytes_per_row
        || checked.packed_bytes != layout.packed_bytes
        || mapped.len() as u64 != checked.staging_bytes
    {
        return Err(GpuError("invalid readback row layout or length".into()));
    }
    let mut pixels = Vec::with_capacity(checked.packed_bytes as usize);
    for row in mapped.chunks_exact(checked.padded_bytes_per_row as usize) {
        for pixel in row[..checked.extent.width as usize * 16]
            .as_chunks::<16>()
            .0
        {
            for bytes in pixel
                .as_chunks::<4>()
                .0
                .iter()
                .take(checked.channels as usize)
            {
                let value = f32::from_le_bytes(*bytes);
                if !value.is_finite() {
                    return Err(GpuError("nonfinite export value".into()));
                }
                match checked.bit_depth {
                    8 => pixels.push((value.clamp(0.0, 1.0) * 255.0).round() as u8),
                    16 => pixels.extend_from_slice(
                        &((value.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes(),
                    ),
                    32 => pixels.extend_from_slice(bytes),
                    _ => unreachable!(),
                }
            }
        }
    }
    Ok(pixels)
}

#[derive(Default)]
pub(crate) struct Mailbox {
    pub charge: std::sync::Weak<PlaneCharge>,
    pub request: Option<ExportRequest>,
    pub next_item: usize,
    pub plane: Option<ExportPlane>,
    pub outstanding: bool,
    pub delivered: bool,
    pub progress: Option<ExportProgress>,
    pub stage: Option<ExportStage>,
    pub last_job: Option<JobId>,
    pub retiring: bool,
}
impl Mailbox {
    pub fn progress(&mut self, stage: ExportStage, error: Option<String>) {
        if let Some(request) = &self.request {
            self.stage = Some(stage);
            self.progress = Some(ExportProgress {
                job_id: request.job_id,
                item_index: self.next_item,
                output_name: request
                    .plan
                    .outputs
                    .get(self.next_item)
                    .map_or_else(String::new, |o| o.name.clone()),
                total_items: request.plan.outputs.len(),
                stage,
                error,
            });
        }
    }
    pub fn cancel(&mut self) {
        if self.request.is_some() {
            self.progress(ExportStage::Cancelled, None);
            self.request = None;
            self.plane = None;
            self.outstanding = false;
            self.delivered = false;
        }
    }
}
