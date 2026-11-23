//! Live format adapter, shared with the BC5 producer (T-081).
//! Numeric planes and header-free BC5 mip chains share job and readback bounds.
use crate::{
    ExportPackingKernel, ExportPackingSource, GpuError, GpuExecutor, PreparedExportPacking,
    export_readback::{PlaneLayout, strip_carrier},
    pool::{FenceId, ResourcePool},
};
use std::sync::{Arc, Mutex};
use texelith_core::{ExportFormat, ExportPlan, Extent, SizePolicy};

/// Layout of bytes delivered to IO. A BC5 producer supplies logical mip
/// extents and offsets; DDS container headers remain an IO responsibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportPayload {
    Numeric { channels: u8, bit_depth: u8 },
    Bc5 { mips: Vec<BlockMip> },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockMip {
    pub extent: Extent,
    pub offset: u64,
    pub byte_len: u64,
}

/// Physical staging geometry, independent of logical image/mip dimensions.
/// The adapter must describe every padded row and all returned CPU bytes.
pub(crate) struct ReadbackLayout {
    pub extent: Extent,
    pub bytes_per_pixel: u32,
    pub staging_bytes: u64,
    pub packed_bytes: u64,
    pub channels: u8,
}

/// Format-specific compute and copy hooks. T-081 can add its codec here without
/// modifying the immutable job scheduler or private mapping/retirement code.
pub(crate) struct PreparedOutput {
    packing: Option<PreparedExportPacking>,
    bc5: Option<crate::bc5::PreparedBc5>,
    metadata: crate::ExportPackingMetadata,
    numeric: PlaneLayout,
}
impl PreparedOutput {
    pub fn metadata(&self) -> &crate::ExportPackingMetadata {
        &self.metadata
    }
    pub fn pass_count(&self) -> usize {
        self.packing.as_ref().map_or(0, |p| p.passes().len())
            + self.bc5.as_ref().map_or(0, |p| p.pass_count())
    }
    pub fn encode_pass(&self, index: usize, encoder: &mut wgpu::CommandEncoder) {
        let packing = self.packing.as_ref().expect("prepared compute");
        if index < packing.passes().len() {
            packing.passes()[index].encode(encoder);
        } else {
            self.bc5
                .as_ref()
                .unwrap()
                .encode_pass(index - packing.passes().len(), encoder);
        }
    }
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError> {
        if let Some(packing) = &mut self.packing {
            packing.mark_submitted(fence)?;
        }
        if let Some(bc5) = &mut self.bc5 {
            bc5.mark_submitted(fence)?;
        }
        Ok(())
    }
    /// Called only after the final compute fence. Release scratch and bindings
    /// before staging admission; encoded output backing must stay until copy.
    pub fn finish_compute(&mut self) {
        self.packing = None;
        if let Some(bc5) = &mut self.bc5 {
            bc5.finish_compute();
        }
    }
    pub fn readback_layout(&self) -> ReadbackLayout {
        if let Some(bc5) = &self.bc5 {
            return ReadbackLayout {
                // Physical buffer geometry, not an image extent. One 256-byte
                // row per chunk matches the existing bounded readback allocator.
                extent: Extent {
                    width: 16,
                    height: (bc5.staging_bytes() / 256) as u32,
                },
                bytes_per_pixel: 16,
                staging_bytes: bc5.staging_bytes(),
                packed_bytes: bc5.byte_len(),
                channels: 2,
            };
        }
        ReadbackLayout {
            extent: self.numeric.extent,
            bytes_per_pixel: 16,
            staging_bytes: self.numeric.staging_bytes,
            packed_bytes: self.numeric.packed_bytes,
            channels: self.numeric.channels,
        }
    }
    pub fn payload(&self) -> ExportPayload {
        if let Some(bc5) = &self.bc5 {
            return ExportPayload::Bc5 {
                mips: bc5.mips().to_vec(),
            };
        }
        ExportPayload::Numeric {
            channels: self.numeric.channels,
            bit_depth: self.numeric.bit_depth,
        }
    }
    pub fn encode_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        carrier: &wgpu::Texture,
        buffer: &wgpu::Buffer,
    ) {
        if let Some(bc5) = &self.bc5 {
            bc5.encode_readback(encoder, buffer);
            return;
        }
        encoder.copy_texture_to_buffer(
            carrier.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.numeric.padded_bytes_per_row),
                    rows_per_image: Some(self.numeric.extent.height),
                },
            },
            wgpu::Extent3d {
                width: self.numeric.extent.width,
                height: self.numeric.extent.height,
                depth_or_array_layers: 1,
            },
        );
    }
    pub fn readback_bytes(&self, mapped: &[u8]) -> Result<Vec<u8>, GpuError> {
        if let Some(bc5) = &self.bc5 {
            return bc5.readback_bytes(mapped);
        }
        strip_carrier(mapped, self.numeric)
    }
}

pub(crate) fn source_boundary(
    graph: &texelith_core::Graph,
    reference: texelith_core::OutputRef,
) -> texelith_core::Boundary {
    use texelith_core::{Boundary, MaterialNodeParams as P};
    let mut node = graph
        .material_node(reference.node)
        .expect("validated export reference");
    loop {
        match &node.params {
            P::HeightFromAlbedo { boundary, .. }
            | P::NormalFromHeight { boundary, .. }
            | P::HeightFromNormal { boundary, .. }
            | P::HorizonAo { boundary, .. }
            | P::Curvature { boundary, .. }
            | P::PaintMask { boundary, .. }
            | P::Levels { boundary, .. }
            | P::Blur { boundary, .. }
            | P::Seamless { boundary, .. } => return *boundary,
            _ => match node.inputs.first() {
                Some(edge) => {
                    node = graph
                        .material_node(edge.source.node)
                        .expect("validated parent")
                }
                None => return Boundary::Clamp,
            },
        }
    }
}

pub fn validate_format(format: ExportFormat) -> Result<(), GpuError> {
    match format {
        ExportFormat::Png8 | ExportFormat::Png16 | ExportFormat::Exr32 | ExportFormat::DdsBc5 => {
            Ok(())
        }
    }
}
pub fn plane_layout(plan: &ExportPlan, item: usize) -> Result<PlaneLayout, GpuError> {
    let output = plan
        .outputs
        .get(item)
        .ok_or_else(|| GpuError("invalid export item".into()))?;
    validate_format(output.format)?;
    let extent = match plan.size {
        SizePolicy::Keep => plan.source_extent,
        SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
            Extent::new(width, height)?
        }
    };
    PlaneLayout::new(
        extent,
        if output.format == ExportFormat::DdsBc5 {
            3
        } else {
            output.channels.len() as u8
        },
        output.format.bit_depth(),
    )
}
pub(crate) fn prepare(
    kernel: &ExportPackingKernel,
    executor: &GpuExecutor,
    plan: &ExportPlan,
    item: usize,
    sources: &[ExportPackingSource<'_>],
    carrier: &wgpu::Texture,
    pool: Arc<Mutex<ResourcePool>>,
) -> Result<PreparedOutput, GpuError> {
    let numeric = plane_layout(plan, item)?;
    // Preserve signed XYZ through packing/resampling. The internal carrier has
    // 16-bit precision; BC5 quantizes XY only after normal-aware mip generation.
    let mut codec_plan;
    let packing_plan = if plan.outputs[item].format == ExportFormat::DdsBc5 {
        codec_plan = plan.clone();
        let output = &mut codec_plan.outputs[item];
        let mut z = output.channels[0];
        if let texelith_core::ChannelSource::Map(reference, _) = z.source {
            z.source = texelith_core::ChannelSource::Map(reference, texelith_core::Component::B);
        }
        z.invert = false;
        output.channels.push(z);
        output.format = ExportFormat::Png16;
        output.quantization = None;
        output.import_hints = None;
        &codec_plan
    } else {
        plan
    };
    let bc5 = if plan.outputs[item].format == ExportFormat::DdsBc5 {
        let source = match plan.outputs[item].channels[0].source {
            texelith_core::ChannelSource::Map(reference, _) => sources
                .iter()
                .find(|s| s.reference == reference)
                .map(|s| s.texture),
            _ => None,
        };
        Some(crate::bc5::Bc5Kernel::new(executor.device()).prepare(
            executor.device(),
            carrier,
            source,
            crate::bc5::Bc5Quality::Default,
            pool.clone(),
        )?)
    } else {
        None
    };
    // Admit codec storage before packing queues uniform writes. If either
    // admission fails, no unsubmitted queue-write payload escapes its charge.
    let packing = kernel.prepare(executor, packing_plan, item, sources, carrier, pool)?;
    let mut metadata = packing.metadata().clone();
    metadata.semantics = plan.outputs[item].semantics();
    Ok(PreparedOutput {
        metadata,
        packing: Some(packing),
        bc5,
        numeric,
    })
}
