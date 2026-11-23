use super::*;
use crate::{
    MaterialError,
    paint_mask::{MAX_SEGMENTS_PER_BATCH, PaintKernel, SEGMENT_BYTES},
};
use texelith_core::{Boundary, BrushStroke, MaterialNode, MaterialNodeParams, stroke_digest};

struct Checkpoint {
    extent: Extent,
    boundary: Boundary,
    count: usize,
    digest: [u8; 32],
    output: Lease,
}
#[derive(Default)]
pub(super) struct State {
    kernel: Option<PaintKernel>,
    checkpoint: Option<Checkpoint>,
}
impl State {
    pub(super) fn evict(&mut self) -> bool {
        self.checkpoint.take().is_some()
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, MaterialNodeParams::PaintMask { .. })
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    if let MaterialNodeParams::PaintMask {
        strokes,
        stroke_digest,
        ..
    } = &node.params
        && strokes.is_empty()
        && *stroke_digest != [0; 32]
    {
        return Err(MaterialError::InvalidRequest(
            "paint stroke digest has no vector replay data".into(),
        ));
    }
    // Pixel coverage/mask (8 B), one evictable immutable prefix (4 B),
    // segment buffer + queue-write staging, uniform + staging, zero dummy.
    u64::from(extent.width)
        .checked_mul(u64::from(extent.height))
        .and_then(|n| n.checked_mul(12))
        .and_then(|n| n.checked_add(2 * SEGMENT_BYTES * MAX_SEGMENTS_PER_BATCH as u64 + 128 + 4))
        .ok_or(MaterialError::Overflow)
}
fn prefix(
    checkpoint: &Checkpoint,
    extent: Extent,
    boundary: Boundary,
    strokes: &[BrushStroke],
) -> bool {
    checkpoint.extent == extent
        && checkpoint.boundary == boundary
        && checkpoint.count <= strokes.len()
        && stroke_digest(&strokes[..checkpoint.count])
            .is_ok_and(|digest| digest == checkpoint.digest)
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let MaterialNodeParams::PaintMask {
        strokes,
        boundary,
        stroke_digest,
        ..
    } = &context.node.params
    else {
        unreachable!("registered paint")
    };
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    // Prior checkpoint stays reachable by Runtime::allocate's eviction loop.
    let pixel_bytes = u64::from(context.extent.width) * u64::from(context.extent.height) * 8;
    let leases = runtime.allocate(&[
        texture_desc(context.extent, wgpu::TextureFormat::R32Float),
        texture_desc(Extent::new(1, 1)?, wgpu::TextureFormat::R32Float),
        AllocationDesc::Staging {
            bytes: pixel_bytes + 2 * SEGMENT_BYTES * MAX_SEGMENTS_PER_BATCH as u64 + 128,
        },
    ])?;
    let output = leases[0].clone();
    let state = &mut runtime.nodes.paint_mask;
    let checkpoint = state
        .checkpoint
        .as_ref()
        .filter(|c| prefix(c, context.extent, *boundary, strokes));
    let start = checkpoint.map_or(0, |c| c.count);
    let input = checkpoint.map_or_else(|| leases[1].clone(), |c| c.output.clone());
    if start > 0 {
        crate::paint_mask::checkpoint_hit(pixel_bytes / 2);
    }
    let kernel = state
        .kernel
        .get_or_insert_with(|| PaintKernel::new(&runtime.device));
    let prepared = kernel.prepare(&runtime.device, input.texture(), output.texture())?;
    let mut retained = leases;
    retained.push(input);
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    prepared.initialize(&runtime.queue, &mut encoder, start > 0);
    let (_, done, _) = runtime.submit(encoder, &retained)?;
    runtime.shared.lock().unwrap().compute_submissions += 1;
    runtime.wait(&done)?;
    for stroke in &strokes[start..] {
        let mut offset = 0;
        loop {
            if !runtime.current(context.work) {
                return Err(Failure::Cancelled);
            }
            let mut encoder = runtime.device.create_command_encoder(&Default::default());
            offset = prepared.batch(&runtime.queue, &mut encoder, stroke, *boundary, offset);
            let (_, done, _) = runtime.submit(encoder, &retained)?;
            runtime.shared.lock().unwrap().compute_submissions += 1;
            runtime.wait(&done)?;
            if offset >= stroke.points.len() {
                break;
            }
        }
    }
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    prepared.finish(&runtime.queue, &mut encoder);
    let (_, done, _) = runtime.submit(encoder, &retained)?;
    runtime.shared.lock().unwrap().compute_submissions += 1;
    runtime.wait(&done)?;
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    runtime.nodes.paint_mask.checkpoint = Some(Checkpoint {
        extent: context.extent,
        boundary: *boundary,
        count: strokes.len(),
        digest: *stroke_digest,
        output: output.clone(),
    });
    Ok(output)
}
