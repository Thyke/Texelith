use super::*;
use crate::{MaterialError, metallic::MetallicKernel};
use texelith_core::{InputPort, MaterialNode, MaterialNodeParams};

#[derive(Default)]
pub(super) struct State(Option<MetallicKernel>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, MaterialNodeParams::Metallic { .. })
}
pub(super) fn scratch(_: &MaterialNode, _: Extent) -> std::result::Result<u64, MaterialError> {
    Ok(crate::metallic::SCRATCH_BYTES)
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let color = context.input_at(InputPort::Color).expect("validated color");
    let mask = context.input_at(InputPort::Mask);
    // Admit output and uploads while all prior cached state remains evictable.
    let mut leases = runtime.allocate(&[
        texture_desc(context.extent, wgpu::TextureFormat::R32Float),
        AllocationDesc::Buffer { bytes: 64 },
        AllocationDesc::Staging { bytes: 64 },
    ])?;
    let output = leases[0].clone();
    let kernel = runtime
        .nodes
        .metallic
        .0
        .get_or_insert_with(|| MetallicKernel::new(&runtime.device, runtime.queue.clone()));
    let prepared = kernel.prepare_proxy(
        &runtime.device,
        color.texture(),
        output.texture(),
        mask.as_ref().map(Lease::texture),
        leases[1].buffer(),
        &context.node.params,
    )?;
    leases.push(color);
    leases.extend(mask);
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    prepared.encode(&mut encoder);
    let (_, done, _) = runtime.submit(encoder, &leases)?;
    runtime.shared.lock().unwrap().compute_submissions += 1;
    runtime.wait(&done)?;
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    Ok(output)
}
