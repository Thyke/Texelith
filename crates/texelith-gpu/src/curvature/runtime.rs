use super::*;
use crate::{CurvatureParameters, MaterialError, PreparedCurvature};
use texelith_core::{MaterialNode, MaterialNodeParams as P};

#[derive(Default)]
pub(super) struct State(Option<wgpu::ComputePipeline>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, P::Curvature { .. })
}
pub(super) fn scratch(
    _node: &MaterialNode,
    _extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(crate::curvature::SCRATCH_BYTES)
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let edge = &context.node.inputs[0];
    let convention = context
        .work
        .material
        .as_ref()
        .expect("material work")
        .snapshot
        .graph
        .output_normal_convention(edge.source)?
        .ok_or_else(|| GpuError("missing normal convention".into()))?;
    let parameters =
        CurvatureParameters::from_material(&context.node.params, convention, context.output.port)?;
    let input = context.input();
    let output = runtime
        .allocate(&[texture_desc(context.extent, wgpu::TextureFormat::R32Float)])?
        .remove(0);
    let pipeline = runtime
        .nodes
        .curvature
        .0
        .get_or_insert_with(|| crate::curvature::pipeline(&runtime.device));
    let mut prepared = PreparedCurvature::prepare(
        &runtime.executor,
        input.texture(),
        output.texture(),
        &parameters,
        runtime.pool.clone(),
        Some(pipeline),
    )?;
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    for pass in prepared.passes() {
        pass.encode(&mut encoder);
    }
    let (fence, done, _) = runtime.submit(encoder, &[input, output.clone()])?;
    prepared.mark_submitted(fence)?;
    runtime.shared.lock().unwrap().compute_submissions += 1;
    runtime.wait(&done)?;
    prepared.complete(fence);
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    Ok(output)
}
