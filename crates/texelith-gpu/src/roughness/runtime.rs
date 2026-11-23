use super::*;
use crate::MaterialError;
use crate::PreparedRoughness;
use texelith_core::MaterialNodeParams;
use texelith_core::{MaterialNode, MaterialNodeParams as P};

impl Runtime {
    pub(super) fn roughness(
        &mut self,
        work: &Work,
        color: Lease,
        mask: Option<Lease>,
        extent: Extent,
        parameters: &MaterialNodeParams,
    ) -> Result<Lease> {
        let output = self
            .allocate(&[texture_desc(extent, wgpu::TextureFormat::R32Float)])?
            .remove(0);
        let pipeline = self
            .nodes
            .roughness
            .0
            .get_or_insert_with(|| crate::roughness::pipeline(&self.device));
        let mut prepared = PreparedRoughness::prepare(
            &self.executor,
            color.texture(),
            mask.as_ref().map(Lease::texture),
            output.texture(),
            parameters,
            self.pool.clone(),
            Some(pipeline),
        )?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let mut leases = vec![color, output.clone()];
        leases.extend(mask);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for pass in prepared.passes() {
            pass.encode(&mut encoder);
        }
        let (fence, done, _) = self.submit(encoder, &leases)?;
        prepared.mark_submitted(fence)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        prepared.complete(fence);
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(output)
    }
}

#[derive(Default)]
pub(super) struct State(Option<wgpu::ComputePipeline>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, P::Roughness { .. })
}
pub(super) fn scratch(
    node: &MaterialNode,
    _extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(match &node.params {
        P::Roughness { .. } => crate::roughness::SCRATCH_BYTES,
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work, node, extent, ..
    } = *context;
    Ok(match node.params {
        P::Roughness { .. } => runtime.roughness(
            work,
            context
                .input_at(texelith_core::InputPort::Color)
                .expect("validated color"),
            context.input_at(texelith_core::InputPort::Mask),
            extent,
            &node.params,
        )?,
        _ => unreachable!("registered parameters"),
    })
}
