use super::*;
use crate::MaterialError;
use crate::{AoParameters, AoResources, HorizonAoKernel};
use texelith_core::{MaterialNode, MaterialNodeParams as P};

impl Runtime {
    pub(super) fn horizon_ao(
        &mut self,
        work: &Work,
        input: Lease,
        source: Extent,
        params: AoParameters,
    ) -> Result<Lease> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let extent = Extent::new(input.texture().width(), input.texture().height())?;
        let table_bytes = params.sample_bytes()?;
        let leases = self.allocate(&[
            texture_desc(extent, wgpu::TextureFormat::R32Float),
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Buffer { bytes: table_bytes },
            AllocationDesc::Staging {
                bytes: 64 + table_bytes,
            },
        ])?;
        let key = (params.directions, params.steps);
        if !self.nodes.horizon_ao.0.contains_key(&key) {
            self.nodes.horizon_ao.0.insert(
                key,
                HorizonAoKernel::new(self.device.clone(), key.0, key.1)?,
            );
        }
        let pass = self.nodes.horizon_ao.0[&key].prepare(
            &self.queue,
            AoResources {
                input: input.texture(),
                output: leases[0].texture(),
                uniform: leases[1].buffer(),
                samples: leases[2].buffer(),
            },
            source,
            params,
        )?;
        let mut used = leases;
        used.push(input);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if self.current(work) {
            pass.encode(&mut encoder);
        }
        // Flush queued writes even when superseded during preparation. Retain
        // every ledger owner until the real submission fence acknowledges them.
        let (_, done, _) = self.submit(encoder, &used)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(used[0].clone())
    }
}

#[derive(Default)]
pub(super) struct State(HashMap<(u32, u32), HorizonAoKernel>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, P::HorizonAo { .. })
}
pub(super) fn scratch(
    node: &MaterialNode,
    _extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(match &node.params {
        P::HorizonAo {
            directions,
            steps,
            radius_source_px,
            bias,
            boundary,
        } => crate::AoParameters {
            directions: *directions,
            steps: *steps,
            radius_source_px: *radius_source_px,
            bias: *bias,
            boundary: *boundary,
        }
        .scratch_bytes()
        .map_err(|e| MaterialError::InvalidRequest(e.to_string()))?,
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work,
        node,
        source_extent,
        ..
    } = *context;
    let input = context.input();
    Ok(match node.params {
        P::HorizonAo {
            directions,
            steps,
            radius_source_px,
            bias,
            boundary,
        } => runtime.horizon_ao(
            work,
            input,
            source_extent,
            crate::AoParameters {
                directions,
                steps,
                radius_source_px,
                bias,
                boundary,
            },
        )?,
        _ => unreachable!("registered parameters"),
    })
}
