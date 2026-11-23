use super::*;
use crate::MaterialError;
use crate::seamless_offset::{OffsetKernels, OffsetTransform, update};
use texelith_core::Boundary;
use texelith_core::{MaterialNode, MaterialNodeParams as P};

impl Runtime {
    pub(super) fn seamless_offset(
        &mut self,
        work: &Work,
        input: Lease,
        extent: Extent,
        overlap: f32,
        boundary: Boundary,
    ) -> Result<Lease> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let transform = OffsetTransform::new(overlap)?;
        let mut leases = self.allocate(&[
            texture_desc(extent, input.texture().format()),
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Staging { bytes: 64 },
        ])?;
        let key = BindingKey {
            input: input.0.id,
            output: leases[0].0.id,
            uniform: leases[1].0.id,
            weights: None,
            // Task-local pipeline identity, separate from legacy Pass indices.
            pipeline: 57_000
                + usize::from(input.texture().format() == wgpu::TextureFormat::Rgba16Float),
            source_width: input.texture().width(),
            source_height: input.texture().height(),
        };
        let kernels = self
            .nodes
            .seamless_offset
            .0
            .get_or_insert_with(|| OffsetKernels::new(&self.device, self.queue.clone()));
        let mut pool = self.pool.lock().unwrap();
        if let Some(pass) = pool.backing.bindings.get(&key) {
            update(pass, transform, boundary);
        } else {
            let pass = kernels.prepare(
                &self.device,
                input.texture(),
                leases[0].texture(),
                leases[1].buffer(),
                transform,
                boundary,
            )?;
            pool.backing.bindings.insert(key, pass.0);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        pool.backing.bindings[&key].encode(&mut encoder);
        drop(pool);
        leases.push(input);
        if !self.current(work) {
            // Queued uniforms must still be flushed and retired at a real fence.
            let (_, done, _) = self.submit(
                self.device.create_command_encoder(&Default::default()),
                &leases,
            )?;
            self.wait(&done)?;
            return Err(Failure::Cancelled);
        }
        let (_, done, _) = self.submit(encoder, &leases)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(leases[0].clone())
    }
}

#[derive(Default)]
pub(super) struct State(Option<OffsetKernels>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(
        node.params,
        P::Seamless {
            method: texelith_core::SeamlessMethod::OffsetBlend { .. },
            ..
        }
    )
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(match &node.params {
        P::Seamless {
            method: texelith_core::SeamlessMethod::OffsetBlend { .. },
            ..
        } => crate::seamless_offset::scratch_bytes(extent),
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work, node, extent, ..
    } = *context;
    let input = context.input();
    Ok(match node.params {
        P::Seamless {
            method: texelith_core::SeamlessMethod::OffsetBlend { overlap },
            boundary,
        } => runtime.seamless_offset(work, input, extent, overlap, boundary)?,
        _ => unreachable!("registered parameters"),
    })
}
