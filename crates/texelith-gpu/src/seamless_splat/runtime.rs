use super::*;
use crate::MaterialError;
use crate::seamless_splat::{SplatKernels, SplatTransform, update};
use texelith_core::Boundary;
use texelith_core::{MaterialNode, MaterialNodeParams as P};

impl Runtime {
    pub(super) fn seamless_splat(
        &mut self,
        work: &Work,
        input: Lease,
        extent: Extent,
        transform: SplatTransform,
        boundary: Boundary,
    ) -> Result<Lease> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let format = if input.texture().format() == wgpu::TextureFormat::Rgba8Unorm {
            wgpu::TextureFormat::R32Float
        } else {
            input.texture().format()
        };
        let mut leases = self.allocate(&[
            texture_desc(extent, format),
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Staging { bytes: 64 },
        ])?;
        let key = BindingKey {
            input: input.0.id,
            output: leases[0].0.id,
            uniform: leases[1].0.id,
            weights: None,
            // Task-local pipeline identity, separate from legacy Pass indices.
            pipeline: 58_000
                + match input.texture().format() {
                    wgpu::TextureFormat::Rgba16Float => 1,
                    wgpu::TextureFormat::Rgba8Unorm => 2,
                    _ => 0,
                },
            source_width: input.texture().width(),
            source_height: input.texture().height(),
        };
        let kernels = self
            .nodes
            .seamless_splat
            .0
            .get_or_insert_with(|| SplatKernels::new(&self.device, self.queue.clone()));
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
pub(super) struct State(Option<SplatKernels>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(
        node.params,
        P::Seamless {
            method: texelith_core::SeamlessMethod::Splat { .. },
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
            method: texelith_core::SeamlessMethod::Splat { .. },
            ..
        } => crate::seamless_splat::scratch_bytes(extent),
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work, node, extent, ..
    } = *context;
    let mut input = context.input();
    // The generic scalar source adapter area-restricts proxies. Splats sample
    // the original continuous field instead; the resident typed upload already
    // belongs to the same ledger, so this adds no image allocation.
    let graph = &work
        .material
        .as_ref()
        .expect("material work")
        .snapshot
        .graph;
    if let Some(source_node) = graph.material_node(node.inputs[0].source.node)
        && let P::HeightSource(source) = &source_node.params
        && let Some((_, upload, _)) = runtime.sources.iter().find(|(s, _, _)| s == source)
        && extent != source.extent
    {
        input = upload.clone();
    }
    Ok(match node.params {
        P::Seamless {
            method:
                texelith_core::SeamlessMethod::Splat {
                    seed,
                    grid,
                    overlap,
                },
            boundary,
        } => runtime.seamless_splat(
            work,
            input,
            extent,
            SplatTransform::new(seed, grid, overlap)?,
            boundary,
        )?,
        _ => unreachable!("registered parameters"),
    })
}
