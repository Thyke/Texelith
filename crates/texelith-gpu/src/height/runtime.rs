use super::*;
use crate::MaterialError;
use crate::{HeightParameters, PreparedHeight};
use texelith_core::{Boundary, CacheKey};
use texelith_core::{MaterialNode, MaterialNodeParams as P};

pub(super) struct HeightState {
    key: CacheKey,
    boundary: Boundary,
    prepared: PreparedHeight,
    // Bind groups keep handles, so retain the ledger owners for their lifetime.
    input: Lease,
    output: Lease,
}
impl Runtime {
    pub(super) fn height_from_albedo(
        &mut self,
        work: &Work,
        input: Lease,
        key: CacheKey,
        extent: Extent,
        params: &HeightParameters,
    ) -> Result<Lease> {
        // Leave the completed chain evictable while admitting the new output.
        // A weight edit may fit only after releasing the previous chain/output.
        let output = self
            .allocate(&[texture_desc(extent, wgpu::TextureFormat::R32Float)])?
            .remove(0);
        let mut cached = self.nodes.height.0.take().filter(|state| {
            state.key == key && state.boundary == params.boundary && state.prepared.is_ready()
        });
        let hit = cached.is_some();
        let mut state = if let Some(mut state) = cached.take() {
            state
                .prepared
                .reweight(&self.executor, output.texture(), params)?;
            state.output = output.clone();
            self.shared.lock().unwrap().pyramid_cache_hits += 1;
            state
        } else {
            let source = SourceDescriptor {
                content_hash: key.0,
                extent: Extent::new(input.texture().width(), input.texture().height())?,
                interpretation: texelith_core::InputInterpretation::LinearHeight,
            };
            let prepared = PreparedHeight::new(
                &self.executor,
                &source,
                input.texture(),
                output.texture(),
                params,
                self.pool.clone(),
            )?;
            HeightState {
                key,
                boundary: params.boundary,
                prepared,
                input,
                output: output.clone(),
            }
        };
        let passes = if hit {
            state.prepared.reconstruction_passes()
        } else {
            state.prepared.passes()
        }
        .to_vec();
        let leases = [state.input.clone(), output.clone()];
        for (index, pass) in passes.iter().enumerate() {
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
            let mut encoder = self.device.create_command_encoder(&Default::default());
            pass.encode(&mut encoder);
            let (fence, done, _) = self.submit(encoder, &leases)?;
            state.prepared.mark_submitted(fence)?;
            self.shared.lock().unwrap().compute_submissions += 1;
            self.wait(&done)?;
            state.prepared.complete(fence, index + 1 == passes.len());
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
        }
        self.nodes.height.0 = Some(state);
        Ok(output)
    }
}

#[derive(Default)]
pub(super) struct State(Option<HeightState>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        self.0.take().is_some()
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(
        node.params,
        P::HeightFromAlbedo { .. } | P::NormalizeHeight {}
    )
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(match &node.params {
        P::NormalizeHeight {} => crate::PreparedMinMaxRemap::scratch_bytes(extent),
        P::HeightFromAlbedo { .. } => crate::height::scratch_bytes(extent)
            .map_err(|e| MaterialError::InvalidRequest(e.to_string()))?,
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work, node, extent, ..
    } = *context;
    let input = context.input();
    Ok(match node.params {
        P::NormalizeHeight {} => {
            let output = runtime
                .allocate(&[texture_desc(extent, wgpu::TextureFormat::R32Float)])?
                .remove(0);
            let mut remap = crate::PreparedMinMaxRemap::new(
                &runtime.executor,
                input.texture(),
                output.texture(),
                0.5,
                runtime.pool.clone(),
            )?;
            let leases = [input, output.clone()];
            for index in 0..remap.passes().len() {
                if !runtime.current(work) {
                    return Err(Failure::Cancelled);
                }
                let mut encoder = runtime.device.create_command_encoder(&Default::default());
                remap.passes()[index].encode(&mut encoder);
                let (fence, done, _) = runtime.submit(encoder, &leases)?;
                remap.mark_submitted(fence)?;
                runtime.shared.lock().unwrap().compute_submissions += 1;
                runtime.wait(&done)?;
                remap.complete(fence);
            }
            output
        }
        P::HeightFromAlbedo {
            ref band_weights,
            coarse_weight,
            remove_coarse,
            boundary,
        } => runtime.height_from_albedo(
            work,
            input,
            context.input_key(),
            extent,
            &crate::HeightParameters {
                band_weights: band_weights.clone(),
                coarse_weight,
                remove_coarse,
                boundary,
            },
        )?,
        _ => unreachable!("registered parameters"),
    })
}
