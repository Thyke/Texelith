use super::*;
use crate::MaterialError;
use crate::{NormalParameters, PreparedNormal};
use texelith_core::{Boundary, CacheKey};
use texelith_core::{MaterialNode, MaterialNodeParams as P};

pub(super) struct NormalState {
    key: CacheKey,
    boundary: Boundary,
    prepared: PreparedNormal,
    // Retain ledger ownership for every texture kept alive by the bind groups.
    input: Lease,
    output: Lease,
}
impl Runtime {
    pub(super) fn normal_from_height(
        &mut self,
        work: &Work,
        input: Lease,
        key: CacheKey,
        extent: Extent,
        source: Extent,
        parameters: &NormalParameters,
    ) -> Result<Lease> {
        // A replacement may fit only after allocate evicts the completed state.
        let output = self
            .allocate(&[texture_desc(extent, wgpu::TextureFormat::Rgba16Float)])?
            .remove(0);
        let cached = self.nodes.normal_filters.0.take().filter(|s| {
            s.key == key
                && s.boundary == parameters.boundary
                && s.prepared.is_ready()
                && (parameters.scale_weights.len() == 1 || s.prepared.has_pyramid())
        });
        let hit = cached.is_some();
        let mut state = if let Some(mut state) = cached {
            state
                .prepared
                .reweight(&self.executor, output.texture(), parameters)?;
            state.output = output.clone();
            state
        } else {
            NormalState {
                key,
                boundary: parameters.boundary,
                prepared: PreparedNormal::new(
                    &self.executor,
                    input.texture(),
                    output.texture(),
                    &SourceDescriptor {
                        content_hash: key.0,
                        extent: source,
                        interpretation: texelith_core::InputInterpretation::LinearHeight,
                    },
                    parameters,
                    self.pool.clone(),
                )?,
                input,
                output: output.clone(),
            }
        };
        let passes = if hit {
            Vec::new()
        } else {
            state.prepared.pyramid_passes().to_vec()
        };
        let leases = [state.input.clone(), output.clone()];
        for index in 0..=passes.len() {
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
            let mut encoder = self.device.create_command_encoder(&Default::default());
            if index < passes.len() {
                passes[index].encode(&mut encoder);
            } else {
                state.prepared.encode(&mut encoder);
            }
            let (fence, done, _) = self.submit(encoder, &leases)?;
            state.prepared.mark_submitted(fence)?;
            self.shared.lock().unwrap().compute_submissions += 1;
            self.wait(&done)?;
            state.prepared.complete(fence, index == passes.len());
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
        }
        self.nodes.normal_filters.0 = Some(state);
        Ok(output)
    }
}

#[derive(Default)]
pub(super) struct State(Option<NormalState>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        self.0.take().is_some()
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, P::NormalFromHeight { .. })
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    Ok(match &node.params {
        P::NormalFromHeight { scale_weights, .. } => {
            crate::normal_filters::scratch_bytes(extent, scale_weights.len())
                .map_err(|e| MaterialError::InvalidRequest(e.to_string()))?
        }
        _ => unreachable!("registered parameters"),
    })
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work,
        node,
        extent,
        source_extent,
        ..
    } = *context;
    let input = context.input();
    Ok(match node.params {
        P::NormalFromHeight {
            strength,
            filter,
            ref scale_weights,
            boundary,
            convention,
        } if (crate::NormalParameters {
            strength,
            filter,
            scale_weights: scale_weights.clone(),
            boundary,
            convention,
        })
        .is_legacy() =>
        {
            runtime.material_pass(
                work,
                Pass::Normal { strength },
                input,
                extent,
                source_extent,
            )?
        }
        P::NormalFromHeight {
            strength,
            filter,
            ref scale_weights,
            boundary,
            convention,
        } => runtime.normal_from_height(
            work,
            input,
            context.input_key(),
            extent,
            source_extent,
            &crate::NormalParameters {
                strength,
                filter,
                scale_weights: scale_weights.clone(),
                boundary,
                convention,
            },
        )?,
        _ => unreachable!("registered parameters"),
    })
}
