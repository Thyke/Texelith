use super::*;
use crate::{
    MaterialError,
    builtin::{is_color_source, source_descriptor},
};
use texelith_core::{Boundary, MaterialNode, MaterialNodeParams as P};
#[derive(Default)]
pub(super) struct State {
    normal_proxy: Option<crate::poisson_solver::normal_proxy::Kernel>,
}
impl State {
    pub(super) fn evict(&mut self) -> bool {
        false
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(
        node.params,
        P::LegacySource(_)
            | P::ColorSource(_)
            | P::HeightSource(_)
            | P::NormalSource { .. }
            | P::Levels {
                boundary: Boundary::Clamp,
                ..
            }
            | P::Blur {
                boundary: Boundary::Clamp,
                ..
            }
    )
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    match node.params {
        P::Blur {
            sigma_source_px, ..
        } if sigma_source_px != 0.0 => crate::material::map_bytes(MapKind::Height, extent)
            .checked_add(384)
            .ok_or(MaterialError::Overflow),
        _ => Ok(128),
    }
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let NodeContext {
        work,
        node,
        extent,
        source_extent,
        ..
    } = *context;
    debug_assert_eq!(context.output.port, node.primary_output());
    Ok(if let Some(source) = source_descriptor(node) {
        let (_, upload, convention) = runtime
            .sources
            .iter()
            .find(|(s, _, _)| s == source)
            .ok_or_else(|| GpuError("requested source hash is not installed".into()))?;
        let upload = upload.clone();
        let conversion = match &node.params {
            P::LegacySource(_) => Some(Pass::Source(source.interpretation)),
            P::HeightSource(_) if upload.texture().format() == wgpu::TextureFormat::R32Float => {
                None
            }
            P::HeightSource(_) => Some(Pass::Source(
                texelith_core::InputInterpretation::LinearHeight,
            )),
            P::ColorSource(_) => Some(Pass::ColorSource),
            P::NormalSource {
                convention: expected,
                ..
            } if Some(*expected) == *convention => Some(Pass::NormalSource(*expected)),
            _ => {
                return Err(GpuError("typed source layout or convention mismatch".into()).into());
            }
        };
        if extent != source_extent && matches!(node.params, P::NormalSource { .. }) {
            return normal_proxy(runtime, context, upload);
        }
        let full = if let Some(operation) = conversion {
            runtime.material_pass(work, operation, upload, source_extent, source_extent)?
        } else {
            upload
        };
        // Height performs luminance before area restriction, preserving
        // scalar precision instead of requantizing a color proxy.
        if extent == source_extent || is_color_source(node) {
            full
        } else {
            runtime.material_pass(work, Pass::Downsample, full, extent, source_extent)?
        }
    } else {
        let input = context.input();
        match node.params {
            P::Levels {
                black,
                white,
                gamma,
                ..
            } => runtime.material_pass(
                work,
                Pass::Levels {
                    black,
                    white,
                    gamma,
                },
                input,
                extent,
                source_extent,
            )?,
            P::Blur {
                sigma_source_px: 0.0,
                ..
            } => input,
            P::Blur {
                sigma_source_px, ..
            } => {
                let scratch = runtime.material_pass(
                    work,
                    Pass::Blur {
                        sigma_source_px,
                        axis: Axis::X,
                    },
                    input,
                    extent,
                    source_extent,
                )?;
                runtime.material_pass(
                    work,
                    Pass::Blur {
                        sigma_source_px,
                        axis: Axis::Y,
                    },
                    scratch,
                    extent,
                    source_extent,
                )?
            }
            _ => unreachable!("registered parameters"),
        }
    })
}

// T-050: fused source decoding/restriction; all backing and bindings stay in the
// shared pool so eviction cannot leave uncharged textures held by a leaf cache.
fn normal_proxy(runtime: &mut Runtime, context: &NodeContext<'_>, input: Lease) -> Result<Lease> {
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let mut leases = runtime.allocate(&[
        texture_desc(context.extent, wgpu::TextureFormat::Rgba16Float),
        AllocationDesc::Buffer { bytes: 64 },
        AllocationDesc::Staging { bytes: 64 },
    ])?;
    let kernel =
        runtime.nodes.builtin.normal_proxy.get_or_insert_with(|| {
            crate::poisson_solver::normal_proxy::Kernel::new(&runtime.device)
        });
    let key = BindingKey {
        input: input.0.id,
        output: leases[0].0.id,
        uniform: leases[1].0.id,
        weights: None,
        // Same source operation; full conversion has equal input/output extents,
        // whereas this specialization always has a smaller output allocation.
        pipeline: Pass::NormalSource(texelith_core::NormalConvention::OpenGl).index(),
        source_width: context.source_extent.width,
        source_height: context.source_extent.height,
    };
    let mut pool = runtime.pool.lock().unwrap();
    let prepared = pool.backing.bindings.entry(key).or_insert_with(|| {
        kernel.prepare(
            &runtime.device,
            runtime.queue.clone(),
            input.texture(),
            leases[0].texture(),
            leases[1].buffer(),
        )
    });
    // Pooled uniforms may have been used by a different binding since last time.
    runtime.queue.write_buffer(
        leases[1].buffer(),
        0,
        bytemuck::bytes_of(&crate::NodeUniform::new(
            context.extent,
            context.source_extent,
        )),
    );
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    prepared.encode(&mut encoder);
    drop(pool);
    leases.push(input);
    let (_, done, _) = runtime.submit(encoder, &leases)?;
    runtime.shared.lock().unwrap().compute_submissions += 1;
    runtime.wait(&done)?;
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    Ok(leases[0].clone())
}
