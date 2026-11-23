use super::*;
use crate::{
    MaterialError,
    seamless_histogram::{HistogramKernels, HistogramResources, cache_event, scratch_bytes},
};
use texelith_core::{CacheKey, MaterialNode, MaterialNodeParams as P, SeamlessMethod};

struct Mapping {
    key: CacheKey,
    resources: HistogramResources,
    reservation: Lease,
}
#[derive(Default)]
pub(super) struct State {
    kernels: Option<HistogramKernels>,
    mapping: Option<Mapping>,
}
impl State {
    pub(super) fn evict(&mut self) -> bool {
        self.mapping.take().is_some()
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(
        node.params,
        P::Seamless {
            method: SeamlessMethod::Histogram { .. },
            ..
        }
    )
}
pub(super) fn scratch(_: &MaterialNode, extent: Extent) -> std::result::Result<u64, MaterialError> {
    // The registry supplies evaluation extent. A proxy can cover up to 4x each
    // axis of original content; reserve its full histogram, never proxy pixels.
    let bound = Extent::new((extent.width * 4).min(4096), (extent.height * 4).min(4096))?;
    Ok(scratch_bytes(bound))
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let P::Seamless {
        method: SeamlessMethod::Histogram { base, strength },
        boundary,
    } = &context.node.params
    else {
        unreachable!("histogram registration")
    };
    if !runtime.current(context.work) {
        return Err(Failure::Cancelled);
    }
    let mut input = context.input();
    let graph = &context
        .work
        .material
        .as_ref()
        .expect("material work")
        .snapshot
        .graph;
    if let Some(source_node) = graph.material_node(context.node.inputs[0].source.node)
        && let P::HeightSource(source) = &source_node.params
        && let Some((_, upload, _)) = runtime.sources.iter().find(|(s, _, _)| s == source)
        && context.extent != source.extent
    {
        input = upload.clone();
    }
    let source = Extent::new(input.texture().width(), input.texture().height())?;
    let format = if input.texture().format() == wgpu::TextureFormat::Rgba8Unorm {
        wgpu::TextureFormat::R32Float
    } else {
        input.texture().format()
    };
    // Keep the old mapping reachable by eviction until replacement admission.
    let output = runtime
        .allocate(&[texture_desc(context.extent, format)])?
        .remove(0);
    let key = context.input_key();
    let hit = runtime
        .nodes
        .seamless_histogram
        .mapping
        .as_ref()
        .is_some_and(|m| m.key == key);
    let mapping = if hit {
        runtime.nodes.seamless_histogram.mapping.take().unwrap()
    } else {
        let reservation = runtime
            .allocate(&[AllocationDesc::Staging {
                bytes: scratch_bytes(source),
            }])?
            .remove(0);
        runtime.nodes.seamless_histogram.mapping = None;
        Mapping {
            key,
            resources: HistogramResources::new(&runtime.device, &runtime.queue, source),
            reservation,
        }
    };
    cache_event(hit);
    let kernels = runtime
        .nodes
        .seamless_histogram
        .kernels
        .get_or_insert_with(|| HistogramKernels::new(&runtime.device));
    let prepared = kernels.prepare(
        &runtime.device,
        &runtime.queue,
        input.texture(),
        output.texture(),
        &mapping.resources,
        base,
        *boundary,
        *strength,
    )?;
    let retained = [input, output.clone(), mapping.reservation.clone()];
    // Flush queued uploads even if this generation was replaced during prepare.
    if !runtime.current(context.work) {
        let encoder = runtime.device.create_command_encoder(&Default::default());
        let (_, done, _) = runtime.submit(encoder, &retained)?;
        runtime.wait(&done)?;
        return Err(Failure::Cancelled);
    }
    for pass in if hit { 3..4 } else { 0..4 } {
        if !runtime.current(context.work) {
            return Err(Failure::Cancelled);
        }
        let mut encoder = runtime.device.create_command_encoder(&Default::default());
        prepared.encode_pass(&mut encoder, pass);
        let (_, done, _) = runtime.submit(encoder, &retained)?;
        runtime.shared.lock().unwrap().compute_submissions += 1;
        runtime.wait(&done)?;
        if !runtime.current(context.work) {
            return Err(Failure::Cancelled);
        }
    }
    runtime.nodes.seamless_histogram.mapping = Some(mapping);
    Ok(output)
}
