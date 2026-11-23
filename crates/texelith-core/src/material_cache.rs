//! Explicit cache schema 2. Field order and tags are fixed by the canonical
//! encoder and golden-byte tests; schema 1 remains in `cache`.

use crate::cache::{extent_bytes, float_bytes, u32_bytes};
use crate::{
    Boundary, CacheKey, CacheRevisions, CoreError, Extent, InputInterpretation, InputPort,
    MapFormat, MaskMode, MaterialNode, MaterialNodeParams, NormalConvention, NormalFilter,
    OutputPort, SeamlessBase, SeamlessMethod, SourceDescriptor,
};

pub fn material_node_cache_key(
    node: &MaterialNode,
    output: OutputPort,
    inputs: &[CacheKey],
    extent: Extent,
    source_extent: Extent,
    revisions: CacheRevisions,
) -> Result<CacheKey, CoreError> {
    let bytes =
        canonical_material_node_bytes(node, output, inputs, extent, source_extent, revisions)?;
    Ok(CacheKey(*blake3::hash(&bytes).as_bytes()))
}

/// Encodes finite canonical parameters, ordered port/key pairs, source and
/// evaluation extents, and explicit revisions. Node IDs and view state are absent.
/// Callers supply parent content keys in the same order as `node.inputs`.
pub fn canonical_material_node_bytes(
    node: &MaterialNode,
    output: OutputPort,
    inputs: &[CacheKey],
    extent: Extent,
    source_extent: Extent,
    revisions: CacheRevisions,
) -> Result<Vec<u8>, CoreError> {
    node.validate()?;
    extent.validate()?;
    source_extent.validate()?;
    if !node.has_output(output) {
        return Err(CoreError::InvalidMaterial("invalid cache output port"));
    }
    if inputs.len() != node.inputs.len() {
        return Err(CoreError::InvalidInputKeyCount {
            expected: node.inputs.len(),
            actual: inputs.len(),
        });
    }
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(b"texelith.node-cache\0");
    u32_bytes(&mut bytes, 2);
    u32_bytes(&mut bytes, revisions.algorithm);
    u32_bytes(&mut bytes, revisions.color_policy);
    bytes.push(node_tag(&node.params));
    bytes.push(output_tag(output));
    bytes.push(match output.map_kind().format() {
        MapFormat::R32Float => 0,
        MapFormat::Rgba16Float => 1,
    });
    parameters(&mut bytes, &node.params);
    u32_bytes(&mut bytes, inputs.len() as u32);
    for (edge, key) in node.inputs.iter().zip(inputs) {
        bytes.push(match edge.port {
            InputPort::Color => 0,
            InputPort::Height => 1,
            InputPort::Normal => 2,
            InputPort::Mask => 3,
        });
        bytes.push(output_tag(edge.source.port));
        bytes.extend_from_slice(&key.0);
    }
    extent_bytes(&mut bytes, extent);
    extent_bytes(&mut bytes, source_extent);
    Ok(bytes)
}

fn node_tag(params: &MaterialNodeParams) -> u8 {
    match params {
        MaterialNodeParams::LegacySource(_) => 0,
        MaterialNodeParams::ColorSource(_) => 1,
        MaterialNodeParams::HeightSource(_) => 2,
        MaterialNodeParams::NormalSource { .. } => 3,
        MaterialNodeParams::HeightFromAlbedo { .. } => 4,
        MaterialNodeParams::NormalFromHeight { .. } => 5,
        MaterialNodeParams::HeightFromNormal { .. } => 6,
        MaterialNodeParams::HorizonAo { .. } => 7,
        MaterialNodeParams::Curvature { .. } => 8,
        MaterialNodeParams::Roughness { .. } => 9,
        MaterialNodeParams::Metallic { .. } => 10,
        MaterialNodeParams::PaintMask { .. } => 11,
        MaterialNodeParams::Levels { .. } => 12,
        MaterialNodeParams::Blur { .. } => 13,
        MaterialNodeParams::Seamless { .. } => 14,
        MaterialNodeParams::NormalizeHeight {} => 15,
    }
}

fn output_tag(output: OutputPort) -> u8 {
    match output {
        OutputPort::Albedo => 0,
        OutputPort::Height => 1,
        OutputPort::Normal => 2,
        OutputPort::Ao => 3,
        OutputPort::Curvature => 4,
        OutputPort::Edge => 5,
        OutputPort::Cavity => 6,
        OutputPort::Roughness => 7,
        OutputPort::Metallic => 8,
        OutputPort::Mask => 9,
    }
}

fn boundary(bytes: &mut Vec<u8>, value: Boundary) {
    bytes.push(match value {
        Boundary::Clamp => 0,
        Boundary::Repeat => 1,
    });
}

fn convention(bytes: &mut Vec<u8>, value: NormalConvention) {
    bytes.push(match value {
        NormalConvention::OpenGl => 0,
        NormalConvention::DirectX => 1,
    });
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        float_bytes(bytes, *value);
    }
}

fn float_vector(bytes: &mut Vec<u8>, values: &[f32]) {
    u32_bytes(bytes, values.len() as u32);
    floats(bytes, values);
}

fn source(bytes: &mut Vec<u8>, value: &SourceDescriptor) {
    bytes.extend_from_slice(&value.content_hash);
    extent_bytes(bytes, value.extent);
    bytes.push(match value.interpretation {
        InputInterpretation::SrgbPhoto => 0,
        InputInterpretation::LinearHeight => 1,
        InputInterpretation::LinearNormal => 2,
    });
}

fn base(bytes: &mut Vec<u8>, value: &SeamlessBase) {
    match value {
        SeamlessBase::OffsetBlend { overlap } => {
            bytes.push(0);
            float_bytes(bytes, *overlap);
        }
        SeamlessBase::Splat {
            seed,
            grid,
            overlap,
        } => {
            bytes.push(1);
            u32_bytes(bytes, *seed);
            u32_bytes(bytes, *grid);
            float_bytes(bytes, *overlap);
        }
    }
}

fn seamless(bytes: &mut Vec<u8>, value: &SeamlessMethod) {
    match value {
        SeamlessMethod::OffsetBlend { overlap } => {
            base(bytes, &SeamlessBase::OffsetBlend { overlap: *overlap })
        }
        SeamlessMethod::Splat {
            seed,
            grid,
            overlap,
        } => base(
            bytes,
            &SeamlessBase::Splat {
                seed: *seed,
                grid: *grid,
                overlap: *overlap,
            },
        ),
        SeamlessMethod::Histogram {
            base: inner,
            strength,
        } => {
            bytes.push(2);
            base(bytes, inner);
            float_bytes(bytes, *strength);
        }
    }
}

fn parameters(bytes: &mut Vec<u8>, params: &MaterialNodeParams) {
    match params {
        MaterialNodeParams::NormalizeHeight {} => {}
        MaterialNodeParams::LegacySource(value)
        | MaterialNodeParams::ColorSource(value)
        | MaterialNodeParams::HeightSource(value) => source(bytes, value),
        MaterialNodeParams::NormalSource {
            source: value,
            convention: normal_convention,
        } => {
            source(bytes, value);
            convention(bytes, *normal_convention);
        }
        MaterialNodeParams::HeightFromAlbedo {
            band_weights,
            coarse_weight,
            remove_coarse,
            boundary: mode,
        } => {
            float_vector(bytes, band_weights);
            float_bytes(bytes, *coarse_weight);
            bytes.push(u8::from(*remove_coarse));
            boundary(bytes, *mode);
        }
        MaterialNodeParams::NormalFromHeight {
            strength,
            filter,
            scale_weights,
            boundary: mode,
            convention: normal_convention,
        } => {
            float_bytes(bytes, *strength);
            bytes.push(match filter {
                NormalFilter::Central => 0,
                NormalFilter::Sobel => 1,
                NormalFilter::Scharr => 2,
            });
            float_vector(bytes, scale_weights);
            boundary(bytes, *mode);
            convention(bytes, *normal_convention);
        }
        MaterialNodeParams::HeightFromNormal {
            boundary: mode,
            v_cycles,
            pre_smooth,
            post_smooth,
            coarsest_iterations,
            relative_tolerance,
            absolute_tolerance,
        } => {
            boundary(bytes, *mode);
            for count in [*v_cycles, *pre_smooth, *post_smooth, *coarsest_iterations] {
                u32_bytes(bytes, count);
            }
            floats(bytes, &[*relative_tolerance, *absolute_tolerance]);
        }
        MaterialNodeParams::HorizonAo {
            directions,
            steps,
            radius_source_px,
            bias,
            boundary: mode,
        } => {
            u32_bytes(bytes, *directions);
            u32_bytes(bytes, *steps);
            floats(bytes, &[*radius_source_px, *bias]);
            boundary(bytes, *mode);
        }
        MaterialNodeParams::Curvature {
            edge_scale,
            cavity_scale,
            boundary: mode,
        } => {
            floats(bytes, &[*edge_scale, *cavity_scale]);
            boundary(bytes, *mode);
        }
        MaterialNodeParams::Roughness {
            curve,
            base,
            invert,
        } => {
            u32_bytes(bytes, curve.len() as u32);
            for knot in curve {
                floats(bytes, knot);
            }
            float_bytes(bytes, *base);
            bytes.push(u8::from(*invert));
        }
        MaterialNodeParams::Metallic {
            picked_color,
            radius,
            threshold,
            feather,
            mask_mode,
        } => {
            floats(bytes, picked_color);
            floats(bytes, &[*radius, *threshold, *feather]);
            bytes.push(match mask_mode {
                MaskMode::Replace => 0,
                MaskMode::Add => 1,
                MaskMode::Subtract => 2,
            });
        }
        MaterialNodeParams::PaintMask {
            boundary: mode,
            extent,
            stroke_digest,
            ..
        } => {
            extent_bytes(bytes, *extent);
            bytes.extend_from_slice(stroke_digest);
            // Repeat preserves the original frozen schema-2 parameter bytes.
            if *mode == Boundary::Clamp {
                bytes.push(0);
            }
        }
        MaterialNodeParams::Levels {
            black,
            white,
            gamma,
            boundary: mode,
        } => {
            floats(bytes, &[*black, *white, *gamma]);
            boundary(bytes, *mode);
        }
        MaterialNodeParams::Blur {
            sigma_source_px,
            boundary: mode,
        } => {
            float_bytes(bytes, *sigma_source_px);
            boundary(bytes, *mode);
        }
        MaterialNodeParams::Seamless {
            method,
            boundary: mode,
        } => {
            seamless(bytes, method);
            boundary(bytes, *mode);
        }
    }
}
