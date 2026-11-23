use texelith_core::*;
use texelith_gpu::TypedSourcePixels;

/// Original deterministic nonconstant photo, matching the Faz 0 fixture formula.
/// Salt changes content identity before warmup, rather than falsifying its hash.
pub fn fixture_pixels(size: u32, salt: u8) -> TypedSourcePixels {
    let mut pixels = Vec::with_capacity(size as usize * size as usize * 4);
    for y in 0..size {
        for x in 0..size {
            pixels.extend_from_slice(&[
                ((x.wrapping_mul(17) ^ y.wrapping_mul(3)) as u8).wrapping_add(salt),
                (y.wrapping_mul(13) ^ (x >> 3)) as u8,
                (x.wrapping_add(y).wrapping_mul(7)) as u8,
                255,
            ]);
        }
    }
    TypedSourcePixels::Rgba8 {
        pixels: pixels.into(),
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

pub fn fixture_graph(source: SourceDescriptor, maximum: bool) -> Result<Graph, String> {
    let mut dto = MaterialGraphDto::from(material_from_source(source).map_err(|e| e.to_string())?);
    for node in &mut dto.graph {
        match &mut node.params {
            MaterialNodeParams::NormalFromHeight { filter, .. } => *filter = NormalFilter::Scharr,
            MaterialNodeParams::HorizonAo {
                directions, steps, ..
            } if maximum => {
                *directions = 32;
                *steps = 32;
            }
            _ => {}
        }
    }
    Graph::try_from(dto).map_err(|e| e.to_string())
}

pub fn scenario_plan(graph: &Graph, profile: &str) -> Result<ExportPlan, String> {
    if profile != "canonical" && profile != "exr" {
        return crate::export_jobs::build_plan(
            graph,
            &crate::export_jobs::ExportOptions {
                preset: profile.into(),
                ..Default::default()
            },
        );
    }
    // Start from the T-095 verified Blender contract; add the ADR 0006 edge map.
    let mut plan = BuiltinExportPresets
        .find("blender-v1", 1)
        .unwrap()
        .bind(graph, &[])
        .map_err(|e| e.to_string())?;
    plan.outputs
        .iter_mut()
        .find(|o| o.name == "occlusion")
        .unwrap()
        .name = "ao".into();
    let edge = graph
        .material_outputs()
        .and_then(|m| m.get(&MapKind::Edge))
        .ok_or("Missing edge")?;
    plan.outputs.push(ExportOutput {
        name: "edge".into(),
        format: ExportFormat::Png8,
        channels: vec![ChannelSpec {
            source: ChannelSource::Map(*edge, Component::R),
            invert: false,
            transfer: Transfer::Linear,
        }],
        normal_has_negative_z: false,
        quantization: None,
        import_hints: None,
    });
    if profile == "exr" {
        let height = plan
            .outputs
            .iter_mut()
            .find(|o| o.name == "height")
            .ok_or("Missing height")?;
        height.format = ExportFormat::Exr32;
        height.quantization = None;
        height.import_hints = None;
    }
    plan.profile.id = format!("t096-{profile}-v1");
    validate_export_for_graph(&plan, graph).map_err(|e| e.to_string())?;
    Ok(plan)
}
