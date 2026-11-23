use texelith_core::*;

pub fn extent(w: u32, h: u32) -> Extent {
    Extent::new(w, h).unwrap()
}

pub fn reference(port: OutputPort) -> OutputRef {
    let id = match port {
        OutputPort::Albedo => 1,
        OutputPort::Height => 2,
        OutputPort::Normal => 3,
        OutputPort::Ao => 4,
        OutputPort::Roughness => 5,
        OutputPort::Metallic => 6,
        _ => 7,
    };
    OutputRef {
        node: NodeId(id),
        port,
    }
}

pub fn map(port: OutputPort, component: Component) -> ChannelSpec {
    ChannelSpec {
        source: ChannelSource::Map(reference(port), component),
        invert: false,
        transfer: Transfer::Linear,
    }
}

pub fn constant(value: f32) -> ChannelSpec {
    ChannelSpec {
        source: ChannelSource::Constant(value),
        invert: false,
        transfer: Transfer::Linear,
    }
}

pub fn output(name: &str, format: ExportFormat, channels: Vec<ChannelSpec>) -> ExportOutput {
    ExportOutput {
        name: name.into(),
        format,
        channels,
        normal_has_negative_z: false,
        quantization: None,
        import_hints: None,
    }
}

pub fn plan(source_extent: Extent, outputs: Vec<ExportOutput>) -> ExportPlan {
    ExportPlan {
        source_extent,
        size: SizePolicy::Keep,
        naming: "T_{name}_{map}".into(),
        profile: EngineProfile {
            id: "t078-explicit-v1".into(),
            revision: 1,
            engine_safe_srgb: false,
            normal_convention: NormalConvention::OpenGl,
        },
        outputs,
        displacement: DisplacementSettings::default(),
        padding: Boundary::Repeat,
    }
}

// Explicit assignments, deliberately independent of BuiltinExportPresets.
pub fn packed_outputs() -> Vec<ExportOutput> {
    use Component::R;
    use OutputPort::{Ao, Metallic, Roughness};
    let mut smoothness = map(Roughness, R);
    smoothness.invert = true;
    vec![
        output(
            "orm",
            ExportFormat::Png8,
            vec![map(Ao, R), map(Roughness, R), map(Metallic, R)],
        ),
        output(
            "mask_map",
            ExportFormat::Png8,
            vec![map(Metallic, R), map(Ao, R), constant(1.0), smoothness],
        ),
        output(
            "gltf_metallic_roughness",
            ExportFormat::Png8,
            vec![map(Ao, R), map(Roughness, R), map(Metallic, R)],
        ),
        output(
            "urp_metallic_smoothness",
            ExportFormat::Png8,
            vec![map(Metallic, R), constant(0.0), constant(0.0), smoothness],
        ),
    ]
}
