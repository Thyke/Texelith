use texelith_core::{
    BuiltinExportPresets, Component, ExportFormat, ExportPresetTable, MapKind, NormalConvention,
    PresetChannelSource, Transfer,
};

fn output<'a>(
    preset: &'a texelith_core::ExportPreset,
    name: &str,
) -> &'a texelith_core::PresetOutput {
    preset.outputs.iter().find(|o| o.name == name).unwrap()
}
fn map(channel: texelith_core::PresetChannelSpec) -> (MapKind, Component) {
    match channel.source {
        PresetChannelSource::Map { map, component, .. } => (map, component),
        other => panic!("expected mapped channel, got {other:?}"),
    }
}

#[test]
fn all_six_engine_profiles_have_exact_mapping_and_transfer_contracts() {
    use Component::{B, G, R};
    use MapKind::{Ao, Metallic, Normal, Roughness};
    let presets = BuiltinExportPresets.presets();
    assert_eq!(presets.len(), 6);
    for p in presets {
        assert_eq!(
            output(p, "albedo")
                .channels
                .iter()
                .map(|c| c.transfer)
                .collect::<Vec<_>>(),
            [Transfer::Srgb; 3],
            "{}",
            p.profile.id
        );
        assert_eq!(
            output(p, "normal")
                .channels
                .iter()
                .map(|c| c.transfer)
                .collect::<Vec<_>>(),
            [Transfer::Linear; 3]
        );
        assert_eq!(
            output(p, "normal")
                .channels
                .iter()
                .map(|c| map(*c))
                .collect::<Vec<_>>(),
            [(Normal, R), (Normal, G), (Normal, B)]
        );
        assert!(p.profile.engine_safe_srgb);
    }
    let get = |id| presets.iter().find(|p| p.profile.id == id).unwrap();
    let unreal = get("unreal-v1");
    assert_eq!(unreal.profile.normal_convention, NormalConvention::DirectX);
    assert_eq!(
        output(unreal, "orm")
            .channels
            .iter()
            .map(|c| map(*c))
            .collect::<Vec<_>>(),
        [(Ao, R), (Roughness, R), (Metallic, R)]
    );
    assert_eq!(output(unreal, "height").format, ExportFormat::Png16);
    assert_eq!(output(unreal, "normal_bc5").format, ExportFormat::DdsBc5);
    let urp = get("unity-urp-v1");
    let ms = output(urp, "metallic_smoothness");
    assert_eq!(
        ms.channels.iter().map(|c| c.transfer).collect::<Vec<_>>(),
        [Transfer::Linear; 4]
    );
    assert_eq!(map(ms.channels[0]), (Metallic, R));
    assert!(ms.channels[3].invert);
    assert_eq!(
        output(urp, "occlusion")
            .import_hints
            .as_ref()
            .unwrap()
            .assignments[0]
            .component,
        G
    );
    let hdrp = get("unity-hdrp-v1");
    let mask = output(hdrp, "mask_map");
    assert_eq!(
        mask.channels.iter().map(|c| c.transfer).collect::<Vec<_>>(),
        [Transfer::Linear; 4]
    );
    assert_eq!(map(mask.channels[0]), (Metallic, R));
    assert_eq!(map(mask.channels[1]), (Ao, R));
    assert_eq!(mask.channels[2].source, PresetChannelSource::Constant(1.0));
    assert!(mask.channels[3].invert);
    for id in ["godot-v1", "blender-v1", "gltf2-v1"] {
        assert_eq!(get(id).profile.normal_convention, NormalConvention::OpenGl);
    }
    let gltf = output(get("gltf2-v1"), "orm");
    assert_eq!(
        gltf.channels.iter().map(|c| map(*c)).collect::<Vec<_>>(),
        [(Ao, R), (Roughness, R), (Metallic, R)]
    );
}
