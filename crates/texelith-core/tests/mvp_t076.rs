use std::{fs, path::PathBuf};
use texelith_core::*;

fn fixture_path(id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("presets")
        .join(format!("{id}.json"))
}

#[test]
fn built_in_presets_match_versioned_json_golden_fixtures() {
    let table = BuiltinExportPresets;
    let presets = table.presets();
    assert_eq!(presets.len(), 6);
    let mut ids = presets
        .iter()
        .map(|preset| preset.profile.id.as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(
        ids,
        [
            "blender-v1",
            "gltf2-v1",
            "godot-v1",
            "unity-hdrp-v1",
            "unity-urp-v1",
            "unreal-v1"
        ]
    );
    for preset in presets {
        let path = fixture_path(&preset.profile.id);
        let fixture: ExportPreset =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            &fixture, preset,
            "fixture differs for {}",
            preset.profile.id
        );
    }
}

fn find<'a>(table: &'a [ExportPreset], id: &str) -> &'a ExportPreset {
    table
        .iter()
        .find(|preset| preset.profile.id == id && preset.profile.revision == 1)
        .unwrap()
}

fn named<'a>(preset: &'a ExportPreset, name: &str) -> &'a PresetOutput {
    preset
        .outputs
        .iter()
        .find(|output| output.name == name)
        .unwrap()
}

#[test]
fn presets_encode_distinct_engine_contracts() {
    let presets = BuiltinExportPresets.presets();
    let unreal = find(presets, "unreal-v1");
    assert_eq!(unreal.profile.normal_convention, NormalConvention::DirectX);
    assert_eq!(named(unreal, "orm").format, ExportFormat::Png8);
    assert_eq!(named(unreal, "height").format, ExportFormat::Png16);
    assert_eq!(named(unreal, "normal_bc5").format, ExportFormat::DdsBc5);
    let hdrp = find(presets, "unity-hdrp-v1");
    assert!(hdrp.outputs.iter().any(|output| output.name == "mask_map"));
    let urp = find(presets, "unity-urp-v1");
    assert!(!urp.outputs.iter().any(|output| output.name == "mask_map"));
    assert!(
        urp.outputs
            .iter()
            .any(|output| output.name == "metallic_smoothness")
    );
    let gltf = find(presets, "gltf2-v1");
    let orm = named(gltf, "orm");
    assert_eq!(
        orm.import_hints
            .as_ref()
            .unwrap()
            .assignments
            .iter()
            .map(|item| item.component)
            .collect::<Vec<_>>(),
        [Component::R, Component::G, Component::B]
    );
    assert!(
        orm.import_hints
            .as_ref()
            .unwrap()
            .instructions
            .iter()
            .any(|line| line.contains("occlusion uses R"))
    );
    assert_eq!(
        find(presets, "godot-v1").profile.normal_convention,
        NormalConvention::OpenGl
    );
    assert!(
        find(presets, "blender-v1")
            .outputs
            .iter()
            .any(|output| output.name == "roughness")
    );
}

#[test]
fn table_lookup_requires_exact_preset_revision() {
    let table = BuiltinExportPresets;
    assert!(table.find("unity-urp-v1", 1).is_some());
    assert!(table.find("unity-urp-v1", 2).is_none());
    assert!(table.find("unknown", 1).is_none());
}

#[test]
fn adr_channel_assignments_and_color_spaces() {
    use Component::{A, B, G, R};
    use MapKind::{Albedo, Ao, Height, Metallic, Normal, Roughness};
    use TextureUsage::{
        Albedo as ColorUse, DetailMask, Height as HeightUse, Metallic as MetalUse,
        Normal as NormalUse, Occlusion, Roughness as RoughUse, Smoothness,
    };

    // Each entry specifies source map/component (or a constant), inversion and usage.
    type Entry = (
        Option<(MapKind, Component)>,
        f32,
        bool,
        Option<TextureUsage>,
    );
    let map = |kind, component, invert, usage| (Some((kind, component)), 0.0, invert, Some(usage));
    let constant = |value, usage| (None, value, false, usage);
    let rgb = |kind, usage| {
        vec![
            map(kind, R, false, usage),
            map(kind, G, false, usage),
            map(kind, B, false, usage),
        ]
    };
    let orm = || {
        vec![
            map(Ao, R, false, Occlusion),
            map(Roughness, R, false, RoughUse),
            map(Metallic, R, false, MetalUse),
        ]
    };
    for preset in BuiltinExportPresets.presets() {
        let mut expected: Vec<(&str, Vec<Entry>)> = vec![
            ("albedo", rgb(Albedo, ColorUse)),
            ("normal", rgb(Normal, NormalUse)),
        ];
        match preset.profile.id.as_str() {
            "unreal-v1" => expected.extend([
                ("orm", orm()),
                ("height", vec![map(Height, R, false, HeightUse)]),
                (
                    "normal_bc5",
                    vec![
                        map(Normal, R, false, NormalUse),
                        (Some((Normal, G)), 0.0, false, None),
                    ],
                ),
            ]),
            "unity-urp-v1" => expected.extend([
                (
                    "metallic_smoothness",
                    vec![
                        map(Metallic, R, false, MetalUse),
                        constant(0.0, None),
                        constant(0.0, None),
                        map(Roughness, R, true, Smoothness),
                    ],
                ),
                (
                    "occlusion",
                    vec![
                        constant(1.0, None),
                        map(Ao, R, false, Occlusion),
                        constant(1.0, None),
                    ],
                ),
            ]),
            "unity-hdrp-v1" => expected.push((
                "mask_map",
                vec![
                    map(Metallic, R, false, MetalUse),
                    map(Ao, R, false, Occlusion),
                    constant(1.0, Some(DetailMask)),
                    map(Roughness, R, true, Smoothness),
                ],
            )),
            "godot-v1" | "gltf2-v1" => expected.push(("orm", orm())),
            "blender-v1" => expected.extend([
                ("roughness", vec![map(Roughness, R, false, RoughUse)]),
                ("metallic", vec![map(Metallic, R, false, MetalUse)]),
                ("occlusion", vec![map(Ao, R, false, Occlusion)]),
                ("height", vec![map(Height, R, false, HeightUse)]),
            ]),
            id => panic!("unexpected preset {id}"),
        }
        assert_eq!(
            preset.outputs.len(),
            expected.len(),
            "{}",
            preset.profile.id
        );
        for (name, entries) in expected {
            let output = named(preset, name);
            let hints = output.import_hints.as_ref().unwrap();
            let color = name == "albedo";
            assert_eq!(
                hints.color_tag,
                if color {
                    ExportColorTag::Srgb
                } else {
                    ExportColorTag::Linear
                },
                "{} {name}",
                preset.profile.id
            );
            assert_eq!(
                output.channels.len(),
                entries.len(),
                "{} {name}",
                preset.profile.id
            );
            let mut assignments = Vec::new();
            for (index, (channel, (source, value, invert, usage))) in
                output.channels.iter().zip(entries).enumerate()
            {
                match (channel.source, source) {
                    (PresetChannelSource::Map { map, component, .. }, Some(expected)) => {
                        assert_eq!((map, component), expected)
                    }
                    (PresetChannelSource::Constant(actual), None) => assert_eq!(actual, value),
                    _ => panic!("wrong source: {} {name} channel {index}", preset.profile.id),
                }
                assert_eq!(
                    channel.invert, invert,
                    "{} {name} channel {index}",
                    preset.profile.id
                );
                assert_eq!(
                    channel.transfer,
                    if color {
                        Transfer::Srgb
                    } else {
                        Transfer::Linear
                    },
                    "{} {name} channel {index}",
                    preset.profile.id
                );
                if let Some(usage) = usage {
                    assignments.push(TextureAssignment {
                        usage,
                        component: [R, G, B, A][index],
                    });
                }
            }
            assert_eq!(
                hints.assignments, assignments,
                "{} {name}",
                preset.profile.id
            );
        }
    }
}

#[test]
fn all_builtin_presets_bind_to_a_valid_graph() {
    let graph = material_from_source(SourceDescriptor {
        content_hash: [76; 32],
        extent: Extent {
            width: 16,
            height: 16,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    })
    .unwrap();
    for preset in BuiltinExportPresets.presets() {
        let plan = preset
            .bind(&graph, &[])
            .unwrap_or_else(|error| panic!("{}: {error:?}", preset.profile.id));
        assert_eq!(plan.outputs.len(), preset.outputs.len());
        validate_export_for_graph(&plan, &graph).unwrap();
    }
}
