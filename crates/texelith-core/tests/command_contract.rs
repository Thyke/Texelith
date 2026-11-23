use std::collections::HashSet;
use texelith_core::{
    Brush, BrushSample, CommandId, CommandRequest, Effect, GestureId, ImportToken,
    MATERIAL_PARAMETER_DESCRIPTORS, MapKind, MaskMode, NodeId, OpaqueLocator, ParameterAddress,
    ParameterId, ParameterValue, PayloadSchema, StrokeId, command_descriptors,
};

#[test]
fn descriptors_and_request_payloads_have_a_one_to_one_mapping() {
    let address = ParameterAddress {
        node: NodeId(2),
        parameter: ParameterId::Gamma,
    };
    let requests = [
        (
            CommandRequest::NewFromImage {
                token: ImportToken(1),
                source: OpaqueLocator("source-token".into()),
            },
            "document.new_from_image",
            PayloadSchema::Import,
        ),
        (
            CommandRequest::SetParameter {
                address,
                value: ParameterValue::Float(2.0),
                gesture: Some(GestureId(5)),
            },
            "node.set_parameter",
            PayloadSchema::Parameter {
                parameters: &MATERIAL_PARAMETER_DESCRIPTORS,
            },
        ),
        (
            CommandRequest::BeginGesture {
                gesture: GestureId(5),
                address,
            },
            "edit.begin_gesture",
            PayloadSchema::BeginGesture,
        ),
        (
            CommandRequest::CommitGesture {
                gesture: GestureId(5),
            },
            "edit.commit_gesture",
            PayloadSchema::Gesture,
        ),
        (
            CommandRequest::CancelGesture {
                gesture: GestureId(5),
            },
            "edit.cancel_gesture",
            PayloadSchema::Gesture,
        ),
        (
            CommandRequest::BeginStroke {
                node: NodeId(4),
                stroke_id: StrokeId(6),
                brush: Brush {
                    radius: 0.2,
                    hardness: 0.5,
                    opacity: 0.75,
                    mode: MaskMode::Add,
                },
                sample: BrushSample {
                    uv: [0.1, 0.5],
                    pressure: 1.0,
                },
            },
            "mask.begin_stroke",
            PayloadSchema::BeginStroke,
        ),
        (
            CommandRequest::AppendStroke {
                stroke_id: StrokeId(6),
                samples: vec![BrushSample {
                    uv: [0.3, 0.5],
                    pressure: 0.5,
                }],
            },
            "mask.append_stroke",
            PayloadSchema::AppendStroke,
        ),
        (
            CommandRequest::CommitStroke {
                stroke_id: StrokeId(6),
            },
            "mask.commit_stroke",
            PayloadSchema::Stroke,
        ),
        (
            CommandRequest::CancelStroke {
                stroke_id: StrokeId(6),
            },
            "mask.cancel_stroke",
            PayloadSchema::Stroke,
        ),
        (CommandRequest::Undo, "edit.undo", PayloadSchema::Empty),
        (CommandRequest::Redo, "edit.redo", PayloadSchema::Empty),
        (
            CommandRequest::SelectOutput { output: NodeId(3) },
            "view.select_output",
            PayloadSchema::Output,
        ),
        (
            CommandRequest::SelectMap { map: MapKind::Edge },
            "material.select_map",
            PayloadSchema::Map,
        ),
        (
            CommandRequest::SavePng {
                destination: OpaqueLocator("destination-token".into()),
                map: MapKind::Normal,
            },
            "image.save_png",
            PayloadSchema::Export,
        ),
        (
            CommandRequest::SetVisible {
                map: MapKind::Height,
                visible: false,
            },
            "material.set_visible",
            PayloadSchema::MapVisible,
        ),
        (
            CommandRequest::SetLocked {
                map: MapKind::Height,
                locked: true,
            },
            "material.set_locked",
            PayloadSchema::MapLocked,
        ),
        (
            CommandRequest::SetMode {
                mode: Default::default(),
            },
            "view.set_mode",
            PayloadSchema::ViewMode,
        ),
        (
            CommandRequest::SetMesh {
                mesh: Default::default(),
            },
            "view.set_mesh",
            PayloadSchema::ViewMesh,
        ),
        (
            CommandRequest::SetLight {
                light: Default::default(),
            },
            "view.set_light",
            PayloadSchema::ViewLight,
        ),
        (
            CommandRequest::SetTiles {
                tiles: Default::default(),
            },
            "view.set_tiles",
            PayloadSchema::ViewTiles,
        ),
        (
            CommandRequest::SetPom {
                pom: Default::default(),
            },
            "view.set_pom",
            PayloadSchema::ViewPom,
        ),
        (
            CommandRequest::SetCamera {
                camera: Default::default(),
            },
            "view.set_camera",
            PayloadSchema::ViewCamera,
        ),
        (
            CommandRequest::CaptureA {},
            "view.capture_a",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::CaptureB {},
            "view.capture_b",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::SetSplit { fraction: 0.5 },
            "view.set_split",
            PayloadSchema::ViewSplit,
        ),
        (
            CommandRequest::SetExportSettings {
                settings: texelith_core::ExportPlan {
                    source_extent: texelith_core::Extent {
                        width: 1,
                        height: 1,
                    },
                    size: texelith_core::SizePolicy::Keep,
                    padding: texelith_core::Boundary::Repeat,
                    displacement: texelith_core::DisplacementSettings::default(),
                    naming: "{name}".into(),
                    profile: texelith_core::EngineProfile {
                        id: "custom".into(),
                        revision: 1,
                        engine_safe_srgb: false,
                        normal_convention: texelith_core::NormalConvention::OpenGl,
                    },
                    outputs: vec![texelith_core::ExportOutput {
                        name: "data".into(),
                        format: texelith_core::ExportFormat::Png8,
                        channels: vec![texelith_core::ChannelSpec {
                            source: texelith_core::ChannelSource::Constant(1.0),
                            invert: false,
                            transfer: texelith_core::Transfer::Linear,
                        }],
                        normal_has_negative_z: false,
                        quantization: None,
                        import_hints: None,
                    }],
                },
            },
            "export.set_settings",
            PayloadSchema::ExportSettings,
        ),
        (
            CommandRequest::RequestExport {},
            "export.request",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::BatchStart {},
            "batch.start",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::BatchCancel {},
            "batch.cancel",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::BatchRetryFailed {},
            "batch.retry_failed",
            PayloadSchema::Empty,
        ),
        (
            CommandRequest::OpenProject {
                token: texelith_core::ProjectOpenToken(84),
                source: OpaqueLocator("project-token".into()),
            },
            "project.open",
            PayloadSchema::ProjectOpen,
        ),
        (
            CommandRequest::ConfigureMaterial {
                seamless: true,
                painted_mask: false,
            },
            "material.configure",
            PayloadSchema::MaterialOptions,
        ),
        (
            CommandRequest::InspectSeams { enabled: true },
            "view.inspect_seams",
            PayloadSchema::Toggle,
        ),
        (
            CommandRequest::CancelExport { job_id: 82 },
            "export.cancel",
            PayloadSchema::ExportJob,
        ),
        (
            CommandRequest::SaveExportPreset {
                destination: OpaqueLocator("presets-token".into()),
                preset: texelith_core::ExportPresetTable::presets(
                    &texelith_core::BuiltinExportPresets,
                )[0]
                .clone(),
            },
            "export.save_preset",
            PayloadSchema::ExportPreset,
        ),
    ];
    let requests = requests
        .into_iter()
        .chain([
            (
                CommandRequest::SaveProject {
                    destination: OpaqueLocator("save".into()),
                },
                "project.save",
                PayloadSchema::ProjectSave,
            ),
            (
                CommandRequest::SaveProjectAs {
                    destination: OpaqueLocator("save-as".into()),
                },
                "project.save_as",
                PayloadSchema::ProjectSave,
            ),
        ])
        .collect::<Vec<_>>();
    let descriptors = command_descriptors();
    assert_eq!(descriptors.len(), requests.len());
    let mut ids = HashSet::new();
    for (request, stable_id, payload) in requests {
        assert!(ids.insert(request.id()));
        assert_eq!(request.id().as_str(), stable_id);
        let matching: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.id == request.id())
            .collect();
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].payload, payload);
        assert!(!matching[0].label.is_empty());
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["id"], stable_id);
        assert_eq!(
            serde_json::from_value::<CommandRequest>(value).unwrap(),
            request
        );
        let descriptor_json = serde_json::to_value(matching[0]).unwrap();
        assert_eq!(descriptor_json["id"], stable_id);
        assert_eq!(
            serde_json::from_value::<CommandId>(descriptor_json["id"].clone()).unwrap(),
            request.id()
        );
    }
    assert!(
        descriptors
            .iter()
            .all(|descriptor| ids.contains(&descriptor.id))
    );
}

#[test]
fn json_routing_rejects_unknown_ids_mismatched_payloads_and_unknown_fields() {
    for bad in [
        r#"{"id":"unknown.command"}"#,
        r#"{"id":"node.set_parameter","payload":{"output":2}}"#,
        r#"{"id":"view.select_output","payload":{"output":2,"unknown":true}}"#,
        r#"{"id":"view.select_output","payload":{"output":2},"unknown":true}"#,
        r#"{"id":"node.set_parameter","payload":{"address":{"node":2,"parameter":"Unknown"},"value":{"Float":1.0}}}"#,
        r#"{"id":"edit.undo","payload":{"output":2}}"#,
    ] {
        assert!(
            serde_json::from_str::<CommandRequest>(bad).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn effect_dtos_roundtrip_opaque_tokens_without_native_services() {
    let effects = [
        Effect::LoadImage {
            token: ImportToken(900),
            source: OpaqueLocator("opaque:input/ü".into()),
        },
        Effect::SavePng {
            destination: OpaqueLocator("opaque:output".into()),
            map: MapKind::Height,
            output: NodeId(3),
            generation: 42,
        },
    ];
    for effect in effects {
        let json = serde_json::to_string(&effect).unwrap();
        assert_eq!(serde_json::from_str::<Effect>(&json).unwrap(), effect);
    }
}
