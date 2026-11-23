//! Mask blending API and behavioral contracts.
use texelith_core::*;

fn map(port: OutputPort, component: Component) -> ChannelSpec {
    ChannelSpec {
        source: ChannelSource::Map(
            OutputRef {
                node: NodeId(1),
                port,
            },
            component,
        ),
        invert: false,
        transfer: Transfer::Linear,
    }
}

fn constant(value: f32) -> ChannelSpec {
    ChannelSpec {
        source: ChannelSource::Constant(value),
        invert: false,
        transfer: Transfer::Linear,
    }
}

fn plan() -> ExportPlan {
    ExportPlan {
        source_extent: Extent {
            width: 17,
            height: 13,
        },
        size: SizePolicy::Keep,
        padding: Boundary::Repeat,
        displacement: DisplacementSettings::default(),
        naming: "T_{name}_{map}".into(),
        profile: EngineProfile {
            id: "custom".into(),
            revision: 1,
            engine_safe_srgb: false,
            normal_convention: NormalConvention::OpenGl,
        },
        outputs: vec![ExportOutput {
            name: "packed".into(),
            format: ExportFormat::Png8,
            channels: vec![
                map(OutputPort::Roughness, Component::R),
                constant(0.0),
                constant(1.0),
            ],
            normal_has_negative_z: false,
            quantization: None,
            import_hints: None,
        }],
    }
}

#[test]
fn arbitrary_components_constants_and_inversion_are_validated() {
    let mut p = plan();
    for component in [Component::R, Component::G, Component::B, Component::A] {
        p.outputs[0].channels[0] = map(OutputPort::Albedo, component);
        assert!(validate_export(&p).is_ok());
    }
    p.outputs[0].channels[1] = constant(0.25);
    p.outputs[0].channels[1].invert = true;
    assert!(validate_export(&p).is_ok());
    for bad in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        p.outputs[0].channels[1] = constant(bad);
        assert!(validate_export(&p).is_err());
    }
}

#[test]
fn engine_safe_srgb_rejects_mixed_color_and_data_but_raw_mix_is_allowed() {
    let mut p = plan();
    p.outputs[0].channels[0] = map(OutputPort::Albedo, Component::R);
    p.outputs[0].channels[0].transfer = Transfer::Srgb;
    p.outputs[0].channels[1] = map(OutputPort::Roughness, Component::R);
    assert!(validate_export(&p).is_ok());
    p.profile.engine_safe_srgb = true;
    assert!(validate_export(&p).is_err());
    p.outputs[0].channels[0].transfer = Transfer::Linear;
    p.outputs[0].channels[1].transfer = Transfer::Srgb;
    assert!(
        validate_export(&p).is_err(),
        "data must never receive sRGB transfer"
    );
}

#[test]
fn formats_limit_channels_and_bc5_rejects_negative_z() {
    let mut p = plan();
    p.outputs[0].format = ExportFormat::Png16;
    assert!(validate_export(&p).is_ok());
    p.outputs[0].format = ExportFormat::Exr32;
    assert!(
        validate_export(&p).is_err(),
        "EXR displacement has one Y channel"
    );
    p.outputs[0].channels = vec![map(OutputPort::Height, Component::R)];
    assert!(validate_export(&p).is_ok());
    p.outputs[0].format = ExportFormat::DdsBc5;
    p.outputs[0].channels = vec![
        map(OutputPort::Normal, Component::R),
        map(OutputPort::Normal, Component::G),
    ];
    assert!(validate_export(&p).is_ok(), "Keep preserves NPOT BC5");
    p.size = SizePolicy::Resize {
        width: 16,
        height: 16,
    };
    assert!(validate_export(&p).is_ok());
    p.outputs[0].normal_has_negative_z = true;
    assert!(validate_export(&p).is_err());
}

#[test]
fn explicit_sizes_are_bounded_and_keep_preserves_npot() {
    let mut p = plan();
    assert!(validate_export(&p).is_ok());
    p.source_extent = Extent {
        width: 1,
        height: 1,
    };
    for (width, height, valid) in [
        (1, 1, true),
        (4096, 4096, true),
        (0, 1, false),
        (1, 0, false),
        (4097, 1, false),
        (1, 4097, false),
    ] {
        for size in [
            SizePolicy::Resize { width, height },
            SizePolicy::Pad { width, height },
        ] {
            p.size = size;
            assert_eq!(validate_export(&p).is_ok(), valid, "{width}x{height}");
        }
    }
}

#[test]
fn export_commands_capture_settings_and_only_dirty_the_document() {
    let source = SourceDescriptor {
        content_hash: [75; 32],
        extent: Extent {
            width: 16,
            height: 16,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 1,
        graph: material_from_source(source).unwrap(),
    });
    let ids: Vec<_> = command_descriptors()
        .iter()
        .map(|d| d.id.as_str())
        .collect();
    for id in ["export.set_settings", "export.request"] {
        assert!(ids.contains(&id));
    }
    let before = session.snapshot().unwrap();
    let revision = session.document_revision();
    let request_id = session.request_id();
    let mut settings = plan();
    settings.source_extent = before
        .graph
        .material_output_extent(before.graph.selected_output_ref())
        .unwrap();
    settings.outputs[0].channels[0].source = ChannelSource::Map(
        before.graph.material_outputs().unwrap()[&MapKind::Roughness],
        Component::R,
    );
    let set: CommandRequest = serde_json::from_value(serde_json::json!({
        "id": "export.set_settings", "payload": { "settings": settings }
    }))
    .unwrap();
    session.dispatch(set).unwrap();
    assert_eq!(session.snapshot().unwrap(), before);
    assert!(session.document_revision() > revision);
    assert_eq!(session.request_id(), request_id);
    assert!(session.is_dirty());
    let saved_revision = session.document_revision();
    let request: CommandRequest = serde_json::from_value(serde_json::json!({
        "id": "export.request", "payload": {}
    }))
    .unwrap();
    let outcome = session.dispatch(request).unwrap();
    assert_eq!(outcome.effects.len(), 1);
    assert!(
        matches!(&outcome.effects[0], Effect::Export { plan: captured, .. } if captured == &settings)
    );
    assert_eq!(session.document_revision(), saved_revision);
    assert_eq!(session.snapshot().unwrap(), before);
}

#[test]
fn naming_rejects_nonportable_and_unknown_tokens() {
    let mut p = plan();
    for good in ["T_{name}_{map}", "{preset}_{index}"] {
        p.naming = good.into();
        assert!(validate_export(&p).is_ok(), "{good}");
    }
    for bad in [
        "",
        "../{name}",
        "C:\\{name}",
        "/{name}",
        "{unknown}",
        "CON",
        "a|b",
        "a\nb",
    ] {
        p.naming = bad.into();
        assert!(validate_export(&p).is_err(), "{bad:?}");
    }
}

#[test]
fn convention_conversion_is_applied_once() {
    let normal = [0.3, 0.4, 0.8660254];
    let gl = convert_normal_convention(normal, NormalConvention::OpenGl, NormalConvention::OpenGl);
    let dx = convert_normal_convention(normal, NormalConvention::OpenGl, NormalConvention::DirectX);
    assert_eq!(gl, normal);
    assert_eq!(dx, [normal[0], -normal[1], normal[2]]);
    assert_eq!(
        convert_normal_convention(dx, NormalConvention::DirectX, NormalConvention::DirectX),
        dx
    );
    assert_eq!(
        convert_normal_convention(dx, NormalConvention::DirectX, NormalConvention::OpenGl),
        normal
    );
}

fn export_session() -> (Session, ExportPlan) {
    let source = SourceDescriptor {
        content_hash: [75; 32],
        extent: Extent {
            width: 17,
            height: 13,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let graph = material_from_source(source).unwrap();
    let mut settings = plan();
    settings.outputs[0].channels[0].source = ChannelSource::Map(
        graph.material_outputs().unwrap()[&MapKind::Roughness],
        Component::R,
    );
    (
        Session::from_snapshot(EvaluationSnapshot {
            generation: 10,
            graph,
        }),
        settings,
    )
}

#[test]
fn invalid_references_and_stale_extents_are_rejected_transactionally() {
    let (mut session, settings) = export_session();
    let before = session.snapshot();
    let revision = session.document_revision();
    let history = session.history();
    for reference in [
        OutputRef {
            node: NodeId(999),
            port: OutputPort::Roughness,
        },
        OutputRef {
            node: NodeId(1),
            port: OutputPort::Roughness,
        },
    ] {
        let mut bad = settings.clone();
        bad.outputs[0].channels[0].source = ChannelSource::Map(reference, Component::R);
        assert!(
            session
                .dispatch(CommandRequest::SetExportSettings { settings: bad })
                .is_err()
        );
    }
    let mut bad = settings.clone();
    bad.source_extent.width += 1;
    assert!(
        session
            .dispatch(CommandRequest::SetExportSettings { settings: bad })
            .is_err()
    );
    assert_eq!(session.snapshot(), before);
    assert_eq!(session.history(), history);
    assert_eq!(session.document_revision(), revision);
    assert_eq!(session.export_settings(), None);
    assert!(
        Session::new()
            .dispatch(CommandRequest::RequestExport {})
            .is_err()
    );
    assert!(session.dispatch(CommandRequest::RequestExport {}).is_err());
}

#[test]
fn export_capture_survives_live_graph_settings_and_source_replacement() {
    let (mut session, settings) = export_session();
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: settings.clone(),
        })
        .unwrap();
    let before = session.snapshot().unwrap();
    let effect = session
        .dispatch(CommandRequest::RequestExport {})
        .unwrap()
        .effects
        .remove(0);
    let Effect::Export {
        plan: captured,
        capture,
    } = &effect
    else {
        panic!("export effect")
    };
    assert_eq!(captured, &settings);
    assert_eq!(capture.graph, before.graph);
    assert_eq!(capture.generation, before.generation);
    assert_eq!(capture.sources[0].content_hash, [75; 32]);
    let encoded = serde_json::to_value(&effect).unwrap();
    assert_eq!(serde_json::from_value::<Effect>(encoded).unwrap(), effect);
    let roughness = before.graph.material_outputs().unwrap()[&MapKind::Roughness];
    session
        .dispatch(CommandRequest::SetParameter {
            address: ParameterAddress {
                node: roughness.node,
                parameter: ParameterId::Base,
            },
            value: ParameterValue::Float(0.7),
            gesture: None,
        })
        .unwrap();
    let next = session
        .dispatch(CommandRequest::RequestExport {})
        .unwrap()
        .effects
        .remove(0);
    let Effect::Export { capture: next, .. } = next else {
        panic!("export effect")
    };
    assert_ne!(capture.graph_hash, next.graph_hash);
    let mut changed = settings.clone();
    changed.naming = "changed_{map}".into();
    session
        .dispatch(CommandRequest::SetExportSettings { settings: changed })
        .unwrap();
    session
        .dispatch(CommandRequest::NewFromImage {
            token: ImportToken(77),
            source: OpaqueLocator("source".into()),
        })
        .unwrap();
    session
        .complete_import(
            ImportToken(77),
            SourceDescriptor {
                content_hash: [99; 32],
                extent: settings.source_extent,
                interpretation: InputInterpretation::LinearHeight,
            },
        )
        .unwrap();
    assert_eq!(capture.graph, before.graph);
    assert_eq!(captured, &settings);
    assert_eq!(capture.sources[0].content_hash, [75; 32]);
}

#[test]
fn export_settings_noop_undo_redo_preserve_evaluation_and_preview_identity() {
    let (mut session, settings) = export_session();
    let before = session.snapshot();
    let request_id = session.request_id();
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: settings.clone(),
        })
        .unwrap();
    let revision = session.document_revision();
    let history = session.history();
    let outcome = session
        .dispatch(CommandRequest::SetExportSettings {
            settings: settings.clone(),
        })
        .unwrap();
    assert!(outcome.snapshot.is_none());
    assert_eq!(session.document_revision(), revision);
    assert_eq!(session.history(), history);
    assert!(history.byte_count > std::mem::size_of::<ExportPlan>());
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(session.export_settings(), None);
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(session.export_settings(), Some(&settings));
    assert_eq!(session.snapshot(), before);
    assert_eq!(session.request_id(), request_id);
}

#[test]
fn channel_semantics_expose_bit_depth_raw_warning_and_generated_alpha() {
    let mut p = plan();
    for (format, depth) in [
        (ExportFormat::Png8, 8),
        (ExportFormat::Png16, 16),
        (ExportFormat::Exr32, 32),
        (ExportFormat::DdsBc5, 8),
    ] {
        assert_eq!(format.bit_depth(), depth);
    }
    let output = &mut p.outputs[0];
    output.channels[0] = map(OutputPort::Albedo, Component::R);
    output.channels[0].transfer = Transfer::Srgb;
    let semantics = output.semantics();
    assert_eq!(semantics.interpretation, ExportInterpretation::Packed);
    assert_eq!(semantics.color_tag, ExportColorTag::Raw);
    assert_eq!(semantics.warnings, vec![ExportWarning::MixedTransfer]);
    assert_eq!(semantics.bit_depth, 8);
    let alpha = map(OutputPort::Albedo, Component::A);
    assert_eq!(alpha.effective_source(), ChannelSource::Constant(1.0));
    output.channels = vec![map(OutputPort::Roughness, Component::G)];
    assert!(validate_export(&p).is_err(), "scalar data has only R");
    p.outputs[0].channels = vec![map(OutputPort::Normal, Component::A)];
    assert!(validate_export(&p).is_err(), "normal has xyz only");
}

#[test]
fn engine_safe_srgb_allows_linear_alpha_but_rejects_partial_rgb_transfer() {
    let mut p = plan();
    p.profile.engine_safe_srgb = true;
    p.outputs[0].channels = [Component::R, Component::G, Component::B]
        .map(|c| {
            let mut channel = map(OutputPort::Albedo, c);
            channel.transfer = Transfer::Srgb;
            channel
        })
        .to_vec();
    p.outputs[0].channels.push(constant(1.0));
    assert!(
        validate_export(&p).is_ok(),
        "linear alpha is valid with sRGB RGB"
    );
    assert_eq!(p.outputs[0].semantics().color_tag, ExportColorTag::Srgb);
    p.outputs[0].channels[1].transfer = Transfer::Linear;
    assert!(validate_export(&p).is_err());
    p.profile.engine_safe_srgb = false;
    assert!(validate_export(&p).is_ok());
    p.outputs[0].channels[3] = map(OutputPort::Albedo, Component::R);
    p.outputs[0].channels[3].transfer = Transfer::Srgb;
    assert!(
        validate_export(&p).is_err(),
        "destination alpha is always linear"
    );
}

#[test]
fn bc5_requires_components_from_one_normal_and_errors_identify_negative_z() {
    let mut p = plan();
    p.size = SizePolicy::Resize {
        width: 16,
        height: 16,
    };
    p.outputs[0].format = ExportFormat::DdsBc5;
    p.outputs[0].channels = vec![
        map(OutputPort::Normal, Component::R),
        map(OutputPort::Normal, Component::G),
    ];
    p.outputs[0].normal_has_negative_z = true;
    assert!(
        validate_export(&p)
            .unwrap_err()
            .to_string()
            .contains("negative-Z")
    );
    p.outputs[0].normal_has_negative_z = false;
    p.outputs[0].channels[1].source = ChannelSource::Map(
        OutputRef {
            node: NodeId(2),
            port: OutputPort::Normal,
        },
        Component::G,
    );
    assert!(validate_export(&p).is_err());
}

#[test]
fn naming_and_padding_reject_portability_collisions_and_cropping() {
    let mut p = plan();
    for bad in [
        "CON.png",
        "CONIN$",
        "CONOUT$.png",
        "nul.txt",
        "LPT9",
        "COM¹",
        ".",
        "   ",
        "{name",
        "name}",
        "{{name}}",
        "trailing.",
        "trailing ",
    ] {
        p.naming = bad.into();
        assert!(validate_export(&p).is_err(), "{bad:?}");
    }
    p.naming = "{name}".into();
    let mut duplicate = p.outputs[0].clone();
    duplicate.name = "PACKED".into();
    p.outputs.push(duplicate);
    assert!(validate_export(&p).is_err());
    p.outputs.pop();
    p.size = SizePolicy::Pad {
        width: 16,
        height: 16,
    };
    assert!(validate_export(&p).is_err());
    for extent in [
        Extent {
            width: 0,
            height: 1,
        },
        Extent {
            width: 4097,
            height: 1,
        },
    ] {
        p.source_extent = extent;
        p.size = SizePolicy::Keep;
        assert!(validate_export(&p).is_err());
    }
}

#[test]
fn typed_preset_table_looks_up_exact_revision_without_json() {
    struct Table(Vec<ExportPreset>);
    impl ExportPresetTable for Table {
        fn presets(&self) -> &[ExportPreset] {
            &self.0
        }
    }
    let table = Table(vec![preset()]);
    assert_eq!(table.find("custom", 1), Some(&table.0[0]));
    assert!(table.find("custom", 2).is_none());
    assert!(table.find("unknown", 1).is_none());
}

#[test]
fn format_precision_cannot_be_overridden_by_unknown_settings_fields() {
    let mut value = serde_json::to_value(plan()).unwrap();
    value["outputs"][0]["bit_depth"] = serde_json::json!(32);
    assert!(serde_json::from_value::<ExportPlan>(value).is_err());
    let mut p = plan();
    for count in [0, 5] {
        p.outputs[0].channels = vec![constant(0.5); count];
        assert!(validate_export(&p).is_err());
    }
    p.outputs[0].channels = vec![map(OutputPort::Albedo, Component::R)];
    p.outputs[0].format = ExportFormat::Exr32;
    assert!(
        validate_export(&p).is_err(),
        "EXR32 is linear height displacement only"
    );
    p.outputs[0].channels = vec![map(OutputPort::Height, Component::R)];
    assert!(validate_export(&p).is_ok());
    assert_eq!(
        p.outputs[0].semantics().interpretation,
        ExportInterpretation::Displacement
    );
    assert_eq!(p.outputs[0].semantics().color_tag, ExportColorTag::Linear);
    assert_eq!(p.outputs[0].semantics().bit_depth, 32);
}

#[test]
fn captured_normal_metadata_converts_once_and_hash_retains_seeds() {
    let (session, mut p) = export_session();
    let mut snapshot = session.snapshot().unwrap();
    let normal = snapshot.graph.material_outputs().unwrap()[&MapKind::Normal];
    let mut dto = MaterialGraphDto::from(snapshot.graph);
    let node = dto
        .graph
        .iter_mut()
        .find(|node| node.id == normal.node)
        .unwrap();
    let MaterialNodeParams::NormalFromHeight { convention, .. } = &mut node.params else {
        panic!("normal node")
    };
    *convention = NormalConvention::DirectX;
    snapshot.graph = Graph::try_from(dto).unwrap();
    p.outputs[0].channels = vec![
        map(OutputPort::Normal, Component::R),
        map(OutputPort::Normal, Component::G),
        map(OutputPort::Normal, Component::B),
    ];
    for channel in &mut p.outputs[0].channels {
        let ChannelSource::Map(_, component) = channel.source else {
            unreachable!()
        };
        channel.source = ChannelSource::Map(normal, component);
    }
    p.profile.normal_convention = NormalConvention::DirectX;
    let capture = ExportCapture::new(&snapshot, 1, &p).unwrap();
    let from = capture
        .graph
        .output_normal_convention(normal)
        .unwrap()
        .unwrap();
    let n = [0.3, -0.4, 0.8660254];
    assert_eq!(
        convert_normal_convention(n, from, p.profile.normal_convention),
        n
    );
    assert_eq!(
        p.outputs[0].semantics().interpretation,
        ExportInterpretation::Normal
    );
    let mut dto = MaterialGraphDto::from(snapshot.graph.clone());
    let node = dto
        .graph
        .iter_mut()
        .find(|node| matches!(node.params, MaterialNodeParams::Seamless { .. }))
        .unwrap();
    let MaterialNodeParams::Seamless { method, .. } = &mut node.params else {
        unreachable!()
    };
    *method = SeamlessMethod::Splat {
        seed: 75,
        grid: 4,
        overlap: 0.25,
    };
    snapshot.graph = Graph::try_from(dto).unwrap();
    let seeded = ExportCapture::new(&snapshot, 1, &p).unwrap();
    let mut dto = MaterialGraphDto::from(snapshot.graph.clone());
    let node = dto
        .graph
        .iter_mut()
        .find(|node| {
            matches!(
                node.params,
                MaterialNodeParams::Seamless {
                    method: SeamlessMethod::Splat { .. },
                    ..
                }
            )
        })
        .unwrap();
    let MaterialNodeParams::Seamless {
        method: SeamlessMethod::Splat { seed, .. },
        ..
    } = &mut node.params
    else {
        unreachable!()
    };
    *seed = 76;
    snapshot.graph = Graph::try_from(dto).unwrap();
    let reseeded = ExportCapture::new(&snapshot, 1, &p).unwrap();
    assert_ne!(seeded.graph_hash, reseeded.graph_hash);
    snapshot.generation += 1;
    let same_recipe = ExportCapture::new(&snapshot, 2, &p).unwrap();
    assert_eq!(reseeded.graph_hash, same_recipe.graph_hash);
    assert_eq!(reseeded.sources, same_recipe.sources);
}

#[test]
fn legacy_graph_capture_and_constant_outputs_preserve_source_identity() {
    let (mut session, mut p) = export_session();
    session
        .dispatch(CommandRequest::NewFromImage {
            token: ImportToken(1),
            source: OpaqueLocator("source".into()),
        })
        .unwrap();
    session
        .complete_import(
            ImportToken(1),
            SourceDescriptor {
                content_hash: [33; 32],
                extent: p.source_extent,
                interpretation: InputInterpretation::LinearHeight,
            },
        )
        .unwrap();
    p.outputs[0].channels = vec![constant(1.0)];
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: p.clone(),
        })
        .unwrap();
    let effect = session
        .dispatch(CommandRequest::RequestExport {})
        .unwrap()
        .effects
        .remove(0);
    let Effect::Export { capture, .. } = effect else {
        panic!("export effect")
    };
    assert!(!capture.graph.is_material());
    assert_eq!(capture.sources[0].content_hash, [33; 32]);
    p.outputs[0].channels = vec![ChannelSpec {
        source: ChannelSource::Map(
            OutputRef {
                node: NodeId(2),
                port: OutputPort::Normal,
            },
            Component::R,
        ),
        invert: false,
        transfer: Transfer::Linear,
    }];
    assert!(validate_export_for_graph(&p, &capture.graph).is_err());
    p.outputs[0].channels[0].source = ChannelSource::Map(
        OutputRef {
            node: NodeId(2),
            port: OutputPort::Height,
        },
        Component::R,
    );
    assert!(validate_export_for_graph(&p, &capture.graph).is_ok());
}

fn preset() -> ExportPreset {
    let p = plan();
    ExportPreset {
        profile: p.profile,
        naming: p.naming,
        size: p.size,
        padding: p.padding,
        displacement: p.displacement,
        outputs: vec![PresetOutput {
            name: "packed".into(),
            format: ExportFormat::Png8,
            channels: vec![
                PresetChannelSpec {
                    source: PresetChannelSource::Map {
                        map: MapKind::Roughness,
                        component: Component::R,
                        missing: MissingMap::Constant(0.5),
                    },
                    invert: true,
                    transfer: Transfer::Linear,
                },
                PresetChannelSpec {
                    source: PresetChannelSource::Constant(0.0),
                    invert: false,
                    transfer: Transfer::Linear,
                },
                PresetChannelSpec {
                    source: PresetChannelSource::Constant(1.0),
                    invert: false,
                    transfer: Transfer::Linear,
                },
            ],
            quantization: None,
            import_hints: Some(ExportImportHints {
                color_tag: ExportColorTag::Linear,
                normal_map: false,
                normal_compression: false,
                assignments: vec![TextureAssignment {
                    usage: TextureUsage::Smoothness,
                    component: Component::R,
                }],
                instructions: vec!["Disable sRGB for packed data".into()],
            }),
        }],
    }
}

#[test]
fn png_only_accepts_scalar_rgb_and_rgba_for_each_precision() {
    let mut p = plan();
    for format in [ExportFormat::Png8, ExportFormat::Png16] {
        p.outputs[0].format = format;
        for count in 0..=5 {
            p.outputs[0].channels = vec![constant(0.5); count];
            assert_eq!(
                validate_export(&p).is_ok(),
                matches!(count, 1 | 3 | 4),
                "{format:?}, {count} channels"
            );
        }
    }
}

#[test]
fn displacement_and_integer_quantization_are_finite_and_format_specific() {
    let mut p = plan();
    assert_eq!(p.displacement.midlevel, 0.5);
    assert_eq!(p.displacement.scale, 1.0);
    assert_eq!(p.displacement.units, DisplacementUnits::Unitless);
    for midlevel in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        p.displacement.midlevel = midlevel;
        assert!(validate_export(&p).is_err());
    }
    p.displacement.midlevel = 0.25;
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
        p.displacement.scale = scale;
        assert!(validate_export(&p).is_err());
    }
    p.displacement.scale = 2.0;
    for (min, max, valid) in [
        (0.0, 1.0, true),
        (0.2, 0.8, true),
        (0.5, 0.5, false),
        (0.8, 0.2, false),
        (-0.1, 1.0, false),
        (0.0, 1.1, false),
        (f32::NAN, 1.0, false),
        (0.0, f32::INFINITY, false),
    ] {
        p.outputs[0].quantization = Some(IntegerQuantization {
            clamp_min: min,
            clamp_max: max,
            dither: IntegerDither::Ordered4x4,
        });
        for format in [ExportFormat::Png8, ExportFormat::Png16] {
            p.outputs[0].format = format;
            assert_eq!(
                validate_export(&p).is_ok(),
                valid,
                "{format:?} [{min}, {max}]"
            );
        }
    }
    p.outputs[0].channels = vec![map(OutputPort::Height, Component::R)];
    p.outputs[0].format = ExportFormat::Exr32;
    p.outputs[0].quantization = Some(IntegerQuantization::default());
    assert!(
        validate_export(&p).is_err(),
        "float displacement must not use integer clamp/dither"
    );
    p.outputs[0].quantization = None;
    assert!(validate_export(&p).is_ok());
    p.outputs[0].format = ExportFormat::DdsBc5;
    p.outputs[0].channels = vec![
        map(OutputPort::Normal, Component::R),
        map(OutputPort::Normal, Component::G),
    ];
    p.size = SizePolicy::Resize {
        width: 16,
        height: 16,
    };
    p.outputs[0].quantization = Some(IntegerQuantization::default());
    assert!(
        validate_export(&p).is_err(),
        "BC5 has its own block quantization"
    );
    p.outputs[0].quantization = None;
    assert!(validate_export(&p).is_ok());
}

#[test]
fn conversion_metadata_roundtrips_and_survives_settings_undo_and_job_capture() {
    let (mut session, mut settings) = export_session();
    settings.size = SizePolicy::Pad {
        width: 32,
        height: 16,
    };
    settings.displacement = DisplacementSettings {
        midlevel: 0.25,
        scale: 12.5,
        units: DisplacementUnits::Centimeters,
    };
    settings.outputs[0].quantization = Some(IntegerQuantization {
        clamp_min: 0.1,
        clamp_max: 0.9,
        dither: IntegerDither::Ordered4x4,
    });
    for padding in [Boundary::Clamp, Boundary::Repeat] {
        settings.padding = padding;
        let json = serde_json::to_value(&settings).unwrap();
        assert_eq!(
            serde_json::from_value::<ExportPlan>(json).unwrap(),
            settings
        );
        session
            .dispatch(CommandRequest::SetExportSettings {
                settings: settings.clone(),
            })
            .unwrap();
    }
    let effect = session
        .dispatch(CommandRequest::RequestExport {})
        .unwrap()
        .effects
        .remove(0);
    let captured = effect.clone();
    let json = serde_json::to_value(&effect).unwrap();
    assert_eq!(serde_json::from_value::<Effect>(json).unwrap(), captured);
    let mut changed = settings.clone();
    changed.displacement = DisplacementSettings::default();
    changed.outputs[0].quantization = None;
    changed.padding = Boundary::Clamp;
    let before = session.snapshot();
    let request_id = session.request_id();
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: changed.clone(),
        })
        .unwrap();
    session.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(session.export_settings(), Some(&settings));
    session.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(session.export_settings(), Some(&changed));
    assert_eq!(session.snapshot(), before);
    assert_eq!(session.request_id(), request_id);
    let Effect::Export { plan, .. } = captured else {
        panic!("export effect")
    };
    assert_eq!(plan, settings);
}

#[test]
fn one_semantic_preset_binds_to_remapped_ids_and_different_extents() {
    let (session, _) = export_session();
    let graph = session.snapshot().unwrap().graph;
    let preset = preset();
    let first = preset.bind(&graph, &[]).unwrap();
    let second_graph = material_from_source(SourceDescriptor {
        content_hash: [76; 32],
        extent: Extent {
            width: 31,
            height: 23,
        },
        interpretation: InputInterpretation::SrgbPhoto,
    })
    .unwrap();
    let mut dto = MaterialGraphDto::from(second_graph);
    for node in &mut dto.graph {
        node.id.0 += 100;
        for edge in &mut node.inputs {
            edge.source.node.0 += 100;
        }
    }
    for reference in dto.outputs.values_mut() {
        reference.node.0 += 100;
    }
    dto.selected_output.node.0 += 100;
    let graph = Graph::try_from(dto).unwrap();
    let second = preset.bind(&graph, &[]).unwrap();
    assert_eq!(
        first.source_extent,
        Extent {
            width: 17,
            height: 13
        }
    );
    assert_eq!(
        second.source_extent,
        Extent {
            width: 31,
            height: 23
        }
    );
    assert_eq!(
        second.outputs[0].channels[0].source,
        ChannelSource::Map(
            graph.material_outputs().unwrap()[&MapKind::Roughness],
            Component::R
        )
    );
    assert_ne!(
        first.outputs[0].channels[0].source,
        second.outputs[0].channels[0].source
    );
    assert!(second.outputs[0].channels[0].invert);
    assert_eq!(
        second.outputs[0].import_hints,
        preset.outputs[0].import_hints
    );
    assert_eq!(
        serde_json::from_value::<ExportPreset>(serde_json::to_value(&preset).unwrap()).unwrap(),
        preset
    );
    let mut session = Session::from_snapshot(EvaluationSnapshot {
        generation: 1,
        graph,
    });
    session
        .dispatch(CommandRequest::SetExportSettings {
            settings: second.clone(),
        })
        .unwrap();
    let effect = session
        .dispatch(CommandRequest::RequestExport {})
        .unwrap()
        .effects
        .remove(0);
    assert!(matches!(effect, Effect::Export { plan, .. } if plan == second));
}

#[test]
fn absent_preset_maps_use_declared_fallback_or_report_an_error() {
    let (session, _) = export_session();
    let mut dto = MaterialGraphDto::from(session.snapshot().unwrap().graph);
    dto.outputs.remove(&MapKind::Roughness);
    let graph = Graph::try_from(dto).unwrap();
    let mut preset = preset();
    let bound = preset.bind(&graph, &[]).unwrap();
    assert_eq!(
        bound.outputs[0].channels[0].source,
        ChannelSource::Constant(0.5)
    );
    assert!(
        bound.outputs[0].channels[0].invert,
        "fallback preserves smoothness inversion"
    );
    for missing in [
        MissingMap::Error,
        MissingMap::Constant(f32::NAN),
        MissingMap::Constant(-0.1),
        MissingMap::Constant(1.1),
    ] {
        preset.outputs[0].channels[0].source = PresetChannelSource::Map {
            map: MapKind::Roughness,
            component: Component::R,
            missing,
        };
        assert!(preset.bind(&graph, &[]).is_err());
    }
    preset.outputs[0].channels[0].source = PresetChannelSource::Map {
        map: MapKind::Roughness,
        component: Component::R,
        missing: MissingMap::OmitOutput,
    };
    let mut required = preset.outputs[0].clone();
    required.name = "required".into();
    required.channels[0].source = PresetChannelSource::Constant(0.5);
    preset.outputs.push(required);
    let bound = preset.bind(&graph, &[]).unwrap();
    assert_eq!(bound.outputs.len(), 1);
    assert_eq!(bound.outputs[0].name, "required");
}

#[test]
fn import_hints_cannot_override_color_or_normal_semantics() {
    let (session, _) = export_session();
    let graph = session.snapshot().unwrap().graph;
    let mut p = preset().bind(&graph, &[]).unwrap();
    let good = p.outputs[0].import_hints.clone().unwrap();
    p.outputs[0].import_hints.as_mut().unwrap().color_tag = ExportColorTag::Srgb;
    assert!(validate_export(&p).is_err());
    p.outputs[0].import_hints = Some(good.clone());
    p.outputs[0].import_hints.as_mut().unwrap().normal_map = true;
    assert!(validate_export(&p).is_err());
    p.outputs[0].import_hints = Some(good.clone());
    p.outputs[0]
        .import_hints
        .as_mut()
        .unwrap()
        .normal_compression = true;
    assert!(validate_export(&p).is_err());
    p.outputs[0].import_hints = Some(good);
    p.outputs[0].import_hints.as_mut().unwrap().assignments[0].component = Component::A;
    assert!(validate_export(&p).is_err());
}

#[test]
fn absent_map_fallback_does_not_hide_an_invalid_component() {
    let (session, _) = export_session();
    let mut dto = MaterialGraphDto::from(session.snapshot().unwrap().graph);
    dto.outputs.remove(&MapKind::Roughness);
    let graph = Graph::try_from(dto).unwrap();
    let mut preset = preset();
    preset.outputs[0].channels[0].source = PresetChannelSource::Map {
        map: MapKind::Roughness,
        component: Component::G,
        missing: MissingMap::Constant(0.5),
    };
    assert!(preset.bind(&graph, &[]).is_err());
}

#[test]
fn preset_binding_preserves_normal_target_and_rejects_known_negative_z_bc5() {
    let (session, _) = export_session();
    let graph = session.snapshot().unwrap().graph;
    let mut preset = preset();
    preset.size = SizePolicy::Resize {
        width: 16,
        height: 16,
    };
    preset.profile.normal_convention = NormalConvention::DirectX;
    let output = &mut preset.outputs[0];
    output.format = ExportFormat::DdsBc5;
    output.channels = [Component::R, Component::G]
        .map(|component| PresetChannelSpec {
            source: PresetChannelSource::Map {
                map: MapKind::Normal,
                component,
                missing: MissingMap::Error,
            },
            invert: false,
            transfer: Transfer::Linear,
        })
        .to_vec();
    output.import_hints = Some(ExportImportHints {
        color_tag: ExportColorTag::Linear,
        normal_map: true,
        normal_compression: true,
        assignments: vec![],
        instructions: vec!["Import as BC5 normal; sRGB off".into()],
    });
    let bound = preset.bind(&graph, &[]).unwrap();
    assert_eq!(bound.profile.normal_convention, NormalConvention::DirectX);
    assert_eq!(
        bound.outputs[0].channels[0].source,
        ChannelSource::Map(
            graph.material_outputs().unwrap()[&MapKind::Normal],
            Component::R
        )
    );
    assert!(
        bound.outputs[0]
            .import_hints
            .as_ref()
            .unwrap()
            .normal_compression
    );
    assert!(
        preset
            .bind(&graph, &[MapKind::Normal])
            .unwrap_err()
            .to_string()
            .contains("negative-Z")
    );
}
