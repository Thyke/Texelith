use texelith_core::*;
use texelith_ui_egui::export_perf::{Event, Trace, fixture_graph, fixture_pixels, scenario_plan};

#[test]
fn four_k_rgba_is_admissible_and_incompressible_pixels_round_trip() {
    use texelith_io::encoders::{PngChannels, PngColorIntent, encode_png8};
    use texelith_ui_egui::batch_jobs::{CPU_BUDGET, INTERACTIVE_RESERVATION, export_reservation};
    let reservation = export_reservation(4096, 4096, 4).unwrap();
    assert!(reservation + INTERACTIVE_RESERVATION <= CPU_BUDGET);
    let mut seed = 0x12345678u32;
    let pixels: Vec<_> = (0..4096 * 4096 * 4)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect();
    let png = encode_png8(
        4096,
        4096,
        PngChannels::Rgba,
        PngColorIntent::Linear,
        &pixels,
    )
    .unwrap();
    assert!(
        png.len() > pixels.len(),
        "exercise incompressible output growth"
    );
    assert!((png.capacity() as u64) < reservation);
    // The import decoder's separate 256 MiB input+decode budget rejects a
    // worst-case 64 MiB PNG. Round-trip NPOT noise through that bounded API.
    let sample = &pixels[..257 * 193 * 4];
    let png = encode_png8(257, 193, PngChannels::Rgba, PngColorIntent::Srgb, sample).unwrap();
    let decoded = texelith_io::decode_bytes(&png, texelith_io::ColorIntent::PhotoSrgb).unwrap();
    assert_eq!((decoded.width, decoded.height), (257, 193));
    assert_eq!(decoded.rgba8, sample);
    assert!(export_reservation(u32::MAX, u32::MAX, 4).is_err());
}

#[test]
fn clock_includes_admission_evaluation_encoding_flush_and_publication() {
    let mut trace = Trace::default();
    trace.record(100, Event::Accepted).unwrap();
    trace.record(120, Event::Waiting("albedo".into())).unwrap();
    trace.record(320, Event::Readback("albedo".into())).unwrap();
    trace.record(330, Event::Encoding("albedo".into())).unwrap();
    trace.record(430, Event::Encoded("albedo".into())).unwrap();
    // Encoding is not publication. Include subsequent write, sync and rename.
    assert!(trace.summary(true).is_err());
    trace.record(600, Event::Published).unwrap();
    let metric = trace.summary(true).unwrap();
    assert_eq!(metric.request_to_disk_ns, 500);
    assert_eq!(metric.outputs[0].evaluation_readback_ns, 200);
    assert_eq!(metric.outputs[0].cpu_encode_ns, 100);
    assert!(trace.summary(false).is_err());
    assert!(trace.record(700, Event::Published).is_err());
}

#[test]
fn malformed_or_failed_traces_cannot_claim_a_gate_pass() {
    let mut trace = Trace::default();
    assert!(trace.record(0, Event::Published).is_err());
    trace.record(20, Event::Accepted).unwrap();
    assert!(trace.record(10, Event::Waiting("height".into())).is_err());
    assert!(trace.summary(false).is_err());
}

#[test]
fn canonical_is_seven_images_and_variants_use_production_preset_contracts() {
    let pixels = fixture_pixels(32, 0);
    let descriptor = pixels.descriptor(Extent::new(32, 32).unwrap()).unwrap();
    let graph = fixture_graph(descriptor, false).unwrap();
    let canonical = scenario_plan(&graph, "canonical").unwrap();
    assert_eq!(canonical.outputs.len(), 7);
    for (name, format, channels) in [
        ("albedo", ExportFormat::Png8, 3),
        ("normal", ExportFormat::Png8, 3),
        ("ao", ExportFormat::Png8, 1),
        ("roughness", ExportFormat::Png8, 1),
        ("metallic", ExportFormat::Png8, 1),
        ("edge", ExportFormat::Png8, 1),
        ("height", ExportFormat::Png16, 1),
    ] {
        let output = canonical.outputs.iter().find(|o| o.name == name).unwrap();
        assert_eq!(output.format, format);
        assert_eq!(output.channels.len(), channels);
    }
    for preset in BuiltinExportPresets.presets() {
        assert_eq!(
            scenario_plan(&graph, &preset.profile.id).unwrap(),
            preset.bind(&graph, &[]).unwrap()
        );
    }
    let exr = scenario_plan(&graph, "exr").unwrap();
    assert_eq!(exr.outputs.len(), 7);
    assert_eq!(
        exr.outputs
            .iter()
            .filter(|o| o.format == ExportFormat::Exr32)
            .count(),
        1
    );
    validate_export_for_graph(&exr, &graph).unwrap();
    assert!(
        exr.outputs
            .iter()
            .find(|o| o.format == ExportFormat::Exr32)
            .unwrap()
            .import_hints
            .is_none(),
        "float displacement must not retain PNG16 import instructions"
    );
}

#[test]
fn fixture_is_nonconstant_scharr_offset_and_quality_is_explicit() {
    let pixels = fixture_pixels(32, 0);
    let descriptor = pixels.descriptor(Extent::new(32, 32).unwrap()).unwrap();
    let graph = fixture_graph(descriptor.clone(), false).unwrap();
    let max = fixture_graph(descriptor, true).unwrap();
    let nodes = graph.topological_material_nodes();
    assert!(nodes.iter().any(|n| matches!(
        n.params,
        MaterialNodeParams::NormalFromHeight {
            filter: NormalFilter::Scharr,
            ..
        }
    )));
    assert!(nodes.iter().any(|n| matches!(
        n.params,
        MaterialNodeParams::Seamless {
            method: SeamlessMethod::OffsetBlend { .. },
            ..
        }
    )));
    assert!(nodes.iter().any(|n| matches!(
        n.params,
        MaterialNodeParams::HorizonAo {
            directions: 8,
            steps: 8,
            ..
        }
    )));
    assert!(max.topological_material_nodes().iter().any(|n| matches!(
        n.params,
        MaterialNodeParams::HorizonAo {
            directions: 32,
            steps: 32,
            ..
        }
    )));
    assert_ne!(
        fixture_pixels(32, 1)
            .descriptor(Extent::new(32, 32).unwrap())
            .unwrap()
            .content_hash,
        pixels
            .descriptor(Extent::new(32, 32).unwrap())
            .unwrap()
            .content_hash
    );
}

#[test]
fn diagnostic_arguments_reject_unknown_profiles_and_preserve_native_report_paths() {
    use texelith_ui_egui::export_perf::Config;
    let parsed = Config::parse(
        [
            "--export-perf",
            "--profile",
            "unreal-v1",
            "--quality",
            "maximum",
            "--report",
            "target/evidence/T-096/ölçüm.json",
        ]
        .map(Into::into),
    )
    .unwrap();
    assert_eq!(parsed.profile, "unreal-v1");
    assert!(parsed.maximum);
    assert_eq!(
        parsed.report,
        std::path::PathBuf::from("target/evidence/T-096/ölçüm.json")
    );
    assert!(Config::parse(["--profile", "unknown"].map(Into::into)).is_err());
    assert!(Config::parse(["--quality", "proxy"].map(Into::into)).is_err());
    assert!(Config::parse(["--size", "4097"].map(Into::into)).is_err());
}

#[test]
fn real_export_measures_publication_after_warmup_and_cache_invalidation() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let config = texelith_ui_egui::export_perf::Config {
        size: 32,
        report: std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("evidence/T-096/test-{stamp}.json")),
        ..Default::default()
    };
    let report = texelith_ui_egui::export_perf::exercise(&config).unwrap();
    assert_eq!(report["cache_invalidated"], true);
    assert!(report["compute_submissions"].as_u64().unwrap() > 0);
    assert!(report["cache_misses"].as_u64().unwrap() > 0);
    assert_eq!(report["metric"]["outputs"].as_array().unwrap().len(), 7);
    assert!(report["metric"]["request_to_disk_ns"].as_u64().unwrap() > 0);
    assert_eq!(report["files"].as_array().unwrap().len(), 7);
    // A reduced fixture is functional evidence only, never a 4K performance pass.
    assert_eq!(report["passed"], false);
    // Same immutable destination must fail; no cached-only/no-overwrite substitution.
    assert!(texelith_ui_egui::export_perf::exercise(&config).is_err());
}
