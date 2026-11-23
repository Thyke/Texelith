use super::harness_contract::{aggregate_timings, select_suite, validate_report};
use serde_json::{Value, json};

const CHECKS: &[&str] = &[
    "constant",
    "ramp",
    "impulse",
    "npot_257x193",
    "singleton_axes",
    "repeat_clamp",
    "scalar_oracle",
    "color_oracle",
    "linear_color",
    "height_precision",
    "same_seed_repeat",
    "different_seeds",
    "nonzero_coverage",
    "periodic_coordinates",
    "source_correlation",
    "bounded_neighbors",
    "seam_reduction",
    "structured_tradeoff",
    "disabled_lossless",
    "proxy_scaling",
    "cancellation",
    "low_budget",
];

#[test]
fn suite_is_registered_exactly_once_in_maps_and_all_and_unknown_or_empty_names_fail() {
    assert!(
        super::metadata_contract_registered(),
        "public splat transform contract must be activated by the node registry"
    );
    for suite in ["seamless_splat", "maps", "all"] {
        let names = select_suite(suite).expect("T-058 verification suite must be registered");
        assert_eq!(
            names.iter().filter(|&&n| n == "seamless_splat").count(),
            1,
            "missing or duplicate splat suite in {suite}"
        );
    }
    // ADR 0012 keeps `viewport` as an alias that covers every resource suite;
    // viewport-specific suites (e.g. T-063 viewport_resources) may extend it.
    let viewport = select_suite("viewport").unwrap();
    for name in select_suite("resources").unwrap() {
        assert!(
            viewport.contains(&name),
            "viewport alias must include {name}"
        );
    }
    for name in ["", " ", "splat", "seamless-splat", "unknown"] {
        assert!(select_suite(name).is_err());
    }
}

// Schema fixture only: never write this synthetic timing as measured evidence.
fn report() -> Value {
    let checks: Vec<_> = CHECKS
        .iter()
        .map(|name| json!({"name":name,"passed":true}))
        .collect();
    let mut r = json!({
        "passed":true,"suite":"seamless_splat","size":4096,"seed":58,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "manifest_hash":"a".repeat(64),
        "manifest":{"seed":58,"grid":4,"overlap":0.25,"boundary":"repeat","epsilon":1e-8,"algorithm_revision":1,"hash":"wrapped_u32_avalanche_v1","window":"separable_cubic_compact_pow4","max_neighbors":9},
        "checks":checks,
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,"p95_ms":2.0,"submission_ms":0.1,"peak_bytes":536870912u64,"scope":"full_node","includes_source":true,"cache_hits_used_for_timing":0,"cancellation_errors":0},
        "resources":{"budget_bytes":1073741824u64,"peak_in_flight":1,"peak_pending":1,"cancellation_requests":1,"low_budget_rejections":1,"cancelled_publications":0,"validation_errors":0},
        "numerical":{"constant_max_error":1e-6,"period_max_error":1e-6,"scalar_max_error":1e-5,"color_max_error":0.0004,"minimum_coverage":0.25,"maximum_neighbors":9,"same_seed_identical":true,"different_seed_rms":0.03,"correlation_max_error":0.0004},
        "seams":[
            {"fixture":"discontinuous_noise_256x256","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03},
            {"fixture":"discontinuous_noise_257x193","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03}]
    });
    // Existing harness boundary: padded canonical JSON as typed source bytes.
    let mut bytes = serde_json::to_vec(&r["manifest"]).unwrap();
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let hash = texelith_core::SourceDescriptor::from_rgba8(
        &bytes,
        texelith_core::Extent::new((bytes.len() / 4) as u32, 1).unwrap(),
        texelith_core::InputInterpretation::LinearHeight,
    )
    .unwrap()
    .content_hash;
    r["manifest_hash"] = json!(hash.iter().map(|v| format!("{v:02x}")).collect::<String>());
    r["maximum_quality"] = r["timing"].clone();
    r["maximum_quality"]["grid"] = json!(8);
    r["maximum_quality"]["overlap"] = json!(0.5);
    r["maximum_quality"]["p95_ms"] = json!(24.0);
    r
}

#[test]
fn report_validator_rejects_missing_oracles_unbounded_work_holes_and_fake_performance() {
    let valid = report();
    assert!(
        validate_report(&valid).is_ok(),
        "splat report schema/registration not implemented: {:?}",
        validate_report(&valid)
    );
    for (path, value) in [
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/manifest_hash", json!("0".repeat(64))),
        ("/manifest/seed", json!(59)),
        ("/manifest/grid", json!(8)),
        ("/manifest/overlap", json!(0.5)),
        ("/manifest/epsilon", json!(1e-4)),
        ("/manifest/boundary", json!("clamp")),
        ("/manifest/max_neighbors", json!(64)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/scope", json!("blend_only")),
        ("/timing/includes_source", json!(false)),
        ("/timing/cache_hits_used_for_timing", json!(1)),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/grid", json!(4)),
        ("/maximum_quality/overlap", json!(0.25)),
        ("/maximum_quality/measurements", json!(0)),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/scope", json!("blend_only")),
        ("/resources/budget_bytes", json!(0)),
        ("/resources/budget_bytes", json!(1024)),
        ("/resources/peak_in_flight", json!(2)),
        ("/resources/peak_pending", json!(2)),
        ("/resources/cancellation_requests", json!(0)),
        ("/resources/low_budget_rejections", json!(0)),
        ("/resources/cancelled_publications", json!(1)),
        ("/resources/validation_errors", json!(1)),
        ("/numerical/constant_max_error", json!(2.1e-5)),
        ("/numerical/period_max_error", json!(2.1e-5)),
        ("/numerical/scalar_max_error", json!(2.1e-5)),
        ("/numerical/color_max_error", json!(0.0006)),
        ("/numerical/correlation_max_error", json!(0.0006)),
        ("/numerical/minimum_coverage", json!(0)),
        ("/numerical/maximum_neighbors", json!(64)),
        ("/numerical/same_seed_identical", json!(false)),
        ("/numerical/different_seed_rms", json!(0)),
        ("/seams", json!([])),
        ("/seams/0/result_boundary_rms", json!(0.051)),
        ("/seams/1/result_interior_rms", json!(0.019)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_report(&bad).is_err(),
            "accepted invalid splat evidence: {path}"
        );
    }
    for (i, name) in CHECKS.iter().enumerate() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(i);
        assert!(
            validate_report(&bad).is_err(),
            "accepted missing check {}",
            name
        );
    }
}

#[test]
fn full_node_timing_sums_passes_and_keeps_maximum_quality_separate() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![8.0, 8.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!((timing.measurements, timing.p95_ms), (100, 16.0));
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(validate_report(&bad).is_err());
    assert!(
        validate_report(&report()).is_ok(),
        "maximum-quality p95 is reported separately, without the default gate"
    );
}
