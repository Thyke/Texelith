use super::harness_contract::{aggregate_timings, select_suite};
#[path = "../../examples/mvp_support/t057/report.rs"]
mod evidence;
use serde_json::{Value, json};

#[test]
fn seamless_offset_suite_is_registered_in_maps_and_all_and_unknown_names_fail() {
    for suite in ["seamless_offset", "maps", "all"] {
        assert!(
            select_suite(suite).unwrap().contains(&"seamless_offset"),
            "missing aggregate membership: {suite}"
        );
    }
    for suite in ["", " ", "seamless", "seamless-offset", "unknown"] {
        assert!(select_suite(suite).is_err());
    }
}

// Synthetic schema fixture only. Never written as hardware/performance evidence.
fn report() -> Value {
    let manifest = json!({"overlap":0.25,"boundary":"repeat","epsilon":1e-8,"window":"cubic_smoothstep","algorithm_revision":1});
    let checks: Vec<_> = evidence::REQUIRED_CHECKS
        .iter()
        .map(|n| json!({"name":n,"passed":true}))
        .collect();
    let mut r = json!({
        "passed":true,"suite":"seamless_offset","size":4096,"seed":57,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "manifest_hash":evidence::manifest_hash(&manifest).unwrap(),"manifest":manifest,"checks":checks,
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,"p95_ms":2.0,
            "submission_ms":0.1,"peak_bytes":536870912,"scope":"full_node","includes_source":true,
            "cache_hits_used_for_timing":0,"cancellation_errors":0},
        "resources":{"budget_bytes":1073741824u64,"peak_in_flight":1,"peak_pending":1,
            "cancellation_requests":1,"low_budget_rejections":1,"cancelled_publications":0,"validation_errors":0},
        "numerical":{"constant_max_error":1e-6,"period_max_error":1e-6,"scalar_max_error":1e-5,
            "color_max_error":0.0004,"minimum_coverage":0.25},
        "seams":[
            {"fixture":"discontinuous_noise_256x256","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03},
            {"fixture":"discontinuous_noise_257x193","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03}]
    });
    r["maximum_quality"] = r["timing"].clone();
    r["maximum_quality"]["overlap"] = json!(0.5);
    r["maximum_quality"]["p95_ms"] = json!(24.0);
    r
}

#[test]
fn report_rejects_missing_oracles_fake_cancellation_partial_timing_bad_hash_and_seam_regressions() {
    let valid = report();
    assert!(evidence::validate(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/suite", json!("height")),
        ("/size", json!(0)),
        ("/seed", json!(58)),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/manifest_hash", json!("0".repeat(64))),
        ("/manifest/overlap", json!(0.5)),
        ("/manifest/boundary", json!("clamp")),
        ("/manifest/epsilon", json!(1e-4)),
        ("/manifest/algorithm_revision", json!(2)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/submission_ms", json!(-1)),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/scope", json!("blend_only")),
        ("/timing/includes_source", json!(false)),
        ("/timing/cache_hits_used_for_timing", json!(1)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/overlap", json!(0.25)),
        ("/maximum_quality/p95_ms", json!(-1)),
        ("/maximum_quality/measurements", json!(0)),
        ("/maximum_quality/scope", json!("blend_only")),
        ("/maximum_quality/includes_source", json!(false)),
        ("/resources/budget_bytes", json!(0)),
        ("/resources/budget_bytes", json!(1073741825u64)),
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
        ("/numerical/minimum_coverage", json!(0.0)),
        ("/seams", json!([])),
        ("/seams/0/source_boundary_rms", json!(0)),
        ("/seams/0/result_boundary_rms", json!(0.051)),
        ("/seams/1/result_interior_rms", json!(0.019)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            evidence::validate(&bad).is_err(),
            "accepted invalid field: {path}"
        );
    }
    for i in 0..valid["checks"].as_array().unwrap().len() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(i);
        assert!(
            evidence::validate(&bad).is_err(),
            "accepted missing check {i}"
        );
    }
    for key in [
        "maximum_quality",
        "timing",
        "numerical",
        "resources",
        "manifest_hash",
        "seams",
    ] {
        let mut bad = valid.clone();
        bad.as_object_mut().unwrap().remove(key);
        assert!(
            evidence::validate(&bad).is_err(),
            "accepted missing field {key}"
        );
    }
}

#[test]
fn report_sums_every_pass_excludes_warmup_and_reports_maximum_quality_independently() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![8.0, 8.0]; 100]);
    let total = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(total.measurements, 100);
    assert_eq!(total.p95_ms, 16.0);
    let mut r = report();
    r["timing"]["p95_ms"] = json!(total.p95_ms);
    assert!(evidence::validate(&r).is_err());
    assert!(
        evidence::validate(&report()).is_ok(),
        "24 ms maximum quality is reported, not silently folded into default timing"
    );
}
