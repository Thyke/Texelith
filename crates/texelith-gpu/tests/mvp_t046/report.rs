use super::harness_contract::{aggregate_timings, select_suite, validate_pyramid_report};
use serde_json::{Value, json};

#[test]
fn pyramid_suite_is_in_maps_and_all_and_empty_or_unknown_suites_fail() {
    for suite in ["pyramid", "maps", "all"] {
        assert!(
            select_suite(suite).unwrap().contains(&"pyramid"),
            "missing pyramid in {suite}"
        );
    }
    for suite in ["", " ", "pyramids", "unknown"] {
        assert!(select_suite(suite).is_err());
    }
}

fn valid_report() -> Value {
    json!({
        "passed": true, "suite": "pyramid", "size": 4096,
        "adapter": {"name": "RTX 5070 Ti", "backend": "Dx12"},
        "seed": 42, "manifest": {"filter": "binomial5", "filter_revision": 1, "boundary": "repeat", "band_weight": 1.0},
        "manifest_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks": [
            {"name": "source_transfer_alpha", "passed": true},
            {"name": "signed_source_hashes", "passed": true},
            {"name": "constant", "passed": true}, {"name": "ramp", "passed": true},
            {"name": "impulse", "passed": true}, {"name": "npot_257x193", "passed": true},
            {"name": "singleton_axes", "passed": true}, {"name": "repeat_clamp", "passed": true},
            {"name": "manual_bilinear", "passed": true}, {"name": "reconstruction", "passed": true},
            {"name": "determinism", "passed": true}, {"name": "cancellation", "passed": true},
            {"name": "low_budget", "passed": true}, {"name": "shared_cache", "passed": true}
        ],
        "timing": {"kind": "completion_upper_bound", "warmups": 10, "measurements": 100,
            "p95_ms": 2.0, "submission_ms": 0.2, "peak_bytes": 201326592,
            "cancellation_errors": 0, "scope": "full_node"},
        "maximum_quality": {"kind": "completion_upper_bound", "warmups": 10, "measurements": 100,
            "p95_ms": 24.0, "submission_ms": 0.3, "levels": 13, "scope": "full_node"},
        "pyramid": {"levels": 13, "budget_bytes": 1073741824u64,
            "scratch_peak_bytes": 201326592, "peak_in_flight": 1,
            "cancellation_requests": 1, "cancelled_publications": 0, "validation_errors": 0,
            "low_budget_rejections": 1, "shared_cache_hits": 1}
    })
}

#[test]
fn report_requires_oracles_full_node_timings_max_quality_and_actual_cancel_budget_checks() {
    let valid = valid_report();
    assert!(validate_pyramid_report(&valid).is_ok());
    for (path, value) in [
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/scope", json!("one_pass")),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/cancellation_errors", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/measurements", json!(0)),
        ("/maximum_quality/warmups", json!(9)),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/submission_ms", Value::Null),
        ("/maximum_quality/levels", json!(12)),
        ("/maximum_quality/scope", json!("one_pass")),
        ("/pyramid/levels", json!(12)),
        ("/pyramid/budget_bytes", json!(2147483648u64)),
        ("/pyramid/budget_bytes", json!(0)),
        ("/pyramid/scratch_peak_bytes", json!(201326593)),
        ("/pyramid/scratch_peak_bytes", json!(0)),
        ("/pyramid/peak_in_flight", json!(2)),
        ("/pyramid/cancellation_requests", json!(0)),
        ("/pyramid/cancelled_publications", json!(1)),
        ("/pyramid/validation_errors", json!(1)),
        ("/pyramid/low_budget_rejections", json!(0)),
        ("/pyramid/shared_cache_hits", json!(0)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        assert!(
            validate_pyramid_report(&invalid).is_err(),
            "accepted invalid report at {path}"
        );
    }
    let count = valid["checks"].as_array().unwrap().len();
    for index in 0..count {
        let mut incomplete = valid.clone();
        incomplete["checks"].as_array_mut().unwrap().remove(index);
        assert!(
            validate_pyramid_report(&incomplete).is_err(),
            "missing check {index} accepted"
        );
    }
    for field in ["pyramid", "timing", "maximum_quality", "manifest_hash"] {
        let mut missing = valid.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(validate_pyramid_report(&missing).is_err());
    }
}

#[test]
fn full_node_aggregation_sums_all_passes_after_exact_warmup_count() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![4.0, 4.0, 4.0, 4.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0, "a four-pass node cannot report 4 ms");
    let mut report = valid_report();
    report["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(
        validate_pyramid_report(&report).is_err(),
        "16 ms must fail the strict default gate"
    );
}
