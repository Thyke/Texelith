use super::harness_contract::{aggregate_timings, select_suite, validate_report};
use serde_json::{Value, json};

const CHECKS: &[&str] = &[
    "cpu_oracle",
    "stroke_order",
    "opacity_hardness_endpoints",
    "wrapped_edge",
    "clamp_edge",
    "undo_replay",
    "proxy_full_alignment",
    "stable_digest",
    "npot",
    "singleton_axes",
    "cancellation",
    "low_budget",
    "bounded_batches",
];

// Synthetic schema fixture, never persisted as hardware evidence. Implementer
// registers the node-local validator through ADR 0012's nested validator entry.
fn report() -> Value {
    json!({
        "passed": true, "suite": "paint", "size": 4096, "seed": 56,
        "adapter": {"name": "RTX 5070 Ti", "backend": "Dx12"},
        "manifest": {"boundary": "Repeat", "radius_uv": 0.02, "hardness": 0.5,
            "opacity": 0.75, "mode": "Add", "stroke_digest": "56".repeat(32),
            "stroke_count": 20, "point_count": 1024, "algorithm_revision": 1},
        "manifest_hash": "0123456789abcdef".repeat(4),
        "checks": CHECKS.iter().map(|name| json!({"name": name, "passed": true})).collect::<Vec<_>>(),
        "timing": {"kind": "completion_upper_bound", "scope": "full_node", "warmups": 10,
            "measurements": 100, "p95_ms": 2.0, "submission_ms": 0.1, "peak_bytes": 268435456,
            "cancellation_errors": 0, "cache_hits_used_for_timing": 0},
        "maximum_quality": {"kind": "completion_upper_bound", "scope": "full_node",
            "warmups": 10, "measurements": 100, "p95_ms": 24.0, "submission_ms": 0.2,
            "point_count": 100000},
        "paint": {"budget_bytes": 1073741824u64, "peak_in_flight": 1, "peak_pending": 1,
            "cancellation_requests": 1, "cancelled_pass_batches": 1,
            "cancelled_publications": 0, "validation_errors": 0, "low_budget_rejections": 1,
            "max_segments_per_batch": 256, "peak_uploaded_segments": 256,
            "checkpoint_hits": 1, "checkpoint_bytes": 67108864,
            "replay_max_error": 1e-6, "scalar_max_error": 1e-6}
    })
}

#[test]
fn paint_suite_is_registered_once_in_maps_and_all_and_unknown_or_empty_names_fail() {
    for name in ["paint", "maps", "all"] {
        assert_eq!(
            select_suite(name)
                .expect("register the paint suite")
                .iter()
                .filter(|&&suite| suite == "paint")
                .count(),
            1
        );
    }
    for name in ["", " ", "paint_mask", "paints", "unknown"] {
        assert!(select_suite(name).is_err());
    }
}

#[test]
fn full_node_timing_sums_batches_excludes_ten_warmups_and_uses_strict_p95_limit() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![5.0, 6.0, 5.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0);
    assert!(aggregate_timings(&rows[..10], 10).is_err());
    assert!(aggregate_timings(&[vec![f64::NAN]], 0).is_err());
    assert!(aggregate_timings(&[vec![]], 0).is_err());
}

#[test]
fn paint_reports_require_numerical_coverage_real_cancellation_budget_and_separate_maximum_timing() {
    let valid = report();
    assert!(
        validate_report(&valid).is_ok(),
        "register a node-local paint report validator"
    );
    for (path, value) in [
        ("/passed", json!(false)),
        ("/checks", json!([])),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", json!(-1.0)),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/scope", json!("one_batch")),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/cache_hits_used_for_timing", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/warmups", json!(0)),
        ("/maximum_quality/measurements", json!(1)),
        ("/maximum_quality/scope", json!("one_batch")),
        ("/paint/budget_bytes", json!(1073741825u64)),
        ("/paint/peak_in_flight", json!(2)),
        ("/paint/peak_pending", json!(2)),
        ("/paint/cancellation_requests", json!(0)),
        ("/paint/cancelled_pass_batches", json!(0)),
        ("/paint/cancelled_publications", json!(1)),
        ("/paint/validation_errors", json!(1)),
        ("/paint/low_budget_rejections", json!(0)),
        ("/paint/max_segments_per_batch", json!(0)),
        ("/paint/peak_uploaded_segments", json!(100000)),
        ("/paint/checkpoint_hits", json!(0)),
        ("/paint/checkpoint_bytes", json!(1073741825u64)),
        ("/paint/replay_max_error", json!(0.01)),
        ("/paint/scalar_max_error", json!(0.01)),
        ("/manifest/stroke_digest", json!("bad")),
        ("/manifest/point_count", json!(100001)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_report(&bad).is_err(),
            "invalid paint report accepted at {path}"
        );
    }
    for required in CHECKS {
        let mut missing = valid.clone();
        missing["checks"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["name"] != *required);
        assert!(
            validate_report(&missing).is_err(),
            "missing {required} was accepted"
        );
    }
}
