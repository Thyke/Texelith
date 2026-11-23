//! Synthetic report fixtures test validation, never hardware performance evidence.
use super::harness_contract::{aggregate_timings, select_suite, validate_report};
use serde_json::{Value, json};

const CHECKS: &[&str] = &[
    "flat_zero",
    "bump_positive",
    "bowl_negative",
    "output_separation",
    "convention_equivalence",
    "steep_finite",
    "cpu_oracle",
    "npot_257x193",
    "singleton_axes",
    "repeat_clamp",
    "determinism",
    "cancellation",
    "low_budget",
];

fn report() -> Value {
    json!({
        "passed":true,"suite":"curvature","size":4096,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},"seed":52,
        "manifest":{"edge_scale":1.0,"cavity_scale":1.0,"boundary":"repeat"},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks": CHECKS.iter().map(|name| json!({"name":name,"passed":true})).collect::<Vec<_>>(),
        "timing":{"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":100,
            "p95_ms":2.0,"submission_ms":0.1,"peak_bytes":536870912,"cancellation_errors":0},
        "maximum_quality":{"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":100,
            "p95_ms":21.0,"submission_ms":0.1}
    })
}

#[test]
fn curvature_suite_is_registered_in_map_aggregates_and_unknown_suites_fail() {
    // Curvature is a map suite; the viewport aggregate is not applicable here.
    for name in ["curvature", "maps", "all"] {
        assert!(
            select_suite(name).unwrap().contains(&"curvature"),
            "missing curvature in {name}"
        );
    }
    for name in ["", " ", "curvatures", "unknown"] {
        assert!(select_suite(name).is_err());
    }
}

#[test]
fn reports_require_full_node_timing_oracles_and_separate_maximum_quality() {
    let valid = report();
    assert!(validate_report(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/manifest_hash", json!("not-a-hash")),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/p95_ms", json!(-1.0)),
        ("/timing/submission_ms", Value::Null),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825_u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/scope", json!("last_output_pass")),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/warmups", json!(9)),
        ("/maximum_quality/measurements", json!(99)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/submission_ms", Value::Null),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/scope", json!("last_output_pass")),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_report(&bad).is_err(),
            "accepted invalid curvature report at {path}"
        );
    }
    for (i, name) in CHECKS.iter().enumerate() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(i);
        assert!(
            validate_report(&bad).is_err(),
            "accepted missing curvature check {}",
            name
        );
    }
}

#[test]
fn p95_sums_every_pass_excludes_warmups_and_enforces_strict_default_limit() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![4.0, 6.0, 6.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0);
    assert!(aggregate_timings(&rows[..10], 10).is_err());
    let mut fixture = report();
    fixture["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(validate_report(&fixture).is_err());
    fixture["timing"]["p95_ms"] = json!(15.999);
    assert!(
        validate_report(&fixture).is_ok(),
        "maximum quality is measured separately from the default <16ms gate"
    );
}
