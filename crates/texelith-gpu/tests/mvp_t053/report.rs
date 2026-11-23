use super::harness_contract::{aggregate_timings, select_suite, validate_roughness_report};
use serde_json::{Value, json};

#[test]
fn roughness_suite_is_registered_in_maps_and_all_and_unknown_empty_suites_fail() {
    for name in ["roughness", "maps", "all"] {
        let selected = select_suite(name).unwrap();
        assert_eq!(
            selected.iter().filter(|name| **name == "roughness").count(),
            1
        );
    }
    for name in ["", " ", "roughnes", "smoothness", "unknown"] {
        assert!(select_suite(name).is_err(), "unexpected suite {name}");
    }
}

// Schema fixture only: these synthetic times/counters are never hardware evidence.
fn report() -> Value {
    json!({
        "passed":true,"suite":"roughness","size":4096,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},"seed":53,
        "manifest":{"curve":[[0.0,0.0],[1.0,1.0]],"base":0.5,"invert":false,"mask":"constant_one"},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[
            {"name":"knot_interpolation","passed":true},{"name":"curve_endpoints","passed":true},
            {"name":"sixteen_knots","passed":true},{"name":"mask_zero_one_blend","passed":true},
            {"name":"optional_mask_one","passed":true},{"name":"invert_after_blend","passed":true},
            {"name":"invert_twice_identity","passed":true},{"name":"linear_rec709","passed":true},
            {"name":"alpha_ignored","passed":true},{"name":"no_srgb_encode","passed":true},
            {"name":"bounded_finite","passed":true},{"name":"constant","passed":true},
            {"name":"ramp","passed":true},{"name":"impulse","passed":true},
            {"name":"npot_257x193","passed":true},{"name":"singleton_axes","passed":true},
            {"name":"determinism","passed":true},{"name":"cache_invalidation","passed":true},
            {"name":"cancellation","passed":true},{"name":"low_budget","passed":true}
        ],
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":2.0,"submission_ms":0.2,"peak_bytes":536870912,
            "cancellation_errors":0,"scope":"full_node"},
        "maximum_quality":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":24.0,"submission_ms":0.3,"knot_count":16,"scope":"full_node"},
        "roughness":{"budget_bytes":1073741824u64,"peak_in_flight":1,"cancellation_requests":1,
            "cancelled_publications":0,"validation_errors":0,"low_budget_rejections":1}
    })
}

#[test]
fn reports_reject_missing_oracles_unmeasured_cancellation_bad_budget_and_partial_timing() {
    let valid = report();
    assert!(validate_roughness_report(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/suite", json!("height")),
        ("/size", json!(0)),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", json!(-1.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/submission_ms", Value::Null),
        ("/timing/scope", json!("curve_only")),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/scope", json!("curve_only")),
        ("/maximum_quality/knot_count", json!(2)),
        ("/maximum_quality/warmups", json!(9)),
        ("/maximum_quality/measurements", json!(99)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/submission_ms", Value::Null),
        ("/roughness/budget_bytes", json!(0)),
        ("/roughness/budget_bytes", json!(1073741825u64)),
        ("/roughness/peak_in_flight", json!(2)),
        ("/roughness/cancellation_requests", json!(0)),
        ("/roughness/cancelled_publications", json!(1)),
        ("/roughness/validation_errors", json!(1)),
        ("/roughness/low_budget_rejections", json!(0)),
        ("/manifest_hash", json!("bad")),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_roughness_report(&bad).is_err(),
            "invalid report accepted: {path}"
        );
    }
    for i in 0..valid["checks"].as_array().unwrap().len() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(i);
        assert!(
            validate_roughness_report(&bad).is_err(),
            "missing check {i}"
        );
    }
    for field in ["roughness", "timing", "maximum_quality", "manifest_hash"] {
        let mut bad = valid.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(
            validate_roughness_report(&bad).is_err(),
            "missing field {field}"
        );
    }
}

#[test]
fn full_node_p95_sums_passes_while_maximum_quality_is_reported_separately() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![8.0, 8.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0);
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(validate_roughness_report(&bad).is_err());
    assert!(
        validate_roughness_report(&report()).is_ok(),
        "24 ms maximum quality is separate from default gate"
    );
}
