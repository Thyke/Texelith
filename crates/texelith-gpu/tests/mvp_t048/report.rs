//! Schema fixtures are not GPU measurement evidence.
use super::harness_contract::{aggregate_timings, select_suite, validate_report};
use serde_json::{Value, json};

#[test]
fn normal_suite_is_explicit_in_maps_and_all_without_losing_legacy_baseline() {
    for suite in ["normal", "maps", "all"] {
        assert!(select_suite(suite).unwrap().contains(&"normal"));
    }
    assert!(select_suite("baseline").unwrap().contains(&"normal"));
    for suite in ["", " ", "normals", "unknown"] {
        assert!(select_suite(suite).is_err());
    }
}

fn report() -> Value {
    json!({
        "passed":true,"suite":"normal","size":4096,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},"seed":42,
        "manifest":{"filter":"sobel","scale_weights":[1.0],"strength":1.0,
            "boundary":"repeat","convention":"opengl","filter_revision":1},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[
            {"name":"legacy_central","passed":true},{"name":"sobel","passed":true},
            {"name":"scharr","passed":true},{"name":"multi_scale","passed":true},
            {"name":"gradient_before_normalize","passed":true},{"name":"uv_ramp_y_sign","passed":true},
            {"name":"unit_length","passed":true},{"name":"gl_dx_metadata","passed":true},
            {"name":"no_double_flip","passed":true},{"name":"proxy_scaling","passed":true},
            {"name":"linear_height_precision","passed":true},{"name":"constant","passed":true},
            {"name":"ramp","passed":true},{"name":"impulse","passed":true},
            {"name":"npot_257x193","passed":true},{"name":"singleton_axes","passed":true},
            {"name":"repeat_clamp","passed":true},{"name":"determinism","passed":true},
            {"name":"cancellation","passed":true},{"name":"low_budget","passed":true}
        ],
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":2.0,"submission_ms":0.2,"peak_bytes":536870912,
            "cancellation_errors":0,"scope":"full_node","includes_pyramid":true},
        "maximum_quality":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":24.0,"submission_ms":0.3,"scope":"full_node","includes_pyramid":true,
            "filter":"scharr","scales":16},
        "normal":{"budget_bytes":1073741824u64,"peak_in_flight":1,"peak_pending":1,
            "cancellation_requests":1,"cancelled_pass_batches":1,"cancelled_publications":0,
            "validation_errors":0,"low_budget_rejections":1}
    })
}

#[test]
fn normal_report_requires_oracles_full_node_max_quality_and_exercised_cancel_budget() {
    let valid = report();
    // Keep the existing public validator as the entry point used by the runner.
    assert!(validate_report(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/p95_ms", json!(-1.0)),
        ("/timing/submission_ms", Value::Null),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/scope", json!("last_normalize_pass")),
        ("/timing/includes_pyramid", json!(false)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/warmups", json!(9)),
        ("/maximum_quality/measurements", json!(99)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/submission_ms", Value::Null),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/scope", json!("last_pass")),
        ("/maximum_quality/includes_pyramid", json!(false)),
        ("/maximum_quality/scales", json!(1)),
        ("/maximum_quality/filter", json!("central")),
        ("/normal/budget_bytes", json!(0)),
        ("/normal/budget_bytes", json!(1073741825u64)),
        ("/normal/peak_in_flight", json!(2)),
        ("/normal/peak_pending", json!(2)),
        ("/normal/cancellation_requests", json!(0)),
        ("/normal/cancelled_pass_batches", json!(0)),
        ("/normal/cancelled_publications", json!(1)),
        ("/normal/validation_errors", json!(1)),
        ("/normal/low_budget_rejections", json!(0)),
        ("/manifest_hash", json!("not-a-hash")),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_report(&bad).is_err(),
            "accepted invalid normal report at {path}"
        );
    }
    for index in 0..valid["checks"].as_array().unwrap().len() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(index);
        assert!(
            validate_report(&bad).is_err(),
            "accepted missing oracle {index}"
        );
    }
    for field in ["normal", "timing", "maximum_quality", "manifest_hash"] {
        let mut bad = valid.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(validate_report(&bad).is_err(), "accepted missing {field}");
    }
}

#[test]
fn p95_sums_all_passes_after_warmup_and_maximum_quality_is_reported_separately() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![4.0, 4.0, 4.0, 4.0]; 100]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0);
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(validate_report(&bad).is_err());
    let mut valid = report();
    valid["timing"]["p95_ms"] = json!(15.999);
    assert!(
        validate_report(&valid).is_ok(),
        "maximum-quality 24ms is separate from the default gate"
    );
}
