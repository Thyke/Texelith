use super::harness_contract::{aggregate_timings, select_suite, validate_height_report};
use serde_json::{Value, json};

#[test]
fn height_suite_is_registered_in_maps_and_all_and_typos_fail() {
    for suite in ["height", "maps", "all"] {
        assert!(select_suite(suite).unwrap().contains(&"height"));
    }
    for suite in ["", " ", "heights", "unknown"] {
        assert!(select_suite(suite).is_err());
    }
}

// Schema fixture only. These values are never written as measurement evidence.
fn report() -> Value {
    json!({
        "passed":true,"suite":"height","size":4096,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "seed":42,"manifest":{"band_weights":[1.0],"coarse_weight":1.0,
            "remove_coarse":false,"boundary":"repeat","remap":"none","filter_revision":1},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":[
            {"name":"unit_reconstruction","passed":true},{"name":"band_isolation","passed":true},
            {"name":"coarse_removal","passed":true},{"name":"explicit_levels","passed":true},
            {"name":"flat_min_max","passed":true},{"name":"min_max_remap","passed":true},
            {"name":"proxy_scaling","passed":true},{"name":"srgb_linear_ramp","passed":true},
            {"name":"cache_reuse","passed":true},{"name":"constant","passed":true},
            {"name":"ramp","passed":true},{"name":"impulse","passed":true},
            {"name":"npot_257x193","passed":true},{"name":"singleton_axes","passed":true},
            {"name":"repeat_clamp","passed":true},{"name":"determinism","passed":true},
            {"name":"cancellation","passed":true},{"name":"low_budget","passed":true}
        ],
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":2.0,"submission_ms":0.2,"peak_bytes":536870912,
            "cancellation_errors":0,"scope":"full_node","includes_pyramid":true},
        "maximum_quality":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":24.0,"submission_ms":0.3,"levels":13,"scope":"full_node","includes_pyramid":true},
        "height":{"levels":13,"budget_bytes":1073741824u64,"scratch_peak_bytes":201326592,
            "peak_in_flight":1,"cancellation_requests":1,"cancelled_publications":0,
            "validation_errors":0,"low_budget_rejections":1,"shared_pyramid_cache_hits":1,"output_cache_hits":1}
    })
}

#[test]
fn report_rejects_incomplete_oracles_unmeasured_cancellation_and_partial_node_timing() {
    let valid = report();
    assert!(validate_height_report(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/suite", json!("pyramid")),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/scope", json!("reconstruction_only")),
        ("/timing/includes_pyramid", json!(false)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/warmups", json!(9)),
        ("/maximum_quality/measurements", json!(99)),
        ("/maximum_quality/levels", json!(12)),
        ("/maximum_quality/p95_ms", json!(-1.0)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/submission_ms", Value::Null),
        ("/maximum_quality/kind", json!("cpu_encode_only")),
        ("/maximum_quality/scope", json!("reconstruction_only")),
        ("/maximum_quality/includes_pyramid", json!(false)),
        ("/height/levels", json!(12)),
        ("/height/budget_bytes", json!(0)),
        ("/height/budget_bytes", json!(1073741825u64)),
        ("/height/scratch_peak_bytes", json!(0)),
        ("/height/scratch_peak_bytes", json!(201326593)),
        ("/height/peak_in_flight", json!(2)),
        ("/height/cancellation_requests", json!(0)),
        ("/height/cancelled_publications", json!(1)),
        ("/height/validation_errors", json!(1)),
        ("/height/low_budget_rejections", json!(0)),
        ("/height/shared_pyramid_cache_hits", json!(0)),
        ("/height/output_cache_hits", json!(0)),
        ("/manifest_hash", json!("not-a-hash")),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_height_report(&bad).is_err(),
            "accepted invalid report at {path}"
        );
    }
    for index in 0..valid["checks"].as_array().unwrap().len() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(index);
        assert!(
            validate_height_report(&bad).is_err(),
            "missing check {index} accepted"
        );
    }
    for field in ["height", "timing", "maximum_quality", "manifest_hash"] {
        let mut bad = valid.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(
            validate_height_report(&bad).is_err(),
            "missing field {field} accepted"
        );
    }
}

#[test]
fn default_p95_sums_all_pyramid_and_reconstruction_passes_max_quality_is_separate() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![4.0, 4.0, 4.0, 4.0]; 100]);
    let total = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(total.measurements, 100);
    assert_eq!(total.p95_ms, 16.0);
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(total.p95_ms);
    assert!(validate_height_report(&bad).is_err());
    assert!(
        validate_height_report(&report()).is_ok(),
        "24 ms maximum-quality result is reported separately"
    );
}
