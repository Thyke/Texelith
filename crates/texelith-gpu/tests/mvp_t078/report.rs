use super::{harness_contract, report_contract as contract};
use serde_json::{Value, json};

// Synthetic schema fixture only; never saved as hardware evidence.
fn report() -> Value {
    let manifest = json!({"algorithm_revision":1,"default_output":"orm_rgb8","default_size":"keep",
        "default_transfer":"linear","default_dither":"none","source_quality":"full"});
    let mut report = json!({"passed":true,"suite":"packing","size":4096,"seed":78,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "manifest_hash":contract::manifest_hash(&manifest).unwrap(),"manifest":manifest,
        "checks":contract::REQUIRED_CHECKS.iter().map(|name| json!({"name":name,"passed":true})).collect::<Vec<_>>(),
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,"p95_ms":4.0,
            "scope":"full_node","includes_source":true,"cache_hits_used_for_timing":0,
            "submission_ms":0.2,"peak_bytes":536870912,"cancellation_errors":0},
        "resources":{"budget_bytes":1073741824u64,"peak_in_flight":1,"peak_pending":1,
            "cancellation_requests":1,"cancelled_pass_batches":1,"low_budget_rejections":1,
            "fence_retirements":1,"cancelled_publications":0,"validation_errors":0},
        "numerical":{"scalar_max_error":0.00001,"normal_component_max_error":0.001,
            "normal_length_max_error":0.001,"integer_max_lsb_error":1,"displacement_max_error":0.00001}});
    report["maximum_quality"] = report["timing"].clone();
    report["maximum_quality"]["configuration"] = json!("rgba16_ordered4x4_resize");
    report["maximum_quality"]["p95_ms"] = json!(24.0);
    report
}

#[test]
fn packing_suite_is_registered_in_maps_and_all_and_unknown_names_fail() {
    assert_eq!(
        harness_contract::select_suite("packing").unwrap(),
        ["packing"]
    );
    for name in ["maps", "all"] {
        assert!(
            harness_contract::select_suite(name)
                .unwrap()
                .contains(&"packing")
        );
    }
    for name in ["", " ", "pack", "Packing", "unknown"] {
        assert!(harness_contract::select_suite(name).is_err());
    }
}

#[test]
fn validator_rejects_missing_or_duplicate_behavioral_checks() {
    let valid = report();
    contract::validate(&valid).unwrap();
    for i in 0..contract::REQUIRED_CHECKS.len() {
        let mut missing = valid.clone();
        missing["checks"].as_array_mut().unwrap().remove(i);
        assert!(contract::validate(&missing).is_err());
        let mut duplicate = valid.clone();
        let check = duplicate["checks"][i].clone();
        duplicate["checks"].as_array_mut().unwrap().push(check);
        assert!(contract::validate(&duplicate).is_err());
    }
}

#[test]
fn validator_rejects_fake_timings_cancellation_budget_precision_and_default_drift() {
    let valid = report();
    contract::validate(&valid).unwrap();
    for (path, value) in [
        ("/passed", json!(false)),
        ("/suite", json!("normal")),
        ("/size", json!(0)),
        ("/seed", json!(0)),
        ("/adapter/name", json!("")),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/manifest_hash", json!("0".repeat(64))),
        ("/manifest/source_quality", json!("proxy")),
        ("/manifest/default_transfer", json!("srgb")),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/submission_ms", json!(-1)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/scope", json!("last_pass_only")),
        ("/timing/includes_source", json!(false)),
        ("/timing/cache_hits_used_for_timing", json!(1)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/measurements", json!(0)),
        ("/maximum_quality/configuration", json!("default")),
        ("/maximum_quality/p95_ms", json!(-1)),
        ("/maximum_quality/peak_bytes", json!(1073741825u64)),
        ("/resources/budget_bytes", json!(0)),
        ("/resources/budget_bytes", json!(1024)),
        ("/resources/peak_in_flight", json!(2)),
        ("/resources/peak_pending", json!(2)),
        ("/resources/cancellation_requests", json!(0)),
        ("/resources/cancelled_pass_batches", json!(0)),
        ("/resources/low_budget_rejections", json!(0)),
        ("/resources/fence_retirements", json!(0)),
        ("/resources/cancelled_publications", json!(1)),
        ("/resources/validation_errors", json!(1)),
        ("/numerical/scalar_max_error", json!(2.1e-5)),
        ("/numerical/normal_component_max_error", json!(0.0021)),
        ("/numerical/normal_length_max_error", json!(0.0021)),
        ("/numerical/integer_max_lsb_error", json!(2)),
        ("/numerical/displacement_max_error", json!(2.1e-5)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(contract::validate(&bad).is_err(), "accepted {path}");
    }
    for key in [
        "checks",
        "manifest",
        "manifest_hash",
        "timing",
        "maximum_quality",
        "resources",
        "numerical",
    ] {
        let mut bad = valid.clone();
        bad.as_object_mut().unwrap().remove(key);
        assert!(contract::validate(&bad).is_err(), "accepted missing {key}");
    }
}

#[test]
fn validator_gates_sum_of_all_passes_and_keeps_maximum_quality_separate() {
    let mut samples = vec![vec![1000.0]; 10];
    samples.extend(vec![vec![8.0, 8.0]; 100]);
    let timing = harness_contract::aggregate_timings(&samples, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(timing.p95_ms, 16.0);
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(timing.p95_ms);
    assert!(contract::validate(&bad).is_err());
    contract::validate(&report()).unwrap();
}

#[test]
fn shared_validator_invokes_the_registered_packing_validator() {
    let valid = report();
    harness_contract::validate_report(&valid).unwrap();
    let mut bad = valid;
    bad["numerical"]["normal_length_max_error"] = json!(0.1);
    assert!(harness_contract::validate_report(&bad).is_err());
}

#[test]
fn validator_checks_default_parameters_even_when_manifest_digest_is_recomputed() {
    for (key, value) in [
        ("algorithm_revision", json!(2)),
        ("default_output", json!("albedo_rgb8")),
        ("default_size", json!("resize")),
        ("default_transfer", json!("srgb")),
        ("default_dither", json!("ordered4x4")),
        ("source_quality", json!("proxy")),
    ] {
        let mut bad = report();
        bad["manifest"][key] = value;
        bad["manifest_hash"] = json!(contract::manifest_hash(&bad["manifest"]).unwrap());
        assert!(
            contract::validate(&bad).is_err(),
            "accepted changed default {key}"
        );
    }
}
