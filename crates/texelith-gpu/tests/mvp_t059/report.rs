use super::{harness_contract as shared, report_contract as c};
use serde_json::{Value, json};

#[test]
fn histogram_suite_is_registered_once_in_maps_all_and_keeps_unknown_empty_errors() {
    for name in ["seamless_histogram", "maps", "all"] {
        let names =
            shared::select_suite(name).expect("T-059 suite must be registered via ADR 0012");
        assert_eq!(
            names.iter().filter(|&&n| n == "seamless_histogram").count(),
            1
        );
    }
    let viewport = shared::select_suite("viewport").unwrap();
    for name in shared::select_suite("resources").unwrap() {
        assert!(
            viewport.contains(&name),
            "viewport alias must include {name}"
        );
    }
    for name in ["", " ", "histogram", "seamless-histogram", "unknown"] {
        assert!(shared::select_suite(name).is_err());
    }
}

// In-memory schema fixture, NEVER written as measured evidence.
fn report() -> Value {
    let manifest = c::default_manifest();
    let mut timing = json!({"kind":"completion_upper_bound","scope":"full_node","warmups":10,"measurements":100,
        "p95_ms":3.0,"submission_ms":0.1,"peak_bytes":536870912u64,"cancellation_errors":0,
        "output_cache_hits_used_for_timing":0,"source_lut_cache_hits_used_for_timing":0,"cpu_pixel_conversions":0});
    for pass in c::PASSES {
        timing[format!("includes_{pass}")] = json!(true);
    }
    let mut maximum = timing.clone();
    maximum["base"] = json!("splat");
    maximum["grid"] = json!(8);
    maximum["overlap"] = json!(0.5);
    maximum["strength"] = json!(1.0);
    maximum["p95_ms"] = json!(25.0);
    json!({"passed":true,"suite":"seamless_histogram","size":4096,"seed":59,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},"manifest_hash":c::manifest_hash(&manifest).unwrap(),"manifest":manifest,
        "checks":c::REQUIRED_CHECKS.iter().map(|name|json!({"name":name,"passed":true})).collect::<Vec<_>>(),
        "timing":timing,"maximum_quality":maximum,
        "resources":{"budget_bytes":1073741824u64,"scratch_peak_bytes":67108864u64,"peak_in_flight":1,"peak_pending":1,
            "cancellation_requests":1,"low_budget_rejections":1,"source_lut_cache_hits":1,"source_lut_invalidations":1,"cancelled_publications":0,"validation_errors":0},
        "numerical":{"constant_max_error":1e-6,"period_max_error":1e-6,"scalar_max_error":1e-5,"color_max_error":0.0004,
            "height_precision_max_error":1e-7,"same_backend_identical":true,"histogram_count_mismatch":0,"nonfinite_pixels":0,
            "max_luminance_ratio":4.0,"clipped_pixel_fraction":0.01,"color_mean_absolute_change":0.04},
        "cdf":[{"base":"offset","fixture":"value_noise_seed59_1024","size":1024,"bins":1024,"max_distance":0.02},
            {"base":"splat","fixture":"value_noise_seed59_1024","size":1024,"bins":1024,"max_distance":0.03}],
        "seams":[{"base":"offset","fixture":"discontinuous_noise_257x193","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03},
            {"base":"splat","fixture":"discontinuous_noise_257x193","source_boundary_rms":0.1,"source_interior_rms":0.04,"result_boundary_rms":0.03,"result_interior_rms":0.03}]})
}

#[test]
fn local_evidence_validator_rejects_missing_work_relaxed_thresholds_and_fabricated_coverage() {
    let valid = report();
    c::validate(&valid).expect("local schema fixture");
    for (path, value) in [
        ("/manifest/bins", json!(256)),
        ("/manifest/variance_floor", json!(0.001)),
        ("/manifest/epsilon", json!(1e-4)),
        ("/manifest_hash", json!("0".repeat(64))),
        ("/checks/0/passed", json!(false)),
        ("/size", json!(1024)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/scope", json!("blend_only")),
        ("/timing/measurements", json!(99)),
        ("/timing/warmups", json!(0)),
        ("/timing/p95_ms", json!(-1.0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/timing/output_cache_hits_used_for_timing", json!(1)),
        ("/timing/source_lut_cache_hits_used_for_timing", json!(1)),
        ("/timing/cpu_pixel_conversions", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/grid", json!(4)),
        ("/maximum_quality/overlap", json!(0.25)),
        ("/maximum_quality/base", json!("offset")),
        ("/maximum_quality/scope", json!("blend_only")),
        ("/maximum_quality/p95_ms", json!(-1)),
        ("/resources/budget_bytes", json!(1024)),
        ("/resources/scratch_peak_bytes", json!(0)),
        ("/resources/scratch_peak_bytes", json!(201326593u64)),
        ("/resources/peak_in_flight", json!(2)),
        ("/resources/peak_pending", json!(2)),
        ("/resources/cancellation_requests", json!(0)),
        ("/resources/low_budget_rejections", json!(0)),
        ("/resources/source_lut_cache_hits", json!(0)),
        ("/resources/source_lut_invalidations", json!(0)),
        ("/resources/cancelled_publications", json!(1)),
        ("/resources/validation_errors", json!(1)),
        ("/numerical/constant_max_error", json!(2.1e-5)),
        ("/numerical/scalar_max_error", json!(2.1e-5)),
        ("/numerical/color_max_error", json!(0.0006)),
        ("/numerical/height_precision_max_error", json!(4e-7)),
        ("/numerical/period_max_error", json!(2.1e-5)),
        ("/numerical/histogram_count_mismatch", json!(1)),
        ("/numerical/nonfinite_pixels", json!(1)),
        ("/numerical/max_luminance_ratio", json!(16.1)),
        ("/numerical/clipped_pixel_fraction", json!(1.01)),
        ("/numerical/same_backend_identical", json!(false)),
        ("/cdf/0/max_distance", json!(0.050001)),
        ("/cdf/1/max_distance", json!(0.050001)),
        ("/cdf/0/size", json!(256)),
        ("/cdf/1/fixture", json!("unspecified_noise")),
        ("/seams/0/result_boundary_rms", json!(0.051)),
        ("/seams/1/result_interior_rms", json!(0.019)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            c::validate(&bad).is_err(),
            "accepted invalid evidence: {path}"
        );
    }
    for pass in c::PASSES {
        for quality in ["timing", "maximum_quality"] {
            let mut bad = valid.clone();
            bad[quality][format!("includes_{pass}")] = json!(false);
            assert!(c::validate(&bad).is_err(), "omitted {quality}/{pass}");
        }
    }
    for i in 0..c::REQUIRED_CHECKS.len() {
        let mut bad = valid.clone();
        bad["checks"].as_array_mut().unwrap().remove(i);
        assert!(c::validate(&bad).is_err());
    }
    let mut bad = valid.clone();
    bad["checks"]
        .as_array_mut()
        .unwrap()
        .push(valid["checks"][0].clone());
    assert!(c::validate(&bad).is_err());
    let mut bad = valid;
    bad["cdf"].as_array_mut().unwrap().remove(1);
    assert!(c::validate(&bad).is_err());
}

#[test]
fn shared_validator_must_apply_histogram_contract_after_registry_integration() {
    let valid = report();
    shared::validate_report(&valid).expect("register suite AND node-specific report validator");
    let mut bad = valid;
    bad["cdf"][0]["max_distance"] = json!(0.051);
    assert!(
        shared::validate_report(&bad).is_err(),
        "shared harness bypassed T-059 CDF assertions"
    );
}

#[test]
fn timing_sums_all_passes_and_reports_maximum_quality_separately() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![3.0, 3.0, 3.0, 3.0, 2.0, 2.0]; 100]);
    let t = shared::aggregate_timings(&rows, 10).unwrap();
    assert_eq!((t.measurements, t.p95_ms), (100, 16.0));
    let mut bad = report();
    bad["timing"]["p95_ms"] = json!(t.p95_ms);
    assert!(c::validate(&bad).is_err());
    c::validate(&report())
        .expect("maximum-quality 25 ms is recorded without imposing the default gate");
}
