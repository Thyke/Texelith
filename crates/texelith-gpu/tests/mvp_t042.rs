//! Adapter-free behavioral contract for the MVP harness.
//!
//! The implementer supplies `mvp_support/contract.rs` with the functions used
//! below. The example must use these same validation and aggregation paths.

#[path = "../examples/mvp_support/contract.rs"]
mod contract;

use contract::{
    aggregate_timings, compare_scalar, fixture, normal_from_height, select_adapter, select_suite,
    validate_completed_report, validate_report,
};
use serde_json::{Value, json};

fn valid_report() -> Value {
    json!({
        "passed": true,
        "suite": "baseline",
        "size": 4096,
        "adapter": { "name": "RTX 5070 Ti", "backend": "Dx12" },
        "seed": 42,
        "manifest": { "normal_strength": 2.0 },
        "manifest_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks": [{ "name": "normal", "passed": true }],
        "timing": {
            "kind": "completion_upper_bound",
            "warmups": 10,
            "measurements": 100,
            "p95_ms": 2.0,
            "submission_ms": 0.2,
            "peak_bytes": 1024,
            "cancellation_errors": 0
        }
    })
}

#[test]
fn report_enforces_strict_p95_limit_and_gpu_completion_errors() {
    let mut report = valid_report();
    for (p95, accepted) in [(15.999, true), (16.0, false), (16.001, false)] {
        report["timing"]["p95_ms"] = json!(p95);
        assert_eq!(validate_report(&report).is_ok(), accepted, "p95={p95}");
    }
    report["timing"]["p95_ms"] = json!(2.0);
    assert!(validate_completed_report(&report, Ok(())).is_ok());
    assert!(validate_completed_report(&report, Err("GPU error".into())).is_err());
}

#[test]
fn suite_selection_rejects_unknown_and_empty_aggregates() {
    assert!(!select_suite("baseline").unwrap().is_empty());
    assert!(select_suite("not-a-suite").is_err());
    assert!(select_suite("").is_err());
    for name in ["maps", "viewport", "all"] {
        if let Ok(tests) = select_suite(name) {
            assert!(!tests.is_empty(), "{name} must not silently pass empty");
        }
    }
}

#[test]
fn adapter_selection_requires_requested_name_and_backend() {
    let adapters = [("RTX 5070 Ti", "Dx12"), ("Other GPU", "Vulkan")];
    assert!(select_adapter(&adapters, "RTX 5070 Ti", "dx12").is_ok());
    assert!(select_adapter(&adapters, "missing adapter", "dx12").is_err());
    assert!(select_adapter(&adapters, "RTX 5070 Ti", "vulkan").is_err());
}

#[test]
fn normal_oracle_handles_flat_and_singleton_axes() {
    let flat = normal_from_height(3, 2, &[0.5; 6], 2.0).unwrap();
    assert_eq!(flat.len(), 6);
    for n in flat {
        assert!((n[0]).abs() < 2e-3 && (n[1]).abs() < 2e-3);
        assert!((n[2] - 1.0).abs() < 2e-3);
    }
    let ramp = normal_from_height(1, 3, &[0.0, 0.5, 1.0], 2.0).unwrap();
    for n in ramp {
        assert!(n[0].abs() < 2e-3 && n[1] > 0.0);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((length - 1.0).abs() < 2e-3);
    }
    assert!(normal_from_height(2, 2, &[0.0; 3], 1.0).is_err());
}

#[test]
fn comparisons_reject_different_extents_and_nonfinite_values() {
    assert!(compare_scalar(&[0.0, 1.0], &[0.0], 2e-5).is_err());
    assert!(compare_scalar(&[], &[], 2e-5).is_err());
    assert!(compare_scalar(&[f32::NAN], &[0.0], 2e-5).is_err());
    assert!(compare_scalar(&[0.0], &[0.000_01], 2e-5).is_ok());
    assert!(compare_scalar(&[0.0], &[0.000_03], 2e-5).is_err());
}

#[test]
fn seeded_fixtures_are_repeatable_and_respect_requested_extents() {
    let a = fixture(257, 193, 42).unwrap();
    let b = fixture(257, 193, 42).unwrap();
    let c = fixture(257, 193, 43).unwrap();
    assert_eq!(a.len(), 257 * 193);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(fixture(1, 13, 42).unwrap().len(), 13);
    assert_eq!(fixture(13, 1, 42).unwrap().len(), 13);
    assert!(fixture(0, 1, 42).is_err());
}

#[test]
fn timing_uses_only_measurements_and_sums_every_pass() {
    // Each row is one complete invocation, with one duration per pass in ms.
    let mut invocations = vec![vec![1000.0, 1000.0]; 10];
    invocations.extend(vec![vec![1.0, 2.0, 4.0]; 100]);
    let timing = aggregate_timings(&invocations, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert!((timing.p95_ms - 7.0).abs() < 1e-9);
    assert!(aggregate_timings(&invocations[..10], 10).is_err());
    assert!(aggregate_timings(&[vec![]], 0).is_err());
    assert!(aggregate_timings(&[vec![f64::NAN]], 0).is_err());
}

#[test]
fn report_requires_real_checks_adapter_manifest_and_labeled_timing() {
    let report = valid_report();
    assert!(validate_report(&report).is_ok());
    for invalid in [
        json!({"checks": []}),
        json!({"checks": [{"name": "normal", "passed": false}]}),
        json!({"adapter": null}),
        json!({"manifest_hash": ""}),
        json!({"timing": null}),
        json!({"timing": {"kind": "missing", "warmups": 10, "measurements": 100, "p95_ms": 2.0, "submission_ms": 0.2, "peak_bytes": 1024, "cancellation_errors": 0}}),
        json!({"timing": {"kind": "gpu_timestamp", "warmups": 10, "measurements": 0, "p95_ms": 2.0, "submission_ms": 0.2, "peak_bytes": 1024, "cancellation_errors": 0}}),
    ] {
        let mut candidate = report.clone();
        for (key, value) in invalid.as_object().unwrap() {
            candidate[key] = value.clone();
        }
        assert!(validate_report(&candidate).is_err(), "accepted {invalid}");
    }
    let mut timestamp_report = report;
    timestamp_report["timing"]["kind"] = json!("gpu_timestamp");
    assert!(validate_report(&timestamp_report).is_ok());
}
