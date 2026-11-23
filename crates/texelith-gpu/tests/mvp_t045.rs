//! Production material scheduler and planner contracts.

#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "mvp_t045/material.rs"]
mod material;
#[path = "mvp_t045/reservations.rs"]
mod reservations;
#[path = "mvp_t045/sources.rs"]
mod sources;

#[path = "mvp_t045/recovery.rs"]
mod recovery;

#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;

use serde_json::json;

#[test]
fn resources_suite_is_registered_in_relevant_aggregates() {
    for suite in ["resources", "maps", "viewport", "all"] {
        let selected = harness_contract::select_suite(suite).unwrap();
        assert!(
            selected.contains(&"resources"),
            "missing resources in {suite}"
        );
    }
    for invalid in ["", "resource", "unknown"] {
        assert!(harness_contract::select_suite(invalid).is_err());
    }
}

fn resource_report() -> serde_json::Value {
    json!({
        "passed": true, "suite": "resources", "size": 4096,
        "adapter": {"name": "RTX 5070 Ti", "backend": "Dx12"},
        "seed": 42, "manifest": {"scenario": "material_resources"},
        "manifest_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks": [
            {"name": "typed_sources", "passed": true},
            {"name": "job_identity", "passed": true},
            {"name": "coherent_outputs", "passed": true},
            {"name": "liveness", "passed": true},
            {"name": "cache_eviction", "passed": true},
            {"name": "lease_fence", "passed": true},
            {"name": "low_budget", "passed": true},
            {"name": "cancellation", "passed": true}
        ],
        "timing": {"kind": "completion_upper_bound", "warmups": 10,
            "measurements": 100, "p95_ms": 2.0, "submission_ms": 0.2,
            "peak_bytes": 1_073_741_824u64, "cancellation_errors": 0},
        "resources": {"budget_bytes": 1_073_741_824u64,
            "peak_in_flight": 1, "peak_pending": 1,
            "cancellation_requests": 1, "cancelled_publications": 0,
            "validation_errors": 0, "viewport_reserved_bytes": 150_994_944u64,
            "readback_reserved_bytes": 134_217_728u64}
    })
}

#[test]
fn resources_report_requires_real_resource_assertions_and_one_gib_peak() {
    let valid = resource_report();
    assert!(harness_contract::validate_resources_report(&valid).is_ok());
    for (pointer, value) in [
        ("/timing/peak_bytes", json!(1_073_741_825u64)),
        ("/resources/budget_bytes", json!(2_147_483_648u64)),
        ("/resources/peak_in_flight", json!(2)),
        ("/resources/peak_pending", json!(2)),
        ("/resources/cancellation_requests", json!(0)),
        ("/resources/cancelled_publications", json!(1)),
        ("/resources/validation_errors", json!(1)),
        ("/resources/viewport_reserved_bytes", json!(0)),
        ("/resources/readback_reserved_bytes", json!(0)),
        ("/checks", json!([])),
        ("/checks", json!([{"name": "normal", "passed": true}])),
        ("/checks/0/passed", json!(false)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        assert!(
            harness_contract::validate_resources_report(&invalid).is_err(),
            "accepted invalid resources report at {pointer}"
        );
    }
    for index in 0..8 {
        let mut incomplete = valid.clone();
        incomplete["checks"].as_array_mut().unwrap().remove(index);
        assert!(harness_contract::validate_resources_report(&incomplete).is_err());
    }
    let mut missing = valid;
    missing.as_object_mut().unwrap().remove("resources");
    assert!(harness_contract::validate_resources_report(&missing).is_err());
}
