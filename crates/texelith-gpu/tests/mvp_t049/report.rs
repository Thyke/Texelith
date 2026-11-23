use super::{
    harness_contract::{aggregate_timings, select_suite},
    report_contract::{CHECKS, OPERATORS, validate_poisson_report},
};
use serde_json::{Value, json};

#[test]
fn poisson_suite_is_registered_in_maps_and_all_without_duplicate_or_empty_entries() {
    for name in ["poisson_operators", "maps", "all"] {
        let suites = select_suite(name).unwrap();
        assert_eq!(
            suites.iter().filter(|n| **n == "poisson_operators").count(),
            1
        );
        let mut unique = suites.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), suites.len());
    }
    for name in ["", " ", "poisson_operator", "POISSON_OPERATORS"] {
        assert!(select_suite(name).is_err());
    }
    assert_eq!(select_suite("poisson").unwrap(), vec!["poisson"]);
}

// Schema fixture only: these invented numbers are never saved as hardware evidence.
fn report() -> Value {
    let timing = json!({"kind":"completion_upper_bound","warmups":10,"measurements":100,
        "p95_ms":7.0,"submission_ms":0.3,"peak_bytes":536870912u64,"cancellation_errors":0,
        "scope":"full_operator_sweep","includes_reductions":true,"includes_coarse_operators":true,
        "operators":OPERATORS});
    json!({"passed":true,"suite":"poisson_operators","size":4096,"seed":49,
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "manifest":{"boundary":"repeat","omega":2.0/3.0,"normal_z_floor":0.05,
            "slope_cap":20.0,"coarsening":"ceil_half_area_bilinear_galerkin","algorithm_revision":1},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "checks":CHECKS.iter().map(|n|json!({"name":n,"passed":true})).collect::<Vec<_>>(),
        "timing":timing.clone(),
        "maximum_quality":{"settings":"all 13 levels; complete operator sweep","p95_ms":24.0,
            "operators":OPERATORS,"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "submission_ms":0.4,"peak_bytes":536870912u64,"cancellation_errors":0,
            "scope":"full_operator_sweep","includes_reductions":true,"includes_coarse_operators":true},
        "poisson_operators":{"budget_bytes":1073741824u64,"scratch_peak_bytes":201326592u64,
            "peak_in_flight":1,"validation_errors":0,"cancelled_publications":0,
            "cancellation_requests":1,"low_budget_rejections":1,
            "adjoint_relative_error":0.000001,"scalar_max_error":0.000001,"galerkin_max_error":0.000001}
    })
}

#[test]
fn report_contract_rejects_missing_oracles_untimed_passes_unmeasured_cancellation_and_budget_overruns()
 {
    let valid = report();
    assert!(validate_poisson_report(&valid).is_ok());
    for (path, value) in [
        ("/passed", json!(false)),
        ("/suite", json!("normal")),
        ("/size", json!(0)),
        ("/manifest", Value::Null),
        ("/manifest_hash", json!("not-a-hash")),
        ("/adapter/name", json!("")),
        ("/adapter/backend", Value::Null),
        ("/manifest/omega", json!(1.0)),
        ("/manifest/normal_z_floor", json!(0.01)),
        ("/manifest/slope_cap", json!(200.0)),
        ("/manifest/boundary", json!("clamp")),
        ("/manifest/coarsening", json!("even_stencil")),
        ("/manifest/algorithm_revision", json!(0)),
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/p95_ms", Value::Null),
        ("/timing/submission_ms", json!(-1.0)),
        ("/timing/warmups", json!(9)),
        ("/timing/measurements", json!(99)),
        ("/timing/scope", json!("jacobi_only")),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/includes_reductions", json!(false)),
        ("/timing/includes_coarse_operators", json!(false)),
        ("/timing/operators", json!([])),
        ("/timing/peak_bytes", json!(0)),
        ("/timing/peak_bytes", json!(1073741825u64)),
        ("/timing/cancellation_errors", json!(1)),
        ("/maximum_quality", Value::Null),
        ("/maximum_quality/measurements", json!(99)),
        ("/maximum_quality/p95_ms", Value::Null),
        ("/maximum_quality/settings", json!("")),
        ("/maximum_quality/operators", json!([])),
        ("/maximum_quality/peak_bytes", json!(0)),
        ("/maximum_quality/includes_reductions", json!(false)),
        ("/poisson_operators/scratch_peak_bytes", json!(0)),
        ("/poisson_operators/scratch_peak_bytes", json!(201326593u64)),
        ("/poisson_operators/budget_bytes", json!(0)),
        ("/poisson_operators/budget_bytes", json!(1073741825u64)),
        ("/poisson_operators/peak_in_flight", json!(2)),
        ("/poisson_operators/validation_errors", json!(1)),
        ("/poisson_operators/cancelled_publications", json!(1)),
        ("/poisson_operators/cancellation_requests", json!(0)),
        ("/poisson_operators/low_budget_rejections", json!(0)),
        ("/poisson_operators/adjoint_relative_error", json!(0.000021)),
        ("/poisson_operators/scalar_max_error", json!(0.000021)),
        ("/poisson_operators/galerkin_max_error", json!(0.000021)),
    ] {
        let mut bad = valid.clone();
        *bad.pointer_mut(path).unwrap() = value;
        assert!(
            validate_poisson_report(&bad).is_err(),
            "accepted invalid {path}"
        );
    }
    for name in CHECKS {
        let mut bad = valid.clone();
        bad["checks"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["name"] != *name);
        assert!(
            validate_poisson_report(&bad).is_err(),
            "accepted absent {name}"
        );
    }
    for name in OPERATORS {
        let mut bad = valid.clone();
        bad["timing"]["operators"]
            .as_array_mut()
            .unwrap()
            .retain(|v| v != name);
        assert!(
            validate_poisson_report(&bad).is_err(),
            "accepted untimed {name}"
        );
    }
}

#[test]
fn timing_gate_sums_all_sweep_passes_before_taking_p95() {
    let mut rows = vec![vec![999.0; 3]; 10];
    rows.extend(vec![vec![5.0, 5.0, 5.0]; 94]);
    rows.extend(vec![vec![6.0, 6.0, 6.0]; 6]);
    let timing = aggregate_timings(&rows, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert_eq!(
        timing.p95_ms, 18.0,
        "per-pass p95 would incorrectly hide a failed sweep"
    );
    assert!(timing.p95_ms >= 16.0);
}
