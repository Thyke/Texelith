use super::harness_contract::{aggregate_timings, select_suite, validate_completed_report};
use serde_json::json;
#[path = "../../examples/mvp_support/t050/report_contract.rs"]
mod poisson_contract;

#[test]
fn approved_limits_accept_100ms_full_and_reject_16ms_proxy_and_false_optima() {
    let timing = json!({"size":4096,"kind":"completion_upper_bound","warmups":10,"measurements":100,
        "p95_ms":100.0,"submission_ms":0.1,"peak_bytes":339821780u64,"scratch_peak_bytes":138495188u64,
        "nonconverged_samples":0,"cancellation_errors":0,"relative_optimality_gap":0.00001,"rms":0.00001,
        "objective_residual_l2":0.19000001,"optimum_residual_l2":0.19,"gradient_l2":1.0});
    let mut proxy = timing.clone();
    proxy["size"] = json!(1024);
    proxy["p95_ms"] = json!(15.99);
    let valid = json!({"passed":true,"suite":"poisson","size":4096,"seed":50,
        "manifest":{},"manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "checks":[{"name":"cpu_optimum","passed":true}],
        "timing":timing,"maximum_quality":timing,"interactive_proxy":proxy});
    let validate = |r: &serde_json::Value| {
        poisson_contract::validate(r, |view| validate_completed_report(view, Ok(())))
    };
    assert!(validate(&valid).is_ok());
    for (path, value) in [
        ("/timing/p95_ms", json!(100.01)),
        ("/interactive_proxy/p95_ms", json!(16.0)),
        ("/interactive_proxy/size", json!(512)),
        ("/timing/relative_optimality_gap", json!(0.00101)),
        ("/timing/objective_residual_l2", json!(0.2)),
        ("/maximum_quality/relative_optimality_gap", json!(0.00101)),
        ("/timing/measurements", json!(99)),
        ("/interactive_proxy/nonconverged_samples", json!(1)),
        ("/checks", json!([])),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        assert!(validate(&invalid).is_err(), "accepted invalid {path}");
    }
    assert!(
        poisson_contract::validate(&valid, |view| validate_completed_report(
            view,
            Err("GPU error".into())
        ))
        .is_err()
    );
}

#[test]
fn poisson_suite_is_nonempty_and_registered_once_in_maps_and_all() {
    for name in ["poisson", "maps", "all"] {
        let suites = select_suite(name).expect("T-050 poisson suite must be registered");
        assert_eq!(
            suites.iter().filter(|s| **s == "poisson").count(),
            1,
            "{name}: {suites:?}"
        );
        let mut unique = suites.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), suites.len(), "duplicate suites in {name}");
    }
    for invalid in ["", " ", "Poisson", "poisson_solver", "poisson "] {
        assert!(select_suite(invalid).is_err());
    }
}
#[test]
fn full_node_p95_sums_every_v_cycle_and_setup_and_finalization_before_percentile() {
    // Ten cold warmups; 94 cheap runs and six expensive full solves. Measuring
    // one cycle, or averaging pass percentiles, would hide the failing p95.
    let mut samples = vec![vec![500.0; 10]; 10];
    samples.extend(vec![vec![0.2; 10]; 94]);
    samples.extend(vec![vec![1.7; 10]; 6]);
    let timing = aggregate_timings(&samples, 10).unwrap();
    assert_eq!(timing.measurements, 100);
    assert!((timing.p95_ms - 17.0).abs() < 1e-12);
    assert!(timing.p95_ms >= 16.0);
    let mut maximum = vec![vec![0.0; 18]; 10];
    maximum.extend(vec![vec![1.1; 18]; 100]);
    assert!(
        aggregate_timings(&maximum, 10).unwrap().p95_ms > 16.0,
        "maximum quality timing must be reported separately from default quality"
    );
}
#[test]
fn report_requires_real_assertions_and_gpu_errors_cannot_be_hidden_by_passing_json() {
    // Invented schema fixture, deliberately never persisted as measured evidence.
    let valid = json!({"passed":true,"suite":"poisson","size":4096,"seed":50,
        "manifest":{"boundary":"repeat","v_cycles":8,"pre_smooth":3,"post_smooth":3,
            "coarsest_iterations":32,"relative_tolerance":0.001,"absolute_tolerance":0.000001},
        "manifest_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "adapter":{"name":"RTX 5070 Ti","backend":"Dx12"},
        "checks":[{"name":"periodic_cpu_oracle","passed":true}],
        "timing":{"kind":"completion_upper_bound","warmups":10,"measurements":100,
            "p95_ms":7.0,"submission_ms":0.1,"peak_bytes":536870912u64,"cancellation_errors":0}});
    assert!(validate_completed_report(&valid, Ok(())).is_ok());
    assert!(
        validate_completed_report(&valid, Err("injected GPU validation error".into())).is_err()
    );
    for (path, value) in [
        ("/checks", json!([])),
        ("/checks/0/passed", json!(false)),
        ("/timing/p95_ms", json!(16.0)),
        ("/timing/measurements", json!(99)),
        ("/timing/warmups", json!(9)),
        ("/timing/kind", json!("cpu_encode_only")),
        ("/timing/cancellation_errors", json!(1)),
        ("/passed", json!(false)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        assert!(
            validate_completed_report(&invalid, Ok(())).is_err(),
            "accepted invalid {path}"
        );
    }
}
