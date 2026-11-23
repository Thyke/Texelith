//! T-060 behavioral contract. The proposed production boundary is isolated in node.rs.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "mvp_t060/behavior.rs"]
mod behavior;
#[allow(dead_code)]
#[path = "../examples/mvp_support/t060/numerics.rs"]
mod cancellation;
#[path = "../examples/mvp_support/contract.rs"]
mod contract;
#[path = "mvp_t060/node.rs"]
mod node;
#[path = "../examples/mvp_support/t060/oracle.rs"]
mod oracle;
#[path = "mvp_t060/report.rs"]
mod report;
#[path = "mvp_t060/support.rs"]
mod support;

#[test]
fn generation_cancellation_and_submitted_scratch_retirement() {
    support::with_gpu(|gpu| cancellation::cancellation_budget(gpu).unwrap());
}

#[test]
fn repeated_unsubmitted_cancellation_releases_actual_gpu_memory() {
    support::with_gpu(|gpu| cancellation::unsubmitted_cancellation_budget(gpu).unwrap());
}

#[test]
fn seam_heatmap_is_selectable_and_in_the_map_aggregates() {
    for name in ["seam_heatmap", "maps", "all"] {
        let suites = contract::select_suite(name).expect("suite must be selectable");
        assert_eq!(
            suites.iter().filter(|&&s| s == "seam_heatmap").count(),
            1,
            "{name} must contain the diagnostic exactly once"
        );
    }
}

#[test]
fn empty_and_unknown_suite_names_are_rejected() {
    for name in ["", " ", "seam-heatmap", "unknown"] {
        assert!(contract::select_suite(name).is_err(), "accepted {name:?}");
    }
}

#[test]
fn timing_uses_all_passes_and_excludes_warmups() {
    let mut rows = vec![vec![1000.0]; 10];
    rows.extend(vec![vec![5.0, 4.0, 7.0]; 100]);
    let result = contract::aggregate_timings(&rows, 10).unwrap();
    assert_eq!(result.measurements, 100);
    assert_eq!(result.p95_ms, 16.0);
    assert!(contract::aggregate_timings(&rows[..10], 10).is_err());
}
