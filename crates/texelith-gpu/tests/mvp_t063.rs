//! Preview mesh behavioral contracts.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "../examples/mvp_support/t063/checks.rs"]
mod checks;
#[path = "mvp_t063/report.rs"]
mod report;
#[path = "../examples/mvp_support/t063/report_contract.rs"]
mod report_contract;
#[path = "../examples/mvp_support/contract.rs"]
mod suite_contract;
#[path = "../examples/mvp_support/t063/support.rs"]
mod support;
#[path = "mvp_t063/targets.rs"]
mod targets;
#[path = "mvp_t063/worker.rs"]
mod worker;
