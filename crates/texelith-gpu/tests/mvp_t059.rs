//! T-059 tests-first behavioral contract. No production code or success stubs.
#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t059/behavior.rs"]
mod behavior;
#[path = "mvp_t059/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t059/planning.rs"]
mod planning;
#[path = "../examples/mvp_support/t059/reference.rs"]
mod reference;
#[path = "mvp_t059/regression.rs"]
mod regression;
#[path = "mvp_t059/report.rs"]
mod report;
use harness_contract::histogram_contract as report_contract;
#[path = "mvp_t059/support.rs"]
mod support;
