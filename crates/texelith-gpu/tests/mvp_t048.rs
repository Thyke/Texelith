//! T-048 independent behavioral contract; production code belongs to the next role.
#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t048/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t048/normal.rs"]
mod normal;
#[path = "mvp_t048/plan.rs"]
mod plan;
#[path = "mvp_t048/reference.rs"]
mod reference;
#[path = "mvp_t048/report.rs"]
mod report;
