//! Poisson operator API and numerical contracts.
//! No implementation stubs and no V-cycle driver are supplied by this pass.

#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t049/compute_abi.rs"]
mod compute_abi;
#[path = "mvp_t049/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t049/operators.rs"]
mod operators;
#[path = "../examples/mvp_support/t049/oracle.rs"]
mod oracle;
#[path = "mvp_t049/report.rs"]
mod report;
#[path = "../examples/mvp_support/t049/report_contract.rs"]
mod report_contract;
