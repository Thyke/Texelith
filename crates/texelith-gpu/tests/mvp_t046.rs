//! Production material worker behavioral contracts.

#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t046/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t046/oracle.rs"]
mod oracle;
#[path = "mvp_t046/pyramid.rs"]
mod pyramid;
#[path = "mvp_t046/report.rs"]
mod report;
#[path = "mvp_t046/sources.rs"]
mod sources;

#[path = "mvp_t046/plan.rs"]
mod plan;
