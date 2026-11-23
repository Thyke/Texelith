//! Height preprocessing behavioral contracts.

#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t047/gpu.rs"]
#[allow(dead_code)]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t047/height.rs"]
mod height;
#[path = "mvp_t046/oracle.rs"]
mod oracle;
#[path = "mvp_t047/reference.rs"]
mod reference;
#[path = "mvp_t047/remap.rs"]
mod remap;
#[path = "mvp_t047/report.rs"]
mod report;
