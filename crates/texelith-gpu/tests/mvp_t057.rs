//! T-057 independent behavioral contract; no production implementation or stubs.
#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t057/behavior.rs"]
mod behavior;
#[path = "mvp_t057/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t057/planning.rs"]
mod planning;
#[path = "mvp_t057/proxy.rs"]
mod proxy;
#[path = "../examples/mvp_support/t057/reference.rs"]
mod reference;
#[path = "mvp_t057/report.rs"]
mod report;
#[path = "mvp_t057/transform.rs"]
mod transform;
