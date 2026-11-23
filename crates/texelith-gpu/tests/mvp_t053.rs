//! T-053 tests-first roughness contract; no production implementation or fallback.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "mvp_t053/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t053/material.rs"]
mod material;
#[path = "../examples/mvp_support/t053/oracle.rs"]
mod oracle;
#[path = "mvp_t053/prepared.rs"]
mod prepared;
#[path = "mvp_t053/report.rs"]
mod report;
