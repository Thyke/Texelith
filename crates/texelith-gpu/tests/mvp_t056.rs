//! Tests-first paint-mask contract through the existing public material API.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "mvp_t056/behavior.rs"]
mod behavior;
#[path = "mvp_t056/contracts.rs"]
mod contracts;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t056/reference.rs"]
mod reference;
#[path = "mvp_t056/report.rs"]
mod report;
#[path = "mvp_t056/support.rs"]
mod support;
