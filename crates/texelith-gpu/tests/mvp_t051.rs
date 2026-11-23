//! Tests-first contract for ADR 0004 height horizon AO. No production stubs.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "mvp_t051/behavior.rs"]
mod behavior;
#[path = "mvp_t051/contracts.rs"]
mod contracts;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t051/oracle.rs"]
mod oracle;
#[path = "mvp_t051/support.rs"]
mod support;
