//! Tests-first metallic contract. No production stubs or registration internals.
#[path = "mvp_t054/behavior.rs"]
mod behavior;
#[path = "mvp_t054/contracts.rs"]
mod contracts;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t054/kernel.rs"]
mod kernel;
#[path = "mvp_t054/reference.rs"]
mod reference;
#[path = "mvp_t054/support.rs"]
mod support;
