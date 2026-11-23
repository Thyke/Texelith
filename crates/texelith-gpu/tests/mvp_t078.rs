//! T-078 tests-first contract. The proposed GPU API is documented in
//! docs/evidence/T-078/test-contract.md; no production stubs are provided.
#[path = "mvp_t078/behavior.rs"]
mod behavior;
#[path = "mvp_t078/fixtures.rs"]
mod fixtures;
#[path = "mvp_t078/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
#[allow(clippy::duplicate_mod)] // Independently test the registered validator too.
mod harness_contract;
#[path = "mvp_t078/oracle.rs"]
mod oracle;
#[path = "../examples/mvp_support/t078/reference.rs"]
mod reference;
#[path = "mvp_t078/report.rs"]
mod report;
#[path = "../examples/mvp_support/t078/report_contract.rs"]
mod report_contract;
#[path = "mvp_t078/resources.rs"]
mod resources;
