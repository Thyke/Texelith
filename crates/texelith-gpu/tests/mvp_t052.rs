//! T-052 tests-first public-node contract. No production stubs or registry coupling.
#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t052/behavior.rs"]
mod behavior;
#[path = "mvp_t052/contracts.rs"]
mod contracts;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t052/oracle.rs"]
mod oracle;
#[path = "mvp_t052/report.rs"]
mod report;
#[path = "mvp_t052/support.rs"]
mod support;
