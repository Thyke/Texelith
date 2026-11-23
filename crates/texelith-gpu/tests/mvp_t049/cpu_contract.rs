//! Standalone rustc driver for validating oracles/schema before the GPU API exists.
//! Not a second Cargo target; the dedicated mvp_t049 target includes these modules.
#[path = "../../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "../../examples/mvp_support/t049/oracle.rs"]
mod oracle;
#[path = "report.rs"]
mod report;
#[path = "../../examples/mvp_support/t049/report_contract.rs"]
mod report_contract;
