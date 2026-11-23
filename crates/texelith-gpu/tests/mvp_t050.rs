//! T-050 tests-first behavioral contract. See mvp_t050/README.md for the
//! proposed additive solver API; no implementation or registration stubs here.
#[path = "mvp_t050/gpu.rs"]
mod gpu;
#[path = "mvp_t050/harness.rs"]
mod harness;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t050/material.rs"]
mod material;
#[path = "mvp_t050/oracle.rs"]
mod oracle;
#[path = "mvp_t050/proxy_oracle.rs"]
mod proxy_oracle;
#[path = "mvp_t050/solver.rs"]
mod solver;
