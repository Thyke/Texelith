//! T-058 tests-first contract. No production code, registration or stubs.
#[path = "common/gpu_availability.rs"]
mod gpu_availability;

#[path = "support/abi.rs"]
#[allow(dead_code)]
mod abi;
#[path = "mvp_t058/behavior.rs"]
mod behavior;
#[path = "mvp_t058/gpu.rs"]
mod gpu;
#[path = "../examples/mvp_support/contract.rs"]
mod harness_contract;
#[path = "mvp_t058/planning.rs"]
mod planning;
#[path = "../examples/mvp_support/t058/reference.rs"]
mod reference;
#[path = "mvp_t058/report.rs"]
mod report;
#[path = "mvp_t058/support.rs"]
mod support;

// Keep the pre-implementation target executable using existing public APIs.
// Registration activates the public metadata contract; the suite registration
// test below stays red until that happens, so these tests cannot be skipped green.
macro_rules! splat_metadata_contract {
    ($v:vis seamless_splat {$($exports:tt)*} [$($runtime:tt)*] [$($suite:tt)*]; $($rest:tt)*) => {
        fn metadata_contract_registered() -> bool { true }
        #[path = "mvp_t058/transform.rs"]
        mod transform;
    };
    ($v:vis $module:ident {$($exports:tt)*} [$($runtime:tt)*] [$($suite:tt)*]; $($rest:tt)*) => {
        splat_metadata_contract!($($rest)*);
    };
    () => { fn metadata_contract_registered() -> bool { false } };
}
texelith_gpu::gpu_nodes!(splat_metadata_contract);
