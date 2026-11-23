//! Native eframe bootstrap and immutable shared-device display bridge.

mod app;
pub mod batch_jobs;
pub mod command_adapter;
pub mod compare;
pub mod controls;
mod display;
pub mod export_jobs;
pub mod export_perf;
pub mod input_locale;
pub mod inspectors;
mod integration;
mod io_worker;
pub mod material_ui;
pub mod measurement;
pub mod packing_editor;
pub mod perf_metrics;
mod perf_normal;
mod perf_runtime;
mod platform;
pub mod project_batch;
pub mod routing;
pub mod startup;
pub mod workspace;

pub use app::run;
pub use display::{DisplayBridge, DisplayLease, SharedGpu};
