//! Full-set diagnostics through the production immutable export coordinator.
mod cli;
mod fixture;
mod metrics;
mod runtime;
pub use cli::{Config, dispatch};
pub use fixture::{fixture_graph, fixture_pixels, scenario_plan};
pub(crate) use metrics::mark;
pub use metrics::{Event, Metric, Trace};
pub use runtime::exercise;
