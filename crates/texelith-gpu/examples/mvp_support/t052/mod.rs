//! T-052 public-worker numerical checks and all-three-plane full-node timings.
use crate::reference::abi;
mod numerics;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t052/oracle.rs"]
mod oracle;
mod performance;
#[allow(dead_code, unused_imports)]
#[path = "../../../tests/mvp_t052/support.rs"]
mod support;

use serde_json::{Value, json};
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};

pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value, String> {
    let scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let gpu = support::Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    // Shared test-side readback and independent f64 oracle exercise public ports.
    let (checks, stats) = numerics::run(&gpu);
    let (mut timing, maximum) = performance::run(context)?;
    timing["peak_bytes"] = json!(timing["peak_bytes"].as_u64().unwrap().max(stats.peak_bytes));
    timing["cancellation_errors"] = json!(stats.stale_publications);
    performance::wait(&gpu.device)?;
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(format!("curvature GPU validation: {error}"));
    }
    let manifest = json!({"edge_scale":1.0,"cavity_scale":1.0,"boundary":"repeat","convention":"opengl","outputs":["curvature","edge","cavity"],"fixture":"half_tangent_cycle_9_amplitudes_x_1_over_128_y_1_over_2048","seed":52,"size":context.extent.width});
    let mut encoded = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !encoded.len().is_multiple_of(4) {
        encoded.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &encoded,
        Extent::new((encoded.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|v| format!("{v:02x}"))
    .collect::<String>();
    let mut report = json!({
        "passed":true,"suite":"curvature","size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "seed":52,"manifest":manifest,"manifest_hash":hash,
        "manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four",
        "checks":checks,"timing":timing,"maximum_quality":maximum,
        "curvature":{"outputs_per_node":3,"budget_bytes":support::GIB,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,"cancellation_requests":stats.cancellation_requests,"validation_errors":stats.validation_errors,"stale_publications":stats.stale_publications},
        "accounting_scope":"normal input, three R32 planes, uniforms, upload staging and padded scalar readback; starts at typed normal input, excludes driver allocations"
    });
    if let Err(error) = crate::contract::validate_report(&report) {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
