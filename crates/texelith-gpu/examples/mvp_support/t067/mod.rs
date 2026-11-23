mod checks;
mod interpolation;
mod numerics;
mod png;
mod report_contract;
mod support;
use serde_json::{Value, json};
use std::{sync::Arc, time::Instant};
use support::*;
use texelith_gpu::ibl::Environment;
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    numerics::constant_and_direction(&g)?;
    let (checks, cold_ms, convolution_ms) = checks::worker(
        &g,
        Some(std::path::Path::new(
            "target/evidence/T-067/ibl-sphere-matrix.png",
        )),
    )?;
    let worker = g.worker(BUDGET)?;
    let material = material(&worker)?;
    let environment = Arc::new(Environment::procedural());
    let mut samples = Vec::new();
    for id in 1..=110 {
        let mut intent = request(
            material.clone(),
            id,
            context.extent,
            environment.clone(),
            0.5,
            0.0,
        );
        intent.view.camera.yaw = id as f32 * 0.01;
        let start = Instant::now();
        let frame = render(&worker, intent)?;
        if id > 10 {
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        frame.lease.retire_after_submission();
    }
    samples.sort_by(f64::total_cmp);
    let d = worker.diagnostics();
    worker.join().map_err(|e| e.to_string())?;
    g.finish()?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "IBL GPU validation error",
    )?;
    let p95 = samples[94];
    let passed = p95 < 20.0
        && d.validation_errors == 0
        && d.stale_publications == 0
        && d.peak_bytes <= BUDGET;
    let report = json!({"suite":"ibl","passed":passed,"failure":if passed {""} else {"IBL cached preview exceeds 20 ms or resource contract failed"},"size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend)},
        "checks":report_contract::CHECKS.map(|name|json!({"name":name,"passed":true})),
        "ibl":{"shared_device":true,"cold_load_ms":cold_ms,"convolution_cold_ms":convolution_ms,"peak_bytes":d.peak_bytes.max(checks.peak_bytes),"validation_errors":d.validation_errors+checks.validation_errors,"stale_publications":d.stale_publications+checks.stale_publications,"cancelled_pass_batches":checks.cancelled_pass_batches,"output_bytes":4_892_672,"specular_samples":128,"diffuse_samples":64},
        "timing":{"scope":"cached_worker_preview","kind":"completion_upper_bound","warmups":10,"measurements":100,"p95_ms":p95,"limit_ms":20.0,"includes_readback":false},
        "native_input":"deferred to orchestrator"});
    if passed {
        report_contract::validate(&report)?;
    }
    Ok(report)
}
