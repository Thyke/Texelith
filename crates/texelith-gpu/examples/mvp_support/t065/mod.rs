mod checks;
mod oracle;
mod png;
mod report_contract;
mod support;
use serde_json::{Value, json};
use std::time::Instant;
use support::*;
use texelith_core::{NormalConvention, PreviewMesh};
use texelith_gpu::pbr::PbrSettings;

pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let outcome = run(&g, context);
    g.finish()?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "PBR shared-device validation error",
    )?;
    outcome
}
fn run(g: &Gpu, context: &crate::SuiteContext<'_>) -> Result<Value> {
    checks::analytic(g)?;
    let capture_dir = std::path::Path::new("docs/evidence/T-065");
    let churn = checks::primitives_and_lifetime(g, Some(capture_dir))?;
    checks::budget(g)?;
    let worker = g.worker(BUDGET)?;
    let bundle = material(
        &worker,
        656,
        [128, 128, 255],
        NormalConvention::OpenGl,
        true,
        false,
    )?;
    let mut samples = Vec::with_capacity(100);
    for id in 1..=110 {
        let mut intent = request(
            bundle.clone(),
            id,
            context.extent,
            PreviewMesh::Sphere,
            PbrSettings::default(),
        );
        intent.view.camera.yaw = id as f32 * 0.01;
        intent.view.light.direction = [-0.5, 0.8, 1.0];
        intent.view.light.intensity = 3.;
        let start = Instant::now();
        let frame = render(&worker, intent)?;
        if id > 10 {
            samples.push(start.elapsed().as_secs_f64() * 1000.);
        }
        frame.lease.retire_after_submission();
    }
    samples.sort_by(f64::total_cmp);
    let d = diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())?;
    let p95 = samples[94];
    let report = json!({
        "suite":"pbr","passed":p95<=20.,"failure":if p95>20. {"PBR completion p95 exceeds 20 ms"} else {""},"size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "checks":(report_contract::CHECKS.map(|name|json!({"name":name,"passed":true}))),
        "pbr":{"shared_device":true,"budget_bytes":BUDGET,"peak_bytes":d.peak_bytes.max(churn.peak_bytes),"validation_errors":d.validation_errors+churn.validation_errors,"stale_publications":d.stale_publications+churn.stale_publications,"peak_pending":d.peak_pending.max(churn.peak_pending),"peak_in_flight":d.peak_in_flight.max(churn.peak_in_flight),"primitives":["plane","cube","sphere","cylinder"],"capture_directory":capture_dir},
        "timing":{"kind":"completion_upper_bound","scope":"worker_preview","warmups":10,"measurements":100,"p95_ms":p95,"limit_ms":20.,"includes_readback":false},
        "native_input":"deferred to orchestrator"
    });
    if p95 <= 20. {
        report_contract::validate(&report)?;
    }
    Ok(report)
}
