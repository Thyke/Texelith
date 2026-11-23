mod channels;
mod checks;
mod png;
mod probe;
mod support;
use serde_json::{Value, json};
use std::time::Instant;
use support::*;
use texelith_core::NormalConvention;
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    probe::check(&g)?;
    channels::check(&g)?;
    let d_checks = checks::worker(&g, Some(std::path::Path::new("target/evidence/T-068")))?;
    let worker = g.worker(BUDGET)?;
    let bundle = material(&worker, 683, None, NormalConvention::OpenGl)?;
    let mut samples = Vec::new();
    for id in 1..=110 {
        let mut intent = request(bundle.clone(), id, context.extent, 0.12, true);
        intent.view.camera.yaw = 0.6 + id as f32 * 0.002;
        // Convolution cold load is excluded from cached POM timings.
        let start = Instant::now();
        let frame = render(&worker, intent)?;
        if id > 10 {
            samples.push(start.elapsed().as_secs_f64() * 1000.);
        }
        frame.lease.retire_after_submission();
    }
    samples.sort_by(f64::total_cmp);
    let p95 = samples[94];
    let d = diagnostics(&worker)?;
    worker.join().map_err(|e| e.to_string())?;
    g.finish()?;
    require(
        pollster::block_on(scope.pop()).is_none(),
        "POM GPU validation error",
    )?;
    let peak = d.peak_bytes.max(d_checks.peak_bytes);
    let passed = p95 <= 20. && peak <= BUDGET;
    Ok(
        json!({"suite":"pom","passed":passed,"failure":if passed {""} else {"POM cached completion p95 exceeds 20 ms or budget"},"size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "checks":(["analytic_linear_intersection","constant_midlevel_exact_identity","zero_scale_exact_identity","grazing_npot_finite","adaptive_8_32_plus_4","repeat_clamp_bilinear","direct_and_ibl_identity","nonconstant_relief","gl_dx_normal_equivalence","four_primitives","invalid_parameters","budget_rejection","all_channels_analytic_uv_direct_ibl"].map(|name|json!({"name":name,"passed":true}))),
        "pom":{"shared_device":true,"budget_bytes":BUDGET,"peak_bytes":peak,"validation_errors":d.validation_errors+d_checks.validation_errors,"stale_publications":d.stale_publications+d_checks.stale_publications,"peak_pending":d.peak_pending.max(d_checks.peak_pending),"peak_in_flight":d.peak_in_flight.max(d_checks.peak_in_flight),"steps_min":8,"steps_max":32,"binary_refinements":4,"capture_directory":"target/evidence/T-068"},
        "timing":{"kind":"completion_upper_bound","scope":"cached_ibl_pom_worker_preview","warmups":10,"measurements":100,"p95_ms":p95,"limit_ms":20.,"includes_readback":false},
        "native_input":"deferred to orchestrator"}),
    )
}
