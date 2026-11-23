//! T-063 headless shared-device behavioral and measurement suite.
mod checks;
mod report_contract;
mod support;
use serde_json::{Value, json};
use std::time::Instant;
use support::*;
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{
    pool::{AllocationDesc, FenceId, ResourcePool},
    viewport_resources::ViewportTargetPlan,
    viewport_shared::PreviewContent,
};

fn display_fence() -> Result<()> {
    let desc = AllocationDesc::Texture {
        extent: Extent::new(17, 3).map_err(|e| e.to_string())?,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
    };
    let bytes = desc.byte_len().map_err(|e| e.to_string())?;
    let mut pool = ResourcePool::new(bytes);
    let id = pool.allocate(desc).map_err(|e| e.to_string())?;
    pool.retire_ui(id, FenceId(9)).map_err(|e| e.to_string())?;
    pool.complete(FenceId(8));
    require(
        pool.allocate(desc).is_err() && pool.diagnostics().retired_bytes == bytes,
        "display fence did not prevent reuse",
    )?;
    pool.complete(FenceId(9));
    require(
        pool.allocate(desc).map_err(|e| e.to_string())? == id,
        "display allocation not reusable after final fence",
    )
}
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let g = Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let outcome = run(&g, context);
    g.finish()?;
    let validation = pollster::block_on(scope.pop());
    require(
        validation.is_none(),
        &format!("viewport validation error: {validation:?}"),
    )?;
    outcome
}
fn run(g: &Gpu, context: &crate::SuiteContext<'_>) -> Result<Value> {
    checks::clear_and_ramp(g)?;
    let churn = checks::identity_and_churn(g)?;
    let lifetime = checks::lifetime_and_budget(g)?;
    checks::compute_supersedes_preview(g)?;
    display_fence()?;
    let worker = g.worker(BUDGET)?;
    let bundle = material(&worker, 63, 1, 1)?;
    let mut times = Vec::with_capacity(100);
    for serial in 1..=110 {
        let start = Instant::now();
        let frame = render(
            &worker,
            request(
                bundle.clone(),
                serial,
                context.extent,
                PreviewContent::MaterialRamp,
            ),
        )?;
        if serial > 10 {
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        // Immutable output is valid on the caller's device, and submission of
        // its final copy precedes dropping the UI owner and its queue callback.
        if serial == 110 {
            require(
                !g.read_display(frame.lease.texture())?.is_empty(),
                "shared-device display cannot be read",
            )?;
        }
        frame.lease.retire_after_submission();
    }
    times.sort_by(f64::total_cmp);
    let timing = checks::diagnostics(&worker)?;
    let target_plan = ViewportTargetPlan::new(context.extent.width, context.extent.height)
        .map_err(|e| e.to_string())?;
    let target_bytes: u64 = target_plan
        .allocations()
        .iter()
        .map(|a| a.byte_len().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum();
    let manifest = json!({"task":"T-063","fixture":"original 17x3 linear height ramp","preview_extent":[context.extent.width,context.extent.height],"display_tolerance_codes":1,"target_formats":["Rgba16Float","Depth32Float","Rgba8Unorm"],"sample_count":1});
    let mut manifest_bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !manifest_bytes.len().is_multiple_of(4) {
        manifest_bytes.push(0);
    }
    let manifest_hash = SourceDescriptor::from_rgba8(
        &manifest_bytes,
        Extent::new((manifest_bytes.len() / 4) as u32, 1).map_err(|e| e.to_string())?,
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    let report = json!({
        "suite":"viewport_resources","passed":true,"size":context.extent.width,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "manifest":manifest,"manifest_hash":manifest_hash,
        "checks":report_contract::CHECKS.iter().map(|name| json!({"name":name,"passed":true})).collect::<Vec<_>>(),
        "viewport_resources":{"budget_bytes":BUDGET,"peak_bytes":timing.peak_bytes.max(churn.peak_bytes).max(lifetime.peak_bytes),"target_bytes":target_bytes,
            "validation_errors":timing.validation_errors + churn.validation_errors + lifetime.validation_errors,"stale_publications":timing.stale_publications + churn.stale_publications + lifetime.stale_publications,
            "peak_pending":timing.peak_pending.max(churn.peak_pending),"peak_in_flight":timing.peak_in_flight.max(churn.peak_in_flight),
            "resize_requests":256,"coalesced_requests":churn.coalesced,"low_budget_rejections":1,"shared_device":true},
        "timing":{"kind":"completion_upper_bound","scope":"worker_preview","warmups":10,"measurements":times.len(),"p95_ms":times[94]}
    });
    worker.join().map_err(|e| e.to_string())?;
    report_contract::validate(&report)?;
    Ok(report)
}
