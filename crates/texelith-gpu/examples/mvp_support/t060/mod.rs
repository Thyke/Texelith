//! Headless numerical and completion-bound performance verification.
mod numerics;
#[allow(dead_code)]
mod oracle;
mod performance;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t060/support.rs"]
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
        executor: pollster::block_on(texelith_gpu::GpuExecutor::new(
            context.device.clone(),
            context.queue.clone(),
        ))
        .map_err(|e| e.to_string())?,
    };
    let checks = numerics::run(&gpu)?;
    let (timing, maximum) = performance::run(context, &gpu)?;
    gpu.wait();
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(format!("seam diagnostic GPU validation: {error}"));
    }
    let manifest = json!({"boundary_band_px":4,"gradient":"forward_repeat","ratio_epsilon":1e-8,
        "color_distance":"linear_rgb_rms_alpha_excluded","seed":60,"size":context.extent.width,
        "quality":"fixed_exact_three_pass","extreme_gradient_saturation":1e18});
    let mut bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    let hash = SourceDescriptor::from_rgba8(
        &bytes,
        Extent::new((bytes.len() / 4) as u32, 1).unwrap(),
        InputInterpretation::LinearHeight,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|v| format!("{v:02x}"))
    .collect::<String>();
    let mut report = json!({"passed":true,"suite":"seam_heatmap","size":context.extent.width,"seed":60,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver,"driver_info":context.info.driver_info},
        "manifest":manifest,"manifest_hash":hash,"manifest_hash_encoding":"BLAKE3 of compact JSON padded to a multiple of four",
        "checks":checks,
        "timing":timing,"maximum_quality":maximum,
        "quality_note":"Fixed exact diagnostic has no quality knob; default and maximum run independently at identical settings.",
        "cancellation_scope":"caller generation token checked before/between passes and before publication; no core diagnostic graph variant exists",
        "accounting_scope":"input, heatmap, upload/readback allowance plus all internal pooled scratch/uniform/staging/scalar readback; excludes driver allocations"});
    if let Err(error) = crate::contract::validate_report(&report) {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
