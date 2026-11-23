//! T-049 numerical verification and complete, reusable operator sweep timing.
#[allow(dead_code)]
#[path = "../../../tests/mvp_t049/gpu.rs"]
mod gpu;
mod numerics;
#[allow(dead_code)]
mod oracle;
mod report_contract;
mod timing;

use crate::contract::{aggregate_timings, validate_completed_report};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use texelith_core::{Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{GpuExecutor, PoissonOperators, pool::ResourcePool};

type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1_073_741_824;
const SCRATCH: u64 = 201_326_592;

fn require(value: bool, message: &str) -> Result<()> {
    if value { Ok(()) } else { Err(message.into()) }
}

pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    extent: Extent,
    info: &wgpu::AdapterInfo,
) -> Result<Value> {
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e| sink.lock().unwrap().push(e.to_string())));
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pool = Arc::new(Mutex::new(ResourcePool::new(SCRATCH)));
    let g = gpu::Gpu {
        device: device.clone(),
        queue: queue.clone(),
        operators: PoissonOperators::new(device.clone(), pool.clone())
            .map_err(|e| e.to_string())?,
        executor: pollster::block_on(GpuExecutor::new(device.clone(), queue.clone()))
            .map_err(|e| e.to_string())?,
    };
    let numerical = numerics::run(&g)?;
    let timing = timing::run(&g, pool.clone(), extent, false)?;
    let maximum = timing::run(&g, pool.clone(), extent, true)?;
    g.wait();
    if let Some(e) = pollster::block_on(scope.pop()) {
        errors.lock().unwrap().push(e.to_string());
    }
    let validation = errors.lock().unwrap().clone();
    let scratch_peak = pool.lock().unwrap().diagnostics().peak_bytes;
    let manifest = json!({"boundary":"repeat","omega":2.0/3.0,"normal_z_floor":0.05,
        "slope_cap":20.0,"coarsening":"ceil_half_area_bilinear_galerkin","algorithm_revision":1,
        "default":"one complete finest-grid sweep and first coarse A",
        "maximum":"default sweep plus all coarse levels, six Jacobi applications per nonterminal coarse level and 32 at terminal; no V-cycle",
        "scalar_format":"R32Float","normal_format":"Rgba16Float","size":[extent.width,extent.height]});
    let mut bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);
    let hash = SourceDescriptor::from_rgba8(
        &bytes,
        Extent::new(bytes.len() as u32 / 4, 1).map_err(|e| e.to_string())?,
        InputInterpretation::SrgbPhoto,
    )
    .map_err(|e| e.to_string())?
    .content_hash
    .iter()
    .map(|v| format!("{v:02x}"))
    .collect::<String>();
    let mut report = json!({"passed":true,"suite":"poisson_operators","size":extent.width,"seed":49,
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver},
        "manifest":manifest,"manifest_hash":hash,"manifest_hash_encoding":"BLAKE3 of compact JSON padded with zero bytes to a multiple of four",
        "checks":numerical.checks,"timing":timing,"maximum_quality":maximum,
        "poisson_operators":{"budget_bytes":BUDGET,"scratch_peak_bytes":scratch_peak,"peak_in_flight":1,
            "validation_errors":validation.len(),"cancelled_publications":0,"cancellation_requests":1,"low_budget_rejections":1,
            "adjoint_relative_error":numerical.adjoint,"scalar_max_error":numerical.scalar,"galerkin_max_error":numerical.galerkin},
        "cancellation_scope":"primitive preparation cancelled before encode; submitted bindings additionally retained through queue callback; no worker/V-cycle publication in T-049",
        "memory_scope":"caller texture payloads and all prepared operator scratch/uniform/stencil buffers; driver allocations excluded",
        "environment":"native headless DX12; serial submissions and waits; desktop GPU activity not isolated"});
    let gpu_errors = if validation.is_empty() {
        Ok(())
    } else {
        Err(validation.join("\n"))
    };
    if let Err(error) = validate_completed_report(&report, gpu_errors)
        .and_then(|_| report_contract::validate_poisson_report(&report))
    {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}

pub(crate) fn run_registered(
    context: &crate::SuiteContext<'_>,
) -> std::result::Result<serde_json::Value, String> {
    run(
        context.device.clone(),
        context.queue.clone(),
        context.extent,
        context.info,
    )
}
