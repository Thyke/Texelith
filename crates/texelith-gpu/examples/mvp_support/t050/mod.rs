//! Complete solver calls: setup, every V-cycle, reductions and completion waits.
#[allow(dead_code)]
#[path = "../../../tests/mvp_t050/gpu.rs"]
mod gpu;
mod interactive;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t050/oracle.rs"]
mod oracle;
#[allow(dead_code)]
#[path = "../../../tests/mvp_t050/proxy_oracle.rs"]
mod proxy_oracle;
mod report_contract;
use crate::contract::{aggregate_timings, validate_completed_report};
use oracle::{Grid, analytic, mean, norm, normals, project, residual, rms, slopes};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use texelith_core::{Boundary, Extent, InputInterpretation, NormalConvention, SourceDescriptor};
use texelith_gpu::{PoissonParameters, PoissonSolver, PoissonStop, pool::ResourcePool};
type Result<T> = std::result::Result<T, String>;
const BUDGET: u64 = 1024 * 1024 * 1024;
const SCRATCH: u64 = 192 * 1024 * 1024;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn solve(
    solver: &PoissonSolver,
    n: &wgpu::Texture,
    h: &wgpu::Texture,
    dx: bool,
    p: PoissonParameters,
) -> Result<texelith_gpu::PoissonSolveReport> {
    solver
        .solve(
            n,
            h,
            if dx {
                NormalConvention::DirectX
            } else {
                NormalConvention::OpenGl
            },
            p,
            || true,
        )
        .map_err(|e| e.to_string())
}
fn numerics(gpu: &gpu::Gpu) -> Result<Vec<Value>> {
    let pool = Arc::new(Mutex::new(ResourcePool::new(SCRATCH)));
    let solver = PoissonSolver::new(gpu.device.clone(), gpu.queue.clone(), pool.clone())
        .map_err(|e| e.to_string())?;
    let mut checks = Vec::new();
    for (w, h) in [(1, 1), (1, 31), (29, 1), (32, 16), (17, 13), (257, 193)] {
        for repeat in [true, false] {
            for dx in [false, true] {
                eprintln!("Poisson CPU oracle: {w}x{h}, repeat={repeat}, DX={dx}");
                let g = Grid { w, h, repeat };
                let expected = analytic(g);
                let (normal, q) = gpu.normal(g, &normals(&g.gradient(&expected), dx));
                let (gradient, count) = slopes(&q, dx);
                let mut b = g.transpose(&gradient);
                project(&mut b);
                let output = gpu.scalar(g, 7.0);
                let p = PoissonParameters {
                    boundary: if repeat {
                        Boundary::Repeat
                    } else {
                        Boundary::Clamp
                    },
                    ..Default::default()
                };
                let r = solve(&solver, &normal, &output, dx, p.clone())?;
                let actual = gpu.read(&output);
                let measured = residual(g, &actual, &b);
                require(
                    r.stop == PoissonStop::Converged,
                    &format!("nonconvergence {w}x{h}: {r:?}"),
                )?;
                require(
                    rms(&actual, &expected) <= 1e-3 && mean(&actual).abs() <= 2e-5,
                    "height recovery/gauge",
                )?;
                require(
                    (f64::from(r.residual_l2) - measured).abs() <= 2e-5 * norm(&b).max(1e-6),
                    "GPU/CPU residual disagreement",
                )?;
                require(
                    measured <= f64::from(p.absolute_tolerance)
                        || measured <= f64::from(p.relative_tolerance) * norm(&b)
                        || (repeat
                            && w.is_power_of_two()
                            && h.is_power_of_two()
                            && oracle::optimality_gap(g, &actual, &oracle::periodic_fft(g, &b))
                                <= f64::from(p.relative_tolerance) * oracle::field_norm(&gradient)),
                    "false convergence",
                )?;
                require(
                    r.clamped_pixels == count && r.scalar_readbacks <= r.cycles + 3,
                    "diagnostics/readback bounds",
                )?;
                if repeat && g.len() <= 1024 {
                    require(
                        rms(&actual, &oracle::periodic_spectral(g, &b)) <= 2e-5,
                        "independent spectral oracle",
                    )?;
                }
                checks.push(json!({"name":format!("cpu_oracle_{w}x{h}_repeat{repeat}_dx{dx}"),"passed":true,"rms":rms(&actual,&expected),"cycles":r.cycles,"rhs_l2":r.rhs_l2,"residual_l2":r.residual_l2,"clamped_pixels":r.clamped_pixels}));
            }
        }
    }
    let g = Grid {
        w: 16,
        h: 8,
        repeat: true,
    };
    let mut n = normals(
        &[
            oracle::seeded(g.len()),
            g.sample(|u, v| 0.2 * (9.0 * u + v).sin()),
        ],
        false,
    );
    n[0] = [1.0, 0.0, 0.0];
    n[1] = [0.0, -1.0, 0.0];
    let (normal, q) = gpu.normal(g, &n);
    let (gradient, count) = slopes(&q, false);
    let mut b = g.transpose(&gradient);
    project(&mut b);
    let output = gpu.scalar(g, 9.0);
    let r = solve(
        &solver,
        &normal,
        &output,
        false,
        PoissonParameters::default(),
    )?;
    require(
        r.stop == PoissonStop::Converged
            && r.clamped_pixels == count
            && count == 2
            && rms(&gpu.read(&output), &oracle::periodic_spectral(g, &b)) <= 1e-3,
        "nonintegrable least squares/clamp diagnostics",
    )?;
    let optimum = oracle::periodic_spectral(g, &b);
    let optimum_residual = oracle::objective_residual(g, &optimum, &gradient);
    let actual = gpu.read(&output);
    let actual_residual = oracle::objective_residual(g, &actual, &gradient);
    let gap = oracle::optimality_gap(g, &actual, &optimum);
    let tolerance = 1e-3 * oracle::field_norm(&gradient);
    require(
        optimum_residual > 1e-3
            && gap <= tolerance
            && actual_residual <= optimum_residual.hypot(tolerance),
        "nonintegrable optimum residual",
    )?;
    require(
        oracle::optimality_gap(g, &vec![0.0; g.len()], &optimum) > tolerance,
        "negative control must reject zero height",
    )?;
    checks.push(json!({"name":"nonintegrable_least_squares","passed":true,"normal_equation_residual_l2":r.residual_l2,"optimum_residual_l2":optimum_residual,"objective_residual_l2":actual_residual,"optimality_gap_l2":gap,"tolerance":tolerance,"clamped_pixels":r.clamped_pixels}));
    let g = Grid {
        w: 65,
        h: 47,
        repeat: true,
    };
    let (normal, _) = gpu.normal(g, &normals(&g.gradient(&oracle::seeded(g.len())), false));
    let output = gpu.scalar(g, 0.375);
    for cycles in [1, 16] {
        let p = PoissonParameters {
            v_cycles: cycles,
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-12,
            ..Default::default()
        };
        let r = solve(&solver, &normal, &output, false, p)?;
        require(
            r.stop == PoissonStop::MaxCycles && r.cycles == cycles,
            "iteration cap must report nonconvergence",
        )?;
        checks.push(json!({"name":format!("max_cycles_{cycles}"),"passed":true,"cycles":r.cycles,"residual_l2":r.residual_l2}));
    }
    let before = gpu.bytes(&output);
    let r = solver
        .solve(
            &normal,
            &output,
            NormalConvention::OpenGl,
            PoissonParameters::default(),
            || false,
        )
        .map_err(|e| e.to_string())?;
    require(
        r.stop == PoissonStop::Cancelled && before == gpu.bytes(&output),
        "entry cancellation mutated output",
    )?;
    let mut polls = 0;
    let r = solver
        .solve(
            &normal,
            &output,
            NormalConvention::OpenGl,
            PoissonParameters::default(),
            || {
                polls += 1;
                polls <= 80
            },
        )
        .map_err(|e| e.to_string())?;
    require(
        r.stop == PoissonStop::Cancelled && polls == 81 && r.cycles == 0,
        "in-cycle cancellation",
    )?;
    checks.push(json!({"name":"cancellation_after_submitted_batches","passed":true,"polls":polls}));
    let tiny_pool = Arc::new(Mutex::new(ResourcePool::new(1)));
    let tiny = PoissonSolver::new(gpu.device.clone(), gpu.queue.clone(), tiny_pool.clone())
        .map_err(|e| e.to_string())?;
    let before = gpu.bytes(&output);
    require(
        solve(&tiny, &normal, &output, false, PoissonParameters::default()).is_err()
            && before == gpu.bytes(&output)
            && tiny_pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "atomic budget rejection",
    )?;
    checks.push(json!({"name":"atomic_budget_rejection","passed":true}));
    let (normal, _) = gpu.normal(g, &vec![[0.0, 0.0, 1.0]; g.len()]);
    let r = solve(
        &solver,
        &normal,
        &output,
        false,
        PoissonParameters::default(),
    )?;
    require(
        r.stop == PoissonStop::Converged
            && r.cycles <= 1
            && gpu.read(&output).iter().all(|v| *v == 0.0),
        "exact flat recovery after cancellation",
    )?;
    checks.push(json!({"name":"flat_exact_zero_and_recovery","passed":true}));
    require(
        pool.lock().unwrap().diagnostics().peak_bytes <= SCRATCH,
        "numerical scratch budget",
    )?;
    Ok(checks)
}
fn timing(gpu: &gpu::Gpu, e: Extent, cycle_budget: u32) -> Result<Value> {
    eprintln!(
        "Poisson {cycle_budget}-cycle budget: generating {}x{} reference",
        e.width, e.height
    );
    let g = Grid {
        w: e.width as usize,
        h: e.height as usize,
        repeat: true,
    };
    // The same low-frequency periodic height is used at every benchmark size.
    let height = analytic(g);
    let (normal, quantized) = gpu.normal(g, &normals(&g.gradient(&height), false));
    eprintln!("Poisson {cycle_budget}-cycle budget: normal fixture uploaded");
    let gradient = slopes(&quantized, false).0;
    let mut rhs = g.transpose(&gradient);
    drop(quantized);
    project(&mut rhs);
    let rhs_l2 = norm(&rhs);
    eprintln!("Poisson: computing independent f64 CPU optimum");
    let optimum = oracle::periodic_fft(g, &rhs);
    let optimum_residual = oracle::objective_residual(g, &optimum, &gradient);
    let gradient_l2 = oracle::field_norm(&gradient);
    let output = gpu.scalar(g, 0.0);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let solver = PoissonSolver::new(gpu.device.clone(), gpu.queue.clone(), pool.clone())
        .map_err(|e| e.to_string())?;
    let p = PoissonParameters {
        v_cycles: cycle_budget,
        ..Default::default()
    };
    let mut rows = Vec::new();
    let mut cycles = Vec::new();
    let mut residuals = Vec::new();
    let mut nonconverged = 0;
    let mut submission = 0.0;
    let mut initialization_ms = 0.0;
    let mut residual_history = Vec::new();
    let mut gpu_gap = None;
    eprintln!("Poisson {cycle_budget}-cycle budget: starting complete solves");
    for i in 0..110 {
        let start = Instant::now();
        let r = solve(&solver, &normal, &output, false, p.clone())?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if i == 0 {
            initialization_ms = elapsed;
            eprintln!(
                "Poisson {cycle_budget}-cycle budget: first complete solve {elapsed:.3} ms, stop {:?}, relative residual {}",
                r.stop,
                r.residual_l2 / r.rhs_l2.max(1e-30)
            );
        }
        rows.push(vec![elapsed]);
        if i >= 10 {
            cycles.push(r.cycles);
            residuals.push(r.residual_l2);
            nonconverged += u32::from(r.stop != PoissonStop::Converged);
            submission += r.submission_ms;
            residual_history = r.residual_history;
            gpu_gap = r.optimality_gap_l2;
        }
        if i % 25 == 24 {
            eprintln!(
                "Poisson {cycle_budget}-cycle budget: {}/110 complete solves",
                i + 1
            );
        }
    }
    let timing = aggregate_timings(&rows, 10)?;
    let scratch = pool.lock().unwrap().diagnostics().peak_bytes;
    let total = scratch + g.len() as u64 * 12;
    let actual = gpu.read(&output);
    let measured_residual = residual(g, &actual, &rhs);
    let relative_residual = measured_residual / rhs_l2.max(1e-30);
    let objective_residual = oracle::objective_residual(g, &actual, &gradient);
    let optimality_gap = oracle::optimality_gap(g, &actual, &optimum);
    let relative_optimality_gap = optimality_gap / gradient_l2.max(1e-30);
    let gpu_gap = gpu_gap.ok_or("missing GPU optimality certificate")?;
    require(
        (f64::from(gpu_gap) - optimality_gap).abs() <= 2e-5 * gradient_l2.max(1e-6),
        "GPU certificate disagrees with independent CPU optimum",
    )?;
    require(
        oracle::optimality_gap(g, &vec![0.0; g.len()], &optimum) > 1e-3 * gradient_l2,
        "full-size negative control must reject zero output",
    )?;
    require(
        objective_residual <= optimum_residual.hypot(1e-3 * gradient_l2),
        "full-size objective exceeds least-squares optimum tolerance",
    )?;
    Ok(
        json!({"size":e.width,"objective_residual_l2":objective_residual,"optimum_residual_l2":optimum_residual,"optimality_gap_l2":optimality_gap,"gpu_optimality_gap_l2":gpu_gap,"relative_optimality_gap":relative_optimality_gap,"gradient_l2":gradient_l2,"kind":"completion_upper_bound","warmups":10,"measurements":timing.measurements,"p95_ms":timing.p95_ms,"submission_ms":submission/100.0,"peak_bytes":total,"scratch_peak_bytes":scratch,"cancellation_errors":0,"nonconverged_samples":nonconverged,"cycles":cycles,"residual_l2":residuals,"last_residual_history":residual_history,"rhs_l2":rhs_l2,"measured_residual_l2":measured_residual,"relative_residual":relative_residual,"initialization_ms":initialization_ms,"rms":rms(&actual,&height),"v_cycles":p.v_cycles,"scope":"complete solve, fresh normal-to-RHS, forward FFT, eigenvalue division, inverse FFT, zero DC, R32 write, physical residual and spectral optimality certificate reductions/readback, every requested refinement until convergence; no warm output cache","warmup_scope":"first call includes spectral pipeline/workspace preparation","samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()}),
    )
}
pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value> {
    let gpu = gpu::Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
    };
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = errors.clone();
    gpu.device
        .on_uncaptured_error(Arc::new(move |e| sink.lock().unwrap().push(e.to_string())));
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut checks = numerics(&gpu)?;
    eprintln!("Poisson: {} numerical checks completed", checks.len());
    let default = timing(&gpu, context.extent, 8)?;
    let maximum = timing(&gpu, context.extent, 16)?;
    let proxy = timing(&gpu, Extent::new(1024, 1024).map_err(|e| e.to_string())?, 8)?;
    let worker_proxy = interactive::measure(&gpu)?;
    gpu.wait();
    if let Some(error) = pollster::block_on(scope.pop()) {
        errors.lock().unwrap().push(error.to_string());
    }
    for (name, passed) in [
        (
            "interactive_worker_4k_to_1k_p95",
            worker_proxy["p95_ms"].as_f64().is_some_and(|v| v < 16.0),
        ),
        ("default_convergence", default["nonconverged_samples"] == 0),
        (
            "full_size_cpu_optimality",
            default["relative_optimality_gap"]
                .as_f64()
                .is_some_and(|r| r <= 1e-3)
                && maximum["relative_optimality_gap"]
                    .as_f64()
                    .is_some_and(|r| r <= 1e-3),
        ),
        (
            "full_size_height_rms",
            default["rms"].as_f64().is_some_and(|r| r <= 1e-3)
                && maximum["rms"].as_f64().is_some_and(|r| r <= 1e-3),
        ),
        (
            "maximum_quality_convergence",
            maximum["nonconverged_samples"] == 0,
        ),
        (
            "scratch_192_mib",
            default["scratch_peak_bytes"]
                .as_u64()
                .is_some_and(|n| n <= SCRATCH)
                && maximum["scratch_peak_bytes"]
                    .as_u64()
                    .is_some_and(|n| n <= SCRATCH),
        ),
        (
            "application_budget",
            default["peak_bytes"].as_u64().is_some_and(|n| n <= BUDGET)
                && maximum["peak_bytes"].as_u64().is_some_and(|n| n <= BUDGET),
        ),
        (
            "default_4k_p95",
            default["p95_ms"].as_f64().is_some_and(|n| n <= 100.0),
        ),
        (
            "interactive_1k_p95",
            proxy["p95_ms"].as_f64().is_some_and(|n| n < 16.0),
        ),
        (
            "interactive_1k_optimum",
            proxy["relative_optimality_gap"]
                .as_f64()
                .is_some_and(|n| n <= 1e-3)
                && proxy["nonconverged_samples"] == 0,
        ),
    ] {
        checks.push(json!({"name":name,"passed":passed}));
    }
    let manifest = json!({"boundary":"repeat","v_cycles":8,"pre_smooth":3,"post_smooth":3,"coarsest_iterations":32,"relative_tolerance":0.001,"absolute_tolerance":0.000001,"maximum_v_cycles":16,"fixture":"zero-mean 0.04*cos(2*pi*u)+0.025*cos(4*pi*v), forward UV differences, binary16 normal upload","size":[context.extent.width,context.extent.height],"algorithm_revision":3,"convergence":"normal equation tolerance OR sqrt(F-F_optimum)/norm(g) <= relative_tolerance","full_resolution_p95_limit_ms":100,"interactive_proxy_size":1024,"interactive_p95_exclusive_limit_ms":16,"interactive_worker":{"source_size":4096,"source_format":"RGB8 encoded normal","proxy_filter":"area average of decoded unit normals, normalized, nearest binary16","pre_smooth_sequence":[3,4],"cache_policy":"alternate inactive spectral smoother control to force fresh output; zero result-cache hits required"},"algorithm":"periodic POT in-place f32 radix-2 FFT with bounded residual correction; Galerkin multigrid for NPOT and Neumann"});
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
    let validation = errors.lock().unwrap().clone();
    let mut report = json!({"passed":true,"suite":"poisson","size":context.extent.width,"seed":50,"adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver},"manifest":manifest,"manifest_hash":hash,"checks":checks,"timing":default,"maximum_quality":maximum,"interactive_proxy":proxy,"interactive_worker":worker_proxy,"validation_errors":validation,"memory_scope":"caller normal/output payload plus all solver/operator textures, buffers, twiddles, uniforms, stencils and readback; excludes driver allocations","environment":"serial headless DX12; desktop GPU activity not isolated"});
    let gpu_errors = if validation.is_empty() {
        Ok(())
    } else {
        Err(validation.join("\n"))
    };
    if let Err(error) =
        report_contract::validate(&report, |view| validate_completed_report(view, gpu_errors))
    {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
