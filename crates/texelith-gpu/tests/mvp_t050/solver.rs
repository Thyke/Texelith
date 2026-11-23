use super::{
    gpu::{Gpu, with_gpu},
    oracle::{self as cpu, Grid},
};
use std::sync::{Arc, Mutex};
use texelith_core::{Boundary, NormalConvention};
use texelith_gpu::{
    PoissonParameters, PoissonSolveReport, PoissonSolver, PoissonStop, pool::ResourcePool,
};

const SCRATCH: u64 = 192 * 1024 * 1024;
fn parameters(g: Grid) -> PoissonParameters {
    PoissonParameters {
        boundary: if g.repeat {
            Boundary::Repeat
        } else {
            Boundary::Clamp
        },
        v_cycles: 8,
        pre_smooth: 3,
        post_smooth: 3,
        coarsest_iterations: 32,
        relative_tolerance: 1e-3,
        absolute_tolerance: 1e-6,
    }
}
fn solver(gpu: &Gpu, budget: u64) -> (PoissonSolver, Arc<Mutex<ResourcePool>>) {
    let pool = Arc::new(Mutex::new(ResourcePool::new(budget)));
    let solver = PoissonSolver::new(gpu.device.clone(), gpu.queue.clone(), pool.clone()).unwrap();
    (solver, pool)
}
fn check_report(
    g: Grid,
    actual: &[f64],
    rhs: &[f64],
    r: &PoissonSolveReport,
    p: &PoissonParameters,
) {
    assert!(actual.iter().all(|v| v.is_finite()));
    assert!(r.cycles <= p.v_cycles && r.cycles <= 16);
    assert_eq!(r.residual_history.len(), r.cycles as usize);
    assert!(
        r.residual_history
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    let measured = cpu::residual(g, actual, rhs);
    let rhs_norm = cpu::norm(rhs);
    let scale = rhs_norm.max(1e-6);
    assert!(
        (f64::from(r.rhs_l2) - rhs_norm).abs() <= 2e-5 * scale,
        "wrong RHS L2"
    );
    assert!(
        (f64::from(r.residual_l2) - measured).abs() <= 2e-5 * scale,
        "reported {} != independently measured {measured}",
        r.residual_l2
    );
    if let Some(last) = r.residual_history.last() {
        assert!((last - r.residual_l2).abs() <= f32::EPSILON * r.residual_l2.abs().max(1.0));
    }
    assert!(cpu::mean(actual).abs() <= 2e-5, "height gauge drift");
    assert!(
        r.scalar_readbacks <= r.cycles + 3,
        "only one residual scalar per completed cycle plus bounded setup reads"
    );
    if r.stop == PoissonStop::Converged {
        let certified = if let (Some(gap), Some(gradient)) = (r.optimality_gap_l2, r.gradient_l2) {
            let optimum = cpu::periodic_fft(g, rhs);
            let measured_gap = cpu::optimality_gap(g, actual, &optimum);
            // f32 slopes/FFT/reductions versus the independent f64 oracle.
            assert!((measured_gap - f64::from(gap)).abs() <= 2e-5 * f64::from(gradient).max(1e-6));
            measured_gap <= f64::from(p.relative_tolerance) * f64::from(gradient)
        } else {
            false
        };
        assert!(
            measured <= f64::from(p.absolute_tolerance)
                || measured <= f64::from(p.relative_tolerance) * rhs_norm
                || certified,
            "false convergence: residual={measured}, RHS={rhs_norm}"
        );
    }
}
fn recover(gpu: &Gpu, g: Grid, height: Vec<f64>, dx: bool) {
    let (solver, pool) = solver(gpu, SCRATCH);
    let (normal, quantized) = gpu.normal(g, &cpu::normals(&g.gradient(&height), dx));
    let (gradients, clamped) = cpu::slopes(&quantized, dx);
    let mut rhs = g.transpose(&gradients);
    cpu::project(&mut rhs);
    let target = gpu.scalar(g, 1234.0);
    let p = parameters(g);
    let convention = if dx {
        NormalConvention::DirectX
    } else {
        NormalConvention::OpenGl
    };
    let r = solver
        .solve(&normal, &target, convention, p.clone(), || true)
        .unwrap();
    assert_eq!(r.stop, PoissonStop::Converged, "{g:?}: {r:?}");
    assert_eq!(r.clamped_pixels, clamped);
    assert_eq!(target.format(), wgpu::TextureFormat::R32Float);
    let actual = gpu.read(&target);
    check_report(g, &actual, &rhs, &r, &p);
    assert!(
        cpu::rms(&actual, &height) <= 1e-3,
        "{g:?}: height RMS={}",
        cpu::rms(&actual, &height)
    );
    if g.repeat && g.len() <= 1024 {
        let exact = cpu::periodic_spectral(g, &rhs);
        assert!(
            cpu::rms(&actual, &exact) <= 2e-5,
            "CPU least-squares oracle disagreement"
        );
        let (mg, _) = cpu::Multigrid::new(g).solve(&rhs, 8);
        assert!(
            cpu::rms(&actual, &mg) <= 2e-5,
            "CPU multigrid oracle disagreement"
        );
    }
    assert!(pool.lock().unwrap().diagnostics().peak_bytes <= SCRATCH);
    if cpu::norm(&rhs) > 1e-6 {
        assert!(
            pool.lock().unwrap().diagnostics().peak_bytes > 0,
            "solver scratch must be charged to the supplied shared ledger"
        );
    }
    // solve returns only after submitted work completes; all transient owners
    // must then be evictable from this same ledger, not a private hidden pool.
    drop(solver);
    let pressure = pool
        .lock()
        .unwrap()
        .allocate(texelith_gpu::pool::AllocationDesc::Buffer { bytes: SCRATCH })
        .expect("completed scratch must be evictable after dropping the solver");
    assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, SCRATCH);
    pool.lock().unwrap().release(pressure).unwrap();
}

#[test]
fn periodic_sinusoid_and_neumann_cosine_recover_signed_zero_mean_height() {
    with_gpu(|gpu| {
        for repeat in [true, false] {
            for dx in [false, true] {
                let g = Grid {
                    w: 32,
                    h: 16,
                    repeat,
                };
                recover(gpu, g, cpu::analytic(g), dx);
            }
        }
    });
}
#[test]
fn npot_257_by_193_converges_without_resizing_or_losing_low_frequency_mode() {
    with_gpu(|gpu| {
        for repeat in [true, false] {
            let g = Grid {
                w: 257,
                h: 193,
                repeat,
            };
            recover(gpu, g, cpu::analytic(g), false);
        }
    });
}
#[test]
fn ramps_impulses_and_singleton_axes_preserve_boundary_and_precision_contracts() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 1), (1, 31), (29, 1)] {
            for repeat in [false, true] {
                let g = Grid { w, h, repeat };
                recover(gpu, g, cpu::analytic(g), false);
            }
        }
        let g = Grid {
            w: 17,
            h: 13,
            repeat: false,
        };
        let mut ramp = g.sample(|u, v| 0.08 * u - 0.04 * v);
        cpu::project(&mut ramp);
        recover(gpu, g, ramp, false);
        let mut impulse = vec![0.0; g.len()];
        impulse[6 * g.w + 8] = 0.004;
        cpu::project(&mut impulse);
        recover(gpu, g, impulse, false);
    });
}
#[test]
fn flat_zero_rhs_is_exact_and_near_zero_rhs_uses_absolute_residual() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 33,
            h: 19,
            repeat: true,
        };
        let (solver, _) = solver(gpu, SCRATCH);
        for slope in [0.0, 2f64.powi(-24)] {
            let n = cpu::normals(&[vec![slope; g.len()], vec![0.0; g.len()]], false);
            let (normal, q) = gpu.normal(g, &n);
            let output = gpu.scalar(g, 42.0);
            let p = parameters(g);
            let r = solver
                .solve(
                    &normal,
                    &output,
                    NormalConvention::OpenGl,
                    p.clone(),
                    || true,
                )
                .unwrap();
            assert_eq!(r.stop, PoissonStop::Converged);
            assert!(
                r.cycles <= 1,
                "zero RHS must not run a full eight-cycle solve"
            );
            let actual = gpu.read(&output);
            assert!(
                actual.iter().all(|v| *v == 0.0),
                "periodic constant slope has zero divergence"
            );
            check_report(g, &actual, &g.transpose(&cpu::slopes(&q, false).0), &r, &p);
        }
        let g = Grid {
            w: 17,
            h: 13,
            repeat: false,
        };
        let small: Vec<_> = cpu::analytic(g).iter().map(|v| v * 2e-6).collect();
        let (normal, q) = gpu.normal(g, &cpu::normals(&g.gradient(&small), false));
        let mut rhs = g.transpose(&cpu::slopes(&q, false).0);
        cpu::project(&mut rhs);
        assert!(
            cpu::norm(&rhs) > 1e-6,
            "fixture must exercise a nonzero RHS"
        );
        let output = gpu.scalar(g, 99.0);
        let mut p = parameters(g);
        p.relative_tolerance = 1e-12;
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                p.clone(),
                || true,
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Converged);
        check_report(g, &gpu.read(&output), &rhs, &r, &p);
        assert!(r.residual_l2 <= 1e-6);
    });
}
#[test]
fn nonintegrable_field_solves_least_squares_and_counts_steep_pixels_once() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 16,
            h: 8,
            repeat: true,
        };
        let field = [
            cpu::seeded(g.len()),
            g.sample(|u, v| 0.2 * (9.0 * u + v).sin()),
        ];
        let mut n = cpu::normals(&field, false);
        n[0] = [1.0, 0.0, 0.0];
        n[1] = [0.0, -1.0, 0.0];
        let (normal, q) = gpu.normal(g, &n);
        let (gradients, count) = cpu::slopes(&q, false);
        assert_eq!(count, 2);
        let mut rhs = g.transpose(&gradients);
        cpu::project(&mut rhs);
        let (solver, _) = solver(gpu, SCRATCH);
        let output = gpu.scalar(g, 11.0);
        let p = parameters(g);
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                p.clone(),
                || true,
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Converged);
        assert_eq!(r.clamped_pixels, count);
        let actual = gpu.read(&output);
        check_report(g, &actual, &rhs, &r, &p);
        let optimum = cpu::periodic_spectral(g, &rhs);
        assert!(cpu::rms(&actual, &optimum) <= 1e-3);
        let optimum_residual = cpu::objective_residual(g, &optimum, &gradients);
        let actual_residual = cpu::objective_residual(g, &actual, &gradients);
        let tolerance = f64::from(p.relative_tolerance) * cpu::field_norm(&gradients);
        assert!(optimum_residual > 1e-3, "nonintegrable optimum is nonzero");
        assert!(actual_residual <= optimum_residual.hypot(tolerance));
        assert!(cpu::optimality_gap(g, &actual, &optimum) <= tolerance);
    });
}
#[test]
fn one_cycle_failure_reports_actual_residual_and_sixteen_is_a_hard_cap() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 65,
            h: 47,
            repeat: true,
        };
        let (solver, _) = solver(gpu, SCRATCH);
        let n = cpu::normals(&g.gradient(&cpu::seeded(g.len())), false);
        let (normal, q) = gpu.normal(g, &n);
        let mut rhs = g.transpose(&cpu::slopes(&q, false).0);
        cpu::project(&mut rhs);
        for cycles in [1, 16] {
            let output = gpu.scalar(g, -17.0);
            let mut p = parameters(g);
            p.v_cycles = cycles;
            p.relative_tolerance = 1e-12;
            p.absolute_tolerance = 1e-12;
            let r = solver
                .solve(
                    &normal,
                    &output,
                    NormalConvention::OpenGl,
                    p.clone(),
                    || true,
                )
                .unwrap();
            assert_eq!(
                r.stop,
                PoissonStop::MaxCycles,
                "nonconvergence must be explicit"
            );
            assert_eq!(r.cycles, cycles);
            check_report(g, &gpu.read(&output), &rhs, &r, &p);
            assert!(r.residual_l2 > p.absolute_tolerance);
        }
        for cycles in [0, 17, u32::MAX] {
            let mut p = parameters(g);
            p.v_cycles = cycles;
            let output = gpu.scalar(g, 13.0);
            assert!(
                solver
                    .solve(&normal, &output, NormalConvention::OpenGl, p, || true)
                    .is_err()
            );
            assert!(gpu.read(&output).iter().all(|v| *v == 13.0));
        }
    });
}
#[test]
fn repeat_solves_are_deterministic_reset_initial_guess_and_fix_only_observable_height() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 32,
            h: 16,
            repeat: true,
        };
        let height = cpu::analytic(g);
        let shifted: Vec<_> = height.iter().map(|v| v + 8.0).collect();
        let (a, _) = gpu.normal(g, &cpu::normals(&g.gradient(&height), false));
        let (b, _) = gpu.normal(g, &cpu::normals(&g.gradient(&shifted), false));
        let (solver, _) = solver(gpu, SCRATCH);
        let output = gpu.scalar(g, 999.0);
        let mut previous = None;
        for (n, cycles) in [(&a, 8), (&b, 8), (&a, 16), (&a, 8)] {
            let mut p = parameters(g);
            p.v_cycles = cycles;
            let r = solver
                .solve(n, &output, NormalConvention::OpenGl, p, || true)
                .unwrap();
            assert_eq!(r.stop, PoissonStop::Converged);
            assert!(cpu::mean(&gpu.read(&output)).abs() < 2e-5);
            let bytes = gpu.bytes(&output);
            if let Some(previous) = previous {
                assert_eq!(
                    bytes, previous,
                    "offset, output reuse or extra budget changed result"
                );
            }
            previous = Some(bytes);
        }
    });
}
#[test]
fn cancellation_is_checked_before_work_and_between_passes_then_solver_recovers() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 65,
            h: 47,
            repeat: true,
        };
        let (solver, pool) = solver(gpu, SCRATCH);
        let (normal, _) = gpu.normal(g, &cpu::normals(&g.gradient(&cpu::analytic(g)), false));
        let output = gpu.scalar(g, 5.0);
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                parameters(g),
                || false,
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Cancelled);
        assert_eq!(r.cycles, 0);
        assert!(gpu.read(&output).iter().all(|v| *v == 5.0));
        let mut polls = 0;
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                parameters(g),
                || {
                    polls += 1;
                    polls <= 3
                },
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Cancelled);
        assert_eq!(polls, 4, "stop promptly when token becomes stale");
        assert_eq!(
            r.cycles, 0,
            "cancellation must be checked within the first V-cycle"
        );
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                parameters(g),
                || true,
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Converged);
        assert!(cpu::rms(&gpu.read(&output), &cpu::analytic(g)) < 1e-3);
        assert!(pool.lock().unwrap().diagnostics().peak_bytes <= SCRATCH);
    });
}
#[test]
fn insufficient_shared_budget_is_atomic_and_never_returns_a_success_report() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 65,
            h: 47,
            repeat: true,
        };
        let (normal, _) = gpu.normal(g, &cpu::normals(&g.gradient(&cpu::analytic(g)), false));
        let output = gpu.scalar(g, 0.375);
        let (solver, pool) = solver(gpu, 1);
        assert!(
            solver
                .solve(
                    &normal,
                    &output,
                    NormalConvention::OpenGl,
                    parameters(g),
                    || true
                )
                .is_err()
        );
        assert!(gpu.read(&output).iter().all(|v| *v == 0.375));
        assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
        assert!(pool.lock().unwrap().diagnostics().peak_bytes <= 1);
    });
}

#[test]
fn invalid_parameters_and_operand_extents_fail_before_output_mutation() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 17,
            h: 13,
            repeat: true,
        };
        let (solver, _) = solver(gpu, SCRATCH);
        let (normal, _) = gpu.normal(g, &vec![[0.0, 0.0, 1.0]; g.len()]);
        let output = gpu.scalar(g, 0.75);
        let mut invalid = Vec::new();
        let mut p = parameters(g);
        p.pre_smooth = 0;
        invalid.push(p);
        let mut p = parameters(g);
        p.post_smooth = 0;
        invalid.push(p);
        let mut p = parameters(g);
        p.coarsest_iterations = 0;
        invalid.push(p);
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut p = parameters(g);
            p.relative_tolerance = value;
            invalid.push(p);
            let mut p = parameters(g);
            p.absolute_tolerance = value;
            invalid.push(p);
        }
        for p in invalid {
            assert!(
                solver
                    .solve(&normal, &output, NormalConvention::OpenGl, p, || true)
                    .is_err()
            );
            assert!(gpu.read(&output).iter().all(|v| *v == 0.75));
        }
        let wrong = gpu.scalar(Grid { w: 13, h: 17, ..g }, 0.75);
        assert!(
            solver
                .solve(
                    &normal,
                    &wrong,
                    NormalConvention::OpenGl,
                    parameters(g),
                    || true
                )
                .is_err()
        );
        assert!(gpu.read(&wrong).iter().all(|v| *v == 0.75));
        assert!(
            solver
                .solve(
                    &output,
                    &output,
                    NormalConvention::OpenGl,
                    parameters(g),
                    || true
                )
                .is_err()
        );
        assert!(gpu.read(&output).iter().all(|v| *v == 0.75));
    });
}

#[test]
fn scratch_peak_does_not_scale_with_requested_v_cycle_count() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 65,
            h: 47,
            repeat: true,
        };
        let (normal, _) = gpu.normal(g, &cpu::normals(&g.gradient(&cpu::seeded(g.len())), false));
        let mut peaks = Vec::new();
        for cycles in [1, 16] {
            let (solver, pool) = solver(gpu, SCRATCH);
            let output = gpu.scalar(g, 0.0);
            let mut p = parameters(g);
            p.v_cycles = cycles;
            p.relative_tolerance = 1e-12;
            p.absolute_tolerance = 1e-12;
            let r = solver
                .solve(&normal, &output, NormalConvention::OpenGl, p, || true)
                .unwrap();
            assert_eq!(r.cycles, cycles);
            assert_eq!(r.stop, PoissonStop::MaxCycles);
            let peak = pool.lock().unwrap().diagnostics().peak_bytes;
            assert!(
                peak >= g.len() as u64 * 4 && peak <= SCRATCH,
                "uncharged or excessive scratch"
            );
            peaks.push(peak);
        }
        assert!(
            peaks[1] <= peaks[0] + 4096,
            "V-cycle workspace must be reused: {peaks:?}"
        );
    });
}

#[test]
fn spectral_split_axes_recover_high_frequency_modes_with_independent_residual() {
    with_gpu(|gpu| {
        for n in [2, 4, 8, 32, 64] {
            let g = Grid {
                w: n,
                h: n,
                repeat: true,
            };
            recover(gpu, g, cpu::analytic(g), false);
        }
        for (w, h) in [(2048, 1), (1, 2048), (4096, 8), (8, 4096)] {
            let g = Grid { w, h, repeat: true };
            let mut height = g.sample(|u, v| {
                0.0001 * (std::f64::consts::TAU * 137.0 * u).cos()
                    + 0.00007 * (std::f64::consts::TAU * 91.0 * v).sin()
            });
            cpu::project(&mut height);
            recover(gpu, g, height, false);
        }
    });
}

#[test]
fn spectral_path_cancels_between_passes_and_reports_strict_tolerance_exhaustion() {
    with_gpu(|gpu| {
        let g = Grid {
            w: 32,
            h: 16,
            repeat: true,
        };
        let (solver, _) = solver(gpu, SCRATCH);
        let (normal, q) = gpu.normal(g, &cpu::normals(&g.gradient(&cpu::seeded(g.len())), false));
        let mut rhs = g.transpose(&cpu::slopes(&q, false).0);
        cpu::project(&mut rhs);
        let output = gpu.scalar(g, 13.0);
        let mut polls = 0;
        let r = solver
            .solve(
                &normal,
                &output,
                NormalConvention::OpenGl,
                parameters(g),
                || {
                    polls += 1;
                    polls <= 3
                },
            )
            .unwrap();
        assert_eq!(r.stop, PoissonStop::Cancelled);
        assert_eq!(polls, 4);
        assert_eq!(r.cycles, 0);
        assert!(gpu.read(&output).iter().all(|v| *v == 13.0));
        for cycles in [1, 16] {
            let mut p = parameters(g);
            p.v_cycles = cycles;
            p.relative_tolerance = 1e-12;
            p.absolute_tolerance = 1e-12;
            let r = solver
                .solve(
                    &normal,
                    &output,
                    NormalConvention::OpenGl,
                    p.clone(),
                    || true,
                )
                .unwrap();
            assert_eq!(r.stop, PoissonStop::MaxCycles);
            assert_eq!(r.cycles, cycles);
            check_report(g, &gpu.read(&output), &rhs, &r, &p);
        }
    });
}
