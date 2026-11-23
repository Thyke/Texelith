use super::*;
use crate::{MaterialError, PoissonParameters, PoissonSolver, PoissonStop};
use texelith_core::{MaterialNode, MaterialNodeParams as P};

#[derive(Default)]
pub(super) struct State(Option<(PoissonSolver, Lease, Lease)>);
impl State {
    pub(super) fn evict(&mut self) -> bool {
        self.0.take().is_some()
    }
}
pub(super) fn supports(node: &MaterialNode) -> bool {
    matches!(node.params, P::HeightFromNormal { .. })
}
pub(super) fn scratch(
    node: &MaterialNode,
    extent: Extent,
) -> std::result::Result<u64, MaterialError> {
    let P::HeightFromNormal { boundary, .. } = node.params else {
        unreachable!("registered solver")
    };
    crate::poisson_solver::scratch_bytes(extent, boundary)
        .map_err(|e| MaterialError::InvalidRequest(e.to_string()))
}
pub(super) fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease> {
    let P::HeightFromNormal {
        boundary,
        v_cycles,
        pre_smooth,
        post_smooth,
        coarsest_iterations,
        relative_tolerance,
        absolute_tolerance,
    } = context.node.params
    else {
        unreachable!("registered solver")
    };
    // Leave the old cache reachable by Runtime::allocate's eviction path.
    let mut admission = runtime.allocate(&[
        texture_desc(context.extent, wgpu::TextureFormat::R32Float),
        AllocationDesc::Staging {
            bytes: crate::poisson_solver::scratch_bytes(context.extent, boundary)?,
        },
    ])?;
    let output = admission.remove(0);
    // Reserve through Runtime so other nodes' completed caches can be evicted.
    // Actual solver allocations replace this reservation before any submission.
    drop(admission);
    let input = context.input();
    let convention = context
        .work
        .material
        .as_ref()
        .expect("material work")
        .snapshot
        .graph
        .output_normal_convention(context.node.inputs[0].source)?
        .ok_or_else(|| GpuError("missing normal convention".into()))?;
    let solver =
        if let Some((solver, old_input, old_output)) = runtime.nodes.poisson_solver.0.take() {
            solver.clear_workspace();
            drop((old_input, old_output));
            solver
        } else {
            PoissonSolver::new(
                runtime.device.clone(),
                runtime.queue.clone(),
                runtime.pool.clone(),
            )?
        };
    let shared = runtime.shared.clone();
    let leases = [input.clone(), output.clone()];
    let work = context.work;
    let result = solver.solve_submitted(
        input.texture(),
        output.texture(),
        convention,
        PoissonParameters {
            boundary,
            v_cycles,
            pre_smooth,
            post_smooth,
            coarsest_iterations,
            relative_tolerance,
            absolute_tolerance,
        },
        || shared.lock().unwrap().scheduler.is_current(work),
        |encoder| {
            let (_, done, _) = runtime
                .submit(encoder, &leases)
                .map_err(|e| GpuError(format!("Poisson submission: {e:?}")))?;
            runtime.shared.lock().unwrap().compute_submissions += 1;
            runtime
                .wait(&done)
                .map_err(|e| GpuError(format!("Poisson completion: {e:?}")))
        },
    )?;
    if !runtime.current(work) || result.stop == PoissonStop::Cancelled {
        return Err(Failure::Cancelled);
    }
    if result.stop == PoissonStop::MaxCycles {
        return Err(GpuError(format!(
            "Poisson did not converge after {} cycles: residual {} (RHS {})",
            result.cycles, result.residual_l2, result.rhs_l2
        ))
        .into());
    }
    runtime.nodes.poisson_solver.0 = Some((solver, input, output.clone()));
    Ok(output)
}
