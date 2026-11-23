use super::{BUDGET, Result, SCRATCH, aggregate_timings, gpu::Gpu, require};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use texelith_core::{Boundary, Extent, NormalConvention};
use texelith_gpu::{
    Axis, PoissonGrid, PoissonPass as P, PreparedPoisson,
    pool::{AllocationDesc, ResourcePool},
};

struct Sweep<'a> {
    g: &'a Gpu,
    caller: ResourcePool,
    passes: Vec<PreparedPoisson>,
    inventory: Vec<Value>,
}
impl Sweep<'_> {
    fn texture(&mut self, e: Extent, normal: bool) -> Result<wgpu::Texture> {
        let format = if normal {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::R32Float
        };
        self.caller
            .allocate(AllocationDesc::Texture {
                extent: e,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            })
            .map_err(|e| e.to_string())?;
        Ok(self.g.texture(e, format))
    }
    fn pass(
        &mut self,
        grid: &PoissonGrid,
        op: P,
        inputs: &[&wgpu::Texture],
        output: &wgpu::Texture,
        repetitions: usize,
    ) -> Result<()> {
        let pass = self
            .g
            .operators
            .prepare(grid, op, inputs, output)
            .map_err(|e| e.to_string())?;
        self.inventory.push(json!({"extent":[grid.extent().width,grid.extent().height],"operator":format!("{op:?}"),"applications":repetitions,"dispatches_per_application":pass.dispatch_count()}));
        for _ in 0..repetitions {
            self.passes.push(pass.clone());
        }
        Ok(())
    }
}

pub fn run(
    g: &Gpu,
    pool: Arc<Mutex<ResourcePool>>,
    extent: Extent,
    maximum: bool,
) -> Result<Value> {
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "scratch still in flight before sweep",
    )?;
    let prepare_start = Instant::now();
    let mut s = Sweep {
        g,
        caller: ResourcePool::new(BUDGET - SCRATCH),
        passes: Vec::new(),
        inventory: Vec::new(),
    };
    let grid = PoissonGrid::new(extent, Boundary::Repeat).map_err(|e| e.to_string())?;
    let scalar = Extent::new(1, 1).unwrap();
    let normal = s.texture(extent, true)?;
    let height = s.texture(extent, false)?;
    let u = s.texture(extent, false)?;
    let v = s.texture(extent, false)?;
    let rhs = s.texture(extent, false)?;
    let temp = s.texture(extent, false)?;
    let mean = s.texture(scalar, false)?;
    let count = s.texture(scalar, false)?;
    let norm = s.texture(scalar, false)?;
    // Analytic unit normal (half precision) and bounded smooth height, no sibling
    // map generator. Repeated sweeps overwrite every consumed intermediate.
    let normals: Vec<u16> = (0..extent.width * extent.height)
        .flat_map(|i| {
            [
                if i % 2 == 0 { 0x3800 } else { 0xb800 },
                0x3800,
                0x39a8,
                0x3c00,
            ]
        })
        .collect();
    g.upload(&normal, bytemuck::cast_slice(&normals), 8);
    drop(normals);
    let heights: Vec<f32> = (0..extent.width * extent.height)
        .map(|i| {
            ((std::f64::consts::TAU * f64::from(i % extent.width) / f64::from(extent.width)).sin()
                + (std::f64::consts::TAU * f64::from(i / extent.width) / f64::from(extent.height))
                    .cos()) as f32
                * 2.0f32.powi(-20)
        })
        .collect();
    g.upload(&height, bytemuck::cast_slice(&heights), 4);
    drop(heights);
    for (axis, out) in [(Axis::X, &u), (Axis::Y, &v)] {
        s.pass(
            &grid,
            P::NormalGradient {
                axis,
                convention: NormalConvention::OpenGl,
            },
            &[&normal],
            out,
            1,
        )?;
    }
    s.pass(&grid, P::ClampCount, &[&normal], &count, 1)?;
    for (axis, out) in [(Axis::X, &u), (Axis::Y, &v)] {
        s.pass(&grid, P::Difference(axis), &[&height], out, 1)?;
    }
    s.pass(&grid, P::Transpose, &[&u, &v], &rhs, 1)?;
    s.pass(&grid, P::Laplacian, &[&height], &temp, 1)?;
    s.pass(&grid, P::Mean, &[&height], &mean, 1)?;
    s.pass(&grid, P::ProjectMean, &[&height, &mean], &temp, 1)?;
    s.pass(
        &grid,
        P::Jacobi { omega: 2.0 / 3.0 },
        &[&height, &rhs],
        &temp,
        if maximum { 6 } else { 1 },
    )?;
    let mut coarse_grid = if extent.width > 1 || extent.height > 1 {
        grid.coarsen().map_err(|e| e.to_string())?
    } else {
        grid.clone()
    };
    let mut coarse = s.texture(coarse_grid.extent(), false)?;
    let ac = s.texture(coarse_grid.extent(), false)?;
    s.pass(&grid, P::Restrict, &[&height], &coarse, 1)?;
    s.pass(&grid, P::Prolong, &[&coarse], &temp, 1)?;
    s.pass(&coarse_grid, P::Laplacian, &[&coarse], &ac, 1)?;
    s.pass(&grid, P::Residual, &[&height, &rhs], &temp, 1)?;
    s.pass(&grid, P::L2Norm, &[&temp], &norm, 1)?;
    let mut levels = if extent.width > 1 || extent.height > 1 {
        2
    } else {
        1
    };
    if maximum {
        loop {
            let e = coarse_grid.extent();
            let u = s.texture(e, false)?;
            let v = s.texture(e, false)?;
            let rhs = s.texture(e, false)?;
            let temp = s.texture(e, false)?;
            let terminal = e.width == 1 && e.height == 1;
            for (axis, out) in [(Axis::X, &u), (Axis::Y, &v)] {
                s.pass(&coarse_grid, P::Difference(axis), &[&coarse], out, 1)?;
            }
            s.pass(&coarse_grid, P::Transpose, &[&u, &v], &rhs, 1)?;
            s.pass(&coarse_grid, P::Laplacian, &[&coarse], &temp, 1)?;
            s.pass(&coarse_grid, P::Mean, &[&coarse], &mean, 1)?;
            s.pass(&coarse_grid, P::ProjectMean, &[&coarse, &mean], &temp, 1)?;
            s.pass(
                &coarse_grid,
                P::Jacobi { omega: 2.0 / 3.0 },
                &[&coarse, &rhs],
                &temp,
                if terminal { 32 } else { 6 },
            )?;
            s.pass(&coarse_grid, P::Residual, &[&coarse, &rhs], &temp, 1)?;
            s.pass(&coarse_grid, P::L2Norm, &[&temp], &norm, 1)?;
            if terminal {
                break;
            }
            let next_grid = coarse_grid.coarsen().map_err(|e| e.to_string())?;
            let next = s.texture(next_grid.extent(), false)?;
            s.pass(&coarse_grid, P::Restrict, &[&coarse], &next, 1)?;
            s.pass(&coarse_grid, P::Prolong, &[&next], &temp, 1)?;
            coarse = next;
            coarse_grid = next_grid;
            levels += 1;
        }
    }
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
    g.wait();
    let scratch = pool.lock().unwrap().diagnostics().allocated_bytes;
    let peak = s.caller.diagnostics().allocated_bytes + scratch;
    require(
        peak <= BUDGET && scratch <= SCRATCH,
        "sweep exceeds byte budget",
    )?;
    let mut rows = Vec::with_capacity(110);
    let mut submissions = Vec::with_capacity(110);
    for _ in 0..110 {
        let mut encoder = g.device.create_command_encoder(&Default::default());
        for pass in &s.passes {
            pass.encode(&mut encoder);
        }
        let command = encoder.finish();
        let start = Instant::now();
        g.queue.submit([command]);
        submissions.push(start.elapsed().as_secs_f64() * 1000.0);
        g.wait();
        rows.push(vec![start.elapsed().as_secs_f64() * 1000.0]);
    }
    let timing = aggregate_timings(&rows, 10)?;
    let submission_ms = submissions[10..].iter().sum::<f64>() / 100.0;
    let applications = s.passes.len();
    let dispatches: usize = s.passes.iter().map(PreparedPoisson::dispatch_count).sum();
    // Inspect the measured reduction outputs for nonfinite values and clamp drift.
    let actual_count = g.read(&count).v[0];
    let actual_norm = g.read(&norm).v[0];
    require(
        actual_count == 0.0 && actual_norm.is_finite(),
        "invalid timed reduction outputs",
    )?;
    let result = json!({"kind":"completion_upper_bound","warmups":10,"measurements":timing.measurements,
        "p95_ms":timing.p95_ms,"submission_ms":submission_ms,"peak_bytes":peak,"scratch_bytes":scratch,
        "cancellation_errors":0,"scope":"full_operator_sweep","includes_reductions":true,"includes_coarse_operators":true,
        "operators":super::report_contract::OPERATORS,"operator_applications":applications,"dispatches":dispatches,
        "settings":if maximum {"all hierarchy levels; six Jacobi applications per nonterminal level, 32 terminal; isolated operator applications, no V-cycle"} else {"one full sweep including both gradient/difference axes and first Galerkin coarse level"},
        "levels":levels,"pass_inventory":s.inventory,"cold_prepare_ms":prepare_ms,
        "completion_samples_ms":rows[10..].iter().map(|r|r[0]).collect::<Vec<_>>()});
    drop(s);
    require(
        pool.lock().unwrap().diagnostics().allocated_bytes == 0,
        "completed sweep scratch leaked",
    )?;
    Ok(result)
}
