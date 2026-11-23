//! Bounded, mean-zero normal integration: periodic radix-2 spectral solving and
//! T-049 Galerkin multigrid for NPOT/Neumann grids. Published heights are R32Float.
mod combine;
pub(crate) mod normal_proxy;
mod spectral;
use crate::pool::{AllocationDesc, AllocationId, ResourcePool};
use crate::{Axis, GpuError, PoissonGrid, PoissonOperators, PoissonPass as Op, PreparedPoisson};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use texelith_core::{Boundary, Extent, NormalConvention};

#[derive(Clone, Debug)]
pub struct PoissonParameters {
    pub boundary: Boundary,
    /// Maximum multigrid cycles, or direct spectral solve plus residual-correction
    /// solves on periodic power-of-two grids. The spectral path also certifies
    /// the least-squares objective gap of the published height.
    pub v_cycles: u32,
    /// Smoother controls apply to the NPOT/Neumann multigrid path.
    pub pre_smooth: u32,
    pub post_smooth: u32,
    pub coarsest_iterations: u32,
    /// Relative normal-equation tolerance, or spectral optimality-gap norm
    /// divided by the input gradient norm (user decision, 2026-11-11).
    pub relative_tolerance: f32,
    pub absolute_tolerance: f32,
}
impl Default for PoissonParameters {
    fn default() -> Self {
        Self {
            boundary: Boundary::Repeat,
            v_cycles: 8,
            pre_smooth: 3,
            post_smooth: 3,
            coarsest_iterations: 32,
            relative_tolerance: 1e-3,
            absolute_tolerance: 1e-6,
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
pub enum PoissonStop {
    Converged,
    MaxCycles,
    Cancelled,
}
#[derive(Debug)]
pub struct PoissonSolveReport {
    pub stop: PoissonStop,
    pub cycles: u32,
    pub rhs_l2: f32,
    pub residual_l2: f32,
    /// Spectral certificate: sqrt(F(h) - min F), where F = ||D h - g||^2.
    /// Unlike the normal-equation residual, this measures distance to the
    /// least-squares optimum without amplifying R32 rounding by two derivatives.
    pub optimality_gap_l2: Option<f32>,
    pub gradient_l2: Option<f32>,
    pub residual_history: Vec<f32>,
    pub clamped_pixels: u32,
    pub scalar_readbacks: u32,
    pub submission_ms: f64,
}
impl PoissonSolveReport {
    fn empty() -> Self {
        Self {
            stop: PoissonStop::Cancelled,
            cycles: 0,
            rhs_l2: 0.0,
            residual_l2: 0.0,
            optimality_gap_l2: None,
            gradient_l2: None,
            residual_history: Vec::with_capacity(16),
            clamped_pixels: 0,
            scalar_readbacks: 0,
            submission_ms: 0.0,
        }
    }
}
struct Reservations {
    pool: Arc<Mutex<ResourcePool>>,
    ids: Vec<AllocationId>,
}
impl Drop for Reservations {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        for id in &self.ids {
            let _ = pool.release_transient(*id);
        }
    }
}
struct Arena<'a> {
    device: &'a wgpu::Device,
    reservations: Reservations,
}
impl Arena<'_> {
    fn reserve(&mut self, bytes: u64) -> Result<(), GpuError> {
        let id = self
            .reservations
            .pool
            .lock()
            .unwrap()
            .allocate(AllocationDesc::Buffer { bytes })
            .map_err(|e| GpuError(e.to_string()))?;
        self.reservations.ids.push(id);
        Ok(())
    }
    fn texture(&mut self, e: Extent) -> Result<wgpu::Texture, GpuError> {
        self.reserve(u64::from(e.width) * u64::from(e.height) * 4)?;
        Ok(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Poisson workspace"),
            size: wgpu::Extent3d {
                width: e.width,
                height: e.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }
}
enum Step {
    Operator(PreparedPoisson),
    Combine(combine::Combine),
}
impl Step {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Operator(p) => p.encode(encoder),
            Self::Combine(p) => p.encode(encoder),
        }
    }
}
impl From<PreparedPoisson> for Step {
    fn from(p: PreparedPoisson) -> Self {
        Self::Operator(p)
    }
}
impl From<combine::Combine> for Step {
    fn from(p: combine::Combine) -> Self {
        Self::Combine(p)
    }
}
struct Level {
    smooth: Vec<Step>,
    descend: Vec<Step>,
    ascend: Vec<Step>,
}
struct Workspace {
    normal: wgpu::Texture,
    output: wgpu::Texture,
    boundary: Boundary,
    convention: NormalConvention,
    setup: Vec<Step>,
    count: Step,
    rhs_norm: Step,
    levels: Vec<Level>,
    finish: Vec<Step>,
    scalar: wgpu::Texture,
    readback: wgpu::Buffer,
    _reservations: Reservations,
}
pub struct PoissonSolver {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pool: Arc<Mutex<ResourcePool>>,
    operators: PoissonOperators,
    combine: combine::Kernel,
    workspace: Mutex<Option<Arc<Workspace>>>,
    spectral: Mutex<Option<spectral::Solver>>,
    spectral_kernel: std::sync::OnceLock<spectral::Kernel>,
}
impl PoissonSolver {
    pub(crate) fn clear_workspace(&self) {
        *self.workspace.lock().unwrap() = None;
        *self.spectral.lock().unwrap() = None;
    }
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError> {
        let operators = PoissonOperators::new(device.clone(), pool.clone())?;
        let combine = combine::Kernel::new(&device);
        Ok(Self {
            device,
            queue,
            pool,
            operators,
            combine,
            workspace: Mutex::new(None),
            spectral: Mutex::new(None),
            spectral_kernel: std::sync::OnceLock::new(),
        })
    }
    /// Reset the initial guess and solve on the caller's device. Operands remain
    /// caller-budgeted; cached scratch is charged to the supplied shared pool.
    /// The callback is polled between passes. Submitted work completes before
    /// returning; only `Converged` is suitable for publication.
    pub fn solve(
        &self,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        convention: NormalConvention,
        parameters: PoissonParameters,
        should_continue: impl FnMut() -> bool,
    ) -> Result<PoissonSolveReport, GpuError> {
        let mut submission_ms = 0.0;
        let mut report = self.solve_submitted(
            normal,
            output,
            convention,
            parameters,
            should_continue,
            |encoder| {
                let started = Instant::now();
                let index = self.queue.submit([encoder.finish()]);
                submission_ms += started.elapsed().as_secs_f64() * 1000.0;
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(index),
                        timeout: Some(Duration::from_secs(30)),
                    })
                    .map_err(|e| GpuError(e.to_string()))?;
                Ok(())
            },
        )?;
        report.submission_ms = submission_ms;
        Ok(report)
    }
    pub(crate) fn solve_submitted(
        &self,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        convention: NormalConvention,
        p: PoissonParameters,
        mut current: impl FnMut() -> bool,
        mut submit: impl FnMut(wgpu::CommandEncoder) -> Result<(), GpuError>,
    ) -> Result<PoissonSolveReport, GpuError> {
        validate(normal, output, &p)?;
        let mut report = PoissonSolveReport::empty();
        if !current() {
            return Ok(report);
        }
        if spectral::supports(output, p.boundary) {
            *self.workspace.lock().unwrap() = None;
            let mut spectral = self.spectral.lock().unwrap();
            if !spectral
                .as_ref()
                .is_some_and(|s| s.matches(normal, output, convention))
            {
                *spectral = None;
                *spectral = Some(spectral::Solver::new(self, normal, output, convention)?);
            }
            return spectral
                .as_ref()
                .unwrap()
                .solve(self, &p, &mut current, &mut submit);
        }
        *self.spectral.lock().unwrap() = None;
        let mut cache = self.workspace.lock().unwrap();
        if !cache.as_ref().is_some_and(|w| {
            w.normal == *normal
                && w.output == *output
                && w.boundary == p.boundary
                && w.convention == convention
        }) {
            *cache = None;
            *cache = Some(Arc::new(
                self.prepare(normal, output, convention, p.boundary)?,
            ));
        }
        let w = cache.as_ref().unwrap();
        let mut batch = Batch {
            solver: self,
            workspace: w,
            encoder: self.device.create_command_encoder(&Default::default()),
            pending: 0,
            current: &mut current,
            submit: &mut submit,
        };
        for step in &w.setup {
            if !batch.step(step)? {
                return Ok(report);
            }
        }
        if !batch.step(&w.rhs_norm)? {
            return Ok(report);
        }
        report.rhs_l2 = batch.scalar()?;
        report.residual_l2 = report.rhs_l2;
        report.scalar_readbacks += 1;
        if !batch.step(&w.count)? {
            return Ok(report);
        }
        report.clamped_pixels = batch.scalar()? as u32;
        report.scalar_readbacks += 1;
        if converged(report.residual_l2, report.rhs_l2, &p) {
            report.stop = PoissonStop::Converged;
            return Ok(report);
        }
        for cycle in 0..p.v_cycles {
            if !v_cycle(&mut batch, &w.levels, 0, &p)? {
                return Ok(report);
            }
            for step in &w.finish {
                if !batch.step(step)? {
                    return Ok(report);
                }
            }
            report.residual_l2 = batch.scalar()?;
            report.scalar_readbacks += 1;
            report.cycles = cycle + 1;
            report.residual_history.push(report.residual_l2);
            if !(batch.current)() {
                return Ok(report);
            }
            if converged(report.residual_l2, report.rhs_l2, &p) {
                report.stop = PoissonStop::Converged;
                return Ok(report);
            }
        }
        report.stop = PoissonStop::MaxCycles;
        Ok(report)
    }
    fn prepare(
        &self,
        normal: &wgpu::Texture,
        output: &wgpu::Texture,
        convention: NormalConvention,
        boundary: Boundary,
    ) -> Result<Workspace, GpuError> {
        let extent = crate::executor::texture_extent(output)?;
        let mut arena = Arena {
            device: &self.device,
            reservations: Reservations {
                pool: self.pool.clone(),
                ids: Vec::new(),
            },
        };
        let scalar = arena.texture(Extent::new(1, 1)?)?;
        arena.reserve(256)?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poisson scalar readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut grids = vec![PoissonGrid::new(extent, boundary)?];
        while grids.last().unwrap().extent() != Extent::new(1, 1)? {
            grids.push(grids.last().unwrap().coarsen()?);
        }
        let mut textures = Vec::new();
        for (i, grid) in grids.iter().enumerate() {
            textures.push((
                if i == 0 {
                    output.clone()
                } else {
                    arena.texture(grid.extent())?
                },
                arena.texture(grid.extent())?,
                arena.texture(grid.extent())?,
            ));
        }
        let op = |g: &PoissonGrid, pass, inputs: &[&wgpu::Texture], out: &wgpu::Texture| {
            self.operators.prepare(g, pass, inputs, out).map(Step::from)
        };
        let combine = |arena: &mut Arena<'_>, input, out, correction, mode| {
            self.combine
                .prepare(arena, input, out, correction, mode, boundary)
                .map(Step::from)
        };
        let g = &grids[0];
        let (x, rhs, tmp) = &textures[0];
        let setup = vec![
            op(
                g,
                Op::NormalGradient {
                    axis: Axis::X,
                    convention,
                },
                &[normal],
                tmp,
            )?,
            op(
                g,
                Op::NormalGradient {
                    axis: Axis::Y,
                    convention,
                },
                &[normal],
                rhs,
            )?,
            op(g, Op::Transpose, &[tmp, rhs], x)?,
            op(g, Op::Mean, &[x], &scalar)?,
            op(g, Op::ProjectMean, &[x, &scalar], rhs)?,
            combine(&mut arena, tmp, x, tmp, 0)?,
        ];
        let rhs_norm = op(g, Op::L2Norm, &[rhs], &scalar)?;
        let count = op(g, Op::ClampCount, &[normal], &scalar)?;
        let finish = vec![
            op(g, Op::Mean, &[x], &scalar)?,
            op(g, Op::ProjectMean, &[x, &scalar], tmp)?,
            combine(&mut arena, tmp, x, tmp, 1)?,
            op(g, Op::Residual, &[x, rhs], tmp)?,
            op(g, Op::L2Norm, &[tmp], &scalar)?,
        ];
        let mut levels = Vec::new();
        for (i, g) in grids.iter().enumerate() {
            let (x, rhs, tmp) = &textures[i];
            let smooth = vec![
                op(g, Op::Jacobi { omega: 2.0 / 3.0 }, &[x, rhs], tmp)?,
                combine(&mut arena, tmp, x, tmp, 1)?,
            ];
            let mut descend = Vec::new();
            let mut ascend = Vec::new();
            if let Some((cx, cb, ct)) = textures.get(i + 1) {
                descend = vec![
                    op(g, Op::Residual, &[x, rhs], tmp)?,
                    op(g, Op::Restrict, &[tmp], cb)?,
                    op(&grids[i + 1], Op::Mean, &[cb], &scalar)?,
                    op(&grids[i + 1], Op::ProjectMean, &[cb, &scalar], ct)?,
                    combine(&mut arena, ct, cb, ct, 1)?,
                    combine(&mut arena, ct, cx, ct, 0)?,
                ];
                ascend = vec![
                    combine(&mut arena, x, tmp, cx, 2)?,
                    combine(&mut arena, tmp, x, tmp, 1)?,
                ];
            }
            levels.push(Level {
                smooth,
                descend,
                ascend,
            });
        }
        Ok(Workspace {
            normal: normal.clone(),
            output: output.clone(),
            boundary,
            convention,
            setup,
            count,
            rhs_norm,
            levels,
            finish,
            scalar,
            readback,
            _reservations: arena.reservations,
        })
    }
}
fn converged(residual: f32, rhs: f32, p: &PoissonParameters) -> bool {
    residual <= p.absolute_tolerance || residual <= p.relative_tolerance * rhs
}
fn validate(
    normal: &wgpu::Texture,
    output: &wgpu::Texture,
    p: &PoissonParameters,
) -> Result<(), GpuError> {
    if !(1..=16).contains(&p.v_cycles)
        || p.pre_smooth == 0
        || p.post_smooth == 0
        || p.coarsest_iterations == 0
        || !p.relative_tolerance.is_finite()
        || p.relative_tolerance <= 0.0
        || !p.absolute_tolerance.is_finite()
        || p.absolute_tolerance <= 0.0
    {
        return Err(GpuError("invalid Poisson parameters".into()));
    }
    if normal == output
        || normal.format() != wgpu::TextureFormat::Rgba16Float
        || output.format() != wgpu::TextureFormat::R32Float
        || normal.size() != output.size()
        || !normal
            .usage()
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        || !output
            .usage()
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING)
        || normal.dimension() != wgpu::TextureDimension::D2
        || output.dimension() != wgpu::TextureDimension::D2
        || normal.sample_count() != 1
        || output.sample_count() != 1
        || normal.mip_level_count() != 1
        || output.mip_level_count() != 1
        || normal.depth_or_array_layers() != 1
    {
        return Err(GpuError("invalid Poisson operands".into()));
    }
    crate::executor::texture_extent(output)?;
    Ok(())
}
struct Batch<'a> {
    solver: &'a PoissonSolver,
    workspace: &'a Arc<Workspace>,
    encoder: wgpu::CommandEncoder,
    pending: usize,
    current: &'a mut dyn FnMut() -> bool,
    submit: &'a mut dyn FnMut(wgpu::CommandEncoder) -> Result<(), GpuError>,
}
impl Batch<'_> {
    fn step(&mut self, step: &Step) -> Result<bool, GpuError> {
        if !(self.current)() {
            return Ok(false);
        }
        step.encode(&mut self.encoder);
        self.pending += 1;
        // Bounded batches expose progress and allow stale work to stop inside a cycle.
        if self.pending == 32 {
            self.flush()?;
        }
        Ok(true)
    }
    fn flush(&mut self) -> Result<(), GpuError> {
        let encoder = std::mem::replace(
            &mut self.encoder,
            self.solver
                .device
                .create_command_encoder(&Default::default()),
        );
        let retained = self.workspace.clone();
        let result = (self.submit)(encoder);
        self.solver
            .queue
            .on_submitted_work_done(move || drop(retained));
        self.pending = 0;
        result
    }
    fn scalar(&mut self) -> Result<f32, GpuError> {
        self.encoder.copy_texture_to_buffer(
            self.workspace.scalar.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.workspace.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            self.workspace.scalar.size(),
        );
        self.flush()?;
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.workspace
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.solver
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| GpuError(e.to_string()))?;
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|e| GpuError(e.to_string()))?
            .map_err(|e| GpuError(e.to_string()))?;
        let mapped = self
            .workspace
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError(e.to_string()))?;
        let value = f32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        self.workspace.readback.unmap();
        if !value.is_finite() || value < 0.0 {
            return Err(GpuError("nonfinite Poisson residual/diagnostic".into()));
        }
        Ok(value)
    }
}
fn v_cycle(
    batch: &mut Batch<'_>,
    levels: &[Level],
    index: usize,
    p: &PoissonParameters,
) -> Result<bool, GpuError> {
    let level = &levels[index];
    let terminal = index + 1 == levels.len();
    let count = if terminal {
        p.coarsest_iterations
    } else {
        p.pre_smooth
    };
    for _ in 0..count {
        for step in &level.smooth {
            if !batch.step(step)? {
                return Ok(false);
            }
        }
    }
    if terminal {
        return Ok(true);
    }
    for step in &level.descend {
        if !batch.step(step)? {
            return Ok(false);
        }
    }
    if !v_cycle(batch, levels, index + 1, p)? {
        return Ok(false);
    }
    for step in &level.ascend {
        if !batch.step(step)? {
            return Ok(false);
        }
    }
    for _ in 0..p.post_smooth {
        for step in &level.smooth {
            if !batch.step(step)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub(crate) fn scratch_bytes(extent: Extent, boundary: Boundary) -> Result<u64, GpuError> {
    if boundary == Boundary::Repeat
        && extent.width.is_power_of_two()
        && extent.height.is_power_of_two()
    {
        let pixels = u64::from(extent.width) * u64::from(extent.height);
        let groups = u64::from(extent.width.div_ceil(8)) * u64::from(extent.height.div_ceil(8));
        // One complex array, two reduction arrays, scalar readback, dummy
        // texture, twiddle table and at most 32 prepared 64-byte uniforms.
        return Ok(pixels * 8 + (groups + groups.div_ceil(64)) * 16 + 36 + 2048 * 8 + 32 * 64);
    }
    // Conservative admission includes every retained prepared stencil, reduction,
    // uniform, scalar staging buffer, and both coarse operator work textures.
    let mut bytes = 256u64 + 4;
    let mut e = extent;
    let mut finest = true;
    loop {
        let n = u64::from(e.width) * u64::from(e.height);
        bytes += n * if finest { 8 } else { 28 };
        bytes += 4096 + 2048 * (u64::from(e.width) + u64::from(e.height));
        bytes += 4 * n.div_ceil(64) * if finest { 5 } else { 2 };
        if e.width == 1 && e.height == 1 {
            break;
        }
        e = Extent::new(e.width.div_ceil(2), e.height.div_ceil(2))?;
        finest = false;
    }
    Ok(bytes)
}
