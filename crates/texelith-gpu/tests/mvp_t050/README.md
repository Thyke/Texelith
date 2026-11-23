# Poisson solver test contracts

The CPU oracle runs independently of the GPU solver:

```powershell
New-Item -ItemType Directory -Force target/evidence/T-050 | Out-Null
rustc --edition=2024 --test crates/texelith-gpu/tests/mvp_t050/oracle.rs -o target/evidence/T-050/oracle-tests.exe
& target/evidence/T-050/oracle-tests.exe
```

## Solver API

The solver tests exercise this public leaf boundary. Material tests use the
public typed request, worker and plan APIs independently of node registration.

```rust
#[derive(Clone, Debug)]
pub struct PoissonParameters {
    pub boundary: Boundary,
    pub v_cycles: u32,
    pub pre_smooth: u32,
    pub post_smooth: u32,
    pub coarsest_iterations: u32,
    pub relative_tolerance: f32,
    pub absolute_tolerance: f32,
}
#[derive(Debug, PartialEq, Eq)]
pub enum PoissonStop { Converged, MaxCycles, Cancelled }
#[derive(Debug)]
pub struct PoissonSolveReport {
    pub stop: PoissonStop,
    pub cycles: u32,
    pub rhs_l2: f32,
    pub residual_l2: f32,
    pub residual_history: Vec<f32>, // one scalar per completed V-cycle
    pub clamped_pixels: u32,
    pub scalar_readbacks: u32, // includes initialization and diagnostics
}
impl PoissonSolver {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
        pool: Arc<Mutex<ResourcePool>>) -> Result<Self, GpuError>;
    pub fn solve(&self, normal: &wgpu::Texture, output: &wgpu::Texture,
        convention: NormalConvention, parameters: PoissonParameters,
        should_continue: impl FnMut() -> bool) -> Result<PoissonSolveReport, GpuError>;
}
```

This worker-side synchronous leaf entry point owns no UI policy. Inputs are signed
Rgba16Float normal values with explicit convention; outputs are distinct linear
R32Float heights. No CPU fallback or test GPU kernel is supplied. Each call resets
its initial guess. It returns after its actual GPU work completes, retaining all
submitted scratch through fences. Caller textures are caller-budgeted; internal
allocations and operator/reduction scratch use the supplied shared pool. Unused
scratch is evictable once the solver is dropped. A one-byte scratch budget must
fail atomically without mutating output. Invalid parameters fail before output
mutation. A stale token at entry leaves output untouched; after work begins,
cancelled intermediate output is unspecified and must never be published.

`MaxCycles` is a measured nonconvergence outcome, never a success bit. The material
adapter must report an error containing the residual and publish no output for
nonconvergence. Explicit acceptance for export is a later pipeline policy; no
unimplemented export acceptance API is invented here. Cancellation is a distinct
outcome and is checked within a V-cycle, between passes. Readbacks are bounded by
one residual scalar per completed cycle plus at most three setup/diagnostic
scalars. History and scratch allocation must not grow without the 16-cycle bound.

## Numerical coverage

- `oracle.rs`: original sparse f64 incidence matrix, geometric overlap restriction,
  pixel-center bilinear prolongation, assembled recursive R A P and weighted
  Jacobi V-cycles. Four independently runnable anchors include hand calculations
  and a Fourier pseudoinverse for a nonintegrable periodic field. The Fourier
  oracle is test-only, not a proposed production FFT backend.
- `solver.rs`: periodic sinusoid, natural Neumann cosine, NPOT 257x193,
  1x1/1xN/Nx1, ramp/impulse, moderate slopes, flat exact zero, absolute stopping
  for a small nonzero RHS, clamped-pixel count, nonintegrable least squares,
  measured versus reported L2 residual, failed 1/16-cycle solves, invalid budgets,
  repeated deterministic output, operand/parameter validation, reused scratch
  across 1/16 cycles and cancellation/recovery. Height RMS <=1e-3;
  CPU scalar oracle comparison <=2e-5 RMS; reported norm error <=2e-5 of RHS scale;
  mean magnitude <=2e-5. Expectations account for actual half-float normal upload.
- `material.rs`: 4K bounded planner, all control/cache identities, public normal
  import to signed linear height, GL/DX equivalence, failure without publication,
  shared ledger budget rejection/recovery and real cancellation. No other task's
  leaf is imported to manufacture inputs.
- `harness.rs`: nonempty unique `poisson` registration in `maps`/`all`, rejects
  unknown suites, sums all cycles/setup/finalization before p95, verifies shared
  zero-assertion and GPU-validation-error rejection. It creates no benchmark
  evidence from invented schema samples.

## Verification

Run `cargo test -p texelith-gpu --test mvp_t050 --locked` for the contracts and
`cargo run --release -p texelith-gpu --example verify_mvp -- --suite poisson
--size 4096 --backend dx12 --adapter-name "RTX 5070 Ti"
--report target/evidence/T-050/poisson.json` for hardware measurements.

The hardware runner records adapter/backend, manifest hash, residuals/cycles,
clamp counts and peak bytes. Timings sum complete solves after 10 warmups over
100 measurements, with maximum quality reported separately. Default 4K p95
must be at most 100 ms, and the 1K interactive proxy p95 must stay below
16 ms. Per-dispatch timings cannot substitute for a complete solve. Unit tests alone do not establish hardware performance.
