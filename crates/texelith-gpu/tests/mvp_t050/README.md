# T-050 tests-first contract

Only this test directory and `tests/mvp_t050.rs` are authored in this pass.
No solver, WGSL, shared registry edits, success stubs or new dependencies are
supplied. The production-dependent target is intentionally red before T-050's
sequential implementer pass. The pure CPU oracle can run independently:

```powershell
rustc --edition=2024 --test crates/texelith-gpu/tests/mvp_t050/oracle.rs -o target-evidence/T-050/oracle-tests.exe
& target-evidence/T-050/oracle-tests.exe
```

## Proposed additive leaf API

These are test-author proposed names, not symbols promised by the ADR. Equivalent
API spelling can be reconciled by the implementer without weakening assertions.
Export from the task-owned module; do not alter the node registry just to satisfy
tests. The material tests use existing public typed request/worker/plan APIs and
are independent of T-121's registration refactor.

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

## Implementation and integration follow-through

Run the task's exact cargo and full-node release benchmark commands. The `poisson`
runner must exercise the CPU comparisons, NPOT, nonintegrable/failed iteration,
cancellation and memory tests with real assertions and report adapter/backend,
manifest hash, residuals/cycles, clamp counts, peak bytes, 10 warmups + 100 complete
node timings, and separate maximum-quality timings. Default 4K p95 stays <16 ms;
the maximum-quality result is separately reported. No per-dispatch timing can
substitute for complete solve timing. Shader warmup and initialization scope must
be explicit. These source tests do not certify unimplemented performance.

T-049 currently contains an outdated negative suite-name assertion in
`tests/mvp_t049/report.rs` which explicitly rejects `poisson`. Once this task's
suite is registered, that assertion conflicts with T-050 acceptance. It is
outside this test-author's scope: orchestrator must authorize its owner to remove
only that invalid-name case. Keep `poisson_operators` separately registered.

The native `measure-faz0` gate is deferred to the orchestrator, per dispatch.

## Recorded tests-first baseline (2026-11-11)

There are 25 authored tests. No production implementation was changed.

| Check | Observed result |
| --- | --- |
| `rustfmt --check --edition 2024 crates/texelith-gpu/tests/mvp_t050.rs` | PASS |
| Standalone `rustc --test` oracle command above | PASS: 4/4 anchors |
| Isolated public-contract compile against built debug libraries | PASS: existing-API tests and upload/readback helpers compile without stubs |
| Isolated CPU/public-contract execution | RED: 6 passed, 4 failed; four GPU worker cases deliberately filtered out of this diagnostic run |
| `cargo build -p texelith-gpu --all-targets --locked` | RED, exit 101: E0432, four proposed solver exports absent |
| `cargo test -p texelith-gpu --test mvp_t050 --locked` | RED, exit 101: same missing API; no dedicated tests execute |
| `cargo test -p texelith-gpu --locked` | RED, exit 101: same target prevents full test execution; existing GPU contracts not certified |
| Task's exact release `verify_mvp --suite poisson --size 4096 --backend dx12 --adapter-name "RTX 5070 Ti" --report target-evidence/T-050/poisson.json` command | RED, exit 1 after release compilation: `unknown or empty suite: poisson`; no report or hardware timing produced |
| Task's exact `measure-faz0.ps1` command | DEFERRED to orchestrator; never launched |

The four isolated red cases are two `MaterialPlan` tests returning
`Unsupported { node: NodeId(2) }`, suite registration rejecting `poisson`, and
the generic report validator rejecting that same unregistered suite. The six
passes are the four numerical CPU anchors, binary16 fixture anchors, and
full-node p95 aggregation. This diagnostic subset is not the dedicated cargo
acceptance target and does not count the unexecuted GPU tests as passed.

Raw build/test/release logs and the generated diagnostic wrapper/binaries are
under ignored `target-evidence/T-050/` (`build.log`, `test-t050.log`,
`test-gpu.log`, `verify-poisson.log`, `public-compile.log`, `public-contracts.log`).
The wrapper includes the same oracle/GPU-helper/material/harness modules while
excluding the missing-API solver module; it contains no production substitutions.
GPU numerical behavior, cancellation, memory peaks and performance remain
unmeasured until implementation. No dependencies were added.
