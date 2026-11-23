# T-060 tests-first contract

This pass adds tests and an independent f64 CPU oracle only. It does not add a
shader, production implementation, registry entry, stub, dependency or fixture
download. The dedicated target contains 20 tests. Fixtures are generated in
memory, including NPOT 257x193 and singleton axes.

## Proposed public boundary

There is no seam diagnostic core graph variant or public GPU API in this
checkout. Only `node.rs` anticipates this leaf API, following `PreparedCurvature`:

```rust
// texelith_gpu::seam_heatmap (public module)
impl PreparedSeamHeatmap {
    pub fn new(
        executor: &GpuExecutor,
        input: &wgpu::Texture,       // actual R32Float or linear Rgba16Float output
        heatmap: &wgpu::Texture,     // distinct R32Float, same extent, linear magnitude
        pool: Arc<Mutex<ResourcePool>>,
    ) -> Result<Self, GpuError>;
    pub fn passes(&self) -> &[PreparedPass];
    pub fn mark_submitted(&mut self, fence: FenceId) -> Result<(), GpuError>;
    pub fn complete(&mut self, fence: FenceId);
    pub fn read_metrics(&self) -> Result<SeamMetrics, GpuError>;
}
pub struct SeamMetrics {
    pub boundary_rms: f32,
    pub interior_rms: f32,
    pub ratio: f32,
}
```

`read_metrics` is called after completion on this test worker, never on a UI
thread. Metrics must come from the diagnostic's GPU reduction of the bound
texture. Full-node timing must include all its passes and completion/readback;
tests do not authorize a CPU per-pixel implementation. Input/output texture
payloads are supplied by the caller, and all internal scratch/readback must be
admitted to the shared pool. No production stubs were added. An equivalent
public prepared API can change the spelling in `node.rs` without changing the
behavioral assertions.

## Explicit numerical choices for review

ADR 0005 requires wrapped gradients, four-pixel bands, RMS and a finite ratio;
it does not fix the difference stencil, band anchor, RGB norm, interior location
or zero-denominator value. This contract makes those choices explicit rather
than using whichever formula an implementation happens to expose:

- Forward difference from `(x,y)` to its wrapped right/down neighbor, in pixel
  value units. A singleton axis contributes neither gradient nor RMS samples.
- Scalar distance is absolute difference. Color uses RMS over **linear RGB**,
  excluding alpha, so chromatic discontinuities cannot disappear at equal
  luminance. Source decoding is checked independently; the oracle receives
  actual half-float texture values, avoiding an invented half rounding tolerance.
- Heatmap at each pixel is `sqrt(dx_squared + dy_squared)` over the whole
  texture, without normalization, transfer functions or clipping. A forward
  wrap seam is located in the last column/row. A zero map can be displayed as
  black; colorization belongs to the viewer.
- For each axis of length `n>1`, let `b=min(4,floor(n/2))`. Boundary gradient
  start coordinates are `[0,b)` and `[n-b,n)`. Interior start coordinates are
  `[floor(n/2)-b,floor(n/2)+b)`. Count each directional edge once in its mask,
  using the full orthogonal dimension. Corners carry separate x/y samples.
  Small extents can have overlapping masks; membership never duplicates an
  edge within a mask. A 1x1 image has empty masks and zero metrics.
- RMS is `sqrt(sum(squared_directional_difference)/sample_count)` per mask;
  empty masks yield zero. Ratio is `boundary_rms/max(interior_rms,1e-8)`.
  Flat input yields exactly zero, while a nonzero seam with zero interior has
  a large finite ratio. RMS/heatmap tolerance is `2e-5`; ratio tolerance is
  `2e-5 * max(1, expected_ratio)` for reduction rounding.
- Periodic pixel-center cosine fixtures have a small nonzero finite-difference
  gradient, so absolute boundary RMS must be below `5e-4`, not mathematically
  zero. Injecting a unit defect on either edge must raise RMS above `.2`, ratio
  above `100`, and RMS above `400` times the undamaged value. A ramp's literal
  expected ratio and a step's literal `.25` RMS prevent a zero-only oracle from
  accepting an implementation.

The implementer/reviewer should resolve these unspecified conventions explicitly
if a different contract is needed. Do not silently alter the assertions to fit
an implementation.

## Behavioral coverage and expected red

`behavior.rs` feeds `TypedSourcePixels::{HeightF32,HeightU16,Rgba8}` through
`install_typed_source`, public `MaterialRequest`/`MaterialGraphDto`, and real
`CompletedMaterial` textures. The source-only path tests untreated discontinuous
input; an existing Levels node proves diagnostics observe processed output.
No seamless repair producer or other task's test/leaf module generates inputs.

Tests assert every heatmap pixel and both RMS metrics against the independent
oracle, plus literal seam locations, exact four-pixel membership on both axes
and sides, equal-width interior masks, finite zero-denominator behavior, flat
and periodic inputs, injected seams, NPOT, impulses, scalar precision, sRGB once,
chromatic seams, ignored alpha, unchanged input bytes and deterministic results.
`node.rs` also asserts alias/extent/format rejection, scratch admission failure,
recovery and release after completion. The report contract rejects missing
behavioral/budget/cancellation assertions, encode-only timings, insufficient
samples, default p95 >=16 ms, over-budget memory, and absent maximum-quality
timing. Report JSON is a schema fixture, never performance evidence.

All new GPU diagnostic tests are compile-red until the proposed API exists.
Suite selection and report validation are red until `seam_heatmap` is registered
with its runner and report validator in the appropriate aggregate. The four CPU
oracle self-tests and two input/readback plumbing tests pass independently.

Actual 4K timing, maximum-quality timing and generation cancellation during the
diagnostic are still implementation/perf responsibilities. The report assertions
require those checks but do not pretend to execute a nonexistent diagnostic.
The native foreground-window gate is deferred to the orchestrator.
