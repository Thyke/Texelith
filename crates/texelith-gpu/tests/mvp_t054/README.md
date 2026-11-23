# Metallic mask test contracts

Numeric fixtures and the f64 reference are original. No downloaded assets are used.

## Public boundary

Worker tests use `MaterialGraphDto`, `MaterialNodeParams::Metallic`,
`MaterialRequest`, `MaterialPlan`, and `GpuWorker`. They do not inspect node
registration, dispatch matches, or runtime fields. The registration implementation
may change independently.

A `HeightSource` cannot feed `InputPort::Mask`. To exercise numeric masks
independently of the `PaintMask` producer, the kernel tests supply mask textures. Isolated numeric mask tests use this
small public leaf boundary, modeled on the existing `seamless_offset` prepared
kernel API:

```rust
// texelith_gpu::metallic
impl MetallicKernel {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self;
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        color: &wgpu::Texture,           // Rgba16Float linear RGB
        output: &wgpu::Texture,          // distinct R32Float, same extent
        mask: Option<&wgpu::Texture>,    // numeric R32Float, same extent
        uniform: &wgpu::Buffer,          // caller-owned 64-byte NodeUniform
        params: &MaterialNodeParams,    // Metallic variant, validated
    ) -> Result<PreparedMetallic, GpuError>;
}
impl PreparedMetallic {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder);
}
```

Callers supply all textures and uniform storage; worker allocations must remain
within the shared budget.

## Numerical details

ADR 0004 says “feathered smoothstep then threshold” but does not specify the
interval, orientation, or equality. The tests choose:

- `d = length(linear_rgb - picked_linear_rgb)`.
- For positive feather, `s = 1 - smoothstep(radius, radius + feather, d)`, then
  selection is `1` if `s >= threshold`, otherwise `0`.
- For zero feather, selection is exactly `d <= radius`, independently of
  threshold. Radius equality is selected. Radius zero selects only the center.
- Threshold zero with positive feather selects even zero coverage; threshold one
  selects only coverage one. These endpoint consequences are deliberate assertions.
- With an explicit mask `m`, Replace is `m`, Add is `clamp(selection+m,0,1)`, and
  Subtract is `clamp(selection-m,0,1)`. Mask samples are data, never sRGB or
  thresholded. Missing painted mask is constant one before composition in every mode.

The ADR/core `MaskInput::ConstantOne` contract requires:
absent and explicit constant-one masks must be identical. Replace and Add yield
one; Subtract yields zero. The tests include a
direct absent-versus-explicit-one regression for every mode and literal worker
default checks. Selection, photo decoding and edited-control assertions use an
explicit zero mask with Add through the public prepared kernel and production
ColorSource pass, preserving the original nonconstant numerical assertions.
Worker tests still cover absent defaults, caching, planning, proxy extents,
determinism, budgets and cancellation. Picker/selection assertions no longer
claim to observe color selection through a worker's absent ConstantOne mask.

## Coverage

- Hand-computed center, exact radius and threshold equalities, feather endpoints,
  zero feather/radius, primary colors, black/white and maximum RGB radius.
- Independent f64 CPU oracle with explicit `2e-5` scalar tolerance and finite
  `[0,1]` assertions. Photo oracle includes the actual half-float color boundary;
  direct numeric color fixtures use exactly representable halves.
- Linear picked color, exactly one inverse sRGB transfer for photos, ignored
  alpha, linear R32Float output and non-8/16-bit fractional mask precision.
- All mask modes, saturation, supplied zero/one/fractional masks and absent mask.
- Constants, ramps, impulse, deterministic seed, NPOT 257x193, 1x1, 1xN/Nx1,
  proxy extent/color controls, wrapped translation and identical seam edges.
- Parameter validation, edited-control recomputation, cache identity for every
  control/source/quality, and byte-identical cold-worker results.
- 4K planning within the shared 1 GiB ledger and exact admission boundary;
  worker pre-submission budget rejection and recovery.
- Deterministic scheduler cancellation of a known active token, plus actual
  worker supersession/recovery. Worker test does not pretend to force mid-kernel
  cancellation of a fast pointwise dispatch; the hardware suite must measure it.
- `metallic` membership in `maps`/`all`, unknown/empty suite rejection, full-node
  pass summation after exactly ten warmups and strict `<16 ms` report gate.

## Verification

Run `cargo test -p texelith-gpu --test mvp_t054 --locked` for the contracts and
`cargo run --release -p texelith-gpu --example verify_mvp -- --suite metallic
--size 4096 --backend dx12 --adapter-name "RTX 5070 Ti"
--report target/evidence/T-054/metallic.json` for hardware measurements.

The hardware suite performs CPU-oracle, NPOT, cancellation and budget assertions,
then records adapter/backend, manifest hash, submission latency, peak bytes and
100 uncached full-node timings after 10 warmups. Maximum settings are timed
separately. Synthetic report fixtures validate the schema and do not establish
hardware performance.
