# T-054 tests-first handoff

The original tests-first target contained 22 tests. No production code, node registry entry, shader,
manifest, dependency, or existing test was changed. Device/readback plumbing is
adapted from the repository's existing permissively licensed test helpers; numeric
fixtures and the f64 reference are authored here. No downloaded assets are used.

## Public boundary

Worker tests use `MaterialGraphDto`, `MaterialNodeParams::Metallic`,
`MaterialRequest`, `MaterialPlan`, and `GpuWorker`. They do not inspect node
registration, dispatch matches, or runtime fields, so T-121 may refactor those.

There is currently no public numeric mask source. A `HeightSource` cannot feed
`InputPort::Mask`, and using `PaintMask` would introduce the expressly excluded
T-056 producer dependency. Isolated numeric mask tests therefore propose this
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

Only `kernel.rs` depends on this proposed API. It is intentionally compile-red
until implementation supplies it. An equivalent public prepared-node API may
replace the spelling in that adapter without changing the behavioral assertions.
No production stubs were added. Callers supply all textures and uniform storage;
this API proposal does not authorize unbudgeted worker allocations.

## Provisional numerical details needing orchestrator disposition

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

Review round 1 clarified the accepted ADR/core `MaskInput::ConstantOne` contract:
absent and explicit constant-one masks must be identical. Replace and Add yield
one; Subtract yields zero. The revised target contains 23 tests, including a
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

## Remaining implementation/performance responsibilities

The numeric mask kernel and all worker metallic tests are expected red until
implementation. Suite membership/report tests also remain red until the producer
registers and implements `verify_mvp --suite metallic`.

The implementer/perf pass must add the real 4K suite with 10 warmups and 100
uncached full-node measurements, CPU-oracle/NPOT/cancellation/budget assertions,
adapter/backend identity, manifest hash, submission latency, peak bytes, and a
separate maximum-setting timing. The synthetic JSON in tests is only a validator
fixture, never hardware evidence. No performance result is inferred from unit
tests. Run the required acceptance commands after implementation and T-121 merge.

`scripts/measure-faz0.ps1` requires a foreground native window and is deferred to
the orchestrator per dispatch; its thresholds must remain unchanged.

## Checks in the tests-first pass

- `cargo build -p texelith-gpu --all-targets --locked`: FAIL, exit 101,
  unresolved `texelith_gpu::metallic` in the proposed numeric-mask adapter.
- `cargo test -p texelith-gpu --test mvp_t054 --locked`: FAIL, exit 101,
  same expected compile-red import.
- `cargo test -p texelith-gpu --locked`: FAIL, exit 101, same compile-red
  test target; no claim that the existing runtime suite executed.
- Required release `verify_mvp --suite metallic --size 4096 --backend dx12
  --adapter-name "RTX 5070 Ti" --report target-evidence/T-054/metallic.json`:
  FAIL, exit 1, `unknown or empty suite: metallic`. No 4K performance or
  maximum-setting timing was measured and no successful report was produced.
- Standalone `rustc --test reference.rs`: PASS, 3/3 oracle self-tests.
- Supplementary standalone test harness excluding **only** `mod kernel`:
  compiled against the built repository rlibs, then ran all 18 existing-API
  tests serially. Result: 5 PASS, 13 expected FAIL. Passing tests are the three
  oracle checks, parameter validation, and scheduler cancellation. Failing tests
  reach `Unsupported { node: NodeId(2) }` or missing metallic suite registration.
  Adapter observed: NVIDIA GeForce RTX 5070 Ti, DX12, driver 32.0.16.2114.
  This supplementary run does not replace the failing Cargo target.
- `rustfmt --check` on the six owned Rust files: PASS.
- `git diff --check`: PASS. All delivered source files are new and confined to
  `tests/mvp_t054.rs` and `tests/mvp_t054/`.
- `measure-faz0`: deferred to orchestrator, not run.

Raw logs and the supplementary harness are local, ignored artifacts under
`target-evidence/T-054/`: `build.log`, `dedicated-tests.log`, `all-tests.log`,
`metallic-command.log`, `public-worker-compile.log`, `public-worker-tests.log`,
and `public-worker-tests.rs`. There are no new dependencies.
