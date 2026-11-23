# ADR 0003: Shared-device compute, texture lifetime, and responsiveness

Status: accepted (user-approved 2026-11-10).

## Context

The spec requires a render worker, generation cancellation, downstream caching,
1K drag previews, linear data processing, and nonblocking UI. eframe already owns
the presentation loop. Independent wgpu devices cannot share texture views.
Target hardware is Windows 11 / RTX 5070 Ti with UHD 770 also present; adapter
selection must be recorded, not assumed. No GPU/API behavior was tested in T-001.

## Decision

### Device, threads, and public boundary

Let eframe create the window, wgpu instance/adapter/device/queue and egui renderer.
Request DX12 and a high-performance adapter by default. Support explicit backend
and adapter selection for diagnostics; report the actual adapter, driver/backend,
limits and optional timestamp features. Fail visibly if required storage formats
or 4096 texture dimensions are unavailable; do not silently switch to CPU rendering.
Query/request needed limits through eframe configuration before device creation.
Verify 0.31.1 bootstrap hooks in T-013; report any incompatible API before changing
the architecture. Never create another device for production compute.

Pass cloned shared device/queue handles to one named compute/render worker. The
UI thread owns egui state and renderer registration and presents every frame.
The worker owns compute pipeline creation, encoding, submissions, cache, polling
and readback. The spec's "GPU submission thread" means application compute;
eframe's own presentation submissions remain on eframe's thread.
One I/O worker decodes/encodes with bounded requests; no rayon/runtime is needed.
UI update never waits on a channel, joins, maps a buffer, or polls the GPU to wait.
Shutdown closes mailboxes; worker joins happen outside the frame update loop.

```rust
pub struct RenderRequest {
    pub request_id: u64, pub snapshot: Arc<EvaluationSnapshot>,
    pub output: NodeId, pub extent: Extent, pub quality: Quality,
}
pub enum Quality { Proxy, Full }
pub struct CompletedFrame {
    pub request_id: u64, pub generation: u64, pub quality: Quality,
    pub extent: Extent, pub lease: DisplayLease, /* owns texture + view */
}
impl GpuWorker {
    pub fn spawn(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
        config: GpuConfig) -> Result<Self, GpuError>;
    pub fn request(&self, request: RenderRequest) -> Result<(), GpuError>;
    pub fn try_latest(&self) -> Option<CompletedFrame>;
}
```

The app associates immutable decoded input with its content hash and installs it
before a request referencing it. API includes bounded source upload and export
requests plus nonblocking error/progress events; never copy pixel arrays per drag.
`request` replaces a single latest pending slot and cannot block on a full queue.
Implement that slot with a short mutex swap plus a capacity-one wake channel;
never hold the lock while hashing, encoding, polling or submitting. Completion
publication also replaces a single latest slot, releasing superseded leases.

### Formats, color, and compute ABI

| Role | Format / interpretation |
| --- | --- |
| Decoded source upload | `Rgba8Unorm`, unassociated alpha; immutable |
| Source luminance, levels, blur | `R32Float`, scalar linear height |
| Normal result | `Rgba16Float`, signed unit xyz, alpha 1, linear data |
| egui viewport staging | `Rgba8Unorm`, opaque display codes, not a data cache |

Use R32Float for height to avoid banding and baseline R16Float storage capability
assumptions. Rgba16Float is ample for normals; Rgba32Float is unnecessary.
Check storage-write/sample/copy usage support. Data textures use
`TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC`; add COPY_DST only when needed.
No sRGB storage textures. PNG export is quantized only at its final conversion.

Common group 0: binding 0 `texture_2d<f32>` (unfilterable float, no sampler),
binding 1 write-only storage output (R32Float or Rgba16Float per pipeline),
binding 2 read-only uniform buffer visible to compute. Use `textureLoad` and
explicit clamped coordinates. Workgroups 8 x 8 x 1, ceil division, bounds check.
Bind a different output allocation from every input; never read/write one texture
in-place. Uniforms are a 64-byte, 16-byte aligned structure: four vec4 slots for
evaluation/source dimensions (`u32`), parameters (`f32`), pass flags (`u32`),
and reserved zeros (`u32`). Rust uses repr(C, align(16)) plus bytemuck assertions.
Bind-group layouts differ only in output storage format. Source-preparation and
display conversion use the same convention with their respective output formats.
Blur additionally uses group 1 binding 0, a 128-byte uniform block of eight
vec4<f32> slots for up to 31 normalized weights plus padding; this is separate
from the common 64-byte node uniform. Other pipelines have no group 1.

- Photo: apply the exact sRGB inverse transfer to RGB, then Rec.709 luminance
  `0.2126*r + 0.7152*g + 0.0722*b`; linear-height import uses red without transfer.
  Alpha is ignored explicitly; no ICC/EXIF orientation handling in this phase.
  Document that assumption; reject unsupported formats/bit depths, not silent cast.
- Levels: `pow(clamp((h-black)/(white-black),0,1), 1/gamma)`.
- Blur: normalized separable Gaussian, two passes, clamp boundary. Sigma is in
  original source pixels; scale sigma independently by evaluation/source size per
  axis. Radius ceil(3*sigma_axis), max 15; sigma zero is an identity/cache alias.
  Compute weights once per request/pass, not exp() inside every pixel tap.
- Normal: centered differences on normalized UV, `du=(hR-hL)*width/2`,
  `dv=(hDown-hUp)*height/2`; `normalize(vec3(-strength*du,+strength*dv,1))`.
  Image rows point down, tangent +Y points up (OpenGL-style +Y convention).
  Clamp neighbors; singleton axis derivative is zero. This UV-space strength
  convention makes proxy/full slopes comparable; it is not physical meters.
- Proxy: preserve aspect, longest axis <=1024, each dimension >=1. Generate a
  linear-height box/area downsample from the source, then run the same three
  kernels at that extent. No repeated sRGB conversion, normal downsampling, or
  changes to the full-resolution graph parameters.
- Display: height -> clamped grayscale codes; normal -> xyz*0.5+0.5 RGB codes.
  These are diagnostic map views, not lit surfaces. Use the texture convention
  expected by egui-wgpu (gamma-coded Rgba8Unorm texels, no automatic sRGB decode).
  A ramp/known-normal visual test must detect double transfer or channel flips.
  A future albedo view applies linear-to-sRGB exactly once; data maps never do.

PNG output: height as 16-bit grayscale, normal as 8-bit RGB with encoded signed
components; no sRGB tag on numeric maps. Handle 256-byte readback row padding and
strip it before encoding. Export evaluates the selected graph snapshot at full
size and reuses the same shaders; it must not save the proxy or viewport staging.
Allow one export request; reject another as Busy. Cancel with a visible error if
its graph/source changes before completion. Interactive work retains priority.

### Cancellation, cache, and lifetime

Content keys follow ADR 0002, including shader revision and actual extent. Cache
unchanged ancestors independently; changing strength must not rerun levels/blur.
Never include generation in the content key. Track generation AND request_id in
publication: quality/output can change without a parameter edit. A delayed proxy
cannot overwrite the current full output. UI checks both again before display.

At most one application compute submission is in flight and one request pending.
Before encoding, between node batches and after GPU completion, compare tokens
with latest intent. Already submitted GPU work cannot be cancelled. Do not enqueue
the next obsolete node batch; retire it after completion without publication.
Cancellation counters do not themselves drain GPU queues: this one-in-flight
bound is mandatory. Refresh completed-work callbacks by nonblocking worker polling.
During drag request proxy; on release request full even if the generation did not
change. Keep the last valid image visible while the new one computes.

Pool by exact (extent, format, usage), with byte accounting, last-use stamp and
three states: in-flight, externally leased, reusable. Cache entries hold leases;
evict least-recently-used unpinned entries before allocation. The default hard
application texture/buffer budget is **1 GiB**, including staging, transient,
cached and retired allocations; eframe/window surfaces are reported separately.
A 4K R32 texture is 64 MiB, Rgba16Float 128 MiB, Rgba8 64 MiB. Plan the peak
working set before dispatch, reserve it, and reuse blur scratch after completion.
Never evict in-flight or UI-held resources to satisfy a budget. If it cannot fit,
return a bounded error; do not spin, overallocate or silently reduce full quality.
Bound imported CPU bytes too: max 64 MiB decoded RGBA8, checked dimensions and
decoder allocation limits (256 MiB), one pending decode, no unbounded pixel queue.

The app registers the completed display view through egui-wgpu's native texture
API using eframe's renderer and a linear sampler. Store TextureId plus owning
DisplayLease. Registration is UI-thread work; rendering never reads back pixels.
Free a replaced TextureId only when it cannot be used by a future egui frame;
keep the underlying lease until all earlier UI submissions referencing it have
completed. Establish that fence in eframe's after-render lifecycle: a queue
completion callback after the final referencing submission, acknowledged back to
the worker. Verify the exact hook/order in T-013/T-019. Do not invent an N-frame
delay as a fence. If a reliable hook is unavailable, use immutable display
allocations and drop their last handle after renderer unregistration; exclude
them from reusable pool entries and rely on wgpu's submitted-resource lifetime,
with conservative retirement accounting until queue completion. Never overwrite
a texture still registered with egui. Device loss yields a visible error and
stops new dispatch; automatic device recreation is deferred.

### Verification and Faz 0 gate

Use original CPU reference algorithms and generated fixtures, no GPL code/data.
On hardware: levels/blur maximum absolute error <=2e-5; half-float normal component
error <=2e-3 and decoded normal length error <=2e-3. Include constant, ramps,
impulse, NPOT, 1xN/Nx1, edges and proxy/full analytic slope comparisons. Hash tests
are exact; cross-backend floating results have tolerances. GPU adapter absence is
a failure of explicitly requested GPU checks, never a passing skipped test.

Release-mode perf scenario uses deterministic 4096 x 4096 input, source/levels/
blur warmed, normal strength swept 0..10 at 60 input events/s for 10 seconds,
then release to full. Run three times on AC power, DX12 RTX 5070 Ti, 60 Hz target
presentation (record actual monitor refresh and present mode), no debugger.
Instrument the real eframe update/present path and the same gesture/command
handler as human input; CPU/GPU microbenchmarks alone cannot pass the slider gate.
Record all samples, adapter/driver, commit/worktree identity, shader hashes,
resolution, build/profile, warm-up, skipped/coalesced requests, frame publication
age, CPU/GPU/presentation timing, peak app allocations and validation errors.

Per run: UI update p95 <=4 ms; rendered-frame intervals p95 <=20 ms,
p99 <=33.4 ms, maximum <=100 ms during the drag; fresh proxy publication age
p95 <=50 ms (not just fast frames repeating an old texture); release-to-visible
full-resolution <=250 ms. Timestamp all boundaries with a monotonic clock; measure
publication age from represented input event, and also enforce no gap between
fresh proxy publications >100 ms during steady drag. Require >=500 drag frames
and >=100 fresh publications over the 10-second active interval.
Here "visible" means the first rendered/present-submitted frame using that
completed texture, measured at the verified eframe lifecycle boundary; OS
compositor/physical scanout latency is not measured. Record this limit explicitly.
Separately measure 100 warm 4K normal-only dispatches: GPU duration p95 <16 ms
when timestamp queries are supported; always report submission-to-completion
latency too. Without timestamps, require that conservative latency p95 <16 ms
and label it an upper bound, never report it as pure GPU time.
All three runs must pass; no dropped-outlier averaging. No validation errors,
stale output publications, unbounded pending work or >1 GiB app allocation.
Manual drag, undo, focus-loss and release verification is also required.
Measure cold start to first usable frame separately; report the spec's <1 s
target as PASS/FAIL, but it is not substituted for or folded into the slider gate.

## Alternatives considered

- Compute in `App::update`: risks shader compilation/encoding stalls and violates
  the spec's worker separation. Separate compute device: requires copies/readback.
- Every slider event becomes a job: stale generations still occupy the GPU queue;
  a latest-slot and one-in-flight limit actually bound latency.
- Readback to egui ColorImage: creates a GPU/CPU round trip each drag update.
- R16Float-only scalar pipeline or full Rgba32Float: capability/precision risk
  versus excessive memory. R32 scalar and half normals meet this small phase.
- Tiled execution and CPU fallback now: defer until larger inputs; enforce a
  checked 4K cap/budget and explicit errors rather than making unsupported promises.

## Consequences

Shared-queue lifetime ordering is the main integration risk and is tested early.
Heavy upstream edits may finish more slowly than strength-only edits; stale work
remains bounded. UI frame timing, proxy freshness and full-result latency are
separate measurements. The phase gate applies to the recorded RTX configuration,
not every GPU, backend, screen refresh, packaged build, or eventual 8K workload.

## Open questions for the human

- Approve the proposed frame/freshness/full-result thresholds and 1 GiB budget?
- Approve 16-bit height PNG / 8-bit normal PNG and explicit OpenGL convention as
  the only Faz 0 output options, with no ICC/EXIF management yet?
- If the first build cannot expose reliable shared-device retirement hooks,
  approve the immutable-display allocation fallback while retaining the same
  budget and responsiveness gate? Report measurements before choosing it.
