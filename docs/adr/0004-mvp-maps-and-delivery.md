# ADR 0004: MVP maps, shared compute contracts, and delivery waves

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Milestones and wave graph

Every row is a barrier: integrate and pass its tasks before starting the next row.
Dependencies in task files add semantic prerequisites; they do not remove barriers.
Roles within a task run sequentially. At most three writers and one T4 writer run
per row. Shared GPU registration/worker files deliberately serialize map tasks.
Task scopes target 30-60 minutes of agent implementation, excluding automated
build/measurement waits and subsequent review; stop and return a proposed split
if the bounded slice cannot fit. No silent omission of acceptance to meet time.

```text
M1: source -> all procedural maps -> seamless -> selectable 2D demo
  W01  T-040 dependency/policy gate
  W02  T-041 graph ports | T-042 measurement harness | T-043 typed image inputs
  W03  T-044 map parameters/commands | T-045 material scheduling/budget
  W04  T-046 pyramid | T-055 stroke command model
  W05  T-047 height from albedo
  W06  T-048 normal filters
  W07  T-049 Poisson operators
  W08  T-050 Poisson solver
  W09  T-051 horizon AO
  W10  T-052 curvature
  W11  T-053 roughness
  W12  T-054 metallic selection
  W13  T-056 mask rasterization
  W14  T-057 offset blending
  W15  T-058 periodic splats
  W16  T-059 histogram preservation
  W17  T-060 seam heatmap
  W18  T-061 map-stack integration
  W19  T-062 M1 demo/gate
M2: M1 -> shared-device PBR -> Simple workspace demo
  W01  T-063 viewport resources | T-064 mesh IO | T-070 view commands
  W02  T-065 direct PBR | T-066 HDR IO | T-071 Simple layout
  W03  T-067 IBL | T-072 inspectors/painting
  W04  T-068 POM | T-073 shortcuts/drop/localization
  W05  T-069 tile and A/B integration
  W06  T-074 M2 demo/gate
M3: M2 -> engine exports -> reopenable batch project -> beta artifacts
  W01  T-075 export contracts | T-080 file encoders | T-089 platform bootstrap
  W02  T-076 preset data | T-077 naming/transactions | T-090 native CI matrix
  W03  T-078 packing | T-084 project DTO | T-091 Windows package
  W04  T-079 readback | T-085 project IO | T-092 macOS package
  W05  T-081 BC5 encoding | T-086 batch state | T-093 Linux package
  W06  T-082 export orchestration | T-094 package CI | T-095 engine fixtures
  W07  T-083 packing editor
  W08  T-087 batch execution
  W09  T-088 project/batch UI
  W10  T-096 full-set performance | T-098 platform evidence
  W11  T-097 workflow visual gate
  W12  T-099 final beta gate
```

## Context

The spec's MVP rows, GUI, performance and risks require a material graph beyond
Faz 0's single-input height chain. ADRs 0001-0003 remain authoritative except the
explicit extensions here and in ADRs 0005-0010. Current core exports GraphDto,
NodeParams, Session, node_cache_key; GPU exports GpuExecutor::prepare,
PreparedPass::encode and GpuWorker::{request,install_source,request_save}; IO
exports bounded decode and numeric PNG functions; UI exports run, DisplayBridge
and SharedGpu. Extend these services; retain four crates and their dependency DAG.
Core stays OS/GPU/egui-free and builds for wasm32-unknown-unknown.

Board History and T-021 handoff close Faz 0, including the later human check.
The older gate report correctly retains its earlier manual-access failures.
Its measured 1020 MiB peak leaves only about 4 MiB of the 1 GiB application budget:
more maps cannot simply be pinned into the existing cache.

## Decision

### Graph evolution and compatibility

Supersede ADR 0002's single edge with ordered, typed input ports and named output
references. Graph DTO schema 2 migrates diagnostic schema 1 without changing its
clamped, central-difference, GL numerical behavior. Keep legacy constructors and
the old four-node fixture usable; new material construction is a separate factory.
Cache encoding schema 2 includes port/output tags, all typed parameters, boundary,
quality settings, seeds, stroke digest and algorithm revision. Never hash paths,
generation or view state. Preserve old schema-1 golden tests as migration evidence.

```rust
pub enum Boundary { Clamp, Repeat }
pub enum NormalConvention { OpenGl, DirectX }
pub struct OutputRef { pub node: NodeId, pub port: OutputPort }
pub struct InputEdge { pub port: InputPort, pub source: OutputRef }
pub struct MaterialOutputs { /* map kind -> validated OutputRef */ }
pub fn material_from_source(source: SourceDescriptor) -> Result<Graph, CoreError>;
// Session::dispatch remains the only mutation entry; snapshots remain immutable.
```

Validate cycles, arity, port semantic types, extents, finite parameters and ranges;
limit graphs to 256 nodes, 16 pyramid bands and 4096 per axis. Optional mask ports
have explicit constant-one defaults, never missing arbitrary references. Material
outputs may share ancestors. Imported normals select a separate HeightFromNormal
branch; do not wire Height->Normal->Height into a cycle. All normal-valued edges
carry convention metadata; consumers canonicalize to GL once. Preview and export
decode that metadata, avoiding a second Y flip. Displacement aliases height plus
scale/midlevel metadata until final conversion; smoothness is 1-roughness.

| Node / output | Inputs | GPU format and numerical contract |
| --- | --- | --- |
| ColorSource / albedo | decoded sRGB RGBA8 | Rgba16Float linear RGB, alpha 1 |
| HeightSource / height | linear u16/f32 input | R32Float, no transfer |
| NormalSource / normal | RGB8 encoded xyz + convention | Rgba16Float signed, normalized |
| HeightFromAlbedo | linear color | R32Float; weighted Laplacian bands + coarse base |
| NormalFromHeight | height | Rgba16Float; central legacy, Sobel or Scharr; multi-scale |
| HeightFromNormal | normal | R32Float; mean-zero Poisson result, remap is explicit |
| HorizonAo | height | R32Float [0,1], 1 is unoccluded |
| Curvature | normal | R32Float signed curvature + separate Edge/Cavity R32 outputs |
| Roughness | color, optional mask | R32Float [0,1], piecewise-linear curve then inversion |
| Metallic | color, optional painted mask | R32Float [0,1], range/threshold/feather |
| PaintMask | stroke data | R32Float [0,1] |
| Seamless | color or height | preserves input format; ADR 0005 |
| Levels / Blur | height | retain existing R32Float behavior |

Photo RGB uses ADR 0003 sRGB decoding and Rec.709 luminance, ignoring ICC/EXIF as
before with visible import assumptions. Extend IO with bounded linear PNG16 and
EXR32 height and normal import; old RGBA8 source hashes remain valid. Typed source
hashes include channel layout, canonical decoded bytes, interpretation and extent.
No implicit photo conversion on scalar data. No AI de-lighting claim.

### Shared WGSL and pyramid

Preserve group 0 bindings 0=input texture, 1=distinct write-only storage output,
2=64-byte NodeUniform. Group 1 is node-specific additional inputs/weights/storage
buffers with Rust ABI tests; declare layouts explicitly. 8x8 workgroups, bounds
checks and checked byte counts are mandatory. R32Float uses textureLoad/manual
bilinear, not a required float32-filtering feature. No read/write texture alias.
Common address helper is signed positive modulo for Repeat and clamp otherwise;
singleton-axis derivatives are zero. New tileable material defaults to Repeat;
legacy graph migration defaults Clamp. Include boundary in cache identity.

Pyramid: separable normalized five-tap [1,4,6,4,1]/16, then area-aware downsample
to ceil(n/2), terminating at 1x1 (max 13 levels at 4K). Pixel-center mappings,
odd NPOT area weights and wrap/clamp edges have CPU oracles. Bands are
G_l - upsample(G_l+1); weighted reconstruction plus coarse term produces height.
All weights 1 must reconstruct source within tolerance. Default bands 1; expose
coarse removal for low-frequency suppression, levels/min-max only when requested.
Constant min=max remap produces the configured midlevel (default .5), never NaN.
Pyramid keys include source extent, boundary and filter revision; share across nodes.

Normal gradients are in normalized UV units (image rows down). Sobel derivative
[-1,0,1] x [1,2,1]/8; Scharr [-1,0,1] x [3,10,3]/32, times the axis dimension.
Blend scale gradients before normalize((-strength*du,+strength*dv,1)); DX encoding
negates Y once. Weights/radii are in original source units for proxy consistency.
AO integrates max positive horizon slope over 8 directions x 8 steps by default,
radius 32 source pixels; user bounds directions 4..32, steps 4..32, radius 1..512.
Define AO as mean cos(max(horizon,0)) squared, with bias .01 UV-height slope units;
normalize sampling distances by source width/height. Flat height is exactly 1.
Curvature uses divergence of the normalized normal's tangent components under the
same row convention; a convex analytic bump is positive. Edge=max(c,0),
Cavity=max(-c,0), each scaled/clamped independently. Roughness uses luminance,
ordered finite curve knots (2..16), mask blend against base roughness, then invert.
Metallic selects distance in linear RGB to picked color (radius 0..sqrt(3)),
feathered smoothstep then threshold; zero feather is an exact step. Painted mask
combines via explicit replace/add/subtract mode, clamped, never inferred from color.

### Height from normal: geometric multigrid

Choose matrix-free geometric multigrid, not FFT: arbitrary NPOT and clamp inputs
need no hidden POT resizing. Convert canonical normals to du=-nx/max(nz,.05),
dv=ny/max(nz,.05); cap each slope to +/-20 and report clamped-pixel counts.
Minimize ||D h - g||^2 with consistent forward differences D and adjoint D^T.
Repeat uses periodic edges; Clamp uses natural Neumann edges (no outward edge).
Subtract RHS mean and project height to mean zero every V-cycle, fixing drift
without claiming recovery of unobservable absolute height or low frequencies.
Coarsen to ceil(n/2) with area-weighted restriction and bilinear prolongation;
use Galerkin-consistent coarse operators on odd grids, not the even-grid stencil
silently reused. Weighted Jacobi (omega 2/3), 3 pre/3 post smooths, 32 coarsest
iterations. Default 8 V-cycles, maximum 16; stop on relative L2 residual <=1e-3
(absolute <=1e-6 for near-zero RHS), reduce on GPU, read one scalar per cycle.
Nonconvergence is reported with residual; export fails unless explicitly accepted.
T-049 isolates operators/odd-grid adjoint tests; T-050 owns cycle orchestration.

### Scheduling and memory

Extend worker requests with a bounded material output set and job identity. Keep
latest interactive slot, one compute submission in flight, token checks between
passes, and existing queue-fenced DisplayLease retirement. Multi-output success
publishes one coherent generation; never mix old normals with new height.
Plan live ranges, evict unpinned cached intermediates before reservations, release
scratch at fences and stream exports one output at a time. Reserve viewport and
IBL in the same 1 GiB ledger (eframe surfaces separately reported). Reservations
vary by phase, not fixed per-crate quotas. The 4K viewport bundle is at most
448 MiB: half-float albedo/normal/packed AO-R-M (128 each) plus R32 height (64).
Keep derived edge/cavity unpinned. Budget up to 192 MiB solver/pyramid scratch,
144 MiB viewport targets/IBL, 64 MiB source and 64 MiB miscellaneous, totaling
912 MiB before readback. A 128 MiB readback requires retiring scratch first;
solver and export readback reservations never overlap at this peak. Actual sizes
and retired generations, not this estimate, authorize each allocation. Before a
new full material generation, retire the old material bundle after its draw fence
while retaining its display image; never pin two 448 MiB bundles. Stream/recompute
unpinned outputs. A/B uses proxy textures, not two full 4K sets. If a plan cannot
fit, return a bounded error. No unbudgeted kernel-local texture allocations.

### Measurement contracts

New pipelines compile lazily on the worker; warmup is explicit in timing reports.
T-042 fixes a seed and serialized parameter manifest per reference graph; report
its hash so defaults and quality cannot drift between benchmark tasks.
T-042 creates verify_mvp example: --suite NAME --size 4096 --backend dx12
--adapter-name "RTX 5070 Ti" --report PATH. Unknown suites and zero assertions
fail. Numerical suites add CPU oracles before WGSL; constants, ramps, impulses,
NPOT 257x193, 1xN/Nx1, boundary and deterministic seed tests are mandatory.
Scalar errors <=2e-5, normal components/unit length <=2e-3 unless a stricter
algorithm-specific bound is stated; solver residual is an additional criterion.
Each new map runs 10 warmups + 100 full-node measurements: p95 <16 ms, sum of
all passes, plus submission latency, peak bytes and cancellation errors. Timestamp
absence uses labeled completion upper bounds, never a skipped PASS. Adjustable
worst settings are measured separately from documented defaults, not concealed.
All map tasks also run unchanged scripts/measure-faz0.ps1, three runs. T-096 gates
default 4K full classic set through disk publication at <2 s (ADR 0006), three
independent runs; no warm-result cache masquerading as computation. Failed targets
require optimization or an explicit orchestrator disposition, not relaxed gates.

## Alternatives considered

- FFT: efficient periodic POT integration, but NPOT padding and clamp boundary
  semantics complicate this MVP; multigrid operators are split into two tasks.
- One giant map implementation: hard to validate numerics and monopolizes a writer.
- All outputs permanently resident: does not fit observed memory headroom.
- R16 scalar or new device per preview: precision loss or incompatible leases.

## Consequences

Schema migration and measured memory eviction precede visual features. Algorithm
quality and <16 ms at 4K are gates, not unverified promises. Original algorithms,
synthetic fixtures and permissively licensed dependencies only; no GPL source.
Diagnostics examples are test harnesses, not the deferred end-user CLI/MCP.

## Open questions for the human

Confirm the proposed default quality/time tradeoff if multigrid or histogram nodes
miss <16 ms on target hardware; no implementation task may pre-approve that waiver.
The existing 4K cap and 1 GiB budget remain; 8K and general CPU fallback stay deferred.
