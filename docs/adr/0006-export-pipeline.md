# ADR 0006: Engine presets, channel packing and transactional exports

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

MVP requires engine-ready sets, arbitrary RGBA packing, naming, displacement and
BC5 normals in the Unreal preset. Existing IO writes height PNG16/normal PNG8;
GpuWorker::request_save returns one packed numeric readback and cancels on edits.
Extend this path, keeping its legacy single-map behavior. Compression preview
and BCn validation UI are Faz 2; generating the explicitly requested BC5 file is
MVP. Preset mappings below are versioned product contracts, to be checked against
engine import fixtures in T-095; they are not claims of tested engine integration.

## Decision

Core owns ExportPlan, named channel selections, conversions and preset DTOs. GPU
owns full-resolution evaluation, channel conversion, packing and bounded readback.
IO owns encoders, paths and transaction publication. UI coordinates typed effects;
neither GPU nor IO imports the other, and core never holds native paths.

```rust
pub enum ChannelSource { Map(OutputRef, Component), Constant(f32) }
pub enum Transfer { Linear, Srgb }
pub struct ChannelSpec { pub source: ChannelSource, pub invert: bool,
                         pub transfer: Transfer }
pub enum ExportFormat { Png8, Png16, Exr32, DdsBc5 }
pub enum SizePolicy { Keep, Resize { width: u32, height: u32 },
                      Pad { width: u32, height: u32 } }
pub struct ExportPlan { /* snapshot identity, outputs, preset revision, policy */ }
pub fn validate_export(plan: &ExportPlan) -> Result<(), CoreError>;
// IO takes validated CPU planes/blocks and an owned native destination separately.
```

Every output declares channels, scalar/vector interpretation, bit depth and color
intent. Apply sRGB encode only to albedo RGB at the last conversion; alpha/data
always linear. Default packed maps are fully linear and carry no sRGB tag. A
custom mixture of albedo and data uses per-channel numerical transfer with a raw
file tag and an explicit manifest warning: engines cannot apply sRGB selectively
to RGB channels in one texture. Reject presets claiming engine-safe sRGB sampling
for such mixtures. Select normal xyz component only after conversion to the chosen
convention; then encode 0..1. Resampled normals are renormalized before packing.
Source alpha is ignored in Faz 1; generated output alpha defaults to 1.

| Preset ID | Albedo | Packed/other data | Normal / import instruction |
| --- | --- | --- | --- |
| unreal-v1 | sRGB RGB PNG8 | ORM RGB = AO, Roughness, Metallic; height PNG16 | DX; DDS BC5_UNORM, optional RGB PNG8 source; sRGB off, normal compression |
| unity-urp-v1 | sRGB RGB PNG8 | MetallicSmoothness RGBA = M,0,0,1-R; separate occlusion G=AO, R/B=1 | GL RGB PNG8; import as normal, data sRGB off |
| unity-hdrp-v1 | sRGB RGB PNG8 | Mask Map RGBA = M,AO,detail-mask(1),1-R | GL RGB PNG8; linear Mask Map |
| godot-v1 | sRGB RGB PNG8 | ORM RGB = AO,R,M | GL RGB PNG8, linear ORM |
| blender-v1 | sRGB RGB PNG8 | separate R,M,AO PNG8; height PNG16 | GL RGB PNG8; Non-Color data, Normal Map node |
| gltf2-v1 | sRGB RGB PNG8 | ORM RGB = AO,R,M; occlusion uses R of same image | GL RGB PNG8; metallicRoughness uses G/B |

No Unity Mask Map shortcut that conflates URP and HDRP. glTF preset exports
textures and an assignment manifest, not a glTF scene/material file. Presets
include explicit defaults for absent maps, normal green orientation, roughness
inversion and engine import hints. Custom channel presets are saved versioned
JSON beside user projects, with collision-safe names and the same validation.
Built-in presets are typed static core tables with JSON golden fixtures checked
using dev-only serde_json; production JSON parsing remains entirely in IO.

PNG8 covers albedo/normals/scalars and packed RGB(A); PNG16 covers scalar height
and optional high-precision channel packs. Preserve existing numeric PNG APIs.
EXR32 is in MVP for true float displacement: a single named Y channel, lossless
compression, linear values, no half-float cast. Store signed displacement as
d=(h-midlevel)*scale, midlevel default .5, scale default 1; keep negative values
and units metadata. PNG16 stores normalized h, with midlevel/scale in manifest;
do not pretend unsigned PNG directly represents signed distance. Explicit clamp
ranges and optional deterministic ordered dither apply only to integer quantizing.
KTX2 and other BC formats are deferred: no consumer requirement outweighs another
container/transcoder dependency at MVP. DDS BC5 needs no added codec dependency:
original bounded BC4 endpoint/index fit for R and G per 4x4 block, packed as BC5;
normal Z reconstructs positive sqrt(max(0,1-x*x-y*y)). Reject negative-Z normals
for BC5 with a clear error; PNG is available for full XYZ. Encode mips down to
1x1 with normal renormalization; include DX10 DDS header with BC5_UNORM format.
T-081 checks against a separately written decoder, not the encoder's own helper.
Default BC5 angular p95 <=5 degrees on the fixed positive-Z smooth-normal fixture;
report worst error and steep-normal degradation. This is codec correctness testing,
not the deferred interactive BC preview feature.

Default Keep preserves NPOT dimensions. Request POT via explicit Resize or Pad
(next power of two per axis); preview the new size before export. DDS base width
and height must be multiples of four: Keep rejects nonmultiples, never silently
resizes. Tiny mip blocks use replicated edge texels and valid logical dimensions.
Linear-color/scalar area downsample and bilinear upscale use the shared boundary;
vectors renormalize. Padding uses Repeat or Clamp, recorded in the manifest with
valid-content rectangle and UV scale. Reject dimensions above 4096 and zero sizes.
POT resize is not enabled merely because an engine can compress textures.

Naming supports {name}, {map}, {preset}, {index}; default T_{name}_{map}, stable
batch index padded to four digits. Reject unknown tokens, rooted paths, separators,
dot traversal, empty stems, Windows device names and forbidden/control characters;
normalize duplicate names with an explicit index policy, not silent overwrite.
Preflight all case-insensitive collision keys on every OS for portable exports.
Default no-overwrite. Write each set to a unique sibling staging directory, sync
files, then rename that directory into a new destination. Replacing a set requires
explicit UI selection and backup/rollback protocol; multi-file replacement is not
claimed atomic on all OSes. On crash retain a recoverable transaction manifest;
on error/cancel do not advertise a complete set. Never truncate existing files.

One immutable export job captures graph/source hashes/preset/seed/output list.
Editing the live graph does not mutate that job; this explicitly extends ADR
0003's cancellation-on-edit for material sets only. A document close offers cancel
or completion; the legacy SavePng behavior stays unchanged. Interactive requests
take priority between bounded passes; one GPU submission, one pending export,
one mapped readback plane and one bounded IO encode job. Map staging rows honor
256-byte pitch and keep leases until queue completion. Cancellation retires work
and cleans staging without publishing a partial set. Progress tracks evaluation,
readback, encode and publication separately; errors retain output identity.

T-096 performance fixture: resident decoded 4096-square input, default procedural
graph with height/albedo, Scharr normal, default AO, curvature, roughness, metallic,
offset seam mode; shaders warmed but all result caches invalidated. Time request
acceptance through compute, readback, encode, flush and final set rename on local
SSD. Canonical outputs are albedo RGB8, normal RGB8, AO/R/M/edge PNG8 and height
PNG16 (seven images); each of three runs <2 s. Measure Unreal DDS+ORM and EXR32
variants separately against the same <2 s target; list every failure independently.
Record decode and cold compile separately, settings, file bytes, adapter, drive,
GPU timings and CPU encode time. No cached-only or encode-excluded substitute.

## Alternatives considered

- One global sRGB switch: corrupts numerical packed channels.
- BC5 only as an import hint: fails the explicit MVP BC5 export requirement.
- KTX2 now: unnecessary container and compression tooling for the named consumers.
- Signed displacement in unsigned PNG without metadata: ambiguous zero and range.
- Overwrite files individually: leaves apparently valid but mixed-generation sets.

## Consequences

EXR adds one audited permissive dependency (ADR 0010). BC5 implementation is
isolated from its DDS writer. Engine profile revision travels with every export.
Throughput is a measured product gate; slow lossless encoders may need optimization.

## Open questions for the human

Confirm minimum engine versions for real import smoke testing in T-095; absent
installed engines, ship truthful assignment contracts and mark engine runtime
coverage unverified. No BCn preview or KTX2 support is implied by DDS export.
