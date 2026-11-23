# ADR 0008: Shared-device PBR viewport and material inspection

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

DisplayBridge already registers immutable GPU textures with egui and retires them
by submission completion. SharedGpu exposes the eframe-created device and queue.
MVP adds a lit viewport, meshes, IBL, POM, tiling and A/B without a second device,
readback-driven rendering, a game engine dependency or engine-match shading claims.

## Decision

Keep renderer/pipelines in texelith-gpu, CPU mesh/HDR decoding in texelith-io,
camera/input and render intent in texelith-ui-egui. CPU mesh transfer types live
in core; vector/matrix arithmetic uses small original helpers, not a new engine.
Use the existing worker to render an offscreen color target, then publish through
DisplayBridge as a native egui image. This is rendering inside the egui viewport
on the same wgpu device; no custom egui callback submit or second swapchain.
GPU encoding is off the UI thread. UI asks for a new frame on material/camera/light
change or viewport resize; only latest intent retained. Background animation is
off by default, enabling idle power savings.

```rust
pub struct MeshData { pub vertices: Vec<Vertex>, pub indices: Vec<u32> }
pub struct Vertex { pub position: [f32; 3], pub normal: [f32; 3],
                    pub uv: [f32; 2], pub tangent: [f32; 4] }
pub struct PreviewRequest { /* material leases + camera/light + viewport extent */ }
// GpuWorker::request_preview replaces a bounded intent slot.
// CompletedFrame/DisplayLease retain existing presentation-fence semantics.
```

Color target Rgba16Float for linear HDR lighting, depth Depth32Float, single sample
initially. Tone-map into immutable Rgba8Unorm display codes using one explicit
linear-to-sRGB encode expected by DisplayBridge. Background and GUI colors never
pass through material tone mapping. Resize uses actual framebuffer pixels; cap
long axis 2048 and report preview scale when reduced. Previous valid frame remains
visible while compute runs. Account all targets, mesh buffers, environment and
retired frames against ADR 0004's shared ledger; preflight before allocation.

Group 0 holds camera/light/environment uniforms and IBL samplers/textures; group 1
holds canonical material textures (linear albedo, signed GL normal, R/M/AO/height)
and material uniforms; group 2 holds object transform. Provide explicit compatible
fallback constant textures for missing maps. Material output set is published
coherently. R32 height/scalars use manual bilinear where filtering is unavailable;
do not request float32-filterable merely for convenience. Construct a shading TBN
with tangent handedness and inverse-transpose normal transform; normalized GL
normal is converted once from any input convention before TBN application.
For the full 4K viewport, pack AO/roughness/metallic into one half-float texture
after generation; authoritative graph outputs remain R32Float for export. Retire
the previous material bundle before building the next, retaining its completed
display image as ADR 0004 requires. This bounds coherent publication memory.

BRDF is metallic/roughness Cook-Torrance: GGX NDF, correlated Smith visibility,
Schlick Fresnel with dielectric F0=.04 and metal F0=albedo, diffuse Lambert weighted
by (1-metallic) and (1-F). Alpha=max(roughness^2,.0025), with denominator epsilons.
One white directional light, intensity and rotation; separate environment exposure.
AO multiplies indirect diffuse only, never direct radiance or a blanket normal
shadow. Tone mapping uses a fixed documented ACES-style rational approximation
implemented originally, then exact sRGB transfer; exposure precedes tone mapping.
This is a neutral inspection model, not UE/Unity renderer equivalence.

IBL accepts Radiance .hdr equirectangular and linear EXR via bounded IO; procedural
sky is default, so no licensed bundled HDRI is necessary. Reject nonfinite pixels
and cap environment to 2048x1024. Build 256-face cubemap, 32-face diffuse irradiance,
128-face specular mip chain and 128-square BRDF LUT. Deterministic Hammersley
sampling (128 specular samples, 64 diffuse samples) on worker, bounded per-face/mip
submissions with cancellation. Convolution is cached by HDR content/settings.
IBL precomputation has separate cold-load timing; never hides inside slider timings.
Constant environment should produce spatially constant diffuse under rotation;
analytic energy/roughness tests precede visual metal/dielectric sphere checks.

Primitives: plane, cube, UV sphere and cylinder with correct seams/normals/tangents
and normalized bounds. Load static triangle glTF 2.0/GLB with gltf crate (MIT OR
Apache-2.0 candidate), OBJ with tobj (MIT candidate), audited in T-040. Disable
glTF automatic material/image import. Read bounded local geometry buffers only,
no URLs or network fetch; cap 64 MiB input, 250,000 vertices and 750,000 indices,
128 MiB decoded mesh. Reject unsupported compression, skins, animation, morphs,
nontriangles and malformed/oversized buffers visibly. Honor static node transforms,
units and handedness; do not load imported materials/textures. Missing normals
get area-weighted vertex normals; missing UVs yield a clear unsupported error.
Generate orthonormal tangents from triangle UV derivatives, preserving UV seams
and handedness; degenerate UV triangles use a deterministic fallback with warning.
No MikkTSpace parity promise or extra tangent dependency for this inspection view.

POM is a fragment-space UV ray march on plane/valid tangent meshes, 8..32 steps
depending on view angle plus 4 binary-refinement steps. Height is h-midlevel;
guard grazing rays and zero-scale paths. All material maps use displaced UVs.
Repeat/clamp matches material boundary. No tessellation, silhouette displacement
or self-shadowing claim. Constant height at midlevel produces exact identity;
grazing-angle and normal-convention fixtures verify stability.

Tile inspection supports 1x, 2x2, 3x3 plus seeded random quarter-turn tiles in
2D and on the plane. Transform normal xy with tile rotation. Random rotation
can introduce seams on anisotropic material; label this and keep normal seam
heatmap independent of that diagnostic. 1x is the canonical export material.
A/B stores two compact graph/parameter snapshots with source leases, not undo
textures; split slider uses same camera, lighting and exposure. Render each at
<=1024 proxy, cache at most one image per side; full-res compare is sequential
on demand under budget. Before/after captures source recipe as A. Increment a
comparison identity for edits/view changes; reject stale halves independently
and publish only matching pairs. The live material may be side B without mutating A.

T-063 isolates targets/leases; T-065 direct PBR+primitives; T-067 IBL; T-068 POM;
T-069 tile/A-B and UI integration. GPU-free formula tests run on all CI OSes;
Windows captures at 100/150% scaling show all primitives, custom meshes, dielectric,
metal, roughness sweep, flat/signed normals and HDR range. Shared-device validation
requires zero errors and <=1 GiB accounting. Preview target: frame p95 <=20 ms
on target RTX at 1920x1080 after warmup, plus unchanged Faz 0 interaction gate.

## Alternatives considered

- Second renderer/device or CPU ColorImage readback: breaks resource sharing or
  introduces a frame-sized GPU/CPU round trip.
- Bevy/full game engine: unnecessary layering, build and installer size costs.
- Engine-exact tonemapping, glTF material runtime or displacement tessellation:
  beyond inspection MVP; import geometry and inspect Texelith's material instead.

## Consequences

GPU work is centralized and observable. Static mesh import has explicit limits;
IBL may initially show the procedural sky while cooking. POM is a UV approximation.

## Open questions for the human

No blocking question. Confirm the neutral shader's appearance during M2 review;
engine-match shading remains a later feature and requires engine-specific evidence.
