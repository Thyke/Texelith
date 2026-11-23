# ADR 0005: Deterministic seamless material generation

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

MVP requires offset-and-blend, splat-based and histogram-preserving tiling plus a
seam heatmap. Periodic sampling alone does not remove an input photograph's seam.
Existing kernels clamp and only operate on height; ADR 0004 adds explicit Repeat
and linear-color processing without changing legacy documents.

## Decision

Apply seamless processing to source albedo or authored height before deriving
normal/AO/curvature. A shared transform/seed feeds all correlated source layers;
never independently randomize different material channels. New materials default
to offset blend; disable is lossless. Users may compare all three modes.
Treat the tile as a continuous periodic field sampled at pixel centers: the first
and last pixel need not be equal. Require wrap-crossing derivatives comparable to
interior derivatives, rather than duplicating edge pixels and creating a plateau.

```rust
pub enum SeamlessMethod {
    OffsetBlend { overlap: f32 },
    Splat { seed: u32, grid: u32, overlap: f32 },
    Histogram { base: SeamlessBase, strength: f32 },
}
pub struct SeamlessPlan { pub method: SeamlessMethod, pub boundary: Boundary }
// Typed core parameters; GPU executor owns passes and scratch, no egui types.
```

Offset blend is a separable overlap-add construction. Translate the original by
half a tile and blend four periodically translated copies with normalized smooth
window weights. Each window vanishes (including first derivative) where its copy
crosses the original image boundary; complementary copies supply coverage there.
Use pixel-center UVs, continuous half offsets (not integer floor offsets for NPOT),
manual bilinear loads and overlap in [0.1,0.5] of tile size (default .25).
Sum weights and divide with epsilon 1e-8; constant color/height is invariant.
Document loss of composition/ghosting on structured photos; no universal seam claim.

Splats place patches on a periodic 4x4 lattice by default (grid 2..8). Derive
translations and 0/90/180/270-degree rotations from a specified u32 hash of seed
and wrapped cell coordinates; no runtime RNG state. Smooth compact support and
at least one-cell overlap guarantee nonzero coverage. Evaluate bounded neighbors
per output pixel (no float atomic accumulation), normalize weights, then sample.
Replicate patch centers across the torus; sampling the field at u and u+1 must
match. Record patch assignments for all correlated maps. Tangent normals, if
ever supplied as correlated inputs, rotate xy with UV rotation and renormalize;
MVP default derives normals after splatting the scalar/color source.

Histogram-preserving mode operates on linear luminance, preserving chroma through
a bounded luminance ratio. Build a 1024-bin source CDF with deterministic integer
histogram/reduction; map luminance to clipped Gaussian quantiles via fixed LUT.
Blend latent values with the selected offset/splat weights divided by sqrt(sum
of squared normalized weights); inverse-map through source CDF. Strength lerps
base blend and reconstructed luminance. Flat/low-variance (<1e-6) images use the
base blend. Black pixels avoid division by zero; RGB remains finite and clamped
to source SDR range. This preserves luminance distribution approximately, not
each RGB marginal or perceptual texture structure. Fixed bin/LUT definitions,
epsilon and seed enter algorithm revision/cache keys. Histogram passes count
toward full-node timing; no hidden CPU round trip for per-pixel conversion.

Seam inspection computes wrap gradient magnitude at opposite boundaries and a
four-pixel band, contrasted with equal-width interior gradients. Show heatmap,
absolute RMS and boundary/interior ratio (zero denominator handled explicitly).
Heatmap is diagnostic and never silently repairs data. Provide both 2x2 preview
and the original/transformed comparison, including a high-contrast structured
fixture where blending necessarily alters the pattern.

Quality gates: constant preservation <=2e-5, translated-period sampling <=2e-5,
same-seed result identical on the same backend, distinct seeds visibly distinct,
no black/NaN holes for NPOT/singleton grids. On fixed synthetic discontinuous
noise fixtures, boundary RMS must decrease at least 50% without suppressing
interior RMS by more than 50%. Histogram mode luminance CDF max distance <=.05
on a fixed seeded 1024-square noise fixture; record color/clipping error separately.
Do not impose these empirical thresholds on arbitrary user photographs.

Compute batches use ADR 0004 resource reservations and generation cancellation;
cache source histograms/pyramids by content, not by slider request. T-057, T-058,
T-059 and T-060 each add one mode/diagnostic to the same executor serially, with
CPU oracle, 4K node timing and the unchanged Faz 0 regression gate.

## Alternatives considered

- A hard half-offset alone: relocates seams without repairing them.
- Poisson/image quilting or learned synthesis: larger optimization and licensing
  surface; outside these three requested baseline algorithms.
- Claim exact histogram preservation after blending: inaccurate for correlated
  samples, quantized CDFs and color gamut clipping; report measured CDF error.
- Postprocess each generated map independently: breaks spatial correspondence.

## Consequences

Seeded splats and transform metadata make projects/batches reproducible. Seamless
quality is content dependent and visibly reviewable. Additional histogram scratch
fits the shared budget or reports failure; no unrestricted allocations.

## Open questions for the human

No blocking question. Default offset blend preserves more structure than random
splats; the latter remains an explicit user choice. Confirm quality defaults after
the M1 visual demo rather than treating synthetic seam scores as artistic approval.
