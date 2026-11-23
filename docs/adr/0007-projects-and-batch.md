# ADR 0007: Versioned projects and bounded batch processing

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

The current GraphDto is a diagnostic envelope, not a project format. Session owns
generation and diff history; IO is bounded and native paths live in the frontend.
MVP needs source+graph persistence and applying one recipe to many inputs without
blocking input, retaining full image sets in memory, or duplicating graph logic.

## Decision

Use UTF-8, two-space indented JSON, extension .texelith.json. Core defines serde
DTOs, version checks and validation; IO parses/writes through serde_json. Core
still requires serde_json only for tests. No new parser or database dependency.
Sort nodes by stable ID and assets/presets by stable key; fixed field order, LF,
trailing newline. Reject duplicate object keys, nonfinite numbers and unknown
semantic fields rather than silently discarding future settings. Limits: 8 MiB
project text, 256 graph nodes, 64 assets, JSON depth 64 and 100,000 total brush
points. These are load-validation limits, not an arbitrary undo history cap.

```rust
pub struct ProjectFile {
    pub format: String,          // exactly "texelith.project"
    pub schema_version: u32,     // 1 at first public release
    pub graph: GraphDto,         // schema 2, independent version axis
    pub assets: Vec<AssetRef>,
    pub export: ExportSettings,
    pub view: SavedView,         // camera etc; never changes graph cache keys
}
pub struct AssetRef { pub id: AssetId, pub relative_path: String,
                     pub decoded_hash: [u8; 32], pub interpretation: AssetIntent }
pub fn validate_project(dto: ProjectFile) -> Result<ValidatedProject, CoreError>;
pub fn instantiate_recipe(recipe: &Graph, source: SourceDescriptor)
    -> Result<Graph, CoreError>;
```

Persist typed parameters, normal convention, boundary, seeds, brush vector
strokes, named outputs, export preset ID+revision+overrides and compact view state.
Never persist generations, GPU resources, pixel arrays, undo stack, pending jobs
or native absolute paths. Sidecar assets remain original, untouched files.
Paths use portable slash-separated project-relative references; Save As rebases
references. External sources may use explicit ../ references: resolve and display
them before import, without restricting a legitimate linked-asset workflow.
Reject URLs/device paths, embedded NUL and implicit network retrieval. Canonicalize
at IO boundary, bound bytes before decode, report symlink resolution and missing
assets. Source hash mismatch requires explicit relink/use-changed choice; do not
reuse a cache for a changed file based on path alone. Save never embeds that file.

Open reads and validates into a temporary document, decodes all required sources
within shared CPU budgets, checks hashes, then uses a tokenized Session completion
to atomically replace document+sources and reset history. Errors leave current
document untouched. Project-open token is distinct from render generation; newer
opens invalidate older results. Generation continues monotonically across opens.
Save snapshots data immediately, writes on IO worker to sibling temp, flushes,
then replaces with backup/rollback as required on Windows. Mark clean only when
the saved document revision still equals live revision; view persistence has its
own dirty flag. Never claim std::fs rename-overwrite is portable atomic replacement.

Project schema migration is a pure version-to-version conversion before validation.
Version 1 is initial; unknown future versions fail with supported range. Do not
invent a historical project v0. A separate explicit diagnostic-import path adapts
Faz 0 GraphDto v1 to current graph v2, wrapping source locators only when supplied.
Round-trip golden fixtures test stable text, all enum variants and re-save identity.
Migration tests keep the old central/clamp numerical recipe intact.

Brush editing is parameter history: stroke UV points, radius in normalized source
units, hardness, opacity, mode and deterministic point spacing. Undo stores stroke
insert/remove or edits, not textures; recompute the mask in GPU checkpoints with
content keys. A stroke is one gesture, Escape discards it, focus loss commits it.
No unbounded pointer-event queue: resample at <=radius/4 spacing and cap per-stroke
points with a visible finish/restart request. Persistence uses the same stroke data.

Batch uses a captured immutable recipe and export settings. Deterministically
enumerate a folder (nonrecursive default; explicit recursion bounded to 10,000
files, no symlink traversal) or the multi-file list; accept supported image types,
deduplicate canonical paths, sort portable relative names and preflight all output
collisions before compute. A failed file yields its own error; stop-on-error is
default, continue-on-error is explicit. No automatic retries with different quality.

```rust
pub enum BatchState { Queued, Loading, Evaluating, Encoding, Committed,
                      Failed, Cancelled }
pub struct BatchItem { pub id: u64, pub source: OpaqueLocator, /* state/error */ }
pub enum BatchAction { Start, Cancel, RetryFailed }
// Pure transition/retry policy in core; native scan/load/encode in UI+IO workers.
```

One active item, one decoded next-item slot maximum, one shared GPU execution
stream; CPU reservation includes interactive and batch decoded input, readback
and codec scratch (default 512 MiB total, individual decoder cap remains 256 MiB).
Release the next slot if interactive source loading needs the reservation. GPU
worker checks interactive priority between node batches, avoiding export starvation
by progressing batch when interactive mailbox is empty; no completion deadline
while a continuous drag occupies the GPU. IDs include job, item, generation and
request: equal numeric generations in two documents cannot cross-publish results.
Source uploads must be content-addressed for both documents, not a global "current
pixels" pointer. A/B and batch source leases participate in the same GPU budget.

Cancellation marks unscheduled items cancelled, invalidates pending work, waits
off the UI thread for submitted buffers to retire, and cleans only its staging
directory. Completed sets remain complete. RetryFailed creates a new job with
the same captured recipe; a changed recipe is a new batch. Keep a small durable
JSON job manifest recording recipe digest, source digests, destinations, errors
and state. Restart offers retry of incomplete items; no automatic background
resume or recomputation of already committed files. Project files do not embed
the batch queue. Export transactions follow ADR 0006.

## Alternatives considered

- RON/YAML/TOML: readable but add a parser; existing serde JSON and explicit schema
  deliver version control and machine tooling now.
- Serialize Session wholesale: leaks generations/history and binds files to UI state.
- Parallel full graphs per image: overcommits VRAM and interferes with interaction.
- Mutating the open project source for each item: corrupts undo and visible state.

## Consequences

Linked projects need their source assets to travel with them. Content hashes make
changes explicit. Batch throughput is bounded and cancellable; it is not the
deferred headless CLI. IO compatibility/error-injection tests precede UI wiring.

## Open questions for the human

No blocking question. Embedded/archive projects and recursive scan as default are
deferred; the initial public file format intentionally preserves readable links.
