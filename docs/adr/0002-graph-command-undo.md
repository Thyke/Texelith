# ADR 0002: Core graph, commands, and parameter-diff history

Status: accepted (user-approved 2026-11-10).

## Context

Faz 0 needs three nodes and undo, while later CLI/MCP must perform exactly the
same actions as the UI. The spec keeps graph editing internal and places saved
projects in Faz 1. Sliders must not produce thousands of history entries or retain
GPU textures. Deterministic snapshots also form a boundary for future AI and C ABI.

## Decision

### Model and contracts

Use an internal typed, single-input DAG with stable `NodeId(u64)` values and
`NodeKind::{Source, Levels, Blur, NormalFromHeight}`. Source is an input reference,
not a fourth user compute filter. Each other node has one typed input edge.
Faz 0 constructs `Source -> Levels -> Blur -> NormalFromHeight`; the viewport
can select height or normal, but there is no graph editor or arbitrary rewiring UI.
Keep the DAG representation rather than baking this chain into executor indexing.

```rust
pub struct Extent { pub width: u32, pub height: u32 }
pub enum InputInterpretation { SrgbPhoto, LinearHeight }
pub struct SourceDescriptor {
    pub content_hash: [u8; 32], pub extent: Extent,
    pub interpretation: InputInterpretation,
}
pub enum MapKind { Height, Normal }
pub enum NodeParams {
    Source(SourceDescriptor),
    Levels { black: f32, white: f32, gamma: f32 },
    Blur { sigma_source_px: f32 },
    NormalFromHeight { strength: f32 },
}
pub struct Node { pub id: NodeId, pub input: Option<NodeId>, pub params: NodeParams }
pub struct Graph { /* sorted nodes, selected output, validated edges */ }
pub struct EvaluationSnapshot { pub generation: u64, pub graph: Graph }
pub struct CacheKey(pub [u8; 32]);
pub fn validate(graph: &Graph) -> Result<(), CoreError>;
pub fn node_cache_key(node: &Node, inputs: &[CacheKey],
    extent: Extent, source_extent: Extent) -> Result<CacheKey, CoreError>;
```

All are owned core values. Private graph fields and constructors preserve
invariants; deserialization produces untrusted DTOs and calls validation before
creating live state. The signatures are contracts to implement, not compiled code.
Validation rejects duplicate/missing IDs, cycles, wrong input types, extra source
inputs, zero dimensions, over-4K inputs, nonfinite values and invalid parameters.
Levels: `0 <= black < white <= 1`, white-black >= 1e-6, gamma in [0.1,10].
Blur: sigma in [0,5] source pixels, kernel radius ceil(3*sigma), max 15.
Normal strength in [0,10]. Defaults: levels (0,1,1), blur 0, normal 1.
Height-to-normal input is scalar height; all intermediate levels/blur remain
height. Source photo luminance is specified by ADR 0003, not guessed by the UI.

### Registry and execution boundaries

Core owns the static descriptors, payload validation and authoritative graph
mutation. UI owns presentation and handles effects; it never edits graph fields.
Descriptors carry stable string ID, label, typed payload schema/ranges and
availability. CLI/MCP can later enumerate these without importing egui.
Hotkeys and file-picker widgets are frontend bindings, not core command semantics.

| Stable ID | Payload / behavior | Undo policy |
| --- | --- | --- |
| `document.new_from_image` | Request token + opaque source locator; emit `LoadImage` | Successful load starts a new history |
| `node.set_parameter` | NodeId, typed parameter ID/value; optional gesture ID | One diff or coalesced gesture |
| `edit.begin_gesture` | Unique gesture ID and parameter address | Open transaction |
| `edit.commit_gesture` | Matching gesture ID | Commit one net diff |
| `edit.cancel_gesture` | Matching gesture ID | Restore before value, discard transaction |
| `edit.undo` / `edit.redo` | Empty | Move diff between stacks |
| `view.select_output` | Existing Height/Normal node | View state only |
| `image.save_png` | Opaque destination + selected map | Emit export effect; no graph mutation |

An opaque locator is an uninterpreted UTF-8 frontend token, not an OS path type in
core; native paths live in the UI's request table. `LoadImage`/`SavePng` are typed
requests. Completion calls a separate `complete_import(token, SourceDescriptor)`
entrypoint, not an arbitrary public "set graph" command. The app installs matching
decoded pixels and the new snapshot together before submitting GPU work.

```rust
pub fn command_descriptors() -> &'static [CommandDescriptor];
impl Session {
    pub fn dispatch(&mut self, request: CommandRequest)
        -> Result<CommandOutcome, CommandError>;
    pub fn complete_import(&mut self, token: ImportToken, source: SourceDescriptor)
        -> Result<CommandOutcome, CommandError>;
    pub fn snapshot(&self) -> EvaluationSnapshot;
}
// Outcome: typed Effects, optional new snapshot, history/availability changes.
// Effect: LoadImage, SavePng; scheduling follows snapshot changes in the app.
```

Descriptors and payload enums have a one-to-one mapping, verified in tests.
Unknown IDs/payloads, disabled commands and stale gesture/import tokens return
structured errors without partially mutating state. JSON routing is future-facing
serialization of these DTOs, not a JSON parser inside every slider event.
One pending import token wins; older decode results are discarded. Failed imports
leave the old document/history/output intact. Import tokens are separate from
graph generations, so changing a slider cannot invalidate the latest import.

### Undo, generations, and invalidation

Each successful parameter change, undo, redo, cancelled gesture restore, or
successful input replacement increments a session-wide monotonic `u64` generation.
No-op edits and view-only changes do not. On overflow return an explicit error;
never wrap into a previously issued generation. Source replacement does not reset
the counter. The app separately numbers render requests for quality/view changes.

A parameter diff contains `(NodeId, ParameterId, before, after)`; a transaction can
hold a list but Faz 0 gestures change one parameter. No textures, pixels, paths,
or entire graphs in history. No arbitrary count cap: memory grows with committed
edits; display diagnostic entry/byte counts. A gesture stores the first before and
latest after; each intermediate value still emits a new evaluation snapshot.
A net-zero gesture creates no entry and preserves the redo branch. Redo is cleared
only when a new nonempty diff commits. Cancel restores the original and preserves
redo. Undo/redo during an active gesture first commits that gesture, then executes
the requested operation. Nested gestures or mismatched end tokens are errors.
Focus loss commits; Escape cancels. Numeric entry/keyboard increments outside a
gesture commit individual diffs. Errors roll back the entire request.

Input replacement is a document boundary and clears undo/redo only after decode
success. It is not undoable in Faz 0; retaining every original photo would violate
the cheap-history intent. This limitation is visible in the command label/help.
Selecting output changes the requested render node without dirtying the graph.
Parameter edits dirty that node and its descendants; unchanged ancestors retain
cache keys. Scheduling and cancellation are GPU/app responsibilities, not core
threads. Undo can reuse a previous content key despite having a newer generation.

### Serialization and future project shape

Derive serde for graph/parameter/command DTOs. Use a versioned test envelope
`{ "schema_version": 1, "graph": ..., "selected_output": ... }` for round trips
and diagnostic fixtures. This is **not** a supported user project file yet.
Never serialize runtime generations, GPU caches, undo stacks, absolute native
paths, or image pixel arrays into it. Unknown versions fail explicitly.
A Faz 1 project envelope can add relative asset references, content hashes and
migrations around the same validated graph. Faz 0 has no Save/Open Project action.

### Deterministic content identity

Hash with blake3 over a versioned domain prefix, shader/algorithm revision,
node-kind tag, map/format tag, canonical parameters, ordered input keys,
evaluation extent, and original source extent. Source keys additionally include
the canonical decoded RGBA8 bytes, dimensions, import interpretation and color
policy revision. Different encodings of identical decoded pixels can share a key.
Do not hash source filenames, generation, transient NodeId, JSON text, locale, or
Rust's randomized map iteration order. Encode integers little-endian, enum tags
explicitly, finite f32 as normalized IEEE bits (`-0` becomes `+0`), and variable
fields with length prefixes. Proxy/full dimensions distinguish their keys.
Golden byte/hash fixtures lock the schema; an algorithm change bumps its revision.
Results are deterministic within the same backend/algorithm configuration; float
agreement across vendors is tolerance-based, not a bitwise guarantee.

## Alternatives considered

- GUI callbacks mutating fields: cannot guarantee equivalent CLI/MCP behavior,
  gesture semantics, validation, or undo.
- Texture or entire-session snapshots for undo: costs scale with 4K image storage
  and retain GPU resources; diffs match the required parameter-level history.
- A generic plugin/type-erased graph and dynamic registry: excessive for three
  known nodes. Versioned typed commands provide a later extension seam.
- Shipping project persistence now: explicitly a Faz 1 feature; serde contracts
  and fixtures provide sufficient architectural preparation.

## Consequences

New parameters require coordinated typed schema, descriptor, validation, hash and
shader changes. This explicit work avoids silent frontend drift. Document loads
cannot be undone; gestures remain cheap and interactive. Source data ownership
stays outside core, keeping native I/O and wasm compilation independent.

## Open questions for the human

- Approve successful image import as a new-document boundary that clears history?
- Approve photo import as the default, with explicit linear-height import for
  authored heightmaps, instead of inferring intent from grayscale filenames?
- Normal convention defaults to the OpenGL-style sign defined in ADR 0003;
  engine presets and a convention toggle are deferred. Confirm this phase default.
