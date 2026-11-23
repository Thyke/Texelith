# ADR 0009: Simple workspace through the command registry

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

The current UI has generated controls, command palette, import/PNG and testable
command_adapter/routing modules. Core Session already coalesces gestures and
emits typed effects. MVP should expose the full material without replacing that
path or creating a second model for the future graph editor.

## Decision

Left map stack, central 2D/3D viewport, right inspector and always-visible bottom
export bar use egui panels, not a docking dependency. Map order and keys 1..7:
Albedo, Height, Normal, AO, Roughness, Metallic, Edge. Curvature signed/convex/cavity
and displacement are inspector subviews of Edge/Height. Cards show thumbnail,
selection, visibility, lock and Procedural/Manual badges. AI badge/control is absent
until Faz 2. Visibility disables contribution in preview via neutral defaults,
not deletion from export. Lock rejects parameter/stroke edits, not recomputation
when an ancestor changes; this avoids a misleading freeze that would retain stale
textures. Tooltip states semantics; export inclusion is a separate setting.

Inspector shows 4-6 frequent controls then Advanced: normal filter/convention,
height bands/remap, AO radius/samples, roughness curve, metallic picker/mask,
seamless mode/seed, midlevel. Fields derive schema/ranges from descriptors; no
duplicated numeric validation in widgets. A color picker samples the original
linear source via bounded asynchronous one-pixel readback, never blocks update.
Painting uses normalized source UV and the stroke transaction model in ADR 0007.
Cursor overlay is UI-only; actual mask commit goes through Session.

Export bar selects preset, output size, format and Export; opens a packing panel
for per-channel source/component/inversion, naming preview and save-preset. Until
M3, M1/M2 keep legacy single-map PNG working and explain pending set export rather
than shipping a clickable no-op. M1 demo needs selection/parameters in existing
layout; M2 completes all panels. M3 connects real export/project/batch effects.

| Stable command ID | Payload/effect | Undo semantics |
| --- | --- | --- |
| existing node.set_parameter | Extend typed values: enum/bool/vector/curve | Coalesced parameter diff |
| material.select_map | Map/output ref | View only |
| material.set_visible / set_locked | Map + bool | Small state diff; visibility view-only generation |
| mask.begin_stroke / append / commit / cancel | Stroke ID + bounded UV samples | One stroke insertion diff |
| view.set_mode / set_mesh / set_light | Validated preview intent, asset token | View only |
| view.set_tiles / set_pom / set_camera | Tile seed/count, scale, camera | View only |
| view.capture_a / capture_b / set_split | Snapshot slot / fraction | View only |
| export.configure / export.save_preset | Validated settings or destination token | Settings diff; save is effect |
| export.material / export.cancel | Captured export ID / job ID | Effects, no graph edit |
| project.open / save / save_as | Opaque native-path token | Effect; successful open resets history |
| batch.add / start / cancel / retry_failed | Locator list or job ID | Effects, separate batch state |

Extend enum CommandId and static descriptors together; maintain one-to-one tests
and stable string routing. All user actions including view settings pass registry
validation. View effects do not bump graph generation; request_id increments for
camera/quality/selection. Document edit revision is separate from evaluation
generation so settings/locks persist and dirty the project without invalidating
map content. Save/Open errors and disabled commands leave state intact. Store
labels as localization keys (label fallback preserved for diagnostic consumers).

1..7 select maps only outside text editors, numeric edit focus and palette.
Ctrl+K opens palette; Ctrl/Cmd+Z and Shift+Ctrl/Cmd+Z undo/redo, Ctrl/Cmd+O open,
Ctrl/Cmd+S save project, Ctrl/Cmd+Shift+S Save As. Shortcut/menu/palette/buttons
produce identical CommandRequest. Focus loss commits gestures; Escape cancels
active stroke/slider first, then closes transient UI. Do not intercept OS text
editing shortcuts. Keyboard traversal, visible focus and accessible labels remain.

Drag/drop is handled at the top-level surface, including over panels. One image
dispatches existing new-material import; multiple images offer batch addition.
.texelith.json opens a project, HDR/mesh drops target environment/mesh with explicit
type feedback. If the document is dirty, queue an intent and offer Save/Discard/
Cancel before replacing it. Never load all dropped bytes on the frame thread.
Import/open/save events carry tokens and errors; keep old successful frame on
failure. Progress chips expose job state and cancellation, not implementation logs.

Dark neutral theme, one accent for selection; no shader jargon or GPU job IDs in
primary flows. Advanced diagnostics may show adapter, budget, normal convention,
quality and transfer assumptions. At 1280x720 and 100/125/150/200% scaling, export
bar remains reachable, panels scroll internally and center never has negative
size. Minimum supported logical window 960x600; collapsible side panels below
1280 wide, remembering widths. Thumbnails are <=128-square cached previews under
the GPU budget, not seven full-res readbacks.

Use Fluent catalogs en-US/tr-TR for all new user-facing text and migrate existing
labels while touching controls. fluent-bundle and unic-langid are narrowly scoped
UI dependencies with MIT/Apache-2.0 candidate licenses; T-040 resolves/audits them.
No additional fonts are needed; retain existing notices. Missing keys fall back
to English and fail catalog coverage tests, not to empty labels. Core stores
stable identifiers, never translated string values in hashes/projects.

T-070 adds registry/view contracts, T-071 panels, T-072 inspectors/painting and
T-073 input/localization. Their separate acceptance checks prevent a static
mockup being counted as functioning UI. T-074 records every card, mode, primitive,
paint/undo state, import error and split state at 100/150%; includes manual drag,
focus loss, drop on every panel and keyboard routing. Later T-097 checks packing,
exports, missing-asset relink and batch cancellation. Screenshots alone cannot
verify shortcuts, retirement or uninterrupted responsiveness.

## Alternatives considered

- New callbacks mutating graphs: bypass history, token validation and future CLI.
- Dockable graph-editor shell now: expands MVP and distracts from fixed Simple UX.
- Lock pins old GPU outputs: violates resource budget and hides stale data.
- English literals now, localization later: expensive migration and misses GUI spec.

## Consequences

Core gains typed view/effect DTOs without egui imports. UI tasks own their crate
serially when wiring shared app.rs; visual verification remains a distinct pass.
Not every state change recomputes maps or enters parameter history.

## Open questions for the human

Confirm proposed lock/visibility semantics at M2 demo. The plan uses edit-lock
and preview visibility; freezing maps or disabling export requires separate intent.
