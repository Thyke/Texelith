# ADR 0011: Faz 1 parallel delivery plan

Status: accepted (orchestrator, 2026-11-11).

## Context

T-040 through T-046, T-055 and T-120 are complete; T-047 is in progress.
The existing four crates and their dependency direction remain authoritative.
ADR 0004 imposes row barriers in addition to task edges, and broad source-tree
ownership makes otherwise independent nodes conflict. Removing only task edges
would neither permit dispatch nor make integration safe.

Read against the current worker, planner, suite selector and crate entry points.
They use explicit Rust match dispatch, not a dynamic plugin registry. Registration
markers below are a convention to add during implementation; they do not yet exist.
No production code, dependency or new task is introduced by this planning change.

## Decision

### Authority and readiness

On orchestrator acceptance, this ADR supersedes only ADR 0004's delivery-wave
barriers/blanket shared-GPU serialization, ADR 0005's last-paragraph requirement
to add seamless modes serially, and ADR 0009's blanket whole-UI-crate ownership.
The numerical contracts, product scope, gate thresholds, layering and all other
ADRs remain unchanged. Old task Context references to ADR 0004 refer to those
contracts, not its superseded schedule. T-047's dispatch/ownership is untouched.

Dispatch a task when its explicit producers have merged and its file exclusions
are free. A completed milestone demo is not a producer of a Rust API. Do not add
an edge merely to order competing writers; acquire the file exclusion instead.
Keep at most three concurrent writers and at most one T4 writer. Role-chain
passes within a task stay sequential. Completed T-055 changes only dependency/
wave metadata; its original broad Owns is historical, not a fresh write grant.

Every original Goal, Acceptance and tier is preserved. No splits or new features.
All milestone gates explicitly join their milestone members; M2 also joins M1
evidence, and final acceptance joins M1/M2 plus every M3 member. A failing gate
still prevents acceptance/release even if independent later work has started.

### Shared-file convention

1. Each task owns a named leaf directory (node/solver/encoder/panel), its
   `tests/mvp_tNNN.rs` and `tests/mvp_tNNN/`, its numerical suite under
   `examples/mvp_support/tNNN/`, and its evidence paths. Task Owns lists exact
   exceptions. Shared helpers from completed tasks are read-only unless listed.
2. An `append-only: <file> (<marker>)` grant means one small task-labelled
   registration block at that Rust item/list/match location. The first writer
   adds the marker comment at the existing insertion point, without extracting
   or rewriting the shared implementation. Use `// FAZ1_...: T-NNN begin/end`
   around the task's entries; one marker pair per insertion site is sufficient.
   Initial absent/duplicate marker comments are mechanically reconciled at merge.
3. Append module declarations/re-exports, distinct dispatch cases, scratch
   estimators, task-local runtime state fields/initializers, and suite entries.
   Delegate real logic to the owned leaf. Never replace a wildcard/default,
   broaden a match to swallow another mode, reorder registrations, rename a
   sibling symbol, or reformat the whole shared file. Register each seamless
   method separately. Existing-case replacements (normal, multi-output planner,
   preview/readback lifecycle) use the structural exclusions below.
4. Shader `include_str!` calls and node-specific bind layouts live in their
   leaf module. Prepared passes take the existing device/queue/resource inputs;
   runtime adapters can be child modules of the existing runtime to access its
   private allocation/submission helpers. No new general plugin framework,
   runtime reflection, build-time registry generator or crate dependency.
5. Register the suite runner AND selector/aggregate membership together. Unknown
   and empty suites still fail; preserve baseline assertions. No success stubs
   for unimplemented siblings. An early aggregate runs merged registrations;
   T-062/T-074/T-099 require the entire milestone suite inventory.
6. Parallel worktrees may still collide textually at additive insertion sites.
   The orchestrator integrates one task at a time, preserving the union of
   task-labelled entries. Never resolve by taking one whole file. Rebuild/test
   the affected crate, then run the affected numerical/behavioral suites on the
   integrated revision. A semantic overlap is not a mechanical union: stop that
   parallel dispatch and reconcile ownership before continuing. Agents do no git
   history/branch operations.
7. Changes needing an unlisted file or a non-additive change under an append-only
   grant must report the concrete blocked ownership/API issue. Do not move the
   work to a gate or leave an unregistered implementation to evade acceptance.

| Shared file / location | Marker(s) / rule |
| --- | --- |
| GPU `src/lib.rs` | `FAZ1_GPU_MODULES` |
| GPU `src/worker_runtime.rs` | `FAZ1_GPU_RUNTIME_MODULES`, `FAZ1_GPU_RUNTIME_FIELDS`, `FAZ1_GPU_RUNTIME_INIT`; task-specific state only |
| GPU `src/worker_material.rs` | `FAZ1_MATERIAL_DISPATCH`; distinct node/method cases calling leaf adapters |
| GPU `src/material.rs` | `FAZ1_MATERIAL_SUPPORT`, `FAZ1_MATERIAL_SCRATCH`; distinct node/method cases |
| GPU `examples/verify_mvp.rs` | `FAZ1_SUITE_MODULES`, `FAZ1_SUITE_RUNNERS` |
| GPU `examples/mvp_support/contract.rs` | `FAZ1_SUITE_SELECTION`; named suite and applicable maps/viewport/all entries |
| Core / IO `src/lib.rs` | `FAZ1_CORE_MODULES` / `FAZ1_IO_MODULES` |
| Core `command.rs`, `session.rs` | Explicit full-file CORE-SESSION owner; enums/descriptor and dispatch parity stay atomic |
| UI app/routing/IO worker/catalogs | Explicit UI-SHELL owner; helpers/panels remain in task leaf modules |
| PBR common shader/bind layouts | T-063 creates the live clear/ramp entry in `src/viewport_shared/` and `shaders/viewport_shared/`; T-065/T-067/T-068/T-069 extend it under PBR-SHARED |
| GPU export-format/readback adapter | T-079 creates live `src/export_shared/` routing to packing/readback; T-081 also owns this directory to connect BC5, with T-079 -> T-081 ordering |

### Exclusions: pairs that must not run in parallel

These are symmetric file leases, independent of the dependency graph.

| Exclusion | Incompatible tasks |
| --- | --- |
| In-flight GPU owner | T-047 with **any** GPU writer; wait for its merge/release, without editing its dispatch |
| GPU-STRUCTURAL | Each of T-048, T-052, T-063, T-079 with **any** other GPU writer, including each other |
| PBR-SHARED | Every pair in T-065, T-067, T-068, T-069; IBL and POM share the shader even though neither consumes the other's output |
| CORE-SESSION | Every pair in T-070, T-075, T-084, T-086 |
| UI-SHELL | Every pair in T-061, T-069, T-071, T-072, T-073, T-082, T-083, T-087, T-088, T-089, T-096 |
| Native dependency policy | T-089 and T-090 share `scripts/check-dependencies.ps1` (also ordered by a real target-policy dependency) |

GPU writers are T-047–T-054, T-056–T-060, T-063, T-065, T-067–T-069,
T-078, T-079 and T-081. Other GPU-LEAF pairs may overlap with only additive
registrations in common. T-076 owns the preset table after T-075 registers it;
core session work does not edit that table. IO decoders/encoders/transactions/
project code use separate modules and local error types; IO root implementations
stay read-only. OS packaging scripts and original assets are separate per OS.

Producer APIs must expose the owned integration point, not bury it in a private
leaf that consumers cannot edit. T-063 routes preview through viewport_shared;
T-079 routes export formats through export_shared; T-071 exposes panel slots to
shared app.rs composition. These are the same already-required integration paths,
not a separate plugin or framework deliverable. The baseline paths work before
lighting/BC5 lands; unsupported formats fail rather than returning fake success.

Hardware acceptance has a separate exclusive measurement slot: stop concurrent
GPU workloads and builds, keep the measured app foreground, and run unchanged
T-120-aware gates serially. Evidence tasks write only their own directories and
read a pinned integrated revision. Their presence in a round is not permission
to benchmark while sibling builds run.

### Wave graph by milestone (readiness layers, not barriers)

Already merged roots are T-040–T-046, T-055 and T-120. The table of exact edges
below is authoritative; these groups show branching and the unavoidable joins.

| Milestone | Parallel branches and joins |
| --- | --- |
| M1 independent frontier | T-048, T-049, T-051, T-052, T-053, T-054, T-056, T-057, T-060 use existing contracts; T-047 retains its current work |
| M1 branch continuations | T-049 -> T-050; T-057 -> T-058 -> T-059; imported analytic heights/normals/masks avoid unrelated producer waits |
| M1 integration | All map branch tips -> T-061; all T-040–T-061 plus T-120 -> T-062 |
| M2 contracts and IO | T-070, T-064, T-066 can start before M1; T-070 + T-046 -> T-063 -> T-065 |
| M2 renderer/UI branches | T-065 -> {T-067, T-068} with PBR exclusion; {T-061,T-070} -> T-071 -> T-072 -> T-073 |
| M2 integration | Geometry + HDR + IBL + POM + completed UI -> T-069; all T-063–T-073 and T-062 -> T-074 |
| M3 contract/data/IO branches | T-070 -> T-075 -> {T-076,T-077,T-078,T-086}; T-080 and T-089 start from merged roots; T-076 -> T-084; {T-077,T-084} -> T-085 |
| M3 export branches | T-078 -> T-079 -> T-081; preset + transaction + codec + GPU + UI -> T-082; {T-082,T-085} -> T-083; {T-082,T-085,T-086} -> T-087; {T-083,T-087} -> T-088 |
| M3 packaging branch | T-089 -> {T-090,T-091,T-092,T-093}; all four -> T-094; no Windows -> macOS -> Linux chain |
| M3 verification branches | {T-076,T-080,T-081} -> T-095; {T-082,T-095} -> T-096; {T-069,T-088,T-095} -> T-097; {T-069,T-088,T-094,T-096} -> T-098 |
| Final join | T-062, T-074, and **every** T-075–T-098 -> T-099 |

IBL uses synthetic finite HDR pixels; imported HDR and geometry cross IO -> UI ->
core/GPU only in T-069. BC5 hands a small original block/mip fixture to T-095's
independent CPU decoder, never an IO -> GPU crate dependency. T-085's bounded
JSON/recovery primitives serve the already-required preset/job persistence.
T-098 waits for actual final source producers, not the visual verdict T-097.
Early packages are provisional; final three-platform proofs must identify the
same frozen candidate containing viewport, workflow and performance changes.

### Illustrative three-slot dispatch rounds

This is a feasible ordering, not a new row barrier or an elapsed-time estimate.
Start T-047 as already in flight; start other ready nonconflicting tasks without
waiting for its numerics. Refill slots as producers merge, optionally prioritizing
an available gate. Hardware checks use the separate slot above. Sixteen of the
nineteen rounds contain three tasks; the last rounds reflect UI/evidence joins
and externally supplied CI, not artificial milestone blocking. The one-T4 limit
and all file exclusions hold in every row. T-055 is already done and omitted.

| Round | Tasks |
| --- | --- |
| R01 | T-047, T-070, T-080 |
| R02 | T-057, T-049, T-051 |
| R03 | T-058, T-050, T-053 |
| R04 | T-048, T-075, T-089 |
| R05 | T-052, T-076, T-077 |
| R06 | T-054, T-056, T-059 |
| R07 | T-060, T-078, T-084 |
| R08 | T-061, T-079, T-085 |
| R09 | T-071, T-063, T-086 |
| R10 | T-072, T-081, T-065 |
| R11 | T-073, T-064, T-066 |
| R12 | T-082, T-067, T-090 |
| R13 | T-068, T-083, T-091 |
| R14 | T-087, T-092, T-093 |
| R15 | T-095, T-062, T-069 |
| R16 | T-088, T-094, T-074 |
| R17 | T-096, T-097 |
| R18 | T-098 |
| R19 | T-099 |

### Dependency changes

Old deps below omit the suffix `all earlier wave barriers in ADR 0004`, which
was present in every row and is now removed. New lists are exact task headers;
transitive producers need not be repeated except explicit milestone inventories.

| Task | Old deps | New deps | Reason |
| --- | --- | --- | --- |
| T-048 | T-047 | T-044, T-046 | Uses T-044 normal controls and T-046 pyramid/typed height; albedo reconstruction is not required. |
| T-049 | T-048 | T-044, T-046 | Uses typed normals, boundaries and controls from T-044/T-046; generated normals are unnecessary for operator oracles. |
| T-050 | T-049 | T-049 | Consumes T-049 operators, coarse grids and residual reductions. |
| T-051 | T-050 | T-044, T-046 | Uses T-044 AO parameters and T-046 numeric height; no Poisson solver input is required. |
| T-052 | T-051 | T-044, T-046 | Uses typed normal input and curvature parameters from T-044/T-046; can test imported analytic normals without AO or generated normals. |
| T-053 | T-052 | T-044, T-046 | Uses T-044 curve/mask semantics and T-046 linear color; numeric mask fixtures do not require curvature or painted strokes. |
| T-054 | T-053 | T-044, T-046 | Uses T-044 selection/mask semantics and T-046 linear color; numeric mask fixtures do not require roughness. |
| T-055 | T-044 | T-044 | Completed stroke-model task retains T-044; remove obsolete scheduling suffix only. |
| T-056 | T-054, T-055 | T-046, T-055 | Consumes T-055 strokes and T-046 typed scalar/boundary execution; metallic does not produce masks. |
| T-057 | T-056 | T-044, T-046 | Uses T-044 seamless parameters and T-046 source sampling; paint is unrelated. |
| T-058 | T-057 | T-057 | Consumes T-057 correlated-layer transform and overlap-weight interfaces. |
| T-059 | T-058 | T-058 | Consumes offset/splat weights and base blend interfaces via T-058 (T-057 transitively). |
| T-060 | T-059 | T-044, T-046 | Uses T-044 output/boundary types and T-046 typed samples; seam diagnostics also work on untreated input. |
| T-061 | T-060 | T-047, T-048, T-050, T-051, T-052, T-053, T-054, T-056, T-059, T-060 | Joins all map producers and their transitive prerequisites for actual seven-map/normal-import/mask/seam UI acceptance. |
| T-062 | T-061 | T-040, T-041, T-042, T-043, T-044, T-045, T-046, T-047, T-048, T-049, T-050, T-051, T-052, T-053, T-054, T-055, T-056, T-057, T-058, T-059, T-060, T-061, T-120 | Explicit join of all M1 tasks plus the T-120 measurement fix; no implicit wave completeness. |
| T-063 | T-062 | T-046, T-070 | Uses T-046 material execution/leases and T-070 PreviewRequest view/mesh DTOs; clear/ramp tests need no M1 demo. |
| T-064 | T-062 | T-040 | Uses T-040 audited mesh decoder dependencies; produces IO-local DTOs. |
| T-065 | T-063, T-064, T-070 | T-063 | Uses T-063 targets/leases and core mesh/view DTOs (T-070 transitively); primitive fixtures require no mesh file parser. |
| T-066 | T-064 | T-043 | Reuses T-043 bounded linear EXR decoding policy and approved HDR feature; mesh IO is unrelated. |
| T-067 | T-065, T-066 | T-065 | Uses T-065 shading/resource contracts; synthetic linear HDR tests avoid an IO dependency in GPU. |
| T-068 | T-067 | T-065 | Uses T-065 shared material shader/TBN; IBL does not supply POM inputs (shader write exclusion remains). |
| T-069 | T-068, T-073 | T-064, T-066, T-067, T-068, T-073 | Joins real geometry/HDR IO, both lighting/POM branches and T-073 completed UI/input routes; no gate supplies an API. |
| T-070 | T-062 | T-044, T-055 | Extends T-044 command/revision and T-055 stroke semantics for lock enforcement; no GPU result is needed. |
| T-071 | T-070 | T-061, T-070 | Uses T-061 working map routes/2D views and T-070 visibility/lock state; no M1 gate output is used. |
| T-072 | T-071 | T-071 | Uses T-071 panel slots and its transitive live map/mask/command producers. |
| T-073 | T-072 | T-072 | Consumes complete T-072 controls/gesture routes for focus handling and catalog coverage. |
| T-074 | T-069 | T-062, T-063, T-064, T-065, T-066, T-067, T-068, T-069, T-070, T-071, T-072, T-073 | Explicit M2 join plus M1 gate evidence; validates both renderer and UI branches. |
| T-075 | T-074 | T-070 | Uses T-070 settings/document revision and registry contracts over existing typed graph; no viewport gate required. |
| T-076 | T-075 | T-075 | Consumes T-075 registered preset table/schema. |
| T-077 | T-075, T-080 | T-075 | Consumes T-075 naming/settings DTOs; staged byte-writer/failure fixtures do not need real encoders. |
| T-078 | T-076, T-077 | T-046, T-075 | Uses T-075 channel/size contract and T-046 typed sampling; no preset table or filesystem transaction needed. |
| T-079 | T-078 | T-078 | Consumes T-078 packed planes and conversion/size metadata. |
| T-080 | T-074 | T-040 | Uses T-040 audited codecs; independent CPU plane/block API, no export-plan or M2 dependency. |
| T-081 | T-079, T-080 | T-079 | Consumes T-079 bounded block readback/job identity; DDS writing stays in IO and is joined in T-082/T-095. |
| T-082 | T-081, T-085, T-086 | T-073, T-076, T-077, T-080, T-081 | Joins actual preset data, transactions, containers and GPU normal/readback paths with T-073 app routes; project and batch models are unnecessary. |
| T-083 | T-082 | T-082, T-085 | Uses T-082 working export bar/jobs and T-085 bounded JSON/preset persistence primitives. |
| T-084 | T-076 | T-076 | Consumes T-076 versioned preset IDs/revisions plus T-075 export and T-070 saved-view schemas transitively. |
| T-085 | T-084, T-077 | T-077, T-084 | Consumes T-084 project schema and T-077 filesystem recovery/preflight primitives. |
| T-086 | T-084 | T-075 | Uses T-075 captured export settings with existing graph/stroke recipe types; project-file DTO is not batch state. |
| T-087 | T-083, T-086 | T-082, T-085, T-086 | Uses T-082 export job coordinator, T-085 bounded JSON/recovery primitives for manifests and T-086 state machine; no editor dependency. |
| T-088 | T-085, T-087 | T-083, T-087 | Joins project IO/batch execution transitively through T-087 and T-083 packing scenario for the required workflow aggregate. |
| T-089 | T-074 | T-040, T-120 | Uses T-040 approved native features and T-120 startup/watchdog behavior; no completed viewport needed. |
| T-090 | T-089 | T-089 | Consumes T-089 native target/build policy and feature choices. |
| T-091 | T-089, T-090 | T-089 | Consumes T-089 Windows CRT/runtime inventory; CI YAML is not an installer input. |
| T-092 | T-091 | T-089 | Consumes T-089 macOS deployment/runtime inventory; Windows installer is unrelated. |
| T-093 | T-092 | T-089 | Consumes T-089 Linux runtime/glibc inventory; macOS package is unrelated. |
| T-094 | T-090, T-091, T-092, T-093 | T-090, T-091, T-092, T-093 | Consumes CI matrix and all three native packaging script interfaces. |
| T-095 | T-076, T-080, T-081 | T-076, T-080, T-081 | Consumes T-076 profile tables, T-080 container APIs and T-081 original block/mip evidence (no GPU crate import into IO). |
| T-096 | T-088, T-095 | T-082, T-095 | Uses T-082 real immutable export pipeline and T-095 validated fixtures; project/batch UI does not implement timing. |
| T-097 | T-096, T-098 | T-069, T-088, T-095 | Consumes T-069 comparison UI, T-088 workflow aggregate and T-095 numeric file checker; no performance or platform report input. |
| T-098 | T-094 | T-069, T-088, T-094, T-096 | Consumes T-094 report schema and final source outputs T-069/T-088/T-096 for same-revision native artifacts; can run beside T-097. |
| T-099 | T-097, T-098 | T-062, T-074, T-075, T-076, T-077, T-078, T-079, T-080, T-081, T-082, T-083, T-084, T-085, T-086, T-087, T-088, T-089, T-090, T-091, T-092, T-093, T-094, T-095, T-096, T-097, T-098 | Explicit join of every M3 task and prior M1/M2 evidence; no early export/packaging result substitutes for final integrated acceptance. |

## Alternatives considered

- Remove dependency arrows only: old ADR barriers and broad Owns would still
  serialize dispatch or create unsafe shared implementation edits.
- One writer per crate: simple, but unnecessarily serializes independent kernels
  and IO encoders. Keep only the actual structural/UI/shader exclusions.
- Add a new registry-framework task or split all tasks: expands scope and changes
  task/acceptance ownership. Existing explicit dispatch plus leaf modules is enough;
  structural rewrites remain inside the already-assigned feature tasks.
- Treat all shared-file edits as conflict-free: Rust enum/match/field changes and
  common WGSL layouts can collide semantically. Serialize those owners explicitly.

## Consequences

Parallelism is in authoring and independent CPU checks; merges and hardware
measurements remain serial. Integration rechecks are required even for a clean
text merge. No speedup factor or runtime/performance pass is claimed here.
Leaf fixtures remove incidental input-producer dependencies, but full material
composition is still tested by T-061/T-062 and final gates.

Core remains wasm32-compatible and lower layers do not import upper layers.
R32Float/Rgba16Float precision, linear/data transfer, normal convention, seeds,
generation/job cancellation and the shared 1 GiB GPU/512 MiB batch CPU accounting
are unchanged. Markers grant no permission to weaken these contracts or add
unbudgeted allocations. All algorithms/assets remain original and MIT-compatible;
no new dependencies or network operations are authorized.

## Open questions for the human

No new product decision is needed. The orchestrator must accept this scheduling
supersession and enforce file/measurement leases. Existing quality, platform,
packaging-policy and unsigned-beta questions from ADRs 0004–0010 remain unchanged.

## Validation (run from repository root)

T-031 is a documentation-only change. The following PowerShell audit checks exact
headers/table agreement, graph acyclicity, gate coverage, protected text against
HEAD, narrowed ownership, round coverage and file/T4 exclusions. It does not run
future implementation tasks' cargo/GPU/native acceptance or claim those pass.
Use it before these task definitions are committed; after commit, compare against
the pre-T-031 revision instead of HEAD for the protected-text check.

```powershell
$ErrorActionPreference = 'Stop'
$adr = Get-Content docs/adr/0011-faz1-parallel-plan.md -Raw
$deps = @{}; $texts = @{}
foreach ($id in 48..99) {
    $name = 'T-{0:D3}' -f $id
    $rel = "planning/tasks/$name.md"
    $t = Get-Content $rel -Raw
    $texts[$name] = $t
    $line = [regex]::Match($t, '(?m)^- \*\*Depends on:\*\* ([^\r\n]+)$').Groups[1].Value
    if ($line -notmatch '^T-\d{3}(, T-\d{3})*$') { throw "$name dependency syntax" }
    $deps[$name] = @($line -split ', ')
    foreach ($d in $deps[$name]) {
        if (!(Test-Path "planning/tasks/$d.md")) { throw "$name missing $d" }
    }
    $row = [regex]::Match($adr, "(?m)^\| $name \|.*$").Value.Split('|')
    if ($row.Count -ne 6 -or $row[3].Trim() -ne $line) { throw "$name table mismatch" }
    if ($id -notin @(55,62,74,90,91,92,93,94,95,97,98,99)) {
        if ($t -match '(?m)^- `crates/[^/]+/(src|tests|shaders)/`$') {
            throw "$name broad crate ownership"
        }
    }
    # Compare all protected text, including Goal, Acceptance, tiers and Context.
    $oldLines = & git show ("HEAD:" + $rel)
    if ($LASTEXITCODE -ne 0) { throw "$name baseline unavailable" }
    $old = $oldLines -join "`n"
    function Protected([string]$s) {
        $s = $s.Replace("`r`n", "`n")
        $s = [regex]::Replace($s, '(?m)^- \*\*(Depends on|Milestone / wave):\*\*.*\n', '')
        $s = [regex]::Replace($s, '(?ms)^## (Owns \(may edit\)|Must not touch|Notes for the agent)\n.*?(?=^## |\z)', '')
        return $s.TrimEnd()
    }
    if ((Protected $old) -cne (Protected $t)) { throw "$name protected text changed" }
}
$seen = @{}; $active = @{}
function Visit([string]$n) {
    if ($active[$n]) { throw "cycle at $n" }
    if ($seen[$n]) { return }
    $active[$n] = $true
    foreach ($d in @($deps[$n])) { if ($deps.ContainsKey($d)) { Visit $d } }
    $active[$n] = $false; $seen[$n] = $true
}
foreach ($n in $deps.Keys) { Visit $n }
$joins = @{
    'T-062' = @((40..61 | ForEach-Object { 'T-{0:D3}' -f $_ })) + @('T-120')
    'T-074' = @('T-062') + @((63..73 | ForEach-Object { 'T-{0:D3}' -f $_ }))
    'T-099' = @('T-062','T-074') + @((75..98 | ForEach-Object { 'T-{0:D3}' -f $_ }))
}
foreach ($g in $joins.Keys) {
    if (@(Compare-Object $joins[$g] $deps[$g]).Count) { throw "$g incomplete join" }
}
foreach ($n in $deps.Keys) {
    if (!$joins.ContainsKey($n) -and @($deps[$n] | Where-Object { $_ -in @('T-062','T-074','T-099') }).Count) {
        throw "$n unexpected milestone gate edge"
    }
}
function Conflict([string]$a, [string]$b) {
    $ag = $a -eq 'T-047' -or $texts[$a] -match 'Dispatch exclusions:.*GPU-'
    $bg = $b -eq 'T-047' -or $texts[$b] -match 'Dispatch exclusions:.*GPU-'
    if (($a -eq 'T-047' -or $texts[$a] -match 'Dispatch exclusions:.*GPU-STRUCTURAL') -and $bg) { return $true }
    if (($b -eq 'T-047' -or $texts[$b] -match 'Dispatch exclusions:.*GPU-STRUCTURAL') -and $ag) { return $true }
    foreach ($lock in @('CORE-SESSION','UI-SHELL','PBR-SHARED')) {
        if ($texts[$a] -match "Dispatch exclusions:.*$lock" -and $texts[$b] -match "Dispatch exclusions:.*$lock") { return $true }
    }
    return $false
}
$done = @{}; foreach ($i in @(40..46) + @(55,120)) { $done[('T-{0:D3}' -f $i)] = $true }
$count = 0; $full = 0
foreach ($m in [regex]::Matches($adr, '(?m)^\| R\d{2} \| ([^|]+) \|$')) {
    $row = @($m.Groups[1].Value.Trim() -split ', ')
    if ($row.Count -gt 3) { throw 'writer limit' }
    $t4 = @($row | Where-Object { $texts[$_] -match '\*\*Tier:\*\* T4' }).Count
    if ($t4 -gt 1) { throw 'T4 limit' }
    foreach ($n in $row) {
        if ($done[$n]) { throw "duplicate schedule $n" }
        if ($deps.ContainsKey($n)) {
            foreach ($d in $deps[$n]) { if (!$done[$d]) { throw "$n before $d" } }
        } elseif ($n -ne 'T-047') { throw "unknown schedule member $n" }
        foreach ($other in $row) {
            if ($n -ne $other -and (Conflict $n $other)) { throw "$n conflicts with $other" }
        }
    }
    foreach ($n in $row) { $done[$n] = $true }
    $count++; if ($row.Count -eq 3) { $full++ }
}
foreach ($n in $deps.Keys) { if (!$done[$n]) { throw "unscheduled $n" } }
if ($count -ne 19 -or $full -ne 16) { throw 'schedule occupancy mismatch' }
"PASS: 52 tasks; protected text/tiers; DAG and gate joins; 16/19 full rounds; exclusions."
```

Also run `git diff --check -- planning/tasks docs/adr/0011-faz1-parallel-plan.md`.
Review the dependency reasons against producers' Goals; structural checks cannot
prove that a proposed API dependency is semantically necessary.

