# ADR 0001: Faz 0 workspace, dependencies, and delivery waves

Status: accepted (user-approved 2026-11-10, with T-010 amendments).

## Task dependency graph and waves

Run a wave only after all tasks in the preceding wave pass and are integrated.
This conservative barrier supplements each task's explicit dependency list.
At most three tasks write concurrently; role passes within one task are sequential.
Only the orchestrator changes branches, integrates work, or updates the board.

```text
Wave 1   T-010 workspace and dependency resolution
             |
Wave 2   T-011 core contracts | T-012 image I/O | T-013 native render shell
             |                    |                 |
Wave 3   T-014 commands/undo | T-015 GPU kernels | T-016 CI/policy
             |                    |                 |
Wave 4   T-018 command UI   | T-017 GPU scheduling and resource lifetime
             \                    /
Wave 5              T-019 image-to-preview integration
                                  |
Wave 6              T-020 reproducible performance gate
                                  |
Wave 7              T-021 visual and final acceptance evidence
```

| Task | Suggested tier | Separate tests-first pass | Visual / performance pass |
| --- | --- | --- | --- |
| T-010 | T2 | No, mechanical build checks | Neither |
| T-011 | T2 | Yes | Neither |
| T-012 | T2 | Yes | Neither |
| T-013 | T3 | No, native smoke harness | Visual |
| T-014 | T3 | Yes | Neither |
| T-015 | T4 | Yes, CPU oracles before shaders | GPU numerical measurement |
| T-016 | T2 | No, policy self-tests | Neither |
| T-017 | T4 | Yes, fake completion before worker | GPU stress measurement |
| T-018 | T2 | Yes, command adapter tests | Visual |
| T-019 | T3 | No, integration scenarios | Visual |
| T-020 | T4 | Yes, metric/gate tests | Performance |
| T-021 | T2 | No, verification only | Visual and performance |

## Context

The root spec's Technology, Architecture, Speed Rules, Roadmap, Performance, and
Risks sections define Faz 0: a small native editor, three compute nodes, I/O,
commands, undo, and a smooth normal-intensity slider on a 4096 x 4096 image.
The checkout currently has no Rust workspace and no existing ADR decisions.
Rust is not installed; this task does not run cargo or access the network.
Versions below are a known coherent baseline, **verify on first build**, not a
claim about the newest releases, current advisories, or a verified lockfile.

## Decision

### Workspace and layering

Create four packages under `crates/`, with these allowed local dependencies:

```text
texelith-ui-egui --> texelith-gpu --> texelith-core
        |---------> texelith-io  --> texelith-core
        `------------------------> texelith-core
```

- `texelith-core`: graph/value types, validation, command descriptors and dispatch,
  parameter-diff history, immutable evaluation snapshots, canonical cache hashes.
  No GPU handles, GUI types, filesystem access, OS paths, clocks, or thread pools.
- `texelith-gpu`: native wgpu compute, source preparation, cache/pool, worker,
  output leases, optional asynchronous PNG-export readback. No egui dependency.
- `texelith-io`: bounded PNG/JPEG decoding and PNG writing; CPU image structures.
  Blocking functions called by the application's I/O worker, never by UI update.
- `texelith-ui-egui`: executable and testable library; eframe renderer bootstrap,
  orchestration, command adapter, controls, a single 2D viewport, diagnostics.

Do not create empty AI/CLI/MCP packages. `texelith-ai`, `texelith-cli`, and
`texelith-mcp` are planned; they may consume core and relevant lower services.
A future C ABI adapter owns opaque handles, explicit destruction and error codes;
Rust enums, `Vec`, and traits are not a C ABI. No `cdylib` or ABI promise now.
No docking framework, node editor, async runtime, AI, 3D preview, batch, engine
presets, disk cache, persistent pipeline cache, or project-file UI in Faz 0.

CI's `scripts/check-layering.ps1` consumes `cargo metadata --format-version 1`.
Check workspace package names and every local edge, including build/dev/optional
and target-specific dependencies, against the allowlist above; reject unlisted
workspace packages and aliases that hide edges. Inspect dependency records, not
only the host's active resolve graph. A root manifest policy rejects patches,
git dependencies, and undeclared dependencies until reviewed. Core's external
allowlist is serde/blake3 plus serde_json in dev only. This prevents an indirect
upper-layer dependency being hidden behind a newly introduced workspace crate.
All manifests belong to T-010; later tasks needing dependency changes report to
the orchestrator for a serial T-010 follow-up, rather than editing them in parallel.

### Toolchain and portability

- Edition 2024, resolver 3, workspace `rust-version = "1.95"`; candidate MSRV
  1.95. Pin `rust-toolchain.toml` to **1.99.1**, the known stable patch baseline.
  Both pin and MSRV are **verify on first build**; no rolling `stable` pin.
- T-010 verifies the dependency pair and MSRV, commits neither code nor lockfile
  itself, and hands the generated lockfile to the orchestrator. If an advisory,
  resolver, or API incompatibility requires an upgrade, stop and report the exact
  issue and proposed coherent versions; do not silently change this decision.
- Add rustfmt/clippy and `wasm32-unknown-unknown` as toolchain components/target.
  Check core for wasm without native siblings. Core supports `std` on wasm;
  `no_std`, browser GPU, wasm I/O, and wasm UI are not Faz 0 requirements.
- Native implementation and CI gate Windows MSVC first. macOS/Linux build and
  runtime coverage are explicitly deferred, not implied by wgpu's portability.
- Use errors/results at boundaries, stable typed values and snapshots; future
  executors need not run in the UI process or share GPU resources.

### Direct dependencies and feature policy

Use exact direct versions in `[workspace.dependencies]` and commit `Cargo.lock`.
Every row is **verify on first build**, including license metadata and features.
Select the permissive branch of dual licenses. No other direct crates are allowed.

| Dependency | Version | Intended license | Used by / reason |
| --- | --- | --- | --- |
| eframe | 0.37.0 | MIT OR Apache-2.0 | UI runner; disable defaults, enable `default_fonts,wgpu` |
| egui | 0.37.0 | MIT OR Apache-2.0 | UI controls, same family as eframe |
| egui-wgpu | 0.37.0 | MIT OR Apache-2.0 | UI native texture bridge; no second renderer |
| wgpu | 30.0.1 | MIT OR Apache-2.0 | GPU and UI bootstrap; WGSL and native backends |
| image | 0.25.10 | MIT OR Apache-2.0 | I/O; disable defaults, enable `png,jpeg` only |
| serde | 1.0.229 | MIT OR Apache-2.0 | core + diagnostics, `derive` |
| serde_json | 1.0.154 | MIT OR Apache-2.0 | diagnostic JSON and core serialization tests |
| blake3 | 1.8.7 | CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception | core hashes; choose Apache-2.0, no rayon |
| crossbeam-channel | 0.5.17 | MIT OR Apache-2.0 | GPU/UI bounded worker channels |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | GPU uniform packing; `derive` |
| pollster | 1.0.1 | MIT OR Apache-2.0 | GPU example/test bootstrap only |

`egui-wgpu`, `bytemuck`, and `pollster` are additions to the spec's shorthand list:
they avoid custom renderer integration, unsafe byte packing, and a production
async runtime respectively. Use `std` threads/time/errors for everything else.
T-010 must ensure `cargo tree -d` has one wgpu/egui family; eframe's actual
resolved wgpu version is authoritative when proposing any compatibility fix.

Tooling: rustup (MIT OR Apache-2.0), Rust/cargo/rustfmt/clippy (MIT OR Apache-2.0,
bundled LLVM components Apache-2.0 WITH LLVM-exception), cargo-deny **0.18.2**
(MIT OR Apache-2.0; version/MSRV verify on first build), PowerShell (MIT).
GitHub-hosted runner actions require MIT or similarly permissive licenses and
immutable SHA pins reviewed in T-016. No separate benchmarking crate is needed.
T-016 may fetch/install cargo-deny for checks; no cloud service calls are needed
for local acceptance. Advisory database refresh belongs to authorized CI setup.

The table is not a transitive-license audit. T-010 exports all locked package
names, versions, licenses and license-file references to
`docs/dependencies/faz0-lock-audit.md`; T-016 enforces them with cargo-deny.
Permit MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, CC0-1.0, Unicode-3.0,
Unicode-DFS-2016, BSL-1.0 and Apache-2.0 WITH LLVM-exception where actually needed.
Reject unknown/unlicensed or copyleft packages; inspect license files when SPDX
metadata is insufficient. Do not waive failures or assume disabling a feature
removes every target's transitive dependency. Review bundled font licenses too.

### Scope and ownership conventions

T-010 creates manifests and minimal compilable lib/bin roots; domain tasks then
own those roots serially. No production placeholders using `todo!()`.
Different crates can evolve in parallel after contracts below are agreed.
GPU and UI tasks never share a source file in the same wave. Test-author passes
write failing behavioral tests first, hand them off, and do not run concurrently
with the implementer of that task. Scratch artifacts stay inside task-owned paths.
The phase gate is numerical and visual; a green build alone is not acceptance.

## Alternatives considered

- All seven packages now: adds unused boundaries and dependencies with no phase
  benefit; future frontend seams already exist in core's command API.
- Latest unconstrained dependencies or rolling stable: unreproducible and not
  verifiable in this offline architecture task; resolve and audit the baseline first.
- Single crate: initially simpler, but makes command sharing and wasm isolation
  difficult to enforce. More than four crates would over-fragment this phase.
- Tokio/rayon, EXR/BCn, docking and a graph editor now: unnecessary to demonstrate
  the specified slider. CPU work uses one bounded I/O worker initially.

## Consequences

The first implementation task is a compatibility/license gate. The known versions
may need a reviewed refresh before code work; none are claimed security-current.
Faz 0 opens PNG/JPEG up to 4096 pixels on each axis and writes one PNG map; it is
not the eventual complete material exporter. The single viewport makes GPU
lifetime and responsiveness testable without a 3D material renderer.

## Open questions for the human

- Approve the explicit Windows-only Faz 0 gate and 4K input cap?
- Approve the ADR 0003 numerical meaning of "without stutter" and the 1 GiB
  application GPU budget? These are proposed acceptance defaults, not spec quotes.
- Approve deferring cold-start optimization/persistent pipeline caching after
  recording the cold-start metric in Faz 0? The spec's <1 second target remains.

These defaults make tasks concrete; the orchestrator obtains the approval in
T-001's role chain before dispatching implementation.
