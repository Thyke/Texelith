# ADR 0012: One GPU node registry

Status: proposed by T-121; supersedes ADR 0011's multiple GPU registration markers.

## Decision

`crates/texelith-gpu/src/node_registry.rs` contains the single `gpu_nodes!`
list. Its callback consumers declare kernel modules/re-exports, worker adapters,
statically typed runtime state, eviction order, and verification modules/runners
and suite membership. No new dependency, reflection, build script or shader change.
Existing root exports and the public `seamless_offset` module remain available.
Poisson operators have a suite but no material-node adapter: registering operators
must not advertise the unfinished HeightFromNormal solver as supported.

Each entry has this shape (one appended line):

```rust
curvature {PreparedCurvature} [curvature, "curvature/runtime.rs"] ["curvature", "mvp_support/t052/mod.rs", [curvature_contract, "t052/report_contract.rs"]];
```

The first name is the kernel module; braces list root re-exports. Optional `pub`
before the name exposes the module itself. The first brackets declare a worker
child module and matching state field; paths are relative to `src/worker_runtime.rs`.
The second brackets name a suite and its module, relative to `examples/verify_mvp.rs`.
The suite's nested brackets optionally register a report-validator module, with a
path relative to `examples/mvp_support/contract.rs`. That module provides
`pub fn validate(&serde_json::Value) -> Result<(), String>` for node-specific checks
only; it must not call the shared `validate_report` again. The shared validator
runs generic checks first, then the registered validator. This lets T-052's existing
tests keep calling `validate_report` without central match edits.
An empty bracket pair omits that integration. Existing normal/baseline and
resources/viewport aliases and aggregate suite order are preserved.

Worker adapters use `super::*` and provide `pub(super)` items:

```rust
#[derive(Default)]
struct State; // or owned cached preparations/pipelines
impl State { fn evict(&mut self) -> bool { false } }
fn supports(node: &MaterialNode) -> bool;
fn scratch(node: &MaterialNode, extent: Extent) -> Result<u64, MaterialError>;
fn evaluate(runtime: &mut Runtime, context: &NodeContext<'_>) -> Result<Lease>;
```

These signatures illustrate the contract; qualify scratch's `std::result::Result`
because the runtime's `Result<T>` uses `Failure`. `supports` must be disjoint
from other registrations, including individual seamless methods. Planning and
execution share the same descriptor lookup. Dispatch is one function-pointer call
per node output preparation; pass encoding and runtime state stay statically typed.
`State::evict` releases completed, unpinned resources and returns whether it freed
anything. Registration order preserves height-before-normal eviction. All states
are dropped before pool shutdown. Admit replacement outputs while old cached
state is still reachable by eviction; retain leases through submission fences.

`NodeContext` supplies the work/cancellation identity, node parameters, requested
`output` port, evaluation `extent`, original `source_extent`, and input helpers:
`input()` / `input_key()` for the first edge and `input_at(InputPort)` for named
optional inputs. Allocate through `Runtime::allocate`, submit/wait through its
existing fence helpers, and check `current(work)` between passes. Budget estimates
must include all scratch/staging and must match actual allocations.

Bookkeeping is keyed by a local ordered wrapper of core `OutputRef`, so secondary
ports have distinct cache keys, payload reservations and last uses without changing
core DTOs. Each requested output is evaluated separately in topological order;
a multi-output adapter selects `context.output.port` and returns that texture.
It may reuse budgeted completed state for sibling outputs. This supplies the
Curvature/Edge/Cavity integration point without implementing curvature math.
Zero-input nodes need not call input helpers. Public proxy restrictions and source
conversion behavior remain unchanged; new adapters must honor `context.extent`.

Each suite module provides
`pub(crate) fn run_registered(&crate::SuiteContext<'_>) -> Result<serde_json::Value, String>`.
It owns its numerical checks/report validation and can delegate to its existing
runner. Node-specific report requirements can be registered in the nested validator
brackets. The shared selector derives membership from the same list; empty and
unknown names still fail. Legacy report-validation exports remain compatible.

## Add a node (10-line checklist)

1. Confirm the core node parameters/ports already exist; their contracts remain in core.
2. Create `src/<node>/mod.rs` with the kernel and public prepared-pass types.
3. Preserve ADR 0004's group-0 ABI and put group-1 bindings/math in that module.
4. Create `src/<node>/runtime.rs` as a worker adapter using `super::*`.
5. Define `State: Default` and `evict`, keeping completed budgeted caches evictable.
6. Implement disjoint `supports` and checked `scratch` with the existing precision/budget rules.
7. Implement `evaluate` for `context.output.port`, named inputs, extent and cancellation/fences.
8. Add the node-local suite module and `run_registered` entry with real checks and report validation.
9. Append one entry to `src/node_registry.rs`; no material/worker/lib/example match edits.
10. Format leaf files explicitly, run fmt/clippy/workspace tests/layering, then measure the headless suite.

Rustfmt does not discover modules declared inside macro expansions: run
`rustfmt --edition 2024 --config skip_children=true` on added/changed runtime and
suite files as well as `cargo fmt --all`. T-050 can add a solver entry alongside
operator-only `poisson_operators`; T-052 uses the requested port; T-054 can use
roughness's optional-mask adapter as the plumbing example.

## Validation

Workspace tests: **702 passed, 1 existing native test ignored**. Clippy with
`--workspace --all-targets --locked -- -D warnings`, workspace fmt plus explicit
leaf rustfmt, and `scripts/check-layering.ps1 -SelfTest` passed. Cargo checks used
`--offline`; `CARGO_INCREMENTAL=0` avoids sandbox hard-link warnings. The first
workspace run failed an unchanged IO overwrite test with Windows Access denied
in the external temp directory; full reruns with `TEMP`/`TMP` under this worktree
passed. No existing test assertions, core APIs, shader math or dependencies changed.
The native `measure-faz0` gate is deferred to the orchestrator.

The 4K DX12 RTX 5070 Ti comparison used the preserved original release executable
and the refactored release executable, serially with no overlapping agent builds
or GPU tests. Each existing suite retains 10 warmups/100 completion-bound samples.
The table compares median run p95s in ms; raw per-run failures remain in the
[verification summary](../../crates/texelith-gpu/examples/mvp_support/t121/verification.json).
Initial unpaired runs overlapped a development build and are excluded. Three
paired runs were followed by two additional height pairs to investigate its failed,
variable result. No failed samples were discarded.

| Suite | Default before -> after (ms) | Change | Maximum before -> after (ms) | Change |
| --- | --- | --- | --- | --- |
| height (5 pairs) | 9.7305 -> 11.7190 | +20.44% | 9.5105 -> 10.2146 | +7.40% |
| normal (3 pairs) | 5.3579 -> 4.9355 | -7.88% | 126.1943 -> 127.1081 | +0.72% |
| poisson_operators (3 pairs) | 11.0613 -> 10.9851 | -0.69% | 19.9498 -> 20.3002 | +1.76% |
| roughness (3 pairs) | 0.7573 -> 0.7310 | -3.47% | 1.2102 -> 1.0552 | -12.81% |
| ao (3 pairs) | 10.0949 -> 9.4181 | -6.70% | 382.4470 -> 384.7493 | +0.60% |
| seamless_offset (3 pairs) | 6.8517 -> 6.2230 | -9.18% | 6.5598 -> 6.2988 | -3.98% |

**Acceptance remains failed:** height's default (+20.44%) and maximum-quality
(+7.40%) medians exceed the 5% allowance.
Its initial three-pair comparison was also a failure (+18.27%). Other suite
medians meet the allowance. All numerical checks and parameter manifests match;
some individual paired runs still fail the existing absolute 16 ms gate, retained
separately in the JSON. The measured height kernel, pass encoding and timing loop
are unchanged; variable completion bounds do not establish the slowdown's cause.
An isolated hardware rerun or explicit orchestrator disposition is still required.
Raw reports/logs and the baseline executable are under `target-evidence/T-121/`
(git-ignored); the small committed JSON includes exact commands and binary hashes.
