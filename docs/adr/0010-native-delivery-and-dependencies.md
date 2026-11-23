# ADR 0010: Native builds, dependency gate and beta packaging

Status: accepted (orchestrator, 2026-11-11, under user autonomy mandate).

## Context

Current CI is Windows only; MSRV is 1.95, pin is 1.99.1, egui/wgpu are 0.37/30.
Local target hardware is Windows 11 RTX 5070 Ti. macOS/Linux results must come
from CI; a successful cross-compile is not runtime verification. Public beta needs
installable artifacts on each OS, <50 MB installer target, and honest signing status.
No network research or dependency resolution was performed for this architecture.

## Decision

### Dependencies and policy ownership

Keep the four-crate DAG of ADR 0001. T-040 exclusively edits manifests, lockfile,
dependency policy and notices before domain work. No dependency upgrades to the
egui/wgpu family are implied. Resolve exact compatible versions from available
registry metadata when implementing, record source/license/features/MSRV, update
the exact-version policy and commit lockfile only through the orchestrator.
These are candidates pending that gate, not verified license/version assertions:

| Addition / scope | Candidate license | Justification and minimum feature policy |
| --- | --- | --- |
| exr, IO only | BSD-3-Clause | float32 displacement and HDR input; disable rayon/optional defaults if supported |
| gltf, IO only | MIT OR Apache-2.0 | glTF/GLB static geometry; no automatic import/images/network |
| tobj, IO only | MIT | bounded OBJ geometry; no material texture loader |
| fluent-bundle, UI only | MIT OR Apache-2.0 | localized messages; no runtime elsewhere |
| unic-langid, UI only | MIT OR Apache-2.0 | locale selection for Fluent |
| serde_json, IO direct | MIT OR Apache-2.0 (already locked) | project/preset/job manifest parsing; core remains dev-only |
| image hdr feature | MIT OR Apache-2.0 (already locked) | Radiance HDR decoding; retain PNG/JPEG, no all-formats default |
| eframe x11/wayland features | MIT OR Apache-2.0 | Linux native windows; verify available flags in locked version |

T-040 must inspect all newly resolved transitive licenses, including targets not
installed locally. New dependencies allowed only MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause, ISC, Zlib or Unicode-3.0; new fonts OFL-1.1 only with scoped exception.
Previously audited Faz 0 license exceptions are grandfathered by exact package
identity, not a general whitelist for new packages. This narrows ADR 0001's broad
allowlist for additions. If candidates introduce other licenses, unsupported MSRV
or incompatible features, report blocked with the concrete resolution; do not
relax deny.toml. No new math, async runtime, compression engine, installer crate,
AI/ort, CLI or MCP dependency. Dependency/tool fetches only; no network in app.

### Build and CI matrix

Windows x86_64-pc-windows-msvc / DX12 default; macOS aarch64-apple-darwin / Metal;
Linux x86_64-unknown-linux-gnu / Vulkan, X11 and Wayland. Intel macOS is initially
build-only if runner capacity permits, not a universal-binary promise. Preserve
explicit backend/adapter override and report actual selection. Keep graceful
error for missing adapters/features, not silent CPU renderer fallback.
T-089 isolates target-specific native window/backend assumptions and uses
std::path correctly; local checks prove Windows only until hosted jobs run.
Declare minimum OS, glibc and macOS deployment targets in the build inventory;
use target-scoped static MSVC CRT or explicitly package its required runtime.
Load heavy compute/IBL pipelines on first use after the initial usable frame,
on the worker, and report cold pipeline time separately. Keep <1 s cold-start
measurement honest; an empty unresponsive shell is not a usable first frame.

T-090 extends ci.yml to pinned runner OS releases available at implementation.
Each OS: cargo fmt --check, clippy --all-targets -D warnings, build/test --locked,
layering/dependency policy self-tests and gates, cargo deny licenses/bans/sources.
Run MSRV 1.95.0 check and core wasm32 check in a dedicated matrix job. Use pwsh
on all platforms and isolate Windows-only P/Invoke measurement script to Windows.
Pin actions to audited immutable SHAs; reuse the already pinned checkout/cache
where compatible. Cache key includes OS, architecture, toolchain and lock hash.
Linux runner installs required X11/Wayland/Vulkan development/runtime packages
from the OS package manager only as build prerequisites; inventory in CI docs.
CI CPU tests must not silently skip explicitly requested GPU tests. Separate
optional GPU capability probe reports unavailable, then a dedicated GPU smoke job
fails if it was configured to require an adapter. A missing hosted hardware GPU
does not invalidate CPU build results or create a GPU PASS. Software Vulkan/Xvfb
may verify lifecycle, labeled software, never counted as hardware/perf evidence.

### Packaging

Use checked-in native packaging scripts, no new Rust packaging dependency. Tools
are build prerequisites outside the distributed application, with exact versions,
licenses and artifact inventory recorded before execution. No GPL library/code
is linked or bundled. Native OS utilities are execution tools, not vendored code;
Linux dpkg/fakeroot (GPL tools) may build .deb only as explicit build-tool scope,
never become dependencies or distributed files. If the project's tool policy
forbids even using those installed utilities, report blocked for a human decision;
do not substitute an undocumented installer. PowerShell is MIT; script source is MIT.

| Platform | Beta artifact / installation | Verification |
| --- | --- | --- |
| Windows | ZIP containing EXE, notices, install.ps1/uninstall.ps1; per-user LocalAppData install and Start Menu shortcut | fresh-user install, launch, export, uninstall preserves user projects |
| macOS | .app bundle in .dmg using system hdiutil; drag to Applications | bundle plist/architecture/resources, attach/copy/launch/remove in macOS CI |
| Linux | .deb using system dpkg-deb; desktop entry/icon/MIME and binary under /opt/texelith | package metadata, dependency closure, isolated install/launch/remove in Ubuntu CI |

Windows portable ZIP alone does not count: include and test the per-user installer
entry script, no admin requirement, paths with spaces/non-ASCII supported. This is
an intentionally minimal script installer, not MSI/EXE wizard integration. Build
icons from original simple vector source in packaging assets (no stock-art download).
No AppImage tooling or GPL runtime payload. All packages include MIT project
license text and THIRD_PARTY_NOTICES with exact bundled dependencies/fonts.
Build size target is compressed artifact <50,000,000 bytes, measured individually;
record installed size separately. Strip where safe, no UPX, no omitted notices.
Project/source assets are user data and never removed by uninstall.

T-094 separate package workflow builds on matching native runner and retains
artifacts+SHA256+build identity+license inventory. It does not publish a release,
upload to a store or request signing secrets. T-098 consumes supplied CI evidence
locally; orchestrator triggers/collects hosted runs under its authorization.
T-099 distinguishes artifact-built, installed, launched, GPU-tested and signed
per platform. Local Windows cannot stand in for macOS/Linux results.

Signing/notarization is deferred for beta: unsigned Windows/macOS packages are
clearly labeled in installer docs; OS trust prompts expected. Do not advise
disabling protections globally. Public release publication/signing is a separate
orchestrator/human decision; absence of keys is not an excuse to claim signed.
Record macOS ad-hoc bundle signing when needed for local runnable artifacts as
ad-hoc only, not Developer ID or notarization. No code signing in this design task.

### Gates and evidence

T-062/T-074/T-099 gate demos; source fixtures are generated/original, screenshots
<=2 MB each, full maps/raw logs under target-evidence/T-xxx. Keep failed numerical
results separate from human disposition. T-096 keeps 4K export <2 s and old slider
gate; T-099 also records cold start <1 s per platform where actually measurable.
No claimed macOS/Linux visual/perf pass from CPU CI. Missing hosted evidence means
platform release gate blocked, while implemented artifact scripts can still be
reported complete with their local checks. Never turn NOT RUN into PASS.

## Alternatives considered

- MSI/WiX, NSIS, AppImage or installer Rust crates now: add toolchain/license and
  maintenance scope; native scripts produce testable beta installs first.
- Shipping only Windows while listing all three supported: contradicts MVP scope.
- Requiring signing secrets to build: prevents reproducible fork/CI artifacts.
- Allow every old license exception for new dependencies: widens the user's policy.

## Consequences

Build, runtime, GPU and distribution claims have separate evidence. Native tools
need explicit inventory; the minimal Windows installer trades polish for a small
dependency surface. A later installer upgrade does not change document/graph APIs.

## Open questions for the human

Confirm script-based per-user Windows installation and unsigned beta delivery.
Confirm native Linux packaging-tool license exception if needed; no new GPL
dependency or artifact payload is permitted. Confirm whether Intel macOS runtime
support is required before calling the beta generally available.
