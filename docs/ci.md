# Continuous integration

`ci.yml` runs on every push, pull request and manual dispatch. Windows, Ubuntu
and macOS hosted runners each run formatting, Clippy with warnings denied, and
`cargo test --workspace --locked -- --nocapture`. A separate job checks MSRV
1.95.0 and the core crate on wasm32. The policy job checks layering, dependency
versions/features/licenses, and cargo-deny licenses, bans and sources. Rust
1.99.1 and its components come from `rust-toolchain.toml`; actions use immutable
commit SHAs. Cargo caches include runner OS/architecture, toolchain and lockfile.
`--locked` permits downloads; CI does not assume an offline/pre-populated cache.

Linux build prerequisites are listed in `scripts/install-linux-ci-deps.ps1`:
compiler/pkg-config, X11/XCB, Wayland, xkbcommon, udev, EGL and Vulkan headers.
The locked rfd defaults use XDG portals/Wayland on Linux, not GTK; its macOS
backend is target-scoped Cocoa. eframe enables X11/Wayland only on applicable
targets. No manifest feature change or new Rust dependency is needed.
PowerShell Core (`pwsh`) is explicitly selected on all three hosted platforms;
Unix packaging scripts are invoked through Bash. macOS deployment minimum is
13.0 (`.cargo/config.toml`).

## GPU coverage

Hosted CI sets `TEXELITH_CI_NO_GPU=1`. Only GPU-dependent test entry points return
early, printing `SKIP GPU: no GPU adapter` with the test name. CPU oracles,
validation, scheduling and schema tests still run. Rust's standard test harness
counts an early return as `ok`; the explicit skip messages and job summary mean
these results **are not GPU assertion passes**. No assertion failure is caught or
converted into a skip. Unset the variable to require the existing real-device
checks (many historical suites require Windows/DX12); adapter/device/validation
or numerical failures then fail normally. Native window and performance gates,
including `measure-faz0.ps1`, are separate manual gates.

## Packages

`package.yml` runs only for `v*` tags or manual dispatch. It uploads artifacts;
it does not publish releases. Windows builds a ZIP with the per-user PowerShell
installer and tests isolated install/uninstall. Apple Silicon macOS produces a
DMG plus ZIP, and Linux x86_64 produces a DEB plus tar.gz. Native package scripts
retain their metadata, minimum OS, size, checksum and license checks. Ubuntu
22.04 packaging deliberately preserves the glibc 2.35 floor; macOS uses the
latest hosted runner and the package script requires native ARM64. Native launch checks remain opt-in script `--verify` modes and
are reported as not run in hosted packaging. Windows is unsigned; macOS uses
ad-hoc signing, without Developer ID/notarization. Hosted builds and downloads
must still confirm platform compilation, package tooling and artifact contents;
CPU CI does not establish rendering, performance or release readiness.

## README badge snippet

Badge:

```markdown
[![CI](https://github.com/Thyke/Texelith/actions/workflows/ci.yml/badge.svg)](https://github.com/Thyke/Texelith/actions/workflows/ci.yml)
```

Cross-target checks require installed Rust targets and target C toolchains/SDKs.
In particular, the Apple ARM64 blake3 build needs Apple C headers; installing
only the Rust standard library on Windows is insufficient.
