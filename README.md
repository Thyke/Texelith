# Texelith

Photo → engine-ready, tileable PBR material set. Lightweight, GPU-native, Rust.

[![CI](https://github.com/Thyke/Texelith/actions/workflows/ci.yml/badge.svg)](https://github.com/Thyke/Texelith/actions/workflows/ci.yml)

## Features

- Seven material maps: albedo, height, normal, ambient occlusion, roughness, metallic and edge (curvature).
- Seamless tools and a seam heatmap.
- 3D preview with procedural sky, HDRI lighting and a Preview ball.
- One-click export presets for Unity, Unreal and Godot, including DDS BC5 normals for Unreal.
- Project files and batch processing.

![Texelith split view with a synthetic brick material](docs/images/hero.png)

[Split view](docs/images/split-view.png) · [Export options](docs/images/export.png)

## System requirements

Windows 10 or later and a GPU with DirectX 12 or Vulkan support. Linux and macOS builds are experimental.

## Download and install

Download the latest build from [GitHub Releases](https://github.com/Thyke/Texelith/releases), unzip it and run `Texelith.exe`.

## Build from source

Install the Rust toolchain specified in `rust-toolchain.toml`, then run from the repository root:

```sh
cargo run -p texelith-ui-egui --release
```

## Quick start

1. Open an image.
2. Adjust the material maps and inspect the result in the preview.
3. Choose an export preset and export the material set.

## Roadmap

Phase 2 plans layers and painting tools. See [ROADMAP.md](ROADMAP.md) for the roadmap.

## AI-assisted development

Texelith is developed with AI assistance: Claude Code orchestrates the work and OpenAI Codex agents implement and test changes. Every change is reviewed, verified and merged by the maintainer.

## License

[MIT](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled components.
