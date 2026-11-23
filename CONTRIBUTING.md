# Contributing

## Build and run

Install the Rust toolchain specified in `rust-toolchain.toml`. From the repository root:

```sh
cargo run -p texelith-ui-egui --release
```

## Check changes

```sh
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

## Pull requests

Describe the problem, the change and how you tested it. Keep changes focused and use the pull request template. Follow the [code of conduct](CODE_OF_CONDUCT.md). For architecture and dependency boundaries, see the [architecture overview](docs/architecture.md) and [architecture decisions](docs/adr/).

## Commits

Write a short, descriptive commit subject. Do not add AI co-author trailers.

## AI-assisted development

Texelith is developed with AI assistance: Claude Code orchestrates the work and OpenAI Codex agents implement and test changes. Every change is reviewed, verified and merged by the maintainer.
