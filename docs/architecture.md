# Architecture

The workspace has four crates. Dependencies point towards the lower layers; lower layers do not depend on the UI.

- `texelith-core` defines material graphs, commands, validation and evaluation data.
- `texelith-gpu` evaluates material operations and renders with wgpu.
- `texelith-io` handles image, project and export I/O.
- `texelith-ui-egui` provides the desktop interface and coordinates the other crates.

The UI sends commands to core and requests image loading through I/O. GPU work evaluates the material graph for previews and export; I/O writes the resulting material files. See the [architecture decisions](adr/) for contracts and design details.
