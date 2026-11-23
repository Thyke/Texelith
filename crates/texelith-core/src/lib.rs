//! Owned, validated graph values and frontend-independent command contracts.
//!
//! # Canonical encoding
//!
//! Cache schema 1 is the concatenation below, with no implicit padding. Integers
//! are little-endian; tags are one byte. Parameters use finite IEEE-754 `f32`
//! bits, with either signed zero encoded as `0u32`. JSON, field insertion order,
//! node/edge IDs, selected output and snapshot generations never enter the key.
//!
//! 1. The literal bytes `texelith.node-cache\0`, then schema `1u32`.
//! 2. Algorithm revision `u32`, then color-policy revision `u32`.
//! 3. Node tag (Source=0, Levels=1, Blur=2, NormalFromHeight=3), map tag
//!    (Height=0, Normal=1), format tag (R32Float=0, Rgba16Float=1).
//! 4. Parameters in this fixed order: Source: 32 content-hash bytes, original
//!    descriptor width/height `u32`, interpretation tag (SrgbPhoto=0,
//!    LinearHeight=1); Levels: black/white/gamma; Blur: sigma_source_px;
//!    NormalFromHeight: strength.
//! 5. Input-key count `u32`, then each 32-byte key in input-port order.
//! 6. Evaluation width/height `u32`, then original source width/height `u32`.
//!
//! The key is BLAKE3 of those bytes. `SourceDescriptor::content_hash` must be
//! BLAKE3 of decoded, tightly packed, row-major, unassociated RGBA8 bytes (before
//! color conversion), never file bytes. [`SourceDescriptor::from_rgba8`] provides
//! that boundary without retaining pixels. Dimensions and interpretation are
//! encoded separately. Color policy 1 is ADR 0003's sRGB inverse transfer and
//! Rec.709 luminance / linear red policy, ignoring alpha, ICC and EXIF orientation.
//! Algorithm revision 1 uses ADR 0003's clamped kernels and OpenGL normal sign.
//! Bump the corresponding revision when that behavior changes, and the encoding
//! schema if this layout changes. [`CacheRevisions`] allows explicit backend
//! algorithm revisions; production defaults are [`CacheRevisions::CURRENT`].
//!
//! Graph serde retains the diagnostic [`GraphDto`] envelope for legacy graphs and
//! uses [`MaterialGraphDto`] schema 2 for typed material graphs. Both store nodes
//! sorted by ID and validate before producing live state. [`MaterialGraphDto`]
//! supports explicit validated schema-1 migration. Schema-2 cache encoding uses
//! [`canonical_material_node_bytes`], preserving schema-1 bytes on the old API.
//! Its frozen tags and field order are documented in `docs/evidence/T-041/implementation.md`.
//! Snapshots deliberately have no serde implementation: generations are runtime
//! state, not diagnostic project data. Command DTOs describe requests and effects;
//! [`Session`] owns dispatch and parameter-diff history; native I/O remains in
//! frontends that handle its typed effects.

mod cache;
mod command;
mod graph;
mod material;
mod material_cache;
mod material_parameters;
mod session;
mod stroke;

pub use cache::*;
pub use command::*;
pub use graph::*;
pub use material::*;
pub use material_cache::*;
pub use material_parameters::*;
pub use session::*;
pub use stroke::*;
// FAZ1_CORE_MODULES: T-075 begin
mod export;
mod export_presets;
pub use export::*;
pub use export_presets::*;
// FAZ1_CORE_MODULES: T-075 end
// FAZ1_CORE_MODULES: T-086 begin
mod batch;
pub use batch::*;
// FAZ1_CORE_MODULES: T-086 end

// FAZ1_CORE_MODULES: T-084 begin
mod project;
pub use project::*;
// FAZ1_CORE_MODULES: T-084 end
