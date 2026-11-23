//! Strict project decoding adapters for shared runtime types. Keep project schema
//! checks local rather than changing the command wire format of earlier tasks.
use crate::*;
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(remote = "crate::Camera", deny_unknown_fields)]
pub(super) struct Camera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: [f32; 3],
}
#[derive(Deserialize)]
#[serde(remote = "crate::Light", deny_unknown_fields)]
pub(super) struct Light {
    direction: [f32; 3],
    intensity: f32,
    environment: Option<OpaqueLocator>,
}
#[derive(Deserialize)]
#[serde(remote = "crate::Tiles", deny_unknown_fields)]
pub(super) struct Tiles {
    count: u32,
    seed: u32,
    random_rotation: bool,
}
#[derive(Deserialize)]
#[serde(remote = "crate::Pom", deny_unknown_fields)]
pub(super) struct Pom {
    enabled: bool,
    scale: f32,
}
#[derive(Deserialize)]
#[serde(remote = "crate::MapView", deny_unknown_fields)]
struct MapView {
    visible: bool,
    locked: bool,
}
#[derive(Deserialize)]
struct MapEntry(#[serde(deserialize_with = "MapView::deserialize")] crate::MapView);

pub(super) fn maps<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<MapKind, crate::MapView>, D::Error> {
    Ok(BTreeMap::<MapKind, MapEntry>::deserialize(d)?
        .into_iter()
        .map(|(k, v)| (k, v.0))
        .collect())
}

#[derive(Deserialize)]
#[serde(remote = "crate::SizePolicy", deny_unknown_fields)]
enum SizePolicy {
    Keep,
    Resize { width: u32, height: u32 },
    Pad { width: u32, height: u32 },
}
#[derive(Deserialize)]
#[serde(remote = "crate::ExportPlan", deny_unknown_fields)]
pub(super) struct ExportPlan {
    source_extent: Extent,
    #[serde(deserialize_with = "SizePolicy::deserialize")]
    size: crate::SizePolicy,
    naming: String,
    profile: EngineProfile,
    outputs: Vec<ExportOutput>,
    #[serde(default)]
    displacement: DisplacementSettings,
    padding: Boundary,
}
