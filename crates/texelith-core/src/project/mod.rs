use crate::{
    Camera, CoreError, ExportPlan, Graph, GraphDto, InputInterpretation, Light, MapKind, MapView,
    MaterialGraphDto, Pom, PreviewMesh, Tiles, ViewMode, ViewState, validate_export_for_graph,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod wire;

pub type AssetId = u64;
pub type AssetIntent = InputInterpretation;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetRef {
    pub id: AssetId,
    pub relative_path: String,
    pub decoded_hash: [u8; 32],
    pub interpretation: AssetIntent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedView {
    pub mode: ViewMode,
    pub mesh: PreviewMesh,
    #[serde(deserialize_with = "wire::Light::deserialize")]
    pub light: Light,
    #[serde(deserialize_with = "wire::Tiles::deserialize")]
    pub tiles: Tiles,
    #[serde(deserialize_with = "wire::Pom::deserialize")]
    pub pom: Pom,
    #[serde(deserialize_with = "wire::Camera::deserialize")]
    pub camera: Camera,
    #[serde(deserialize_with = "wire::maps")]
    pub maps: BTreeMap<MapKind, MapView>,
    pub split_fraction: f32,
}

impl From<ViewState> for SavedView {
    fn from(v: ViewState) -> Self {
        Self {
            mode: v.mode,
            mesh: v.mesh,
            light: v.light,
            tiles: v.tiles,
            pom: v.pom,
            camera: v.camera,
            maps: v.maps,
            split_fraction: v.split_fraction,
        }
    }
}
impl From<SavedView> for ViewState {
    fn from(v: SavedView) -> Self {
        Self {
            mode: v.mode,
            mesh: v.mesh,
            light: v.light,
            tiles: v.tiles,
            pom: v.pom,
            camera: v.camera,
            maps: v.maps,
            split_fraction: v.split_fraction,
            capture_a: None,
            capture_b: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectFile {
    pub format: String,
    pub schema_version: u32,
    pub graph: MaterialGraphDto,
    pub assets: Vec<AssetRef>,
    #[serde(deserialize_with = "wire::ExportPlan::deserialize")]
    pub export: ExportPlan,
    pub view: SavedView,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedProject {
    pub(crate) dto: ProjectFile,
    pub(crate) graph: Graph,
}
impl ValidatedProject {
    pub fn document(&self) -> &ProjectFile {
        &self.dto
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn into_document(self) -> ProjectFile {
        self.dto
    }
}

pub fn migrate_project(dto: ProjectFile) -> Result<ProjectFile, CoreError> {
    if dto.schema_version != 1 {
        return Err(CoreError::UnknownSchema(dto.schema_version));
    }
    Ok(dto)
}
pub fn adapt_diagnostic_graph(dto: GraphDto) -> Result<MaterialGraphDto, CoreError> {
    MaterialGraphDto::try_from(dto)
}

fn valid_component(part: &str) -> bool {
    if part == ".." {
        return true;
    }
    let base = part
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches([' ', '.'])
        .to_ascii_uppercase();
    let port = base
        .strip_prefix("COM")
        .or_else(|| base.strip_prefix("LPT"));
    !part.is_empty()
        && part != "."
        && !part.ends_with(['.', ' '])
        && !part.chars().any(|c| "<>\"|?*".contains(c))
        && !matches!(
            base.as_str(),
            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
        )
        && !port.is_some_and(|n| {
            matches!(
                n,
                "1" | "2"
                    | "3"
                    | "4"
                    | "5"
                    | "6"
                    | "7"
                    | "8"
                    | "9"
                    | "\u{00b9}"
                    | "\u{00b2}"
                    | "\u{00b3}"
            )
        })
}

fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && !path.contains(':')
        && !path.chars().any(char::is_control)
        && path.split('/').all(valid_component)
}

pub fn validate_project(dto: ProjectFile) -> Result<ValidatedProject, CoreError> {
    let mut dto = migrate_project(dto)?;
    if dto.format != "texelith.project" {
        return Err(CoreError::InvalidMaterial("invalid project format"));
    }
    if dto.assets.len() > 64 {
        return Err(CoreError::InvalidMaterial("too many project assets"));
    }
    let mut ids = BTreeSet::new();
    for asset in &dto.assets {
        if !ids.insert(asset.id) || !valid_path(&asset.relative_path) {
            return Err(CoreError::InvalidMaterial("invalid project asset"));
        }
    }
    let graph = Graph::try_from(dto.graph.clone())?;
    for node in &dto.graph.graph {
        let source = match &node.params {
            crate::MaterialNodeParams::LegacySource(source)
            | crate::MaterialNodeParams::ColorSource(source)
            | crate::MaterialNodeParams::HeightSource(source)
            | crate::MaterialNodeParams::NormalSource { source, .. } => source,
            _ => continue,
        };
        if !dto.assets.iter().any(|asset| {
            asset.decoded_hash == source.content_hash
                && asset.interpretation == source.interpretation
        }) {
            return Err(CoreError::InvalidMaterial("missing matching source asset"));
        }
    }
    dto.graph.graph.sort_by_key(|node| node.id);
    dto.assets.sort_by_key(|asset| asset.id);
    validate_export_for_graph(&dto.export, &graph)?;
    let v = &dto.view;
    if !v.light.intensity.is_finite()
        || v.light.intensity < 0.0
        || !v.light.direction.iter().all(|x| x.is_finite())
        || v.light.direction.iter().all(|x| *x == 0.0)
        || !(1..=3).contains(&v.tiles.count)
        || !v.pom.scale.is_finite()
        || v.pom.scale < 0.0
        || !v.camera.yaw.is_finite()
        || !v.camera.pitch.is_finite()
        || !v.camera.distance.is_finite()
        || v.camera.distance <= 0.0
        || !v.camera.target.iter().all(|x| x.is_finite())
        || !v.split_fraction.is_finite()
        || !(0.0..=1.0).contains(&v.split_fraction)
    {
        return Err(CoreError::InvalidMaterial("invalid saved view"));
    }
    if let PreviewMesh::Custom(locator) = &v.mesh
        && !valid_path(&locator.0)
    {
        return Err(CoreError::InvalidMaterial("invalid mesh reference"));
    }
    if let Some(locator) = &v.light.environment
        && !valid_path(&locator.0)
    {
        return Err(CoreError::InvalidMaterial("invalid environment reference"));
    }
    Ok(ValidatedProject { dto, graph })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectOpenToken(pub u64);
