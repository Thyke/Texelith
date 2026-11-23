//! Bounded project/preset/manifest JSON, explicit linked-asset checks and durable saves.
//!
//! Callers own schema validation, worker scheduling and the user's relink decision.
//! No helper installs a document into a live session or accepts changed pixels.

mod assets;
mod json;
mod persistence;

pub use assets::{
    AssetLocation, AssetResolution, rebase_project, resolve_assets, resolve_assets_with,
};
pub use json::{CanonicalDocument, MAX_JSON_BYTES, MAX_JSON_DEPTH, canonical_json, parse_json};
pub use persistence::{
    FailurePoint, RecoveryOutcome, RecoveryPaths, SaveOutcome, read_json, recover_json,
    recover_project, recover_with, recovery_paths, save_json, save_json_with, save_project,
    save_project_with,
};

use std::{fmt, io};
use texelith_core::{
    CoreError, MATERIAL_GRAPH_SCHEMA_VERSION, ProjectFile, ValidatedProject, validate_project,
};

#[derive(Debug)]
pub enum ProjectIoError {
    Io(io::Error),
    Json(serde_json::Error),
    Core(CoreError),
    Image(crate::IoError),
    Invalid(&'static str),
    DuplicateKey(String),
    /// An unsupported version is not corruption and must never trigger fallback.
    UnsupportedSchema {
        found: u64,
        supported_min: u32,
        supported_max: u32,
    },
    /// Both copies are retained at `paths` for a later recovery attempt.
    Rollback {
        publication: io::Error,
        rollback: io::Error,
        paths: Box<RecoveryPaths>,
    },
}

impl fmt::Display for ProjectIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "project IO: {e}"),
            Self::Json(e) => write!(f, "project JSON: {e}"),
            Self::Core(e) => write!(f, "project validation: {e}"),
            Self::Image(e) => write!(f, "project asset: {e}"),
            Self::Invalid(e) => f.write_str(e),
            Self::DuplicateKey(key) => write!(f, "duplicate JSON key: {key:?}"),
            Self::UnsupportedSchema {
                found,
                supported_min,
                supported_max,
            } => write!(
                f,
                "unsupported project schema {found}; supported range {supported_min}..={supported_max}"
            ),
            Self::Rollback {
                publication,
                rollback,
                paths,
            } => write!(
                f,
                "save failed: {publication}; rollback failed: {rollback}; backup: {}",
                paths.backup.display()
            ),
        }
    }
}
impl std::error::Error for ProjectIoError {}
impl From<io::Error> for ProjectIoError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for ProjectIoError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
impl From<CoreError> for ProjectIoError {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}
impl From<crate::IoError> for ProjectIoError {
    fn from(e: crate::IoError) -> Self {
        Self::Image(e)
    }
}

/// Checks text limits and duplicate keys before deserializing/validating a new DTO.
pub fn read_project(bytes: &[u8]) -> Result<ValidatedProject, ProjectIoError> {
    let value = parse_json(bytes)?;
    probe_future_schemas(&value, 1, true)?;
    // Inspect the version before v1 deserialization: a newer schema can change
    // required fields and shapes, which must not be mistaken for corruption.
    if let Some(found) = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        && found != 1
    {
        return Err(ProjectIoError::UnsupportedSchema {
            found,
            supported_min: 1,
            supported_max: 1,
        });
    }
    let dto: ProjectFile = serde_json::from_value(value)?;
    Ok(validate_project(dto)?)
}

fn probe_future_schemas(
    value: &serde_json::Value,
    supported: u32,
    project_root: bool,
) -> Result<(), ProjectIoError> {
    match value {
        serde_json::Value::Object(fields) => {
            if let Some(found) = fields
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                && found > u64::from(supported)
            {
                return Err(ProjectIoError::UnsupportedSchema {
                    found,
                    supported_min: supported,
                    supported_max: supported,
                });
            }
            // Only the project's graph has an independent v2 schema. Inspect
            // all other sections (including unknown fields) for future v1
            // extensions, without accepting them during strict deserialization.
            for (key, child) in fields {
                let supported = if project_root && key == "graph" {
                    MATERIAL_GRAPH_SCHEMA_VERSION
                } else {
                    1
                };
                probe_future_schemas(child, supported, false)?;
            }
        }
        serde_json::Value::Array(elements) => {
            for child in elements {
                probe_future_schemas(child, 1, false)?;
            }
        }
        _ => {}
    }
    Ok(())
}
