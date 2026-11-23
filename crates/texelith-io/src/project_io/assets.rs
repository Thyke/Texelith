use super::ProjectIoError;
use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};
use texelith_core::{
    AssetId, Extent, InputInterpretation, PreviewMesh, ProjectFile, SourceDescriptor,
    validate_project,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetLocation {
    /// Absolute requested location, before symlink resolution.
    pub requested: PathBuf,
    pub canonical: PathBuf,
    pub symlink_resolved: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetResolution {
    Matched {
        id: AssetId,
        location: AssetLocation,
        source: SourceDescriptor,
    },
    Missing {
        id: AssetId,
        requested: PathBuf,
    },
    /// The caller must obtain an explicit user choice before using this source.
    Changed {
        id: AssetId,
        location: AssetLocation,
        expected: [u8; 32],
        source: SourceDescriptor,
    },
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// Default probe uses the existing bounded RGBA8 import/hash boundary. Numeric
/// PNG16/EXR sources can supply their typed descriptor via `resolve_assets_with`.
pub fn resolve_assets(
    project: &ProjectFile,
    path: &Path,
) -> Result<Vec<AssetResolution>, ProjectIoError> {
    resolve_assets_with(project, path, |path, interpretation| {
        let intent = match interpretation {
            InputInterpretation::SrgbPhoto => crate::ColorIntent::PhotoSrgb,
            InputInterpretation::LinearHeight | InputInterpretation::LinearNormal => {
                crate::ColorIntent::LinearHeight
            }
        };
        let image = crate::decode_path(path, intent)?;
        Ok(SourceDescriptor::from_rgba8(
            &image.rgba8,
            Extent::new(image.width, image.height)?,
            interpretation,
        )?)
    })
}

/// Decodes sequentially so no full decoded image sets accumulate here. The probe
/// must honour the shared CPU reservation and return the actual decoded identity.
pub fn resolve_assets_with(
    project: &ProjectFile,
    path: &Path,
    mut probe: impl FnMut(&Path, InputInterpretation) -> Result<SourceDescriptor, ProjectIoError>,
) -> Result<Vec<AssetResolution>, ProjectIoError> {
    let validated = validate_project(project.clone())?;
    let base = fs::canonicalize(parent(path))?;
    let mut resolutions = Vec::with_capacity(project.assets.len());
    for asset in &validated.document().assets {
        let requested = base.join(&asset.relative_path);
        let canonical = match fs::canonicalize(&requested) {
            Ok(path) => path,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                resolutions.push(AssetResolution::Missing {
                    id: asset.id,
                    requested,
                });
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let symlink_resolved = resolved_link(&base, &asset.relative_path)?;
        let source = probe(&canonical, asset.interpretation)?;
        if source.interpretation != asset.interpretation {
            return Err(ProjectIoError::Invalid(
                "asset probe changed interpretation",
            ));
        }
        source.extent.validate()?;
        let matches_extent = validated.document().graph.graph.iter().all(|node| {
            let expected = match &node.params {
                texelith_core::MaterialNodeParams::LegacySource(s)
                | texelith_core::MaterialNodeParams::ColorSource(s)
                | texelith_core::MaterialNodeParams::HeightSource(s)
                | texelith_core::MaterialNodeParams::NormalSource { source: s, .. } => s,
                _ => return true,
            };
            expected.content_hash != asset.decoded_hash
                || expected.interpretation != asset.interpretation
                || expected.extent == source.extent
        });
        let location = AssetLocation {
            requested,
            canonical,
            symlink_resolved,
        };
        resolutions.push(
            if source.content_hash == asset.decoded_hash && matches_extent {
                AssetResolution::Matched {
                    id: asset.id,
                    location,
                    source,
                }
            } else {
                AssetResolution::Changed {
                    id: asset.id,
                    location,
                    expected: asset.decoded_hash,
                    source,
                }
            },
        );
    }
    Ok(resolutions)
}

fn resolved_link(base: &Path, reference: &str) -> io::Result<bool> {
    let mut prefix = base.to_path_buf();
    let mut found = false;
    for component in Path::new(reference).components() {
        prefix.push(component);
        found |= fs::symlink_metadata(&prefix)?.file_type().is_symlink();
    }
    Ok(found)
}

/// Resolve existing prefixes before processing `..`, preserving symlink semantics
/// even when the final linked asset is missing.
fn target(base: &Path, reference: &str) -> io::Result<PathBuf> {
    let mut result = base.to_path_buf();
    for component in Path::new(reference).components() {
        match component {
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(name) => {
                result.push(name);
                match fs::canonicalize(&result) {
                    Ok(canonical) => result = canonical,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "nonrelative asset reference",
                ));
            }
        }
    }
    Ok(result)
}

fn relative(base: &Path, target: &Path) -> Result<String, ProjectIoError> {
    let from: Vec<_> = base.components().collect();
    let to: Vec<_> = target.components().collect();
    if from.first() != to.first() {
        return Err(ProjectIoError::Invalid(
            "Save As cannot express linked asset across filesystem roots",
        ));
    }
    let shared = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let mut parts = vec![".."; from.len() - shared];
    for part in &to[shared..] {
        let Component::Normal(name) = part else {
            return Err(ProjectIoError::Invalid("nonportable asset location"));
        };
        parts.push(
            name.to_str()
                .ok_or(ProjectIoError::Invalid("non UTF-8 asset location"))?,
        );
    }
    Ok(parts.join("/"))
}

/// Returns a new snapshot; neither success nor failure mutates the live document.
/// Save the result first and update the frontend's path only on committed success.
pub fn rebase_project(
    project: &ProjectFile,
    old: &Path,
    new: &Path,
) -> Result<ProjectFile, ProjectIoError> {
    let mut dto = validate_project(project.clone())?.into_document();
    let old_base = fs::canonicalize(parent(old))?;
    let new_base = fs::canonicalize(parent(new))?;
    let rebase = |reference: &str| relative(&new_base, &target(&old_base, reference)?);
    for asset in &mut dto.assets {
        asset.relative_path = rebase(&asset.relative_path)?;
    }
    if let PreviewMesh::Custom(locator) = &mut dto.view.mesh {
        locator.0 = rebase(&locator.0)?;
    }
    if let Some(locator) = &mut dto.view.light.environment {
        locator.0 = rebase(&locator.0)?;
    }
    Ok(validate_project(dto)?.into_document())
}
