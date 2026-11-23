use super::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path};
use texelith_io::{export_transactions as tx, project_io as json};

#[cfg(test)]
#[path = "publication_tests.rs"]
mod publication_tests;

#[derive(Clone, Debug)]
pub enum InputSelection {
    Folder { path: PathBuf, recursive: bool },
    Files(Vec<PathBuf>),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSource {
    pub path: PathBuf,
    pub name: String,
}
fn supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["png", "jpg", "jpeg", "exr"].contains(&e.to_ascii_lowercase().as_str()))
}
fn portable(path: &Path) -> Result<String, String> {
    Ok(path
        .to_str()
        .ok_or("Batch paths must be Unicode")?
        .replace('\\', "/"))
}
pub fn enumerate(selection: &InputSelection) -> Result<Vec<BatchSource>, String> {
    fn scan(
        path: &Path,
        root: &Path,
        recursive: bool,
        out: &mut Vec<BatchSource>,
        visited: &mut usize,
        depth: usize,
    ) -> Result<(), String> {
        if depth > 64 {
            return Err("Batch directory depth limit exceeded".into());
        }
        for entry in std::fs::read_dir(path).map_err(err)? {
            *visited += 1;
            if *visited > 20_000 {
                return Err("Batch directory entry limit exceeded".into());
            }
            let entry = entry.map_err(err)?;
            let kind = entry.file_type().map_err(err)?;
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() && recursive {
                scan(&entry.path(), root, true, out, visited, depth + 1)?;
            } else if kind.is_file() && supported(&entry.path()) {
                out.push(BatchSource {
                    path: std::fs::canonicalize(entry.path()).map_err(err)?,
                    name: portable(entry.path().strip_prefix(root).map_err(err)?)?,
                });
                if out.len() > 10_000 {
                    return Err("Batch file limit exceeded".into());
                }
            }
        }
        Ok(())
    }
    let mut sources = Vec::new();
    match selection {
        InputSelection::Folder { path, recursive } => {
            if std::fs::symlink_metadata(path)
                .map_err(err)?
                .file_type()
                .is_symlink()
            {
                return Err("Batch root cannot be a symlink".into());
            }
            scan(path, path, *recursive, &mut sources, &mut 0, 0)?;
        }
        InputSelection::Files(paths) => {
            if paths.len() > 10_000 {
                return Err("Batch file limit exceeded".into());
            }
            for path in paths {
                let metadata = std::fs::symlink_metadata(path).map_err(err)?;
                if !metadata.is_file() || metadata.file_type().is_symlink() || !supported(path) {
                    continue;
                }
                let path = std::fs::canonicalize(path).map_err(err)?;
                sources.push(BatchSource {
                    name: portable(Path::new(path.file_name().ok_or("Missing filename")?))?,
                    path,
                });
            }
        }
    }
    // Tie-break equal relative names by canonical path, independently of selection order.
    sources.sort_by(|a, b| (&a.name, &a.path).cmp(&(&b.name, &b.path)));
    let mut seen = BTreeSet::new();
    sources.retain(|s| seen.insert(s.path.clone()));
    if sources.is_empty() {
        return Err("No supported batch images".into());
    }
    Ok(sources)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeItem {
    pub index: usize,
    pub source: BatchSource,
    pub destination: PathBuf,
    pub material_name: String,
    pub decoded_hash: Option<[u8; 32]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub recipe_digest: String,
    pub checkpoint: BatchCheckpoint,
    pub native: Vec<NativeItem>,
}
pub struct Prepared {
    pub job: BatchJob,
    pub native: Vec<NativeItem>,
    pub manifest: PathBuf,
}
fn digest(checkpoint: &BatchCheckpoint) -> Result<String, String> {
    let text = json::canonical_json(&serde_json::to_value(&checkpoint.recipe).map_err(err)?)
        .map_err(err)?;
    // Reuse core's BLAKE3 boundary, as TypedSourcePixels does, without a dependency.
    let mut bytes = b"texelith.batch-recipe.v1\0".to_vec();
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&text);
    let width = bytes.len().div_ceil(4).min(4096) as u32;
    let height = bytes.len().div_ceil(width as usize * 4) as u32;
    bytes.resize(width as usize * height as usize * 4, 0);
    let hash = SourceDescriptor::from_rgba8(
        &bytes,
        Extent::new(width, height).map_err(err)?,
        InputInterpretation::SrgbPhoto,
    )
    .map_err(err)?
    .content_hash;
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}
pub fn save_manifest(path: &Path, job: &BatchJob, native: &[NativeItem]) -> Result<(), String> {
    let existing = parse_manifest(json::read_json(path).map_err(err)?)?;
    if existing.checkpoint.job != job.id() || existing.checkpoint.recipe != job.checkpoint().recipe
    {
        return Err("Batch manifest belongs to another job; choose explicit recovery".into());
    }
    write_manifest(path, job, native)
}
fn manifest_value(job: &BatchJob, native: &[NativeItem]) -> Result<serde_json::Value, String> {
    let checkpoint = job.checkpoint();
    let manifest = Manifest {
        schema_version: 1,
        recipe_digest: digest(&checkpoint)?,
        checkpoint,
        native: native.to_vec(),
    };
    serde_json::to_value(manifest).map_err(err)
}
fn write_manifest(path: &Path, job: &BatchJob, native: &[NativeItem]) -> Result<(), String> {
    let value = manifest_value(job, native)?;
    match json::save_json(path, &value).map_err(err)? {
        json::SaveOutcome::Committed => Ok(()),
        json::SaveOutcome::CommittedWithCleanup { error, .. } => {
            Err(format!("Manifest committed; recovery required: {error}"))
        }
    }
}
/// T-077 stages and flushes the complete record. Installing the staged file admits
/// exactly one new name without replacing a destination created after preflight.
pub(super) fn create_manifest(
    path: &Path,
    job: &BatchJob,
    native: &[NativeItem],
) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(err)?;
    let paths = json::recovery_paths(path).map_err(err)?;
    let entries = [
        &paths.destination,
        &paths.temporary,
        &paths.backup,
        &paths.backup_temporary,
    ];
    let names: Vec<_> = entries
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    tx::preflight(parent, &names).map_err(err)?;
    let bytes = json::canonical_json(&manifest_value(job, native)?).map_err(err)?;
    let staging = parent.join(format!(
        ".batch-manifest-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    tx::publish_set(
        &staging,
        &[tx::ExportFile::new("manifest.json")],
        tx::PublishOptions {
            max_file_bytes: json::MAX_JSON_BYTES as u64,
            ..Default::default()
        },
        |_, writer| writer.write_all(&bytes),
        |_| Ok(()),
    )
    .map_err(err)?;
    let result = install_manifest(&staging.join("manifest.json"), path).map_err(err);
    // Only these two entries belong to this staging transaction.
    let cleanup = (|| -> std::io::Result<()> {
        match std::fs::remove_file(staging.join("manifest.json")) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        std::fs::remove_file(staging.join(".texelith-transaction"))?;
        std::fs::remove_dir(&staging)
    })();
    result?;
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(err)?;
    cleanup.map_err(err)
}
#[cfg(not(windows))]
fn install_manifest(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::hard_link(source, destination)
}
#[cfg(windows)]
fn install_manifest(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(source: *const u16, destination: *const u16, flags: u32) -> i32;
    }
    let wide = |path: &Path| -> std::io::Result<Vec<u16>> {
        let mut text: Vec<_> = path.as_os_str().encode_wide().collect();
        if text.contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "NUL in manifest path",
            ));
        }
        text.push(0);
        Ok(text)
    };
    let source = wide(source)?;
    let destination = wide(destination)?;
    // SAFETY: buffers are NUL terminated and live through the call. WRITE_THROUGH
    // without REPLACE_EXISTING rejects any destination appearing after preflight.
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 8) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
pub(super) fn create_derived_manifest(
    base: &Path,
    job: &BatchJob,
    native: &[NativeItem],
) -> Result<PathBuf, String> {
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid manifest name")?;
    let path = base.with_file_name(format!("{stem}.job-{}.json", job.id()));
    create_manifest(&path, job, native)?;
    Ok(path)
}
fn parse_manifest(value: serde_json::Value) -> Result<Manifest, String> {
    let manifest: Manifest = serde_json::from_value(value).map_err(err)?;
    if manifest.schema_version != 1 || digest(&manifest.checkpoint)? != manifest.recipe_digest {
        return Err("Invalid batch manifest version or recipe digest".into());
    }
    let _ = BatchJob::restore(manifest.checkpoint.clone()).map_err(err_debug)?;
    let mut seen = BTreeSet::new();
    for item in &manifest.checkpoint.items {
        let native = manifest
            .native
            .iter()
            .find(|n| n.index == item.index)
            .ok_or("Missing native batch item")?;
        if !seen.insert(item.index) || portable(&native.source.path)? != item.input.source.0 {
            return Err("Manifest source identity mismatch".into());
        }
        let destinations = destinations(
            &manifest.checkpoint.recipe.export,
            &native.destination,
            &native.material_name,
        )?;
        if destinations != item.input.destinations {
            return Err("Manifest destination identity mismatch".into());
        }
    }
    if seen.len() != manifest.native.len() {
        return Err("Unexpected native batch entries".into());
    }
    Ok(manifest)
}
pub fn recover_manifest(path: &Path) -> Result<Prepared, String> {
    json::recover_with(path, |bytes| {
        let value = json::parse_json(bytes)?;
        for (section, supported) in [
            (&value, 1),
            (&value["checkpoint"], 1),
            (&value["checkpoint"]["recipe"]["graph"], 2),
        ] {
            if let Some(found) = section["schema_version"].as_u64()
                && found > u64::from(supported)
            {
                return Err(json::ProjectIoError::UnsupportedSchema {
                    found,
                    supported_min: 1,
                    supported_max: supported,
                });
            }
        }
        parse_manifest(value).map(|_| ()).map_err(|_| {
            json::ProjectIoError::Invalid("Invalid batch manifest schema, mappings or digest")
        })
    })
    .map_err(err)?;
    let mut manifest = parse_manifest(json::read_json(path).map_err(err)?)?;
    let mut job = BatchJob::restore(manifest.checkpoint.clone()).map_err(err_debug)?;
    for item in job.items() {
        let native = manifest
            .native
            .iter()
            .find(|n| n.index == item.index)
            .ok_or("Missing native item")?;
        if let Some(digest) = native.decoded_hash {
            let upload = decode(&native.source.path, job.recipe())?;
            if upload.descriptor.content_hash != digest
                || (item.source_digest != [0; 32] && item.source_digest != digest)
            {
                return Err(format!("Source changed: {}", native.source.name));
            }
        }
        if item.state == BatchState::Committed || native.destination.exists() {
            for destination in &item.destinations {
                if !Path::new(&destination.0).is_file() {
                    return Err("Committed batch output is missing".into());
                }
            }
            let assignment =
                json::read_json(&native.destination.join("assignment.json")).map_err(err)?;
            let sources: Vec<SourceDescriptor> =
                serde_json::from_value(assignment["sources"].clone()).map_err(err)?;
            if sources.len() != 1 || Some(sources[0].content_hash) != native.decoded_hash {
                return Err("Committed batch source identity mismatch".into());
            }
            let graph = instantiate_recipe(&job.recipe().graph, sources[0].clone()).map_err(err)?;
            let mut plan = job.export_settings().clone();
            plan.source_extent = sources[0].extent;
            let capture = ExportCapture::new(
                &EvaluationSnapshot {
                    generation: 0,
                    graph,
                },
                0,
                &plan,
            )
            .map_err(err)?;
            if serde_json::to_value(capture.graph_hash).map_err(err)? != assignment["graph_hash"]
                || serde_json::to_value(&plan).map_err(err)? != assignment["plan"]
            {
                return Err("Committed batch recipe identity mismatch".into());
            }
            // Publication can win a crash before the next manifest flush. Validate
            // the complete set, then retain it instead of recomputing on resume.
            let saved = manifest
                .checkpoint
                .items
                .iter_mut()
                .find(|i| i.index == item.index)
                .ok_or("Missing checkpoint item")?;
            saved.state = BatchState::Committed;
            saved.error = None;
        }
    }
    job = BatchJob::restore(manifest.checkpoint).map_err(err_debug)?;
    Ok(Prepared {
        job,
        native: manifest.native,
        manifest: path.to_owned(),
    })
}
pub(super) fn destinations(
    plan: &ExportPlan,
    destination: &Path,
    name: &str,
) -> Result<Vec<OpaqueLocator>, String> {
    let mut outputs = Vec::new();
    for (index, output) in plan.outputs.iter().enumerate() {
        let stem = tx::expand_name(
            &plan.naming,
            &tx::NameContext {
                name,
                map: &output.name,
                preset: &plan.profile.id,
                index: index as u32,
            },
        )
        .map_err(err)?;
        let extension = match output.format {
            ExportFormat::Png8 | ExportFormat::Png16 => "png",
            ExportFormat::Exr32 => "exr",
            ExportFormat::DdsBc5 => "dds",
        };
        outputs.push(OpaqueLocator(portable(
            &destination.join(format!("{stem}.{extension}")),
        )?));
    }
    outputs.push(OpaqueLocator(portable(
        &destination.join("assignment.json"),
    )?));
    Ok(outputs)
}
pub(super) fn prepare(submission: Submission) -> Result<Prepared, String> {
    let sources = enumerate(&submission.selection)?;
    std::fs::create_dir_all(&submission.destination).map_err(err)?;
    let root = std::fs::canonicalize(&submission.destination).map_err(err)?;
    let mut native = Vec::new();
    let mut inputs = Vec::new();
    let mut directories = BTreeSet::new();
    // Preflight the complete output set before decoding or submitting any graph.
    for (index, source) in sources.into_iter().enumerate() {
        let relative = Path::new(&source.name).with_extension("");
        let destination = root.join(&relative);
        let material_name = relative
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("Invalid source name")?
            .to_owned();
        if !directories.insert(portable(&destination)?.to_lowercase())
            || destination.try_exists().map_err(err)?
        {
            return Err(format!("Batch output collision: {}", destination.display()));
        }
        inputs.push(BatchInput {
            source: OpaqueLocator(portable(&source.path)?),
            source_digest: [0; 32],
            destinations: destinations(&submission.plan, &destination, &material_name)?,
        });
        native.push(NativeItem {
            index,
            source,
            destination,
            material_name,
            decoded_hash: None,
        });
    }
    let mut job = BatchJob::new(
        submission.id,
        submission.capture,
        submission.plan,
        inputs.clone(),
        submission.continue_on_error,
    )
    .map_err(err_debug)?;
    for (input, native) in inputs.iter_mut().zip(&mut native) {
        if let Ok(upload) = decode(&native.source.path, job.recipe()) {
            input.source_digest = upload.descriptor.content_hash;
            native.decoded_hash = Some(upload.descriptor.content_hash);
        }
    }
    for item in &native {
        std::fs::create_dir_all(item.destination.parent().ok_or("Missing output parent")?)
            .map_err(err)?;
    }
    job = BatchJob::new(
        submission.id,
        job.recipe().clone(),
        job.export_settings().clone(),
        inputs,
        submission.continue_on_error,
    )
    .map_err(err_debug)?;
    let parent = submission
        .manifest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(err)?;
    create_manifest(&submission.manifest, &job, &native)?;
    Ok(Prepared {
        job,
        native,
        manifest: submission.manifest,
    })
}

pub(super) fn decode(
    path: &Path,
    capture: &ExportCapture,
) -> Result<texelith_gpu::TypedSourceUpload, String> {
    let _cpu = memory::reserve_background();
    let kind = match capture.sources[0].interpretation {
        InputInterpretation::SrgbPhoto => crate::material_ui::ImportKind::Photo,
        InputInterpretation::LinearHeight => crate::material_ui::ImportKind::Height,
        InputInterpretation::LinearNormal => crate::material_ui::ImportKind::Normal,
    };
    let convention = capture
        .graph
        .material_nodes()
        .iter()
        .find_map(|node| {
            if let MaterialNodeParams::NormalSource { convention, .. } = node.params {
                Some(convention)
            } else {
                None
            }
        })
        .unwrap_or(NormalConvention::OpenGl);
    crate::material_ui::presenter::decode(path, kind, convention)
}
