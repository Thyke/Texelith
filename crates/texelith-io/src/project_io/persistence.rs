use super::{
    CanonicalDocument, MAX_JSON_BYTES, ProjectIoError, canonical_json, parse_json, read_project,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};
use texelith_core::ProjectFile;

/// The historical checkpoint name is retained for the project IO contract. A
/// backup is installed without moving away the destination's only directory entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePoint {
    AfterTempFlush,
    AfterBackupRename,
    AfterPublishRename,
    BeforeRollbackRestore,
    BeforeBackupCleanup,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryPaths {
    pub destination: PathBuf,
    pub temporary: PathBuf,
    pub backup: PathBuf,
    pub backup_temporary: PathBuf,
}

#[derive(Debug)]
pub enum SaveOutcome {
    Committed,
    /// The document is committed, but a backup remains. Resolve recovery before
    /// another save. Explicitly retaining this outcome avoids a false "failed save".
    CommittedWithCleanup {
        paths: RecoveryPaths,
        error: io::Error,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    Unchanged,
    RestoredBackup,
    PromotedTemp,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub fn recovery_paths(path: &Path) -> Result<RecoveryPaths, ProjectIoError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("invalid project filename"))?;
    crate::export_transactions::expand_name(
        "{name}",
        &crate::export_transactions::NameContext {
            name,
            map: "",
            preset: "",
            index: 0,
        },
    )?;
    Ok(RecoveryPaths {
        destination: parent.join(name),
        temporary: parent.join(format!("{name}.texelith-tmp")),
        backup: parent.join(format!("{name}.texelith-bak")),
        backup_temporary: parent.join(format!("{name}.texelith-backup-tmp")),
    })
}

fn regular(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => Err(invalid(
            "persistence entry is not a regular file (symlinks are refused)",
        )),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn remove(path: &Path) -> io::Result<()> {
    if regular(path)? {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, ProjectIoError> {
    if !regular(path)? {
        return Err(io::Error::from(io::ErrorKind::NotFound).into());
    }
    let mut file = File::open(path)?;
    if file.metadata()?.len() > MAX_JSON_BYTES as u64 {
        return Err(ProjectIoError::Invalid("JSON exceeds 8 MiB"));
    }
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        if n > MAX_JSON_BYTES.saturating_sub(bytes.len()) {
            return Err(ProjectIoError::Invalid("JSON exceeds 8 MiB"));
        }
        bytes.reserve_exact(n);
        bytes.extend_from_slice(&chunk[..n]);
    }
    Ok(bytes)
}

/// Read a preset or manifest using the same bounded reader as project recovery.
pub fn read_json(path: &Path) -> Result<serde_json::Value, ProjectIoError> {
    parse_json(&read_bounded(path)?)
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> io::Result<()> {
    File::open(path.parent().unwrap())?.sync_all()
}
#[cfg(not(unix))]
fn sync_parent(_: &Path) -> io::Result<()> {
    // Windows uses sync_publication's final-file FlushFileBuffers barrier.
    Ok(())
}

fn sync_publication(path: &Path) -> io::Result<()> {
    // ReplaceFileW has no supported write-through flag. Flush the final file
    // handle AFTER replacement, retaining the backup if opening/flushing fails.
    // File::sync_all calls FlushFileBuffers on Windows; it needs write access.
    #[cfg(windows)]
    OpenOptions::new().write(true).open(path)?.sync_all()?;
    sync_parent(path)
}

#[cfg(not(windows))]
fn replace(source: &Path, destination: &Path) -> io::Result<()> {
    // Same-directory rename is the atomic replacement primitive on POSIX.
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ReplaceFileW(
            replaced: *const u16,
            replacement: *const u16,
            backup: *const u16,
            flags: u32,
            exclude: *mut std::ffi::c_void,
            reserved: *mut std::ffi::c_void,
        ) -> i32;
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }
    let wide = |path: &Path| -> io::Result<Vec<u16>> {
        let mut out: Vec<u16> = path.as_os_str().encode_wide().collect();
        if out.contains(&0) {
            return Err(invalid("NUL in persistence path"));
        }
        out.push(0);
        Ok(out)
    };
    let source = wide(source)?;
    let dest = wide(destination)?;
    let exists = regular(destination)?;
    // SAFETY: NUL-terminated path buffers outlive the call. Optional pointers are
    // null; flags are documented constants. No Rust memory is owned by Windows.
    let result = unsafe {
        if exists {
            ReplaceFileW(
                dest.as_ptr(),
                source.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            // MOVEFILE_WRITE_THROUGH, without REPLACE_EXISTING: a newly created
            // destination is never overwritten during first publication.
            MoveFileExW(source.as_ptr(), dest.as_ptr(), 8)
        }
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn stage_copy(source: &Path, destination: &Path) -> io::Result<()> {
    // A separate staged copy makes even non-hardlink filesystems safe. A crash
    // during the copy never exposes a partial backup as a recovery candidate.
    let source = File::open(source)?;
    if source.metadata()?.len() > MAX_JSON_BYTES as u64 {
        return Err(invalid("previous JSON exceeds 8 MiB"));
    }
    let mut staged = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let copied = io::copy(&mut source.take(MAX_JSON_BYTES as u64 + 1), &mut staged)?;
    if copied > MAX_JSON_BYTES as u64 {
        return Err(invalid("previous JSON exceeds 8 MiB"));
    }
    staged.flush()?;
    staged.sync_all()?;
    Ok(())
}

fn backup(paths: &RecoveryPaths) -> io::Result<()> {
    stage_copy(&paths.destination, &paths.backup_temporary)?;
    fs::rename(&paths.backup_temporary, &paths.backup)?;
    sync_publication(&paths.backup)
}

fn publish_copy(source: &Path, paths: &RecoveryPaths) -> io::Result<()> {
    // Preserve the recovery candidate until the replacement passes its durable
    // publication barrier. The backup staging entry is never a valid candidate.
    remove(&paths.backup_temporary)?;
    stage_copy(source, &paths.backup_temporary)?;
    replace(&paths.backup_temporary, &paths.destination)?;
    sync_publication(&paths.destination)
}

fn rollback(
    paths: &RecoveryPaths,
    publication: io::Error,
    had_destination: bool,
    published: bool,
    checkpoint: &mut impl FnMut(FailurePoint) -> io::Result<()>,
) -> ProjectIoError {
    let restored = (|| {
        if had_destination && regular(&paths.backup)? {
            checkpoint(FailurePoint::BeforeRollbackRestore)?;
            publish_copy(&paths.backup, paths)?;
            remove(&paths.backup)?;
            sync_parent(&paths.destination)?;
        } else if published {
            // Retain the new file as a recoverable temp before clearing a failed
            // first save; never delete the sole complete copy on this path.
            fs::rename(&paths.destination, &paths.temporary)?;
            sync_parent(&paths.destination)?;
            return Ok(());
        }
        remove(&paths.temporary)?;
        remove(&paths.backup_temporary)?;
        Ok(())
    })();
    match restored {
        Ok(()) => ProjectIoError::Io(publication),
        Err(rollback) => ProjectIoError::Rollback {
            publication,
            rollback,
            paths: Box::new(paths.clone()),
        },
    }
}

pub fn save_project(path: &Path, project: &ProjectFile) -> Result<SaveOutcome, ProjectIoError> {
    save_project_with(path, project, |_| Ok(()))
}
pub fn save_project_with(
    path: &Path,
    project: &ProjectFile,
    checkpoint: impl FnMut(FailurePoint) -> io::Result<()>,
) -> Result<SaveOutcome, ProjectIoError> {
    save_json_with(path, project, checkpoint)
}
pub fn save_json(
    path: &Path,
    document: &(impl CanonicalDocument + ?Sized),
) -> Result<SaveOutcome, ProjectIoError> {
    save_json_with(path, document, |_| Ok(()))
}

/// Single-writer helper, also used by presets and job manifests. Sibling recovery
/// entries prevent another save until explicitly resolved. Run save/recover on
/// the same serialized IO worker; these are not cross-process lock services.
/// Windows flushes the final file handle after replacement and before removing
/// the backup. If that durability barrier fails, the save returns an IO error
/// and retains the backup; recovery must resolve the uncertain publication.
pub fn save_json_with(
    path: &Path,
    document: &(impl CanonicalDocument + ?Sized),
    mut checkpoint: impl FnMut(FailurePoint) -> io::Result<()>,
) -> Result<SaveOutcome, ProjectIoError> {
    let bytes = canonical_json(document)?;
    let paths = recovery_paths(path)?;
    let had_destination = regular(&paths.destination)?;
    let parent = paths.destination.parent().unwrap();
    let siblings = [&paths.temporary, &paths.backup, &paths.backup_temporary];
    let names: Vec<_> = siblings
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    crate::export_transactions::preflight(parent, &names)?;
    if !had_destination {
        crate::export_transactions::preflight(
            parent,
            &[paths.destination.file_name().unwrap().to_str().unwrap()],
        )?;
    }
    let mut temporary = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&paths.temporary)?;
    let prepared = temporary
        .write_all(&bytes)
        .and_then(|()| temporary.flush())
        .and_then(|()| temporary.sync_all());
    drop(temporary);
    if let Err(e) = prepared.and_then(|()| checkpoint(FailurePoint::AfterTempFlush)) {
        // No old file was touched. A complete temp is retained if cleanup fails.
        let _ = remove(&paths.temporary);
        return Err(e.into());
    }
    if had_destination
        && let Err(e) = backup(&paths).and_then(|()| checkpoint(FailurePoint::AfterBackupRename))
    {
        return Err(rollback(&paths, e, true, false, &mut checkpoint));
    }
    if let Err(e) = replace(&paths.temporary, &paths.destination) {
        return Err(rollback(&paths, e, had_destination, false, &mut checkpoint));
    }
    if let Err(e) = checkpoint(FailurePoint::AfterPublishRename) {
        return Err(rollback(&paths, e, had_destination, true, &mut checkpoint));
    }
    // Do not roll back a failed durability barrier by consuming the backup.
    // Keep it available alongside the published file for explicit recovery.
    sync_publication(&paths.destination)?;
    if had_destination
        && let Err(error) = checkpoint(FailurePoint::BeforeBackupCleanup)
            .and_then(|()| remove(&paths.backup))
            .and_then(|()| sync_parent(&paths.destination))
    {
        return Ok(SaveOutcome::CommittedWithCleanup { paths, error });
    }
    Ok(SaveOutcome::Committed)
}

pub fn recover_project(path: &Path) -> Result<RecoveryOutcome, ProjectIoError> {
    recover_with(path, |bytes| read_project(bytes).map(|_| ()))
}
pub fn recover_json(path: &Path) -> Result<RecoveryOutcome, ProjectIoError> {
    recover_with(path, |bytes| parse_json(bytes).map(|_| ()))
}

/// Recovery validates candidates with the consumer's schema before touching the
/// destination. A valid main file is the newest publication and wins over a
/// leftover backup, regardless of copy timestamps. Only a missing/invalid main
/// falls back to a valid backup, then a complete temp. Unsupported schemas and
/// unreadable candidates fail without fallback. Failed validation retains all
/// entries; publication failure retains the selected candidate. Successful
/// recovery flushes the publication before removing siblings and is idempotent.
pub fn recover_with(
    path: &Path,
    validate: impl Fn(&[u8]) -> Result<(), ProjectIoError>,
) -> Result<RecoveryOutcome, ProjectIoError> {
    let paths = recovery_paths(path)?;
    for entry in [
        &paths.destination,
        &paths.temporary,
        &paths.backup,
        &paths.backup_temporary,
    ] {
        regular(entry)?;
    }
    let mut failure = None;
    let mut selected = None;
    for (candidate, outcome) in [
        (&paths.destination, RecoveryOutcome::Unchanged),
        (&paths.backup, RecoveryOutcome::RestoredBackup),
        (&paths.temporary, RecoveryOutcome::PromotedTemp),
    ] {
        if !regular(candidate)? {
            continue;
        }
        match read_bounded(candidate).and_then(|bytes| validate(&bytes)) {
            Ok(()) => {
                selected = Some((candidate, outcome));
                break;
            }
            // An unreadable main is not evidence of an invalid document. Do not
            // overwrite it just because a permission or device error hid it.
            Err(error @ (ProjectIoError::Io(_) | ProjectIoError::UnsupportedSchema { .. }))
            | Err(error @ ProjectIoError::Core(texelith_core::CoreError::UnknownSchema(_))) => {
                return Err(error);
            }
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        }
    }
    let Some((candidate, outcome)) = selected else {
        return Err(failure.unwrap_or_else(|| io::Error::from(io::ErrorKind::NotFound).into()));
    };
    if outcome != RecoveryOutcome::Unchanged {
        publish_copy(candidate, &paths)?;
    } else {
        // Even a valid main may be an interrupted, not-yet-flushed publication.
        sync_publication(&paths.destination)?;
    }
    remove(&paths.backup)?;
    remove(&paths.temporary)?;
    remove(&paths.backup_temporary)?;
    sync_parent(&paths.destination)?;
    Ok(outcome)
}
