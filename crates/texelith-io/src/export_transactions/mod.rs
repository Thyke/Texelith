use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

mod normalization;

static NEXT: AtomicU64 = AtomicU64::new(0);
const MANIFEST: &str = ".texelith-transaction";

pub struct NameContext<'a> {
    pub name: &'a str,
    pub map: &'a str,
    pub preset: &'a str,
    pub index: u32,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn validate_name(name: &str) -> io::Result<()> {
    if name.is_empty() {
        return Err(invalid("Name is required"));
    }
    if name.len() > 255 || name.encode_utf16().count() > 255 {
        return Err(invalid(
            "Name is too long (maximum 255 bytes or UTF-16 characters)",
        ));
    }
    if name.ends_with(['.', ' '])
        || name.contains("..")
        || name
            .chars()
            .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
    {
        return Err(invalid(
            "Name contains invalid characters, '..', or a trailing dot or space",
        ));
    }
    let base = name
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches([' ', '.'])
        .to_ascii_uppercase();
    let port = base
        .strip_prefix("COM")
        .or_else(|| base.strip_prefix("LPT"));
    if matches!(
        base.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
    ) || port.is_some_and(|n| {
        matches!(
            n,
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
        )
    }) {
        return Err(invalid("Name is reserved by Windows; choose another name"));
    }
    Ok(())
}

pub fn expand_name(template: &str, context: &NameContext<'_>) -> io::Result<String> {
    let mut out = String::new();
    let mut rest = template;
    while !rest.is_empty() {
        let Some(pos) = rest.find(['{', '}']) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        if !rest.starts_with('{') {
            return Err(invalid("unmatched brace"));
        }
        let end = rest.find('}').ok_or_else(|| invalid("unmatched brace"))?;
        match &rest[1..end] {
            "name" => out.push_str(context.name),
            "map" => out.push_str(context.map),
            "preset" => out.push_str(context.preset),
            "index" => out.push_str(&format!("{:04}", context.index)),
            _ => {
                return Err(invalid(
                    "Naming template has an unknown token; use {name}, {map}, {preset} or {index}",
                ));
            }
        }
        rest = &rest[end + 1..];
    }
    validate_name(&out)?;
    Ok(out)
}

fn key(name: &str) -> String {
    normalization::nfc(&normalization::nfc(name.trim_end_matches([' ', '.'])).to_lowercase())
}

fn validate_names(names: &[&str]) -> io::Result<HashSet<String>> {
    let mut seen = HashSet::new();
    for name in names {
        validate_name(name)?;
        if !seen.insert(key(name)) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "duplicate name",
            ));
        }
    }
    Ok(seen)
}

/// Pure filename validation shared with naming previews; performs no IO.
pub fn validate_file_names(names: &[&str]) -> io::Result<()> {
    validate_names(names).map(|_| ())
}

pub fn preflight(root: &Path, names: &[&str]) -> io::Result<()> {
    let seen = validate_names(names)?;
    for entry in read_parent(root)? {
        let entry = entry?;
        if seen.contains(&key(&entry.file_name().to_string_lossy())) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination collision",
            ));
        }
    }
    Ok(())
}

fn read_parent(root: &Path) -> io::Result<impl Iterator<Item = io::Result<fs::DirEntry>>> {
    match fs::read_dir(root) {
        Ok(entries) => Ok(Some(entries).into_iter().flatten()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None.into_iter().flatten()),
        Err(error) => Err(error),
    }
}

fn preflight_destination(parent: &Path, name: &str, policy: CollisionPolicy) -> io::Result<bool> {
    let wanted = key(name);
    let mut exists = false;
    for entry in read_parent(parent)? {
        let entry = entry?;
        if key(&entry.file_name().to_string_lossy()) != wanted {
            continue;
        }
        if policy == CollisionPolicy::NoOverwrite || entry.file_name() != name {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination collision",
            ));
        }
        // Do not follow symlinks: replacement must own the actual directory entry.
        if !entry.file_type()?.is_dir() {
            return Err(invalid("replacement destination is not a directory"));
        }
        check_recovery_journal(parent, &entry.path())?;
        exists = true;
    }
    Ok(exists)
}

#[derive(Clone, Debug)]
pub struct ExportFile {
    name: String,
}
impl ExportFile {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CollisionPolicy {
    #[default]
    NoOverwrite,
    Replace,
}
#[derive(Clone, Copy, Debug)]
pub struct PublishOptions {
    pub collision: CollisionPolicy,
    pub max_file_bytes: u64,
}
impl Default for PublishOptions {
    fn default() -> Self {
        Self {
            collision: CollisionPolicy::NoOverwrite,
            max_file_bytes: 512 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePoint {
    AfterEncode(usize),
    AfterFlush(usize),
    AfterBackupRename,
    AfterPublishRename,
    BeforeRollbackRetire,
    BeforeRollbackRestore,
    /// One entry has actually been deleted; rollback is no longer possible.
    AfterBackupCleanup(usize),
}

/// Both variants mean the new, complete set has been committed. An `Err` means
/// publication failed; downcast `io::Error::get_ref()` to `RollbackFailure` to
/// inspect a failed rollback, or to `RecoveryRequired` when an earlier journal
/// must be resolved before replacement can start.
#[derive(Debug)]
pub enum PublishOutcome {
    Committed,
    CommittedWithCleanup {
        backup: PathBuf,
        recovery_manifest: PathBuf,
        error: io::Error,
    },
}

/// Replacement is refused until the caller resolves the journal's recovery or
/// pending backup cleanup. No staging, encoding or backup rename has started
/// when this is returned by the initial preflight.
#[derive(Debug)]
pub struct RecoveryRequired {
    pub recovery_manifest: PathBuf,
}

impl std::fmt::Display for RecoveryRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "resolve recovery or pending backup cleanup before replacing this set; recovery manifest: {}",
            self.recovery_manifest.display()
        )
    }
}

impl std::error::Error for RecoveryRequired {}

fn check_recovery_journal(parent: &Path, destination: &Path) -> io::Result<()> {
    let recovery_manifest = destination.join(MANIFEST);
    match fs::symlink_metadata(&recovery_manifest) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
        Ok(metadata) if metadata.is_file() => {
            let record = fs::read_to_string(&recovery_manifest)?;
            let last = |field: &str| {
                record
                    .lines()
                    .rev()
                    .find_map(|line| line.strip_prefix(field))
            };
            // Successful sets retain their historical journal. A committed
            // journal is resolved only after its exact backup sibling is gone.
            // Rollback failures must be resolved even if that backup is absent.
            if last("state=") == Some("committed") {
                let cleaned = match last("backup=") {
                    None => true,
                    Some(name) if validate_name(name).is_ok() => {
                        fs::symlink_metadata(parent.join(name))
                            .is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
                    }
                    Some(_) => false,
                };
                if cleaned {
                    return Ok(());
                }
            }
        }
        Ok(_) => {}
    }
    Err(io::Error::other(RecoveryRequired { recovery_manifest }))
}

#[derive(Debug)]
pub struct RollbackFailure {
    pub publication_error: io::Error,
    pub rollback_error: io::Error,
    pub recovery_manifest: PathBuf,
    pub backup: Option<PathBuf>,
}

impl std::fmt::Display for RollbackFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "publication failed: {}; rollback failed: {}; recovery manifest: {}",
            self.publication_error,
            self.rollback_error,
            self.recovery_manifest.display()
        )
    }
}

impl std::error::Error for RollbackFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.rollback_error)
    }
}

struct BoundedWriter<'a> {
    file: &'a mut File,
    remaining: u64,
}
impl Write for BoundedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(invalid("encoded file exceeds limit"));
        }
        let n = self.file.write(bytes)?;
        self.remaining -= n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
struct Staging {
    path: PathBuf,
    retain: bool,
}
impl Drop for Staging {
    fn drop(&mut self) {
        if !self.retain {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn journal(directory: &Path, record: &str) -> io::Result<()> {
    // Windows FlushFileBuffers needs a full write handle; append-only access
    // cannot sync. The journal has one writer, so seeking is sufficient.
    let mut manifest = OpenOptions::new()
        .write(true)
        .open(directory.join(MANIFEST))?;
    manifest.seek(SeekFrom::End(0))?;
    writeln!(manifest, "{record}")?;
    manifest.sync_all()
}

fn rollback<C>(
    publication_error: io::Error,
    staging: &mut Staging,
    destination: &Path,
    backup: Option<&Path>,
    published: bool,
    checkpoint: &mut C,
) -> io::Error
where
    C: FnMut(FailurePoint) -> io::Result<()>,
{
    let mut manifest_directory = destination;
    let result = (|| {
        if published {
            checkpoint(FailurePoint::BeforeRollbackRetire)?;
            fs::rename(destination, &staging.path)?;
        }
        manifest_directory = &staging.path;
        if let Some(backup) = backup {
            checkpoint(FailurePoint::BeforeRollbackRestore)?;
            // A new entry may have appeared after the original was backed up.
            // Restoring must not replace that entry, even on OSes whose rename
            // can replace an existing regular file.
            let parent = destination
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            preflight(
                parent,
                &[destination.file_name().unwrap().to_str().unwrap()],
            )?;
            fs::rename(backup, destination)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => publication_error,
        Err(rollback_error) => {
            // The only journal can still be inside the published directory if
            // retiring it failed. Neither that journal nor staging may be erased.
            staging.retain = true;
            let _ = journal(manifest_directory, "state=rollback-failed");
            io::Error::other(RollbackFailure {
                publication_error,
                rollback_error,
                recovery_manifest: manifest_directory.join(MANIFEST),
                backup: backup.map(Path::to_path_buf),
            })
        }
    }
}

fn cleanup_backup<C>(directory: &Path, deleted: &mut usize, checkpoint: &mut C) -> io::Result<()>
where
    C: FnMut(FailurePoint) -> io::Result<()>,
{
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            cleanup_backup(&entry.path(), deleted, checkpoint)?;
        } else {
            fs::remove_file(entry.path())?;
        }
        checkpoint(FailurePoint::AfterBackupCleanup(*deleted))?;
        *deleted += 1;
    }
    fs::remove_dir(directory)
}
fn unique_dir(parent: &Path, tag: &str) -> io::Result<PathBuf> {
    loop {
        let path = parent.join(format!(
            ".texelith-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

pub fn publish_set<E, C>(
    destination: &Path,
    files: &[ExportFile],
    options: PublishOptions,
    mut encode: E,
    mut checkpoint: C,
) -> io::Result<PublishOutcome>
where
    E: FnMut(&ExportFile, &mut dyn Write) -> io::Result<()>,
    C: FnMut(FailurePoint) -> io::Result<()>,
{
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let dest_name = destination
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("invalid destination"))?;
    validate_name(dest_name)?;
    if files.is_empty() {
        return Err(invalid("empty export set"));
    }
    let names: Vec<&str> = files.iter().map(ExportFile::name).collect();
    validate_names(&names)?;
    if names.iter().any(|n| key(n) == key(MANIFEST)) {
        return Err(invalid("reserved transaction filename"));
    }
    preflight_destination(parent, dest_name, options.collision)?;
    let mut staging = Staging {
        path: unique_dir(parent, "stage")?,
        retain: false,
    };
    let mut manifest = File::create(staging.path.join(MANIFEST))?;
    writeln!(manifest, "destination={dest_name}\nstate=staging")?;
    manifest.sync_all()?;
    drop(manifest);
    for (index, entry) in files.iter().enumerate() {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(staging.path.join(entry.name()))?;
        encode(
            entry,
            &mut BoundedWriter {
                file: &mut file,
                remaining: options.max_file_bytes,
            },
        )?;
        checkpoint(FailurePoint::AfterEncode(index))?;
        file.flush()?;
        file.sync_all()?;
        checkpoint(FailurePoint::AfterFlush(index))?;
    }
    let mut backup = None;
    if preflight_destination(parent, dest_name, options.collision)? {
        let path = unique_dir(parent, "backup")?;
        fs::remove_dir(&path)?;
        journal(
            &staging.path,
            &format!(
                "backup={}\nstate=backup-planned",
                path.file_name().unwrap().to_string_lossy()
            ),
        )?;
        fs::rename(destination, &path)?;
        backup = Some(path);
        if let Err(error) = checkpoint(FailurePoint::AfterBackupRename) {
            return Err(rollback(
                error,
                &mut staging,
                destination,
                backup.as_deref(),
                false,
                &mut checkpoint,
            ));
        }
    }
    if let Err(error) = preflight(parent, &[dest_name])
        .and_then(|()| journal(&staging.path, "state=publish-planned"))
        .and_then(|()| fs::rename(&staging.path, destination))
    {
        return Err(rollback(
            error,
            &mut staging,
            destination,
            backup.as_deref(),
            false,
            &mut checkpoint,
        ));
    }
    if let Err(error) = checkpoint(FailurePoint::AfterPublishRename)
        .and_then(|()| journal(destination, "state=committed\ncleanup=backup-pending"))
    {
        return Err(rollback(
            error,
            &mut staging,
            destination,
            backup.as_deref(),
            true,
            &mut checkpoint,
        ));
    }
    // This path no longer contains our staging directory after publication.
    // Do not let Drop remove a new entry created at that now-free path.
    staging.retain = true;
    if let Some(path) = backup
        && let Err(error) = cleanup_backup(&path, &mut 0, &mut checkpoint)
    {
        return Ok(PublishOutcome::CommittedWithCleanup {
            backup: path,
            recovery_manifest: destination.join(MANIFEST),
            error,
        });
    }
    Ok(PublishOutcome::Committed)
}
