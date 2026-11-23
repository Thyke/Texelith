use super::*;
use crossbeam_channel::{Receiver, Sender, bounded, select};
use texelith_io::project_io as io;

pub(super) enum Task {
    Open {
        token: ProjectOpenToken,
        path: PathBuf,
        overrides: BTreeMap<AssetId, PathBuf>,
        accepted: BTreeMap<AssetId, [u8; 32]>,
    },
    Save {
        path: PathBuf,
        old: Option<PathBuf>,
        document: Box<ProjectFile>,
        assets: BTreeMap<AssetId, PathBuf>,
        revision: u64,
        view_revision: u64,
    },
}
pub(super) struct Saved {
    pub path: PathBuf,
    pub revision: u64,
    pub view_revision: u64,
    pub warning: Option<String>,
}
pub(super) enum Reply {
    Opened(ProjectOpenToken, Result<Box<Opened>, String>),
    Saved(Result<Saved, String>),
}
pub(super) struct Worker {
    pub send: Sender<Task>,
    pub receive: Receiver<Reply>,
    stop: Sender<()>,
}
impl Worker {
    pub fn spawn() -> Result<Self, String> {
        let (send, receive) = bounded::<Task>(1);
        let (reply, results) = bounded(1);
        let (stop, stopped) = bounded(1);
        std::thread::Builder::new().name("texelith-project-io".into()).spawn(move || loop {
            let task = select! { recv(stopped) -> _ => break, recv(receive) -> r => match r { Ok(t) => t, Err(_) => break } };
            let _reservation = crate::batch_jobs::memory::reserve();
            let result = match task {
                Task::Open { token, path, overrides, accepted } => Reply::Opened(token, open(token,path,overrides,accepted).map(Box::new)),
                Task::Save { path, old, document, assets, revision, view_revision } => Reply::Saved(save(path,old,*document,assets,revision,view_revision)),
            };
            select! { recv(stopped) -> _ => break, send(reply,result) -> r => if r.is_err() { break } }
        }).map_err(|e| e.to_string())?;
        Ok(Self {
            send,
            receive: results,
            stop,
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
    }
}

fn err(e: impl ToString) -> String {
    e.to_string()
}
fn absolute(path: &std::path::Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    Ok(std::fs::canonicalize(parent).map_err(err)?.join(
        path.file_name()
            .ok_or_else(|| text("project-invalid-path"))?,
    ))
}
fn convention(dto: &ProjectFile) -> NormalConvention {
    dto.graph
        .graph
        .iter()
        .find_map(|n| {
            if let MaterialNodeParams::NormalSource { convention, .. } = n.params {
                Some(convention)
            } else {
                None
            }
        })
        .unwrap_or(NormalConvention::OpenGl)
}
fn source_mut(params: &mut MaterialNodeParams) -> Option<&mut SourceDescriptor> {
    match params {
        MaterialNodeParams::LegacySource(s)
        | MaterialNodeParams::ColorSource(s)
        | MaterialNodeParams::HeightSource(s)
        | MaterialNodeParams::NormalSource { source: s, .. } => Some(s),
        _ => None,
    }
}
fn open(
    token: ProjectOpenToken,
    path: PathBuf,
    overrides: BTreeMap<AssetId, PathBuf>,
    accepted: BTreeMap<AssetId, [u8; 32]>,
) -> Result<Opened, String> {
    let path = absolute(&path)?;
    io::recover_project(&path).map_err(err)?;
    let value = io::read_json(&path).map_err(err)?;
    let mut dto = io::read_project(&io::canonical_json(&value).map_err(err)?)
        .map_err(err)?
        .into_document();
    let convention = convention(&dto);
    let mut assets = BTreeMap::new();
    let mut uploads = VecDeque::new();
    let mut issues = vec![];
    let mut changed = false;
    let mut retained = 0usize;
    for asset in &mut dto.assets {
        let requested = overrides
            .get(&asset.id)
            .cloned()
            .unwrap_or_else(|| path.parent().unwrap().join(&asset.relative_path));
        let canonical = match std::fs::canonicalize(&requested) {
            Ok(path) => path,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                issues.push(AssetIssue {
                    id: asset.id,
                    path: requested,
                    changed: false,
                    decoded_hash: None,
                });
                continue;
            }
            Err(e) => return Err(err(e)),
        };
        assets.insert(asset.id, canonical.clone());
        let kind = match asset.interpretation {
            InputInterpretation::SrgbPhoto => crate::material_ui::ImportKind::Photo,
            InputInterpretation::LinearHeight => crate::material_ui::ImportKind::Height,
            InputInterpretation::LinearNormal => crate::material_ui::ImportKind::Normal,
        };
        let upload = crate::material_ui::presenter::decode(&canonical, kind, convention)?;
        retained += match &upload.pixels {
            texelith_gpu::TypedSourcePixels::Rgba8 { pixels, .. }
            | texelith_gpu::TypedSourcePixels::NormalRgb8 { pixels, .. } => pixels.len(),
            texelith_gpu::TypedSourcePixels::HeightU16(p) => p.len() * 2,
            texelith_gpu::TypedSourcePixels::HeightF32(p) => p.len() * 4,
        };
        if retained > 256 * 1024 * 1024 {
            return Err(text("project-memory-limit"));
        }
        let old_hash = asset.decoded_hash;
        let differs = old_hash != upload.descriptor.content_hash
            || dto
                .graph
                .graph
                .iter_mut()
                .filter_map(|n| source_mut(&mut n.params))
                .any(|s| s.content_hash == old_hash && s.extent != upload.descriptor.extent);
        if differs && accepted.get(&asset.id) != Some(&upload.descriptor.content_hash) {
            issues.push(AssetIssue {
                id: asset.id,
                path: canonical,
                changed: true,
                decoded_hash: Some(upload.descriptor.content_hash),
            });
        } else {
            for source in dto
                .graph
                .graph
                .iter_mut()
                .filter_map(|n| source_mut(&mut n.params))
            {
                if source.content_hash == old_hash && source.interpretation == asset.interpretation
                {
                    *source = upload.descriptor.clone();
                }
            }
            changed |= differs || overrides.contains_key(&asset.id);
            asset.decoded_hash = upload.descriptor.content_hash;
            uploads.push_back(upload);
        }
    }
    // Store portable references only after explicit acceptance; use the same
    // checked rebasing helper as Save As rather than persisting native paths.
    if issues.is_empty() {
        if uploads.is_empty() {
            return Err(text("project-no-relink"));
        }
        for asset in &mut dto.assets {
            asset.relative_path = relative(path.parent().unwrap(), &assets[&asset.id])?;
        }
        validate_project(dto.clone()).map_err(err)?;
    } else {
        uploads.clear();
    }
    Ok(Opened {
        token,
        path,
        document: dto,
        assets,
        uploads,
        issues,
        changed,
        accepted,
    })
}
fn relative(base: &std::path::Path, target: &std::path::Path) -> Result<String, String> {
    let b: Vec<_> = base.components().collect();
    let t: Vec<_> = target.components().collect();
    let common = b.iter().zip(&t).take_while(|(a, b)| a == b).count();
    if common == 0 {
        return Err(text("project-different-volume"));
    }
    let mut parts = vec!["..".to_owned(); b.len() - common];
    parts.extend(
        t[common..]
            .iter()
            .map(|p| p.as_os_str().to_string_lossy().into_owned()),
    );
    Ok(parts.join("/"))
}
fn save(
    path: PathBuf,
    old: Option<PathBuf>,
    mut document: ProjectFile,
    assets: BTreeMap<AssetId, PathBuf>,
    revision: u64,
    view_revision: u64,
) -> Result<Saved, String> {
    let path = absolute(&path)?;
    if !path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".texelith.json")
    {
        return Err(text("project-extension"));
    }
    if document.assets.is_empty() {
        let source = document
            .graph
            .graph
            .iter_mut()
            .find_map(|n| source_mut(&mut n.params))
            .ok_or_else(|| text("project-no-relink"))?;
        let target =
            std::fs::canonicalize(assets.get(&1).ok_or_else(|| text("project-no-relink"))?)
                .map_err(err)?;
        document.assets.push(AssetRef {
            id: 1,
            relative_path: relative(path.parent().unwrap(), &target)?,
            decoded_hash: source.content_hash,
            interpretation: source.interpretation,
        });
    } else if let Some(old) = old {
        document = io::rebase_project(&document, &old, &path).map_err(err)?;
    }
    validate_project(document.clone()).map_err(err)?;
    let warning = match io::save_project(&path, &document).map_err(err)? {
        io::SaveOutcome::Committed => None,
        io::SaveOutcome::CommittedWithCleanup { error, .. } => Some(error.to_string()),
    };
    Ok(Saved {
        path,
        revision,
        view_revision,
        warning,
    })
}
