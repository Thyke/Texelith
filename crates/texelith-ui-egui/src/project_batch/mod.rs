//! Project persistence and batch controls share the application's command route.
mod panel;
pub mod smoke;
pub(crate) mod snapshots;
mod worker;
use crate::{command_adapter::UiAction, controls::Controls, input_locale::labels::text};
use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
};
use texelith_core::*;
use texelith_gpu::TypedSourceUpload;

#[derive(Clone, Debug)]
pub struct AssetIssue {
    pub id: AssetId,
    pub path: PathBuf,
    pub changed: bool,
    decoded_hash: Option<[u8; 32]>,
}
#[derive(Clone, Debug)]
pub enum AssetChoice {
    Cancel,
    UseChanged,
    Relink(PathBuf),
}
#[derive(Clone, Copy, Debug)]
pub enum RestartChoice {
    KeepStopped,
    RetryIncomplete,
}

/// A restart is opt-in and derives only incomplete items from the captured recipe.
pub fn restart_batch(job: &BatchJob, choice: RestartChoice) -> Result<Option<BatchJob>, String> {
    match choice {
        RestartChoice::KeepStopped => Ok(None),
        RestartChoice::RetryIncomplete => job
            .resume_incomplete(job.id().checked_add(1).ok_or("Batch IDs exhausted")?)
            .map(Some)
            .map_err(|e| format!("{e:?}")),
    }
}

pub(crate) struct Opened {
    token: ProjectOpenToken,
    path: PathBuf,
    document: ProjectFile,
    assets: BTreeMap<AssetId, PathBuf>,
    uploads: VecDeque<TypedSourceUpload>,
    issues: Vec<AssetIssue>,
    changed: bool,
    accepted: BTreeMap<AssetId, [u8; 32]>,
}
#[derive(Default)]
pub(crate) struct State {
    worker: Option<worker::Worker>,
    pending: VecDeque<worker::Task>,
    inflight: usize,
    latest: Option<ProjectOpenToken>,
    opened: Option<Opened>,
    uploading: Option<[u8; 32]>,
    pub path: Option<PathBuf>,
    assets: BTreeMap<AssetId, PathBuf>,
    pub feedback: String,
    pub save_picker: bool,
    pub save_path: String,
    pub batch_open: bool,
    pub batch_folder: String,
    pub batch_destination: String,
    pub batch_manifest: String,
    pub batch_continue: bool,
    pub batch_recover: Option<PathBuf>,
    pub relink_path: String,
    pub restart_offer: bool,
    pending_source: Option<(ImportToken, PathBuf)>,
    saving: bool,
    document_id: u64,
    saved_document_id: u64,
}
impl Controls {
    pub fn resolve_batch_restart(&mut self, choice: RestartChoice) -> Result<(), String> {
        let job = self.session.batch().ok_or("No recovered batch")?;
        if let Some(job) = restart_batch(job, choice)? {
            self.session.set_batch(job).map_err(|e| format!("{e:?}"))?;
            self.action(UiAction::Request(CommandRequest::BatchStart {}));
        }
        self.projects.restart_offer = false;
        Ok(())
    }
    pub(crate) fn project_upload_pending(&self) -> bool {
        self.projects.uploading.is_some()
    }
    pub(crate) fn remember_project_source(&mut self, token: ImportToken, path: PathBuf) {
        self.projects.pending_source = Some((token, path));
        self.projects.latest = None;
        self.projects.opened = None;
    }
    pub(crate) fn commit_project_source(&mut self, token: ImportToken) {
        if self
            .projects
            .pending_source
            .as_ref()
            .is_some_and(|(t, _)| *t == token)
        {
            let (_, path) = self.projects.pending_source.take().unwrap();
            self.projects.assets = [(1, path)].into();
            self.projects.path = None;
            self.projects.document_id += 1;
        }
    }
    pub fn project_path(&self) -> Option<&std::path::Path> {
        self.projects.path.as_deref()
    }
    pub fn project_save_picker_open(&self) -> bool {
        self.projects.save_picker
    }
    pub fn project_feedback(&self) -> &str {
        &self.projects.feedback
    }
    pub fn project_busy(&self) -> bool {
        self.projects.inflight > 0
            || !self.projects.pending.is_empty()
            || self
                .projects
                .opened
                .as_ref()
                .is_some_and(|o| o.issues.is_empty())
    }
    pub fn project_asset_issues(&self) -> &[AssetIssue] {
        self.projects.opened.as_ref().map_or(&[], |o| &o.issues)
    }
    fn project_enqueue(&mut self, task: worker::Task) -> Result<(), String> {
        if self.projects.worker.is_none() {
            self.projects.worker = Some(worker::Worker::spawn()?);
        }
        // One running operation plus one replacement intent; never an unbounded queue.
        if !self.projects.pending.is_empty() {
            return Err(text("project-busy"));
        }
        self.projects.pending.push_back(task);
        Ok(())
    }
    pub fn open_project_path(&mut self, path: PathBuf) {
        self.next_import = self
            .next_import
            .checked_add(1)
            .expect("project tokens exhausted");
        let source = self.locator(path);
        self.action(UiAction::Request(CommandRequest::OpenProject {
            token: ProjectOpenToken(self.next_import),
            source,
        }));
    }
    pub fn cancel_project_open(&mut self) {
        if let Some(token) = self.projects.latest.take() {
            let _ = self.session.fail_project_open(token);
        }
        self.projects.opened = None;
        self.projects.uploading = None;
        self.projects
            .pending
            .retain(|t| !matches!(t, worker::Task::Open { .. }));
        self.projects.feedback.clear();
    }
    pub fn save_project_to(&mut self, path: PathBuf, save_as: bool) -> Result<(), String> {
        if self.projects.saving {
            return Err(text("project-busy"));
        }
        self.action(UiAction::FocusLost);
        let plan = self.packing_plan()?;
        self.action(UiAction::Request(CommandRequest::SetExportSettings {
            settings: plan,
        }));
        let destination = self.locator(path);
        self.action(UiAction::Request(if save_as {
            CommandRequest::SaveProjectAs { destination }
        } else {
            CommandRequest::SaveProject { destination }
        }));
        if self.projects.saving {
            Ok(())
        } else {
            Err(self.projects.feedback.clone())
        }
    }
    pub(crate) fn project_save_intent(&mut self, save_as: bool) {
        if !save_as && let Some(path) = self.projects.path.clone() {
            if let Err(e) = self.save_project_to(path, false) {
                self.projects.feedback = e;
            }
        } else {
            self.projects.save_picker = true;
            self.projects.save_path = self
                .projects
                .path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Material.texelith.json".into());
        }
    }
    pub(crate) fn route_project_effect(&mut self, effect: Effect) -> Option<Effect> {
        let retained = matches!(effect, Effect::OpenProject { .. }).then(|| effect.clone());
        let result = match effect {
            Effect::OpenProject { token, source } => {
                self.projects.latest = Some(token);
                self.projects.opened = None;
                self.projects.uploading = None;
                self.projects
                    .pending
                    .retain(|t| !matches!(t, worker::Task::Open { .. }));
                self.take_path(source)
                    .ok_or_else(|| text("project-invalid-path"))
                    .and_then(|path| {
                        self.project_enqueue(worker::Task::Open {
                            token,
                            path,
                            overrides: BTreeMap::new(),
                            accepted: BTreeMap::new(),
                        })
                    })
            }
            Effect::SaveProject { destination, .. } => (|| {
                if self.projects.saving {
                    return Err(text("project-busy"));
                }
                let path = self
                    .take_path(destination)
                    .ok_or_else(|| text("project-invalid-path"))?;
                let snapshot = self
                    .session
                    .snapshot()
                    .ok_or_else(|| text("export-no-material"))?;
                let document = ProjectFile {
                    format: "texelith.project".into(),
                    schema_version: 1,
                    graph: snapshot.graph.into(),
                    assets: self.session.project_assets().to_vec(),
                    export: self
                        .session
                        .export_settings()
                        .cloned()
                        .ok_or_else(|| text("export-no-material"))?,
                    view: self.session.view_state().clone().into(),
                };
                self.project_enqueue(worker::Task::Save {
                    path,
                    old: self.projects.path.clone(),
                    document: Box::new(document),
                    assets: self.projects.assets.clone(),
                    revision: self.session.document_revision(),
                    view_revision: self.session.view_revision(),
                })?;
                self.projects.saving = true;
                self.projects.saved_document_id = self.projects.document_id;
                Ok(())
            })(),
            other => return Some(other),
        };
        self.projects.feedback = result.err().unwrap_or_else(|| text("project-working"));
        retained
    }
    pub fn poll_project_io(&mut self) {
        self.effects
            .retain(|e| !matches!(e, Effect::OpenProject { .. } | Effect::SaveProject { .. }));
        if let Some(task) = self.projects.pending.pop_front() {
            let worker = self.projects.worker.as_ref().unwrap();
            match worker.send.try_send(task) {
                Ok(()) => self.projects.inflight += 1,
                Err(crossbeam_channel::TrySendError::Full(task)) => {
                    self.projects.pending.push_front(task)
                }
                Err(_) => {
                    self.projects.feedback = text("project-worker-closed");
                    self.projects.saving = false;
                }
            }
        }
        loop {
            let reply = self
                .projects
                .worker
                .as_ref()
                .and_then(|w| w.receive.try_recv().ok());
            let Some(reply) = reply else {
                break;
            };
            self.projects.inflight = self.projects.inflight.saturating_sub(1);
            match reply {
                worker::Reply::Opened(token, result) if self.projects.latest == Some(token) => {
                    match result {
                        Ok(opened) => {
                            self.projects.feedback.clear();
                            self.projects.opened = Some(*opened);
                        }
                        Err(error) => {
                            let _ = self.session.fail_project_open(token);
                            self.projects.latest = None;
                            self.projects.feedback = error;
                            self.input.feedback = None;
                        }
                    }
                }
                worker::Reply::Opened(..) => {}
                worker::Reply::Saved(result) => {
                    self.projects.saving = false;
                    match result {
                        Ok(saved) => {
                            if self.projects.saved_document_id != self.projects.document_id {
                                continue;
                            }
                            let clean = self
                                .session
                                .mark_project_saved(saved.revision, saved.view_revision);
                            self.projects.path = Some(saved.path);
                            self.projects.feedback = saved.warning.unwrap_or_else(|| {
                                text(if clean {
                                    "project-saved"
                                } else {
                                    "project-saved-edits"
                                })
                            });
                            if clean && self.input.pending.is_some() {
                                self.resolve_dirty(crate::input_locale::DirtyChoice::Discard);
                            }
                        }
                        Err(error) => self.projects.feedback = error,
                    }
                }
            }
        }
    }
    pub fn take_project_upload(&mut self) -> Option<TypedSourceUpload> {
        if self.projects.uploading.is_some() {
            return None;
        }
        let opened = self.projects.opened.as_mut()?;
        if !opened.issues.is_empty() {
            return None;
        }
        let upload = opened.uploads.pop_front()?;
        self.projects.uploading = Some(upload.descriptor.content_hash);
        Some(upload)
    }
    pub fn ack_project_upload(&mut self, hash: [u8; 32]) -> bool {
        if self.projects.uploading != Some(hash) {
            return false;
        }
        self.projects.uploading = None;
        let Some(opened) = &self.projects.opened else {
            return false;
        };
        if !opened.uploads.is_empty() {
            return true;
        }
        let opened = self.projects.opened.take().unwrap();
        let result = validate_project(opened.document)
            .map_err(|e| e.to_string())
            .and_then(|p| {
                self.session
                    .complete_project_open(opened.token, p)
                    .map_err(|e| e.to_string())
            });
        match result {
            Ok(_) => {
                self.projects.path = Some(opened.path);
                self.projects.assets = opened.assets;
                self.projects.latest = None;
                self.projects.document_id += 1;
                self.input.feedback = None;
                if opened.changed {
                    // A changed linked source was explicitly accepted. Keep its new
                    // identity dirty until the user saves the updated recipe.
                    self.session.mark_project_modified();
                }
                self.workspace.clear_thumbnails();
                self.inspectors.cancel_transients();
                self.adapter.sync(&self.session);
                self.render_revision += 1;
                self.packing.draft = None;
                self.packing.open = false;
                self.restore_project_export();
                self.projects.feedback = text("project-opened");
            }
            Err(error) => self.projects.feedback = error,
        }
        true
    }
    pub fn fail_project_upload(&mut self, error: String) {
        self.cancel_project_open();
        self.projects.feedback = error;
    }
    pub fn resolve_project_asset(&mut self, choice: AssetChoice) -> Result<(), String> {
        if matches!(choice, AssetChoice::Cancel) {
            self.cancel_project_open();
            return Ok(());
        }
        let opened = self
            .projects
            .opened
            .as_ref()
            .ok_or_else(|| text("project-no-relink"))?;
        let issue = opened
            .issues
            .first()
            .ok_or_else(|| text("project-no-relink"))?;
        let mut overrides = opened.assets.clone();
        let mut accepted = opened.accepted.clone();
        match choice {
            AssetChoice::UseChanged if issue.changed => {
                accepted.insert(
                    issue.id,
                    issue
                        .decoded_hash
                        .ok_or_else(|| text("project-no-relink"))?,
                );
            }
            AssetChoice::Relink(path) => {
                overrides.insert(issue.id, path);
                accepted.remove(&issue.id);
            }
            _ => return Err(text("project-no-relink")),
        };
        let task = worker::Task::Open {
            token: opened.token,
            path: opened.path.clone(),
            overrides,
            accepted,
        };
        self.projects.opened = None;
        self.project_enqueue(task)
    }
    fn restore_project_export(&mut self) {
        let Some(plan) = self.session.export_settings() else {
            return;
        };
        let graph = self.session.snapshot().unwrap().graph;
        let mut preset = ExportPreset {
            profile: plan.profile.clone(),
            size: plan.size,
            naming: plan.naming.clone(),
            padding: plan.padding,
            displacement: plan.displacement,
            outputs: vec![],
        };
        for output in &plan.outputs {
            let channels = output
                .channels
                .iter()
                .map(|channel| {
                    let source = match channel.source {
                        ChannelSource::Constant(v) => PresetChannelSource::Constant(v),
                        ChannelSource::Map(reference, component) => {
                            let map = graph
                                .material_outputs()
                                .unwrap()
                                .iter()
                                .find(|(_, r)| **r == reference)
                                .map(|(m, _)| *m)
                                .unwrap_or(reference.port.map_kind());
                            PresetChannelSource::Map {
                                map,
                                component,
                                missing: MissingMap::Error,
                            }
                        }
                    };
                    PresetChannelSpec {
                        source,
                        invert: channel.invert,
                        transfer: channel.transfer,
                    }
                })
                .collect();
            preset.outputs.push(PresetOutput {
                name: output.name.clone(),
                format: output.format,
                channels,
                quantization: output.quantization,
                import_hints: output.import_hints.clone(),
            });
        }
        self.export_options.preset = preset.profile.id.clone();
        self.export_options.size = preset.size;
        self.native_files.naming = None;
        self.packing.draft = Some(preset);
    }
}
