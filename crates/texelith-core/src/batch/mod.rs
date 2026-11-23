use crate::{
    CoreError, ExportCapture, ExportPlan, Graph, MaterialNodeParams, NodeParams, OpaqueLocator,
    SourceDescriptor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

mod checkpoint;
pub use checkpoint::*;

/// Substitute the single recipe source without changing the original graph.
/// Paint masks use the replacement extent; their strokes remain in source UV units.
pub fn instantiate_recipe(recipe: &Graph, source: SourceDescriptor) -> Result<Graph, CoreError> {
    if let Some(mut dto) = recipe.material.clone() {
        let mut count = 0;
        for node in &mut dto.graph {
            let target = match &mut node.params {
                MaterialNodeParams::LegacySource(s)
                | MaterialNodeParams::ColorSource(s)
                | MaterialNodeParams::HeightSource(s)
                | MaterialNodeParams::NormalSource { source: s, .. } => Some(s),
                MaterialNodeParams::PaintMask { extent, .. } => {
                    *extent = source.extent;
                    None
                }
                _ => None,
            };
            if let Some(target) = target {
                *target = source.clone();
                count += 1;
            }
        }
        if count != 1 {
            return Err(CoreError::InvalidMaterial(
                "batch recipe must have exactly one source",
            ));
        }
        Graph::try_from(dto)
    } else {
        let mut nodes = recipe.nodes().to_vec();
        let mut count = 0;
        for node in &mut nodes {
            if let NodeParams::Source(target) = &mut node.params {
                *target = source.clone();
                count += 1;
            }
        }
        if count != 1 {
            return Err(CoreError::InvalidMaterial(
                "batch recipe must have exactly one source",
            ));
        }
        Graph::new(nodes, recipe.selected_output())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchState {
    Queued,
    Loading,
    Evaluating,
    Encoding,
    Committed,
    Failed,
    Cancelled,
}
impl BatchState {
    fn is_active(self) -> bool {
        matches!(self, Self::Loading | Self::Evaluating | Self::Encoding)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchAction {
    Start,
    Cancel,
    RetryFailed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchItemId {
    pub job: u64,
    pub index: usize,
}
/// A single phase's completion identity. Never reconstruct this from a live job
/// when delivering a worker result: retain the token issued with that work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchWorkId {
    pub item: BatchItemId,
    pub generation: u64,
    pub request: u64,
}

/// Native preflight supplies canonical destination collision keys for every
/// output (including sidecars). Core compares them case-insensitively for portable
/// exports, but never resolves them as paths or probes existing files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchInput {
    pub source: OpaqueLocator,
    pub source_digest: [u8; 32],
    pub destinations: Vec<OpaqueLocator>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchItem {
    pub id: BatchItemId,
    pub index: usize,
    pub source: OpaqueLocator,
    pub source_digest: [u8; 32],
    pub destinations: Vec<OpaqueLocator>,
    pub state: BatchState,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    UnknownItem {
        index: usize,
    },
    InvalidTransition {
        index: usize,
        from: BatchState,
        to: BatchState,
    },
    ActiveItem,
    StoppedOnError,
    DuplicateJobId {
        id: u64,
    },
    NoFailedItems,
    NoIncompleteItems,
    StaleCompletion {
        received: BatchWorkId,
    },
    RequestOverflow,
    InvalidCheckpoint(&'static str),
    DuplicateDestination {
        destination: OpaqueLocator,
    },
    InvalidRecipe(CoreError),
}
#[derive(Clone, Debug, PartialEq)]
pub struct BatchJob {
    id: u64,
    recipe: ExportCapture,
    settings: ExportPlan,
    items: Vec<BatchItem>,
    continue_on_error: bool,
    pending: Option<BatchWorkId>,
    next_request: u64,
}
impl BatchJob {
    pub fn new(
        id: u64,
        recipe: ExportCapture,
        settings: ExportPlan,
        sources: Vec<BatchInput>,
        continue_on_error: bool,
    ) -> Result<Self, BatchError> {
        // Public captures are transport DTOs: do not trust their cached metadata.
        let expected = ExportCapture::new(
            &crate::EvaluationSnapshot {
                generation: recipe.generation,
                graph: recipe.graph.clone(),
            },
            recipe.document_revision,
            &settings,
        )
        .map_err(BatchError::InvalidRecipe)?;
        if recipe != expected || recipe.sources.len() != 1 {
            return Err(BatchError::InvalidCheckpoint("invalid captured recipe"));
        }
        instantiate_recipe(&recipe.graph, recipe.sources[0].clone())
            .map_err(BatchError::InvalidRecipe)?;
        let items = sources
            .into_iter()
            .enumerate()
            .map(|(index, input)| BatchItem {
                id: BatchItemId { job: id, index },
                index,
                source: input.source,
                source_digest: input.source_digest,
                destinations: input.destinations,
                state: BatchState::Queued,
                error: None,
            })
            .collect::<Vec<_>>();
        validate_items(&items, settings.outputs.len())?;
        Ok(Self {
            id,
            recipe,
            settings,
            items,
            continue_on_error,
            pending: None,
            next_request: 1,
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn items(&self) -> &[BatchItem] {
        &self.items
    }
    pub fn recipe(&self) -> &ExportCapture {
        &self.recipe
    }
    pub fn export_settings(&self) -> &ExportPlan {
        &self.settings
    }
    pub fn pending_work(&self) -> Option<BatchWorkId> {
        self.pending
    }
    pub fn can_start(&self) -> bool {
        self.pending.is_none()
            && self.items.iter().any(|i| i.state == BatchState::Queued)
            && (self.continue_on_error || !self.items.iter().any(|i| i.state == BatchState::Failed))
            && self.next_request < u64::MAX
    }
    pub fn can_retry_failed(&self) -> bool {
        self.pending.is_none() && self.items.iter().any(|i| i.state == BatchState::Failed)
    }
    pub fn can_cancel(&self) -> bool {
        self.items
            .iter()
            .any(|i| i.state == BatchState::Queued || i.state.is_active())
    }

    /// Gets the token for scheduling a queued item, or inspecting its active phase.
    pub fn work_id(&self, index: usize) -> Result<BatchWorkId, BatchError> {
        let item = self
            .items
            .iter()
            .find(|i| i.index == index)
            .ok_or(BatchError::UnknownItem { index })?;
        Ok(self
            .pending
            .filter(|work| work.item == item.id)
            .unwrap_or(BatchWorkId {
                item: item.id,
                generation: self.recipe.generation,
                request: self.next_request,
            }))
    }

    pub fn transition(&mut self, work: BatchWorkId, to: BatchState) -> Result<(), BatchError> {
        self.validate_work(work)?;
        let index = work.item.index;
        let pos = self
            .items
            .iter()
            .position(|item| item.index == index)
            .ok_or(BatchError::UnknownItem { index })?;
        let from = self.items[pos].state;
        if !matches!(
            (from, to),
            (BatchState::Queued, BatchState::Loading)
                | (BatchState::Loading, BatchState::Evaluating)
                | (BatchState::Evaluating, BatchState::Encoding)
                | (BatchState::Encoding, BatchState::Committed)
        ) {
            return Err(BatchError::InvalidTransition { index, from, to });
        }
        if to == BatchState::Loading {
            if self.pending.is_some() {
                return Err(BatchError::ActiveItem);
            }
            if !self.continue_on_error && self.items.iter().any(|i| i.state == BatchState::Failed) {
                return Err(BatchError::StoppedOnError);
            }
        }
        let next_request = self
            .next_request
            .checked_add(1)
            .ok_or(BatchError::RequestOverflow)?;
        self.items[pos].state = to;
        self.pending = if to == BatchState::Committed {
            None
        } else if to == BatchState::Loading {
            Some(work)
        } else {
            Some(BatchWorkId {
                request: self.next_request,
                ..work
            })
        };
        self.next_request = next_request;
        Ok(())
    }
    pub fn fail(&mut self, work: BatchWorkId, error: String) -> Result<(), BatchError> {
        self.validate_work(work)?;
        if error.is_empty() {
            return Err(BatchError::InvalidCheckpoint("empty item error"));
        }
        let index = work.item.index;
        let item = self
            .items
            .iter_mut()
            .find(|item| item.index == index)
            .ok_or(BatchError::UnknownItem { index })?;
        if !matches!(
            item.state,
            BatchState::Loading | BatchState::Evaluating | BatchState::Encoding
        ) {
            return Err(BatchError::InvalidTransition {
                index,
                from: item.state,
                to: BatchState::Failed,
            });
        }
        item.state = BatchState::Failed;
        item.error = Some(error);
        self.pending = None;
        Ok(())
    }
    pub fn cancel(&mut self) -> Result<(), BatchError> {
        for item in &mut self.items {
            if matches!(
                item.state,
                BatchState::Queued
                    | BatchState::Loading
                    | BatchState::Evaluating
                    | BatchState::Encoding
            ) {
                item.state = BatchState::Cancelled;
            }
        }
        Ok(())
    }
    pub fn retry_failed(&self, id: u64) -> Result<Self, BatchError> {
        self.requeue(id, true)
    }

    /// Explicit restart choice, including interrupted/cancelled/queued work.
    /// Committed destinations are always excluded; nothing resumes on restore.
    pub fn resume_incomplete(&self, id: u64) -> Result<Self, BatchError> {
        self.requeue(id, false)
    }

    /// A cancelled phase remains owned until its submitted work has retired.
    pub fn retire_cancelled(&mut self, work: BatchWorkId) -> Result<(), BatchError> {
        if self.pending != Some(work) {
            return Err(BatchError::StaleCompletion { received: work });
        }
        if !self
            .items
            .iter()
            .any(|i| i.id == work.item && i.state == BatchState::Cancelled)
        {
            return Err(BatchError::ActiveItem);
        }
        self.pending = None;
        Ok(())
    }

    fn validate_work(&self, work: BatchWorkId) -> Result<(), BatchError> {
        if work.item.job != self.id || work.generation != self.recipe.generation {
            return Err(BatchError::StaleCompletion { received: work });
        }
        let item = self
            .items
            .iter()
            .find(|i| i.index == work.item.index)
            .ok_or(BatchError::UnknownItem {
                index: work.item.index,
            })?;
        let expected = self.work_id(item.index)?;
        if expected != work {
            return Err(BatchError::StaleCompletion { received: work });
        }
        Ok(())
    }

    fn requeue(&self, id: u64, failed_only: bool) -> Result<Self, BatchError> {
        if id <= self.id {
            return Err(BatchError::DuplicateJobId { id });
        }
        if self.pending.is_some() {
            return Err(BatchError::ActiveItem);
        }
        let items: Vec<_> = self
            .items
            .iter()
            .filter(|i| {
                if failed_only {
                    i.state == BatchState::Failed
                } else {
                    i.state != BatchState::Committed
                }
            })
            .map(|i| BatchItem {
                id: BatchItemId {
                    job: id,
                    index: i.index,
                },
                index: i.index,
                source: i.source.clone(),
                source_digest: i.source_digest,
                destinations: i.destinations.clone(),
                state: BatchState::Queued,
                error: None,
            })
            .collect();
        if items.is_empty() {
            return Err(if failed_only {
                BatchError::NoFailedItems
            } else {
                BatchError::NoIncompleteItems
            });
        }
        Ok(Self {
            id,
            recipe: self.recipe.clone(),
            settings: self.settings.clone(),
            items,
            continue_on_error: self.continue_on_error,
            pending: None,
            next_request: 1,
        })
    }
}

fn validate_items(items: &[BatchItem], output_count: usize) -> Result<(), BatchError> {
    if items.is_empty() || items.len() > 10_000 {
        return Err(BatchError::InvalidCheckpoint(
            "batch requires 1..=10000 items",
        ));
    }
    let mut sources = HashSet::new();
    let mut source_keys = HashSet::new();
    let mut destinations = HashSet::new();
    let mut previous = None;
    for item in items {
        if item.index >= 10_000
            || previous.is_some_and(|index| index >= item.index)
            || item.id.index != item.index
        {
            return Err(BatchError::InvalidCheckpoint("invalid stable item index"));
        }
        previous = Some(item.index);
        if item.source.0.is_empty() || !sources.insert(&item.source.0) {
            return Err(BatchError::InvalidCheckpoint(
                "empty or duplicate source token",
            ));
        }
        source_keys.insert(item.source.0.to_lowercase());
        if item.destinations.len() < output_count {
            return Err(BatchError::InvalidCheckpoint("missing output destinations"));
        }
        for destination in &item.destinations {
            if destination.0.is_empty() {
                return Err(BatchError::InvalidCheckpoint("empty destination token"));
            }
            if !destinations.insert(destination.0.to_lowercase()) {
                return Err(BatchError::DuplicateDestination {
                    destination: destination.clone(),
                });
            }
        }
        if (item.state == BatchState::Failed) != item.error.is_some()
            || item.error.as_ref().is_some_and(String::is_empty)
        {
            return Err(BatchError::InvalidCheckpoint("invalid item error state"));
        }
    }
    if items.iter().filter(|i| i.state.is_active()).count() > 1 {
        return Err(BatchError::InvalidCheckpoint("multiple active items"));
    }
    // Reject input/output aliasing too; equality keys are canonicalized by IO.
    if let Some(destination) = items
        .iter()
        .flat_map(|i| &i.destinations)
        .find(|d| source_keys.contains(&d.0.to_lowercase()))
    {
        return Err(BatchError::DuplicateDestination {
            destination: destination.clone(),
        });
    }
    Ok(())
}
