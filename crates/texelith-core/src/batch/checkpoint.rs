use super::*;

pub const BATCH_CHECKPOINT_VERSION: u32 = 1;

/// Serializable recipe data, independent of ProjectFile and runtime generations.
/// T-087 hashes the serialized recipe and writes the durable manifest at IO.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRecipeDto {
    pub graph: Graph,
    pub export: ExportPlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchCheckpointItem {
    pub index: usize,
    pub input: BatchInput,
    pub state: BatchState,
    pub error: Option<String>,
}

/// Untrusted recovery data. Parsing and durable writes belong to IO; converting
/// with BatchJob::restore validates the recipe and the entire destination set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchCheckpoint {
    pub schema_version: u32,
    pub job: u64,
    pub recipe: BatchRecipeDto,
    pub items: Vec<BatchCheckpointItem>,
    pub continue_on_error: bool,
}

impl BatchJob {
    pub fn checkpoint(&self) -> BatchCheckpoint {
        BatchCheckpoint {
            schema_version: BATCH_CHECKPOINT_VERSION,
            job: self.id,
            recipe: BatchRecipeDto {
                graph: self.recipe.graph.clone(),
                export: self.settings.clone(),
            },
            items: self
                .items
                .iter()
                .map(|item| BatchCheckpointItem {
                    index: item.index,
                    input: BatchInput {
                        source: item.source.clone(),
                        source_digest: item.source_digest,
                        destinations: item.destinations.clone(),
                    },
                    state: item.state,
                    error: item.error.clone(),
                })
                .collect(),
            continue_on_error: self.continue_on_error,
        }
    }

    /// Restores a quiescent record, never scheduled work. Queued and interrupted
    /// phases become Cancelled; explicit resume_incomplete creates a fresh job.
    /// IO must verify source digests and committed sets before offering resume.
    pub fn restore(checkpoint: BatchCheckpoint) -> Result<Self, BatchError> {
        if checkpoint.schema_version != BATCH_CHECKPOINT_VERSION {
            return Err(BatchError::InvalidCheckpoint(
                "unsupported batch checkpoint version",
            ));
        }
        let capture = ExportCapture::new(
            &crate::EvaluationSnapshot {
                generation: 0,
                graph: checkpoint.recipe.graph,
            },
            0,
            &checkpoint.recipe.export,
        )
        .map_err(BatchError::InvalidRecipe)?;
        let mut job = Self::new(
            checkpoint.job,
            capture,
            checkpoint.recipe.export,
            checkpoint.items.iter().map(|i| i.input.clone()).collect(),
            checkpoint.continue_on_error,
        )?;
        for (item, saved) in job.items.iter_mut().zip(checkpoint.items) {
            item.index = saved.index;
            item.id.index = saved.index;
            item.state = saved.state;
            item.error = saved.error;
        }
        validate_items(&job.items, job.settings.outputs.len())?;
        for item in &mut job.items {
            if item.state == BatchState::Queued || item.state.is_active() {
                item.state = BatchState::Cancelled;
            }
        }
        Ok(job)
    }
}
