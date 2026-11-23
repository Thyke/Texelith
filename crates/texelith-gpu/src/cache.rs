//! Content identity is the core BLAKE3 schema, independent of request tokens.
use crate::GpuError;
use std::collections::HashMap;
use texelith_core::{
    CacheKey, CacheRevisions, EvaluationSnapshot, Extent, NodeId, NodeParams,
    node_cache_key_with_revisions,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheAddress {
    pub content: CacheKey,
    pub format: wgpu::TextureFormat,
}

pub struct EvaluationCache<T> {
    entries: HashMap<CacheAddress, (T, u64)>,
    clock: u64,
}
impl<T> Default for EvaluationCache<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
        }
    }
}
impl<T> EvaluationCache<T> {
    pub fn insert(&mut self, address: CacheAddress, value: T) {
        self.clock += 1;
        self.entries.insert(address, (value, self.clock));
    }
    pub fn get(&mut self, address: &CacheAddress) -> Option<&T> {
        self.clock += 1;
        self.entries.get_mut(address).map(|(value, stamp)| {
            *stamp = self.clock;
            &*value
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub(crate) fn weight(&self, weight: impl Fn(&T) -> u64) -> u64 {
        self.entries.values().map(|(v, _)| weight(v)).sum()
    }
    pub(crate) fn retain(&mut self, keep: impl Fn(&CacheAddress) -> bool) -> usize {
        let before = self.entries.len();
        self.entries.retain(|key, _| keep(key));
        before - self.entries.len()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub(crate) fn evict_oldest(&mut self) -> bool {
        let key = self
            .entries
            .iter()
            .min_by_key(|(_, (_, stamp))| *stamp)
            .map(|(key, _)| *key);
        if let Some(key) = key {
            self.entries.remove(&key);
            true
        } else {
            false
        }
    }
}

pub fn evaluation_keys(
    snapshot: &EvaluationSnapshot,
    extent: Extent,
    revisions: CacheRevisions,
) -> Result<Vec<(NodeId, CacheKey)>, GpuError> {
    extent.validate()?;
    let mut keys = HashMap::new();
    let mut sources = HashMap::new();
    let mut ordered = Vec::new();
    for node in snapshot.graph.topological_nodes() {
        let source = match &node.params {
            NodeParams::Source(source) => source.extent,
            _ => sources[&node.input.expect("validated input")],
        };
        let inputs: Vec<_> = node.input.map(|id| keys[&id]).into_iter().collect();
        let key = node_cache_key_with_revisions(node, &inputs, extent, source, revisions)?;
        keys.insert(node.id, key);
        sources.insert(node.id, source);
        ordered.push((node.id, key));
    }
    Ok(ordered)
}
