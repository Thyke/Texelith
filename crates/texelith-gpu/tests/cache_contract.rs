//! Exercise the worker's real cache and planner, not a local hash-map oracle.
mod worker_support;

use texelith_core::{CacheKey, CacheRevisions, EvaluationSnapshot, NodeId};
use texelith_gpu::cache::{CacheAddress, EvaluationCache, evaluation_keys};
use wgpu::TextureFormat;
use worker_support::{PROXY, SOURCE, snapshot};

fn keys(snapshot: &EvaluationSnapshot) -> Vec<(NodeId, CacheKey)> {
    evaluation_keys(snapshot, SOURCE, CacheRevisions::CURRENT).unwrap()
}

fn address(index: usize, key: CacheKey) -> CacheAddress {
    CacheAddress {
        content: key,
        format: if index == 3 {
            TextureFormat::Rgba16Float
        } else {
            TextureFormat::R32Float
        },
    }
}

#[test]
fn cache_normal_strength_edit_hits_source_levels_blur_and_misses_normal() {
    let original = keys(&snapshot(1, 7, 1.0));
    let edited = keys(&snapshot(2, 7, 9.0));
    let mut cache = EvaluationCache::default();
    for (index, (_, key)) in original.iter().enumerate() {
        cache.insert(address(index, *key), index);
    }
    assert_eq!(cache.len(), 4);
    for index in 0..3 {
        assert_eq!(original[index], edited[index]);
        assert_eq!(cache.get(&address(index, edited[index].1)), Some(&index));
    }
    assert_ne!(original[3].1, edited[3].1);
    assert!(cache.get(&address(3, edited[3].1)).is_none());
}

#[test]
fn cache_undo_new_generation_can_hit_older_normal_content() {
    let before = keys(&snapshot(1, 7, 1.0));
    let edit = keys(&snapshot(2, 7, 9.0));
    let undo = keys(&snapshot(3, 7, 1.0));
    let mut cache = EvaluationCache::default();
    cache.insert(address(3, before[3].1), "before");
    cache.insert(address(3, edit[3].1), "edited");
    assert_eq!(undo, before);
    assert_eq!(cache.get(&address(3, undo[3].1)), Some(&"before"));
    assert_eq!(cache.get(&address(3, edit[3].1)), Some(&"edited"));
}

#[test]
fn cache_identity_ignores_generation_and_transient_node_ids() {
    let original = snapshot(1, 7, 1.0);
    let mut renumbered = original.graph.nodes().to_vec();
    for node in &mut renumbered {
        node.id.0 += 100;
        node.input = node.input.map(|id| NodeId(id.0 + 100));
    }
    let later = EvaluationSnapshot {
        generation: u64::MAX,
        graph: texelith_core::Graph::new(renumbered, NodeId(104)).unwrap(),
    };
    let first = keys(&original);
    let second = keys(&later);
    assert_eq!(
        first.iter().map(|(_, key)| key).collect::<Vec<_>>(),
        second.iter().map(|(_, key)| key).collect::<Vec<_>>()
    );
}

#[test]
fn cache_extent_shader_revision_and_format_changes_miss() {
    let graph = snapshot(1, 7, 1.0);
    let full = keys(&graph);
    let proxy = evaluation_keys(&graph, PROXY, CacheRevisions::CURRENT).unwrap();
    let shader_changed = evaluation_keys(
        &graph,
        SOURCE,
        CacheRevisions {
            algorithm: CacheRevisions::CURRENT.algorithm + 1,
            ..CacheRevisions::CURRENT
        },
    )
    .unwrap();
    let mut cache = EvaluationCache::default();
    for (index, (_, key)) in full.iter().enumerate() {
        cache.insert(address(index, *key), index);
    }
    for index in 0..4 {
        for changed in [&proxy, &shader_changed] {
            assert_ne!(full[index].1, changed[index].1);
            assert!(cache.get(&address(index, changed[index].1)).is_none());
        }
        let wrong_format = CacheAddress {
            content: full[index].1,
            format: TextureFormat::Rgba32Float,
        };
        assert!(cache.get(&wrong_format).is_none());
    }
}

#[test]
fn cache_replacing_source_invalidates_every_descendant() {
    let before = keys(&snapshot(1, 7, 1.0));
    let after = keys(&snapshot(2, 8, 1.0));
    let mut cache = EvaluationCache::default();
    for (index, (_, key)) in before.iter().enumerate() {
        cache.insert(address(index, *key), index);
    }
    for index in 0..4 {
        assert_ne!(before[index].1, after[index].1);
        assert!(cache.get(&address(index, after[index].1)).is_none());
    }
}

#[test]
fn cache_repeat_is_deterministic_and_negative_zero_is_canonical() {
    let positive = keys(&snapshot(1, 7, 0.0));
    let negative = keys(&snapshot(2, 7, -0.0));
    assert_eq!(positive, negative);
    for _ in 0..100 {
        assert_eq!(positive, keys(&snapshot(999, 7, 0.0)));
    }
}

#[test]
fn cache_replacement_and_clear_release_owning_leases() {
    use std::sync::Arc;
    let key = address(3, keys(&snapshot(1, 7, 1.0))[3].1);
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let mut cache = EvaluationCache::default();
    cache.insert(key, lease);
    assert_eq!(weak.strong_count(), 1);
    cache.insert(key, Arc::new(()));
    assert!(weak.upgrade().is_none());
    let last = Arc::clone(cache.get(&key).unwrap());
    let last_weak = Arc::downgrade(&last);
    drop(last);
    cache.clear();
    assert!(cache.is_empty());
    assert!(last_weak.upgrade().is_none());
}
