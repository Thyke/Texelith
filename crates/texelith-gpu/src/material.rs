//! Validated material requests and a fence-by-fence live-resource plan.
use crate::{
    DisplayLease, GpuError, Quality,
    builtin::{is_color_source, source_descriptor},
    worker::node_registration,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use texelith_core::{
    CacheKey, CacheRevisions, EvaluationSnapshot, Extent, MapKind, MaterialOutputs, NodeId,
    NormalConvention, OutputRef, material_node_cache_key,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobId(pub u64);

#[derive(Clone, Debug)]
pub struct MaterialRequest {
    pub job_id: JobId,
    pub request_id: u64,
    pub snapshot: Arc<EvaluationSnapshot>,
    pub outputs: MaterialOutputs,
    pub extent: Extent,
    pub quality: Quality,
}

pub struct CompletedMaterial {
    pub job_id: JobId,
    pub request_id: u64,
    pub generation: u64,
    pub extent: Extent,
    pub quality: Quality,
    pub normal_convention: Option<NormalConvention>,
    /// Drop after submitting the final draw referencing these immutable maps.
    pub outputs: BTreeMap<MapKind, DisplayLease>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceReservations {
    pub viewport_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterialError {
    BudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
    Unsupported {
        node: NodeId,
    },
    InvalidRequest(String),
    Overflow,
}
impl std::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MaterialError {}
impl From<texelith_core::CoreError> for MaterialError {
    fn from(e: texelith_core::CoreError) -> Self {
        Self::InvalidRequest(e.to_string())
    }
}
impl From<MaterialError> for GpuError {
    fn from(e: MaterialError) -> Self {
        Self(e.to_string())
    }
}

impl MaterialRequest {
    pub(crate) fn validate(&self) -> Result<(), MaterialError> {
        self.extent.validate()?;
        if self.outputs.is_empty() || !self.snapshot.graph.is_material() {
            return Err(MaterialError::InvalidRequest(
                "nonempty material outputs required".into(),
            ));
        }
        for (kind, output) in &self.outputs {
            if output.port.map_kind() != *kind {
                return Err(MaterialError::InvalidRequest("output kind mismatch".into()));
            }
            let source = self.snapshot.graph.material_output_extent(*output)?;
            let expected = match self.quality {
                Quality::Full => source,
                Quality::Proxy => crate::proxy_extent(source)
                    .map_err(|e| MaterialError::InvalidRequest(e.to_string()))?,
            };
            if self.extent != expected {
                return Err(MaterialError::InvalidRequest(
                    "extent does not match quality".into(),
                ));
            }
        }
        Ok(())
    }
}

// Local ordering keeps the core DTO/API unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputKey(pub OutputRef);
impl From<OutputRef> for OutputKey {
    fn from(output: OutputRef) -> Self {
        Self(output)
    }
}
impl std::ops::Deref for OutputKey {
    type Target = OutputRef;
    fn deref(&self) -> &OutputRef {
        &self.0
    }
}
impl Ord for OutputKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.node, self.port.map_kind()).cmp(&(other.node, other.port.map_kind()))
    }
}
impl PartialOrd for OutputKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct MaterialPlan {
    peak: u64,
    outputs: BTreeMap<MapKind, CacheKey>,
    pub(crate) nodes: Vec<OutputKey>,
    pub(crate) keys: BTreeMap<OutputKey, CacheKey>,
    pub(crate) last_use: BTreeMap<OutputKey, usize>,
}
fn add(a: u64, b: u64) -> Result<u64, MaterialError> {
    a.checked_add(b).ok_or(MaterialError::Overflow)
}
pub(crate) fn map_bytes(kind: MapKind, extent: Extent) -> u64 {
    u64::from(extent.width)
        * u64::from(extent.height)
        * if matches!(kind, MapKind::Normal | MapKind::Albedo) {
            8
        } else {
            4
        }
}
// Visit named outputs, not just node IDs: secondary outputs have their own
// cache identities, payload reservations and last consumers.
fn required_outputs(request: &MaterialRequest) -> Vec<OutputKey> {
    let graph = &request.snapshot.graph;
    let mut needed = BTreeSet::new();
    let mut stack: Vec<_> = request
        .outputs
        .values()
        .copied()
        .map(OutputKey::from)
        .collect();
    while let Some(output) = stack.pop() {
        if needed.insert(output) {
            let node = graph
                .material_node(output.node)
                .expect("validated output ancestry");
            stack.extend(node.inputs.iter().map(|edge| OutputKey::from(edge.source)));
        }
    }
    graph
        .topological_material_nodes()
        .into_iter()
        .flat_map(|node| {
            needed
                .iter()
                .filter(move |output| output.node == node.id)
                .copied()
        })
        .collect()
}
impl MaterialPlan {
    pub fn new(
        request: &MaterialRequest,
        budget_bytes: u64,
        reservations: ResourceReservations,
    ) -> Result<Self, MaterialError> {
        request.validate()?;
        let graph = &request.snapshot.graph;
        let nodes = required_outputs(request);
        let mut keys = BTreeMap::new();
        let mut last_use = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        // Installed sources remain resident. Conservatively allow the largest typed
        // upload representation (RGBA16); actual registry occupancy is charged by the pool.
        let mut sources = BTreeMap::new();
        for (index, output) in nodes.iter().enumerate() {
            let node = graph.material_node(output.node).expect("planned node");
            if node_registration(node).is_none() {
                return Err(MaterialError::Unsupported { node: node.id });
            }
            let extent = graph.material_output_extent(output.0)?;
            let inputs: Vec<_> = node
                .inputs
                .iter()
                .map(|e| keys[&OutputKey::from(e.source)])
                .collect();
            keys.insert(
                *output,
                material_node_cache_key(
                    node,
                    output.port,
                    &inputs,
                    request.extent,
                    extent,
                    CacheRevisions::CURRENT,
                )?,
            );
            last_use.insert(*output, index);
            for edge in &node.inputs {
                last_use.insert(edge.source.into(), index);
            }
            sizes.insert(
                *output,
                map_bytes(
                    output.port.map_kind(),
                    if is_color_source(node) {
                        extent
                    } else {
                        request.extent
                    },
                ),
            );
            if let Some(source) = source_descriptor(node) {
                sources.insert(
                    source.content_hash,
                    map_bytes(output.port.map_kind(), source.extent),
                );
            }
        }
        for output in request.outputs.values() {
            last_use.insert((*output).into(), nodes.len());
        }
        let source_bytes = sources.values().try_fold(0u64, |sum, n| add(sum, *n))?;
        let fixed = add(source_bytes, reservations.viewport_bytes)?;
        let mut live = 0u64;
        let mut peak = fixed;
        for (index, output) in nodes.iter().enumerate() {
            let node = graph.material_node(output.node).expect("planned node");
            live = add(live, sizes[output])?;
            let scratch = (node_registration(node)
                .expect("validated registration")
                .scratch)(node, request.extent)?;
            // Source conversion at full resolution precedes the proxy downsample.
            let full = if source_descriptor(node).is_some() && request.quality == Quality::Proxy {
                map_bytes(
                    output.port.map_kind(),
                    graph.material_output_extent(output.0)?,
                )
            } else {
                0
            };
            peak = peak.max(add(add(add(fixed, live)?, scratch)?, full)?);
            for (id, end) in &last_use {
                if *end == index {
                    live -= sizes[id];
                }
            }
        }
        // FAZ1_MATERIAL_SCRATCH: T-069 public color restriction retains the
        // canonical full input alongside its admitted half-float proxy.
        if request.quality == Quality::Proxy && request.outputs.contains_key(&MapKind::Albedo) {
            peak = peak.max(add(
                add(fixed, live)?,
                map_bytes(MapKind::Albedo, request.extent),
            )?);
        }
        // Readback is a separate phase after scratch retirement.
        peak = peak.max(add(add(fixed, live)?, reservations.readback_bytes)?);
        if peak > budget_bytes {
            return Err(MaterialError::BudgetExceeded {
                required_bytes: peak,
                budget_bytes,
            });
        }
        let outputs = request
            .outputs
            .iter()
            .map(|(k, o)| (*k, keys[&OutputKey::from(*o)]))
            .collect();
        Ok(Self {
            peak,
            outputs,
            nodes,
            keys,
            last_use,
        })
    }
    pub fn peak_bytes(&self) -> u64 {
        self.peak
    }
    pub fn output_keys(&self) -> &BTreeMap<MapKind, CacheKey> {
        &self.outputs
    }
}

#[cfg(test)]
mod registry_plan_tests {
    use super::*;
    use texelith_core::{InputInterpretation, SourceDescriptor, material_from_source};

    #[test]
    fn secondary_ports_are_distinct_and_share_their_ancestor() {
        let extent = Extent::new(17, 13).unwrap();
        let graph = material_from_source(SourceDescriptor {
            content_hash: [121; 32],
            extent,
            interpretation: InputInterpretation::LinearNormal,
        })
        .unwrap();
        let outputs: MaterialOutputs = graph
            .material_outputs()
            .unwrap()
            .iter()
            .filter(|(kind, _)| {
                matches!(kind, MapKind::Curvature | MapKind::Edge | MapKind::Cavity)
            })
            .map(|(kind, output)| (*kind, *output))
            .collect();
        assert_eq!(outputs.len(), 3);
        let request = MaterialRequest {
            job_id: JobId(121),
            request_id: 1,
            snapshot: Arc::new(EvaluationSnapshot {
                generation: 1,
                graph,
            }),
            outputs,
            extent,
            quality: Quality::Full,
        };
        let planned = required_outputs(&request);
        assert_eq!(planned.len(), 4);
        assert_eq!(planned[0].port.map_kind(), MapKind::Normal);
        assert_eq!(planned[1].node, planned[2].node);
        assert_eq!(planned[2].node, planned[3].node);
        let mut port_keys = BTreeMap::new();
        for output in request.outputs.values() {
            port_keys.insert(OutputKey::from(*output), output.port);
        }
        assert_eq!(
            port_keys.len(),
            3,
            "secondary ports must not alias in caches or liveness"
        );
        for output in request.outputs.values() {
            assert!(planned.contains(&(*output).into()));
        }
    }
}
