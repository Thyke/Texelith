//! Render identity is independent of graph generation (release and view changes
//! can request a different result without changing any parameter).
use std::sync::Arc;
use texelith_core::{EvaluationSnapshot, Extent, NodeId, NodeParams};
use texelith_gpu::{CompletedFrame, Quality, RenderRequest};

#[derive(Default)]
pub struct RenderRouting {
    next: u64,
    pub latest: Option<RenderRequest>,
}

impl RenderRouting {
    pub fn issue(
        &mut self,
        snapshot: EvaluationSnapshot,
        output: NodeId,
        quality: Quality,
    ) -> Result<RenderRequest, String> {
        let source = snapshot
            .graph
            .nodes()
            .iter()
            .find_map(|node| match &node.params {
                NodeParams::Source(source) => Some(source.extent),
                _ => None,
            })
            .ok_or("missing source")?;
        let extent = match quality {
            Quality::Full => source,
            Quality::Proxy => texelith_gpu::proxy_extent(source).map_err(|e| e.to_string())?,
        };
        self.next = self
            .next
            .checked_add(1)
            .ok_or("render request IDs exhausted")?;
        let request = RenderRequest {
            request_id: self.next,
            snapshot: Arc::new(snapshot),
            output,
            extent,
            quality,
        };
        self.latest = Some(request.clone());
        Ok(request)
    }

    pub fn matches(
        &self,
        id: u64,
        generation: u64,
        output: NodeId,
        extent: Extent,
        quality: Quality,
    ) -> bool {
        self.latest.as_ref().is_some_and(|r| {
            r.request_id == id
                && r.snapshot.generation == generation
                && r.output == output
                && r.extent == extent
                && r.quality == quality
        })
    }

    pub fn accepts(&self, frame: &CompletedFrame) -> bool {
        self.matches(
            frame.request_id,
            frame.generation,
            frame.output,
            frame.extent,
            frame.quality,
        )
    }
}
