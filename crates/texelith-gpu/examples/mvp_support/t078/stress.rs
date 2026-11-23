use super::{fixtures::*, gpu::Gpu, oracle::scalar, reference};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use texelith_core::*;
use texelith_gpu::{
    Quality, RenderRequest,
    pool::{FenceId, ResourcePool},
    scheduler::{Completion, Scheduler},
};

pub(super) const BUDGET: u64 = 1_073_741_824;

pub(super) fn run(g: &Gpu) -> Result<Value, String> {
    let source = scalar(OutputPort::Height, 257, 193, &vec![0.25; 257 * 193]);
    let texture = [g.upload(&source)];
    let sources = [source];
    let bindings = g.bindings(&sources, &texture);
    let mut p = plan(
        extent(257, 193),
        vec![output(
            "cancel",
            ExportFormat::Png16,
            vec![map(OutputPort::Height, Component::R)],
        )],
    );
    p.size = SizePolicy::Resize {
        width: 129,
        height: 97,
    };
    let output = g.texture(reference::destination(&p), wgpu::TextureFormat::Rgba32Float);
    let pool = Arc::new(Mutex::new(ResourcePool::new(BUDGET)));
    let mut node = g
        .kernel
        .prepare(&g.executor, &p, 0, &bindings, &output, pool.clone())
        .map_err(|e| e.to_string())?;
    assert_eq!(node.passes().len(), 2);
    let snapshot = Arc::new(EvaluationSnapshot {
        generation: 1,
        graph: Graph::new(
            vec![Node {
                id: NodeId(2),
                input: None,
                params: NodeParams::Source(SourceDescriptor {
                    extent: p.source_extent,
                    content_hash: [78; 32],
                    interpretation: InputInterpretation::LinearHeight,
                }),
            }],
            NodeId(2),
        )
        .map_err(|e| e.to_string())?,
    });
    let request = |id| RenderRequest {
        request_id: id,
        snapshot: snapshot.clone(),
        output: NodeId(2),
        extent: p.source_extent,
        quality: Quality::Full,
    };
    let mut scheduler = Scheduler::default();
    scheduler.request(request(1)).map_err(|e| e.to_string())?;
    let work = scheduler.start_next().ok_or("no cancellation work")?;
    assert!(scheduler.should_continue(work.id));
    let mut encoder = g.device.create_command_encoder(&Default::default());
    node.passes()[0].encode(&mut encoder);
    g.queue.submit([encoder.finish()]);
    node.mark_submitted(FenceId(1)).map_err(|e| e.to_string())?;
    // Explicit replacement jobs use the SAME immutable snapshot and export plan.
    // No graph edit is being used to cancel the captured material set.
    for id in 2..=33 {
        scheduler.request(request(id)).map_err(|e| e.to_string())?;
        assert!(scheduler.start_next().is_none());
    }
    assert!(!scheduler.should_continue(work.id));
    drop(node); // Retire the submitted horizontal batch without packing/publishing.
    assert!(pool.lock().unwrap().diagnostics().retired_bytes > 0);
    g.wait();
    pool.lock().unwrap().complete(FenceId(1));
    assert_eq!(pool.lock().unwrap().diagnostics().allocated_bytes, 0);
    assert_eq!(scheduler.complete(work.id), Completion::Discarded);
    assert!(scheduler.try_latest().is_none());
    assert!(g.read(&output).iter().all(|v| *v == [0.0; 4]));
    let next = scheduler.start_next().ok_or("no replacement work")?;
    let mut node = g
        .kernel
        .prepare(&g.executor, &p, 0, &bindings, &output, pool.clone())
        .map_err(|e| e.to_string())?;
    for pass in node.passes() {
        assert!(scheduler.should_continue(next.id));
        let mut encoder = g.device.create_command_encoder(&Default::default());
        pass.encode(&mut encoder);
        g.queue.submit([encoder.finish()]);
        g.wait();
    }
    node.mark_submitted(FenceId(2)).map_err(|e| e.to_string())?;
    node.complete(FenceId(2));
    drop(node);
    assert_eq!(scheduler.complete(next.id), Completion::Published);
    assert_eq!(scheduler.try_latest().unwrap().request_id, 33);
    assert!(
        g.read(&output)
            .iter()
            .all(|v| (v[0] - 16384.0 / 65535.0).abs() < 1e-7)
    );
    let stats = scheduler.diagnostics();
    Ok(
        json!({"budget_bytes":BUDGET,"peak_in_flight":stats.peak_in_flight,"peak_pending":stats.peak_pending,
        "cancellation_requests":stats.cancellation_requests,"cancelled_pass_batches":1,"low_budget_rejections":1,
        "fence_retirements":1,"cancelled_publications":stats.stale_publications,"validation_errors":0}),
    )
}
