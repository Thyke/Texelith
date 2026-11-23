//! Compile-time ADR 0003 boundary. No adapter or second device is created.
use std::{sync::Arc, time::Instant};
use texelith_core::{EvaluationSnapshot, Extent, NodeId};
use texelith_gpu::{
    CompletedFrame, DisplayLease, GpuConfig, GpuError, GpuWorker, Quality, RenderRequest,
    SourceUpload,
};

#[test]
fn scheduler_worker_accepts_eframe_shared_handles_and_owned_immutable_requests() {
    let _spawn: fn(Arc<wgpu::Device>, Arc<wgpu::Queue>, GpuConfig) -> Result<GpuWorker, GpuError> =
        GpuWorker::spawn;
    let _request: fn(&GpuWorker, RenderRequest) -> Result<(), GpuError> = GpuWorker::request;
    let _receive: fn(&GpuWorker) -> Option<CompletedFrame> = GpuWorker::try_latest;
    let _source: fn(&GpuWorker, SourceUpload) -> Result<(), GpuError> = GpuWorker::install_source;
    let _close: fn(&GpuWorker) = GpuWorker::close;
    let _save: fn(&GpuWorker, RenderRequest) -> Result<u64, texelith_gpu::SaveError> =
        GpuWorker::request_save;
    let _save_receive: fn(
        &GpuWorker,
    ) -> Option<
        Result<texelith_gpu::CompletedReadback, texelith_gpu::SaveError>,
    > = GpuWorker::try_save;
    fn send<T: Send>() {}
    fn send_sync<T: Send + Sync>() {}
    fn cloneable<T: Clone>() {}
    send::<RenderRequest>();
    send::<CompletedFrame>();
    send::<SourceUpload>();
    send_sync::<GpuWorker>();
    send_sync::<DisplayLease>();
    cloneable::<DisplayLease>();
    let _shape =
        |request: RenderRequest| -> (u64, Arc<EvaluationSnapshot>, NodeId, Extent, Quality) {
            (
                request.request_id,
                request.snapshot,
                request.output,
                request.extent,
                request.quality,
            )
        };
    let _timing = |frame: &CompletedFrame| -> (Instant, Instant, Instant) {
        (frame.requested_at, frame.submitted_at, frame.completed_at)
    };
    let _view: fn(&DisplayLease) -> &wgpu::TextureView = DisplayLease::view;
    let _texture: fn(&DisplayLease) -> &wgpu::Texture = DisplayLease::texture;
}

#[test]
fn pool_worker_default_hard_budget_is_one_gib() {
    assert_eq!(GpuConfig::default().budget_bytes, 1024 * 1024 * 1024);
}
