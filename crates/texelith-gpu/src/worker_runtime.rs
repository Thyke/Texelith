use super::*;
use crate::{
    Axis, GpuExecutor, Pass,
    cache::{CacheAddress, EvaluationCache, evaluation_keys},
    pool::{AllocationDesc, AllocationId, FenceId, PoolError},
    resources::{BindingKey, Resource},
    scheduler::{Completion, Work, WorkKind},
};
use crossbeam_channel::Receiver;
use std::{
    collections::HashMap,
    future::Future,
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};
use texelith_core::{CacheRevisions, NodeParams};
#[path = "export_readback/runtime.rs"]
mod export_runtime;
#[path = "worker_material.rs"]
mod material_runtime;
#[path = "viewport_resources/runtime.rs"]
mod viewport_runtime;
// FAZ1_GPU_RUNTIME_MODULES: T-065 begin
#[path = "pbr/runtime.rs"]
mod pbr_runtime;
#[path = "pbr/mips.rs"]
mod preview_mips;
// FAZ1_GPU_RUNTIME_MODULES: T-065 end
// FAZ1_GPU_RUNTIME_MODULES: T-067 begin
#[path = "ibl/runtime.rs"]
mod ibl_runtime;
// FAZ1_GPU_RUNTIME_MODULES: T-067 end
// FAZ1_GPU_RUNTIME_MODULES: T-069
#[path = "compare/runtime.rs"]
mod compare_runtime;
// Runtime adapters are children so allocation/submission internals stay private.
macro_rules! register_nodes {
    ($( $visibility:vis $module:ident { $($export:ident),* } [$($state:ident, $runtime:literal)?] [$($suite:literal, $runner:literal, [$($validator:ident, $validation:literal)?])?]; )*) => {
        $($(#[path = $runtime] mod $state;) ?)*
        #[derive(Default)]
        struct NodeStates {
            $($($state: $state::State,) ?)*
        }
        impl NodeStates {
            fn evict_one(&mut self) -> bool {
                $($(if self.$state.evict() { return true; }) ?)*
                false
            }
        }
        const NODE_REGISTRATIONS: &[NodeRegistration] = &[
            $($(NodeRegistration {
                supports: $state::supports,
                scratch: $state::scratch,
                evaluate: $state::evaluate,
            },) ?)*
        ];
    };
}
crate::gpu_nodes!(register_nodes);

pub(crate) struct NodeRegistration {
    supports: fn(&texelith_core::MaterialNode) -> bool,
    pub(crate) scratch:
        fn(&texelith_core::MaterialNode, Extent) -> std::result::Result<u64, crate::MaterialError>,
    evaluate: fn(&mut Runtime, &NodeContext<'_>) -> Result<Lease>,
}
pub(crate) fn node_registration(
    node: &texelith_core::MaterialNode,
) -> Option<&'static NodeRegistration> {
    NODE_REGISTRATIONS
        .iter()
        .find(|entry| (entry.supports)(node))
}

struct NodeContext<'a> {
    work: &'a Work,
    node: &'a texelith_core::MaterialNode,
    output: texelith_core::OutputRef,
    plan: &'a crate::MaterialPlan,
    extent: Extent,
    source_extent: Extent,
    live: &'a std::collections::BTreeMap<crate::material::OutputKey, Lease>,
}
impl NodeContext<'_> {
    fn input(&self) -> Lease {
        self.live[&self.node.inputs[0].source.into()].clone()
    }
    fn input_key(&self) -> texelith_core::CacheKey {
        self.plan.keys[&self.node.inputs[0].source.into()]
    }
    fn input_at(&self, port: texelith_core::InputPort) -> Option<Lease> {
        self.node
            .inputs
            .iter()
            .find(|edge| edge.port == port)
            .map(|edge| self.live[&edge.source.into()].clone())
    }
}

struct Owner {
    id: AllocationId,
    resource: Resource,
    pool: Arc<Mutex<ResourcePool>>,
    transient: bool,
}
impl Drop for Owner {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        let _ = if self.transient {
            pool.release_transient(self.id)
        } else {
            pool.release(self.id)
        };
    }
}
#[derive(Clone)]
struct Lease(Arc<Owner>);
impl Lease {
    fn texture(&self) -> &wgpu::Texture {
        match &self.0.resource {
            Resource::Texture { texture, .. } => texture,
            _ => unreachable!("texture plan"),
        }
    }
    fn view(&self) -> &wgpu::TextureView {
        match &self.0.resource {
            Resource::Texture { view, .. } => view,
            _ => unreachable!("texture plan"),
        }
    }
    fn buffer(&self) -> &wgpu::Buffer {
        match &self.0.resource {
            Resource::Buffer(buffer) => buffer,
            _ => unreachable!("buffer plan"),
        }
    }
}
struct DisplayOwner {
    lease: Option<Lease>,
    queue: Arc<wgpu::Queue>,
}
impl Drop for DisplayOwner {
    fn drop(&mut self) {
        // The last UI owner must be dropped AFTER unregistering the TextureId
        // and submitting the final referencing frame. Keep the allocation pinned
        // until a real queue callback, even if the worker has already closed.
        let lease = self.lease.take();
        if let Some(lease) = &lease {
            lease.0.pool.lock().unwrap().ui_retiring(lease.0.id, true);
        }
        self.queue.on_submitted_work_done(move || {
            if let Some(lease) = &lease {
                lease.0.pool.lock().unwrap().ui_retiring(lease.0.id, false);
            }
            drop(lease);
        });
    }
}
/// Immutable display allocation. Clones retain the same owning lease. Unregister
/// the native egui texture and submit its final frame before dropping the last
/// clone. Drop establishes a queue completion fence; no frame delay is assumed.
#[derive(Clone)]
pub struct DisplayLease(Arc<DisplayOwner>);

/// Weak provenance prevents forged or mixed bundles without pinning old maps.
pub(super) struct MaterialIdentity {
    pub(super) recipe: MaterialRequest,
    job: crate::JobId,
    generation: u64,
    request: u64,
    extent: Extent,
    quality: Quality,
    normal: Option<texelith_core::NormalConvention>,
    outputs: std::collections::BTreeMap<MapKind, std::sync::Weak<DisplayOwner>>,
}
impl MaterialIdentity {
    fn new(bundle: &CompletedMaterial, request: MaterialRequest) -> Self {
        Self {
            recipe: request,
            job: bundle.job_id,
            generation: bundle.generation,
            request: bundle.request_id,
            extent: bundle.extent,
            quality: bundle.quality,
            normal: bundle.normal_convention,
            outputs: bundle
                .outputs
                .iter()
                .map(|(kind, lease)| (*kind, Arc::downgrade(&lease.0)))
                .collect(),
        }
    }
    pub(super) fn matches(&self, bundle: &CompletedMaterial) -> bool {
        self.job == bundle.job_id
            && self.generation == bundle.generation
            && self.request == bundle.request_id
            && self.extent == bundle.extent
            && self.quality == bundle.quality
            && self.normal == bundle.normal_convention
            && self.outputs.len() == bundle.outputs.len()
            && self.outputs.iter().all(|(kind, owner)| {
                bundle
                    .outputs
                    .get(kind)
                    .is_some_and(|lease| owner.ptr_eq(&Arc::downgrade(&lease.0)))
            })
    }
}
impl DisplayLease {
    pub fn texture(&self) -> &wgpu::Texture {
        self.0.lease.as_ref().expect("live lease").texture()
    }
    pub fn view(&self) -> &wgpu::TextureView {
        self.0.lease.as_ref().expect("live lease").view()
    }
    /// Consume this UI owner at the verified after-render boundary. Other clones
    /// remain pinned; the final owner establishes the retirement callback.
    pub fn retire_after_submission(self) {
        drop(self);
    }
}

#[derive(Debug)]
enum Failure {
    Gpu(GpuError),
    Pool(PoolError),
    Cancelled,
}
impl From<GpuError> for Failure {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}
impl From<texelith_core::CoreError> for Failure {
    fn from(e: texelith_core::CoreError) -> Self {
        Self::Gpu(e.into())
    }
}
impl From<PoolError> for Failure {
    fn from(e: PoolError) -> Self {
        Self::Pool(e)
    }
}
type Result<T> = std::result::Result<T, Failure>;

#[derive(Clone)]
enum Reference {
    Existing(Lease),
    Planned(usize),
}
impl Reference {
    fn resolve(&self, leases: &[Lease]) -> Lease {
        match self {
            Self::Existing(lease) => lease.clone(),
            Self::Planned(index) => leases[*index].clone(),
        }
    }
}
struct PlannedPass {
    operation: Pass,
    input: Reference,
    output: Reference,
    source: Extent,
    uniform: usize,
    weights: Option<usize>,
}
#[derive(Default)]
struct Plan {
    allocations: Vec<AllocationDesc>,
    passes: Vec<PlannedPass>,
    cache: Vec<(CacheAddress, Reference)>,
    output: Option<Reference>,
    readback: Option<usize>,
    hits: u64,
    misses: u64,
}
impl Plan {
    fn allocate(&mut self, desc: AllocationDesc) -> usize {
        let i = self.allocations.len();
        self.allocations.push(desc);
        i
    }
    fn texture(&mut self, extent: Extent, format: wgpu::TextureFormat) -> Reference {
        Reference::Planned(self.allocate(texture_desc(extent, format)))
    }
    fn pass(&mut self, operation: Pass, input: Reference, output: Reference, source: Extent) {
        let uniform = self.allocate(AllocationDesc::Buffer { bytes: 64 });
        let weights = matches!(operation, Pass::Blur { .. })
            .then(|| self.allocate(AllocationDesc::Buffer { bytes: 128 }));
        self.passes.push(PlannedPass {
            operation,
            input,
            output,
            source,
            uniform,
            weights,
        });
    }
}
fn texture_desc(extent: Extent, format: wgpu::TextureFormat) -> AllocationDesc {
    AllocationDesc::Texture {
        extent,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    }
}

struct Runtime {
    export: Option<export_runtime::State>,
    export_kernel: crate::ExportPackingKernel,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    executor: GpuExecutor,
    config: GpuConfig,
    shared: Arc<Mutex<Mailboxes>>,
    pool: Arc<Mutex<ResourcePool>>,
    wake: Receiver<()>,
    cache: EvaluationCache<Lease>,
    sources: Vec<(
        SourceDescriptor,
        Lease,
        Option<texelith_core::NormalConvention>,
    )>,
    viewport_reservation: Option<DisplayLease>,
    viewport_bytes: u64,
    readback_reservation: Option<DisplayLease>,
    nodes: NodeStates,
    viewport: Option<crate::viewport_shared::ViewportRenderer>,
    preview_mips: preview_mips::State,
    // FAZ1_GPU_RUNTIME_FIELDS: T-069
    compare_proxy: Option<wgpu::ComputePipeline>,
    compare_mesh: Option<(Arc<texelith_core::MeshData>, crate::pbr::PackedMesh)>,
    // FAZ1_GPU_RUNTIME_FIELDS: T-065 CPU-only immutable primitive cache
    pbr_meshes: Option<crate::pbr::Meshes>,
    // FAZ1_GPU_RUNTIME_FIELDS: T-067 begin
    ibl: Option<ibl_runtime::CachedEnvironment>,
    ibl_kernel: Option<crate::ibl::IblKernel>,
    // FAZ1_GPU_RUNTIME_FIELDS: T-067 end
    fence: u64,
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.preview_mips.cached = None;
        self.export = None;
        self.nodes = NodeStates::default();
        self.cache.clear();
        self.sources.clear();
        self.viewport_reservation = None;
        self.readback_reservation = None;
        {
            let mut state = self.shared.lock().unwrap();
            state.viewport_reserved_bytes = 0;
            state.readback_reserved_bytes = 0;
        }
        self.pool.lock().unwrap().shutdown();
    }
}

struct Evaluated {
    output: Lease,
    pixels: Option<Vec<u8>>,
    kind: MapKind,
    submitted_at: Instant,
    completed_at: Instant,
}

pub(super) fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: GpuConfig,
    shared: Arc<Mutex<Mailboxes>>,
    pool: Arc<Mutex<ResourcePool>>,
    wake: Receiver<()>,
) {
    let scopes = error_scopes(&device);
    let initialized = drive_future(&device, GpuExecutor::new(device.clone(), queue.clone()));
    let executor = match finish_scopes(&device, &shared, scopes, initialized.map_err(Failure::Gpu))
    {
        Ok(executor) => executor,
        Err(error) => {
            shared.lock().unwrap().scheduler.close();
            report_failure(&shared, error);
            return;
        }
    };
    let mut runtime = Runtime {
        export: None,
        export_kernel: crate::ExportPackingKernel::default(),
        device,
        queue,
        executor,
        config,
        shared,
        pool,
        wake,
        cache: EvaluationCache::default(),
        sources: Vec::new(),
        viewport_reservation: None,
        viewport_bytes: 0,
        readback_reservation: None,
        nodes: NodeStates::default(),
        viewport: None,
        preview_mips: Default::default(),
        // FAZ1_GPU_RUNTIME_INIT: T-069
        compare_proxy: None,
        compare_mesh: None,
        // FAZ1_GPU_RUNTIME_INIT: T-065 lazy primitive cache
        pbr_meshes: None,
        // FAZ1_GPU_RUNTIME_INIT: T-067 begin
        ibl: None,
        ibl_kernel: None,
        // FAZ1_GPU_RUNTIME_INIT: T-067 end
        fence: 0,
    };
    loop {
        if runtime.shared.lock().unwrap().scheduler.closed() {
            break;
        }
        runtime.retire_export();
        let clear_results = runtime.shared.lock().unwrap().clear_results.take();
        if let Some(done) = clear_results {
            runtime.preview_mips.cached = None;
            runtime.cache.clear();
            while runtime.nodes.evict_one() {}
            let mut state = runtime.shared.lock().unwrap();
            state.frame = None;
            state.material = None;
            state.preview = None;
            state.material_identity = None;
            state.retained_materials.clear();
            let _ = done.try_send(());
        }
        let upload = runtime.shared.lock().unwrap().upload.take();
        if let Some(upload) = upload {
            let hash = upload.descriptor.content_hash;
            let outcome = runtime.upload(upload);
            {
                let mut state = runtime.shared.lock().unwrap();
                state.uploading = false;
                if outcome.is_ok() && !state.scheduler.closed() {
                    state.ready = Some(hash);
                }
            }
            if let Err(error) = outcome {
                report_failure(&runtime.shared, error);
            }
            continue;
        }
        let work = runtime.shared.lock().unwrap().scheduler.start_next();
        if let Some(work) = work {
            let result = if work.preview.is_some() {
                runtime.evaluate_preview(&work)
            } else if work.material.is_some() {
                runtime.evaluate_material(&work)
            } else {
                runtime.evaluate(&work)
            };
            if let Err(error) = result {
                {
                    let mut state = runtime.shared.lock().unwrap();
                    match &error {
                        Failure::Cancelled => {
                            state.scheduler.cancel_active(work.id);
                            state.cancelled_pass_batches += 1;
                        }
                        Failure::Gpu(error) => {
                            state.scheduler.fail_active(work.id, error.to_string());
                        }
                        Failure::Pool(error) => {
                            state.scheduler.fail_active(work.id, error.to_string());
                        }
                    }
                }
                if !matches!(error, Failure::Cancelled) {
                    report_failure(&runtime.shared, error);
                }
            }
            continue;
        }
        if runtime.export_step() {
            continue;
        }
        if let Err(error) = runtime.device.poll(wgpu::PollType::Poll) {
            runtime.shared.lock().unwrap().scheduler.close();
            report_failure(&runtime.shared, Failure::Gpu(GpuError(error.to_string())));
        }
        let _ = runtime.wake.recv_timeout(Duration::from_millis(1));
    }
    // Submitted work is retired inside evaluate/upload before reaching here.
    // External display leases can outlive this worker and keep their accounting.
}

fn report_failure(shared: &Arc<Mutex<Mailboxes>>, failure: Failure) {
    let mut state = shared.lock().unwrap();
    if state.device_lost {
        return;
    }
    if state.scheduler.closed() && state.export.request.is_some() {
        state
            .export
            .progress(crate::ExportStage::Failed, Some(format!("{failure:?}")));
        state.export.request = None;
        state.export.plane = None;
    }
    let event = match failure {
        Failure::Pool(PoolError::BudgetExceeded {
            budget_bytes,
            required_bytes,
        }) => WorkerEvent::BudgetExceeded {
            budget_bytes,
            required_bytes,
        },
        Failure::Pool(error) => WorkerEvent::Error(GpuError(error.to_string())),
        Failure::Gpu(error) => WorkerEvent::Error(error),
        Failure::Cancelled => return,
    };
    state.event = Some(event);
    // A request/source admission error must not consume an unrelated pending
    // upload or permanently close the worker. Fatal device paths close explicitly.
}

struct ThreadWake(std::thread::Thread);
fn error_scopes(device: &wgpu::Device) -> Vec<(wgpu::ErrorFilter, wgpu::ErrorScopeGuard)> {
    [
        wgpu::ErrorFilter::OutOfMemory,
        wgpu::ErrorFilter::Internal,
        wgpu::ErrorFilter::Validation,
    ]
    .into_iter()
    .map(|filter| (filter, device.push_error_scope(filter)))
    .collect()
}
fn finish_scopes<T>(
    device: &wgpu::Device,
    shared: &Mutex<Mailboxes>,
    scopes: Vec<(wgpu::ErrorFilter, wgpu::ErrorScopeGuard)>,
    mut result: Result<T>,
) -> Result<T> {
    for (filter, scope) in scopes.into_iter().rev() {
        if let Some(error) = drive_future(device, scope.pop()) {
            if filter == wgpu::ErrorFilter::Validation {
                shared.lock().unwrap().validation_errors += 1;
            }
            result = Err(GpuError(error.to_string()).into());
        }
    }
    result
}
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
// Only used on the worker, including shader initialization and error scopes.
fn drive_future<T>(device: &wgpu::Device, future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

impl Runtime {
    fn current(&self, work: &Work) -> bool {
        self.shared.lock().unwrap().scheduler.is_current(work)
    }
    fn allocate(&mut self, descriptions: &[AllocationDesc]) -> Result<Vec<Lease>> {
        let resources = loop {
            let outcome = self
                .pool
                .lock()
                .unwrap()
                .allocate_backed(&self.device, descriptions);
            match outcome {
                Ok(resources) => break resources,
                Err(PoolError::BudgetExceeded { .. }) if self.cache.evict_oldest() => {
                    self.shared.lock().unwrap().cache_evictions += 1;
                    continue;
                }
                Err(PoolError::BudgetExceeded { .. }) if self.nodes.evict_one() => {
                    continue;
                }
                Err(PoolError::BudgetExceeded { .. }) if self.ibl.take().is_some() => continue, // T-067 evictable cache
                Err(PoolError::BudgetExceeded { .. })
                    if self.preview_mips.cached.take().is_some() =>
                {
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        };
        Ok(resources
            .into_iter()
            .map(|(id, resource)| {
                Lease(Arc::new(Owner {
                    id,
                    resource,
                    pool: self.pool.clone(),
                    transient: false,
                }))
            })
            .collect())
    }

    fn submit(
        &mut self,
        encoder: wgpu::CommandEncoder,
        leases: &[Lease],
    ) -> Result<(FenceId, Receiver<()>, Instant)> {
        self.fence = self
            .fence
            .checked_add(1)
            .ok_or_else(|| GpuError("queue fence exhausted".into()))?;
        let fence = FenceId(self.fence);
        {
            let mut pool = self.pool.lock().unwrap();
            for lease in leases {
                pool.mark_submitted(lease.0.id, fence)?;
            }
        }
        let submitted_at = Instant::now();
        self.queue.submit([encoder.finish()]);
        self.shared.lock().unwrap().queue_submit_ns += submitted_at.elapsed().as_nanos() as u64;
        let (tx, rx) = bounded(1);
        let pool = self.pool.clone();
        self.queue.on_submitted_work_done(move || {
            pool.lock().unwrap().complete(fence);
            let _ = tx.try_send(());
        });
        Ok((fence, rx, submitted_at))
    }
    fn wait<T>(&self, receiver: &Receiver<T>) -> Result<T> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(value) = receiver.try_recv() {
                return Ok(value);
            }
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| GpuError(e.to_string()))?;
            if self.shared.lock().unwrap().device_lost {
                return Err(GpuError("device lost".into()).into());
            }
            if Instant::now() > deadline {
                self.shared.lock().unwrap().scheduler.close();
                return Err(GpuError("GPU completion timeout".into()).into());
            }
            // Callbacks need polling on this worker. Sleeping here can add a
            // coarse Windows timer quantum even to a sub-millisecond dispatch.
            // Yield only while a submission is active; the idle loop blocks.
            std::thread::yield_now();
        }
    }
    fn upload(&mut self, upload: TypedSourceUpload) -> Result<()> {
        let checked = upload.pixels.descriptor(upload.descriptor.extent)?;
        if checked != upload.descriptor {
            return Err(GpuError("source content hash does not match typed bytes".into()).into());
        }
        if self.sources.iter().any(|(s, _, _)| *s == checked) {
            return Ok(());
        }
        let extent = checked.extent;
        let (format, data, convention): (_, Vec<u8>, _) = match &upload.pixels {
            TypedSourcePixels::Rgba8 { pixels, .. } => {
                (wgpu::TextureFormat::Rgba8Unorm, pixels.to_vec(), None)
            }
            TypedSourcePixels::HeightU16(pixels) => (
                wgpu::TextureFormat::R32Float,
                pixels
                    .iter()
                    .flat_map(|v| (f32::from(*v) / 65535.0).to_le_bytes())
                    .collect(),
                None,
            ),
            TypedSourcePixels::HeightF32(pixels) => (
                wgpu::TextureFormat::R32Float,
                pixels.iter().flat_map(|v| v.to_le_bytes()).collect(),
                None,
            ),
            TypedSourcePixels::NormalRgb8 { pixels, convention } => (
                wgpu::TextureFormat::Rgba8Unorm,
                pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|v| [v[0], v[1], v[2], 255])
                    .collect(),
                Some(*convention),
            ),
        };
        // A bounded, content-addressed resident registry. Eviction never substitutes
        // another image: a request for an evicted hash returns a missing-source error.
        const MAX_SOURCE_BYTES: u64 = 128 * 1024 * 1024;
        while self.sources.len() >= 8
            || self
                .sources
                .iter()
                .map(|(s, _, _)| u64::from(s.extent.width) * u64::from(s.extent.height) * 4)
                .sum::<u64>()
                + data.len() as u64
                > MAX_SOURCE_BYTES
        {
            let protected = self.shared.lock().unwrap().export.request.clone();
            let victim = self
                .sources
                .iter()
                .position(|(source, _, _)| {
                    !self.shared.lock().unwrap().scheduler.retains_source(source)
                        && protected
                            .as_ref()
                            .is_none_or(|request| !request.capture.sources.contains(source))
                })
                .ok_or_else(|| {
                    GpuError("source upload Busy: immutable export retains source registry".into())
                })?;
            self.sources.remove(victim);
        }
        let desc = AllocationDesc::Texture {
            extent,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        };
        let staging = AllocationDesc::Readback {
            extent,
            bytes_per_pixel: 4,
        }
        .byte_len()?;
        let scopes = error_scopes(&self.device);
        let result = (|| {
            let leases = self.allocate(&[desc, AllocationDesc::Staging { bytes: staging }])?;
            self.queue.write_texture(
                leases[0].texture().as_image_copy(),
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(extent.width * 4),
                    rows_per_image: Some(extent.height),
                },
                size(extent),
            );
            let encoder = self.device.create_command_encoder(&Default::default());
            let (_, finished, _) = self.submit(encoder, &leases)?;
            self.wait(&finished)?;
            self.sources.push((checked, leases[0].clone(), convention));
            Ok(())
        })();
        finish_scopes(&self.device, &self.shared, scopes, result)
    }
}

fn size(extent: Extent) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: 1,
    }
}

impl Runtime {
    fn cached(&mut self, address: CacheAddress, plan: &mut Plan) -> Option<Reference> {
        if let Some(lease) = self.cache.get(&address) {
            plan.hits += 1;
            Some(Reference::Existing(lease.clone()))
        } else {
            plan.misses += 1;
            None
        }
    }
    fn plan(&mut self, work: &Work) -> Result<Plan> {
        let request = &work.request;
        let graph = &request.snapshot.graph;
        let source = crate::scheduler::source(request)
            .map_err(|e| GpuError(e.to_string()))?
            .clone();
        let (_, rgba, _) = self
            .sources
            .iter()
            .find(|(s, _, _)| *s == source)
            .ok_or_else(|| GpuError("requested source hash is not installed".into()))?;
        if rgba.texture().format() != wgpu::TextureFormat::Rgba8Unorm {
            return Err(GpuError("typed height requires material request".into()).into());
        }
        let rgba = rgba.clone();
        let mut chain = Vec::new();
        let mut node = graph
            .node(request.output)
            .ok_or_else(|| GpuError("missing output".into()))?;
        loop {
            chain.push(node);
            let Some(input) = node.input else {
                break;
            };
            node = graph.node(input).expect("validated graph");
        }
        chain.reverse();
        let keys: HashMap<_, _> =
            evaluation_keys(&request.snapshot, request.extent, CacheRevisions::CURRENT)?
                .into_iter()
                .collect();
        let mut plan = Plan::default();
        let mut previous = None;
        for node in chain {
            let format = if node.params.output_kind() == MapKind::Normal {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            };
            let address = CacheAddress {
                content: keys[&node.id],
                format,
            };
            if let Some(hit) = self.cached(address, &mut plan) {
                previous = Some(hit);
                continue;
            }
            let output = match node.params {
                NodeParams::Source(_) => {
                    if request.extent == source.extent {
                        let output = plan.texture(source.extent, format);
                        plan.pass(
                            Pass::Source(source.interpretation),
                            Reference::Existing(rgba.clone()),
                            output.clone(),
                            source.extent,
                        );
                        output
                    } else {
                        let full_key =
                            texelith_core::node_cache_key(node, &[], source.extent, source.extent)?;
                        let full_address = CacheAddress {
                            content: full_key,
                            format,
                        };
                        let full = if let Some(hit) = self.cached(full_address, &mut plan) {
                            hit
                        } else {
                            let full = plan.texture(source.extent, format);
                            plan.pass(
                                Pass::Source(source.interpretation),
                                Reference::Existing(rgba.clone()),
                                full.clone(),
                                source.extent,
                            );
                            plan.cache.push((full_address, full.clone()));
                            full
                        };
                        let output = plan.texture(request.extent, format);
                        plan.pass(Pass::Downsample, full, output.clone(), source.extent);
                        output
                    }
                }
                NodeParams::Blur {
                    sigma_source_px: 0.0,
                } => previous.clone().expect("validated parent"),
                NodeParams::Blur { sigma_source_px } => {
                    let scratch = plan.texture(request.extent, format);
                    let output = plan.texture(request.extent, format);
                    plan.pass(
                        Pass::Blur {
                            sigma_source_px,
                            axis: Axis::X,
                        },
                        previous.clone().expect("validated parent"),
                        scratch.clone(),
                        source.extent,
                    );
                    plan.pass(
                        Pass::Blur {
                            sigma_source_px,
                            axis: Axis::Y,
                        },
                        scratch,
                        output.clone(),
                        source.extent,
                    );
                    output
                }
                NodeParams::Levels {
                    black,
                    white,
                    gamma,
                } => {
                    let output = plan.texture(request.extent, format);
                    plan.pass(
                        Pass::Levels {
                            black,
                            white,
                            gamma,
                        },
                        previous.clone().expect("validated parent"),
                        output.clone(),
                        source.extent,
                    );
                    output
                }
                NodeParams::NormalFromHeight { strength } => {
                    let output = plan.texture(request.extent, format);
                    plan.pass(
                        Pass::Normal { strength },
                        previous.clone().expect("validated parent"),
                        output.clone(),
                        source.extent,
                    );
                    output
                }
            };
            plan.cache.push((address, output.clone()));
            previous = Some(output);
        }
        let output = previous.expect("validated nonempty chain");
        let kind = graph
            .node(request.output)
            .expect("validated output")
            .params
            .output_kind();
        if work.kind == WorkKind::Interactive {
            let display = plan.texture(request.extent, wgpu::TextureFormat::Rgba8Unorm);
            plan.pass(Pass::Display(kind), output, display.clone(), source.extent);
            plan.output = Some(display);
        } else {
            plan.readback = Some(plan.allocate(AllocationDesc::Readback {
                extent: request.extent,
                bytes_per_pixel: if kind == MapKind::Height { 4 } else { 8 },
            }));
            plan.output = Some(output);
        }
        // wgpu owns queue-write staging; reserve its payload until completion in
        // addition to the persistent uniforms. Backend heap overhead is separate.
        let staged: u64 = plan
            .passes
            .iter()
            .map(|p| if p.weights.is_some() { 192 } else { 64 })
            .sum();
        if staged > 0 {
            plan.allocate(AllocationDesc::Staging { bytes: staged });
        }
        Ok(plan)
    }

    fn evaluate(&mut self, work: &Work) -> Result<()> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let plan = self.plan(work)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let scopes = error_scopes(&self.device);
        let result = self.execute_plan(work, plan);
        let evaluated = finish_scopes(&self.device, &self.shared, scopes, result)?;
        self.publish(work, evaluated);
        Ok(())
    }
    fn execute_plan(&mut self, work: &Work, plan: Plan) -> Result<Evaluated> {
        let leases = self.allocate(&plan.allocations)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Texelith bounded evaluation"),
            });
        let mut used = leases.clone();
        let mut encoded_all = true;
        for pass in &plan.passes {
            if !self.current(work) {
                encoded_all = false;
                break;
            }
            let input = pass.input.resolve(&leases);
            let output = pass.output.resolve(&leases);
            let uniform = &leases[pass.uniform];
            let weights = pass.weights.map(|i| &leases[i]);
            let key = BindingKey {
                input: input.0.id,
                output: output.0.id,
                uniform: uniform.0.id,
                weights: weights.map(|w| w.0.id),
                pipeline: pass.operation.index(),
                source_width: pass.source.width,
                source_height: pass.source.height,
            };
            let mut pool = self.pool.lock().unwrap();
            if let Some(prepared) = pool.backing.bindings.get_mut(&key) {
                prepared.update(pass.operation)?;
            } else {
                let prepared = self.executor.prepare_pooled(
                    pass.operation,
                    input.texture(),
                    output.texture(),
                    pass.source,
                    (uniform.buffer(), weights.map(Lease::buffer)),
                )?;
                pool.backing.bindings.insert(key, prepared);
            }
            pool.backing.bindings[&key].encode(&mut encoder);
            drop(pool);
            used.push(input);
            used.push(output);
        }
        let output = plan
            .output
            .as_ref()
            .expect("planned output")
            .resolve(&leases);
        used.push(output.clone());
        let kind = work
            .request
            .snapshot
            .graph
            .node(work.request.output)
            .expect("validated output")
            .params
            .output_kind();
        let bpp = if kind == MapKind::Height { 4 } else { 8 };
        let padded = (work.request.extent.width * bpp).div_ceil(256) * 256;
        if encoded_all && let Some(index) = plan.readback {
            encoder.copy_texture_to_buffer(
                output.texture().as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: leases[index].buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(work.request.extent.height),
                    },
                },
                size(work.request.extent),
            );
        }
        // Even a cancelled, partially encoded batch must flush queued uniform
        // writes and retire their staging before another batch can start.
        let (fence, finished, submitted_at) = self.submit(encoder, &used)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&finished)?;
        self.pool.lock().unwrap().complete(fence);
        if !encoded_all {
            return Err(Failure::Cancelled);
        }
        for (address, reference) in &plan.cache {
            self.cache.insert(*address, reference.resolve(&leases));
        }
        self.trim_cache();
        {
            let mut state = self.shared.lock().unwrap();
            state.cache_hits += plan.hits;
            state.cache_misses += plan.misses;
        }
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let pixels = if let Some(index) = plan.readback {
            let buffer = leases[index].buffer();
            let (tx, rx) = bounded(1);
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.try_send(result);
                });
            self.wait(&rx)?.map_err(|e| GpuError(e.to_string()))?;
            let converted = (|| {
                let mapped = buffer
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|e| GpuError(e.to_string()))?;
                quantize(&mapped, work.request.extent, kind, padded)
            })();
            buffer.unmap();
            Some(converted?)
        } else {
            None
        };
        let completed_at = Instant::now();
        Ok(Evaluated {
            output,
            pixels,
            kind,
            submitted_at,
            completed_at,
        })
    }
    fn publish(&self, work: &Work, evaluated: Evaluated) {
        let Evaluated {
            output,
            pixels,
            kind,
            submitted_at,
            completed_at,
        } = evaluated;
        let mut state = self.shared.lock().unwrap();
        let completion = state.scheduler.complete(work.id);
        let mut old = None;
        match completion {
            Completion::Published => {
                state.scheduler.try_latest();
                old = state.frame.replace(CompletedFrame {
                    request_id: work.request.request_id,
                    generation: work.request.snapshot.generation,
                    quality: work.request.quality,
                    extent: work.request.extent,
                    output: work.request.output,
                    lease: DisplayLease(Arc::new(DisplayOwner {
                        lease: Some(output),
                        queue: self.queue.clone(),
                    })),
                    requested_at: work.requested_at,
                    submitted_at,
                    completed_at,
                });
            }
            Completion::Saved => {
                let save_id = state.scheduler.save_id().expect("completed save slot");
                state.readback = Some(Ok(CompletedReadback {
                    save_id,
                    request_id: work.request.request_id,
                    generation: work.request.snapshot.generation,
                    output: work.request.output,
                    extent: work.request.extent,
                    kind,
                    pixels: pixels.expect("save readback"),
                    requested_at: work.requested_at,
                    submitted_at,
                    completed_at,
                }));
            }
            _ => {}
        }
        drop(state);
        drop(old);
    }
}

fn quantize(
    mapped: &[u8],
    extent: Extent,
    kind: MapKind,
    padded: u32,
) -> std::result::Result<Vec<u8>, GpuError> {
    let channels = if kind == MapKind::Height { 2 } else { 3 };
    let mut pixels = Vec::with_capacity(extent.width as usize * extent.height as usize * channels);
    for row in mapped
        .chunks_exact(padded as usize)
        .take(extent.height as usize)
    {
        match kind {
            MapKind::Height => {
                for pixel in row[..extent.width as usize * 4].as_chunks::<4>().0 {
                    let value = f32::from_le_bytes(*pixel);
                    pixels.extend_from_slice(
                        &((value.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes(),
                    );
                }
            }
            MapKind::Normal => {
                for pixel in row[..extent.width as usize * 8].as_chunks::<8>().0 {
                    for channel in pixel[..6].as_chunks::<2>().0 {
                        let value = half(u16::from_le_bytes([channel[0], channel[1]]));
                        pixels.push(((value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                }
            }
            _ => {
                return Err(GpuError(format!(
                    "readback of {kind:?} is unsupported in this phase"
                )));
            }
        }
    }
    Ok(pixels)
}
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = f32::from(bits & 1023);
    sign * match exponent {
        0 => fraction * 2.0f32.powi(-24),
        31 => {
            if fraction == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + fraction / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15),
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn factory_nodes_have_at_most_one_runtime_registration() {
        for interpretation in [
            texelith_core::InputInterpretation::SrgbPhoto,
            texelith_core::InputInterpretation::LinearHeight,
            texelith_core::InputInterpretation::LinearNormal,
        ] {
            let graph = texelith_core::material_from_source(SourceDescriptor {
                content_hash: [121; 32],
                extent: Extent::new(17, 13).unwrap(),
                interpretation,
            })
            .unwrap();
            for node in graph.material_nodes() {
                let matches = NODE_REGISTRATIONS
                    .iter()
                    .filter(|entry| (entry.supports)(node))
                    .count();
                assert!(matches <= 1, "overlapping registration: {:?}", node.params);
            }
        }
    }
}
