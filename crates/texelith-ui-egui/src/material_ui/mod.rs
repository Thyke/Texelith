//! M1 material discovery, presentation and command-route diagnostics.
pub mod labels;
pub(crate) mod presenter;
mod runtime;
pub mod smoke;
pub(crate) mod snapshots;
pub(crate) use runtime::MaterialIntegration;
use texelith_core::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportKind {
    Photo,
    Height,
    Normal,
}

pub const MAPS: [MapKind; 7] = [
    MapKind::Albedo,
    MapKind::Height,
    MapKind::Normal,
    MapKind::Ao,
    MapKind::Roughness,
    MapKind::Metallic,
    MapKind::Edge,
];

pub fn map_label(map: MapKind) -> &'static str {
    match map {
        MapKind::Albedo => "Albedo",
        MapKind::Height => "Height",
        MapKind::Normal => "Normal",
        MapKind::Ao => "AO",
        MapKind::Roughness => "Roughness",
        MapKind::Metallic => "Metallic",
        MapKind::Edge => "Edge",
        MapKind::Curvature => "Curvature",
        MapKind::Cavity => "Cavity",
        MapKind::Mask => "Painted mask",
    }
}

/// Registration-derived planning prevents the UI advertising unfinished producers.
pub fn supported(snapshot: &EvaluationSnapshot, map: MapKind) -> bool {
    request(snapshot.clone(), map, 1, texelith_gpu::Quality::Full).is_ok_and(|r| {
        texelith_gpu::MaterialPlan::new(&r, 1024 * 1024 * 1024, Default::default()).is_ok()
    })
}

pub fn request(
    snapshot: EvaluationSnapshot,
    map: MapKind,
    id: u64,
    quality: texelith_gpu::Quality,
) -> Result<texelith_gpu::MaterialRequest, String> {
    let output = *snapshot
        .graph
        .material_outputs()
        .and_then(|m| m.get(&map))
        .ok_or("Map unavailable for this source")?;
    let source = snapshot
        .graph
        .material_output_extent(output)
        .map_err(|e| e.to_string())?;
    let extent = if quality == texelith_gpu::Quality::Proxy {
        texelith_gpu::proxy_extent(source).map_err(|e| e.to_string())?
    } else {
        source
    };
    Ok(texelith_gpu::MaterialRequest {
        job_id: texelith_gpu::JobId(id),
        request_id: id,
        snapshot: std::sync::Arc::new(snapshot),
        outputs: [(map, output)].into(),
        extent,
        quality,
    })
}

pub fn options(graph: &Graph) -> (bool, bool) {
    texelith_core::material_options(graph)
}

pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct Wake(std::thread::Thread);
    impl std::task::Wake for Wake {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Wake(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => {
                std::thread::park_timeout(std::time::Duration::from_millis(10))
            }
        }
    }
}
