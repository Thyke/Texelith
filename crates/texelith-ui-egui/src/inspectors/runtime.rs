//! Auxiliary requests share the material worker and its resource ledger. A single
//! background readback owns its immutable lease until queue completion.
use super::*;
use crate::inspectors::PickRequest;

enum Job {
    Pick(PickRequest),
    Thumbnail { map: MapKind, generation: u64 },
}
enum ResultEvent {
    Pick(PickRequest, Result<[f32; 3], String>),
    Thumbnail(MapKind, u64, Result<Option<egui::ColorImage>, String>),
}
pub(super) struct Auxiliary {
    send: Sender<(Job, CompletedMaterial)>,
    recv: Receiver<ResultEvent>,
    stop: Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    pending: Option<(u64, Job)>,
    reading: bool,
    failed_thumbnails: std::collections::BTreeSet<(u64, MapKind)>,
}
impl Auxiliary {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        current_generation: Arc<AtomicU64>,
    ) -> Result<Self, String> {
        let (send, receive) = bounded::<(Job, CompletedMaterial)>(1);
        let (events, recv) = bounded(1);
        let (stop, stopped) = bounded(1);
        let thread=std::thread::Builder::new().name("texelith-inspector-readback".into()).spawn(move || loop {
            let (job,bundle)=crossbeam_channel::select! {recv(stopped) -> _ => break,recv(receive) -> r => match r {Ok(r)=>r,Err(_)=>break}};
            let result=match job {
                Job::Pick(request) => {
                    let result=read_pixel(&device,&queue,&bundle,request.uv);
                    ResultEvent::Pick(request,result)
                }
                Job::Thumbnail{map,generation} => {
                    let image=presenter::present_thumbnail(&device,&queue,bundle,map,&current_generation);
                    ResultEvent::Thumbnail(map,generation,image)
                }
            };
            crossbeam_channel::select! {recv(stopped) -> _ => break,send(events,result) -> r => if r.is_err() {break}}
        }).map_err(|e|e.to_string())?;
        Ok(Self {
            send,
            recv,
            stop,
            thread: Some(thread),
            pending: None,
            reading: false,
            failed_thumbnails: Default::default(),
        })
    }
    pub fn poll(&mut self, c: &mut Controls) {
        while let Ok(event) = self.recv.try_recv() {
            self.reading = false;
            match event {
                ResultEvent::Pick(request, Ok(color)) => {
                    c.complete_color_pick(&request, color);
                }
                ResultEvent::Pick(request, Err(error)) => {
                    if c.inspectors.pending.as_ref() == Some(&request) {
                        c.inspectors.pending = None;
                        c.set_inspector_error(error);
                    }
                }
                ResultEvent::Thumbnail(map, generation, Ok(Some(image))) => {
                    if c.session()
                        .snapshot()
                        .is_some_and(|s| s.generation == generation)
                    {
                        c.cache_map_thumbnail(map, generation, &image);
                    }
                }
                ResultEvent::Thumbnail(_, _, Ok(None)) => {}
                ResultEvent::Thumbnail(map, generation, Err(error)) => {
                    if c.session()
                        .snapshot()
                        .is_some_and(|s| s.generation == generation)
                    {
                        self.failed_thumbnails.insert((generation, map));
                        c.set_inspector_error(error);
                    }
                }
            }
        }
    }
    pub fn receive(&mut self, bundle: CompletedMaterial) -> Option<CompletedMaterial> {
        if self
            .pending
            .as_ref()
            .is_some_and(|(id, _)| *id == bundle.request_id)
        {
            let (_, job) = self.pending.take().unwrap();
            self.reading = self.send.try_send((job, bundle)).is_ok();
            None
        } else {
            Some(bundle)
        }
    }
    pub fn failed(&mut self, c: &mut Controls) {
        if let Some((_, job)) = self.pending.take() {
            match job {
                Job::Pick(request) => {
                    if c.inspectors.pending.as_ref() == Some(&request) {
                        c.inspectors.pending = None;
                    }
                }
                Job::Thumbnail { map, generation } => {
                    self.failed_thumbnails.insert((generation, map));
                }
            }
        }
    }
    pub fn busy(&self) -> bool {
        self.reading || self.pending.is_some()
    }
    /// Let the next foreground request replace obsolete work in the GPU mailbox.
    pub fn cancel_stale(&mut self, generation: u64) -> Option<u64> {
        if self.pending.as_ref().is_some_and(|(_, job)| {
            matches!(job, Job::Thumbnail { generation: captured, .. } if *captured != generation)
        }) {
            self.pending.take().map(|(id, _)| id)
        } else {
            None
        }
    }
    pub fn request(
        &mut self,
        c: &mut Controls,
        gpu: &GpuWorker,
        next: &mut u64,
    ) -> Option<(u64, u64, MapKind)> {
        if self.busy() {
            return None;
        }
        let snapshot = c.session().snapshot()?;
        let generation = snapshot.generation;
        self.failed_thumbnails.retain(|(g, _)| *g == generation);
        let job = if !c.inspectors.submitted
            && let Some(p) = c.inspectors.pending.clone()
        {
            if p.generation != generation || p.output != snapshot.graph.selected_output_ref() {
                c.inspectors.pending = None;
                return None;
            }
            Job::Pick(p)
        } else {
            // Gestures get all compute time; refresh the retained card images
            // after release instead of starting work for intermediate edits.
            let history = c.session().history();
            if history.active_gesture.is_some() || history.active_stroke.is_some() {
                return None;
            }
            let map = super::super::MAPS.into_iter().find(|map| {
                c.thumbnail_generation(*map) != Some(generation)
                    && !self.failed_thumbnails.contains(&(generation, *map))
                    && super::super::supported(&snapshot, *map)
            })?;
            Job::Thumbnail { map, generation }
        };
        *next = next.checked_add(1).expect("request IDs exhausted");
        let (map, request) = match &job {
            Job::Pick(_) => {
                let source =
                    snapshot
                        .graph
                        .material_nodes()
                        .iter()
                        .find_map(|n| match n.params {
                            MaterialNodeParams::ColorSource(ref source) => {
                                Some((n.id, source.extent))
                            }
                            _ => None,
                        })?;
                let output = OutputRef {
                    node: source.0,
                    port: OutputPort::Albedo,
                };
                (
                    MapKind::Albedo,
                    MaterialRequest {
                        job_id: JobId(*next),
                        request_id: *next,
                        snapshot: Arc::new(snapshot),
                        outputs: [(MapKind::Albedo, output)].into(),
                        extent: source.1,
                        quality: Quality::Full,
                    },
                )
            }
            Job::Thumbnail { map, .. } => (*map, thumbnail_request(snapshot, *map, *next).ok()?),
        };
        match gpu.request_material(request) {
            Ok(()) => {
                if matches!(job, Job::Pick(_)) {
                    c.inspectors.submitted = true;
                }
                self.pending = Some((*next, job));
                Some((*next, generation, map))
            }
            Err(error) => {
                c.set_inspector_error(error.to_string());
                None
            }
        }
    }
}

pub(super) fn thumbnail_request(
    snapshot: EvaluationSnapshot,
    map: MapKind,
    id: u64,
) -> Result<MaterialRequest, String> {
    super::super::request(snapshot, map, id, Quality::Proxy)
}
impl Drop for Auxiliary {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
fn read_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bundle: &CompletedMaterial,
    uv: [f32; 2],
) -> Result<[f32; 3], String> {
    let texture = bundle
        .outputs
        .get(&MapKind::Albedo)
        .ok_or("Original source missing")?
        .texture();
    if texture.format() != wgpu::TextureFormat::Rgba16Float {
        return Err("Picker expects linear RGBA16 source".into());
    }
    let x = ((uv[0] * texture.width() as f32).floor() as u32).min(texture.width() - 1);
    let y = ((uv[1] * texture.height() as f32).floor() as u32).min(texture.height() - 1);
    // Exactly one texel, one 256-byte aligned staging row, never a full image.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("original source one-pixel readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut source = texture.as_image_copy();
    source.origin = wgpu::Origin3d { x, y, z: 0 };
    encoder.copy_texture_to_buffer(
        source,
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|e| e.to_string())?;
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let color = std::array::from_fn(|i| {
        super::super::presenter::half(u16::from_le_bytes([mapped[2 * i], mapped[2 * i + 1]]))
    });
    drop(mapped);
    buffer.unmap();
    if color.iter().any(|v| !v.is_finite()) {
        Err("Nonfinite source color".into())
    } else {
        Ok(color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_thumbnail_work_is_cancelled_and_late_results_cannot_replace_the_cache() {
        let mut c = Controls::new_material();
        c.open_image("fixture.png".into());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            panic!()
        };
        c.complete_import(
            token,
            SourceDescriptor {
                content_hash: [1; 32],
                extent: Extent::new(32, 32).unwrap(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        )
        .unwrap();
        let snapshot = c.session().snapshot().unwrap();
        let map = MapKind::Normal;
        let old = egui::ColorImage::filled([32, 32], egui::Color32::GREEN);
        c.cache_map_thumbnail(map, snapshot.generation, &old);
        let node = snapshot
            .graph
            .material_nodes()
            .iter()
            .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
            .unwrap()
            .id;
        c.action(crate::command_adapter::UiAction::Change {
            address: ParameterAddress {
                node,
                parameter: ParameterId::Strength,
            },
            value: ParameterValue::Float(2.0),
        });
        let current = c.session().snapshot().unwrap().generation;
        assert_ne!(current, snapshot.generation);
        let (events, recv) = bounded(1);
        let mut auxiliary = Auxiliary {
            send: bounded(1).0,
            recv,
            stop: bounded(1).0,
            thread: None,
            pending: Some((
                42,
                Job::Thumbnail {
                    map,
                    generation: snapshot.generation,
                },
            )),
            reading: false,
            failed_thumbnails: Default::default(),
        };
        assert_eq!(auxiliary.cancel_stale(current), Some(42));
        assert!(!auxiliary.busy());
        events
            .send(ResultEvent::Thumbnail(
                map,
                snapshot.generation,
                Ok(Some(egui::ColorImage::filled([32, 32], egui::Color32::RED))),
            ))
            .unwrap();
        auxiliary.poll(&mut c);
        assert_eq!(c.workspace.thumbnails[&map].image, old);
        assert_eq!(c.thumbnail_generation(map), Some(snapshot.generation));
        events
            .send(ResultEvent::Thumbnail(
                map,
                current,
                Err("readback failed".into()),
            ))
            .unwrap();
        auxiliary.poll(&mut c);
        assert_eq!(c.workspace.thumbnails[&map].image, old);
        assert!(auxiliary.failed_thumbnails.contains(&(current, map)));
    }
}
