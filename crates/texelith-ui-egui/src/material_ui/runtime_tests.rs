use super::*;
use crate::command_adapter::UiAction;
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

#[test]
fn studio_split_pairs_selected_map_with_lit_material_and_refreshes_selection() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = Harness::new();
    let mut c = Controls::new_material();
    h.decoded(&mut c, height_upload(0.5));
    h.until(&mut c, |r| r.publications > 0);
    c.action(UiAction::Request(CommandRequest::CaptureA {}));
    let capture = c.session().view_state().capture_a;
    let history = c.session().history().undo_entries;
    c.preview.split_view = true;
    c.preview.compare_enabled = true;
    c.action(UiAction::Request(CommandRequest::SetMode {
        mode: texelith_core::ViewMode::ThreeD,
    }));
    h.runtime.viewport.set_extent(Extent {
        width: 96,
        height: 128,
    });
    let mut previous = None;
    for map in [MapKind::Height, MapKind::Normal] {
        c.action(UiAction::Request(CommandRequest::SelectMap { map }));
        let baseline = h.runtime.viewport.publications;
        h.until(&mut c, |r| r.viewport.publications > baseline);
        assert!(
            h.runtime.viewport.error.is_none(),
            "{:?}",
            h.runtime.viewport.error
        );
        let (left, right) = h.runtime.viewport.take_ready().expect("paired frame");
        let right = right.expect("lit pane");
        assert_eq!(left.generation, right.generation);
        assert_eq!(left.extent, right.extent);
        assert_eq!(
            left.extent,
            Extent {
                width: 96,
                height: 128
            }
        );
        let map_pixels = crate::compare::smoke::readback(&h.device, &h.queue, &left).unwrap();
        let lit_pixels = crate::compare::smoke::readback(&h.device, &h.queue, &right).unwrap();
        assert_ne!(
            map_pixels, lit_pixels,
            "Split must show different map/lit content"
        );
        if map == MapKind::Height {
            assert!(
                map_pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[0] == p[1] && p[1] == p[2])
            );
        } else {
            assert_ne!(
                previous.as_ref().unwrap(),
                &map_pixels,
                "map selection refreshes left pane"
            );
        }
        previous = Some(map_pixels);
    }
    assert_eq!(c.session().view_state().capture_a, capture);
    assert_eq!(c.session().history().undo_entries, history);
}

#[path = "thumbnail_tests.rs"]
mod thumbnails;

#[path = "defaults_tests.rs"]
mod defaults;

#[path = "live_drag_tests.rs"]
mod live_drag;

struct Harness {
    runtime: MaterialIntegration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Harness {
    fn new() -> Self {
        let backends = if cfg!(windows) {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::PRIMARY
        };
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = super::super::block_on(instance.enumerate_adapters(backends))
            .into_iter()
            .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .min_by_key(|a| a.get_info().device_type != wgpu::DeviceType::DiscreteGpu)
            .expect("T-061 regressions require a native GPU");
        eprintln!("T-061 regression adapter: {:?}", adapter.get_info());
        let (device, queue) =
            super::super::block_on(adapter.request_device(&Default::default())).unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let gpu =
            Arc::new(GpuWorker::spawn(device.clone(), queue.clone(), Default::default()).unwrap());
        let runtime = MaterialIntegration::new(gpu, device.clone(), queue.clone()).unwrap();
        Self {
            runtime,
            device,
            queue,
        }
    }

    fn decoded(&mut self, c: &mut Controls, upload: TypedSourceUpload) -> ImportToken {
        // These publication fixtures poll the 2D presenter without a viewport render loop.
        c.action(UiAction::Request(CommandRequest::SetMode {
            mode: ViewMode::TwoD,
        }));
        c.preview.split_view = false;
        c.open_image("injected-decoded-source.exr".into());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            panic!("import must emit LoadImage")
        };
        self.runtime.latest_import = Some(token);
        // Inject a decoded result at the same boundary as Event::Loaded;
        // GPU admission, asynchronous validation and acknowledgement stay real.
        self.runtime.loaded(c, token, Ok(upload));
        token
    }

    fn until(&mut self, c: &mut Controls, done: impl Fn(&MaterialIntegration) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.runtime.tick(c);
            if done(&self.runtime) {
                break;
            }
            assert!(Instant::now() < deadline, "{}", self.runtime.status);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.runtime.gpu.close();
        self.runtime.gpu.join().unwrap();
    }
}

fn height_upload(offset: f32) -> TypedSourceUpload {
    let extent = Extent {
        width: 32,
        height: 32,
    };
    let values: Vec<_> = (0..1024)
        .map(|i| offset + 0.125 * (std::f32::consts::TAU * (i % 32) as f32 / 32.0).sin())
        .collect();
    let pixels = TypedSourcePixels::HeightF32(values.into());
    let descriptor = pixels.descriptor(extent).unwrap();
    TypedSourceUpload { descriptor, pixels }
}

#[test]
fn failed_async_upload_preserves_session_and_allows_the_next_import() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = Harness::new();
    let mut c = Controls::new_material();
    h.decoded(&mut c, height_upload(0.5));
    h.until(&mut c, |r| r.publications > 0);
    let normal = c
        .session()
        .snapshot()
        .unwrap()
        .graph
        .material_nodes()
        .iter()
        .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
        .unwrap()
        .id;
    let publications = h.runtime.publications;
    c.action(UiAction::Change {
        address: ParameterAddress {
            node: normal,
            parameter: ParameterId::Strength,
        },
        value: ParameterValue::Float(0.25),
    });
    h.until(&mut c, |r| r.publications > publications);
    let before = c.session().snapshot().unwrap();
    let history = c.session().history();
    let preview = h.runtime.image.clone();
    assert_eq!(history.undo_entries, 1);

    // Length validation succeeds synchronously; content validation fails on the GPU worker.
    let mut bad = height_upload(2.0);
    bad.descriptor.content_hash = [99; 32];
    h.decoded(&mut c, bad);
    assert_eq!(
        c.session().snapshot(),
        Some(before.clone()),
        "queued upload must not commit"
    );
    assert_eq!(c.session().history(), history);
    h.until(&mut c, |r| r.status.contains("failed"));
    assert_eq!(c.session().snapshot(), Some(before.clone()));
    assert_eq!(c.session().history(), history);
    assert_eq!(h.runtime.image, preview);
    assert!(h.runtime.ready);
    assert!(h.runtime.pending_upload.is_none());

    let publications = h.runtime.publications;
    h.decoded(&mut c, height_upload(-2.0));
    h.until(&mut c, |r| r.publications > publications);
    assert_ne!(c.session().snapshot(), Some(before));
    assert_eq!(c.session().history().undo_entries, 0);
}

#[test]
fn signed_and_hdr_exr_heights_reach_height_normal_and_ao_without_clipping() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h = Harness::new();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target-evidence/T-061/tests");
    std::fs::create_dir_all(&dir).unwrap();
    let mut c = Controls::new_material();
    for (id, offset) in [(1, -3.0), (2, 2.0)] {
        let upload = height_upload(offset);
        let TypedSourcePixels::HeightF32(values) = &upload.pixels else {
            unreachable!()
        };
        let path = dir.join(format!("signed-hdr-{}-{offset}.exr", std::process::id()));
        std::fs::write(
            &path,
            texelith_io::encoders::encode_exr32(32, 32, values).unwrap(),
        )
        .unwrap();
        let upload =
            presenter::decode(&path, ImportKind::Height, NormalConvention::OpenGl).unwrap();
        h.runtime.gpu.install_typed_source(upload.clone()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(event) = h.runtime.gpu.try_event() {
                assert!(
                    matches!(event, WorkerEvent::SourceReady { content_hash } if content_hash == upload.descriptor.content_hash)
                );
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        c.open_image(path.clone());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            panic!()
        };
        c.complete_import(token, upload.descriptor).unwrap();
        c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
            seamless: false,
            painted_mask: false,
        }));
        let snapshot = c.session().snapshot().unwrap();
        let mut request =
            super::super::request(snapshot.clone(), MapKind::Height, id, Quality::Full).unwrap();
        request.outputs = [MapKind::Height, MapKind::Normal, MapKind::Ao]
            .into_iter()
            .map(|map| (map, snapshot.graph.material_outputs().unwrap()[&map]))
            .collect();
        h.runtime.gpu.request_material(request).unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        let bundle = loop {
            if let Some(event) = h.runtime.gpu.try_event() {
                panic!("{event:?}");
            }
            if let Some(bundle) = h.runtime.gpu.try_latest_material() {
                break bundle;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        };
        let height = presenter::read_texture(
            &h.device,
            &h.queue,
            bundle.outputs[&MapKind::Height].texture(),
        )
        .unwrap();
        for (actual, expected) in height.iter().zip(values.iter()) {
            assert!(
                (actual[0] - expected).abs() < 1e-6,
                "offset {offset}: {} != {expected}",
                actual[0]
            );
        }
        let normal = presenter::read_texture(
            &h.device,
            &h.queue,
            bundle.outputs[&MapKind::Normal].texture(),
        )
        .unwrap();
        assert!(
            normal.iter().any(|p| p[0].abs() > 0.1),
            "height variation must produce slopes"
        );
        assert!(
            normal
                .iter()
                .all(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1.0).abs() < 0.002)
        );
        let ao =
            presenter::read_texture(&h.device, &h.queue, bundle.outputs[&MapKind::Ao].texture())
                .unwrap();
        assert!(
            ao.iter().any(|p| p[0] < 0.99),
            "varying height must occlude"
        );
        std::fs::remove_file(path).unwrap();
    }
}
