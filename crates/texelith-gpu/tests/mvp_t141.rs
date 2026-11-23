//! Headless regression for the user-visible environment/background path.
#[path = "../examples/mvp_support/t067/png.rs"]
mod png;
#[allow(dead_code)]
#[path = "../examples/mvp_support/t067/support.rs"]
mod support;

use std::sync::Arc;
use texelith_core::{Camera, Extent, PreviewMesh};
use texelith_gpu::{ibl::Environment, pbr::PbrSettings, viewport_shared::PreviewContent};

#[test]
fn sky_background_and_ibl_respond_independently_on_sphere_and_plane() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = if cfg!(windows) {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("real GPU required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let g = support::Gpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let worker = g.worker(support::BUDGET).unwrap();
    let material = support::material(&worker).unwrap();
    let extent = Extent::new(384, 256).unwrap();
    let sky = Arc::new(Environment::procedural());
    let mut id = 2;
    let mut render = |mesh: PreviewMesh,
                      settings: PbrSettings,
                      environment: Option<Arc<Environment>>,
                      exposure: f32| {
        let mut request = support::request(material.clone(), id, extent, sky.clone(), 0.12, 1.0);
        id += 1;
        request.view.camera = Camera {
            yaw: 0.45,
            pitch: 0.3,
            distance: if mesh == PreviewMesh::Plane { 2.7 } else { 3.6 },
            ..Default::default()
        };
        request.view.mesh = mesh;
        request.content = if let Some(environment) = environment {
            PreviewContent::Ibl {
                settings,
                environment,
                exposure,
            }
        } else {
            PreviewContent::Pbr(settings)
        };
        let frame = support::render(&worker, request).unwrap();
        g.read(frame.lease.texture(), 4).unwrap()
    };
    let settings = PbrSettings {
        background_visible: true,
        roughness: Some(0.12),
        metallic: Some(1.0),
        ..Default::default()
    };
    let center =
        |pixels: &[u8]| pixels[((128 * 384 + 192) * 4)..((128 * 384 + 192) * 4 + 3)].to_vec();
    let difference = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum::<u64>()
    };
    for (mesh, name) in [
        (PreviewMesh::Sphere, "sphere"),
        (PreviewMesh::Plane, "plane"),
    ] {
        let settings = PbrSettings {
            metallic: Some(if name == "plane" { 0.0 } else { 1.0 }),
            ..settings
        };
        let visible = render(mesh.clone(), settings, Some(sky.clone()), 0.);
        let hidden = render(
            mesh.clone(),
            PbrSettings {
                background_visible: false,
                ..settings
            },
            Some(sky.clone()),
            0.,
        );
        assert_eq!(
            center(&visible),
            center(&hidden),
            "background toggle changed material lighting"
        );
        assert_ne!(
            &visible[..3],
            &visible[(255 * 384 * 4)..(255 * 384 * 4 + 3)],
            "missing sky gradient"
        );
        assert_eq!(
            &hidden[..3],
            &hidden[(255 * 384 * 4)..(255 * 384 * 4 + 3)],
            "solid background changed with direction"
        );
        let none = render(mesh.clone(), settings, None, 0.);
        assert_ne!(
            center(&visible),
            center(&none),
            "IBL does not light the mesh"
        );
        let dim = render(mesh.clone(), settings, Some(sky.clone()), -3.);
        assert!(
            center(&visible).iter().map(|v| u32::from(*v)).sum::<u32>()
                > center(&dim).iter().map(|v| u32::from(*v)).sum::<u32>(),
            "environment intensity has no effect"
        );
        let rotated = render(
            mesh.clone(),
            PbrSettings {
                environment_rotation: 1.6,
                ..settings
            },
            Some(sky.clone()),
            0.,
        );
        assert!(
            difference(&visible, &rotated) > 1000,
            "environment rotation has no visible effect"
        );
        let blurred = render(
            mesh,
            PbrSettings {
                background_blur: 0.8,
                ..settings
            },
            Some(sky.clone()),
            0.,
        );
        assert_eq!(
            center(&visible),
            center(&blurred),
            "background blur changed material lighting"
        );
        assert!(
            difference(&visible, &blurred) > 1000,
            "background blur has no effect"
        );
        if let Some(dir) = std::env::var_os("TEXELITH_T141_EVIDENCE") {
            let dir = std::path::PathBuf::from(dir);
            png::save(&dir.join(format!("{name}-sky.png")), 384, 256, &visible).unwrap();
            png::save(&dir.join(format!("{name}-none.png")), 384, 256, &none).unwrap();
        }
    }
    // A replacement decoded environment goes through exactly this same path.
    let constant =
        Arc::new(Environment::new(Extent::new(2, 1).unwrap(), &[[0.03, 0.1, 0.9]; 2]).unwrap());
    let replacement = render(PreviewMesh::Sphere, settings, Some(constant), 0.);
    assert!(
        replacement[2] > replacement[0] + 50,
        "replacement environment is absent from the background"
    );
    let sky_frame = render(PreviewMesh::Sphere, settings, Some(sky.clone()), 0.);
    assert_ne!(
        center(&replacement),
        center(&sky_frame),
        "replacement environment is absent from mesh lighting"
    );
    worker.close();
    worker.join().unwrap();
    g.finish().unwrap();
    assert!(
        pollster::block_on(scope.pop()).is_none(),
        "wgpu validation error"
    );
}
