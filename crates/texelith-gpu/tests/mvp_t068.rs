#[path = "../examples/mvp_support/t068/channels.rs"]
mod channels;
#[path = "../examples/mvp_support/t068/checks.rs"]
mod checks;
#[path = "../examples/mvp_support/contract.rs"]
mod contract;
#[path = "../examples/mvp_support/t068/png.rs"]
mod png;
#[path = "../examples/mvp_support/t068/probe.rs"]
mod probe;
#[path = "../examples/mvp_support/t068/support.rs"]
mod support;

#[test]
fn pom_belongs_to_viewport_and_all_only() {
    assert_eq!(contract::select_suite("pom").unwrap(), ["pom"]);
    for name in ["viewport", "all"] {
        assert_eq!(
            contract::select_suite(name)
                .unwrap()
                .iter()
                .filter(|s| **s == "pom")
                .count(),
            1
        );
    }
    assert!(!contract::select_suite("maps").unwrap().contains(&"pom"));
    assert!(contract::select_suite("").is_err());
    assert!(contract::select_suite("unknown").is_err());
}

#[test]
fn surface_parameters_reject_nonfinite_and_out_of_range_midlevels() {
    use texelith_gpu::{pbr::PbrSettings, pom::SurfaceSettings};
    for midlevel in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert!(
            PbrSettings {
                surface: SurfaceSettings {
                    midlevel,
                    ..Default::default()
                },
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    for midlevel in [0.0, 0.3, 1.0] {
        assert!(
            PbrSettings {
                surface: SurfaceSettings {
                    midlevel,
                    ..Default::default()
                },
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
    }
}

#[test]
fn real_gpu_bounded_pom_and_shared_worker_contracts() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    use std::sync::Arc;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let instance = wgpu::Instance::new(desc);
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("POM requires an actual GPU");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("T-068 {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let g = support::Gpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
    };
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::Validation);
    probe::check(&g).unwrap();
    channels::check(&g).unwrap();
    checks::worker(&g, None).unwrap();
    g.finish().unwrap();
    assert!(pollster::block_on(scope.pop()).is_none());
}
