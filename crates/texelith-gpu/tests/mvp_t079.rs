//! T-079: exercise the public worker on a real shared device, never a CPU substitute.
#[path = "../examples/mvp_support/t079/checks.rs"]
mod checks;

#[test]
fn bounded_immutable_export_contract() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("T-079 requires a real GPU; missing hardware is not a passing skip");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("T-079 adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    checks::run(
        std::sync::Arc::new(device),
        std::sync::Arc::new(queue),
        1024,
    )
    .unwrap();
}

#[test]
fn numeric_pitch_contract() {
    use texelith_gpu::export_readback::{PlaneLayout, strip_carrier};
    for width in [1, 17, 257] {
        for channels in [1, 3, 4] {
            for bits in [8, 16, 32] {
                let layout = PlaneLayout::new(
                    texelith_core::Extent::new(width, 3).unwrap(),
                    channels,
                    bits,
                )
                .unwrap();
                assert_eq!(layout.padded_bytes_per_row % 256, 0);
                assert_eq!(
                    layout.staging_bytes,
                    u64::from(layout.padded_bytes_per_row) * 3
                );
                let mut carrier = vec![0xcc; layout.staging_bytes as usize];
                for y in 0..3usize {
                    for x in 0..width as usize {
                        for c in 0..4 {
                            let value = if bits == 32 {
                                -0.25 + c as f32
                            } else {
                                c as f32 / 4.0
                            };
                            let offset = y * layout.padded_bytes_per_row as usize + x * 16 + c * 4;
                            carrier[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                let bytes = strip_carrier(&carrier, layout).unwrap();
                assert_eq!(
                    bytes.len(),
                    width as usize * 3 * channels as usize * (bits / 8) as usize
                );
                for pixel in bytes.chunks_exact(channels as usize * (bits / 8) as usize) {
                    for c in 0..channels as usize {
                        match bits {
                            8 => assert_eq!(pixel[c], (c as f32 / 4.0 * 255.0).round() as u8),
                            16 => assert_eq!(
                                u16::from_be_bytes(pixel[c * 2..c * 2 + 2].try_into().unwrap()),
                                (c as f32 / 4.0 * 65535.0).round() as u16
                            ),
                            32 => assert_eq!(
                                f32::from_le_bytes(pixel[c * 4..c * 4 + 4].try_into().unwrap()),
                                -0.25 + c as f32
                            ),
                            _ => unreachable!(),
                        }
                    }
                }
                assert!(strip_carrier(&carrier[..carrier.len() - 1], layout).is_err());
            }
        }
    }
    assert!(
        PlaneLayout::new(
            texelith_core::Extent {
                width: 0,
                height: 1
            },
            1,
            8
        )
        .is_err()
    );
    assert!(PlaneLayout::new(texelith_core::Extent::new(1, 1).unwrap(), 2, 8).is_err());
}

#[test]
fn unsupported_formats_and_invalid_carriers_fail_explicitly() {
    use texelith_gpu::{
        export_readback::{PlaneLayout, strip_carrier},
        export_shared::validate_format,
    };
    validate_format(texelith_core::ExportFormat::DdsBc5).unwrap(); // T-081 now supports BC5.
    let layout = PlaneLayout::new(texelith_core::Extent::new(1, 1).unwrap(), 1, 32).unwrap();
    let mut carrier = vec![0; layout.staging_bytes as usize];
    carrier[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(strip_carrier(&carrier, layout).is_err());
    carrier[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(strip_carrier(&carrier, layout).is_err());
    let mut forged = layout;
    forged.padded_bytes_per_row = 16;
    assert!(strip_carrier(&carrier, forged).is_err());
}
