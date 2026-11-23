use crate::gpu_availability;

use super::{
    checks,
    support::{self, Gpu},
};
use std::sync::{Arc, Mutex, OnceLock};
use texelith_core::Extent;
use texelith_gpu::{
    CompletedPreview, GpuError, GpuWorker, PreviewRequest, viewport_shared::PreviewContent,
};

fn with_gpu(run: impl FnOnce(&Gpu)) {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let gpu = GPU
        .get_or_init(|| {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    ..Default::default()
                }))
                .expect("T-063 needs a real GPU; absence is not a passing skip");
            let info = adapter.get_info();
            assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
            eprintln!("T-063 adapter/backend: {info:?}");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: 4096,
                        ..Default::default()
                    },
                    required_features: wgpu::Features::empty(),
                    ..Default::default()
                }))
                .unwrap();
            Mutex::new(Gpu {
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
        .lock()
        .unwrap();
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    run(&gpu);
    gpu.finish().unwrap();
    assert!(
        pollster::block_on(scope.pop()).is_none(),
        "T-063 wgpu validation failure"
    );
}

#[test]
fn preview_api_is_nonblocking_owned_and_uses_core_view_dtos() {
    let _: fn(&GpuWorker, PreviewRequest) -> Result<(), GpuError> = GpuWorker::request_preview;
    let _: fn(&GpuWorker) -> Option<CompletedPreview> = GpuWorker::try_latest_preview;
    fn send<T: Send>() {}
    send::<PreviewRequest>();
    send::<CompletedPreview>();
    let _core_view = |r: &PreviewRequest| -> texelith_core::ViewState { r.view.clone() };
}

#[test]
fn clear_and_material_ramp_have_one_display_transfer_and_deterministic_bytes() {
    with_gpu(|g| checks::clear_and_ramp(g).unwrap());
}

#[test]
fn resize_camera_burst_and_job_changes_publish_only_latest_matching_material() {
    with_gpu(|g| {
        checks::identity_and_churn(g).unwrap();
    });
}

#[test]
fn targets_and_materials_share_budget_and_keep_previous_display_on_rejection() {
    with_gpu(|g| {
        checks::lifetime_and_budget(g).unwrap();
    });
}

#[test]
fn material_compute_and_preview_share_one_submission_authority() {
    with_gpu(|g| checks::compute_supersedes_preview(g).unwrap());
}

#[test]
fn invalid_preview_intents_do_not_replace_the_current_valid_frame() {
    with_gpu(|g| {
        let worker = g.worker(support::BUDGET).unwrap();
        let material = support::material(&worker, 63, 1, 1).unwrap();
        let extent = Extent::new(17, 3).unwrap();
        let valid = support::request(material.clone(), 1, extent, PreviewContent::MaterialRamp);
        worker.request_preview(valid).unwrap();
        let mut invalid =
            support::request(material.clone(), 2, extent, PreviewContent::MaterialRamp);
        invalid.view.camera.yaw = f32::NAN;
        assert!(worker.request_preview(invalid).is_err());
        let mut invalid =
            support::request(material.clone(), 2, extent, PreviewContent::MaterialRamp);
        invalid.extent.width = 0;
        assert!(worker.request_preview(invalid).is_err());
        let invalid = support::request(
            material,
            2,
            extent,
            PreviewContent::Clear([f32::INFINITY, 0.0, 0.0, 1.0]),
        );
        assert!(worker.request_preview(invalid).is_err());
        let frame = support::complete(&worker).unwrap();
        assert_eq!(
            frame.request_id, 1,
            "invalid intent cancelled a valid request"
        );
        assert!(!g.read_display(frame.lease.texture()).unwrap().is_empty());
        worker.close();
        // The real device/queue remain usable after worker shutdown.
        worker.join().unwrap();
        g.finish().unwrap();
    });
}

#[test]
fn retained_viewport_bundle_survives_auxiliary_work_without_material_evaluation() {
    with_gpu(|g| {
        let worker = g.worker(support::BUDGET).unwrap();
        let bundle = support::material(&worker, 140, 1, 1).unwrap();
        let auxiliary = support::material(&worker, 141, 1, 1).unwrap();
        let extent = Extent::new(64, 64).unwrap();
        let request = || support::request(bundle.clone(), 2, extent, PreviewContent::MaterialRamp);
        assert!(worker.request_preview(request()).is_err());
        let misses = worker.diagnostics().cache_misses;
        worker.request_retained_preview(request()).unwrap();
        let frame = support::complete(&worker).unwrap();
        assert_eq!(frame.job_id, bundle.job_id);
        assert_eq!(worker.diagnostics().cache_misses, misses);
        assert!(
            worker.request_retained_preview(request()).is_err(),
            "stale frame ID accepted"
        );
        let mut forged = support::request(auxiliary, 3, extent, PreviewContent::MaterialRamp);
        forged.generation += 1;
        assert!(worker.request_retained_preview(forged).is_err());
        worker.close();
        worker.join().unwrap();
        assert!(worker.request_retained_preview(request()).is_err());
    });
}

#[test]
fn viewport_area_mips_preserve_npot_white_and_average_subpixel_checker() {
    with_gpu(|g| {
        let shader = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("T-140 mip verification"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/viewport_shared/mips.wgsl").into(),
            ),
        });
        let pipeline = g
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        for (width, height, checker) in [
            (17u32, 3u32, false),
            (1, 17, false),
            (17, 1, false),
            (16, 16, true),
        ] {
            let levels = width.max(height).ilog2() + 1;
            let descriptor = texelith_gpu::pool::AllocationDesc::MipTexture {
                extent: Extent::new(width, height).unwrap(),
            };
            let expected_bytes: u64 = (0..levels)
                .map(|level| {
                    u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * 8
                })
                .sum();
            assert_eq!(descriptor.byte_len().unwrap(), expected_bytes);
            let desc = wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            };
            let source = g.device.create_texture(&desc);
            let target = g.device.create_texture(&wgpu::TextureDescriptor {
                mip_level_count: levels,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                ..desc
            });
            let pixels: Vec<u16> = (0..width * height)
                .flat_map(|index| {
                    [if checker && (index % width + index / width).is_multiple_of(2) {
                        0
                    } else {
                        0x3c00
                    }; 4]
                })
                .collect();
            g.queue.write_texture(
                source.as_image_copy(),
                bytemuck::cast_slice(&pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(height),
                },
                source.size(),
            );
            let mut encoder = g.device.create_command_encoder(&Default::default());
            for level in 0..levels {
                let view = |texture: &wgpu::Texture, base_mip_level| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                };
                let input = if level == 0 {
                    view(&source, 0)
                } else {
                    view(&target, level - 1)
                };
                let output = view(&target, level);
                let group = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&input),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&output),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(
                    (width >> level).max(1).div_ceil(8),
                    (height >> level).max(1).div_ceil(8),
                    1,
                );
            }
            let readback = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    mip_level: levels - 1,
                    ..target.as_image_copy()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
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
            g.queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            g.finish().unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range().unwrap();
            let expected = if checker { 0x3800u16 } else { 0x3c00 };
            for value in mapped[..8].as_chunks::<2>().0 {
                assert_eq!(u16::from_le_bytes(*value), expected);
            }
            drop(mapped);
            readback.unmap();
        }
    });
}
