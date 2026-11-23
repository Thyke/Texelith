//! Independent ABI driver and production executor, sharing only readback/accounting.
use crate::reference::abi::{
    BlurWeights, NodeUniform, decode_half, row_layout, strip_rows, workgroups,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    num::NonZeroU64,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use texelith_core::{Extent, InputInterpretation, MapKind};
use texelith_gpu::{Axis, GpuExecutor, Pass};
use wgpu::util::DeviceExt;

const BUDGET: u64 = 1024 * 1024 * 1024;
pub const SHADERS: [&str; 6] = [
    "source",
    "downsample",
    "levels",
    "blur",
    "normal",
    "display",
];

#[derive(Default)]
struct Accounting {
    live: AtomicU64,
    peak: AtomicU64,
    count: AtomicU64,
}

struct Reservation {
    accounting: Arc<Accounting>,
    bytes: u64,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.accounting
            .live
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

pub struct Image {
    pub texture: wgpu::Texture,
    pub extent: Extent,
    pub format: wgpu::TextureFormat,
    _reservation: Reservation,
}

pub struct Harness {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: BTreeMap<&'static str, wgpu::ComputePipeline>,
    executor: Option<GpuExecutor>,
    accounting: Arc<Accounting>,
    errors: Arc<Mutex<Vec<String>>>,
    pub measurements: Vec<Value>,
}

fn bpp(format: wgpu::TextureFormat) -> u32 {
    if format == wgpu::TextureFormat::Rgba16Float {
        8
    } else {
        4
    }
}

fn size(e: Extent) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: e.width,
        height: e.height,
        depth_or_array_layers: 1,
    }
}

impl Harness {
    /// Only the headless bootstrap creates a device; the driver takes shared handles.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let callback_errors = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            callback_errors.lock().unwrap().push(format!("{error}"));
        }));
        Self {
            device,
            queue,
            pipelines: BTreeMap::new(),
            executor: None,
            accounting: Arc::default(),
            errors,
            measurements: Vec::new(),
        }
    }

    pub fn compile(&mut self) -> Result<(), String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
        for name in SHADERS {
            let path = root.join(format!("{name}.wgsl"));
            let source = std::fs::read_to_string(&path)
                .map_err(|e| format!("implementation pending: {}: {e}", path.display()))?;
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let format = match name {
                "normal" => wgpu::TextureFormat::Rgba16Float,
                "display" => wgpu::TextureFormat::Rgba8Unorm,
                _ => wgpu::TextureFormat::R32Float,
            };
            let common = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ADR common ABI"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(64),
                            },
                            count: None,
                        },
                    ],
                });
            let weights = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ADR blur weights ABI"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(128),
                        },
                        count: None,
                    }],
                });
            let groups = if name == "blur" {
                vec![Some(&common), Some(&weights)]
            } else {
                vec![Some(&common)]
            };
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(name),
                    bind_group_layouts: &groups,
                    immediate_size: 0,
                });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(format!("{name}: {error}"));
            }
            self.pipelines.insert(name, pipeline);
        }
        Ok(())
    }

    pub fn enable_production(&mut self) -> Result<(), String> {
        self.executor = Some(
            pollster::block_on(GpuExecutor::new(self.device.clone(), self.queue.clone()))
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    pub fn production(&self) -> bool {
        self.executor.is_some()
    }

    pub fn verify_rejections(&self, checks: &mut Vec<Value>) -> Result<(), String> {
        let executor = self
            .executor
            .as_ref()
            .ok_or("production executor missing")?;
        let e = Extent::new(17, 11).unwrap();
        let input = self.image(e, wgpu::TextureFormat::R32Float, false)?;
        let output = self.image(e, wgpu::TextureFormat::R32Float, false)?;
        let no_storage = self.image(e, wgpu::TextureFormat::R32Float, true)?;
        let wrong_extent = self.image(
            Extent::new(16, 11).unwrap(),
            wgpu::TextureFormat::R32Float,
            false,
        )?;
        let rgba = self.image(e, wgpu::TextureFormat::Rgba8Unorm, false)?;
        let levels = Pass::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 1.0,
        };
        let alias = input.texture.clone();
        for (name, result) in [
            (
                "aliased_texture_handle",
                executor.prepare(levels, &input.texture, &alias, e),
            ),
            (
                "output_format",
                executor.prepare(levels, &input.texture, &rgba.texture, e),
            ),
            (
                "input_format",
                executor.prepare(levels, &rgba.texture, &output.texture, e),
            ),
            (
                "output_usage",
                executor.prepare(levels, &input.texture, &no_storage.texture, e),
            ),
            (
                "mismatched_extent",
                executor.prepare(levels, &input.texture, &wrong_extent.texture, e),
            ),
            (
                "invalid_levels",
                executor.prepare(
                    Pass::Levels {
                        black: 0.5,
                        white: 0.5,
                        gamma: 1.0,
                    },
                    &input.texture,
                    &output.texture,
                    e,
                ),
            ),
            (
                "downsample_source_extent",
                executor.prepare(Pass::Downsample, &wrong_extent.texture, &output.texture, e),
            ),
        ] {
            checks.push(json!({ "name": format!("production/reject/{name}"), "driver": "production", "passed": result.is_err() }));
            if result.is_ok() {
                return Err(format!("production accepted invalid {name}"));
            }
        }
        let _uniform_bytes = self.reserve(64)?;
        let mut prepared = executor
            .prepare(levels, &input.texture, &output.texture, e)
            .map_err(|e| e.to_string())?;
        let rejected = prepared.update(Pass::Normal { strength: 1.0 }).is_err();
        checks.push(json!({ "name": "production/reject/pipeline_update", "driver": "production", "passed": rejected }));
        if !rejected {
            return Err("production accepted incompatible pipeline update".into());
        }
        self.check_errors()
    }

    pub fn run_blur(
        &mut self,
        input: &Image,
        uniform: NodeUniform,
        weights: BlurWeights,
        sigma_source_px: f32,
        vertical: bool,
    ) -> Result<(Image, Vec<u8>), String> {
        self.run_operation(
            "blur",
            input,
            uniform,
            Some(weights),
            Some(Pass::Blur {
                sigma_source_px,
                axis: if vertical { Axis::Y } else { Axis::X },
            }),
        )
    }

    fn reserve(&self, bytes: u64) -> Result<Reservation, String> {
        let live = self.accounting.live.load(Ordering::Relaxed);
        if live.checked_add(bytes).is_none_or(|n| n > BUDGET) {
            return Err("test driver exceeded 1 GiB payload budget".into());
        }
        self.accounting.live.store(live + bytes, Ordering::Relaxed);
        self.accounting
            .peak
            .fetch_max(live + bytes, Ordering::Relaxed);
        self.accounting.count.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation {
            accounting: self.accounting.clone(),
            bytes,
        })
    }

    pub fn image(
        &self,
        extent: Extent,
        format: wgpu::TextureFormat,
        upload: bool,
    ) -> Result<Image, String> {
        extent.validate().map_err(|e| e.to_string())?;
        let reservation = self
            .reserve(u64::from(extent.width) * u64::from(extent.height) * u64::from(bpp(format)))?;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | if upload {
                wgpu::TextureUsages::COPY_DST
            } else {
                wgpu::TextureUsages::STORAGE_BINDING
            };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("verifier fixture"),
            size: size(extent),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        Ok(Image {
            texture,
            extent,
            format,
            _reservation: reservation,
        })
    }

    pub fn upload(
        &self,
        extent: Extent,
        format: wgpu::TextureFormat,
        bytes: &[u8],
    ) -> Result<Image, String> {
        let image = self.image(extent, format, true)?;
        let tight = extent
            .width
            .checked_mul(bpp(format))
            .ok_or("upload overflow")?;
        if bytes.len() != tight as usize * extent.height as usize {
            return Err("upload byte count mismatch".into());
        }
        self.queue.write_texture(
            image.texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tight),
                rows_per_image: Some(extent.height),
            },
            size(extent),
        );
        Ok(image)
    }

    pub fn upload_height(&self, input: &crate::reference::oracle::Height) -> Result<Image, String> {
        let values: Vec<f32> = input.pixels.iter().map(|h| *h as f32).collect();
        self.upload(
            input.extent,
            wgpu::TextureFormat::R32Float,
            bytemuck::cast_slice(&values),
        )
    }

    /// Submits, asynchronously maps and polls only in this headless verifier.
    pub fn run(
        &mut self,
        name: &'static str,
        input: &Image,
        uniform: NodeUniform,
        weights: Option<BlurWeights>,
    ) -> Result<(Image, Vec<u8>), String> {
        self.run_operation(name, input, uniform, weights, None)
    }

    fn run_operation(
        &mut self,
        name: &'static str,
        input: &Image,
        uniform: NodeUniform,
        weights: Option<BlurWeights>,
        operation: Option<Pass>,
    ) -> Result<(Image, Vec<u8>), String> {
        let expected_input = match name {
            "source" => wgpu::TextureFormat::Rgba8Unorm,
            "display" if uniform.flags[0] == 1 => wgpu::TextureFormat::Rgba16Float,
            _ => wgpu::TextureFormat::R32Float,
        };
        if input.format != expected_input {
            return Err(format!(
                "{name}: input format {:?} != {expected_input:?}",
                input.format
            ));
        }
        let extent =
            Extent::new(uniform.dimensions[0], uniform.dimensions[1]).map_err(|e| e.to_string())?;
        let format = match name {
            "normal" => wgpu::TextureFormat::Rgba16Float,
            "display" => wgpu::TextureFormat::Rgba8Unorm,
            _ => wgpu::TextureFormat::R32Float,
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let started = Instant::now();
        let output = self.image(extent, format, false)?;
        let _uniform_bytes = self.reserve(64)?;
        let _weights_bytes = if weights.is_some() {
            Some(self.reserve(128)?)
        } else {
            None
        };
        let independent = if self.executor.is_none() {
            let buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test node ABI"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let pipeline = self
                .pipelines
                .get(name)
                .ok_or_else(|| format!("pipeline {name} unavailable"))?;
            let input_view = input.texture.create_view(&Default::default());
            let output_view = output.texture.create_view(&Default::default());
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            let weights_group = weights.map(|weights| {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("test blur weights"),
                        contents: bytemuck::bytes_of(&weights),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("test weights"),
                    layout: &pipeline.get_bind_group_layout(1),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            });
            Some((group, weights_group))
        } else {
            None
        };

        let prepared = if let Some(executor) = &self.executor {
            let operation = if let Some(operation) = operation {
                operation
            } else {
                match name {
                    "source" => Pass::Source(if uniform.flags[0] == 0 {
                        InputInterpretation::SrgbPhoto
                    } else {
                        InputInterpretation::LinearHeight
                    }),
                    "downsample" => Pass::Downsample,
                    "levels" => Pass::Levels {
                        black: uniform.parameters[0],
                        white: uniform.parameters[1],
                        gamma: uniform.parameters[2],
                    },
                    "normal" => Pass::Normal {
                        strength: uniform.parameters[0],
                    },
                    "display" => Pass::Display(if uniform.flags[0] == 0 {
                        MapKind::Height
                    } else {
                        MapKind::Normal
                    }),
                    // Blur always supplies its original source-pixel sigma via run_blur.
                    "blur" => return Err("production blur requires source-pixel sigma".into()),
                    _ => return Err("unknown production pass".into()),
                }
            };
            let source = Extent::new(uniform.dimensions[2], uniform.dimensions[3])
                .map_err(|e| e.to_string())?;
            let initial = match operation {
                Pass::Source(InputInterpretation::SrgbPhoto) => {
                    Pass::Source(InputInterpretation::LinearHeight)
                }
                Pass::Source(_) => Pass::Source(InputInterpretation::SrgbPhoto),
                Pass::Levels { .. } => Pass::Levels {
                    black: 0.0,
                    white: 1.0,
                    gamma: 1.0,
                },
                Pass::Blur { .. } => Pass::Blur {
                    sigma_source_px: 0.0,
                    axis: Axis::X,
                },
                Pass::Normal { .. } => Pass::Normal { strength: 0.0 },
                other => other,
            };
            let mut prepared = executor
                .prepare(initial, &input.texture, &output.texture, source)
                .map_err(|e| e.to_string())?;
            if prepared.uniform_bytes() != 64 + if weights.is_some() { 128 } else { 0 } {
                return Err("production uniform allocation accounting mismatch".into());
            }
            // Encode/submit the initial values, then update the SAME bindings.
            // Queue ordering makes this safe; no production polling is involved.
            let mut first = self.device.create_command_encoder(&Default::default());
            prepared.encode(&mut first);
            self.queue.submit([first.finish()]);
            prepared.update(operation).map_err(|e| e.to_string())?;
            Some(prepared)
        } else {
            None
        };
        let (_, padded) = row_layout(extent.width, bpp(format))?;
        let readback_size = u64::from(padded) * u64::from(extent.height);
        let _readback_bytes = self.reserve(readback_size)?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test async readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(prepared) = &prepared {
            prepared.encode(&mut encoder);
        }
        if let Some((group, weights_group)) = &independent {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[name]);
            pass.set_bind_group(0, group, &[]);
            if let Some(group) = weights_group {
                pass.set_bind_group(1, group, &[]);
            }
            let [x, y, z] = workgroups(extent)?;
            pass.dispatch_workgroups(x, y, z);
        }
        encoder.copy_texture_to_buffer(
            output.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(extent.height),
                },
            },
            size(extent),
        );
        let encoded_ms = started.elapsed().as_secs_f64() * 1000.0;
        let submitted = Instant::now();
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        loop {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| e.to_string())?;
            match receiver.try_recv() {
                Ok(result) => {
                    result.map_err(|e| e.to_string())?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err("map callback disconnected".into());
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if submitted.elapsed() > Duration::from_secs(30) {
                return Err("GPU readback timeout".into());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let completion_ms = submitted.elapsed().as_secs_f64() * 1000.0;
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let bytes = strip_rows(&mapped, extent.width, extent.height, bpp(format))?;
        drop(mapped);
        readback.unmap();
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(format!("{name}: {error}"));
        }
        self.check_errors()?;
        self.measurements.push(json!({ "driver": if self.production() { "production" } else { "independent" }, "dispatches": if self.production() { 2 } else { 1 }, "pass": name, "extent": [extent.width, extent.height], "output_format": format!("{format:?}"), "encode_and_allocate_ms": encoded_ms, "submit_to_mapped_ms_upper_bound": completion_ms, "readback_bytes": readback_size }));
        Ok((output, bytes))
    }

    pub fn check_errors(&self) -> Result<(), String> {
        let errors = self.errors.lock().unwrap();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("GPU errors: {errors:?}"))
        }
    }

    pub fn report(&self) -> Value {
        json!({ "allocation_count": self.accounting.count.load(Ordering::Relaxed), "peak_payload_bytes": self.accounting.peak.load(Ordering::Relaxed), "live_payload_bytes": self.accounting.live.load(Ordering::Relaxed), "budget_bytes": BUDGET, "allocation_scope": "requested test textures, node/weight uniforms and padded staging buffers; excludes wgpu/driver internals", "validation_errors": self.errors.lock().unwrap().clone(), "measurements": self.measurements })
    }
}

pub fn scalar(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f64::from(f32::from_le_bytes(*p)))
        .collect()
}

pub fn normals(bytes: &[u8]) -> Vec<[f64; 4]> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|p| std::array::from_fn(|c| decode_half(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]]))))
        .collect()
}
