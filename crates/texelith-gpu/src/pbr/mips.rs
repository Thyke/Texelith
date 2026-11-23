use super::*;

type MaterialKey = std::sync::Weak<CompletedMaterial>;
type FilteredMaterial = (MaterialKey, [Lease; 6]);

#[derive(Default)]
pub(super) struct State {
    pub(super) cached: Option<FilteredMaterial>,
    pipeline: Option<wgpu::ComputePipeline>,
}

impl Runtime {
    pub(super) fn filtered_material(
        &mut self,
        work: &Work,
        maps: [Lease; 6],
    ) -> Result<[Lease; 6]> {
        // Pool allocation IDs and per-document request counters can both be
        // reused. Weak bundle identity cannot alias a replacement allocation.
        let key = Arc::downgrade(&work.preview.as_ref().expect("preview").material);
        if let Some((old, filtered)) = &self.preview_mips.cached
            && old.ptr_eq(&key)
        {
            return Ok(filtered.clone());
        }
        // The previous cache stays visible to admission/eviction until allocation.
        let descriptions: Vec<_> = maps
            .iter()
            .map(|map| AllocationDesc::MipTexture {
                extent: Extent {
                    width: map.texture().width(),
                    height: map.texture().height(),
                },
            })
            .collect();
        let filtered = self.allocate(&descriptions)?;
        if self.preview_mips.pipeline.is_none() {
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("viewport mip ABI"),
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
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("viewport mip pipeline"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("viewport area mip filter"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../../shaders/viewport_shared/mips.wgsl").into(),
                    ),
                });
            self.preview_mips.pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("viewport area mip filter"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
        }
        let pipeline = self.preview_mips.pipeline.as_ref().expect("mip pipeline");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (source, target) in maps.iter().zip(&filtered) {
            for level in 0..target.texture().mip_level_count() {
                let view = |texture: &wgpu::Texture, mip| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                };
                let input = if level == 0 {
                    view(source.texture(), 0)
                } else {
                    view(target.texture(), level - 1)
                };
                let output = view(target.texture(), level);
                let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewport mip level"),
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(
                    (target.texture().width() >> level).max(1).div_ceil(8),
                    (target.texture().height() >> level).max(1).div_ceil(8),
                    1,
                );
            }
        }
        let mut leases = filtered.clone();
        leases.extend(maps);
        let (_, done, _) = self.submit(encoder, &leases)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let filtered: [Lease; 6] = filtered
            .try_into()
            .unwrap_or_else(|_| unreachable!("six maps"));
        self.preview_mips.cached = Some((key, filtered.clone()));
        Ok(filtered)
    }
}
