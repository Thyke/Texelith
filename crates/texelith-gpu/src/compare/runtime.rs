use super::*;
impl Runtime {
    pub(super) fn prepare_compare_mesh(
        &mut self,
        content: &crate::viewport_shared::PreviewContent,
    ) -> bool {
        let crate::viewport_shared::PreviewContent::Mesh { mesh, .. } = content else {
            return false;
        };
        if self
            .compare_mesh
            .as_ref()
            .is_none_or(|(source, _)| !Arc::ptr_eq(source, mesh))
        {
            self.compare_mesh = Some((
                mesh.clone(),
                crate::pbr::PackedMesh {
                    vertices: mesh
                        .vertices
                        .iter()
                        .map(|v| {
                            [
                                v.position[0],
                                v.position[1],
                                v.position[2],
                                v.normal[0],
                                v.normal[1],
                                v.normal[2],
                                v.uv[0],
                                v.uv[1],
                                v.tangent[0],
                                v.tangent[1],
                                v.tangent[2],
                                v.tangent[3],
                            ]
                        })
                        .collect(),
                    indices: mesh.indices.clone(),
                },
            ));
        }
        true
    }
    pub(super) fn compare_color_proxy(
        &mut self,
        work: &Work,
        input: Lease,
        extent: Extent,
    ) -> Result<Lease> {
        let leases = self.allocate(&[texture_desc(extent, wgpu::TextureFormat::Rgba16Float)])?;
        let pipeline = self.compare_proxy.get_or_insert_with(|| {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("T-069 linear area proxy"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../../shaders/compare/proxy.wgsl").into(),
                    ),
                });
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("T-069 proxy ABI"),
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
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("T-069 proxy"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("T-069 proxy"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("T-069 proxy"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(leases[0].view()),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
        let (_, done, _) = self.submit(encoder, &[leases[0].clone(), input])?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(leases[0].clone())
    }
}
