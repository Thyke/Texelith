use super::*;
use crate::viewport_resources::ViewportTargetPlan;
use crate::viewport_shared::{MATERIAL_MAPS, PreviewContent, ViewportBindings, ViewportRenderer};

impl Runtime {
    pub(super) fn render_pbr(&mut self, work: &Work) -> Result<Lease> {
        let request = work.preview.as_ref().expect("preview work");
        let (settings, environment, environment_exposure) = match &request.content {
            PreviewContent::Pbr(settings) => (*settings, None, 0.0),
            PreviewContent::Ibl {
                settings,
                environment,
                exposure,
            } => (*settings, Some(environment.clone()), *exposure),
            PreviewContent::Mesh {
                settings,
                environment,
                exposure,
                ..
            } => (*settings, environment.clone(), *exposure),
            _ => unreachable!("lit request"),
        };
        let plan = ViewportTargetPlan::new(request.extent.width, request.extent.height)?;
        let custom = self.prepare_compare_mesh(&request.content);
        let mesh_index = if custom {
            0
        } else {
            crate::pbr::mesh_index(&request.view.mesh)?
        };
        let meshes = self.pbr_meshes.get_or_insert_with(Default::default);
        let mesh = if custom {
            &self.compare_mesh.as_ref().expect("custom mesh").1
        } else {
            &meshes.0[mesh_index]
        };
        let vertex_bytes = (mesh.vertices.len() * 48) as u64;
        let index_bytes = (mesh.indices.len() * 4) as u64;
        let index_count = mesh.indices.len() as u32;
        self.viewport_reservation = None;
        self.viewport_bytes = 0;
        self.shared.lock().unwrap().viewport_reserved_bytes = 0;
        let environment = environment
            .map(|e| self.environment(work, &e))
            .transpose()?;
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| GpuError(e.to_string()))?;
        let mut descriptions = plan.allocations().to_vec();
        descriptions.extend([AllocationDesc::Buffer { bytes: 64 }; 3]);
        descriptions.extend([
            AllocationDesc::Buffer {
                bytes: vertex_bytes,
            },
            AllocationDesc::Buffer { bytes: index_bytes },
        ]);
        let mut map_indices = [None; 6];
        let mut staging_bytes = 192 + vertex_bytes + index_bytes;
        for (i, kind) in MATERIAL_MAPS.iter().enumerate() {
            if !request.material.outputs.contains_key(kind) {
                let desc = AllocationDesc::Texture {
                    extent: Extent {
                        width: 1,
                        height: 1,
                    },
                    format: if i < 2 {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        wgpu::TextureFormat::R32Float
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                };
                staging_bytes += desc.byte_len()?;
                map_indices[i] = Some(descriptions.len());
                descriptions.push(desc);
            }
        }
        descriptions.push(AllocationDesc::Staging {
            bytes: staging_bytes,
        });
        let actual: u64 = descriptions
            .iter()
            .map(|d| d.byte_len())
            .collect::<std::result::Result<Vec<_>, _>>()?
            .iter()
            .sum();
        let reserved = self.shared.lock().unwrap().reservations.viewport_bytes;
        if reserved > actual {
            descriptions.push(AllocationDesc::Staging {
                bytes: reserved - actual,
            });
        }
        // Admit while every old resource remains reachable by normal pool eviction.
        let mut leases = self.allocate(&descriptions)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let maps: [Lease; 6] = std::array::from_fn(|i| match map_indices[i] {
            Some(index) => leases[index].clone(),
            None => request.material.outputs[&MATERIAL_MAPS[i]]
                .0
                .lease
                .as_ref()
                .expect("material lease")
                .clone(),
        });
        for (i, index) in map_indices.iter().enumerate() {
            if let Some(index) = index {
                let (bytes, count) = fallback(i);
                self.queue.write_texture(
                    leases[*index].texture().as_image_copy(),
                    &bytes[..count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(count as u32),
                        rows_per_image: Some(1),
                    },
                    size(Extent {
                        width: 1,
                        height: 1,
                    }),
                );
            }
        }
        let maps = self.filtered_material(work, maps)?;
        if self.viewport.is_none() {
            self.viewport = Some(ViewportRenderer::new(&self.device));
        }
        let key = std::array::from_fn(|i| {
            if i < 8 {
                leases[i].0.id
            } else {
                maps[i - 8].0.id
            }
        });
        self.pool
            .lock()
            .unwrap()
            .backing
            .pbr_bindings
            .entry(key)
            .or_insert_with(|| {
                self.viewport.as_ref().expect("viewport").prepare_pbr(
                    &self.device,
                    ViewportBindings {
                        hdr: leases[0].view(),
                        depth: leases[1].view(),
                        display: leases[2].view(),
                        scene_uniform: leases[3].buffer(),
                        material_uniform: leases[4].buffer(),
                        object_uniform: leases[5].buffer(),
                        material: std::array::from_fn(|i| maps[i].view()),
                        extent: plan.extent,
                    },
                    leases[6].buffer(),
                    leases[7].buffer(),
                    index_count,
                )
            });
        let mesh = if custom {
            &self.compare_mesh.as_ref().expect("custom mesh").1
        } else {
            &self.pbr_meshes.as_ref().expect("primitives").0[mesh_index]
        };
        self.queue
            .write_buffer(leases[6].buffer(), 0, bytemuck::cast_slice(&mesh.vertices));
        self.queue
            .write_buffer(leases[7].buffer(), 0, bytemuck::cast_slice(&mesh.indices));
        let mut scene = settings.scene(request.view.camera, &request.view.light, plan.extent);
        scene[3][0] = environment_exposure;
        self.queue
            .write_buffer(leases[3].buffer(), 0, bytemuck::cast_slice(&scene));
        let mut material = settings.material(request.material.normal_convention);
        material[3] = crate::compare::tile_uniform(request.view.tiles);
        material[2] = settings.surface.uniform(
            request.view.pom,
            request
                .material
                .outputs
                .contains_key(&texelith_core::MapKind::Height),
        );
        self.queue
            .write_buffer(leases[4].buffer(), 0, bytemuck::cast_slice(&material));
        let identity: [f32; 16] = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        self.queue
            .write_buffer(leases[5].buffer(), 0, bytemuck::cast_slice(&identity));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("T-065 shared-device PBR"),
            });
        if let Some(env) = &environment {
            self.encode_ibl(
                &mut encoder,
                (&leases, &maps, env),
                key,
                plan.extent,
                index_count,
            );
        } else {
            self.pool.lock().unwrap().backing.pbr_bindings[&key].encode(&mut encoder);
        }
        if let Some(env) = environment {
            leases.extend(env);
        }
        leases.extend(maps);
        let (_, done, _) = self.submit(encoder, &leases)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(leases[2].clone())
    }
}
fn fallback(index: usize) -> ([u8; 8], usize) {
    let mut bytes = [0; 8];
    if index < 2 {
        let rgba: [u16; 4] = if index == 0 {
            [0x3800, 0x3800, 0x3800, 0x3c00]
        } else {
            [0, 0, 0x3c00, 0x3c00]
        };
        for (out, value) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(rgba) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        (bytes, 8)
    } else {
        let value: f32 = match index {
            2 | 4 => 1.0,
            3 => 0.0,
            _ => 0.5,
        };
        bytes[..4].copy_from_slice(&value.to_le_bytes());
        (bytes, 4)
    }
}
