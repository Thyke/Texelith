use super::*;
use crate::viewport_resources::ViewportTargetPlan;
use crate::viewport_shared::{MATERIAL_MAPS, ViewportBindings, ViewportRenderer};

impl Runtime {
    pub(super) fn evaluate_preview(&mut self, work: &Work) -> Result<()> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let scopes = error_scopes(&self.device);
        let outcome = self.render_preview(work);
        let output = finish_scopes(&self.device, &self.shared, scopes, outcome)?;
        let request = work.preview.as_ref().expect("preview work");
        let plan = ViewportTargetPlan::new(request.extent.width, request.extent.height)?;
        let mut state = self.shared.lock().unwrap();
        if state.scheduler.complete(work.id) == Completion::Published {
            let old = state.preview.replace(CompletedPreview {
                job_id: request.job_id,
                generation: request.generation,
                request_id: request.request_id,
                extent: plan.extent,
                scale: plan.scale,
                lease: DisplayLease(Arc::new(DisplayOwner {
                    lease: Some(output),
                    queue: self.queue.clone(),
                })),
            });
            drop(state);
            drop(old);
        }
        Ok(())
    }

    fn render_preview(&mut self, work: &Work) -> Result<Lease> {
        // T-065 extends the same scheduler, admission ledger and publication fence.
        if matches!(
            work.preview.as_ref().expect("preview work").content,
            crate::viewport_shared::PreviewContent::Pbr(_)
                | crate::viewport_shared::PreviewContent::Ibl { .. }
                | crate::viewport_shared::PreviewContent::Mesh { .. }
        ) {
            return self.render_pbr(work);
        }
        let request = work.preview.as_ref().expect("preview work");
        let mut material_maps = MATERIAL_MAPS;
        if let crate::viewport_shared::PreviewContent::Map { map, .. } = request.content
            && !material_maps.contains(&map)
        {
            material_maps[5] = map;
        }
        let plan = ViewportTargetPlan::new(request.extent.width, request.extent.height)?;
        // Consume the placeholder reservation with real budgeted targets. Any
        // unused reserved capacity remains withheld until this draw completes.
        self.viewport_reservation = None;
        self.viewport_bytes = 0;
        self.shared.lock().unwrap().viewport_reserved_bytes = 0;
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| GpuError(e.to_string()))?;

        let mut descriptions = plan.allocations().to_vec();
        descriptions.extend([AllocationDesc::Buffer { bytes: 64 }; 3]);
        let mut map_indices = [None; 6];
        let mut staging_bytes = 3 * 64;
        for (index, kind) in material_maps.iter().enumerate() {
            if !request.material.outputs.contains_key(kind) {
                let desc = AllocationDesc::Texture {
                    extent: Extent {
                        width: 1,
                        height: 1,
                    },
                    format: if index < 2 {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        wgpu::TextureFormat::R32Float
                    },
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                };
                staging_bytes += desc.byte_len()?;
                map_indices[index] = Some(descriptions.len());
                descriptions.push(desc);
            }
        }
        descriptions.push(AllocationDesc::Staging {
            bytes: staging_bytes,
        });
        let reserved = self.shared.lock().unwrap().reservations.viewport_bytes;
        let actual: u64 = descriptions
            .iter()
            .map(|d| d.byte_len())
            .collect::<std::result::Result<Vec<_>, _>>()?
            .iter()
            .sum();
        if reserved > actual {
            descriptions.push(AllocationDesc::Staging {
                bytes: reserved - actual,
            });
        }
        // Atomic admission includes targets, uniforms, fallbacks and queue writes.
        let mut leases = self.allocate(&descriptions)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let maps: [Lease; 6] = std::array::from_fn(|i| {
            if let Some(index) = map_indices[i] {
                leases[index].clone()
            } else {
                request.material.outputs[&material_maps[i]]
                    .0
                    .lease
                    .as_ref()
                    .expect("material lease")
                    .clone()
            }
        });
        if self.viewport.is_none() {
            self.viewport = Some(ViewportRenderer::new(&self.device));
        }
        let key = std::array::from_fn(|i| {
            if i < 6 {
                leases[i].0.id
            } else {
                maps[i - 6].0.id
            }
        });
        {
            let mut pool = self.pool.lock().unwrap();
            pool.backing.preview_bindings.entry(key).or_insert_with(|| {
                self.viewport
                    .as_ref()
                    .expect("initialized viewport")
                    .prepare(
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
                    )
            });
        }
        // Cancellation after these writes still submits/drains their staging.
        self.queue.write_buffer(
            leases[3].buffer(),
            0,
            bytemuck::bytes_of(&request.content.uniform(plan.extent)),
        );
        let mut material_uniform =
            crate::NodeUniform::new(request.material.extent, request.material.extent);
        material_uniform.parameters = [1.0, 0.0, 1.0, 0.5];
        self.queue
            .write_buffer(leases[4].buffer(), 0, bytemuck::bytes_of(&material_uniform));
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        self.queue
            .write_buffer(leases[5].buffer(), 0, bytemuck::cast_slice(&identity));
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared-device preview"),
            });
        self.pool.lock().unwrap().backing.preview_bindings[&key].encode(&mut encoder);
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
