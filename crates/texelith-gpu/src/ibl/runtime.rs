use super::*;
pub(super) struct CachedEnvironment {
    key: [u64; 2],
    pub(super) outputs: Vec<Lease>,
}
impl Runtime {
    pub(super) fn encode_ibl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: (&[Lease], &[Lease; 6], &[Lease]),
        key: [AllocationId; 14],
        extent: Extent,
        index_count: u32,
    ) {
        let (leases, maps, env) = resources;
        let key: [AllocationId; 17] =
            std::array::from_fn(|i| if i < 14 { key[i] } else { env[i - 13].0.id });
        let mut pool = self.pool.lock().unwrap();
        pool.backing
            .ibl_bindings
            .entry(key)
            .or_insert_with(|| {
                self.viewport.as_ref().expect("viewport").prepare_pbr_ibl(
                    &self.device,
                    crate::viewport_shared::ViewportBindings {
                        hdr: leases[0].view(),
                        depth: leases[1].view(),
                        display: leases[2].view(),
                        scene_uniform: leases[3].buffer(),
                        material_uniform: leases[4].buffer(),
                        object_uniform: leases[5].buffer(),
                        material: std::array::from_fn(|i| maps[i].view()),
                        extent,
                    },
                    (leases[6].buffer(), leases[7].buffer(), index_count),
                    [env[1].view(), env[2].view(), env[3].view()],
                )
            })
            .encode(encoder);
    }
    pub(super) fn environment(
        &mut self,
        work: &Work,
        environment: &crate::ibl::Environment,
    ) -> Result<Vec<Lease>> {
        if let Some(cached) = &self.ibl
            && cached.key == environment.key()
        {
            return Ok(cached.outputs.clone());
        }
        // Retire the single previous environment before admitting a replacement.
        self.ibl = None;
        let mut desc = crate::ibl::output_allocations().to_vec();
        let bytes = environment.pixels().len() as u64 * 16;
        desc.extend([
            AllocationDesc::Buffer { bytes },
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Staging { bytes: bytes + 64 },
        ]);
        let leases = self.allocate(&desc)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let convolution_start = Instant::now();
        let kernel = self
            .ibl_kernel
            .get_or_insert_with(|| crate::ibl::IblKernel::new(&self.device));
        self.queue.write_buffer(
            leases[4].buffer(),
            0,
            bytemuck::cast_slice(environment.pixels()),
        );
        let prepared: Vec<_> = crate::ibl::passes()
            .map(|(mode, _, _, size)| {
                kernel.prepare(
                    &self.device,
                    leases[4].buffer(),
                    leases[if mode == 0 { 2 } else { 0 }].view(),
                    leases[mode as usize].view(),
                    leases[5].buffer(),
                    size,
                )
            })
            .collect();
        for ((mode, face, mip, size), pass) in crate::ibl::passes().zip(prepared) {
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
            let e = environment.extent();
            let uniform: [u32; 16] = [
                mode, face, mip, size, e.width, e.height, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            self.queue
                .write_buffer(leases[5].buffer(), 0, bytemuck::cast_slice(&uniform));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("T-067 bounded face/mip"),
                });
            pass.encode(&mut encoder);
            let (_, done, _) = self.submit(encoder, &leases)?;
            self.shared.lock().unwrap().compute_submissions += 1;
            self.wait(&done)?;
        }
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let outputs = leases[..4].to_vec();
        self.shared.lock().unwrap().ibl_convolution_ns =
            convolution_start.elapsed().as_nanos() as u64;
        self.ibl = Some(CachedEnvironment {
            key: environment.key(),
            outputs: outputs.clone(),
        });
        Ok(outputs)
    }
}
