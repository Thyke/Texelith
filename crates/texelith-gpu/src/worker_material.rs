use super::*;
use crate::MaterialPlan;
use crate::material::OutputKey;
use std::collections::{BTreeMap, BTreeSet};

impl Runtime {
    pub(super) fn trim_cache(&mut self) {
        let limit = self.config.budget_bytes.min(1024 * 1024 * 1024) * 3 / 8;
        while self.cache.len() > self.config.max_cache_entries
            || self.cache.weight(|lease| {
                let t = lease.texture();
                u64::from(t.width())
                    * u64::from(t.height())
                    * if t.format() == wgpu::TextureFormat::Rgba16Float {
                        8
                    } else {
                        4
                    }
            }) > limit
        {
            if !self.cache.evict_oldest() {
                break;
            }
            self.shared.lock().unwrap().cache_evictions += 1;
        }
    }

    pub(super) fn material_pass(
        &mut self,
        work: &Work,
        operation: Pass,
        input: Lease,
        extent: Extent,
        source: Extent,
    ) -> Result<Lease> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let has_weights = matches!(operation, Pass::Blur { .. });
        let mut descriptions = vec![
            texture_desc(extent, operation.output_format()),
            AllocationDesc::Buffer { bytes: 64 },
            AllocationDesc::Staging {
                bytes: if has_weights { 192 } else { 64 },
            },
        ];
        if has_weights {
            descriptions.push(AllocationDesc::Buffer { bytes: 128 });
        }
        let mut leases = self.allocate(&descriptions)?;
        let key = BindingKey {
            input: input.0.id,
            output: leases[0].0.id,
            uniform: leases[1].0.id,
            weights: has_weights.then(|| leases[3].0.id),
            pipeline: operation.index(),
            source_width: source.width,
            source_height: source.height,
        };
        let mut pool = self.pool.lock().unwrap();
        if let Some(prepared) = pool.backing.bindings.get_mut(&key) {
            prepared.update(operation)?;
        } else {
            let prepared = self.executor.prepare_pooled(
                operation,
                input.texture(),
                leases[0].texture(),
                source,
                (leases[1].buffer(), has_weights.then(|| leases[3].buffer())),
            )?;
            pool.backing.bindings.insert(key, prepared);
        }
        leases.push(input);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        pool.backing.bindings[&key].encode(&mut encoder);
        drop(pool);
        let (_, done, _) = self.submit(encoder, &leases)?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&done)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        Ok(leases[0].clone())
    }

    pub(super) fn evaluate_material(&mut self, work: &Work) -> Result<()> {
        let request = work.material.as_ref().expect("material work");
        let reservations = self.shared.lock().unwrap().reservations;
        let plan = MaterialPlan::new(request, self.config.budget_bytes, reservations).map_err(
            |e| match e {
                crate::MaterialError::BudgetExceeded {
                    required_bytes,
                    budget_bytes,
                } => Failure::Pool(PoolError::BudgetExceeded {
                    required_bytes,
                    budget_bytes,
                }),
                other => Failure::Gpu(other.into()),
            },
        )?;
        let evicted = self
            .cache
            .retain(|key| plan.keys.values().any(|k| *k == key.content));
        self.shared.lock().unwrap().cache_evictions += evicted as u64;
        // Reservation tokens have no backing buffer; they withhold capacity for
        // the shared-device viewport and its IBL allocations.
        if self.viewport_bytes != reservations.viewport_bytes {
            self.viewport_reservation = None;
            self.viewport_bytes = 0;
            self.shared.lock().unwrap().viewport_reserved_bytes = 0;
        }
        self.readback_reservation = None;
        self.shared.lock().unwrap().readback_reserved_bytes = 0;
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| GpuError(e.to_string()))?;
        if reservations.viewport_bytes > 0 && self.viewport_reservation.is_none() {
            let lease = self
                .allocate(&[AllocationDesc::Staging {
                    bytes: reservations.viewport_bytes,
                }])?
                .pop();
            self.viewport_reservation = Some(DisplayLease(Arc::new(DisplayOwner {
                lease,
                queue: self.queue.clone(),
            })));
            self.viewport_bytes = reservations.viewport_bytes;
            self.shared.lock().unwrap().viewport_reserved_bytes = reservations.viewport_bytes;
        }
        let scopes = error_scopes(&self.device);
        let outcome = self.execute_material(work, &plan, reservations);
        let outputs = finish_scopes(&self.device, &self.shared, scopes, outcome)?;
        let bundle = CompletedMaterial {
            job_id: request.job_id,
            request_id: request.request_id,
            generation: request.snapshot.generation,
            extent: request.extent,
            quality: request.quality,
            normal_convention: request
                .outputs
                .get(&MapKind::Normal)
                .map(|output| request.snapshot.graph.output_normal_convention(*output))
                .transpose()?
                .flatten(),
            outputs: outputs
                .into_iter()
                .map(|(kind, lease)| {
                    (
                        kind,
                        DisplayLease(Arc::new(DisplayOwner {
                            lease: Some(lease),
                            queue: self.queue.clone(),
                        })),
                    )
                })
                .collect(),
        };
        let mut state = self.shared.lock().unwrap();
        if state.scheduler.complete(work.id) == Completion::Published {
            state.scheduler.try_latest_material();
            state.retained_materials.retain(|identity| {
                identity
                    .outputs
                    .values()
                    .any(|owner| owner.strong_count() > 0)
            });
            if state.retained_materials.len() == 8 {
                state.retained_materials.pop_front();
            }
            state
                .retained_materials
                .push_back(MaterialIdentity::new(&bundle, request.clone()));
            state.material_identity = Some(MaterialIdentity::new(&bundle, request.clone()));
            let old = state.material.replace(bundle);
            drop(state);
            drop(old);
        }
        Ok(())
    }

    pub(super) fn execute_material(
        &mut self,
        work: &Work,
        plan: &MaterialPlan,
        reservations: ResourceReservations,
    ) -> Result<BTreeMap<MapKind, Lease>> {
        let request = work.material.as_ref().expect("material work");
        let graph = &request.snapshot.graph;
        let mut live = BTreeMap::<OutputKey, Lease>::new();
        let mut needed = BTreeSet::new();
        let mut stack: Vec<_> = request
            .outputs
            .values()
            .copied()
            .map(OutputKey::from)
            .collect();
        while let Some(id) = stack.pop() {
            if !needed.insert(id) {
                continue;
            }
            let node = graph.material_node(id.node).expect("validated node");
            let format = if matches!(id.port.map_kind(), MapKind::Albedo | MapKind::Normal) {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::R32Float
            };
            if let Some(cached) = self.cache.get(&CacheAddress {
                content: plan.keys[&id],
                format,
            }) {
                live.insert(id, cached.clone());
                self.shared.lock().unwrap().cache_hits += 1;
            } else {
                self.shared.lock().unwrap().cache_misses += 1;
                stack.extend(node.inputs.iter().map(|e| OutputKey::from(e.source)));
            }
        }
        for (index, id) in plan.nodes.iter().enumerate() {
            if !needed.contains(id) || live.contains_key(id) {
                continue;
            }
            if !self.current(work) {
                return Err(Failure::Cancelled);
            }
            let node = graph.material_node(id.node).expect("planned node");
            let extent = request.extent;
            let source_extent = graph.material_output_extent(id.0)?;
            let registration = node_registration(node)
                .ok_or_else(|| GpuError("Unsupported material node".into()))?;
            let output = (registration.evaluate)(
                self,
                &NodeContext {
                    work,
                    node,
                    output: id.0,
                    plan,
                    extent,
                    source_extent,
                    live: &live,
                },
            )?;
            // Immutable sets reuse declared maps (notably height) across output
            // requests. These leases remain evictable under the existing LRU cap.
            if request.outputs.values().any(|o| *o == id.0)
                || (work.export_job.is_some()
                    && graph
                        .material_outputs()
                        .is_some_and(|maps| maps.values().any(|o| *o == id.0)))
            {
                self.cache.insert(
                    CacheAddress {
                        content: plan.keys[id],
                        format: output.texture().format(),
                    },
                    output.clone(),
                );
            }
            live.insert(*id, output);
            live.retain(|id, _| plan.last_use[id] > index);
        }
        // Every submitted pass has completed; scratch can now be evicted before
        // streaming one readback. The token is charged alongside all output leases.
        if reservations.readback_bytes > 0 {
            let mut reservation = self.allocate(&[AllocationDesc::Staging {
                bytes: reservations.readback_bytes,
            }])?;
            self.shared.lock().unwrap().readback_reserved_bytes = reservations.readback_bytes;
            self.readback_reservation = Some(DisplayLease(Arc::new(DisplayOwner {
                lease: reservation.pop(),
                queue: self.queue.clone(),
            })));
        }
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let outputs = request
            .outputs
            .iter()
            .map(|(kind, output)| (*kind, live[&OutputKey::from(*output)].clone()))
            .collect();
        // FAZ1_MATERIAL_DISPATCH: T-069 -- public albedo is restricted only after
        // full-size color consumers have finished. Keep their canonical inputs.
        let mut outputs: BTreeMap<MapKind, Lease> = outputs;
        if let Some(color) = outputs.get(&MapKind::Albedo).cloned()
            && (color.texture().width() != request.extent.width
                || color.texture().height() != request.extent.height)
        {
            outputs.insert(
                MapKind::Albedo,
                self.compare_color_proxy(work, color, request.extent)?,
            );
        }
        self.trim_cache();
        Ok(outputs)
    }
}
