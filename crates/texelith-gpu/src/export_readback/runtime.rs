use super::*;
use crate::{ExportPlane, ExportRequest, ExportStage, MaterialPlan, export_readback::PlaneCharge};
use texelith_core::{ChannelSource, ExportCapture, OutputRef};

pub(super) struct State {
    request: ExportRequest,
    item: usize,
    maps: Vec<(OutputRef, Lease)>,
    carrier: Option<Lease>,
    packing: Option<crate::export_shared::PreparedOutput>,
    pass: usize,
    metadata: Option<crate::ExportPackingMetadata>,
}
impl Runtime {
    pub(super) fn retire_export(&mut self) {
        let active = self
            .shared
            .lock()
            .unwrap()
            .export
            .request
            .as_ref()
            .map(|r| r.job_id);
        if self
            .export
            .as_ref()
            .is_some_and(|s| Some(s.request.job_id) != active)
        {
            self.export = None;
        }
        self.shared.lock().unwrap().export.retiring = self.export.is_some() || active.is_some();
    }

    pub(super) fn export_step(&mut self) -> bool {
        let (request, item) = {
            let state = self.shared.lock().unwrap();
            let Some(request) = &state.export.request else {
                return false;
            };
            if state.export.outstanding || state.export.next_item >= request.plan.outputs.len() {
                return false;
            }
            (request.clone(), state.export.next_item)
        };
        let snapshot = Arc::new(EvaluationSnapshot {
            generation: request.capture.generation,
            graph: request.capture.graph.clone(),
        });
        let work = self.shared.lock().unwrap().scheduler.start_export(
            RenderRequest {
                request_id: request.job_id.0,
                snapshot: snapshot.clone(),
                output: NodeId(0),
                extent: request.plan.source_extent,
                quality: Quality::Full,
            },
            request.job_id,
        );
        let Some(work) = work else {
            return false;
        };
        let scopes = error_scopes(&self.device);
        let outcome = (|| {
            if self.export.is_none() {
                self.readback_reservation = None;
                let verified = ExportCapture::new(
                    &snapshot,
                    request.capture.document_revision,
                    &request.plan,
                )?;
                if verified != *request.capture {
                    return Err(GpuError(
                        "export capture identity does not match graph/source hashes".into(),
                    )
                    .into());
                }
                if request.capture.sources.iter().any(|source| {
                    !self
                        .sources
                        .iter()
                        .any(|(installed, _, _)| installed == source)
                }) {
                    return Err(
                        GpuError("immutable export source hash is not installed".into()).into(),
                    );
                }
                self.export = Some(State {
                    request,
                    item,
                    maps: Vec::new(),
                    carrier: None,
                    packing: None,
                    pass: 0,
                    metadata: None,
                });
                self.shared.lock().unwrap().export.retiring = true;
            }
            let mut export = self.export.take().unwrap();
            if export.item != item {
                export.item = item;
                export.maps.clear();
                export.carrier = None;
                export.packing = None;
                export.metadata = None;
                export.pass = 0;
            }
            let result = self.advance_export(&work, &mut export);
            self.export = Some(export);
            result
        })();
        let outcome = finish_scopes(&self.device, &self.shared, scopes, outcome);
        let mut shared = self.shared.lock().unwrap();
        shared.readback_reserved_bytes = 0;
        match outcome {
            Ok(Some(plane)) if shared.scheduler.is_current(&work) => {
                shared.export.charge = Arc::downgrade(&plane.charge);
                shared.export.plane = Some(plane);
                shared.export.outstanding = true;
                shared.export.progress(ExportStage::Encoding, None);
                self.export.as_mut().unwrap().carrier = None;
                self.export.as_mut().unwrap().packing = None;
            }
            Ok(_) => {}
            Err(error) => {
                if matches!(error, Failure::Cancelled) {
                    shared.cancelled_pass_batches += 1;
                } else if shared
                    .export
                    .request
                    .as_ref()
                    .is_some_and(|r| Some(r.job_id) == work.export_job)
                {
                    let message = match error {
                        Failure::Gpu(e) => e.to_string(),
                        Failure::Pool(e) => e.to_string(),
                        Failure::Cancelled => unreachable!(),
                    };
                    shared.export.progress(ExportStage::Failed, Some(message));
                    shared.export.request = None;
                    shared.scheduler.cancel_export();
                }
            }
        }
        shared.scheduler.cancel_active(work.id);
        drop(shared);
        self.retire_export();
        true
    }

    fn transient(&mut self, descriptions: &[AllocationDesc]) -> Result<Vec<Lease>> {
        let mut leases = self.allocate(descriptions)?;
        for lease in &mut leases {
            Arc::get_mut(&mut lease.0).expect("new owner").transient = true;
        }
        Ok(leases)
    }

    fn advance_export(&mut self, work: &Work, export: &mut State) -> Result<Option<ExportPlane>> {
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let plan = &export.request.plan;
        let output = &plan.outputs[export.item];
        // A channel can refer to a different node of the same map kind. Evaluate
        // each distinct OutputRef rather than collapsing them into MaterialOutputs.
        let needed = output
            .channels
            .iter()
            .find_map(|channel| match channel.source {
                ChannelSource::Map(reference, _)
                    if !export.maps.iter().any(|(r, _)| *r == reference) =>
                {
                    Some(reference)
                }
                _ => None,
            });
        if let Some(reference) = needed.filter(|_| export.carrier.is_none()) {
            let request = MaterialRequest {
                job_id: export.request.job_id,
                request_id: work.request.request_id,
                snapshot: work.request.snapshot.clone(),
                outputs: [(reference.port.map_kind(), reference)].into(),
                extent: plan.source_extent,
                quality: Quality::Full,
            };
            let planned = MaterialPlan::new(
                &request,
                self.config.budget_bytes,
                ResourceReservations::default(),
            )
            .map_err(GpuError::from)?;
            let mut evaluation = work.clone();
            evaluation.material = Some(request);
            let maps =
                self.execute_material(&evaluation, &planned, ResourceReservations::default())?;
            export
                .maps
                .push((reference, maps[&reference.port.map_kind()].clone()));
            return Ok(None);
        }
        if export.carrier.is_none() {
            // Keep-size numeric packing needs only 288 bytes of uniforms and
            // optionally a 1x1 dummy. Admit that headroom with the carrier so
            // normal LRU eviction preserves reusable ancestors when they fit.
            let bounded_packing = matches!(plan.size, texelith_core::SizePolicy::Keep)
                && output.format != texelith_core::ExportFormat::DdsBc5;
            if !bounded_packing {
                self.cache.clear();
            }
            // Retire completed pyramid/scratch owners before LRU pressure can
            // discard the much smaller map results needed by subsequent outputs.
            while self.nodes.evict_one() {}
            let layout = crate::export_shared::plane_layout(plan, export.item)?;
            let mut descriptions = vec![texture_desc(
                layout.extent,
                wgpu::TextureFormat::Rgba32Float,
            )];
            if bounded_packing {
                descriptions.push(AllocationDesc::Staging { bytes: 512 });
            }
            let mut admitted = self.transient(&descriptions)?;
            let carrier = admitted.remove(0);
            drop(admitted); // Release the headroom token before private packing admission.
            let sources = export
                .maps
                .iter()
                .map(|(reference, lease)| {
                    Ok(crate::ExportPackingSource {
                        reference: *reference,
                        texture: lease.texture(),
                        boundary: crate::export_shared::source_boundary(
                            &work.request.snapshot.graph,
                            *reference,
                        ),
                        normal_convention: work
                            .request
                            .snapshot
                            .graph
                            .output_normal_convention(*reference)?
                            .unwrap_or(texelith_core::NormalConvention::OpenGl),
                    })
                })
                .collect::<std::result::Result<Vec<_>, texelith_core::CoreError>>()?;
            let packing = crate::export_shared::prepare(
                &self.export_kernel,
                &self.executor,
                plan,
                export.item,
                &sources,
                carrier.texture(),
                self.pool.clone(),
            )?;
            export.metadata = Some(packing.metadata().clone());
            export.packing = Some(packing);
            export.carrier = Some(carrier);
            export.pass = 0;
        }
        if let Some(packing) = &mut export.packing
            && export.pass < packing.pass_count()
        {
            // Preparation queued uniform writes, so flush this pass even if an
            // edit arrived during preparation. Cancellation is checked after its fence.
            let mut encoder = self.device.create_command_encoder(&Default::default());
            packing.encode_pass(export.pass, &mut encoder);
            packing.mark_submitted(FenceId(
                self.fence
                    .checked_add(1)
                    .ok_or_else(|| GpuError("queue fence exhausted".into()))?,
            ))?;
            let mut leases: Vec<_> = export.maps.iter().map(|(_, lease)| lease.clone()).collect();
            leases.push(export.carrier.as_ref().unwrap().clone());
            let (_, finished, _) = self.submit(encoder, &leases)?;
            self.shared.lock().unwrap().compute_submissions += 1;
            self.wait(&finished)?;
            export.pass += 1;
            if export.pass == packing.pass_count() {
                packing.finish_compute();
                export.maps.clear();
                self.shared
                    .lock()
                    .unwrap()
                    .export
                    .progress(ExportStage::Readback, None);
            }
            return Ok(None);
        }
        let prepared = export.packing.as_mut().expect("prepared output");
        let layout = prepared.readback_layout();
        let readback = AllocationDesc::Readback {
            extent: layout.extent,
            bytes_per_pixel: layout.bytes_per_pixel,
        };
        if readback.byte_len()? != layout.staging_bytes {
            return Err(GpuError("export adapter staging size mismatch".into()).into());
        }
        let leases = self.transient(&[
            readback,
            // Also bound the CPU plane while its IO consumer owns it.
            AllocationDesc::Staging {
                bytes: layout.packed_bytes,
            },
        ])?;
        self.shared.lock().unwrap().readback_reserved_bytes = layout.staging_bytes;
        let carrier = export.carrier.as_ref().unwrap();
        let buffer = leases[0].buffer();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        prepared.encode_readback(&mut encoder, carrier.texture(), buffer);
        prepared.mark_submitted(FenceId(
            self.fence
                .checked_add(1)
                .ok_or_else(|| GpuError("queue fence exhausted".into()))?,
        ))?;
        let (_, finished, _) = self.submit(encoder, &[leases[0].clone(), carrier.clone()])?;
        self.shared.lock().unwrap().compute_submissions += 1;
        self.wait(&finished)?;
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        let (tx, rx) = bounded(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.try_send(result);
            });
        let result: Result<Vec<u8>> = (|| {
            self.wait(&rx)?.map_err(|e| GpuError(e.to_string()))?;
            let mapped = buffer
                .slice(..)
                .get_mapped_range()
                .map_err(|e| GpuError(e.to_string()))?;
            Ok(prepared.readback_bytes(&mapped)?)
        })();
        buffer.unmap();
        let pixels = result?;
        if pixels.len() as u64 != layout.packed_bytes {
            return Err(GpuError("export adapter CPU plane size mismatch".into()).into());
        }
        if !self.current(work) {
            return Err(Failure::Cancelled);
        }
        self.pool.lock().unwrap().retain(leases[1].0.id)?;
        let charge = Arc::new(PlaneCharge {
            pool: self.pool.clone(),
            id: leases[1].0.id,
            job: export.request.job_id,
            item: export.item,
        });
        Ok(Some(ExportPlane {
            job_id: export.request.job_id,
            item_index: export.item,
            output_name: output.name.clone(),
            generation: export.request.capture.generation,
            graph_hash: export.request.capture.graph_hash,
            document_revision: export.request.capture.document_revision,
            format: output.format,
            channels: layout.channels,
            payload: prepared.payload(),
            metadata: export.metadata.as_ref().unwrap().clone(),
            pixels,
            charge,
        }))
    }
}
