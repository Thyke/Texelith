use super::*;
use std::{
    io,
    time::{Duration, Instant},
};
use texelith_io::export_transactions::{self as tx, ExportFile, FailurePoint};

fn error(e: impl ToString) -> io::Error {
    io::Error::other(e.to_string())
}
fn folder_error(destination: &std::path::Path, error: io::Error) -> io::Error {
    let message = match error.kind() {
        io::ErrorKind::PermissionDenied => "Folder is not writable; choose another export folder",
        io::ErrorKind::NotFound => "Folder could not be found; check the export path",
        io::ErrorKind::NotADirectory => "Export path contains a file where a folder is required",
        io::ErrorKind::AlreadyExists => {
            "Destination already exists; choose another name or confirm replacement"
        }
        _ => return error,
    };
    io::Error::new(
        error.kind(),
        format!("{message}\nDestination: {}\n{error}", destination.display()),
    )
}

fn update(shared: &Mutex<Shared>, stage: ExportStage, index: usize, name: &str) {
    if let Some(s) = &mut shared.lock().unwrap().status {
        s.stage = stage;
        s.completed_items = index;
        s.output_name = name.into();
    }
}
pub(super) fn run(gpu: &GpuWorker, shared: &Mutex<Shared>, stop: &AtomicBool, job: Job) {
    let id = job.request.job_id();
    let plan = job.request.plan();
    let capture = job.request.capture();
    let check = || {
        if job.cancel.load(Ordering::Acquire) || stop.load(Ordering::Acquire) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Export cancelled",
            ))
        } else {
            Ok(())
        }
    };
    let mut admitted = false;
    let mut publishing = false;
    let mut records = Vec::new();
    let mut output_identity = plan.outputs[0].name.clone();
    let result = (|| {
        check()?;
        let mut files = Vec::new();
        for (output_index, output) in plan.outputs.iter().enumerate() {
            let stem = tx::expand_name(
                &plan.naming,
                &tx::NameContext {
                    name: &job.options.material_name,
                    map: &output.name,
                    preset: &plan.profile.id,
                    index: output_index as u32,
                },
            )?;
            let extension = match output.format {
                ExportFormat::Png8 | ExportFormat::Png16 => "png",
                ExportFormat::Exr32 => "exr",
                ExportFormat::DdsBc5 => "dds",
            };
            files.push(ExportFile::new(format!("{stem}.{extension}")));
        }
        files.push(ExportFile::new("assignment.json"));
        if let Some(parent) = job
            .options
            .destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| folder_error(&job.options.destination, e))?;
        }
        let mut index = 0;
        let outcome = tx::publish_set(
            &job.options.destination,
            &files,
            tx::PublishOptions {
                collision: if job.options.replace_existing {
                    tx::CollisionPolicy::Replace
                } else {
                    tx::CollisionPolicy::NoOverwrite
                },
                ..Default::default()
            },
            |file, writer| {
                check()?;
                if index == plan.outputs.len() {
                    let manifest = serde_json::json!({
                        "schema_version": 1, "job_id": id.0,
                        "graph_hash": capture.graph_hash, "sources": capture.sources,
                        "generation": capture.generation, "document_revision": capture.document_revision,
                        "plan": plan, "outputs": records,
                    });
                    return writer.write_all(&serde_json::to_vec_pretty(&manifest).map_err(error)?);
                }
                output_identity = plan.outputs[index].name.clone();
                crate::export_perf::mark(crate::export_perf::Event::Waiting(
                    output_identity.clone(),
                ));
                if !admitted {
                    #[cfg(test)]
                    crate::batch_jobs::regressions::export_phase(
                        &job.options.destination,
                        "gpu",
                        capture,
                    );
                    let started = Instant::now();
                    loop {
                        check()?;
                        match gpu.request_export(job.request.clone()) {
                            Ok(()) => {
                                admitted = true;
                                break;
                            }
                            Err(texelith_gpu::SaveError::Busy)
                                if started.elapsed() < Duration::from_secs(90) =>
                            {
                                let _ = gpu.try_export_progress();
                                std::thread::sleep(Duration::from_millis(2));
                            }
                            Err(e) => return Err(error(e)),
                        }
                    }
                }
                let started = Instant::now();
                let plane = loop {
                    check()?;
                    if let Some(progress) = gpu.try_export_progress() {
                        if progress.job_id != id {
                            return Err(error("Unexpected export job progress"));
                        }
                        if progress.stage == ExportStage::Failed {
                            return Err(error(
                                progress.error.unwrap_or("GPU export failed".into()),
                            ));
                        }
                        update(shared, progress.stage, index, &output_identity);
                    }
                    if let Some(plane) = gpu.try_export_plane() {
                        break plane;
                    }
                    if started.elapsed() > Duration::from_secs(90) {
                        return Err(error("Export evaluation/readback timed out"));
                    }
                    std::thread::sleep(Duration::from_millis(2));
                };
                crate::export_perf::mark(crate::export_perf::Event::Readback(
                    output_identity.clone(),
                ));
                let encoded = (|| {
                    if plane.job_id != id
                        || plane.item_index != index
                        || plane.graph_hash != capture.graph_hash
                        || plane.generation != capture.generation
                        || plane.document_revision != capture.document_revision
                        || plane.output_name != output_identity
                    {
                        return Err(error("Export plane identity mismatch"));
                    }
                    update(shared, ExportStage::Encoding, index, &output_identity);
                    check()?;
                    let reservation = crate::batch_jobs::export_phase_reservation(
                        plane.metadata.extent.width,
                        plane.metadata.extent.height,
                        u64::from(plane.channels),
                        plane.format,
                    )
                    .map_err(error)?;
                    let mut cpu =
                        crate::batch_jobs::memory::reserve_encode(reservation, || check().is_ok())
                            .map_err(|e| check().err().unwrap_or_else(|| error(e)))?;
                    check()?;
                    crate::export_perf::mark(crate::export_perf::Event::Encoding(
                        output_identity.clone(),
                    ));
                    let bytes = encode::encode(&plane).map_err(error)?;
                    crate::export_perf::mark(crate::export_perf::Event::Encoded(
                        output_identity.clone(),
                    ));
                    cpu.retain(bytes.capacity() as u64);
                    check()?;
                    writer.write_all(&bytes)?;
                    records.push(serde_json::json!({
                    "name": plane.output_name, "file": file.name(), "format": plane.format,
                    "channels": plane.channels, "extent": plane.metadata.extent,
                    "semantics": plane.metadata.semantics, "valid_content": plane.metadata.valid_content,
                    "uv_scale": plane.metadata.uv_scale, "padding": plane.metadata.padding,
                    "displacement": plane.metadata.displacement,
                    "import_hints": plan.outputs[index].import_hints,
                    "encoded_bytes": bytes.len(),
                }));
                    Ok(())
                })();
                gpu.acknowledge_export(
                    plane,
                    encoded.as_ref().map(|_| ()).map_err(ToString::to_string),
                )
                .map_err(error)?;
                encoded?;
                index += 1;
                Ok(())
            },
            |point| {
                if matches!(
                    point,
                    FailurePoint::AfterEncode(_)
                        | FailurePoint::AfterFlush(_)
                        | FailurePoint::AfterBackupRename
                        | FailurePoint::AfterPublishRename
                ) {
                    check()?;
                }
                if point == FailurePoint::AfterFlush(plan.outputs.len()) {
                    #[cfg(test)]
                    crate::batch_jobs::regressions::export_phase(
                        &job.options.destination,
                        "publication",
                        capture,
                    );
                    gpu.begin_export_publication(id).map_err(error)?;
                    publishing = true;
                    update(
                        shared,
                        ExportStage::Publishing,
                        plan.outputs.len(),
                        "assignment.json",
                    );
                }
                Ok(())
            },
        ).map_err(|e| folder_error(&job.options.destination, e))?;
        match outcome {
            tx::PublishOutcome::Committed => {
                // Retain recovery journals on failure, remove them only after
                // publication and all backup cleanup have succeeded.
                std::fs::remove_file(job.options.destination.join(".texelith-transaction"))?;
                crate::export_perf::mark(crate::export_perf::Event::Published);
                Ok(())
            }
            tx::PublishOutcome::CommittedWithCleanup { .. } => Err(error(
                "Export committed; backup cleanup requires recovery before another replacement",
            )),
        }
    })();
    if publishing {
        let _ = gpu.finish_export(id, result.as_ref().map(|_| ()).map_err(ToString::to_string));
    } else if admitted {
        let _ = gpu.cancel_export(id);
    }
    // Retire mailbox progress before admitting a subsequent job; GPU fences remain worker-owned.
    let _ = gpu.try_export_progress();
    let mut state = shared.lock().unwrap();
    if let Some(status) = &mut state.status {
        status.stage = match &result {
            Ok(()) => ExportStage::Completed,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => ExportStage::Cancelled,
            Err(_) => ExportStage::Failed,
        };
        status.error = result.err().map(|e| format!("{output_identity}: {e}"));
        if status.stage == ExportStage::Completed {
            status.completed_items = status.total_items;
        }
    }
    state.busy = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_failures_keep_specific_messages_and_os_detail() {
        for (kind, reason) in [
            (io::ErrorKind::PermissionDenied, "not writable"),
            (io::ErrorKind::NotFound, "could not be found"),
            (io::ErrorKind::NotADirectory, "file where a folder"),
            (io::ErrorKind::AlreadyExists, "already exists"),
        ] {
            let error = folder_error(
                std::path::Path::new("captured-folder"),
                io::Error::new(kind, "technical OS detail"),
            );
            assert_eq!(error.kind(), kind);
            let detail = error.to_string();
            assert!(detail.lines().next().unwrap().contains(reason));
            assert!(detail.contains("captured-folder"));
            assert!(detail.contains("technical OS detail"));
        }
        let gpu_error = folder_error(
            std::path::Path::new("set"),
            io::Error::other("GPU export failed"),
        );
        assert_eq!(gpu_error.to_string(), "GPU export failed");
    }
}
