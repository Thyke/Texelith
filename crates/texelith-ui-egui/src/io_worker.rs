use crossbeam_channel::{Receiver, Sender, bounded, select};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::JoinHandle,
};
use texelith_core::{Extent, ImportToken, InputInterpretation, MapKind, SourceDescriptor};
use texelith_gpu::{CompletedReadback, SourceUpload};

type PresetResult = Result<(crate::packing_editor::PresetLibrary, Option<String>), String>;

/// Preset persistence is serialized separately from image decode/readback, with
/// one request/result slot. No JSON or filesystem work runs on the frame thread.
pub(crate) struct PresetIo {
    pub requests: Sender<texelith_core::ExportPreset>,
    pub results: Receiver<PresetResult>,
    stop: Sender<()>,
}
impl PresetIo {
    pub fn spawn(path: Option<PathBuf>) -> Result<Self, String> {
        let (requests, receive) = bounded::<texelith_core::ExportPreset>(1);
        let (send, results) = bounded(1);
        let (stop, stopped) = bounded(1);
        std::thread::Builder::new().name("texelith-preset-io".into()).spawn(move || {
            let path = path.map(Ok).unwrap_or_else(user_preset_path);
            let mut library = path.as_ref().map_err(Clone::clone).and_then(|p| load_presets(p));
            let initial = library.clone().map(|l| (l, None));
            select! { recv(stopped) -> _ => return, send(send, initial) -> r => if r.is_err() { return } }
            loop {
                let preset = select! { recv(stopped) -> _ => break, recv(receive) -> r => match r { Ok(r) => r, Err(_) => break } };
                let result = (|| {
                    let mut updated = library.as_ref().map_err(Clone::clone)?.clone();
                    updated.presets.retain(|p| p.profile.id != preset.profile.id);
                    let id = preset.profile.id.clone();
                    updated.presets.push(preset);
                    updated.validate()?;
                    let document = serde_json::to_value(&updated).map_err(|e| e.to_string())?;
                    let path = path.as_ref().map_err(Clone::clone)?;
                    let parent = path.parent().ok_or("Invalid preset path")?;
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    match texelith_io::project_io::save_json(path, &document).map_err(|e| e.to_string())? {
                        texelith_io::project_io::SaveOutcome::Committed => {},
                        texelith_io::project_io::SaveOutcome::CommittedWithCleanup { error, .. } => {
                            library = Ok(updated);
                            return Err(format!("Preset committed; recovery required: {error}"));
                        }
                    }
                    library = Ok(updated.clone());
                    Ok((updated, Some(id)))
                })();
                select! { recv(stopped) -> _ => break, send(send, result) -> r => if r.is_err() { break } }
            }
        }).map_err(|e| e.to_string())?;
        Ok(Self {
            requests,
            results,
            stop,
        })
    }
}
impl Drop for PresetIo {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
    }
}

fn user_preset_path() -> Result<PathBuf, String> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
    };
    Ok(base
        .ok_or("User configuration directory unavailable")?
        .join("Texelith")
        .join("export-presets-v1.json"))
}
fn load_presets(path: &std::path::Path) -> Result<crate::packing_editor::PresetLibrary, String> {
    use texelith_io::project_io::{ProjectIoError, read_json};
    let value = match read_json(path) {
        Ok(value) => value,
        Err(ProjectIoError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Default::default());
        }
        Err(error) => return Err(error.to_string()),
    };
    let library: crate::packing_editor::PresetLibrary =
        serde_json::from_value(value).map_err(|e| e.to_string())?;
    library.validate()?;
    Ok(library)
}

pub(crate) enum IoRequest {
    Load {
        token: ImportToken,
        path: PathBuf,
        interpretation: InputInterpretation,
    },
    Save {
        path: PathBuf,
        readback: CompletedReadback,
    },
}
pub(crate) enum IoResult {
    Loaded {
        token: ImportToken,
        result: Result<SourceUpload, String>,
    },
    Saved {
        generation: u64,
        extent: Extent,
        kind: MapKind,
        result: Result<(), String>,
    },
}
pub(crate) struct IoWorker {
    pub requests: Sender<IoRequest>,
    pub results: Receiver<IoResult>,
    pub generation: Arc<AtomicU64>,
    stop: Sender<()>,
    thread: Mutex<Option<JoinHandle<()>>>,
}
impl IoWorker {
    pub fn spawn() -> Result<Self, String> {
        Self::spawn_with_fixture(false)
    }
    pub fn spawn_with_fixture(generated_fixture: bool) -> Result<Self, String> {
        let (requests, receive) = bounded(1);
        let (send, results) = bounded(1);
        let (stop, stopped) = bounded(1);
        let generation = Arc::new(AtomicU64::new(0));
        let current = generation.clone();
        let thread = std::thread::Builder::new().name("texelith-image-io".into()).spawn(move || loop {
            let request = select! { recv(stopped) -> _ => break, recv(receive) -> r => match r { Ok(r) => r, Err(_) => break } };
            let cpu = crate::batch_jobs::memory::reserve();
            let result = match request {
                IoRequest::Load { token, path, interpretation } => {
                    let result = (|| {
                        let intent = match interpretation {
                            InputInterpretation::SrgbPhoto => texelith_io::ColorIntent::PhotoSrgb,
                            InputInterpretation::LinearHeight => texelith_io::ColorIntent::LinearHeight,
                            InputInterpretation::LinearNormal => return Err("Normal import is unsupported in this phase".to_string()),
                        };
                        let image = if generated_fixture {
                            let mut rgba8 = Vec::with_capacity(4096 * 4096 * 4);
                            for y in 0..4096_u32 { for x in 0..4096_u32 {
                                rgba8.extend_from_slice(&[(x.wrapping_mul(17) ^ y.wrapping_mul(3)) as u8,
                                    (y.wrapping_mul(13) ^ (x >> 3)) as u8, (x.wrapping_add(y).wrapping_mul(7)) as u8,255]);
                            } }
                            texelith_io::DecodedImage { width:4096,height:4096,rgba8,color_intent:intent }
                        } else { texelith_io::decode_path(path, intent).map_err(|e| e.to_string())? };
                        let extent = Extent { width: image.width, height: image.height };
                        // Both hashing and the one immutable Arc allocation are off the UI thread.
                        let descriptor = SourceDescriptor::from_rgba8(&image.rgba8, extent, interpretation).map_err(|e| e.to_string())?;
                        Ok(SourceUpload { descriptor, pixels: image.rgba8.into() })
                    })();
                    IoResult::Loaded { token, result }
                }
                IoRequest::Save { path, readback } => {
                    let result = save(&path, &readback, &current);
                    IoResult::Saved { generation: readback.generation, extent: readback.extent, kind: readback.kind, result }
                }
            };
            drop(cpu);
            select! { recv(stopped) -> _ => break, send(send, result) -> r => if r.is_err() { break } }
        }).map_err(|e| e.to_string())?;
        Ok(Self {
            requests,
            results,
            generation,
            stop,
            thread: Mutex::new(Some(thread)),
        })
    }
    pub fn close(&self) {
        let _ = self.stop.try_send(());
    }
    pub fn join(&self) -> Result<(), String> {
        self.close();
        if let Some(thread) = self.thread.lock().unwrap().take() {
            thread
                .join()
                .map_err(|_| "I/O worker panicked".to_string())?;
        }
        Ok(())
    }
}
impl Drop for IoWorker {
    fn drop(&mut self) {
        self.close();
    }
}

fn save(
    path: &std::path::Path,
    readback: &CompletedReadback,
    current: &AtomicU64,
) -> Result<(), String> {
    let fresh = || current.load(Ordering::Acquire) == readback.generation;
    if !fresh() {
        return Err("Save cancelled: document changed".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let staging = parent.join(format!(
        ".texelith-ui-{}-{}.png",
        std::process::id(),
        readback.save_id
    ));
    let encoded = match readback.kind {
        MapKind::Normal => texelith_io::write_normal_png(
            &staging,
            readback.extent.width,
            readback.extent.height,
            &readback.pixels,
            false,
        ),
        MapKind::Height => {
            let values: Vec<u16> = readback
                .pixels
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_be_bytes([p[0], p[1]]))
                .collect();
            texelith_io::write_height_png(
                &staging,
                readback.extent.width,
                readback.extent.height,
                &values,
                false,
            )
        }
        _ => {
            return Err(format!(
                "Save of {:?} is unsupported in this phase",
                readback.kind
            ));
        }
    };
    encoded.map_err(|e| e.to_string())?;
    let result = (|| {
        if !fresh() {
            return Err("Save cancelled: document changed".into());
        }
        // Commit only complete encoded files. Retain a rollback copy on Windows.
        if path.is_dir() {
            return Err("PNG destination is a directory".into());
        }
        let backup = staging.with_extension("backup");
        let existed = path.exists();
        if existed {
            std::fs::rename(path, &backup).map_err(|e| e.to_string())?;
        }
        if let Err(error) = std::fs::rename(&staging, path) {
            if existed {
                let _ = std::fs::rename(&backup, path);
            }
            return Err(error.to_string());
        }
        if existed {
            std::fs::remove_file(backup).map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(staging);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn directory(name: &str) -> PathBuf {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/evidence/T-019")
            .join(format!("test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
    fn readback(kind: MapKind, pixels: Vec<u8>) -> CompletedReadback {
        CompletedReadback {
            save_id: 1,
            request_id: 2,
            generation: 3,
            output: texelith_core::NodeId(3),
            extent: Extent {
                width: 2,
                height: 1,
            },
            kind,
            pixels,
            requested_at: Instant::now(),
            submitted_at: Instant::now(),
            completed_at: Instant::now(),
        }
    }

    #[test]
    fn io_worker_decodes_and_hashes_linear_height_without_changing_channel_bytes() {
        let dir = directory("decode");
        let path = dir.join("input.png");
        texelith_io::write_normal_png(&path, 2, 1, &[128, 9, 240, 64, 250, 2], false).unwrap();
        let worker = IoWorker::spawn().unwrap();
        assert_eq!(worker.requests.capacity(), Some(1));
        assert_eq!(worker.results.capacity(), Some(1));
        worker
            .requests
            .send(IoRequest::Load {
                token: ImportToken(9),
                path: path.clone(),
                interpretation: InputInterpretation::LinearHeight,
            })
            .unwrap();
        let IoResult::Loaded { token, result } =
            worker.results.recv_timeout(Duration::from_secs(5)).unwrap()
        else {
            panic!("expected decode")
        };
        let upload = result.unwrap();
        assert_eq!(token, ImportToken(9));
        assert_eq!(upload.pixels.as_ref(), &[128, 9, 240, 255, 64, 250, 2, 255]);
        assert_eq!(
            upload.descriptor,
            SourceDescriptor::from_rgba8(
                &upload.pixels,
                Extent {
                    width: 2,
                    height: 1
                },
                InputInterpretation::LinearHeight
            )
            .unwrap()
        );
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn io_save_converts_big_endian_height_and_cancellation_preserves_destination() {
        let dir = directory("save");
        let path = dir.join("height.png");
        let current = AtomicU64::new(3);
        save(
            &path,
            &readback(MapKind::Height, vec![0x12, 0x34, 0xab, 0xcd]),
            &current,
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[24..26], &[16, 0], "16-bit grayscale PNG IHDR");
        assert_eq!(
            bytes,
            texelith_io::encode_height_png(2, 1, &[0x1234, 0xabcd]).unwrap()
        );
        current.store(4, Ordering::Release);
        assert!(
            save(
                &path,
                &readback(MapKind::Normal, vec![128, 128, 255, 128, 128, 255]),
                &current
            )
            .unwrap_err()
            .contains("cancelled")
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        current.store(3, Ordering::Release);
        save(
            &path,
            &readback(MapKind::Normal, vec![128, 128, 255, 128, 128, 255]),
            &current,
        )
        .unwrap();
        assert_eq!(
            &std::fs::read(&path).unwrap()[24..26],
            &[8, 2],
            "8-bit RGB PNG IHDR"
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
