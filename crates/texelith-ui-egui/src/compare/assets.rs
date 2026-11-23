use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use texelith_core::{Extent, MeshData, Vertex};
use texelith_gpu::ibl::Environment;
#[derive(Clone)]
pub(crate) enum Load {
    Mesh(PathBuf),
    Hdr(PathBuf),
    Sky,
}
pub(crate) enum Loaded {
    Mesh(Arc<MeshData>),
    Hdr(Arc<Environment>, String),
}
pub(crate) struct Assets {
    send: crossbeam_channel::Sender<(u64, Load)>,
    pub recv: crossbeam_channel::Receiver<(u64, Result<Loaded, String>)>,
}
impl Assets {
    pub fn new() -> Self {
        let (send, receive) = crossbeam_channel::bounded::<(u64, Load)>(1);
        let (events, recv) = crossbeam_channel::bounded(1);
        std::thread::Builder::new()
            .name("texelith-viewport-io".into())
            .spawn(move || {
                while let Ok((id, load)) = receive.recv() {
                    if events.send((id, decode(load))).is_err() {
                        break;
                    }
                }
            })
            .expect("viewport IO thread");
        Self { send, recv }
    }
    pub fn request(&self, id: u64, load: Load) -> Result<(), Load> {
        self.send.try_send((id, load)).map_err(|e| e.into_inner().1)
    }
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("Viewport asset exceeds input limit".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("Viewport asset exceeds input limit".into());
    }
    Ok(bytes)
}
pub(crate) fn decode(load: Load) -> Result<Loaded, String> {
    match load {
        Load::Sky => Ok(Loaded::Hdr(
            Arc::new(Environment::procedural()),
            super::labels::text("sky"),
        )),
        Load::Hdr(path) => {
            let bytes = read(&path, 128 * 1024 * 1024)?;
            let image = texelith_io::decode_environment_bytes(&bytes).map_err(|e| e.to_string())?;
            Ok(Loaded::Hdr(
                Arc::new(
                    Environment::new(
                        Extent::new(image.width, image.height).map_err(|e| e.to_string())?,
                        &image.rgb,
                    )
                    .map_err(|e| e.to_string())?,
                ),
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            ))
        }
        Load::Mesh(path) => {
            use texelith_io::mesh::{MeshFormat, decode_mesh_bytes};
            let bytes = read(&path, 64 * 1024 * 1024)?;
            let format = match path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "obj" => MeshFormat::Obj,
                "gltf" => MeshFormat::Gltf,
                "glb" => MeshFormat::Glb,
                _ => return Err("Choose OBJ, glTF or GLB geometry".into()),
            };
            let mut local = Vec::<(String, Vec<u8>)>::new();
            if format == MeshFormat::Gltf {
                let json: serde_json::Value =
                    serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                let root = path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .canonicalize()
                    .map_err(|e| e.to_string())?;
                let mut remaining = 64 * 1024 * 1024 - bytes.len() as u64;
                for buffer in json["buffers"].as_array().into_iter().flatten() {
                    if let Some(uri) = buffer["uri"].as_str() {
                        if uri.contains(':') || uri.contains('%') || uri.contains('\\') {
                            return Err("Only bounded local geometry buffers are supported".into());
                        }
                        let target = root.join(uri).canonicalize().map_err(|e| e.to_string())?;
                        if !target.starts_with(&root) {
                            return Err("Mesh buffer escapes its local folder".into());
                        }
                        let data = read(&target, remaining)?;
                        remaining -= data.len() as u64;
                        local.push((uri.into(), data));
                    }
                }
            }
            let refs: Vec<_> = local
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice()))
                .collect();
            let mesh = decode_mesh_bytes(&bytes, format, &refs).map_err(|e| e.to_string())?;
            let mesh = MeshData {
                vertices: mesh
                    .vertices
                    .into_iter()
                    .map(|v| Vertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                        tangent: v.tangent,
                    })
                    .collect(),
                indices: mesh.indices,
            };
            texelith_gpu::compare::validate_mesh(&mesh).map_err(|e| e.to_string())?;
            Ok(Loaded::Mesh(Arc::new(mesh)))
        }
    }
}

#[cfg(test)]
mod sky_tests {
    use super::*;

    #[test]
    fn default_and_reloaded_sky_share_blue_atmosphere_and_warm_sun() {
        let default = Environment::procedural();
        let Loaded::Hdr(reloaded, _) = decode(Load::Sky).unwrap() else {
            panic!("sky environment");
        };
        assert_eq!(default.key(), reloaded.key());
        let pixel = |x: usize, y: usize| default.pixels()[y * 256 + x];
        let upper = pixel(128, 58);
        let horizon = pixel(128, 92);
        assert!(upper[2] > upper[0] * 1.3, "blue upper backdrop: {upper:?}");
        assert!(
            horizon[0] > upper[0] * 1.5,
            "soft bright horizon: {horizon:?}"
        );
        let sun = default
            .pixels()
            .iter()
            .max_by(|a, b| a[0].total_cmp(&b[0]))
            .unwrap();
        assert!(sun[0] > 1. && sun[0] > sun[1] && sun[1] > sun[2]);
        assert!(
            default
                .pixels()
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v >= 0.)
        );
    }
}
