//! Bounded, geometry-only static mesh import. Call on the IO worker.
//!
//! No filesystem, material, image or network loading is performed. Coordinates
//! remain right-handed; UVs have no origin flip or color-space conversion.

mod gltf;
mod math;
mod obj;

use math::{cross, dot, normalize, scale, sub};

const MAX_INPUT: usize = 64 * 1024 * 1024;
const MAX_VERTICES: usize = 250_000;
const MAX_INDICES: usize = 750_000;
const MAX_DECODED: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    Gltf,
    Glb,
    Obj,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Orthonormal tangent xyz and bitangent sign: B = cross(N, T) * w.
    pub tangent: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshWarning {
    DegenerateUv,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub bounds: MeshBounds,
    pub warnings: Vec<MeshWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshError(String);

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MeshError {}

fn error(message: impl Into<String>) -> MeshError {
    MeshError(message.into())
}

fn counts(vertices: usize, indices: usize) -> Result<(), MeshError> {
    if vertices > MAX_VERTICES || indices > MAX_INDICES {
        return Err(error("Mesh vertex/index limit exceeded"));
    }
    let bytes = vertices
        .checked_mul(48)
        .and_then(|n| indices.checked_mul(4).and_then(|i| n.checked_add(i)))
        .ok_or_else(|| error("Decoded mesh size overflow"))?;
    if bytes > MAX_DECODED {
        return Err(error("Decoded mesh exceeds 128 MiB"));
    }
    Ok(())
}

/// Decode a bounded static triangle scene, centered at its AABB center with
/// longest extent 2. Local geometry buffers must be provided by exact URI name;
/// only relative local URIs are accepted. No imported textures are allocated.
pub fn decode_mesh_bytes(
    bytes: &[u8],
    format: MeshFormat,
    local_buffers: &[(&str, &[u8])],
) -> Result<ImportedMesh, MeshError> {
    if bytes.len() > MAX_INPUT {
        return Err(error("Mesh input exceeds 64 MiB"));
    }
    let raw = match format {
        MeshFormat::Obj => obj::decode(bytes)?,
        MeshFormat::Gltf | MeshFormat::Glb => gltf::decode(bytes, format, local_buffers)?,
    };
    finish(raw)
}

struct RawVertex {
    position: [f64; 3],
    normal: Option<[f64; 3]>,
    uv: [f64; 2],
}

#[derive(Default)]
struct RawMesh {
    vertices: Vec<RawVertex>,
    indices: Vec<u32>,
}

fn uv_derivatives(raw: &RawMesh, tri: &[u32; 3]) -> ([f64; 2], [f64; 2], f64) {
    let [a, b, c] = tri.map(|i| raw.vertices[i as usize].uv);
    let d1 = [b[0] - a[0], b[1] - a[1]];
    let d2 = [c[0] - a[0], c[1] - a[1]];
    (d1, d2, d1[0] * d2[1] - d1[1] * d2[0])
}

fn split_tangent_vertices(raw: &RawMesh) -> Result<(Vec<usize>, Vec<u32>), MeshError> {
    let mut sources: Vec<_> = (0..raw.vertices.len()).collect();
    let mut indices = Vec::with_capacity(raw.indices.len());
    let mut slots = vec![[None; 3]; raw.vertices.len()];
    for tri in raw.indices.as_chunks::<3>().0 {
        let (_, _, det) = uv_derivatives(raw, tri);
        // Opposite orientations cannot share a tangent frame. Degenerate UVs
        // get their own corners so their fallback does not replace valid frames.
        let orientation = if det < 0.0 {
            0
        } else if det > 0.0 {
            1
        } else {
            2
        };
        for &source in tri {
            let group = &mut slots[source as usize];
            let index = if let Some(index) = group[orientation] {
                index
            } else {
                let index = if group.iter().all(Option::is_none) {
                    source
                } else {
                    counts(sources.len() + 1, raw.indices.len())?;
                    let index = sources.len() as u32;
                    sources.push(source as usize);
                    index
                };
                group[orientation] = Some(index);
                index
            };
            indices.push(index);
        }
    }
    Ok((sources, indices))
}

fn finish(raw: RawMesh) -> Result<ImportedMesh, MeshError> {
    counts(raw.vertices.len(), raw.indices.len())?;
    if raw.vertices.is_empty() || raw.indices.is_empty() || !raw.indices.len().is_multiple_of(3) {
        return Err(error("Mesh contains no complete triangles"));
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for v in &raw.vertices {
        if !v.position.iter().chain(&v.uv).all(|x| x.is_finite())
            || v.normal.is_some_and(|n| !n.iter().all(|x| x.is_finite()))
        {
            return Err(error("Nonfinite mesh attribute"));
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(v.position[axis]);
            max[axis] = max[axis].max(v.position[axis]);
        }
    }
    let extent = sub(max, min).into_iter().fold(0.0, f64::max);
    if !extent.is_finite() || extent <= 0.0 {
        return Err(error("Mesh has zero or invalid bounds"));
    }
    let center = std::array::from_fn(|i| min[i] + (max[i] - min[i]) * 0.5);
    // f64 intermediates keep very small/large finite input coordinates stable.
    let positions: Vec<_> = raw
        .vertices
        .iter()
        .map(|v| sub(v.position, center).map(|x| x / extent * 2.0))
        .collect();
    let mut normals = vec![[0.0; 3]; raw.vertices.len()];
    for tri in raw.indices.as_chunks::<3>().0 {
        let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        if [a, b, c].iter().any(|&i| i >= raw.vertices.len()) {
            return Err(error("Mesh index out of bounds"));
        }
        let e1 = sub(positions[b], positions[a]);
        let e2 = sub(positions[c], positions[a]);
        let face = cross(e1, e2);
        if dot(face, face) <= 0.0 {
            return Err(error("Zero-area mesh triangle"));
        }
        for i in [a, b, c] {
            for (sum, value) in normals[i].iter_mut().zip(face) {
                *sum += value;
            }
        }
    }
    // Synthesize normals on the original topology; a tangent seam must not
    // introduce a hard normal edge in otherwise smooth geometry.
    let (sources, indices) = split_tangent_vertices(&raw)?;
    let mut tangents = vec![[0.0; 3]; sources.len()];
    let mut bitangents = tangents.clone();
    let mut fallback = vec![false; sources.len()];
    let mut warnings = Vec::new();
    for (tri, split) in raw
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .zip(indices.as_chunks::<3>().0)
    {
        let [a, b, c] = tri.map(|i| i as usize);
        let e1 = sub(positions[b], positions[a]);
        let e2 = sub(positions[c], positions[a]);
        let (d1, d2, det) = uv_derivatives(&raw, tri);
        if det == 0.0 {
            for &i in split {
                fallback[i as usize] = true;
            }
            if warnings.is_empty() {
                warnings.push(MeshWarning::DegenerateUv);
            }
        } else {
            let tangent = scale(sub(scale(e1, d2[1]), scale(e2, d1[1])), 1.0 / det);
            let bitangent = scale(sub(scale(e2, d1[0]), scale(e1, d2[0])), 1.0 / det);
            for &i in split {
                for axis in 0..3 {
                    tangents[i as usize][axis] += tangent[axis];
                    bitangents[i as usize][axis] += bitangent[axis];
                }
            }
        }
    }
    let mut vertices = Vec::with_capacity(sources.len());
    let mut bounds = MeshBounds {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };
    for (i, source) in sources.into_iter().enumerate() {
        let v = &raw.vertices[source];
        // Unreferenced vertices have no incident area; choose a stable frame.
        let n = normalize(v.normal.unwrap_or(normals[source]))
            .or_else(|| v.normal.is_none().then_some([0.0, 0.0, 1.0]))
            .ok_or_else(|| error("Invalid zero-length normal"))?;
        let t = if fallback[i] {
            None
        } else {
            normalize(sub(tangents[i], scale(n, dot(n, tangents[i]))))
        }
        .unwrap_or_else(|| {
            let axis = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
                [1.0, 0.0, 0.0]
            } else if n[1].abs() <= n[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            normalize(cross(axis, n)).expect("least-aligned axis is not parallel")
        });
        let sign = if !fallback[i] && dot(cross(n, t), bitangents[i]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let position = positions[source].map(|x| x as f32);
        for (axis, &value) in position.iter().enumerate() {
            bounds.min[axis] = bounds.min[axis].min(value);
            bounds.max[axis] = bounds.max[axis].max(value);
        }
        vertices.push(MeshVertex {
            position,
            normal: n.map(|x| x as f32),
            uv: v.uv.map(|x| x as f32),
            tangent: [t[0] as f32, t[1] as f32, t[2] as f32, sign],
        });
    }
    Ok(ImportedMesh {
        vertices,
        indices,
        bounds,
        warnings,
    })
}
