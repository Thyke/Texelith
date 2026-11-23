use super::{MAX_VERTICES, MeshError, RawMesh, RawVertex, counts, error};
use std::{collections::HashSet, fmt::Write, io::Cursor};

fn index(text: &str, current: usize, total: usize) -> Result<usize, MeshError> {
    let value: i64 = text.parse().map_err(|_| error("Invalid OBJ index"))?;
    let resolved = if value < 0 {
        current as i64 + value
    } else {
        value - 1
    };
    let limit = if value < 0 { current } else { total };
    if value == 0 || resolved < 0 || resolved as u64 >= limit as u64 {
        return Err(error("OBJ index outside attribute array"));
    }
    Ok(resolved as usize)
}

pub(super) fn decode(bytes: &[u8]) -> Result<RawMesh, MeshError> {
    let text = std::str::from_utf8(bytes).map_err(|_| error("OBJ must be UTF-8 text"))?;
    let mut totals = [0usize; 3];
    // Count source attributes before invoking tobj, which otherwise allocates
    // its source arrays before constructing the single-index output.
    for line in text.lines() {
        let mut words = line.split('#').next().unwrap_or("").split_whitespace();
        let slot = match words.next() {
            Some("v") => Some((0, 3)),
            Some("vt") => Some((1, 2)),
            Some("vn") => Some((2, 3)),
            Some("l" | "p") => return Err(error("OBJ lines and points are unsupported")),
            _ => None,
        };
        if let Some((slot, required)) = slot {
            totals[slot] += 1;
            if totals[slot] > MAX_VERTICES {
                return Err(error("OBJ source attribute limit exceeded"));
            }
            let mut count = 0;
            let mut nonzero_normal = false;
            for word in words {
                let value: f32 = word
                    .parse()
                    .map_err(|_| error("Invalid OBJ numeric attribute"))?;
                if !value.is_finite() {
                    return Err(error("Nonfinite OBJ attribute"));
                }
                if count < 3 {
                    nonzero_normal |= value != 0.0;
                }
                count += 1;
            }
            if count < required {
                return Err(error("Incomplete OBJ attribute"));
            }
            if slot == 2 && !nonzero_normal {
                return Err(error("Zero-length OBJ normal"));
            }
        }
    }
    let mut current = [0; 3];
    let mut corners = HashSet::new();
    let (mut emitted, mut indices) = (0usize, 0usize);
    // A synthetic zero normal makes missing corners explicit to tobj instead
    // of allowing its missing-attribute fill policy to borrow another normal.
    // Real normals were checked above, so the sentinel is unambiguous.
    let mut geometry = String::from("vn 0 0 0\n");
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut words = line.split_whitespace();
        match words.next() {
            Some(kind @ ("v" | "vt" | "vn")) => {
                let slot = match kind {
                    "v" => 0,
                    "vt" => 1,
                    _ => 2,
                };
                current[slot] += 1;
                geometry.push_str(line);
                geometry.push('\n');
            }
            Some("f") => {
                let mut count = 0;
                let mut canonical = String::from("f");
                for corner in words {
                    count += 1;
                    if count > 3 {
                        return Err(error("Only triangle OBJ faces are supported"));
                    }
                    let mut fields = corner.split('/');
                    let pos = index(fields.next().unwrap_or(""), current[0], totals[0])?;
                    let uv = fields
                        .next()
                        .filter(|v| !v.is_empty())
                        .ok_or_else(|| error("Missing OBJ UV coordinates"))?;
                    let uv = index(uv, current[1], totals[1])?;
                    let normal = fields
                        .next()
                        .filter(|v| !v.is_empty())
                        .map(|n| index(n, current[2], totals[2]))
                        .transpose()?;
                    if fields.next().is_some() {
                        return Err(error("Malformed OBJ face corner"));
                    }
                    if corners.insert((pos, uv, normal)) {
                        emitted += 1;
                        counts(emitted, indices)?;
                    }
                    write!(
                        canonical,
                        " {}/{}/{}",
                        pos + 1,
                        uv + 1,
                        normal.map_or(1, |n| n + 2)
                    )
                    .map_err(|_| error("Cannot format OBJ triangle"))?;
                }
                if count != 3 {
                    return Err(error("Incomplete OBJ triangle"));
                }
                indices += 3;
                counts(emitted, indices)?;
                geometry.push_str(&canonical);
                geometry.push('\n');
            }
            Some("o" | "g") => {
                corners.clear();
                geometry.push_str("g mesh\n");
            }
            // Do not pass material references to the parser at all.
            _ => {}
        }
    }
    drop(corners);
    counts(emitted, indices)?;
    let options = tobj::LoadOptions {
        single_index: true,
        triangulate: false,
        ..Default::default()
    };
    let (models, _) = tobj::load_obj_buf(&mut Cursor::new(geometry), &options, |_| {
        Ok(Default::default())
    })
    .map_err(|e| error(format!("Invalid OBJ: {e}")))?;
    let mut raw = RawMesh {
        vertices: Vec::with_capacity(emitted),
        indices: Vec::with_capacity(indices),
    };
    for model in models {
        let mesh = model.mesh;
        let nv = mesh.positions.len() / 3;
        counts(
            raw.vertices.len() + nv,
            raw.indices.len() + mesh.indices.len(),
        )?;
        if mesh.texcoords.len() != nv * 2 || mesh.normals.len() != nv * 3 {
            return Err(error("Incomplete OBJ UV/normal attributes"));
        }
        let base = raw.vertices.len() as u32;
        for i in 0..nv {
            let position = std::array::from_fn(|axis| f64::from(mesh.positions[i * 3 + axis]));
            let uv = std::array::from_fn(|axis| f64::from(mesh.texcoords[i * 2 + axis]));
            let normal: [f64; 3] =
                std::array::from_fn(|axis| f64::from(mesh.normals[i * 3 + axis]));
            let normal = (normal != [0.0; 3]).then_some(normal);
            raw.vertices.push(RawVertex {
                position,
                normal,
                uv,
            });
        }
        for index in mesh.indices {
            if index as usize >= nv {
                return Err(error("OBJ index outside positions"));
            }
            raw.indices.push(index + base);
        }
    }
    Ok(raw)
}
