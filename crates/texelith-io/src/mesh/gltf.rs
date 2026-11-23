use super::{
    MAX_INDICES, MAX_INPUT, MAX_VERTICES, MeshError, MeshFormat, RawMesh, RawVertex, counts, error,
    math,
};
use ::gltf::{
    accessor::{DataType, Dimensions},
    mesh::{Mode, Semantic},
};
use serde_json::Value;

fn uint(value: &Value, name: &str) -> Result<usize, MeshError> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| error(format!("Invalid or overflowing {name}")))
}

fn optional_uint(value: &Value, name: &str, default: usize) -> Result<usize, MeshError> {
    value.get(name).map_or(Ok(default), |v| uint(v, name))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], MeshError> {
    value.get(name).map_or(Ok(&[]), |v| {
        v.as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| error(format!("Invalid {name} array")))
    })
}

fn word(bytes: &[u8], offset: usize) -> Result<u32, MeshError> {
    let data = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| error("Truncated GLB header"))?;
    Ok(u32::from_le_bytes(
        data.try_into().map_err(|_| error("Invalid GLB header"))?,
    ))
}

fn container(bytes: &[u8], format: MeshFormat) -> Result<(&[u8], Option<&[u8]>), MeshError> {
    if format == MeshFormat::Gltf {
        return Ok((bytes, None));
    }
    if word(bytes, 0)? != 0x4654_6c67
        || word(bytes, 4)? != 2
        || word(bytes, 8)? as usize != bytes.len()
    {
        return Err(error("Invalid GLB magic, version or declared length"));
    }
    let mut cursor = 12usize;
    let mut json = None;
    let mut bin = None;
    while cursor < bytes.len() {
        let length = word(bytes, cursor)? as usize;
        let kind = word(bytes, cursor + 4)?;
        let start = cursor
            .checked_add(8)
            .ok_or_else(|| error("GLB chunk overflow"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| error("GLB chunk overflow"))?;
        let chunk = bytes
            .get(start..end)
            .ok_or_else(|| error("Truncated GLB chunk"))?;
        if !length.is_multiple_of(4) || (json.is_none() && kind != 0x4e4f_534a) {
            return Err(error("Invalid GLB chunk order/alignment"));
        }
        match kind {
            0x4e4f_534a if json.is_none() => json = Some(chunk),
            0x004e_4942 if bin.is_none() => bin = Some(chunk),
            _ => return Err(error("Unsupported or duplicate GLB chunk")),
        }
        cursor = end;
    }
    Ok((json.ok_or_else(|| error("Missing GLB JSON chunk"))?, bin))
}

fn compression(value: &Value) -> bool {
    match value {
        Value::Array(a) => a.iter().any(compression),
        Value::Object(o) => o.iter().any(|(k, v)| {
            let compressed = |name: &str| {
                matches!(
                    name,
                    "KHR_draco_mesh_compression" | "EXT_meshopt_compression"
                )
            };
            (k == "extensions"
                && v.as_object()
                    .is_some_and(|extensions| extensions.keys().any(|name| compressed(name))))
                || (matches!(k.as_str(), "extensionsUsed" | "extensionsRequired")
                    && v.as_array().is_some_and(|names| {
                        names
                            .iter()
                            .any(|name| name.as_str().is_some_and(compressed))
                    }))
                || compression(v)
        }),
        _ => false,
    }
}

fn local_uri(uri: &str) -> bool {
    // A byte table is not permission to resolve a scheme, absolute path or UNC.
    !uri.is_empty()
        && !uri.starts_with(['/', '\\'])
        && !uri.contains([':', '\\', '%', '?', '#'])
        && !uri.split('/').any(|part| part == "..")
        && !uri.chars().any(char::is_control)
}

fn resolve<'a>(
    root: &Value,
    bin: Option<&'a [u8]>,
    input_size: usize,
    local: &[(&str, &'a [u8])],
) -> Result<Vec<&'a [u8]>, MeshError> {
    let sources = array(root, "buffers")?;
    if sources.len() > MAX_VERTICES {
        return Err(error("Too many geometry buffers"));
    }
    let mut total = input_size;
    let mut buffers = Vec::with_capacity(sources.len());
    for (i, source) in sources.iter().enumerate() {
        let length = uint(&source["byteLength"], "buffer byteLength")?;
        if length == 0 || length > MAX_INPUT {
            return Err(error("Geometry buffer size limit exceeded"));
        }
        let data = if let Some(uri) = source.get("uri") {
            let uri = uri
                .as_str()
                .ok_or_else(|| error("Invalid geometry buffer URI"))?;
            if !local_uri(uri) {
                return Err(error(
                    "Only relative local geometry buffer URIs are supported",
                ));
            }
            let mut matches = local.iter().filter(|(name, _)| *name == uri);
            let data = matches
                .next()
                .ok_or_else(|| error("Local geometry buffer not supplied"))?
                .1;
            if matches.next().is_some() {
                return Err(error("Ambiguous local geometry buffer"));
            }
            total = total
                .checked_add(data.len())
                .ok_or_else(|| error("Mesh input size overflow"))?;
            if total > MAX_INPUT {
                return Err(error("Aggregate mesh input exceeds 64 MiB"));
            }
            data
        } else if i == 0 {
            bin.ok_or_else(|| error("Geometry buffer has no supplied GLB payload"))?
        } else {
            return Err(error("Only the first GLB buffer may omit its URI"));
        };
        if data.len() < length {
            return Err(error("Truncated geometry buffer"));
        }
        buffers.push(&data[..length]);
    }
    Ok(buffers)
}

fn view_range<'a>(
    view: ::gltf::buffer::View<'_>,
    buffers: &[&'a [u8]],
) -> Result<&'a [u8], MeshError> {
    let end = view
        .offset()
        .checked_add(view.length())
        .ok_or_else(|| error("Buffer view overflow"))?;
    buffers
        .get(view.buffer().index())
        .and_then(|data| data.get(view.offset()..end))
        .ok_or_else(|| error("Buffer view outside geometry buffer"))
}

fn range(
    view: ::gltf::buffer::View<'_>,
    offset: usize,
    count: usize,
    size: usize,
) -> Result<(), MeshError> {
    let stride = view.stride().unwrap_or(size);
    let end = count
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(size))
        .and_then(|n| n.checked_add(offset))
        .ok_or_else(|| error("Accessor range overflow or zero count"))?;
    if stride < size || end > view.length() {
        return Err(error("Accessor outside buffer view or invalid stride"));
    }
    Ok(())
}

fn validate_accessors(
    accessors: &[::gltf::Accessor<'_>],
    buffers: &[&[u8]],
) -> Result<(), MeshError> {
    // Bound all payload work before scanning even the first sparse index. Each
    // consumed vertex has at most position, normal and UV data, plus indices.
    let mut elements = 0usize;
    for a in accessors {
        elements = elements
            .checked_add(a.count())
            .filter(|&n| n <= 3 * MAX_VERTICES + MAX_INDICES)
            .ok_or_else(|| error("Accessor validation work limit exceeded"))?;
    }
    for a in accessors {
        if a.count() == 0 || a.count() > MAX_INDICES {
            return Err(error("Accessor count limit exceeded"));
        }
        if let Some(view) = a.view() {
            view_range(view.clone(), buffers)?;
            range(view, a.offset(), a.count(), a.size())?;
        } else if a.sparse().is_none() {
            return Err(error("Accessor has no data"));
        }
        if let Some(sparse) = a.sparse() {
            if sparse.count() == 0 || sparse.count() > a.count() {
                return Err(error("Invalid sparse accessor count"));
            }
            let indices = sparse.indices();
            let size = indices.index_type().size();
            view_range(indices.view(), buffers)?;
            view_range(sparse.values().view(), buffers)?;
            range(indices.view(), indices.offset(), sparse.count(), size)?;
            range(
                sparse.values().view(),
                sparse.values().offset(),
                sparse.count(),
                a.size(),
            )?;
            let view = indices.view();
            let stride = view.stride().unwrap_or(size);
            let bytes = view_range(view, buffers)?;
            let mut previous = None;
            for i in 0..sparse.count() {
                let start = indices.offset() + i * stride;
                let mut word = [0u8; 4];
                word[..size].copy_from_slice(&bytes[start..start + size]);
                let index = u32::from_le_bytes(word) as usize;
                if index >= a.count() || previous.is_some_and(|p| index <= p) {
                    return Err(error("Sparse indices must be ordered and in bounds"));
                }
                previous = Some(index);
            }
        }
    }
    Ok(())
}

fn primitive_counts(p: &::gltf::Primitive<'_>) -> Result<(usize, usize), MeshError> {
    if p.mode() != Mode::Triangles {
        return Err(error("Only triangle glTF primitives are supported"));
    }
    let pos = p
        .get(&Semantic::Positions)
        .ok_or_else(|| error("Missing positions"))?;
    if pos.dimensions() != Dimensions::Vec3 || pos.data_type() != DataType::F32 || pos.normalized()
    {
        return Err(error("Positions must be float32 VEC3"));
    }
    let uv = p
        .get(&Semantic::TexCoords(0))
        .ok_or_else(|| error("Missing UV coordinates"))?;
    if uv.dimensions() != Dimensions::Vec2
        || !matches!(uv.data_type(), DataType::F32 | DataType::U8 | DataType::U16)
        || (uv.data_type() != DataType::F32 && !uv.normalized())
    {
        return Err(error("Unsupported UV accessor"));
    }
    for (semantic, attr) in p.attributes() {
        if attr.count() != pos.count() {
            return Err(error("Attribute count differs from positions"));
        }
        if matches!(semantic, Semantic::Joints(_) | Semantic::Weights(_)) {
            return Err(error("Skinned attributes are unsupported"));
        }
        if semantic == Semantic::Normals
            && (attr.dimensions() != Dimensions::Vec3
                || attr.data_type() != DataType::F32
                || attr.normalized())
        {
            return Err(error("Normals must be float32 VEC3"));
        }
    }
    let indices = if let Some(a) = p.indices() {
        if a.dimensions() != Dimensions::Scalar
            || !matches!(a.data_type(), DataType::U8 | DataType::U16 | DataType::U32)
            || a.normalized()
        {
            return Err(error("Invalid triangle index accessor"));
        }
        a.count()
    } else {
        pos.count()
    };
    if !indices.is_multiple_of(3) {
        return Err(error("Incomplete triangle indices"));
    }
    counts(pos.count(), indices)?;
    Ok((pos.count(), indices))
}

pub(super) fn decode(
    bytes: &[u8],
    format: MeshFormat,
    local: &[(&str, &[u8])],
) -> Result<RawMesh, MeshError> {
    let (json, bin) = container(bytes, format)?;
    let mut root: Value =
        serde_json::from_slice(json).map_err(|e| error(format!("Invalid glTF JSON: {e}")))?;
    if compression(&root) {
        return Err(error("Compressed geometry is unsupported"));
    }
    if !array(&root, "skins")?.is_empty() || !array(&root, "animations")?.is_empty() {
        return Err(error("Skins and animation are unsupported"));
    }
    for name in ["nodes", "meshes", "accessors", "bufferViews", "scenes"] {
        if array(&root, name)?.len() > MAX_VERTICES {
            return Err(error("glTF metadata count limit exceeded"));
        }
    }
    for node in array(&root, "nodes")? {
        if node.get("skin").is_some() || node.get("weights").is_some() {
            return Err(error("Skin/morph nodes are unsupported"));
        }
        if node.get("matrix").is_some()
            && ["translation", "rotation", "scale"]
                .iter()
                .any(|k| node.get(k).is_some())
        {
            return Err(error("A node cannot contain both matrix and TRS"));
        }
    }
    // Preflight advertised sizes in JSON before any geometry/decoder allocation.
    for accessor in array(&root, "accessors")? {
        let count = uint(&accessor["count"], "accessor count")?;
        if count == 0 || count > MAX_INDICES {
            return Err(error("Accessor count limit exceeded"));
        }
        optional_uint(accessor, "byteOffset", 0)?;
    }
    for view in array(&root, "bufferViews")? {
        let length = uint(&view["byteLength"], "view byteLength")?;
        let offset = optional_uint(view, "byteOffset", 0)?;
        if length > MAX_INPUT || offset.checked_add(length).is_none_or(|n| n > MAX_INPUT) {
            return Err(error("Buffer view limit exceeded"));
        }
    }
    let buffers = resolve(&root, bin, bytes.len(), local)?;
    if let Some(meshes) = root.get_mut("meshes").and_then(Value::as_array_mut) {
        for mesh in meshes {
            if mesh.get("weights").is_some() {
                return Err(error("Morph targets are unsupported"));
            }
            let primitives = mesh
                .get_mut("primitives")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| error("Missing primitives"))?;
            if primitives.len() > MAX_VERTICES {
                return Err(error("Primitive count limit exceeded"));
            }
            for p in primitives {
                if p.get("targets").is_some() {
                    return Err(error("Morph targets are unsupported"));
                }
                if let Some(o) = p.as_object_mut() {
                    o.remove("material");
                }
            }
        }
    }
    // Materials/images are outside this importer, including their validity/URIs.
    if let Some(o) = root.as_object_mut() {
        for key in ["materials", "images", "textures", "samplers"] {
            o.remove(key);
        }
    }
    let json = serde_json::from_value(root).map_err(|e| error(format!("Invalid glTF: {e}")))?;
    let doc = ::gltf::Document::from_json(json).map_err(|e| error(format!("Invalid glTF: {e}")))?;
    let meshes: Vec<_> = doc.meshes().collect();
    let mut sizes = Vec::with_capacity(meshes.len());
    let (mut source_v, mut source_i) = (0usize, 0usize);
    for mesh in &meshes {
        let (mut v, mut i) = (0, 0);
        for p in mesh.primitives() {
            let (pv, pi) = primitive_counts(&p)?;
            v += pv;
            i += pi;
            counts(v, i)?;
        }
        source_v += v;
        source_i += i;
        counts(source_v, source_i)?;
        sizes.push((v, i));
    }
    // Validate the entire node forest iteratively, including unreachable cycles.
    let mut parents = vec![0usize; doc.nodes().len()];
    for node in doc.nodes() {
        for child in node.children() {
            parents[child.index()] += 1;
            if parents[child.index()] > 1 {
                return Err(error("Node has multiple parents"));
            }
        }
    }
    let nodes: Vec<_> = doc.nodes().collect();
    let mut queue: Vec<_> = parents
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == 0)
        .map(|(i, _)| i)
        .collect();
    let mut cursor = 0;
    while cursor < queue.len() {
        for child in nodes[queue[cursor]].children() {
            queue.push(child.index());
        }
        cursor += 1;
    }
    if queue.len() != nodes.len() {
        return Err(error("Cyclic glTF node graph"));
    }
    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .ok_or_else(|| error("No glTF scene"))?;
    if scene.nodes().len() > nodes.len() || scene.nodes().any(|n| parents[n.index()] != 0) {
        return Err(error("Scene must contain unique root nodes"));
    }
    let mut stack: Vec<_> = scene.nodes().map(|n| (n.index(), math::IDENTITY)).collect();
    stack.reverse();
    let mut seen = vec![false; nodes.len()];
    let mut instances = Vec::new();
    let (mut total_v, mut total_i) = (0, 0);
    while let Some((index, parent)) = stack.pop() {
        if seen[index] {
            return Err(error("Scene has duplicate or non-root nodes"));
        }
        seen[index] = true;
        let node = &nodes[index];
        let local = node.transform().matrix();
        if !local.iter().flatten().all(|x| x.is_finite()) {
            return Err(error("Nonfinite node transform"));
        }
        let world = math::multiply(parent, local.map(|c| c.map(f64::from)));
        if !world.iter().flatten().all(|x| x.is_finite())
            || world[0][3] != 0.0
            || world[1][3] != 0.0
            || world[2][3] != 0.0
            || world[3][3] != 1.0
        {
            return Err(error("Invalid non-affine node transform"));
        }
        let (_, det) = math::cofactors(world);
        if !det.is_finite() || det == 0.0 {
            return Err(error("Singular node transform"));
        }
        if let Some(mesh) = node.mesh() {
            let (v, i) = sizes[mesh.index()];
            total_v += v;
            total_i += i;
            counts(total_v, total_i)?;
            instances.push((mesh.index(), world));
        }
        let start = stack.len();
        for child in node.children() {
            stack.push((child.index(), world));
        }
        stack[start..].reverse();
    }
    // Ignore unused accessor payloads, and validate each consumed accessor once
    // even when primitives or instances alias it. Geometry/instance preflight
    // above must finish before any sparse index scan or geometry allocation.
    let mut used_meshes = vec![false; meshes.len()];
    let mut used_accessors = vec![false; doc.accessors().len()];
    let mut accessors = Vec::new();
    for &(index, _) in &instances {
        if used_meshes[index] {
            continue;
        }
        used_meshes[index] = true;
        for p in meshes[index].primitives() {
            for a in [
                p.get(&Semantic::Positions),
                p.get(&Semantic::Normals),
                p.get(&Semantic::TexCoords(0)),
                p.indices(),
            ]
            .into_iter()
            .flatten()
            {
                if !used_accessors[a.index()] {
                    used_accessors[a.index()] = true;
                    accessors.push(a);
                }
            }
        }
    }
    validate_accessors(&accessors, &buffers)?;
    let mut raw = RawMesh {
        vertices: Vec::with_capacity(total_v),
        indices: Vec::with_capacity(total_i),
    };
    for (index, world) in instances {
        let (cof, det) = math::cofactors(world);
        for p in meshes[index].primitives() {
            let (nv, _) = primitive_counts(&p)?;
            let base = raw.vertices.len() as u32;
            let reader = p.reader(|buffer| buffers.get(buffer.index()).copied());
            let positions = reader
                .read_positions()
                .ok_or_else(|| error("Invalid position data"))?;
            let uvs = reader
                .read_tex_coords(0)
                .ok_or_else(|| error("Invalid UV data"))?
                .into_f32();
            let mut normals = reader.read_normals();
            for (position, uv) in positions.zip(uvs) {
                if !position.iter().chain(&uv).all(|x| x.is_finite()) {
                    return Err(error("Nonfinite position/UV attribute"));
                }
                let normal = normals
                    .as_mut()
                    .and_then(Iterator::next)
                    .map(|n| math::normal(cof, det, n.map(f64::from)));
                raw.vertices.push(RawVertex {
                    position: math::position(world, position.map(f64::from)),
                    normal,
                    uv: uv.map(f64::from),
                });
            }
            let indices: Vec<_> = if p.indices().is_some() {
                reader
                    .read_indices()
                    .ok_or_else(|| error("Invalid index data"))?
                    .into_u32()
                    .collect()
            } else {
                (0..nv as u32).collect()
            };
            for tri in indices.as_chunks::<3>().0 {
                if tri.iter().any(|&i| i as usize >= nv) {
                    return Err(error("Triangle index outside positions"));
                }
                let mut tri = [tri[0] + base, tri[1] + base, tri[2] + base];
                if det < 0.0 {
                    tri.swap(1, 2);
                }
                raw.indices.extend(tri);
            }
        }
    }
    Ok(raw)
}
