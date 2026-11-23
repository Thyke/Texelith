use serde_json::{Value, json};
use std::fmt::Write;

pub const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
pub const UV: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
pub const OBJ: &[u8] = b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";

pub fn obj_vertices(count: usize) -> Vec<u8> {
    let mut text = String::new();
    for i in 0..count {
        let [x, y, z] = TRIANGLE[i % 3];
        writeln!(text, "v {x} {y} {z}").unwrap();
    }
    text.push_str("vt 0 0\nvt 1 0\nvt 0 1\n");
    for start in (0..count / 3 * 3).step_by(3) {
        writeln!(text, "f {}/1 {}/2 {}/3", start + 1, start + 2, start + 3).unwrap();
    }
    match count % 3 {
        1 => writeln!(text, "f {count}/1 2/2 3/3").unwrap(),
        2 => writeln!(text, "f {}/1 {count}/2 3/3", count - 1).unwrap(),
        _ => {}
    }
    text.into_bytes()
}

pub fn obj_indices(count: usize) -> Vec<u8> {
    assert!(count.is_multiple_of(3));
    let mut bytes = OBJ.to_vec();
    for _ in 1..count / 3 {
        bytes.extend_from_slice(b"f 1/1 2/2 3/3\n");
    }
    bytes
}

/// Original synthetic geometry; no material/image import or downloaded fixtures.
pub struct GltfFixture {
    pub json: Value,
    pub bin: Vec<u8>,
}

impl GltfFixture {
    pub fn triangle() -> Self {
        Self::new(&TRIANGLE, Some(&UV), None, Some(&[0, 1, 2]))
    }

    pub fn animated() -> Self {
        let mut fixture = Self::triangle();
        let input = fixture.json["accessors"].as_array().unwrap().len();
        let offset = fixture.bin.len();
        for time in [0.0f32, 1.0, 2.0] {
            fixture.bin.extend_from_slice(&time.to_le_bytes());
        }
        let view = fixture.json["bufferViews"].as_array().unwrap().len();
        fixture.json["bufferViews"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "buffer": 0, "byteOffset": offset, "byteLength": 12
            }));
        fixture.json["accessors"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "bufferView": view, "componentType": 5126, "count": 3, "type": "SCALAR",
                "min": [0.0], "max": [2.0]
            }));
        fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
        fixture.json["animations"] = json!([{
            "samplers": [{"input": input, "output": 0}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]
        }]);
        fixture
    }

    pub fn new(
        positions: &[[f32; 3]],
        uvs: Option<&[[f32; 2]]>,
        normals: Option<&[[f32; 3]]>,
        indices: Option<&[u32]>,
    ) -> Self {
        let mut fixture = Self {
            json: json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {}, "mode": 4}]}],
                "buffers": [{"uri": "geometry.bin", "byteLength": 0}],
                "bufferViews": [],
                "accessors": []
            }),
            bin: Vec::new(),
        };
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for position in positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
        }
        let position = fixture.floats(positions, "VEC3");
        fixture.json["accessors"][position]["min"] = json!(min);
        fixture.json["accessors"][position]["max"] = json!(max);
        fixture.json["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = json!(position);
        if let Some(uvs) = uvs {
            let uv = fixture.floats(uvs, "VEC2");
            fixture.json["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = json!(uv);
        }
        if let Some(normals) = normals {
            let normal = fixture.floats(normals, "VEC3");
            fixture.json["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] = json!(normal);
        }
        if let Some(indices) = indices {
            let start = fixture.bin.len();
            for value in indices {
                fixture.bin.extend_from_slice(&value.to_le_bytes());
            }
            let index = fixture.accessor(start, indices.len(), 5125, "SCALAR");
            fixture.json["meshes"][0]["primitives"][0]["indices"] = json!(index);
        }
        fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
        fixture
    }

    fn floats<const N: usize>(&mut self, values: &[[f32; N]], kind: &str) -> usize {
        let start = self.bin.len();
        for value in values.iter().flatten() {
            self.bin.extend_from_slice(&value.to_le_bytes());
        }
        self.accessor(start, values.len(), 5126, kind)
    }

    fn accessor(&mut self, start: usize, count: usize, component: u32, kind: &str) -> usize {
        let views = self.json["bufferViews"].as_array_mut().unwrap();
        let view = views.len();
        views.push(json!({
            "buffer": 0, "byteOffset": start, "byteLength": self.bin.len() - start
        }));
        let accessors = self.json["accessors"].as_array_mut().unwrap();
        let index = accessors.len();
        accessors.push(json!({
            "bufferView": view, "componentType": component, "count": count, "type": kind
        }));
        index
    }

    pub fn document(&self) -> Vec<u8> {
        serde_json::to_vec(&self.json).unwrap()
    }

    pub fn glb(&self) -> Vec<u8> {
        let mut document = self.json.clone();
        document["buffers"][0]
            .as_object_mut()
            .unwrap()
            .remove("uri");
        let mut json = serde_json::to_vec(&document).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = self.bin.clone();
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        for word in [0x4654_6c67, 2, u32::try_from(total).unwrap()] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&u32::try_from(json.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&0x4e4f_534au32.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&u32::try_from(bin.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&0x004e_4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    pub fn repeated_vertices(count: usize) -> Self {
        let positions: Vec<_> = (0..count).map(|i| TRIANGLE[i % 3]).collect();
        let uvs: Vec<_> = (0..count).map(|i| UV[i % 3]).collect();
        let mut indices: Vec<_> = (0..count / 3 * 3).map(|i| i as u32).collect();
        match count % 3 {
            1 => indices.extend_from_slice(&[(count - 1) as u32, 1, 2]),
            2 => indices.extend_from_slice(&[(count - 2) as u32, (count - 1) as u32, 2]),
            _ => {}
        }
        Self::new(&positions, Some(&uvs), None, Some(&indices))
    }

    pub fn repeated_indices(count: usize) -> Self {
        assert!(count.is_multiple_of(3));
        let indices: Vec<_> = (0..count).map(|i| (i % 3) as u32).collect();
        Self::new(&TRIANGLE, Some(&UV), None, Some(&indices))
    }
}
