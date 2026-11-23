//! T-064 behavioral contract. API proposals and unspecified policies are recorded
//! in docs/evidence/T-064/test-contract.md; no importer implementation lives here.

#[path = "mvp_t064/fixtures.rs"]
mod fixtures;

use fixtures::{GltfFixture, OBJ, TRIANGLE, UV};
use serde_json::json;
use texelith_io::mesh::{ImportedMesh, MeshFormat, MeshWarning, decode_mesh_bytes};

const EPS: f32 = 2e-5;
const MAX_INPUT: usize = 64 * 1024 * 1024;
const MAX_VERTICES: usize = 250_000;
const MAX_INDICES: usize = 750_000;
const MAX_DECODED: usize = 128 * 1024 * 1024;

fn gltf(fixture: &GltfFixture) -> ImportedMesh {
    decode_mesh_bytes(
        &fixture.document(),
        MeshFormat::Gltf,
        &[("geometry.bin", fixture.bin.as_slice())],
    )
    .unwrap()
}

fn obj(bytes: &[u8]) -> ImportedMesh {
    decode_mesh_bytes(bytes, MeshFormat::Obj, &[]).unwrap()
}

fn rejected(bytes: &[u8], format: MeshFormat, buffers: &[(&str, &[u8])]) -> String {
    let error = decode_mesh_bytes(bytes, format, buffers).expect_err("invalid mesh accepted");
    let message = error.to_string();
    assert!(
        !message.trim().is_empty(),
        "error must be visible to the user"
    );
    message.to_lowercase()
}

fn reject_gltf(fixture: &GltfFixture) -> String {
    rejected(
        &fixture.document(),
        MeshFormat::Gltf,
        &[("geometry.bin", fixture.bin.as_slice())],
    )
}

fn near(actual: f32, expected: f32) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= EPS,
        "expected {expected}, got {actual}"
    );
}

fn near3(actual: [f32; 3], expected: [f32; 3]) {
    for axis in 0..3 {
        near(actual[axis], expected[axis]);
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn assert_mesh(mesh: &ImportedMesh) {
    assert!(!mesh.vertices.is_empty());
    assert!(!mesh.indices.is_empty() && mesh.indices.len().is_multiple_of(3));
    assert!(mesh.vertices.len() <= MAX_VERTICES);
    assert!(mesh.indices.len() <= MAX_INDICES);
    // DTO payload: 12 f32 scalars per vertex and one u32 per index, no textures.
    assert!(mesh.vertices.len() * 48 + mesh.indices.len() * 4 <= MAX_DECODED);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            assert!(vertex.position[axis].is_finite());
            min[axis] = min[axis].min(vertex.position[axis]);
            max[axis] = max[axis].max(vertex.position[axis]);
        }
        assert!(vertex.uv.iter().all(|value| value.is_finite()));
        let tangent = [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]];
        near(dot(vertex.normal, vertex.normal), 1.0);
        near(dot(tangent, tangent), 1.0);
        near(dot(vertex.normal, tangent), 0.0);
        assert!(matches!(vertex.tangent[3], -1.0 | 1.0));
    }
    near3(mesh.bounds.min, min);
    near3(mesh.bounds.max, max);
    let mut longest: f32 = 0.0;
    for axis in 0..3 {
        near((min[axis] + max[axis]) * 0.5, 0.0);
        longest = longest.max(max[axis] - min[axis]);
    }
    near(longest, 2.0);
    for triangle in mesh.indices.as_chunks::<3>().0 {
        assert!(triangle.iter().all(|&i| (i as usize) < mesh.vertices.len()));
        let a = &mesh.vertices[triangle[0] as usize];
        let b = &mesh.vertices[triangle[1] as usize];
        let c = &mesh.vertices[triangle[2] as usize];
        let face = cross(sub(b.position, a.position), sub(c.position, a.position));
        assert!(dot(face, face) > 0.0, "fixture triangles are nondegenerate");
        for vertex in [a, b, c] {
            assert!(dot(face, vertex.normal) > 0.0, "normal/winding mismatch");
        }
    }
}

/// Serialize scalar bits explicitly: no struct padding, platform hash seed, or
/// tolerance can hide nondeterministic ordering/handedness on repeated imports.
fn geometry_bits(mesh: &ImportedMesh) -> Vec<u32> {
    let mut bits = Vec::new();
    for vertex in &mesh.vertices {
        bits.extend(vertex.position.map(f32::to_bits));
        bits.extend(vertex.normal.map(f32::to_bits));
        bits.extend(vertex.uv.map(f32::to_bits));
        bits.extend(vertex.tangent.map(f32::to_bits));
    }
    bits.extend(&mesh.indices);
    bits.extend(mesh.bounds.min.map(f32::to_bits));
    bits.extend(mesh.bounds.max.map(f32::to_bits));
    bits
}

#[test]
fn obj_gltf_and_glb_import_a_triangle_without_materials() {
    let fixture = GltfFixture::triangle();
    let meshes = [
        obj(OBJ),
        gltf(&fixture),
        decode_mesh_bytes(&fixture.glb(), MeshFormat::Glb, &[]).unwrap(),
    ];
    for mesh in &meshes {
        assert_mesh(mesh);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert!(mesh.warnings.is_empty());
        for vertex in &mesh.vertices {
            near3(vertex.normal, [0.0, 0.0, 1.0]);
            near3(vertex.tangent[..3].try_into().unwrap(), [1.0, 0.0, 0.0]);
            near(vertex.tangent[3], 1.0);
            // Geometry UVs are numeric: no sRGB transfer or image-origin flip.
            assert!(UV.contains(&vertex.uv));
        }
        for expected in [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]] {
            assert!(mesh.vertices.iter().any(|v| v.position == expected));
        }
    }
    assert_eq!(geometry_bits(&meshes[1]), geometry_bits(&meshes[2]));
}

#[test]
fn nonindexed_gltf_and_negative_obj_indices_are_supported() {
    let fixture = GltfFixture::new(&TRIANGLE, Some(&UV), None, None);
    let gltf = gltf(&fixture);
    let negative = obj(b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nf -3/-3 -2/-2 -1/-1\n");
    for mesh in [gltf, negative] {
        assert_mesh(&mesh);
        assert_eq!(mesh.indices.len(), 3);
    }
}

#[test]
fn supplied_obj_normals_are_renormalized_without_flipping_y_or_z() {
    let mesh =
        obj(b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 7\nf 1/1/1 2/2/1 3/3/1\n");
    assert_mesh(&mesh);
    for vertex in mesh.vertices {
        near3(vertex.normal, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn normalization_preserves_aspect_ratio_instead_of_scaling_axes_independently() {
    let fixture = GltfFixture::new(
        &[[10.0, 20.0, 3.0], [14.0, 20.0, 3.0], [10.0, 22.0, 3.0]],
        Some(&UV),
        None,
        Some(&[0, 1, 2]),
    );
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    near3(mesh.bounds.min, [-1.0, -0.5, 0.0]);
    near3(mesh.bounds.max, [1.0, 0.5, 0.0]);
}

#[test]
fn hierarchical_trs_and_column_major_matrix_produce_the_same_scene() {
    let mut hierarchy = GltfFixture::triangle();
    hierarchy.json["nodes"] = json!([
        {"translation": [10.0, 20.0, 30.0], "scale": [2.0, 3.0, 1.0], "children": [1]},
        {"mesh": 0, "translation": [1.0, 2.0, 0.0],
         "rotation": [0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2]}
    ]);
    let mut matrix = GltfFixture::triangle();
    matrix.json["nodes"] = json!([{"mesh": 0, "matrix": [
        0.0, 3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 12.0, 26.0, 30.0, 1.0
    ]}]);
    let a = gltf(&hierarchy);
    let b = gltf(&matrix);
    assert_mesh(&a);
    assert_mesh(&b);
    near3(a.bounds.min, [-2.0 / 3.0, -1.0, 0.0]);
    near3(a.bounds.max, [2.0 / 3.0, 1.0, 0.0]);
    for va in &a.vertices {
        let vb = b.vertices.iter().find(|vb| vb.uv == va.uv).unwrap();
        near3(va.position, vb.position);
        near3(va.normal, vb.normal);
        for axis in 0..4 {
            near(va.tangent[axis], vb.tangent[axis]);
        }
    }
}

#[test]
fn scene_instances_and_primitives_share_one_normalization() {
    let mut fixture = GltfFixture::triangle();
    fixture.json["nodes"] = json!([{"mesh": 0}, {"mesh": 0, "translation": [8.0, 0.0, 0.0]}]);
    fixture.json["scenes"][0]["nodes"] = json!([0, 1]);
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    assert_eq!(mesh.indices.len(), 6);
    assert_eq!(mesh.vertices.len(), 6);
    near3(mesh.bounds.min, [-1.0, -0.2, 0.0]);
    near3(mesh.bounds.max, [1.0, 0.2, 0.0]);
    for x in [-1.0, -0.6, 0.6, 1.0] {
        assert!(
            mesh.vertices
                .iter()
                .any(|v| (v.position[0] - x).abs() <= EPS)
        );
    }
    // A second primitive is geometry too, even though it reuses the accessors.
    let primitive = fixture.json["meshes"][0]["primitives"][0].clone();
    fixture.json["meshes"][0]["primitives"]
        .as_array_mut()
        .unwrap()
        .push(primitive);
    let doubled = gltf(&fixture);
    assert_mesh(&doubled);
    assert_eq!(doubled.indices.len(), 12);
    near3(doubled.bounds.min, mesh.bounds.min);
    near3(doubled.bounds.max, mesh.bounds.max);
}

#[test]
fn nonuniform_scale_uses_inverse_transpose_for_normals() {
    let n = std::f32::consts::FRAC_1_SQRT_2;
    let mut fixture = GltfFixture::new(
        &[[0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
        Some(&UV),
        Some(&[[-n, 0.0, n]; 3]),
        Some(&[0, 1, 2]),
    );
    fixture.json["nodes"][0]["scale"] = json!([2.0, 3.0, 0.5]);
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    let length = 17.0f32.sqrt();
    for vertex in mesh.vertices {
        near3(vertex.normal, [-1.0 / length, 0.0, 4.0 / length]);
        near3(
            vertex.tangent[..3].try_into().unwrap(),
            [4.0 / length, 0.0, 1.0 / length],
        );
    }
}

#[test]
fn reflected_transform_preserves_front_faces_and_flips_tangent_handedness() {
    let mut fixture = GltfFixture::new(
        &TRIANGLE,
        Some(&UV),
        Some(&[[0.0, 0.0, 1.0]; 3]),
        Some(&[0, 1, 2]),
    );
    fixture.json["nodes"][0]["scale"] = json!([-1.0, 1.0, 1.0]);
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    for vertex in mesh.vertices {
        near3(vertex.normal, [0.0, 0.0, 1.0]);
        near3(vertex.tangent[..3].try_into().unwrap(), [-1.0, 0.0, 0.0]);
        near(vertex.tangent[3], -1.0);
    }
}

#[test]
fn missing_normals_are_area_weighted_at_shared_vertices() {
    let fixture = GltfFixture::new(
        &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        Some(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        None,
        Some(&[0, 1, 2, 0, 3, 1]),
    );
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    let shared = &mesh.vertices[mesh.indices[0] as usize];
    near3(
        shared.normal,
        [0.0, 1.0 / 5.0f32.sqrt(), 2.0 / 5.0f32.sqrt()],
    );
}

#[test]
fn obj_uv_seams_and_hard_normal_edges_are_preserved() {
    let seam = obj(b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.25 0\nvt 0.75 1\nf 1/1 2/2 3/3\nf 1/5 3/6 4/4\n");
    assert_mesh(&seam);
    for position in [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]] {
        let vertices: Vec<_> = seam
            .vertices
            .iter()
            .filter(|v| v.position == position)
            .collect();
        assert_eq!(vertices.len(), 2);
        assert_ne!(vertices[0].uv, vertices[1].uv);
    }
    let hard = obj(b"v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nvn 0 1 0\nf 1/1/1 2/2/1 3/3/1\nf 1/1/2 4/3/2 2/2/2\n");
    assert_mesh(&hard);
    let corner: Vec<_> = hard
        .vertices
        .iter()
        .filter(|v| v.position == [-1.0; 3])
        .collect();
    assert_eq!(corner.len(), 2);
    assert_ne!(corner[0].normal, corner[1].normal);
}

#[test]
fn gltf_uv_seams_are_not_welded_by_position() {
    let fixture = GltfFixture::new(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        Some(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.25, 0.0],
            [0.75, 1.0],
            [0.0, 1.0],
        ]),
        None,
        Some(&[0, 1, 2, 3, 4, 5]),
    );
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    assert_eq!(mesh.vertices.len(), 6);
    let corner: Vec<_> = mesh
        .vertices
        .iter()
        .filter(|v| v.position == [-1.0, -1.0, 0.0])
        .collect();
    assert_eq!(corner.len(), 2);
    assert_ne!(corner[0].uv, corner[1].uv);
}

#[test]
fn mirrored_uvs_reconstruct_the_bitangent_with_negative_handedness() {
    let fixture = GltfFixture::new(
        &TRIANGLE,
        Some(&[[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        None,
        Some(&[0, 1, 2]),
    );
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    for vertex in mesh.vertices {
        let tangent = vertex.tangent[..3].try_into().unwrap();
        near3(tangent, [-1.0, 0.0, 0.0]);
        near(vertex.tangent[3], -1.0);
        near3(
            cross(vertex.normal, tangent).map(|v| v * vertex.tangent[3]),
            [0.0, 1.0, 0.0],
        );
    }
}

#[test]
fn degenerate_uvs_emit_a_warning_and_a_deterministic_orthonormal_fallback() {
    let fixture = GltfFixture::new(&TRIANGLE, Some(&[[0.25, 0.75]; 3]), None, Some(&[0, 1, 2]));
    let first = gltf(&fixture);
    assert_mesh(&first);
    assert!(first.warnings.contains(&MeshWarning::DegenerateUv));
    for _ in 0..4 {
        let next = gltf(&fixture);
        assert_eq!(geometry_bits(&next), geometry_bits(&first));
        assert_eq!(next.warnings, first.warnings);
    }
    let obj = obj(b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0.25 0.75\nf 1/1 2/1 3/1\n");
    assert_mesh(&obj);
    assert!(obj.warnings.contains(&MeshWarning::DegenerateUv));
}

#[test]
fn repeated_imports_are_bitwise_deterministic_for_all_three_formats() {
    let fixture = GltfFixture::triangle();
    let json = fixture.document();
    let glb = fixture.glb();
    for (bytes, format, buffers) in [
        (OBJ, MeshFormat::Obj, vec![]),
        (
            json.as_slice(),
            MeshFormat::Gltf,
            vec![("geometry.bin", fixture.bin.as_slice())],
        ),
        (glb.as_slice(), MeshFormat::Glb, vec![]),
    ] {
        let first = decode_mesh_bytes(bytes, format, &buffers).unwrap();
        for _ in 0..4 {
            let next = decode_mesh_bytes(bytes, format, &buffers).unwrap();
            assert_eq!(geometry_bits(&next), geometry_bits(&first));
            assert_eq!(next.warnings, first.warnings);
        }
    }
}

#[test]
fn missing_or_partial_uvs_return_a_clear_error() {
    let fixture = GltfFixture::new(&TRIANGLE, None, None, Some(&[0, 1, 2]));
    assert!(reject_gltf(&fixture).contains("uv"));
    for bytes in [
        &b"v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n"[..],
        &b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nf 1/1 2/2 3\n"[..],
    ] {
        assert!(rejected(bytes, MeshFormat::Obj, &[]).contains("uv"));
    }
}

#[test]
fn invalid_indices_and_incomplete_triangles_are_rejected() {
    for indices in [&[0, 1, 3][..], &[0, 1, u32::MAX][..], &[0, 1][..]] {
        reject_gltf(&GltfFixture::new(&TRIANGLE, Some(&UV), None, Some(indices)));
    }
    for face in [
        "1/1 2/2 4/3",
        "0/1 2/2 3/3",
        "-4/1 -2/2 -1/3",
        "1/1 2/2 3/4",
    ] {
        let bytes = format!("v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nf {face}\n");
        rejected(bytes.as_bytes(), MeshFormat::Obj, &[]);
    }
}

#[test]
fn attribute_counts_must_match_positions() {
    let mut fixture = GltfFixture::triangle();
    fixture.json["accessors"][1]["count"] = json!(2);
    reject_gltf(&fixture);
    let fixture = GltfFixture::new(
        &TRIANGLE,
        Some(&UV),
        Some(&[[0.0, 0.0, 1.0]; 2]),
        Some(&[0, 1, 2]),
    );
    reject_gltf(&fixture);
}

#[test]
fn nonfinite_geometry_attributes_and_node_transforms_are_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Corrupt payloads after valid finite accessor bounds have been written.
        for byte in [0, TRIANGLE.len() * 12, TRIANGLE.len() * 12 + UV.len() * 8] {
            let mut fixture = GltfFixture::new(
                &TRIANGLE,
                Some(&UV),
                Some(&[[0.0, 0.0, 1.0]; 3]),
                Some(&[0, 1, 2]),
            );
            fixture.bin[byte..byte + 4].copy_from_slice(&value.to_le_bytes());
            reject_gltf(&fixture);
        }
    }
    for value in ["NaN", "inf", "-inf"] {
        let bytes =
            format!("v {value} 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n");
        rejected(bytes.as_bytes(), MeshFormat::Obj, &[]);
    }
    // Finite JSON numbers that overflow f32 must not reach normalization.
    let mut fixture = GltfFixture::triangle();
    fixture.json["nodes"][0]["translation"] = json!([1e100, 0.0, 0.0]);
    reject_gltf(&fixture);
}

#[test]
fn singular_transforms_and_zero_area_geometry_fail_cleanly() {
    let mut fixture = GltfFixture::triangle();
    fixture.json["nodes"][0]["scale"] = json!([0.0, 1.0, 1.0]);
    reject_gltf(&fixture);
    for positions in [
        [[0.0; 3]; 3],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ] {
        reject_gltf(&GltfFixture::new(
            &positions,
            Some(&UV),
            None,
            Some(&[0, 1, 2]),
        ));
    }
}

#[test]
fn nontriangle_gltf_modes_and_obj_primitives_are_rejected() {
    for mode in [0, 1, 2, 3, 5, 6] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["meshes"][0]["primitives"][0]["mode"] = json!(mode);
        reject_gltf(&fixture);
    }
    for primitive in ["f 1/1 2/2 3/3 4/4", "l 1/1 2/2", "p 1"] {
        let bytes = format!(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n{primitive}\n"
        );
        rejected(bytes.as_bytes(), MeshFormat::Obj, &[]);
    }
}

#[test]
fn skins_animations_and_morph_targets_are_explicitly_unsupported() {
    let mut skin = GltfFixture::triangle();
    skin.json["skins"] = json!([{"joints": [1]}]);
    skin.json["nodes"] = json!([{"mesh": 0, "skin": 0}, {}]);
    reject_gltf(&skin);
    // Valid scalar/VEC3 animation accessors; rejection must not depend on junk JSON.
    reject_gltf(&GltfFixture::animated());
    let mut morph = GltfFixture::triangle();
    morph.json["meshes"][0]["primitives"][0]["targets"] = json!([{"POSITION": 0}]);
    morph.json["meshes"][0]["weights"] = json!([1.0]);
    reject_gltf(&morph);
}

#[test]
fn draco_and_meshopt_compression_are_rejected_even_when_optional() {
    for required in [false, true] {
        for extension in ["KHR_draco_mesh_compression", "EXT_meshopt_compression"] {
            let mut fixture = GltfFixture::triangle();
            fixture.json["extensionsUsed"] = json!([extension]);
            if required {
                fixture.json["extensionsRequired"] = json!([extension]);
            }
            if extension == "KHR_draco_mesh_compression" {
                fixture.json["meshes"][0]["primitives"][0]["extensions"][extension] =
                    json!({"bufferView": 0, "attributes": {"POSITION": 0}});
            } else {
                fixture.json["bufferViews"][0]["extensions"][extension] = json!({
                    "buffer": 0, "byteOffset": 0, "byteLength": 36,
                    "byteStride": 12, "count": 3, "mode": "ATTRIBUTES", "filter": "NONE"
                });
            }
            reject_gltf(&fixture);
        }
    }
}

#[test]
fn network_geometry_uris_are_rejected_even_if_the_caller_supplies_bytes() {
    for uri in [
        "https://example.invalid/mesh.bin",
        "http://127.0.0.1/mesh.bin",
        "//host/mesh.bin",
        "ftp://host/mesh.bin",
        "file://host/mesh.bin",
    ] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["buffers"][0]["uri"] = json!(uri);
        rejected(
            &fixture.document(),
            MeshFormat::Gltf,
            &[(uri, fixture.bin.as_slice())],
        );
    }
}

#[test]
fn local_buffers_must_be_supplied_completely_and_are_not_opened_implicitly() {
    let fixture = GltfFixture::triangle();
    rejected(&fixture.document(), MeshFormat::Gltf, &[]);
    rejected(
        &fixture.document(),
        MeshFormat::Gltf,
        &[("unrelated.bin", fixture.bin.as_slice())],
    );
    rejected(
        &fixture.document(),
        MeshFormat::Gltf,
        &[("geometry.bin", &fixture.bin[..fixture.bin.len() - 1])],
    );
}

#[test]
fn imported_materials_and_image_uris_do_not_load_or_block_geometry() {
    let mut fixture = GltfFixture::triangle();
    let plain = geometry_bits(&gltf(&fixture));
    fixture.json["images"] = json!([{"uri": "https://example.invalid/not-an-image.png"}]);
    fixture.json["textures"] = json!([{"source": 0}]);
    fixture.json["materials"] =
        json!([{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]);
    fixture.json["meshes"][0]["primitives"][0]["material"] = json!(0);
    assert_eq!(geometry_bits(&gltf(&fixture)), plain);
    let decorated = [
        b"mtllib nonexistent-t064.mtl\nusemtl ignored\n".as_slice(),
        OBJ,
    ]
    .concat();
    assert_eq!(geometry_bits(&obj(&decorated)), geometry_bits(&obj(OBJ)));
}

#[test]
fn malformed_accessor_ranges_and_advertised_counts_are_bounded_errors() {
    for field in ["byteOffset", "byteLength"] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["bufferViews"][0][field] = json!(u64::MAX);
        reject_gltf(&fixture);
    }
    for field in ["count", "byteOffset"] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["accessors"][0][field] = json!(u64::MAX);
        reject_gltf(&fixture);
    }
    let mut fixture = GltfFixture::triangle();
    fixture.json["buffers"][0]["byteLength"] = json!(u64::MAX);
    reject_gltf(&fixture);
}

#[test]
fn truncated_glb_and_invalid_headers_never_return_partial_geometry() {
    let glb = GltfFixture::triangle().glb();
    for end in 0..glb.len() {
        rejected(&glb[..end], MeshFormat::Glb, &[]);
    }
    for (offset, value) in [(0, 0), (4, 1), (8, u32::MAX), (12, u32::MAX)] {
        let mut malformed = glb.clone();
        malformed[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        rejected(&malformed, MeshFormat::Glb, &[]);
    }
}

#[test]
fn empty_inputs_and_documents_without_triangles_are_errors() {
    for format in [MeshFormat::Obj, MeshFormat::Gltf, MeshFormat::Glb] {
        for bytes in [&b""[..], &b"garbage"[..]] {
            rejected(bytes, format, &[]);
        }
    }
    rejected(b"# comment only\n", MeshFormat::Obj, &[]);
    rejected(
        br#"{"asset":{"version":"2.0"},"scenes":[{}],"scene":0}"#,
        MeshFormat::Gltf,
        &[],
    );
}

#[test]
fn input_cap_is_inclusive_and_includes_external_geometry_buffers() {
    let mut bytes = OBJ.to_vec();
    // Short comment lines avoid making a line-length policy part of this cap test.
    let mut comment = vec![b'x'; 1024];
    comment[0] = b'#';
    comment[1023] = b'\n';
    while bytes.len() + comment.len() < MAX_INPUT {
        bytes.extend_from_slice(&comment);
    }
    bytes.extend_from_slice(b"#");
    bytes.resize(MAX_INPUT, b'x');
    assert_mesh(&obj(&bytes));
    bytes.push(b'x');
    for format in [MeshFormat::Obj, MeshFormat::Gltf, MeshFormat::Glb] {
        rejected(&bytes, format, &[]);
    }
    drop(bytes);
    let mut fixture = GltfFixture::triangle();
    fixture.bin.resize(MAX_INPUT, 0);
    fixture.json["buffers"][0]["byteLength"] = json!(MAX_INPUT);
    // Each source is <=64 MiB, but their aggregate is over the input budget.
    reject_gltf(&fixture);
}

#[test]
fn vertex_limit_accepts_the_boundary_and_rejects_one_over() {
    let at_limit = GltfFixture::repeated_vertices(MAX_VERTICES);
    let mesh = gltf(&at_limit);
    assert_mesh(&mesh);
    drop((at_limit, mesh));
    reject_gltf(&GltfFixture::repeated_vertices(MAX_VERTICES + 1));
}

#[test]
fn index_limit_accepts_the_boundary_and_rejects_the_next_triangle() {
    let at_limit = GltfFixture::repeated_indices(MAX_INDICES);
    let mesh = gltf(&at_limit);
    assert_mesh(&mesh);
    assert_eq!(mesh.indices.len(), MAX_INDICES);
    drop((at_limit, mesh));
    reject_gltf(&GltfFixture::repeated_indices(MAX_INDICES + 3));
}

#[test]
fn obj_vertex_and_index_caps_are_enforced_independently() {
    let at_vertices = fixtures::obj_vertices(MAX_VERTICES);
    assert_mesh(&obj(&at_vertices));
    drop(at_vertices);
    rejected(
        &fixtures::obj_vertices(MAX_VERTICES + 1),
        MeshFormat::Obj,
        &[],
    );
    let at_indices = fixtures::obj_indices(MAX_INDICES);
    let mesh = obj(&at_indices);
    assert_mesh(&mesh);
    assert_eq!(mesh.indices.len(), MAX_INDICES);
    drop((at_indices, mesh));
    rejected(
        &fixtures::obj_indices(MAX_INDICES + 3),
        MeshFormat::Obj,
        &[],
    );
}

#[test]
fn scene_expansion_counts_instances_against_both_geometry_limits() {
    for mut fixture in [
        GltfFixture::repeated_vertices(125_001),
        GltfFixture::repeated_indices(375_003),
    ] {
        fixture.json["nodes"] = json!([{"mesh": 0}, {"mesh": 0}]);
        fixture.json["scenes"][0]["nodes"] = json!([0, 1]);
        reject_gltf(&fixture);
    }
}

#[test]
fn interleaved_positions_and_normalized_integer_uvs_are_supported() {
    let mut fixture = GltfFixture::triangle();
    let start = fixture.bin.len();
    for position in TRIANGLE {
        for value in position {
            fixture.bin.extend_from_slice(&value.to_le_bytes());
        }
        fixture.bin.extend_from_slice(&[0; 4]);
    }
    fixture.json["bufferViews"][0] = json!({
        "buffer": 0, "byteOffset": start, "byteLength": 48, "byteStride": 16
    });
    let uv_start = fixture.json["bufferViews"][1]["byteOffset"]
        .as_u64()
        .unwrap() as usize;
    fixture.bin[uv_start..uv_start + 6].copy_from_slice(&[0, 0, 255, 0, 0, 255]);
    fixture.json["bufferViews"][1]["byteLength"] = json!(6);
    fixture.json["accessors"][1]["componentType"] = json!(5121);
    fixture.json["accessors"][1]["normalized"] = json!(true);
    fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    for (vertex, expected) in mesh.vertices.iter().zip(UV) {
        assert_eq!(vertex.uv, expected);
    }
}

#[test]
fn sparse_positions_are_supported_and_sparse_ranges_are_checked() {
    let mut fixture = GltfFixture::triangle();
    let start = fixture.bin.len();
    fixture.bin.extend_from_slice(&[0, 1, 2, 0]);
    let view = fixture.json["bufferViews"].as_array().unwrap().len();
    fixture.json["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "buffer": 0, "byteOffset": start, "byteLength": 3
        }));
    fixture.json["accessors"][0]
        .as_object_mut()
        .unwrap()
        .remove("bufferView");
    fixture.json["accessors"][0]["sparse"] = json!({
        "count": 3, "indices": {"bufferView": view, "componentType": 5121},
        "values": {"bufferView": 0}
    });
    fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    assert_eq!(
        geometry_bits(&mesh),
        geometry_bits(&gltf(&GltfFixture::triangle()))
    );
    for indices in [[0, 0, 2], [0, 2, 1], [0, 1, 3]] {
        fixture.bin[start..start + 3].copy_from_slice(&indices);
        reject_gltf(&fixture);
    }
    fixture.bin[start..start + 3].copy_from_slice(&[0, 1, 2]);
    fixture.json["accessors"][0]["sparse"]["values"]["byteOffset"] = json!(u64::MAX);
    reject_gltf(&fixture);
}

#[test]
fn accessor_shape_and_short_stride_errors_do_not_panic_in_the_decoder() {
    for (accessor, field, value) in [
        (0, "type", json!("VEC2")),
        (0, "componentType", json!(5123)),
        (1, "type", json!("VEC3")),
        (2, "type", json!("VEC2")),
    ] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["accessors"][accessor][field] = value;
        reject_gltf(&fixture);
    }
    let mut fixture = GltfFixture::triangle();
    fixture.json["bufferViews"][0]["byteStride"] = json!(4);
    reject_gltf(&fixture);
}

#[test]
fn cyclic_shared_and_duplicate_scene_nodes_are_rejected() {
    for nodes in [
        json!([{"mesh": 0, "children": [0]}]),
        json!([{"mesh": 0, "children": [1]}, {"children": [0]}]),
        json!([{"children": [2]}, {"children": [2]}, {"mesh": 0}]),
    ] {
        let mut fixture = GltfFixture::triangle();
        fixture.json["nodes"] = nodes;
        reject_gltf(&fixture);
    }
    let mut fixture = GltfFixture::triangle();
    fixture.json["scenes"][0]["nodes"] = json!([0, 0]);
    reject_gltf(&fixture);
    fixture.json["nodes"] = json!([{"children": [1]}, {"mesh": 0}]);
    fixture.json["scenes"][0]["nodes"] = json!([1]);
    reject_gltf(&fixture);
}

#[test]
fn deeply_nested_static_nodes_use_bounded_iterative_traversal() {
    let mut fixture = GltfFixture::triangle();
    let mut nodes: Vec<_> = (0..2048).map(|i| json!({"children": [i + 1]})).collect();
    nodes.push(json!({"mesh": 0}));
    fixture.json["nodes"] = json!(nodes);
    assert_mesh(&gltf(&fixture));
}

#[test]
fn ignored_metadata_names_do_not_masquerade_as_compression_extensions() {
    let mut fixture = GltfFixture::triangle();
    fixture.json["nodes"][0]["name"] = json!("KHR_draco_mesh_compression");
    fixture.json["images"] = json!([{"uri": "EXT_meshopt_compression"}]);
    assert_mesh(&gltf(&fixture));
}

#[test]
fn uri_table_rejects_ambiguous_absolute_and_traversing_buffer_names() {
    let mut fixture = GltfFixture::triangle();
    rejected(
        &fixture.document(),
        MeshFormat::Gltf,
        &[
            ("geometry.bin", &fixture.bin),
            ("geometry.bin", &fixture.bin),
        ],
    );
    for uri in [
        "../geometry.bin",
        "/geometry.bin",
        "C:/geometry.bin",
        "\\\\host\\mesh.bin",
        "%2f%2fhost/mesh.bin",
    ] {
        fixture.json["buffers"][0]["uri"] = json!(uri);
        rejected(
            &fixture.document(),
            MeshFormat::Gltf,
            &[(uri, &fixture.bin)],
        );
    }
}

#[test]
fn obj_group_expansion_is_preflighted() {
    let mut bytes = OBJ.to_vec();
    for _ in 1..(MAX_VERTICES / 3 + 1) {
        bytes.extend_from_slice(b"g repeated\nf 1/1 2/2 3/3\n");
    }
    rejected(&bytes, MeshFormat::Obj, &[]);
}

#[test]
fn obj_partial_normals_are_synthesized_only_for_missing_corners() {
    let mesh =
        obj(b"v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 7\nf 1/1/1 2/2 3/3\n");
    assert_mesh(&mesh);
    for vertex in mesh.vertices {
        near3(vertex.normal, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn reviewer_unused_sparse_aliases_do_not_amplify_validation_work() {
    let mut fixture = GltfFixture::triangle();
    let start = fixture.bin.len();
    for index in 0..MAX_INDICES as u32 {
        fixture.bin.extend_from_slice(&index.to_le_bytes());
    }
    // Poison the last entry so the old implementation fails after one scan,
    // rather than letting this regression execute 75 billion iterations.
    let end = fixture.bin.len();
    fixture.bin[end - 4..].copy_from_slice(&(MAX_INDICES as u32).to_le_bytes());
    let view = fixture.json["bufferViews"].as_array().unwrap().len();
    fixture.json["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "buffer": 0, "byteOffset": start, "byteLength": MAX_INDICES * 4
        }));
    let alias = json!({
        "componentType": 5125, "type": "SCALAR", "count": MAX_INDICES,
        "sparse": {
            "count": MAX_INDICES,
            "indices": {"bufferView": view, "componentType": 5125},
            "values": {"bufferView": view}
        }
    });
    fixture.json["accessors"]
        .as_array_mut()
        .unwrap()
        .extend(std::iter::repeat_n(alias, 100_000));
    fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
    assert!(fixture.document().len() + fixture.bin.len() < MAX_INPUT);
    assert_eq!(
        geometry_bits(&gltf(&fixture)),
        geometry_bits(&gltf(&GltfFixture::triangle()))
    );
}

#[test]
fn reviewer_geometry_limits_are_checked_before_referenced_sparse_payloads() {
    for source_expansion in [false, true] {
        let mut fixture = GltfFixture::repeated_vertices(125_001);
        let start = fixture.bin.len();
        fixture.bin.extend_from_slice(&125_001u32.to_le_bytes());
        let view = fixture.json["bufferViews"].as_array().unwrap().len();
        fixture.json["bufferViews"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "buffer": 0, "byteOffset": start, "byteLength": 4
            }));
        fixture.json["accessors"][0]["sparse"] = json!({
            "count": 1, "indices": {"bufferView": view, "componentType": 5125},
            "values": {"bufferView": 0}
        });
        fixture.json["buffers"][0]["byteLength"] = json!(fixture.bin.len());
        if source_expansion {
            let mesh = fixture.json["meshes"][0].clone();
            fixture.json["meshes"].as_array_mut().unwrap().push(mesh);
        } else {
            fixture.json["nodes"] = json!([{"mesh": 0}, {"mesh": 0}]);
            fixture.json["scenes"][0]["nodes"] = json!([0, 1]);
        }
        assert!(reject_gltf(&fixture).contains("limit"));
    }
}

const MIRRORED_POSITIONS: [[f32; 3]; 4] = [
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
];
const MIRRORED_UVS: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

#[test]
fn reviewer_shared_mirrored_uv_corners_keep_both_tangent_frames() {
    let fixture = GltfFixture::new(
        &MIRRORED_POSITIONS,
        Some(&MIRRORED_UVS),
        None,
        Some(&[0, 1, 2, 1, 3, 2]),
    );
    let obj = obj(b"v -1 0 0\nv 0 0 0\nv 0 1 0\nv 1 0 0\nvt 1 0\nvt 0 0\nvt 0 1\nvt 1 0\nf 1/1 2/2 3/3\nf 2/2 4/4 3/3\n");
    for mesh in [
        gltf(&fixture),
        decode_mesh_bytes(&fixture.glb(), MeshFormat::Glb, &[]).unwrap(),
        obj,
    ] {
        assert_mesh(&mesh);
        assert_eq!(mesh.vertices.len(), 6);
        assert!(mesh.warnings.is_empty());
        for (triangle, sign) in mesh.indices.as_chunks::<3>().0.iter().zip([-1.0, 1.0]) {
            for &index in triangle {
                let vertex = &mesh.vertices[index as usize];
                let tangent = vertex.tangent[..3].try_into().unwrap();
                near3(tangent, [sign, 0.0, 0.0]);
                near(vertex.tangent[3], sign);
                near3(
                    cross(vertex.normal, tangent).map(|v| v * sign),
                    [0.0, 1.0, 0.0],
                );
            }
        }
        for (a, b) in [
            (mesh.indices[1], mesh.indices[3]),
            (mesh.indices[2], mesh.indices[5]),
        ] {
            assert_ne!(a, b);
            assert_eq!(
                mesh.vertices[a as usize].position,
                mesh.vertices[b as usize].position
            );
            assert_eq!(mesh.vertices[a as usize].uv, mesh.vertices[b as usize].uv);
            assert_eq!(
                mesh.vertices[a as usize].normal,
                mesh.vertices[b as usize].normal
            );
        }
    }
}

#[test]
fn reviewer_tangent_splits_preserve_area_weighted_smooth_normals() {
    let mut positions = MIRRORED_POSITIONS;
    positions[3][2] = 1.0;
    let mesh = gltf(&GltfFixture::new(
        &positions,
        Some(&MIRRORED_UVS),
        None,
        Some(&[0, 1, 2, 1, 3, 2]),
    ));
    assert_mesh(&mesh);
    assert_eq!(mesh.vertices.len(), 6);
    let expected = [-1.0 / 5.0f32.sqrt(), 0.0, 2.0 / 5.0f32.sqrt()];
    for i in [1, 2, 3, 5] {
        near3(mesh.vertices[mesh.indices[i] as usize].normal, expected);
    }
}

#[test]
fn degenerate_uv_corners_do_not_replace_a_shared_valid_tangent_frame() {
    let mut uvs = MIRRORED_UVS;
    uvs[3] = uvs[1];
    let fixture = GltfFixture::new(
        &MIRRORED_POSITIONS,
        Some(&uvs),
        None,
        Some(&[0, 1, 2, 1, 3, 2]),
    );
    let mesh = gltf(&fixture);
    assert_mesh(&mesh);
    assert_eq!(mesh.vertices.len(), 6);
    assert_eq!(mesh.warnings, [MeshWarning::DegenerateUv]);
    for (&index, expected) in mesh.indices.iter().zip(
        [[-1.0, 0.0, 0.0]; 3]
            .into_iter()
            .chain([[0.0, -1.0, 0.0]; 3]),
    ) {
        near3(
            mesh.vertices[index as usize].tangent[..3]
                .try_into()
                .unwrap(),
            expected,
        );
    }
    assert_eq!(geometry_bits(&mesh), geometry_bits(&gltf(&fixture)));
}

#[test]
fn reviewer_tangent_expansion_obeys_the_vertex_limit() {
    for (count, accepted) in [(MAX_VERTICES - 2, true), (MAX_VERTICES - 1, false)] {
        let mut positions: Vec<_> = (0..count).map(|i| TRIANGLE[i % 3]).collect();
        let mut uvs: Vec<_> = (0..count).map(|i| UV[i % 3]).collect();
        positions[..4].copy_from_slice(&MIRRORED_POSITIONS);
        uvs[..4].copy_from_slice(&MIRRORED_UVS);
        let mut indices = vec![0, 1, 2, 1, 3, 2];
        indices.extend((4..4 + (count - 4) / 3 * 3).map(|i| i as u32));
        let fixture = GltfFixture::new(&positions, Some(&uvs), None, Some(&indices));
        if accepted {
            let mesh = gltf(&fixture);
            assert_mesh(&mesh);
            assert_eq!(mesh.vertices.len(), MAX_VERTICES);
        } else {
            assert!(reject_gltf(&fixture).contains("limit"));
        }
    }
}
