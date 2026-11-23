use texelith_core::PreviewMesh;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[test]
fn inspection_meshes_have_valid_geometry_and_uv_tangent_frames() {
    for kind in [PreviewMesh::PreviewBall, PreviewMesh::RoundedCube] {
        let mesh = texelith_gpu::pbr::primitive(&kind).unwrap();
        assert!(mesh.vertices.len() > 4000 && mesh.vertices.len() < 10000);
        assert!(mesh.indices.len() < 40000);
        for v in &mesh.vertices {
            assert!(
                v.position
                    .iter()
                    .chain(&v.normal)
                    .chain(&v.uv)
                    .chain(&v.tangent)
                    .all(|x| x.is_finite())
            );
            assert!(dot(v.position, v.position) <= 1.00001);
            assert!((dot(v.normal, v.normal) - 1.).abs() < 2e-6);
            let t = v.tangent[..3].try_into().unwrap();
            assert!((dot(t, t) - 1.).abs() < 2e-6);
            assert!(dot(v.normal, t).abs() < 2e-6);
            assert_eq!(v.tangent[3], 1.);
        }
        for triangle in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = std::array::from_fn(|i| mesh.vertices[triangle[i] as usize]);
            let (e1, e2) = (sub(b.position, a.position), sub(c.position, a.position));
            let geometric = cross(e1, e2);
            assert!(dot(geometric, geometric) > 1e-12, "degenerate {kind:?}");
            assert!(dot(geometric, a.normal) > 0., "inward {kind:?}");
            let (du1, dv1) = (b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]);
            let (du2, dv2) = (c.uv[0] - a.uv[0], c.uv[1] - a.uv[1]);
            let det = du1 * dv2 - du2 * dv1;
            assert!(det.abs() > 1e-8);
            let tangent = std::array::from_fn(|i| (e1[i] * dv2 - e2[i] * dv1) / det);
            let up = std::array::from_fn(|i| -(e2[i] * du1 - e1[i] * du2) / det);
            assert!(dot(tangent, a.tangent[..3].try_into().unwrap()) > 0.);
            assert!(dot(up, cross(a.normal, a.tangent[..3].try_into().unwrap())) > 0.);
            // Longitude wrapping is split on the back, never across a triangle.
            assert!(du1.abs() < 0.04 && du2.abs() < 0.04);
        }
    }
}

#[test]
fn preview_ball_places_wrap_on_back_and_round_cube_preserves_flat_face_density() {
    assert_eq!(PreviewMesh::default(), PreviewMesh::PreviewBall);
    let ball = texelith_gpu::pbr::primitive(&PreviewMesh::PreviewBall).unwrap();
    for v in ball
        .vertices
        .iter()
        .filter(|v| v.uv[0] == 0. || v.uv[0] == 1.)
    {
        assert!(v.position[2] <= 0.000001);
        assert!(v.position[0].abs() < 0.000001);
    }
    let cube = texelith_gpu::pbr::primitive(&PreviewMesh::RoundedCube).unwrap();
    for v in cube.vertices.iter().filter(|v| {
        v.position[0].abs() < 0.38 && v.position[1].abs() < 0.38 && v.position[2] > 0.57
    }) {
        assert_eq!(v.normal, [0., 0., 1.]);
        assert!((v.position[0] - (v.uv[0] * 2. - 1.) * 0.58).abs() < 1e-6);
        assert!((v.position[1] - (1. - v.uv[1] * 2.) * 0.58).abs() < 1e-6);
    }
}
