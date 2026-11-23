use texelith_core::PreviewMesh;
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
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
fn primitives_have_outward_nondegenerate_triangles_and_correct_tangent_frames() {
    for kind in [
        PreviewMesh::Plane,
        PreviewMesh::Cube,
        PreviewMesh::Sphere,
        PreviewMesh::Cylinder,
    ] {
        let mesh = texelith_gpu::pbr::primitive(&kind).unwrap();
        assert!(
            mesh.vertices.len() >= 4
                && mesh.indices.len() >= 6
                && mesh.indices.len().is_multiple_of(3)
        );
        for v in &mesh.vertices {
            assert!(
                v.position
                    .iter()
                    .chain(v.normal.iter())
                    .chain(v.uv.iter())
                    .chain(v.tangent.iter())
                    .all(|v| v.is_finite())
            );
            assert!(dot(v.position, v.position) <= 1.00001);
            assert!((dot(v.normal, v.normal) - 1.).abs() < 2e-6);
            let t = v.tangent[..3].try_into().unwrap();
            assert!((dot(t, t) - 1.).abs() < 2e-6);
            assert!(dot(t, v.normal).abs() < 2e-6);
            assert!(v.uv.iter().all(|x| (0.0..=1.0).contains(x)));
            assert_eq!(v.tangent[3], 1.);
        }
        for tri in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = std::array::from_fn(|i| mesh.vertices[tri[i] as usize]);
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
            assert!(
                dot(up, cross(a.normal, a.tangent[..3].try_into().unwrap())) > 0.,
                "wrong image-row/TBN handedness {kind:?}"
            );
        }
        if matches!(kind, PreviewMesh::Sphere | PreviewMesh::Cylinder) {
            assert!(
                mesh.vertices
                    .iter()
                    .any(|a| mesh.vertices.iter().any(|b| dot(
                        sub(a.position, b.position),
                        sub(a.position, b.position)
                    ) < 1e-10
                        && (a.uv[0] - b.uv[0]).abs() > 0.99)),
                "UV seam must be split"
            );
        }
    }
}
