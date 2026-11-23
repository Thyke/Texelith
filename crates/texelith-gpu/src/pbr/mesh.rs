use std::f32::consts::{PI, TAU};
use texelith_core::{MeshData, PreviewMesh, Vertex};

/// Unit-radius bounds; hard edges, UV seams and cap vertices remain split.
/// UV v increases down the image, while tangent-space +Y points up.
pub fn primitive(kind: &PreviewMesh) -> Result<MeshData, crate::GpuError> {
    let mut mesh = MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    match kind {
        PreviewMesh::PreviewBall => preview_ball(&mut mesh),
        PreviewMesh::RoundedCube => rounded_cube(&mut mesh),
        PreviewMesh::Plane => face(
            &mut mesh,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
            1.0 / 2.0f32.sqrt(),
        ),
        PreviewMesh::Cube => {
            let s = 1.0 / 3.0f32.sqrt();
            for (n, t, b) in [
                ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
                ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
                ([1., 0., 0.], [0., 0., -1.], [0., 1., 0.]),
                ([-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]),
                ([0., 1., 0.], [1., 0., 0.], [0., 0., -1.]),
                ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
            ] {
                face(&mut mesh, n, t, b, s, s);
            }
        }
        PreviewMesh::Sphere => {
            let (slices, stacks) = (64, 32);
            for j in 0..=stacks {
                let phi = PI * j as f32 / stacks as f32;
                for i in 0..=slices {
                    let theta = TAU * i as f32 / slices as f32;
                    let n = [phi.sin() * theta.sin(), phi.cos(), phi.sin() * theta.cos()];
                    mesh.vertices.push(Vertex {
                        position: n,
                        normal: n,
                        uv: [i as f32 / slices as f32, j as f32 / stacks as f32],
                        tangent: [theta.cos(), 0., -theta.sin(), 1.],
                    });
                }
            }
            for j in 0..stacks {
                for i in 0..slices {
                    let a = j * (slices + 1) + i;
                    // One triangle at each pole avoids degenerate geometry.
                    if j > 0 {
                        mesh.indices.extend([a, a + slices + 1, a + 1]);
                    }
                    if j + 1 < stacks {
                        mesh.indices.extend([a + 1, a + slices + 1, a + slices + 2]);
                    }
                }
            }
        }
        PreviewMesh::Cylinder => {
            let r = 1.0 / 2.0f32.sqrt();
            let slices = 64;
            for j in 0..=1 {
                for i in 0..=slices {
                    let theta = TAU * i as f32 / slices as f32;
                    mesh.vertices.push(Vertex {
                        position: [r * theta.sin(), r * (1. - 2. * j as f32), r * theta.cos()],
                        normal: [theta.sin(), 0., theta.cos()],
                        uv: [i as f32 / slices as f32, j as f32],
                        tangent: [theta.cos(), 0., -theta.sin(), 1.],
                    });
                }
            }
            for i in 0..slices {
                mesh.indices.extend([
                    i,
                    i + slices + 1,
                    i + 1,
                    i + 1,
                    i + slices + 1,
                    i + slices + 2,
                ]);
            }
            for sign in [-1.0, 1.0] {
                let start = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    position: [0., sign * r, 0.],
                    normal: [0., sign, 0.],
                    uv: [0.5, 0.5],
                    tangent: [1., 0., 0., 1.],
                });
                for i in 0..=slices {
                    let theta = TAU * i as f32 / slices as f32;
                    let (x, z) = (theta.sin(), theta.cos());
                    mesh.vertices.push(Vertex {
                        position: [x * r, sign * r, z * r],
                        normal: [0., sign, 0.],
                        uv: [0.5 + 0.5 * x, 0.5 + 0.5 * sign * z],
                        tangent: [1., 0., 0., 1.],
                    });
                }
                for i in 0..slices {
                    let (a, b) = (start + 1 + i, start + 2 + i);
                    mesh.indices.extend(if sign > 0. {
                        [start, a, b]
                    } else {
                        [start, b, a]
                    });
                }
            }
        }
        PreviewMesh::Custom(_) => {
            return Err(crate::GpuError(
                "primitive generator requires a built-in mesh".into(),
            ));
        }
    }
    Ok(mesh)
}
fn face(mesh: &mut MeshData, n: [f32; 3], t: [f32; 3], b: [f32; 3], offset: f32, s: f32) {
    let start = mesh.vertices.len() as u32;
    for (x, y, u, v) in [
        (-1., -1., 0., 1.),
        (1., -1., 1., 1.),
        (1., 1., 1., 0.),
        (-1., 1., 0., 0.),
    ] {
        mesh.vertices.push(Vertex {
            position: std::array::from_fn(|i| n[i] * offset + s * (x * t[i] + y * b[i])),
            normal: n,
            uv: [u, v],
            tangent: [t[0], t[1], t[2], 1.],
        });
    }
    mesh.indices
        .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.map(|x| x / length)
}

/// Lathed sphere and pedestal. The only longitude seam faces -Z; arc-length V
/// gives the groove and base the same density as the sphere's equator.
fn preview_ball(mesh: &mut MeshData) {
    let slices = 96u32;
    let mut profile = Vec::new();
    for j in 0..=48 {
        let phi = 2.25 * j as f32 / 48.;
        profile.push([0.78 * phi.sin(), 0.17 + 0.78 * phi.cos()]);
    }
    profile.extend([
        [0.58, -0.35],
        [0.57, -0.39],
        [0.58, -0.42],
        [0.63, -0.44],
        [0.69, -0.46],
        [0.72, -0.49],
        [0.73, -0.52],
        [0.73, -0.56],
        [0.72, -0.58],
        [0.68, -0.60],
        [0.67, -0.62],
        [0.67, -0.64],
        [0.68, -0.66],
        [0.72, -0.68],
        [0.73, -0.70],
        [0.73, -0.74],
        [0.72, -0.77],
        [0.69, -0.80],
        [0.65, -0.81],
        [0.4, -0.81],
        [0., -0.81],
    ]);
    let radius = profile
        .iter()
        .map(|[r, y]| (r * r + y * y).sqrt())
        .fold(0.0f32, f32::max);
    let mut arc = 0.;
    for (j, &[r, y]) in profile.iter().enumerate() {
        let previous = profile[j.saturating_sub(1)];
        let next = profile[(j + 1).min(profile.len() - 1)];
        if j > 0 {
            arc += (r - previous[0]).hypot(y - previous[1]);
        }
        let [nr, ny, _] = if j == 0 {
            [0., 1., 0.]
        } else {
            normalize([previous[1] - next[1], next[0] - previous[0], 0.])
        };
        for i in 0..=slices {
            let theta = PI + TAU * i as f32 / slices as f32;
            mesh.vertices.push(Vertex {
                position: [
                    r * theta.sin() / radius,
                    y / radius,
                    r * theta.cos() / radius,
                ],
                normal: [nr * theta.sin(), ny, nr * theta.cos()],
                uv: [i as f32 / slices as f32, arc / (TAU * 0.78)],
                tangent: [theta.cos(), 0., -theta.sin(), 1.],
            });
        }
    }
    for j in 0..profile.len() as u32 - 1 {
        for i in 0..slices {
            let a = j * (slices + 1) + i;
            if profile[j as usize][0] > 0. {
                mesh.indices.extend([a, a + slices + 1, a + 1]);
            }
            if profile[j as usize + 1][0] > 0. {
                mesh.indices.extend([a + 1, a + slices + 1, a + slices + 2]);
            }
        }
    }
}

/// Six tessellated rounded-box patches. Flat face UVs remain undistorted;
/// normals and analytic UV tangents continue smoothly around the bevels.
fn rounded_cube(mesh: &mut MeshData) {
    let steps = 32u32;
    let half = 0.58;
    let inner = 0.38;
    for (n, t, b) in [
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
        ([1., 0., 0.], [0., 0., -1.], [0., 1., 0.]),
        ([-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]),
        ([0., 1., 0.], [1., 0., 0.], [0., 0., -1.]),
        ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
    ] {
        let start = mesh.vertices.len() as u32;
        for j in 0..=steps {
            for i in 0..=steps {
                let x = half * (2. * i as f32 / steps as f32 - 1.);
                let y = half * (2. * j as f32 / steps as f32 - 1.);
                let p: [f32; 3] = std::array::from_fn(|k| n[k] * half + t[k] * x + b[k] * y);
                let q = p.map(|v| v.clamp(-inner, inner));
                let normal = normalize(std::array::from_fn(|k| p[k] - q[k]));
                let tangent = if x.abs() <= inner {
                    t
                } else {
                    let dot = t.iter().zip(normal).map(|(a, b)| a * b).sum::<f32>();
                    normalize(std::array::from_fn(|k| t[k] - normal[k] * dot))
                };
                mesh.vertices.push(Vertex {
                    position: std::array::from_fn(|k| q[k] + (half - inner) * normal[k]),
                    normal,
                    uv: [i as f32 / steps as f32, 1. - j as f32 / steps as f32],
                    tangent: [tangent[0], tangent[1], tangent[2], 1.],
                });
            }
        }
        for j in 0..steps {
            for i in 0..steps {
                let a = start + j * (steps + 1) + i;
                mesh.indices
                    .extend([a, a + 1, a + steps + 2, a, a + steps + 2, a + steps + 1]);
            }
        }
    }
}
