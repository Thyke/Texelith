//! Seeded tile inspection. UV rows point down; signed tangent normals use +Y up.
use texelith_core::{MeshData, Tiles};

pub fn quarter_turn(cell: [u32; 2], seed: u32) -> u32 {
    let mut h = seed ^ cell[0].wrapping_mul(0x9e3779b9) ^ cell[1].wrapping_mul(0x85ebca6b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h & 3
}
pub fn tile_uv(uv: [f32; 2], tiles: Tiles) -> ([f32; 2], u32) {
    let p = uv.map(|v| v * tiles.count as f32);
    let cell = p.map(|v| v.floor().max(0.) as u32);
    let turns = if tiles.random_rotation {
        quarter_turn(cell, tiles.seed)
    } else {
        0
    };
    let [mut x, mut y] = p.map(|v| v - v.floor() - 0.5);
    for _ in 0..turns {
        (x, y) = (-y, x);
    }
    ([x + 0.5, y + 0.5], turns)
}
pub fn rotate_normal([mut x, mut y, z]: [f32; 3], turns: u32) -> [f32; 3] {
    for _ in 0..turns % 4 {
        (x, y) = (-y, x);
    }
    [x, y, z]
}
pub(crate) fn tile_uniform(tiles: Tiles) -> [f32; 4] {
    [
        tiles.count as f32,
        (tiles.seed & 65535) as f32,
        (tiles.seed >> 16) as f32,
        u32::from(tiles.random_rotation) as f32,
    ]
}
pub fn validate_mesh(mesh: &MeshData) -> Result<(), crate::GpuError> {
    if mesh.vertices.is_empty()
        || mesh.vertices.len() > 250_000
        || mesh.indices.is_empty()
        || mesh.indices.len() > 750_000
        || !mesh.indices.len().is_multiple_of(3)
        || mesh
            .indices
            .iter()
            .any(|i| *i as usize >= mesh.vertices.len())
        || mesh.vertices.iter().any(|v| {
            v.position
                .iter()
                .chain(&v.normal)
                .chain(&v.uv)
                .chain(&v.tangent)
                .any(|v| !v.is_finite())
        })
    {
        return Err(crate::GpuError("invalid bounded triangle mesh".into()));
    }
    Ok(())
}
