use texelith_core::Tiles;
use texelith_gpu::compare::*;

#[test]
fn rotation_is_seeded_and_normals_rotate_back_from_texture_coordinates() {
    let tiles = Tiles {
        count: 3,
        seed: 17,
        random_rotation: true,
    };
    let turns: Vec<_> = (0..9)
        .map(|i| quarter_turn([i % 3, i / 3], tiles.seed))
        .collect();
    assert_eq!(
        turns,
        (0..9)
            .map(|i| quarter_turn([i % 3, i / 3], tiles.seed))
            .collect::<Vec<_>>()
    );
    assert!(turns.iter().any(|t| *t != turns[0]));
    assert_eq!(rotate_normal([1., 0., 0.], 1), [0., 1., 0.]);
    assert_eq!(rotate_normal([0., 1., 0.], 2), [0., -1., 0.]);
    assert_eq!(rotate_normal([0., 0., 1.], 3), [0., 0., 1.]);
    assert_eq!(tile_uv([0.25, 0.75], Tiles::default()).0, [0.25, 0.75]);
}

#[test]
fn rotated_normal_matches_derivatives_of_a_rotated_height_ramp() {
    // h(q)=q.x, q=(-v,u): dh/du=0, dh/dv=-1. GL n=(-du,+dv,1).
    let original = [-1.0, 0.0, 1.0];
    let rotated = rotate_normal(original, 1);
    assert_eq!(rotated, [0.0, -1.0, 1.0]);
    assert_ne!(rotated, [0.0, 1.0, 1.0]);
}
