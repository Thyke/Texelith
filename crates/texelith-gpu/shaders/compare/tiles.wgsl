fn tile_turn(cell: vec2<u32>, seed: u32) -> u32 {
    var h = seed ^ (cell.x * 0x9e3779b9u) ^ (cell.y * 0x85ebca6bu);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    return h & 3u;
}
fn tile_rotation(uv: vec2<f32>) -> u32 {
    if material.reserved1.w < 0.5 { return 0u; }
    return tile_turn(vec2<u32>(max(floor(uv * max(material.reserved1.x,1.0)),vec2(0.0))), (u32(material.reserved1.y) | (u32(material.reserved1.z)<<16u)));
}
fn tiled_uv(uv: vec2<f32>, turns: u32) -> vec2<f32> {
    let count = max(material.reserved1.x,1.0);
    if count == 1.0 && material.reserved1.w < 0.5 { return uv; }
    var p = fract(uv * count) - 0.5;
    for (var i=0u; i<turns; i++) { p = vec2(-p.y,p.x); }
    return p + 0.5;
}
fn tile_normal(n: vec3<f32>, turns: u32) -> vec3<f32> {
    var xy = n.xy;
    for (var i=0u; i<turns; i++) { xy = vec2(-xy.y,xy.x); }
    return vec3(xy,n.z);
}

fn tile_view(v: vec3<f32>, turns: u32) -> vec3<f32> {
    var xy=v.xy;
    for(var i=0u;i<turns;i++) { xy=vec2(-xy.y,xy.x); }
    return vec3(xy,v.z);
}
