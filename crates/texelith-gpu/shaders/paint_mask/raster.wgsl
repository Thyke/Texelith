struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
struct Segment { a: vec4<f32>, b: vec4<f32> }
@group(0) @binding(0) var checkpoint: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> u: NodeUniform;
@group(1) @binding(0) var<storage, read_write> pixels: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read> segments: array<Segment>;

fn coverage(p: vec2<f32>, s: Segment) -> f32 {
    let delta = s.b.xy - s.a.xy;
    let len2 = dot(delta, delta);
    var t = 0.0;
    if len2 > 0.0 { t = clamp(dot(p - s.a.xy, delta) / len2, 0.0, 1.0); }
    let d = distance(p, s.a.xy + delta * t);
    if d >= u.parameters.x { return 0.0; }
    // Stationary samples still contribute to the stroke's maximum coverage.
    var pressure = max(s.a.z, s.b.z);
    if len2 > 0.0 { pressure = mix(s.a.z, s.b.z, t); }
    if u.parameters.y >= 1.0 { return pressure; }
    return pressure * (1.0 - smoothstep(u.parameters.x * u.parameters.y, u.parameters.x, d));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.zw) { return; }
    let pos = id.xy + u.reserved.yz;
    let index = pos.y * u.dimensions.x + pos.x;
    if u.flags.x == 0u {
        var value = 0.0;
        if u.flags.y == 1u { value = textureLoad(checkpoint, vec2<i32>(pos), 0).x; }
        pixels[index] = vec2<f32>(value, 0.0);
        return;
    }
    if u.flags.x == 3u { textureStore(output_map, pos, vec4<f32>(pixels[index].x)); return; }
    let uv = (vec2<f32>(pos) + vec2<f32>(0.5)) / vec2<f32>(u.dimensions.xy);
    var alpha = pixels[index].y;
    for (var i = 0u; i < u.flags.y; i++) {
        if alpha >= 1.0 { break; }
        let s = segments[i];
        let lo = min(s.a.xy, s.b.xy) - vec2<f32>(u.parameters.x);
        let hi = max(s.a.xy, s.b.xy) + vec2<f32>(u.parameters.x);
        if u.flags.z == 0u {
            if all(uv >= lo) && all(uv <= hi) { alpha = max(alpha, coverage(uv, s)); }
            continue;
        }
        // Translation ranges cull periodic images before segment projection.
        let low = max(vec2<i32>(-1), vec2<i32>(ceil(lo - uv)));
        let high = min(vec2<i32>(1), vec2<i32>(floor(hi - uv)));
        for (var y = low.y; y <= high.y; y++) {
            for (var x = low.x; x <= high.x; x++) {
                let p = uv + vec2<f32>(f32(x), f32(y));
                if all(p >= lo) && all(p <= hi) { alpha = max(alpha, coverage(p, s)); }
            }
        }
    }
    var value = pixels[index].x;
    if u.reserved.x == 1u {
        let opacity = alpha * u.parameters.z;
        if u.flags.w == 0u { value += opacity; }
        else if u.flags.w == 1u { value -= opacity; }
        else { value += (1.0 - value) * opacity; }
        alpha = 0.0;
    }
    pixels[index] = vec2<f32>(clamp(value, 0.0, 1.0), alpha);
}
