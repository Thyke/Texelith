struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
struct Sample {
    geometry: vec4<f32>,
    weights: vec4<f32>,
}
@group(0) @binding(0) var height: texture_2d<f32>;
@group(0) @binding(1) var ao: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> u: NodeUniform;
struct Samples { entries: array<Sample, SAMPLE_COUNT>, }
@group(1) @binding(0) var<uniform> samples: Samples;
override DIRECTIONS: u32 = 8u;
override STEPS: u32 = 8u;

fn fetch(p: vec2<i32>, interior: bool) -> f32 {
    if interior { return textureLoad(height, p, 0).r; }
    let size = vec2<i32>(u.dimensions.xy);
    if u.flags.x != 0u {
        return textureLoad(height, ((p % size) + size) % size, 0).r;
    }
    return textureLoad(height, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0).r;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let margin = vec2<i32>(u.parameters.yz);
    let interior = all(p >= margin) && all(p < vec2<i32>(u.dimensions.xy) - margin);
    let center = textureLoad(height, p, 0).r;
    var visibility = 0.0;
    for (var d = 0u; d < DIRECTIONS; d++) {
        var horizon = 0.0;
        for (var k = 0u; k < STEPS; k++) {
            let s = samples.entries[d * STEPS + k];
            let base = p + vec2<i32>(s.geometry.xy);
            // Interpolate differences so constant signed heights stay exactly flat.
            var delta = (fetch(base, interior) - center) * s.weights.x;
            if s.weights.y > 0.0 { delta += (fetch(base + vec2<i32>(1, 0), interior) - center) * s.weights.y; }
            if s.weights.z > 0.0 { delta += (fetch(base + vec2<i32>(0, 1), interior) - center) * s.weights.z; }
            if s.weights.w > 0.0 { delta += (fetch(base + vec2<i32>(1, 1), interior) - center) * s.weights.w; }
            horizon = max(horizon, delta * s.geometry.z - u.parameters.x);
        }
        visibility += 1.0 / (1.0 + horizon * horizon);
    }
    textureStore(ao, p, vec4<f32>(clamp(visibility / f32(DIRECTIONS), 0.0, 1.0)));
}
