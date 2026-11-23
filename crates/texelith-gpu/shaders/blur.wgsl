struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

struct BlurWeights { slots: array<vec4<f32>, 8>, }
@group(1) @binding(0) var<uniform> weights: BlurWeights;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let radius = i32(node.flags.y);
    let limit = vec2<i32>(node.dimensions.xy) - vec2<i32>(1);
    var total = 0.0;
    for (var tap = -radius; tap <= radius; tap += 1) {
        let delta = select(vec2<i32>(tap, 0), vec2<i32>(0, tap), node.flags.x == 1u);
        let index = u32(tap + radius);
        let weight = weights.slots[index / 4u][index % 4u];
        total += textureLoad(input_map, clamp(p + delta, vec2<i32>(0), limit), 0).r * weight;
    }
    textureStore(output_map, p, vec4<f32>(total, 0.0, 0.0, 1.0));
}
