struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
};
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var mask_map: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    var rgb = textureLoad(color, vec2<i32>(id.xy), 0).rgb;
    if any(node.dimensions.xy != node.dimensions.zw) {
        let scale = vec2<f32>(node.dimensions.zw) / vec2<f32>(node.dimensions.xy);
        let lo = vec2<f32>(id.xy) * scale;
        let hi = vec2<f32>(id.xy + vec2<u32>(1u)) * scale;
        // Area restriction in linear RGB; differences preserve constant cells.
        let anchor = textureLoad(color, vec2<i32>(lo), 0).rgb;
        var delta = vec3<f32>(0.0);
        var total = 0.0;
        for (var y = u32(floor(lo.y)); y < min(u32(ceil(hi.y)), node.dimensions.w); y++) {
            for (var x = u32(floor(lo.x)); x < min(u32(ceil(hi.x)), node.dimensions.z); x++) {
                let overlap = max(vec2<f32>(0.0), min(hi, vec2<f32>(f32(x + 1u), f32(y + 1u))) - max(lo, vec2<f32>(f32(x), f32(y))));
                let area = overlap.x * overlap.y;
                delta += (textureLoad(color, vec2<i32>(i32(x), i32(y)), 0).rgb - anchor) * area;
                total += area;
            }
        }
        rgb = anchor + delta / total;
    }
    let distance = length(rgb - node.parameters.xyz);
    let radius = node.parameters.w;
    let threshold = bitcast<f32>(node.reserved.x);
    let feather = bitcast<f32>(node.reserved.y);
    var selected = select(0.0, 1.0, distance <= radius);
    if feather > 0.0 {
        // Divide the difference instead of adding radius + feather: the latter
        // loses small feathers or overflows at legal extreme parameter values.
        let t = clamp((distance - radius) / feather, 0.0, 1.0);
        let coverage = 1.0 - t * t * (3.0 - 2.0 * t);
        selected = select(0.0, 1.0, coverage >= threshold);
    }
    // Optional mask ports resolve to ConstantOne before composition (ADR 0004).
    var mask = 1.0;
    if node.flags.x != 0u {
        mask = clamp(textureLoad(mask_map, vec2<i32>(id.xy), 0).r, 0.0, 1.0);
    }
    var result = mask;
    switch node.flags.y {
        case 0u: {}
        case 1u: { result = clamp(selected + mask, 0.0, 1.0); }
        default: { result = clamp(selected - mask, 0.0, 1.0); }
    }
    textureStore(output_map, vec2<i32>(id.xy), vec4<f32>(result, 0.0, 0.0, 1.0));
}
