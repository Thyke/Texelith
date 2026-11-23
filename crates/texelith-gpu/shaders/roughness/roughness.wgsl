struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
};
struct Curve { knots: array<vec4<f32>, 16> };
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var mask_map: texture_2d<f32>;
@group(1) @binding(1) var<uniform> curve: Curve;

fn remap(value: f32) -> f32 {
    if value <= curve.knots[0].x { return curve.knots[0].y; }
    for (var i = 1u; i < node.flags.x; i++) {
        let a = curve.knots[i - 1u].xy;
        let b = curve.knots[i].xy;
        if value <= b.x {
            let t = clamp((value - a.x) / (b.x - a.x), 0.0, 1.0);
            return a.y * (1.0 - t) + b.y * t;
        }
    }
    return curve.knots[node.flags.x - 1u].y;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    // Area restriction keeps proxies in linear color and covers odd extents.
    let scale = vec2<f32>(node.dimensions.zw) / vec2<f32>(node.dimensions.xy);
    let lo = vec2<f32>(id.xy) * scale;
    let hi = vec2<f32>(id.xy + vec2<u32>(1u)) * scale;
    var rgb = textureLoad(color, vec2<i32>(id.xy), 0).rgb;
    if any(node.dimensions.xy != node.dimensions.zw) {
        // Accumulate deviations so constant colors survive division exactly.
        let anchor = textureLoad(color, vec2<i32>(lo), 0).rgb;
        rgb = vec3<f32>(0.0);
        var total_weight = 0.0;
        for (var y = u32(floor(lo.y)); y < min(u32(ceil(hi.y)), node.dimensions.w); y++) {
            for (var x = u32(floor(lo.x)); x < min(u32(ceil(hi.x)), node.dimensions.z); x++) {
                let weight = max(vec2<f32>(0.0), min(hi, vec2<f32>(f32(x + 1u), f32(y + 1u))) - max(lo, vec2<f32>(f32(x), f32(y))));
                let area = weight.x * weight.y;
                rgb += (textureLoad(color, vec2<i32>(i32(x), i32(y)), 0).rgb - anchor) * area;
                total_weight += area;
            }
        }
        // Rounded NPOT boundaries must use their actual covered area.
        rgb = anchor + rgb / total_weight;
    }
    let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    var mask = 1.0;
    if node.flags.y != 0u { mask = clamp(textureLoad(mask_map, vec2<i32>(id.xy), 0).r, 0.0, 1.0); }
    let mapped = remap(luminance);
    var result = clamp(node.parameters.x * (1.0 - mask) + mapped * mask, 0.0, 1.0);
    if node.flags.z != 0u { result = 1.0 - result; }
    textureStore(output_map, vec2<i32>(id.xy), vec4<f32>(result, 0.0, 0.0, 1.0));
}
