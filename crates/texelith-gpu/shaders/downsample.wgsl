struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    // Exact rational pixel boundaries in units of output pixels avoid fractional
    // boundary drift for NPOT inputs. All products fit u32 under the 4096 cap.
    let out_size = node.dimensions.xy;
    let source_size = node.dimensions.zw;
    let low = id.xy * source_size;
    let high = (id.xy + vec2<u32>(1u)) * source_size;
    let first = low / out_size;
    let end = (high + out_size - vec2<u32>(1u)) / out_size;
    var sum = 0.0;
    for (var y = first.y; y < end.y; y += 1u) {
        let wy = min(high.y, (y + 1u) * out_size.y) - max(low.y, y * out_size.y);
        for (var x = first.x; x < end.x; x += 1u) {
            let wx = min(high.x, (x + 1u) * out_size.x) - max(low.x, x * out_size.x);
            let coord = vec2<i32>(min(vec2<u32>(x, y), source_size - vec2<u32>(1u)));
            sum += textureLoad(input_map, coord, 0).r * f32(wx * wy);
        }
    }
    textureStore(output_map, p, vec4<f32>(sum / f32(source_size.x * source_size.y), 0.0, 0.0, 1.0));
}
