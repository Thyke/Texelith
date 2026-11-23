struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let size = node.dimensions.xy;
    let source = node.dimensions.zw;
    // Integer overlap numerators avoid subtracting imprecise absolute f32
    // endpoints on NPOT grids. Products fit u32 at the 4096 source limit.
    let low = id.xy * source;
    let high = (id.xy + vec2<u32>(1u)) * source;
    let first = low / size;
    let end = (high + size - vec2<u32>(1u)) / size;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var y = first.y; y < end.y; y += 1u) {
        let wy = min(high.y, (y + 1u) * size.y) - max(low.y, y * size.y);
        for (var x = first.x; x < end.x; x += 1u) {
            let wx = min(high.x, (x + 1u) * size.x) - max(low.x, x * size.x);
            // Decode integer RGB8 first so complementary GL/DX bytes produce
            // exactly opposite Y values before half rounding.
            let rgb = round(textureLoad(input_map, vec2<i32>(i32(x), i32(y)), 0).rgb * 255.0);
            let n = normalize(rgb * 2.0 - 255.0);
            let w = f32(wx * wy);
            sum += w * n;
            weight += w;
        }
    }
    let average = sum / weight;
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if dot(average, average) > 1e-12 { n = normalize(average); }
    // Keep the encoded convention; downstream consumers apply the Y flip once.
    // Explicit half rounding avoids backend-dependent storage truncation.
    let spacing = exp2(max(floor(log2(max(abs(n), vec3<f32>(1e-30)))) - 10.0, vec3<f32>(-24.0)));
    n = sign(n) * floor(abs(n) / spacing + 0.5) * spacing;
    textureStore(output_map, id.xy, vec4<f32>(n, 1.0));
}
