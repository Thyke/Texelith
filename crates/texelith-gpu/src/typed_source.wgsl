struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: NodeUniform;
fn linear(v: vec3<f32>) -> vec3<f32> {
    return select(pow((v + 0.055) / 1.055, vec3<f32>(2.4)), v / 12.92, v <= vec3<f32>(0.04045));
}
@compute @workgroup_size(8,8)
fn color(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.xy) { return; }
    let rgb = linear(textureLoad(input, vec2<i32>(id.xy), 0).rgb);
    // Storage conversion can truncate on DX12. Round explicitly to binary16
    // spacing first so the stored value is exact on every backend.
    let spacing = exp2(max(floor(log2(max(rgb, vec3<f32>(1e-30)))) - 10.0, vec3<f32>(-24.0)));
    let rounded = floor(rgb / spacing + 0.5) * spacing;
    textureStore(output, id.xy, vec4<f32>(rounded, 1.0));
}
@compute @workgroup_size(8,8)
fn normal(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.xy) { return; }
    var n = textureLoad(input, vec2<i32>(id.xy), 0).rgb * 2.0 - 1.0;
    // Preserve the source convention carried by the graph edge. Its consumer
    // canonicalizes to GL once; import must not silently flip that metadata.
    n = normalize(n);
    textureStore(output, id.xy, vec4<f32>(n, 1.0));
}
