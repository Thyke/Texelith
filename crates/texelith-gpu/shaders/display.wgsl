struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let value = textureLoad(input_map, p, 0);
    var codes = vec3<f32>(value.r);
    if node.flags.x == 1u { codes = value.xyz * 0.5 + vec3<f32>(0.5); }
    // Diagnostic data codes: no sRGB transfer, alpha is always opaque.
    textureStore(output_map, p, vec4<f32>(clamp(codes, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
