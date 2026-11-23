struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

fn height(p: vec2<i32>) -> f32 {
    let limit = vec2<i32>(node.dimensions.xy) - vec2<i32>(1);
    return textureLoad(input_map, clamp(p, vec2<i32>(0), limit), 0).r;
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let du = (height(p + vec2<i32>(1, 0)) - height(p - vec2<i32>(1, 0))) * f32(node.dimensions.x) * 0.5;
    let dv = (height(p + vec2<i32>(0, 1)) - height(p - vec2<i32>(0, 1))) * f32(node.dimensions.y) * 0.5;
    // Rows increase downward; tangent +Y points upward (OpenGL convention).
    let normal = normalize(vec3<f32>(-node.parameters.x * du, node.parameters.x * dv, 1.0));
    textureStore(output_map, p, vec4<f32>(normal, 1.0));
}
