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
    let height = textureLoad(input_map, p, 0).r;
    let adjusted = clamp((height - node.parameters.x) / (node.parameters.y - node.parameters.x), 0.0, 1.0);
    var result = adjusted;
    // Preserve clamp endpoints exactly, including zero for fractional exponents.
    if adjusted > 0.0 && adjusted < 1.0 {
        result = pow(adjusted, 1.0 / node.parameters.z);
    }
    textureStore(output_map, p, vec4<f32>(result, 0.0, 0.0, 1.0));
}
