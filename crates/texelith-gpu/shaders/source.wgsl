struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_map: texture_2d<f32>;
@group(0) @binding(1) var output_map: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

fn linear(code: vec3<f32>) -> vec3<f32> {
    return select(pow((code + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
                  code / 12.92, code <= vec3<f32>(0.04045));
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let rgba = textureLoad(input_map, p, 0);
    var height = rgba.r;
    if node.flags.x == 0u {
        height = dot(linear(rgba.rgb), vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    textureStore(output_map, p, vec4<f32>(height, 0.0, 0.0, 1.0));
}
