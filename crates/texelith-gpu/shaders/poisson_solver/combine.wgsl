struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var correction: texture_2d<f32>;
fn sample_correction(p: vec2<i32>) -> f32 {
    let n = vec2<i32>(textureDimensions(correction));
    var q = clamp(p, vec2<i32>(0), n - vec2<i32>(1));
    if node.flags.y != 0u { q = ((p % n) + n) % n; }
    return textureLoad(correction, q, 0).x;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    var value = 0.0;
    if node.flags.x != 0u { value = textureLoad(input, p, 0).x; }
    if node.flags.x == 2u {
        let fine = vec2<i32>(node.dimensions.xy);
        let coarse = vec2<i32>(textureDimensions(correction));
        let numerator = (2 * p + vec2<i32>(1)) * coarse - fine;
        let denominator = 2 * fine;
        let base = vec2<i32>(floor(vec2<f32>(numerator) / vec2<f32>(denominator)));
        let f = vec2<f32>(numerator - base * denominator) / vec2<f32>(denominator);
        value += mix(mix(sample_correction(base), sample_correction(base + vec2<i32>(1, 0)), f.x),
                     mix(sample_correction(base + vec2<i32>(0, 1)), sample_correction(base + vec2<i32>(1, 1)), f.x), f.y);
    }
    textureStore(output, p, vec4<f32>(value, 0.0, 0.0, 0.0));
}
