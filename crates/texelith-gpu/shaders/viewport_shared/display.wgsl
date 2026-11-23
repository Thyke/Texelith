@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var display: texture_storage_2d<rgba8unorm, write>;

// Diagnostic background/ramp bypass material tone mapping. This is the single
// linear-to-sRGB boundary; egui samples opaque gamma codes in Rgba8Unorm.
fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {
    let v = clamp(linear, vec3(0.0), vec3(1.0));
    return select(1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055, 12.92 * v, v <= vec3(0.0031308));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(display)) { return; }
    let color = textureLoad(hdr, vec2<i32>(id.xy), 0).rgb;
    textureStore(display, vec2<i32>(id.xy), vec4(encode_srgb(color), 1.0));
}
