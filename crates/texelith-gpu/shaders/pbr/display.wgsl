@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var display: texture_storage_2d<rgba8unorm, write>;
// Original fixed rational fit. x is bounded by finite Rgba16Float storage.
fn tone_map(x: vec3<f32>) -> vec3<f32> {
    return clamp(x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14),vec3(0.0),vec3(1.0));
}
fn encode_srgb(linear: vec3<f32>) -> vec3<f32> {
    let v=clamp(linear,vec3(0.0),vec3(1.0));
    return select(1.055*pow(v,vec3(1.0/2.4))-0.055,12.92*v,v<=vec3(0.0031308));
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=textureDimensions(display)) {return;}
    let sample=textureLoad(hdr,vec2<i32>(id.xy),0);
    // Background is a UI color and bypasses the material tone curve/exposure.
    let linear=select(sample.rgb,tone_map(sample.rgb),sample.a>0.5);
    textureStore(display,vec2<i32>(id.xy),vec4(encode_srgb(linear),1.0));
}
