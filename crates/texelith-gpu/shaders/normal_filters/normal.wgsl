struct Uniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
struct Weights { slots: array<vec4<f32>,4> }
@group(0) @binding(0) var fine: texture_2d<f32>;
@group(0) @binding(1) var result: texture_storage_2d<rgba16float,write>;
@group(0) @binding(2) var<uniform> u: Uniform;
@group(1) @binding(0) var level1: texture_2d<f32>;
@group(1) @binding(1) var level2: texture_2d<f32>;
@group(1) @binding(2) var level3: texture_2d<f32>;
@group(1) @binding(3) var level4: texture_2d<f32>;
@group(1) @binding(4) var level5: texture_2d<f32>;
@group(1) @binding(5) var level6: texture_2d<f32>;
@group(1) @binding(6) var level7: texture_2d<f32>;
@group(1) @binding(7) var level8: texture_2d<f32>;
@group(1) @binding(8) var level9: texture_2d<f32>;
@group(1) @binding(9) var level10: texture_2d<f32>;
@group(1) @binding(10) var level11: texture_2d<f32>;
@group(1) @binding(11) var level12: texture_2d<f32>;
@group(1) @binding(12) var<uniform> weights: Weights;
fn size(level: u32) -> vec2<i32> {
    let divisor = 1u << level;
    return vec2<i32>((u.dimensions.xy + vec2<u32>(divisor-1u))/divisor);
}
fn sample_height(level: u32, pos: vec2<i32>) -> f32 {
    let n = size(level);
    var p = clamp(pos,vec2<i32>(0),n-vec2<i32>(1));
    if u.flags.y != 0u { p = ((pos % n)+n)%n; }
    switch level {
        case 0u: { return textureLoad(fine,p,0).x; }
        case 1u: { return textureLoad(level1,p,0).x; }
        case 2u: { return textureLoad(level2,p,0).x; }
        case 3u: { return textureLoad(level3,p,0).x; }
        case 4u: { return textureLoad(level4,p,0).x; }
        case 5u: { return textureLoad(level5,p,0).x; }
        case 6u: { return textureLoad(level6,p,0).x; }
        case 7u: { return textureLoad(level7,p,0).x; }
        case 8u: { return textureLoad(level8,p,0).x; }
        case 9u: { return textureLoad(level9,p,0).x; }
        case 10u: { return textureLoad(level10,p,0).x; }
        case 11u: { return textureLoad(level11,p,0).x; }
        case 12u: { return textureLoad(level12,p,0).x; }
        default: { return 0.0; }
    }
}
fn gradient(level: u32, p: vec2<i32>) -> vec2<f32> {
    var d = vec2<f32>(0.0);
    var smoothing = vec3<f32>(0.0,1.0,0.0);
    var divisor = 2.0;
    if u.flags.x == 1u { smoothing = vec3<f32>(1.0,2.0,1.0); divisor = 8.0; }
    if u.flags.x == 2u { smoothing = vec3<f32>(3.0,10.0,3.0); divisor = 32.0; }
    for (var t = -1; t <= 1; t++) {
        d.x += smoothing[u32(t+1)] * (sample_height(level,p+vec2<i32>(1,t))-sample_height(level,p+vec2<i32>(-1,t)));
        d.y += smoothing[u32(t+1)] * (sample_height(level,p+vec2<i32>(t,1))-sample_height(level,p+vec2<i32>(t,-1)));
    }
    return d * vec2<f32>(size(level)) / divisor;
}
fn lifted_gradient(level: u32, pixel: vec2<u32>) -> vec2<f32> {
    if level == 0u { return gradient(level,vec2<i32>(pixel)); }
    let p = (vec2<f32>(pixel)+vec2<f32>(0.5))*vec2<f32>(size(level))/vec2<f32>(u.dimensions.xy)-vec2<f32>(0.5);
    let a = vec2<i32>(floor(p));
    let t = fract(p);
    // Interpolate gradients, never already normalized normals. R32 height
    // sampling preserves adjacent 16-bit input codes and signed height values.
    return mix(mix(gradient(level,a),gradient(level,a+vec2<i32>(1,0)),t.x),
        mix(gradient(level,a+vec2<i32>(0,1)),gradient(level,a+vec2<i32>(1,1)),t.x),t.y);
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.xy) { return; }
    var d = vec2<f32>(0.0);
    for (var l = 0u; l < u.flags.w; l++) {
        let weight = weights.slots[l/4u][l%4u];
        if weight != 0.0 { d += weight * lifted_gradient(l,id.xy); }
    }
    var normal = normalize(vec3<f32>(-u.parameters.x*d.x,u.parameters.x*d.y,1.0));
    if u.flags.z != 0u { normal.y = -normal.y; }
    textureStore(result,vec2<i32>(id.xy),vec4<f32>(normal,1.0));
}
