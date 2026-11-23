struct Uniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
@group(0) @binding(0) var normal: texture_2d<f32>;
@group(0) @binding(1) var result: texture_storage_2d<r32float,write>;
@group(0) @binding(2) var<uniform> u: Uniform;

fn canonical(pos: vec2<i32>) -> vec3<f32> {
    let n = vec2<i32>(u.dimensions.xy);
    var p = clamp(pos, vec2<i32>(0), n - vec2<i32>(1));
    if u.flags.x != 0u { p = ((pos % n) + n) % n; }
    var v = normalize(textureLoad(normal, p, 0).xyz);
    if u.flags.y != 0u { v.y = -v.y; }
    return v;
}
fn scaled(value: f32, scale: f32) -> f32 {
    if value <= 0.0 || scale == 0.0 { return 0.0; }
    // The division is bounded below by 1/8192 for normalized tangent fields.
    // Test saturation before multiplying so finite maximum scales cannot overflow.
    if scale >= 1.0 / value { return 1.0; }
    return min(value * scale, 1.0);
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= u.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    var du = 0.0;
    var dv = 0.0;
    if u.dimensions.x > 1u {
        let right = canonical(p + vec2<i32>(1,0)).x;
        let left = canonical(p - vec2<i32>(1,0)).x;
        // Preserve exact cancellation before the backend can contract scaled
        // subtraction into an FMA with a separately rounded operand.
        if right != left { du = (right - left) * f32(u.dimensions.x) * 0.5; }
    }
    if u.dimensions.y > 1u {
        let down = canonical(p + vec2<i32>(0,1)).y;
        let up = canonical(p - vec2<i32>(0,1)).y;
        if down != up { dv = (down - up) * f32(u.dimensions.y) * 0.5; }
    }
    // Image rows run down; canonical tangent +Y runs up.
    let c = du - dv;
    var value = c;
    if u.flags.z == 1u { value = scaled(c, u.parameters.x); }
    if u.flags.z == 2u { value = scaled(-c, u.parameters.y); }
    textureStore(result, p, vec4<f32>(value, 0.0, 0.0, 1.0));
}
