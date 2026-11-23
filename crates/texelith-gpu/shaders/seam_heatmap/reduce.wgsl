struct Uniforms {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> u: Uniforms;
var<workgroup> boundary: array<f32, 64>;
var<workgroup> interior: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let width = u.dimensions.z / 2u;
    let count = width * u.dimensions.w;
    let start = group.x * 4096u;
    var b = 0.0;
    var i = 0.0;
    for (var index = start + lane; index < min(start + 4096u, count); index += 64u) {
        let p = vec2<i32>(i32((index % width) * 2u), i32(index / width));
        b += textureLoad(source, p, 0).x;
        i += textureLoad(source, p + vec2(1, 0), 0).x;
    }
    boundary[lane] = b;
    interior[lane] = i;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step /= 2u) {
        if lane < step {
            boundary[lane] += boundary[lane + step];
            interior[lane] += interior[lane + step];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        if u.flags.x == 0u {
            textureStore(output, vec2<i32>(i32(group.x * 2u), 0), vec4(boundary[0], 0.0, 0.0, 0.0));
            textureStore(output, vec2<i32>(i32(group.x * 2u + 1u), 0), vec4(interior[0], 0.0, 0.0, 0.0));
        } else {
            let b_rms = sqrt(boundary[0]);
            let i_rms = sqrt(interior[0]);
            textureStore(output, vec2(0, 0), vec4(b_rms, 0.0, 0.0, 0.0));
            textureStore(output, vec2(1, 0), vec4(i_rms, 0.0, 0.0, 0.0));
            textureStore(output, vec2(2, 0), vec4(b_rms / max(i_rms, 1e-8), 0.0, 0.0, 0.0));
        }
    }
}
