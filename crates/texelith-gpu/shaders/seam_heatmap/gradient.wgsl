struct Uniforms {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var heatmap: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> u: Uniforms;
@group(1) @binding(0) var partials: texture_storage_2d<r32float, write>;
var<workgroup> boundary: array<f32, 64>;
var<workgroup> interior: array<f32, 64>;

fn energy(a: vec4<f32>, b: vec4<f32>) -> f32 {
    // Saturate extreme height differences before squaring to keep the f32
    // reduction finite. Ordinary material data is unaffected.
    let d = clamp(b.rgb * 0.5 - a.rgb * 0.5, vec3(-5e17), vec3(5e17)) * 2.0;
    if u.flags.x == 0u { return d.x * d.x; }
    return dot(d, d) / 3.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let size = u.dimensions.xy;
    var b = 0.0;
    var i = 0.0;
    if all(p.xy < size) {
        let a = textureLoad(source, vec2<i32>(p.xy), 0);
        var total = 0.0;
        for (var axis = 0u; axis < 2u; axis++) {
            let n = size[axis];
            if n <= 1u { continue; }
            var next = p.xy;
            next[axis] = (next[axis] + 1u) % n;
            let e = energy(a, textureLoad(source, vec2<i32>(next), 0));
            total += e;
            let band = min(4u, n / 2u);
            let middle = n / 2u - band;
            let c = p[axis];
            if c < band || c >= n - band { b += e * u.parameters.x; }
            if c >= middle && c < middle + 2u * band { i += e * u.parameters.x; }
        }
        textureStore(heatmap, vec2<i32>(p.xy), vec4(sqrt(total), 0.0, 0.0, 0.0));
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
        textureStore(partials, vec2<i32>(i32(group.x * 2u), i32(group.y)), vec4(boundary[0], 0.0, 0.0, 0.0));
        textureStore(partials, vec2<i32>(i32(group.x * 2u + 1u), i32(group.y)), vec4(interior[0], 0.0, 0.0, 0.0));
    }
}
