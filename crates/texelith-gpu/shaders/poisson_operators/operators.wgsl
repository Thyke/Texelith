struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
struct GridUniform { size: vec4<u32> }
@group(0) @binding(0) var input0: texture_2d<f32>;
@group(1) @binding(0) var input1: texture_2d<f32>;
@group(1) @binding(1) var input2: texture_2d<f32>;
@group(1) @binding(2) var input3: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(3) var<uniform> grid: GridUniform;
@group(1) @binding(4) var<storage, read> stencil: array<vec4<f32>>;

fn index(p: vec2<i32>, size: vec2<u32>) -> vec2<i32> {
    let n = vec2<i32>(size);
    if grid.size.z != 0u { return ((p % n) + n) % n; }
    return clamp(p, vec2<i32>(0), n - vec2<i32>(1));
}
fn load(p: vec2<i32>) -> f32 {
    return textureLoad(input0, index(p, textureDimensions(input0)), 0).x;
}
fn laplacian(p: vec2<i32>) -> vec2<f32> {
    let n = vec2<i32>(grid.size.xy);
    let center = load(p);
    var a = 0.0;
    var diagonal = 0.0;
    for (var axis = 0u; axis < 2u; axis++) {
        let offset = select(vec2<i32>(1, 0), vec2<i32>(0, 1), axis == 1u);
        if n[axis] > 1 {
            let scale = f32(n[axis]) * f32(n[axis]);
            if grid.size.z != 0u || p[axis] > 0 {
                a += scale * (center - load(p - offset)); diagonal += scale;
            }
            if grid.size.z != 0u || p[axis] + 1 < n[axis] {
                a += scale * (center - load(p + offset)); diagonal += scale;
            }
        }
    }
    return vec2<f32>(a, diagonal);
}
fn transpose(p: vec2<i32>) -> f32 {
    let n = vec2<i32>(grid.size.xy);
    var value = 0.0;
    if n.x > 1 {
        var v = 0.0;
        if grid.size.z != 0u || p.x > 0 { v += textureLoad(input0, index(p - vec2<i32>(1, 0), grid.size.xy), 0).x; }
        if grid.size.z != 0u || p.x + 1 < n.x { v -= textureLoad(input0, p, 0).x; }
        value += f32(n.x) * v;
    }
    if n.y > 1 {
        var v = 0.0;
        if grid.size.z != 0u || p.y > 0 { v += textureLoad(input1, index(p - vec2<i32>(0, 1), grid.size.xy), 0).x; }
        if grid.size.z != 0u || p.y + 1 < n.y { v -= textureLoad(input1, p, 0).x; }
        value += f32(n.y) * v;
    }
    return value;
}
fn restricted(p: vec2<u32>) -> f32 {
    let fine = textureDimensions(input0);
    let coarse = textureDimensions(output);
    let lo = p * fine;
    let hi = (p + vec2<u32>(1)) * fine;
    let first = lo / coarse;
    let end = (hi + coarse - vec2<u32>(1)) / coarse;
    var value = 0.0;
    for (var y = first.y; y < end.y; y++) {
        for (var x = first.x; x < end.x; x++) {
            let cell = vec2<u32>(x, y);
            let overlap = min(hi, (cell + vec2<u32>(1)) * coarse) - max(lo, cell * coarse);
            let weight = vec2<f32>(overlap) / vec2<f32>(fine);
            value += textureLoad(input0, vec2<i32>(cell), 0).x * weight.x * weight.y;
        }
    }
    return value;
}
fn prolonged(p: vec2<i32>) -> f32 {
    let fine = vec2<i32>(textureDimensions(output));
    let coarse = vec2<i32>(textureDimensions(input0));
    // Exact integer numerator, then divide just the fractional remainder. This
    // avoids losing interpolation precision at large NPOT pixel coordinates.
    let numerator = (2 * p + vec2<i32>(1)) * coarse - fine;
    let denominator = 2 * fine;
    let base = vec2<i32>(floor(vec2<f32>(numerator) / vec2<f32>(denominator)));
    let frac = vec2<f32>(numerator - base * denominator) / vec2<f32>(denominator);
    let a = mix(load(base), load(base + vec2<i32>(1, 0)), frac.x);
    let b = mix(load(base + vec2<i32>(0, 1)), load(base + vec2<i32>(1, 1)), frac.x);
    return mix(a, b, frac.y);
}
fn horizontal(p: vec2<i32>, mass: bool) -> f32 {
    let row = stencil[u32(p.x)];
    let center = textureLoad(input0, p, 0).x;
    var value = select(0.0, center, mass);
    for (var i = 0u; i < u32(row.y); i++) {
        let entry = stencil[u32(row.x) + i];
        value += select(entry.y, entry.z, mass) * (textureLoad(input0, vec2<i32>(i32(entry.x), p.y), 0).x - center);
    }
    return value;
}
fn coarse_apply(p: vec2<i32>) -> vec2<f32> {
    let row = stencil[grid.size.x + u32(p.y)];
    let ax = textureLoad(input0, p, 0).x;
    let mx = textureLoad(input1, p, 0).x;
    var result = ax;
    for (var i = 0u; i < u32(row.y); i++) {
        let entry = stencil[u32(row.x) + i];
        let q = vec2<i32>(p.x, i32(entry.x));
        result += entry.z * (textureLoad(input0, q, 0).x - ax)
                + entry.y * (textureLoad(input1, q, 0).x - mx);
    }
    let xrow = stencil[u32(p.x)];
    return vec2<f32>(result, xrow.z * row.w + xrow.w * row.z);
}
@compute @workgroup_size(8, 8)
fn point(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let p = vec2<i32>(id.xy);
    let op = node.flags.x;
    var value = 0.0;
    if op == 0u {
        let n = textureLoad(input0, p, 0).xyz;
        let slope = vec2<f32>(-n.x, select(n.y, -n.y, node.flags.z != 0u)) / max(n.z, 0.05);
        value = clamp(slope[node.flags.y], -20.0, 20.0);
    } else if op == 1u {
        let axis = node.flags.y;
        let n = grid.size[axis];
        if n > 1u && (grid.size.z != 0u || id[axis] + 1u < n) {
            let offset = select(vec2<i32>(1, 0), vec2<i32>(0, 1), axis == 1u);
            value = f32(n) * (load(p + offset) - load(p));
        }
    } else if op == 2u { value = transpose(p); }
    else if op == 3u || op == 5u || op == 8u {
        let a = laplacian(p); value = a.x;
        if op == 8u { value = textureLoad(input1, p, 0).x - a.x; }
        if op == 5u {
            value = load(p);
            if a.y != 0.0 { value += node.parameters.x * (textureLoad(input1, p, 0).x - a.x) / a.y; }
        }
    } else if op == 4u { value = load(p) - textureLoad(input1, vec2<i32>(0), 0).x; }
    else if op == 6u { value = restricted(id.xy); }
    else if op == 7u { value = prolonged(p); }
    else if op == 13u { value = horizontal(p, false); }
    else if op == 14u { value = horizontal(p, true); }
    else if op == 15u {
        let a = coarse_apply(p); value = a.x;
        if node.flags.w == 8u { value = textureLoad(input3, p, 0).x - a.x; }
        if node.flags.w == 5u {
            value = textureLoad(input2, p, 0).x;
            if a.y != 0.0 { value += node.parameters.x * (textureLoad(input3, p, 0).x - a.x) / a.y; }
        }
    }

    textureStore(output, p, vec4<f32>(value, 0.0, 0.0, 0.0));
}

var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(8, 8)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0.0;
    if all(id.xy < node.dimensions.zw) {
        let n = textureLoad(input0, vec2<i32>(id.xy), 0);
        value = n.x;
        if node.flags.x == 10u { value *= value; }
        if node.flags.x == 11u {
            let slope = abs(n.xy / max(n.z, 0.05));
            value = select(0.0, 1.0, n.z < 0.05 || any(slope > vec2<f32>(20.0)));
        }
    }
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        value = sums[0];
        if node.parameters.y != 0.0 {
            if node.flags.w == 9u { value /= node.parameters.z; }
            if node.flags.w == 10u { value = sqrt(max(value, 0.0)); }
        }
        textureStore(output, vec2<i32>(group.xy), vec4<f32>(value, 0.0, 0.0, 0.0));
    }
}
