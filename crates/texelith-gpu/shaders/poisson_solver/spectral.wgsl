struct NodeUniform { dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32> }
@group(0) @binding(0) var normal: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(1) @binding(1) var height: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> partial: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> reduced: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> twiddles: array<vec2<f32>>;

fn address(p: vec2<i32>) -> vec2<i32> {
    let n = vec2<i32>(node.dimensions.xy);
    return (p + n) % n;
}
fn slopes(p: vec2<i32>) -> vec2<f32> {
    let n = textureLoad(normal, address(p), 0).xyz;
    return clamp(vec2<f32>(-n.x, select(n.y, -n.y, node.reserved.x != 0u)) / max(n.z, 0.05), vec2<f32>(-20.0), vec2<f32>(20.0));
}
fn rhs(p: vec2<i32>) -> f32 {
    let g = slopes(p);
    let x = (slopes(p - vec2<i32>(1, 0)).x - g.x) * f32(node.dimensions.x);
    let y = (slopes(p - vec2<i32>(0, 1)).y - g.y) * f32(node.dimensions.y);
    return x + y;
}
fn load_height(p: vec2<i32>) -> f32 { return textureLoad(height, address(p), 0).x; }
fn laplacian(p: vec2<i32>) -> f32 {
    let h = load_height(p);
    let x = ((h - load_height(p - vec2<i32>(1, 0))) + (h - load_height(p + vec2<i32>(1, 0)))) * f32(node.dimensions.x * node.dimensions.x);
    let y = ((h - load_height(p - vec2<i32>(0, 1))) + (h - load_height(p + vec2<i32>(0, 1)))) * f32(node.dimensions.y * node.dimensions.y);
    return x + y;
}
fn linear(p: vec2<u32>) -> u32 { return p.y * node.dimensions.x + p.x; }
@compute @workgroup_size(8, 8)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    var value = rhs(vec2<i32>(id.xy));
    if node.flags.w != 0u { value -= laplacian(vec2<i32>(id.xy)); }
    data[linear(id.xy)] = vec2<f32>(value, 0.0);
}

fn multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
fn twiddle(k: u32, n: u32, inverse: bool) -> vec2<f32> {
    let value = twiddles[k * (4096u/n)];
    return vec2<f32>(value.x, select(value.y, -value.y, inverse));
}
fn location(k: u32, line: u32) -> u32 {
    return select(line * node.dimensions.x + k, k * node.dimensions.x + line, node.flags.x != 0u);
}
// A 4096-point transform starts with one in-place DIF butterfly, followed by
// two independent 2048-point FFTs. Frequencies remain even/odd permuted until
// the inverse undoes the same factorization. No inter-workgroup dependency.
@compute @workgroup_size(8, 8)
fn split(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let k = group.x * 64u + lane;
    let n = node.flags.z;
    if k >= n / 2u { return; }
    let a = data[location(k, group.y)];
    var b = data[location(k + n/2u, group.y)];
    if node.flags.y != 0u {
        b = multiply(b, twiddle(k, n, true));
        data[location(k, group.y)] = (a + b) * 0.5;
        data[location(k+n/2u, group.y)] = (a - b) * 0.5;
    } else {
        data[location(k, group.y)] = a + b;
        data[location(k+n/2u, group.y)] = multiply(a - b, twiddle(k, n, false));
    }
}
var<workgroup> values: array<vec2<f32>, 2048>;
@compute @workgroup_size(8, 8)
fn fft(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let n = min(node.flags.z, 2048u);
    let bits = firstLeadingBit(n);
    let base = group.y * n;
    for (var k = lane; k < n; k += 64u) {
        var reversed = 0u;
        if n > 1u { reversed = reverseBits(k) >> (32u - bits); }
        values[reversed] = data[location(base + k, group.x)];
    }
    workgroupBarrier();
    for (var size = 2u; size <= n; size *= 2u) {
        for (var k = lane; k < n/2u; k += 64u) {
            let offset = k % (size/2u);
            let i = (k / (size/2u)) * size + offset;
            let a = values[i];
            let b = multiply(values[i+size/2u], twiddle(offset, size, node.flags.y != 0u));
            values[i] = a + b;
            values[i+size/2u] = a - b;
        }
        workgroupBarrier();
    }
    for (var k = lane; k < n; k += 64u) {
        data[location(base + k, group.x)] = values[k] / select(1.0, f32(n), node.flags.y != 0u);
    }
}
fn frequency(k: u32, n: u32) -> u32 {
    if n > 2048u { return (k % 2048u) * 2u + k / 2048u; }
    return k;
}
var<workgroup> tile_a: array<vec2<f32>, 64>;
var<workgroup> tile_b: array<vec2<f32>, 64>;
// Only square grids use this in-place tiled permutation. Both tiles belong to
// one workgroup; every pair is visited once. Column FFTs then use contiguous rows.
@compute @workgroup_size(8, 8)
fn transpose(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    if group.x < group.y { return; }
    let p = group.xy * 8u + local.xy;
    let q = group.yx * 8u + local.xy;
    let i = local.y * 8u + local.x;
    let j = local.x * 8u + local.y;
    if all(p < node.dimensions.xy) { tile_a[i] = data[linear(p)]; }
    if all(q < node.dimensions.xy) { tile_b[i] = data[linear(q)]; }
    workgroupBarrier();
    if all(p < node.dimensions.xy) { data[linear(p)] = tile_b[j]; }
    if group.x != group.y && all(q < node.dimensions.xy) { data[linear(q)] = tile_a[j]; }
}
@compute @workgroup_size(8, 8)
fn divide(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let k = vec2<u32>(frequency(id.x, node.dimensions.x), frequency(id.y, node.dimensions.y));
    let n = vec2<f32>(node.dimensions.xy);
    // Fold to the first half to avoid cancellation near 2*pi. The sin-squared
    // form is stable at low frequencies unlike 2-2*cos(theta).
    let folded = min(k, node.dimensions.xy-k);
    let s = sin(3.141592653589793 * vec2<f32>(folded) / n) * n;
    let eigenvalue = 4.0 * dot(s,s);
    var value = vec2<f32>(0.0);
    if any(k != vec2<u32>(0)) { value = data[linear(id.xy)] / eigenvalue; }
    data[linear(id.xy)] = value;
}
@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    data[linear(id.xy)].x += textureLoad(height, vec2<i32>(id.xy), 0).x;
}
@compute @workgroup_size(8, 8)
fn write_height(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(data[linear(id.xy)].x, 0.0, 0.0, 0.0));
}
var<workgroup> sums: array<vec4<f32>, 64>;
fn sum(lane: u32) {
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] += sums[lane+stride]; }
        workgroupBarrier();
    }
}
@compute @workgroup_size(8, 8)
fn diagnose(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var value = vec4<f32>(0.0);
    if all(id.xy < node.dimensions.xy) {
        let p = vec2<i32>(id.xy);
        let b = rhs(p);
        let r = b - laplacian(p);
        let n = textureLoad(normal, p, 0).xyz;
        let clamped = n.z < 0.05 || any(abs(n.xy / max(n.z, 0.05)) > vec2<f32>(20.0));
        let g = slopes(p);
        value = vec4<f32>(b*b, r*r, select(0.0, 1.0, clamped), dot(g,g));
    }
    sums[lane] = value;
    sum(lane);
    if lane == 0u { partial[group.y * ((node.dimensions.x+7u)/8u) + group.x] = sums[0]; }
}
@compute @workgroup_size(8, 8)
fn reduce(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let i = group.x*64u + lane;
    var value = vec4<f32>(0.0);
    if i < node.reserved.y { value = partial[i]; }
    sums[lane] = value;
    sum(lane);
    if lane == 0u { reduced[group.x] = sums[0]; }
}

// Parseval + the normal equations give F(h)-F(h*) = r^T A^+ r.
// The forward transform holds r=b-Ah, unnormalized. DC is the nullspace.
@compute @workgroup_size(8, 8)
fn certificate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    var energy = 0.0;
    if all(id.xy < node.dimensions.xy) {
        let k = vec2<u32>(frequency(id.x, node.dimensions.x), frequency(id.y, node.dimensions.y));
        let n = vec2<f32>(node.dimensions.xy);
        let folded = min(k, node.dimensions.xy-k);
        let s = sin(3.141592653589793 * vec2<f32>(folded) / n) * n;
        if any(k != vec2<u32>(0)) {
            let r = data[linear(id.xy)];
            energy = dot(r,r) / (4.0 * dot(s,s) * n.x * n.y);
        }
    }
    sums[lane] = vec4<f32>(energy, 0.0, 0.0, 0.0);
    sum(lane);
    if lane == 0u { partial[group.y * ((node.dimensions.x+7u)/8u) + group.x] = sums[0]; }
}
