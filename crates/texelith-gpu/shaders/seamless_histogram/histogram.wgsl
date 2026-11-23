struct NodeUniform {
    dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var result: texture_storage_2d<OUTPUT_FORMAT, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var<storage, read_write> partial: array<u32>;
@group(1) @binding(1) var<storage, read_write> moments: array<vec2<f32>>;
// [0,1024] CDF endpoints; 1025 variance; [1032,2055] exact counts as u32 bits.
@group(1) @binding(2) var<storage, read_write> mapping: array<f32>;
@group(1) @binding(3) var<storage, read> gaussian: array<f32>;
var<workgroup> counts: array<atomic<u32>, 1024>;
var<workgroup> sums: array<vec2<f32>, 64>;
fn luminance(v: vec4<f32>) -> f32 {
    if node.flags.w == 0u { return v.r; }
    return dot(v.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}
@compute @workgroup_size(8, 8)
fn histogram(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    for (var i = lane; i < 1024u; i += 64u) { atomicStore(&counts[i], 0u); }
    workgroupBarrier();
    let size = node.dimensions.z * node.dimensions.w;
    let anchor = luminance(textureLoad(source, vec2<i32>(0), 0));
    var sum = vec2<f32>(0.0);
    for (var offset = lane; offset < 4096u; offset += 64u) {
        let i = group.x * 4096u + offset;
        if i < size {
            let v = luminance(textureLoad(source, vec2<i32>(vec2<u32>(i % node.dimensions.z, i / node.dimensions.z)), 0));
            atomicAdd(&counts[min(u32(clamp(v, 0.0, 1.0) * 1024.0), 1023u)], 1u);
            let d = v - anchor;
            sum += vec2<f32>(d, d * d);
        }
    }
    sums[lane] = sum;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step /= 2u) {
        if lane < step { sums[lane] += sums[lane + step]; }
        workgroupBarrier();
    }
    if lane == 0u { moments[group.x] = sums[0]; }
    for (var i = lane; i < 1024u; i += 64u) { partial[group.x * 1024u + i] = atomicLoad(&counts[i]); }
}
@compute @workgroup_size(8, 8)
fn reduce(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let bin = group.x * 64u + lane;
    let groups = (node.dimensions.z * node.dimensions.w + 4095u) / 4096u;
    var total = 0u;
    for (var i = 0u; i < groups; i++) { total += partial[i * 1024u + bin]; }
    mapping[1032u + bin] = bitcast<f32>(total);
}
@compute @workgroup_size(8, 8)
fn cdf(@builtin(local_invocation_index) lane: u32) {
    if lane != 0u { return; }
    let size = node.dimensions.z * node.dimensions.w;
    var total = 0u;
    mapping[0] = 0.0;
    for (var i = 0u; i < 1024u; i++) {
        total += bitcast<u32>(mapping[1032u + i]);
        mapping[i + 1u] = f32(total) / f32(size);
    }
    var sum = vec2<f32>(0.0);
    for (var i = 0u; i < (size + 4095u) / 4096u; i++) { sum += moments[i]; }
    let mean = sum.x / f32(size);
    mapping[1025] = max(0.0, sum.y / f32(size) - mean * mean);
}
fn probability(v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0) * 1024.0;
    let i = min(u32(x), 1023u);
    return mix(mapping[i], mapping[i + 1u], x - f32(i));
}
fn to_gaussian(value: f32) -> f32 {
    let p = clamp(value, 0.000244140625, 0.999755859375);
    var lo = 1u;
    var hi = 8192u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if gaussian[mid] < p { lo = mid + 1u; } else { hi = mid; }
    }
    let t = (p - gaussian[lo - 1u]) / (gaussian[lo] - gaussian[lo - 1u]);
    return (f32(lo - 1u) + t) / 512.0 - 8.0;
}
fn inverse(z: f32) -> f32 {
    let x = clamp((z + 8.0) * 512.0, 0.0, 8192.0);
    let i = min(u32(x), 8191u);
    let p = clamp(mix(gaussian[i], gaussian[i + 1u], x - f32(i)), 0.000244140625, 0.999755859375);
    var lo = 1u;
    var hi = 1024u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if mapping[mid] < p { lo = mid + 1u; } else { hi = mid; }
    }
    let t = (p - mapping[lo - 1u]) / max(mapping[lo] - mapping[lo - 1u], 1e-8);
    return (f32(lo - 1u) + t) / 1024.0;
}
fn address(p: vec2<i32>) -> vec2<i32> {
    let n = vec2<i32>(node.dimensions.zw);
    if node.flags.x != 0u { return ((p % n) + n) % n; }
    return clamp(p, vec2<i32>(0), n - vec2<i32>(1));
}
fn bilinear(base: vec2<i32>, t: vec2<f32>) -> vec4<f32> {
    let a = textureLoad(source, address(base), 0);
    let b = textureLoad(source, address(base + vec2<i32>(1, 0)), 0);
    let c = textureLoad(source, address(base + vec2<i32>(0, 1)), 0);
    let d = textureLoad(source, address(base + vec2<i32>(1, 1)), 0);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
// Match the existing splat rational coordinate construction: retain subtexel
// precision for NPOT sources and quarter turns before conversion to float.
fn coordinate(translation: u32, numerator: i32, denominator: i32, size: i32) -> vec2<f32> {
    let product = numerator * size;
    let remainder = ((product % denominator) + denominator) % denominator;
    let integer = (product - remainder) / denominator;
    let translated = i32(translation) * size;
    let fraction = f32(remainder) / f32(denominator) + f32(translated % 32768) / 32768.0;
    let carry = i32(floor(fraction));
    let wrapped = (((integer + translated / 32768 + carry) % size) + size) % size;
    let centered = fraction - f32(carry) - 0.5;
    let base = floor(centered);
    return vec2<f32>(f32(wrapped) + base, centered - base);
}
fn window(d: vec2<f32>) -> f32 {
    let t = clamp(1.0 - abs(d) / (1.0 + node.parameters.x), vec2<f32>(0.0), vec2<f32>(1.0));
    let w = t * t * (3.0 - 2.0 * t);
    let w2 = w * w;
    let w4 = w2 * w2;
    return w4.x * w4.y;
}
@compute @workgroup_size(8, 8)
fn blend(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let n = vec2<f32>(node.dimensions.zw);
    let anchor = textureLoad(source, vec2<i32>(0), 0);
    var value = vec4<f32>(0.0);
    var total = 0.0;
    var latent = 0.0;
    var squares = 0.0;
    let reconstruct = node.parameters.y != 0.0 && mapping[1025] >= 1e-6;
    let splat = node.flags.z != 0u;
    let taps = select(4u, 9u, splat);
    for (var i = 0u; i < taps; i++) {
        var base: vec2<i32>;
        var t: vec2<f32>;
        var weight: f32;
        if !splat {
            let shift = vec2<u32>(select(0u, 16384u, (i & 1u) == 0u), select(0u, 16384u, (i & 2u) == 0u));
            let output = vec2<i32>(node.dimensions.xy);
            let twice_center = 2 * vec2<i32>(id.xy) + 1;
            let x = coordinate(shift.x, twice_center.x, 2 * output.x, i32(n.x));
            let y = coordinate(shift.y, twice_center.y, 2 * output.y, i32(n.y));
            let numerator = (twice_center + vec2<i32>(select(0, output.x, (i & 1u) == 0u), select(0, output.y, (i & 2u) == 0u))) % (2 * output);
            let uv = vec2<f32>(numerator) / vec2<f32>(2 * output);
            let s = clamp(min(uv, 1.0 - uv) / node.parameters.x, vec2<f32>(0.0), vec2<f32>(1.0));
            let w = s * s * (3.0 - 2.0 * s);
            weight = w.x * w.y;
            base = vec2<i32>(i32(x.x), i32(y.x)); t = vec2<f32>(x.y, y.y);        } else {
            let grid = i32(node.flags.z);
            let output = vec2<i32>(node.dimensions.xy);
            let twice_center = 2 * vec2<i32>(id.xy) + 1;
            let cell = twice_center * grid / (2 * output);
            let c = cell + vec2<i32>(i32(i) % 3 - 1, i32(i) / 3 - 1);
            let numerator = twice_center * grid - (2 * c + 1) * output;
            let denominator = 2 * output * grid;
            weight = window(vec2<f32>(numerator) / vec2<f32>(2 * output));
            let wrapped = vec2<u32>(((c % grid) + grid) % grid);
            var h = node.flags.y ^ (wrapped.x * 0x9e3779b9u) ^ (wrapped.y * 0x85ebca6bu);
            h ^= h >> 16u; h *= 0x7feb352du;
            h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
            var rotated = numerator;
            var divisor = denominator;
            switch h >> 30u {
                case 1u: { rotated = vec2<i32>(-numerator.y, numerator.x); divisor = denominator.yx; }
                case 2u: { rotated = -numerator; }
                case 3u: { rotated = vec2<i32>(numerator.y, -numerator.x); divisor = denominator.yx; }
                default: {}
            }
            let x = coordinate(h & 32767u, rotated.x, divisor.x, i32(n.x));
            let y = coordinate((h >> 15u) & 32767u, rotated.y, divisor.y, i32(n.y));
            base = vec2<i32>(i32(x.x), i32(y.x)); t = vec2<f32>(x.y, y.y);
        }
        let sample = bilinear(base, t);
        value += (sample - anchor) * weight;
        total += weight;
        if reconstruct {
            latent += weight * to_gaussian(probability(luminance(sample)));
            squares += weight * weight;
        }
    }
    let rgb = anchor + value / max(total, 1e-8);
    var out = rgb.rgb;
    if reconstruct {
        let restored = inverse(latent / max(sqrt(squares), 1e-8));
        let l = luminance(rgb);
        let restored_luminance = mix(l, restored, node.parameters.y);
        if node.flags.w != 0u { out = clamp(out * clamp(restored_luminance / max(l, 1e-8), 0.0, 16.0), vec3<f32>(0.0), vec3<f32>(1.0)); }
        else { out = vec3<f32>(restored_luminance); }
    }
    textureStore(result, vec2<i32>(id.xy), vec4<f32>(out, 1.0));
}



