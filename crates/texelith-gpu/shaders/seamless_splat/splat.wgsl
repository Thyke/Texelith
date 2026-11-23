struct NodeUniform {
    dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32>,
}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var result: texture_storage_2d<OUTPUT_FORMAT, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

fn address(p: vec2<i32>) -> vec2<i32> {
    let n = vec2<i32>(node.dimensions.zw);
    if node.flags.x != 0u { return ((p % n) + n) % n; }
    return clamp(p, vec2<i32>(0), n - vec2<i32>(1));
}
fn texel(p: vec2<i32>) -> vec4<f32> {
    let value = textureLoad(source, p, 0);
    if node.flags.w != 0u { return vec4<f32>(value.r, 0.0, 0.0, 1.0); }
    return value;
}
fn bilinear(base: vec2<i32>, t: vec2<f32>) -> vec4<f32> {
    let a = texel(address(base));
    let b = texel(address(base + vec2<i32>(1, 0)));
    let c = texel(address(base + vec2<i32>(0, 1)));
    let d = texel(address(base + vec2<i32>(1, 1)));
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
// Evaluate the two rational coordinate terms separately. Converting a large
// source-space coordinate to f32 first loses fractional texel precision at 4K.
// All products fit i32 at the validated 4096-axis and grid <= 8 bounds.
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
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let grid = i32(node.flags.z);
    let output = vec2<i32>(node.dimensions.xy);
    let twice_center = 2 * vec2<i32>(id.xy) + 1;
    let cell = twice_center * grid / (2 * output);
    let n = vec2<i32>(node.dimensions.zw);
    // Difference accumulation preserves constants exactly before half-float stores.
    let anchor = texel(vec2<i32>(0));
    var value = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = 0; i < 9; i++) {
        let c = cell + vec2<i32>(i % 3 - 1, i / 3 - 1);
        let numerator = twice_center * grid - (2 * c + 1) * output;
        let denominator = 2 * output * grid;
        let d = vec2<f32>(numerator) / vec2<f32>(2 * output);
        let weight = window(d);
        let wrapped = vec2<u32>(((c % grid) + grid) % grid);
        var h = node.flags.y ^ (wrapped.x * 0x9e3779b9u) ^ (wrapped.y * 0x85ebca6bu);
        h ^= h >> 16u; h *= 0x7feb352du;
        h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
        let translation = vec2<u32>(h & 32767u, (h >> 15u) & 32767u);
        var rotated = numerator;
        var divisor = denominator;
        switch h >> 30u {
            case 1u: { rotated = vec2<i32>(-numerator.y, numerator.x); divisor = denominator.yx; }
            case 2u: { rotated = -numerator; }
            case 3u: { rotated = vec2<i32>(numerator.y, -numerator.x); divisor = denominator.yx; }
            default: {}
        }
        let x = coordinate(translation.x, rotated.x, divisor.x, n.x);
        let y = coordinate(translation.y, rotated.y, divisor.y, n.y);
        value += (bilinear(vec2<i32>(i32(x.x), i32(y.x)), vec2<f32>(x.y, y.y)) - anchor) * weight;
        total += weight;
    }
    textureStore(result, vec2<i32>(id.xy), vec4<f32>((anchor + value / max(total, 1e-8)).rgb, 1.0));
}
