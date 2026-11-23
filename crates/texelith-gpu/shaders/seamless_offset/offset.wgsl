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
fn bilinear(p: vec2<f32>) -> vec4<f32> {
    let base = vec2<i32>(floor(p));
    let t = fract(p);
    let a = textureLoad(source, address(base), 0);
    let b = textureLoad(source, address(base + vec2<i32>(1, 0)), 0);
    let c = textureLoad(source, address(base + vec2<i32>(0, 1)), 0);
    let d = textureLoad(source, address(base + vec2<i32>(1, 1)), 0);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
fn window(uv: vec2<f32>) -> f32 {
    let t = clamp(min(uv, vec2<f32>(1.0) - uv) / node.parameters.x, vec2<f32>(0.0), vec2<f32>(1.0));
    let w = t * t * (vec2<f32>(3.0) - 2.0 * t);
    return w.x * w.y;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let n = vec2<f32>(node.dimensions.zw);
    // Keep equal-size texel coordinates exact; avoid a redundant UV/size
    // round trip whose reciprocal error grows with the source dimension.
    var center = vec2<f32>(id.xy);
    // Accumulate deviations to preserve a constant exactly before the binary16
    // store, even on backends whose storage conversion rounds toward zero.
    var anchor: vec4<f32>;
    if all(node.dimensions.xy == node.dimensions.zw) {
        anchor = textureLoad(source, vec2<i32>(id.xy), 0);
    } else {
        center = (center + 0.5) * (n / vec2<f32>(node.dimensions.xy)) - 0.5;
        anchor = bilinear(center);
    }
    var value = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let shift = vec2<f32>(select(0.0, 0.5, (i & 1u) == 0u), select(0.0, 0.5, (i & 2u) == 0u));
        // Work in texel coordinates to retain exact half-texel NPOT offsets;
        // a UV round trip would amplify float error by the source dimension.
        let translated = center + shift * n;
        let p = translated - floor((translated + 0.5) / n) * n;
        let uv = (p + 0.5) / n;
        let weight = window(uv);
        value += (bilinear(p) - anchor) * weight;
        total += weight;
    }
    textureStore(result, vec2<i32>(id.xy), vec4<f32>((anchor + value / max(total, 1e-8)).rgb, 1.0));
}
