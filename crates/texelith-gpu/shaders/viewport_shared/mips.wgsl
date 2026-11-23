@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst = textureDimensions(output);
    if any(id.xy >= dst) { return; }
    let src = textureDimensions(input);
    let span = vec2<f32>(src) / vec2<f32>(dst);
    let start = vec2<f32>(id.xy) * span;
    let base = vec2<i32>(floor(start));
    let offset = fract(start);
    var sum = vec4(0.0);
    var weight = 0.0;
    // At most three source texels per axis, including odd and 1-wide levels.
    for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 3; x++) {
            let p = vec2<i32>(x, y);
            let lo = max(vec2<f32>(p) - offset, vec2(0.0));
            let hi = min(vec2<f32>(p + vec2(1)) - offset, span);
            let overlap = max(hi - lo, vec2(0.0));
            let w = overlap.x * overlap.y;
            if w > 0.0 {
                sum += textureLoad(input, clamp(base + p, vec2(0), vec2<i32>(src) - 1), 0) * w;
                weight += w;
            }
        }
    }
    textureStore(output, vec2<i32>(id.xy), sum / max(weight, 1e-20));
}
