struct NodeUniform {
    dimensions: vec4<u32>,
    parameters: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var input_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var coarse_image: texture_2d<f32>;

fn gaussian(p: vec2<i32>, axis: vec2<i32>, repeat: bool) -> f32 {
    let weights = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var sum = 0.0;
    for (var i = 0; i < 5; i++) {
        sum += weights[i] * load_scalar(input_image, p + axis * (i - 2), repeat);
    }
    return sum;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_size = node.dimensions.xy;
    if any(id.xy >= out_size) { return; }
    let p = vec2<i32>(id.xy);
    let repeat = node.flags.y != 0u;
    let op = node.flags.x;
    var value = 0.0;
    if op == 0u {
        let axis = select(vec2<i32>(1, 0), vec2<i32>(0, 1), node.flags.z != 0u);
        value = gaussian(p, axis, repeat);
    } else if op == 1u || op == 5u {
        // Integer overlap numerators keep odd-grid cell boundaries exact even at 4K.
        let source = node.dimensions.zw;
        let lo = id.xy * source;
        let hi = (id.xy + vec2<u32>(1)) * source;
        let start = lo / out_size;
        let end = (hi + out_size - vec2<u32>(1)) / out_size;
        var anchor = textureLoad(input_image, vec2<i32>(start), 0).r;
        if op == 5u { anchor = gaussian(vec2<i32>(start), vec2<i32>(0, 1), repeat); }
        var total = 0.0;
        for (var y = start.y; y < end.y; y++) {
            let wy = min(hi.y, (y + 1u) * out_size.y) - max(lo.y, y * out_size.y);
            for (var x = start.x; x < end.x; x++) {
                let wx = min(hi.x, (x + 1u) * out_size.x) - max(lo.x, x * out_size.x);
                let cell = vec2<i32>(i32(x), i32(y));
                var sample_value = textureLoad(input_image, cell, 0).r;
                if op == 5u { sample_value = gaussian(cell, vec2<i32>(0, 1), repeat); }
                let weight = (f32(wx) / f32(source.x)) * (f32(wy) / f32(source.y));
                value += weight * (sample_value - anchor);
                total += weight;
            }
        }
        // Preserve arbitrary constants exactly, including on odd grids. Tiny
        // rounding ripples would become full-range noise after min/max remapping.
        value = anchor + value / total;
    } else if op == 2u {
        let center = (vec2<f32>(id.xy) + 0.5) * vec2<f32>(textureDimensions(input_image)) / vec2<f32>(out_size) - 0.5;
        value = bilinear_scalar(input_image, center, repeat);
    } else {
        let center = (vec2<f32>(id.xy) + 0.5) * vec2<f32>(textureDimensions(coarse_image)) / vec2<f32>(out_size) - 0.5;
        let coarse = bilinear_scalar(coarse_image, center, repeat);
        let fine = textureLoad(input_image, p, 0).r;
        if op == 3u { value = fine - coarse; }
        else { value = node.parameters.x * fine + node.parameters.y * coarse; }
    }
    textureStore(output_image, p, vec4<f32>(value, 0.0, 0.0, 1.0));
}
