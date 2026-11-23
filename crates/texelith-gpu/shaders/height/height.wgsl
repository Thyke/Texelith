struct NodeUniform {
    dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32>,
}
@group(0) @binding(0) var input_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;
@group(1) @binding(0) var gaussian_coarse: texture_2d<f32>;
@group(1) @binding(1) var reconstructed_coarse: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = node.dimensions.xy;
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    let op = node.flags.x;
    var value = 0.0;
    if op == 0u {
        // Linear luminance and exact area restriction in one pass. The color
        // source has already crossed the public binary16 transfer boundary.
        let source = node.dimensions.zw;
        let lo = id.xy * source;
        let hi = (id.xy + vec2<u32>(1)) * source;
        let start = lo / size;
        let end = (hi + size - vec2<u32>(1)) / size;
        let anchor = dot(textureLoad(input_image, vec2<i32>(start), 0).rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        var total = 0.0;
        for (var y = start.y; y < end.y; y++) {
            let wy = min(hi.y, (y + 1u) * size.y) - max(lo.y, y * size.y);
            for (var x = start.x; x < end.x; x++) {
                let wx = min(hi.x, (x + 1u) * size.x) - max(lo.x, x * size.x);
                let rgb = textureLoad(input_image, vec2<i32>(i32(x), i32(y)), 0).rgb;
                let weight = (f32(wx) / f32(source.x)) * (f32(wy) / f32(source.y));
                value += (dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) - anchor) * weight;
                total += weight;
            }
        }
        value = anchor + value / total;
    } else if op == 1u {
        value = textureLoad(input_image, p, 0).r * node.parameters.x;
    } else if op == 2u {
        let center = (vec2<f32>(id.xy) + 0.5) * vec2<f32>(textureDimensions(gaussian_coarse)) / vec2<f32>(size) - 0.5;
        let repeat = node.flags.y != 0u;
        let base = bilinear_scalar(gaussian_coarse, center, repeat);
        let accumulated = bilinear_scalar(reconstructed_coarse, center, repeat);
        value = node.parameters.x * (textureLoad(input_image, p, 0).r - base) + accumulated;
    } else if op == 3u || op == 4u {
        // Each output pair contains exact f32 minimum/maximum of an 8x8 tile.
        let maximum = id.x % 2u != 0u;
        let origin = vec2<u32>(id.x / 2u, id.y) * 8u;
        let source = node.dimensions.zw;
        value = select(3.402823466e+38, -3.402823466e+38, maximum);
        for (var y = origin.y; y < min(origin.y + 8u, source.y); y++) {
            for (var x = origin.x; x < min(origin.x + 8u, source.x); x++) {
                var pixel = vec2<u32>(x, y);
                if op == 4u { pixel.x = 2u * x + select(0u, 1u, maximum); }
                let v = textureLoad(input_image, vec2<i32>(pixel), 0).r;
                if maximum { value = max(value, v); } else { value = min(value, v); }
            }
        }
    } else {
        let low = textureLoad(gaussian_coarse, vec2<i32>(0, 0), 0).r;
        let high = textureLoad(gaussian_coarse, vec2<i32>(1, 0), 0).r;
        let v = textureLoad(input_image, p, 0).r;
        value = node.parameters.x;
        if high != low {
            // Power-of-two scaling preserves adjacent values while avoiding
            // overflow and DX12 reciprocal flush-to-zero at extreme magnitudes.
            let magnitude = max(abs(low), abs(high));
            var scale = 1.0;
            if magnitude > 1.844674407e+19 { scale = 5.421010862e-20; }
            if magnitude < 5.421010862e-20 { scale = 1.844674407e+19; }
            let scaled_low = low * scale;
            value = (v * scale - scaled_low) / (high * scale - scaled_low);
            value = clamp(value, 0.0, 1.0);
        }
    }
    textureStore(output_image, p, vec4<f32>(value, 0.0, 0.0, 1.0));
}
