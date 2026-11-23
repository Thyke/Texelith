struct Channels { descriptors: array<vec4<u32>, 4>, constants: vec4<f32>, }
@group(1) @binding(0) var input1: texture_2d<f32>;
@group(1) @binding(1) var input2: texture_2d<f32>;
@group(1) @binding(2) var input3: texture_2d<f32>;
@group(1) @binding(3) var<uniform> channels: Channels;

fn load_channel(c: u32, p: vec2<i32>) -> vec4<f32> {
    switch c {
        case 0u: { return textureLoad(input0, p, 0); }
        case 1u: { return textureLoad(input1, p, 0); }
        case 2u: { return textureLoad(input2, p, 0); }
        default: { return textureLoad(input3, p, 0); }
    }
}
fn sample_channel(c: u32, p: vec2<u32>, boundary: u32) -> vec4<f32> {
    if node.flags.x != 1u {
        let b = select(boundary, node.reserved.x, node.flags.x == 2u);
        return load_channel(c, vec2<i32>(address(i32(p.x), node.dimensions.z, b), address(i32(p.y), node.dimensions.w, b)));
    }
    // Horizontal filtering is a separate f32 pass, shared by swizzled channels.
    let source = node.dimensions.w;
    let destination_size = node.dimensions.y;
    if source == destination_size { return load_channel(c, vec2<i32>(p)); }
    if source > destination_size {
        let bounds = footprint(p.y, source, destination_size);
        let anchor = load_channel(c, vec2<i32>(i32(p.x), i32(bounds.x)));
        var value = vec4<f32>(0.0);
        var total_weight = 0.0;
        for (var offset = 0u; offset * destination_size < bounds.y + source; offset++) {
            let weight = overlap(offset, bounds.y, source, destination_size);
            let y = i32(bounds.x + offset);
            value += (load_channel(c, vec2<i32>(i32(p.x), address(y, source, boundary))) - anchor) * weight;
            total_weight += weight;
        }
        return anchor + value / total_weight;
    }
    let center = (f32(p.y) + 0.5) * f32(source) / f32(destination_size) - 0.5;
    let y = i32(floor(center));
    return mix(load_channel(c, vec2<i32>(i32(p.x), address(y, source, boundary))),
        load_channel(c, vec2<i32>(i32(p.x), address(y+1, source, boundary))), fract(center));
}
const BAYER = array<u32, 16>(0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
fn convert(c: u32, id: vec2<u32>) -> f32 {
        let d = channels.descriptors[c];
        var value = channels.constants[c];
        if d.x != 0u {
            var pixel = sample_channel(c, id, d.z);
            if d.x == 3u {
                // Antipodal/degenerate averages have a deterministic +Z fallback.
                let length2 = dot(pixel.xyz, pixel.xyz);
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if length2 > 1e-20 { normal = pixel.xyz * inverseSqrt(length2); }
                if (d.w & 4u) != 0u { normal.y = -normal.y; }
                pixel = vec4<f32>(normal * 0.5 + 0.5, 1.0);
            }
            if d.x == 2u { pixel.a = 1.0; }
            value = pixel[d.y];
        }
        if (d.w & 1u) != 0u { value = 1.0 - value; }
        if (d.w & 2u) != 0u {
            if value <= 0.0031308 { value *= 12.92; }
            else { value = 1.055 * pow(value, 1.0/2.4) - 0.055; }
        }
        if node.flags.y == 0u {
            value = (value - node.parameters.z) * node.parameters.w;
        } else {
            var offset = 0.0;
            if node.flags.z != 0u { offset = (f32(BAYER[(id.y % 4u)*4u + id.x%4u])+0.5)/16.0-0.5; }
            let codes = f32(node.flags.y);
            value = floor(clamp(clamp(value, node.parameters.x, node.parameters.y)*codes + offset, 0.0, codes)+0.5)/codes;
        }
        return value;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    var result = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    result.r = convert(0u, id.xy);
    if node.flags.w > 1u { result.g = convert(1u, id.xy); }
    if node.flags.w > 2u { result.b = convert(2u, id.xy); }
    if node.flags.w > 3u { result.a = convert(3u, id.xy); }
    textureStore(output_image, id.xy, result);
}


