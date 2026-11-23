@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= node.dimensions.xy) { return; }
    let source = node.dimensions.z;
    let destination_size = node.dimensions.x;
    var value = vec4<f32>(0.0);
    if source > destination_size {
        let bounds = footprint(id.x, source, destination_size);
        let anchor = textureLoad(input0, vec2<i32>(i32(bounds.x), i32(id.y)), 0);
        var total_weight = 0.0;
        for (var offset = 0u; offset * destination_size < bounds.y + source; offset++) {
            let weight = overlap(offset, bounds.y, source, destination_size);
            let x = i32(bounds.x + offset);
            value += (textureLoad(input0, vec2<i32>(address(x, source, node.flags.x), i32(id.y)), 0) - anchor) * weight;
            total_weight += weight;
        }
        value = anchor + value / total_weight;
    } else {
        let center = (f32(id.x) + 0.5) * f32(source) / f32(destination_size) - 0.5;
        let x = i32(floor(center));
        let a = textureLoad(input0, vec2<i32>(address(x, source, node.flags.x), i32(id.y)), 0);
        let b = textureLoad(input0, vec2<i32>(address(x+1, source, node.flags.x), i32(id.y)), 0);
        value = mix(a, b, fract(center));
    }
    textureStore(output_image, id.xy, value);
}

