struct NodeUniform {
    dimensions: vec4<u32>, parameters: vec4<f32>, flags: vec4<u32>, reserved: vec4<u32>,
}
@group(0) @binding(0) var input0: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> node: NodeUniform;

fn address(i: i32, n: u32, repeat: u32) -> i32 {
    let size = i32(n);
    if repeat != 0u { return ((i % size) + size) % size; }
    return clamp(i, 0, size - 1);
}

// Keep the footprint relative to its first source texel. Integer numerators
// preserve NPOT overlaps without subtracting rounded absolute f32 endpoints.
fn footprint(pixel: u32, source: u32, destination_size: u32) -> vec2<u32> {
    let start = pixel * source;
    return vec2<u32>(start / destination_size, start % destination_size);
}

fn overlap(offset: u32, remainder: u32, source: u32, destination_size: u32) -> f32 {
    let lo = max(remainder, offset * destination_size);
    let hi = min(remainder + source, (offset + 1u) * destination_size);
    return f32(hi - lo);
}

