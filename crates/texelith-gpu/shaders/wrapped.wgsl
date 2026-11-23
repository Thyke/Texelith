// Shared address and interpolation functions. No hardware float filtering needed.
fn address(p: vec2<i32>, size: vec2<i32>, repeat: bool) -> vec2<i32> {
    if repeat { return ((p % size) + size) % size; }
    return clamp(p, vec2<i32>(0), size - vec2<i32>(1));
}
fn load_scalar(image: texture_2d<f32>, p: vec2<i32>, repeat: bool) -> f32 {
    return textureLoad(image, address(p, vec2<i32>(textureDimensions(image)), repeat), 0).r;
}
fn bilinear_scalar(image: texture_2d<f32>, p: vec2<f32>, repeat: bool) -> f32 {
    let base = vec2<i32>(floor(p));
    let f = fract(p);
    let anchor = load_scalar(image, base, repeat);
    if anchor == load_scalar(image, base + vec2<i32>(1, 0), repeat)
        && anchor == load_scalar(image, base + vec2<i32>(0, 1), repeat)
        && anchor == load_scalar(image, base + vec2<i32>(1, 1), repeat) {
        return anchor;
    }
    let top = mix(load_scalar(image, base, repeat), load_scalar(image, base + vec2<i32>(1, 0), repeat), f.x);
    let bottom = mix(load_scalar(image, base + vec2<i32>(0, 1), repeat), load_scalar(image, base + vec2<i32>(1, 1), repeat), f.x);
    return mix(top, bottom, f.y);
}
