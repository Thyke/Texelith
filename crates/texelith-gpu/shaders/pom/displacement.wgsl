// Manual bilinear sampling keeps R32 inputs independent of float32-filterable.
fn address(p: vec2<i32>, size: vec2<i32>, repeat: bool) -> vec2<i32> {
    if repeat { return ((p % size) + size) % size; }
    return clamp(p, vec2(0), size - 1);
}
fn surface_sample(map: texture_2d<f32>, uv: vec2<f32>, repeat: bool) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(map));
    var coords = clamp(uv, vec2(0.0), vec2(1.0));
    if repeat { coords = fract(uv); }
    let p = coords * vec2<f32>(size) - 0.5;
    let lo = vec2<i32>(floor(p));
    let f = fract(p);
    let a = textureLoad(map,address(lo,size,repeat),0);
    let b = textureLoad(map,address(lo+vec2(1,0),size,repeat),0);
    let c = textureLoad(map,address(lo+vec2(0,1),size,repeat),0);
    let d = textureLoad(map,address(lo+vec2(1,1),size,repeat),0);
    let top = a + (b-a)*f.x;
    let bottom = c + (d-c)*f.x;
    return top + (bottom-top)*f.y;
}
fn pom_steps(cosine: f32) -> u32 {
    return u32(ceil(mix(32.0, 8.0, clamp(abs(cosine), 0.0, 1.0))));
}
// view.xy follows texture axes (v down), built from the geometric TBN before
// normal mapping. Each normalized height layer occupies scale UV units.
fn pom_uv(map: texture_2d<f32>, uv: vec2<f32>, view: vec3<f32>, params: vec4<f32>) -> vec2<f32> {
    if params.x <= 0.0 || view.z <= 0.0 { return uv; }
    let repeat = params.z > 0.5;
    let ray = view.xy / max(view.z, 0.05) * min(params.x, 1.0);
    if dot(ray,ray) == 0.0 { return uv; }
    let steps = pom_steps(view.z);
    let mid = params.y;
    var above = 1.0 - mid;
    var below = above;
    var last_height = clamp(surface_sample(map, uv + ray * above, repeat).x, 0.0, 1.0) - mid;
    if last_height == above { return uv + ray * above; }
    // March from the top bound to the bottom bound, retaining first crossing.
    for (var i = 1u; i <= steps; i++) {
        below = 1.0 - mid - f32(i) / f32(steps);
        let height = clamp(surface_sample(map, uv + ray * below, repeat).x, 0.0, 1.0) - mid;
        if height >= below { break; }
        above = below;
        last_height = height;
    }
    for (var i = 0u; i < 4u; i++) {
        let z = (above + below) * 0.5;
        let height = clamp(surface_sample(map, uv + ray * z, repeat).x, 0.0, 1.0) - mid;
        if height >= z { below = z; } else { above = z; last_height = height; }
    }
    // Interpolate the final signed distances: constant heights (including
    // midlevel) are exact without a shortcut that could miss an occluding ridge.
    let bottom_height = clamp(surface_sample(map, uv + ray * below, repeat).x, 0.0, 1.0) - mid;
    if last_height == 0.0 && bottom_height == 0.0 { return uv; }
    let a = above - last_height;
    let b = bottom_height - below;
    let z = (above * b + below * a) / max(a + b, 1e-20);
    return uv + ray * z;
}
