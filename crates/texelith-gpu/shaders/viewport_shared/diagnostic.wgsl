struct Scene {
    dimensions: vec4<u32>,
    color: vec4<f32>,
    flags: vec4<u32>,
    reserved: vec4<u32>,
}
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var normal: texture_2d<f32>;
@group(1) @binding(2) var roughness: texture_2d<f32>;
@group(1) @binding(3) var metallic: texture_2d<f32>;
@group(1) @binding(4) var ao: texture_2d<f32>;
@group(1) @binding(5) var height: texture_2d<f32>;
@group(1) @binding(6) var<uniform> material: Scene;
@group(2) @binding(0) var<uniform> object: mat4x4<f32>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(positions[index], 0.0, 1.0);
}

fn sample_height(uv: vec2<f32>) -> f32 {
    let size = textureDimensions(height);
    let p = clamp(uv * vec2<f32>(size) - 0.5, vec2(0.0), vec2<f32>(size - 1u));
    let lo = vec2<i32>(floor(p));
    let hi = min(lo + 1, vec2<i32>(size) - 1);
    let f = fract(p);
    let a = mix(textureLoad(height, lo, 0).x, textureLoad(height, vec2(hi.x, lo.y), 0).x, f.x);
    let b = mix(textureLoad(height, vec2(lo.x, hi.y), 0).x, textureLoad(height, hi, 0).x, f.x);
    return mix(a, b, f.y);
}


fn read_map(map: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(map));
    let p = uv * vec2<f32>(size) - 0.5;
    let lo = vec2<i32>(floor(p));
    let f = fract(p);
    let a = textureLoad(map,((lo % size)+size)%size,0);
    let b = textureLoad(map,(((lo+vec2(1,0)) % size)+size)%size,0);
    let c = textureLoad(map,(((lo+vec2(0,1)) % size)+size)%size,0);
    let d = textureLoad(map,(((lo+vec2(1,1)) % size)+size)%size,0);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
fn decode_codes(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c+0.055)/1.055,vec3(2.4)),c/12.92,c<=vec3(0.04045));
}
fn map_view(pixel: vec2<f32>) -> vec4<f32> {
    let p = pixel / vec2<f32>(scene.dimensions.xy) * f32(scene.flags.z);
    var turns = 0u;
    if scene.flags.w != 0u {
        let cell = vec2<u32>(floor(p));
        var h=scene.reserved.x ^ (cell.x*0x9e3779b9u) ^ (cell.y*0x85ebca6bu);
        h ^= h>>16u; h*=0x7feb352du; h^=h>>15u;
        turns=h&3u;
    }
    var uv=fract(p)-0.5;
    for(var i=0u;i<turns;i++) { uv=vec2(-uv.y,uv.x); }
    uv+=0.5;
    if scene.flags.y==0u { return vec4(read_map(albedo,uv).rgb,1.0); }
    if scene.flags.y==1u {
        var n=read_map(normal,uv).xyz;
        if scene.reserved.y!=0u { n.y = -n.y; }
        for(var i=0u;i<turns;i++) { n=vec3(-n.y,n.x,n.z); }
        n=normalize(n);
        return vec4(decode_codes(clamp(n*0.5+0.5,vec3(0.0),vec3(1.0))),1.0);
    }
    var value=read_map(height,uv).x;
    if scene.flags.y==2u { value=read_map(roughness,uv).x; }
    if scene.flags.y==3u { value=read_map(metallic,uv).x; }
    if scene.flags.y==4u { value=read_map(ao,uv).x; }
    return vec4(decode_codes(vec3(clamp(value,0.0,1.0))),1.0);
}
@fragment fn fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    if scene.flags.x == 0u { return vec4(scene.color.rgb, 1.0); }
    if scene.flags.x == 2u { return map_view(pixel.xy); }
    let value = clamp(sample_height(pixel.xy / vec2<f32>(scene.dimensions.xy)), 0.0, 65504.0);
    return vec4(vec3(value), 1.0);
}
