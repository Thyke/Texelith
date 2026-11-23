struct Scene { camera: vec4<f32>, focus: vec4<f32>, light: vec4<f32>, reserved: vec4<f32> }
struct Material { color_roughness: vec4<f32>, metal_normal: vec4<f32>, reserved0: vec4<f32>, reserved1: vec4<f32> }
@group(0) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(0) var albedo_map: texture_2d<f32>;
@group(1) @binding(1) var normal_map: texture_2d<f32>;
@group(1) @binding(2) var roughness_map: texture_2d<f32>;
@group(1) @binding(3) var metallic_map: texture_2d<f32>;
@group(1) @binding(4) var ao_map: texture_2d<f32>;
@group(1) @binding(5) var height_map: texture_2d<f32>;
@group(1) @binding(6) var<uniform> material: Material;
@group(2) @binding(0) var<uniform> object: mat4x4<f32>;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) tangent: vec4<f32> }
struct Surface { @builtin(position) clip: vec4<f32>, @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) tangent: vec4<f32> }
fn unit(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    return select(fallback, v * inverseSqrt(max(dot(v,v), 1e-20)), dot(v,v) > 1e-20);
}
fn camera_back() -> vec3<f32> {
    return vec3(sin(scene.camera.x)*cos(scene.camera.y), sin(scene.camera.y), cos(scene.camera.x)*cos(scene.camera.y));
}
@vertex fn vertex(v: VertexInput) -> Surface {
    let back = camera_back();
    let right = unit(cross(vec3(0.0,1.0,0.0),back),vec3(1.0,0.0,0.0));
    let up = cross(back,right);
    let eye = scene.focus.xyz + back*scene.camera.z;
    let p = (object*vec4(v.position,1.0)).xyz;
    let view = p-eye;
    let z = -dot(view,back);
    // Right-handed camera, wgpu depth [0,1], 45 degree vertical field of view.
    let near = 0.01;
    let far = 2000000.0;
    let rotation = mat3x3(object[0].xyz,object[1].xyz,object[2].xyz);
    let cofactors = mat3x3(cross(rotation[1],rotation[2]),cross(rotation[2],rotation[0]),cross(rotation[0],rotation[1]));
    let det = dot(rotation[0],cofactors[0]);
    let inverse_transpose = cofactors * (sign(det) / max(abs(det),1e-12));
    var out: Surface;
    out.clip = vec4(dot(view,right)*2.41421356237/scene.camera.w, dot(view,up)*2.41421356237, (far*z-far*near)/(far-near), z);
    out.position = p;
    out.normal = unit(inverse_transpose*v.normal,vec3(0.0,0.0,1.0));
    out.tangent = vec4(unit(rotation*v.tangent.xyz,vec3(1.0,0.0,0.0)),v.tangent.w*sign(det));
    out.uv = v.uv;
    return out;
}
// All canonical maps share one boundary policy and displaced coordinate.
fn sample_level(map: texture_2d<f32>, uv: vec2<f32>, level: i32) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(map, level));
    let repeat = material.reserved0.z > 0.5;
    let coords = select(clamp(uv, vec2(0.0), vec2(1.0)), fract(uv), repeat);
    let p = coords * vec2<f32>(size) - 0.5;
    let lo = vec2<i32>(floor(p));
    let f = fract(p);
    let a = textureLoad(map,address(lo,size,repeat),level);
    let b = textureLoad(map,address(lo+vec2(1,0),size,repeat),level);
    let c = textureLoad(map,address(lo+vec2(0,1),size,repeat),level);
    let d = textureLoad(map,address(lo+vec2(1,1),size,repeat),level);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
fn sample_map(map: texture_2d<f32>, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(map));
    let footprint = max(length(dx*size),length(dy*size));
    let lod = clamp(log2(max(footprint,1.0)),0.0,f32(textureNumLevels(map)-1u));
    let lo = i32(floor(lod));
    let hi = min(lo+1,i32(textureNumLevels(map))-1);
    return mix(sample_level(map,uv,lo),sample_level(map,uv,hi),fract(lod));
}
fn direct_brdf(color: vec3<f32>, r: f32, m: f32, nv: f32, nl: f32, nh: f32, vh: f32) -> vec3<f32> {
    if nv <= 0.0 || nl <= 0.0 { return vec3(0.0); }
    let alpha=max(r*r,0.0025);
    let a2=alpha*alpha;
    // This algebra avoids cancellation at nh=1 with minimum roughness.
    let denominator=(1.0-nh*nh)+nh*nh*a2;
    let distribution=a2/max(3.14159265359*denominator*denominator,1e-12);
    let visibility=0.5/max(nl*sqrt(nv*nv*(1.0-a2)+a2)+nv*sqrt(nl*nl*(1.0-a2)+a2),1e-8);
    let f0=mix(vec3(0.04),color,m);
    let fresnel=f0+(1.0-f0)*pow(1.0-vh,5.0);
    return ((1.0-fresnel)*(1.0-m)*color/3.14159265359+distribution*visibility*fresnel)*nl;
}
@fragment fn fragment(s: Surface) -> @location(0) vec4<f32> {
    // Derivatives precede tile wrapping/POM so discontinuities cannot select
    // a spurious mip or violate derivative uniformity inside the ray march.
    let dx = dpdx(s.uv) * material.reserved1.x;
    let dy = dpdy(s.uv) * material.reserved1.x;
    let geometric=unit(s.normal,vec3(0.0,0.0,1.0));
    let tangent=unit(s.tangent.xyz-geometric*dot(geometric,s.tangent.xyz),vec3(1.0,0.0,0.0));
    let bitangent=cross(geometric,tangent)*s.tangent.w;
    let eye=scene.focus.xyz+camera_back()*scene.camera.z;
    let v=unit(eye-s.position,geometric);
    let turns=tile_rotation(s.uv);
    let uv=pom_uv(height_map,tiled_uv(s.uv,turns),tile_view(vec3(dot(v,tangent),-dot(v,bitangent),dot(v,geometric)),turns),material.reserved0);
    var mapped=sample_map(normal_map,uv,dx,dy).xyz;
    mapped.y *= material.metal_normal.y;
    mapped=tile_normal(mapped,turns);
    let normal_variance = max(0.0, 1.0 - dot(mapped,mapped) - 0.001);
    mapped=unit(mapped,vec3(0.0,0.0,1.0));
    let n=unit(tangent*mapped.x+bitangent*mapped.y+geometric*mapped.z,geometric);
    let l=scene.light.xyz;
    let h=unit(v+l,geometric);
    let color=clamp(sample_map(albedo_map,uv,dx,dy).rgb*material.color_roughness.rgb,vec3(0.0),vec3(1.0));
    let roughness=clamp(select(sample_map(roughness_map,uv,dx,dy).x,material.color_roughness.w,material.color_roughness.w>=0.0),0.0,1.0);
    // Broaden unresolved GGX lobes using normal-map mip variance and the
    // screen-space normal footprint, rather than clipping bright highlights.
    let ndx = dpdx(n);
    let ndy = dpdy(n);
    let variance = min(0.25, normal_variance + 0.5*(dot(ndx,ndx)+dot(ndy,ndy)));
    let r = max(roughness, pow(variance,0.25));
    let m=clamp(select(sample_map(metallic_map,uv,dx,dy).x,material.metal_normal.x,material.metal_normal.x>=0.0),0.0,1.0);
    // AO attenuates indirect diffuse only.
    let direct=direct_brdf(color,r,m,max(dot(n,v),0.0),max(dot(n,l),0.0),clamp(dot(n,h),0.0,1.0),clamp(dot(v,h),0.0,1.0));
    let radiance=direct*scene.light.w+indirect(color,r,m,n,v,clamp(sample_map(ao_map,uv,dx,dy).x,0.0,1.0));
    // Preserve linear HDR until the display pass. Alpha marks covered material
    // (the background clear uses zero); final display is always opaque.
    return vec4(min(radiance*exp2(scene.focus.w),vec3(65504.0)),1.0);
}
