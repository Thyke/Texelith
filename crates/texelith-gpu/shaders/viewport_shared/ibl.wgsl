@group(0) @binding(1) var irradiance:texture_2d<f32>;
@group(0) @binding(2) var specular:texture_2d<f32>;
@group(0) @binding(3) var brdf_lut:texture_2d<f32>;
fn specular_mip(d:vec3<f32>,mip:u32)->vec3<f32> {
    return cube_sample(specular,d,128u>>mip,256u-(256u>>mip));
}
fn environment_direction(d:vec3<f32>)->vec3<f32> {
    let c=cos(scene.reserved.y);let s=sin(scene.reserved.y);
    return vec3(c*d.x+s*d.z,d.y,-s*d.x+c*d.z);
}
fn environment_specular(d:vec3<f32>,roughness:f32)->vec3<f32> {
    let lod=roughness*7.0;let lo=u32(floor(lod));let hi=min(lo+1u,7u);
    return mix(specular_mip(d,lo),specular_mip(d,hi),fract(lod));
}
fn indirect(color:vec3<f32>,r:f32,m:f32,n:vec3<f32>,v:vec3<f32>,ao:f32)->vec3<f32> {
    let nv=max(dot(n,v),0.0);let f0=mix(vec3(0.04),color,m);
    let f=f0+(max(vec3(1.0-r),f0)-f0)*pow(1.0-nv,5.0);
    let diffuse=cube_sample(irradiance,environment_direction(n),32u,0u)*color*(1.0-m)*(1.0-f)*ao;
    let prefiltered=environment_specular(environment_direction(reflect(-v,n)),r);
    let ab=surface_sample(brdf_lut,vec2(nv,r),false).rg;
    return (diffuse+prefiltered*(f0*ab.x+ab.y))*exp2(scene.reserved.x);
}
struct Sky { @builtin(position) clip:vec4<f32>, @location(0) ndc:vec2<f32> }
@vertex fn sky_vertex(@builtin(vertex_index) index:u32)->Sky {
    let p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
    var out:Sky;out.clip=vec4(p[index],1.0,1.0);out.ndc=p[index];return out;
}
@fragment fn sky_fragment(s:Sky)->@location(0) vec4<f32> {
    if scene.reserved.z<0.5 {return vec4(vec3(0.018),0.0);}
    let back=camera_back();
    let right=unit(cross(vec3(0.0,1.0,0.0),back),vec3(1.0,0.0,0.0));
    let up=cross(back,right);
    let ray=normalize(-back+right*s.ndc.x*scene.camera.w/2.41421356237+up*s.ndc.y/2.41421356237);
    let radiance=environment_specular(environment_direction(ray),scene.reserved.w)*exp2(scene.reserved.x);
    // Sky uses its own fixed display mapping; material exposure does not affect it.
    let x=min(radiance,vec3(65504.0));
    let mapped=clamp(x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14),vec3(0.0),vec3(1.0));
    return vec4(mapped,0.0);
}
