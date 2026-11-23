struct Params { work:vec4<u32>, source:vec4<u32>, reserved0:vec4<u32>, reserved1:vec4<u32> }
@group(0) @binding(0) var<storage,read> pixels:array<vec4<f32>>;
@group(0) @binding(1) var cube:texture_2d<f32>;
@group(0) @binding(2) var output_map:texture_storage_2d<rgba16float,write>;
@group(0) @binding(3) var<uniform> params:Params;
fn hammersley(i:u32,count:u32)->vec2<f32> {
    return vec2(f32(i)/f32(count),f32(reverseBits(i))*2.3283064365386963e-10);
}
fn basis(n:vec3<f32>)->mat3x3<f32> {
    let up=select(vec3(0.0,0.0,1.0),vec3(1.0,0.0,0.0),abs(n.z)>0.99);
    let t=normalize(cross(up,n));return mat3x3(t,cross(n,t),n);
}
fn source_tap(p:vec2<i32>)->vec3<f32> {
    let w=i32(params.source.x);let h=i32(params.source.y);
    return pixels[u32(clamp(p.y,0,h-1)*w+((p.x%w+w)%w))].rgb;
}
fn equirect(d:vec3<f32>)->vec3<f32> {
    let uv=vec2(atan2(d.z,d.x)/(2.0*PI)+0.5,acos(clamp(d.y,-1.0,1.0))/PI);
    let p=uv*vec2<f32>(params.source.xy)-0.5;let lo=vec2<i32>(floor(p));let f=fract(p);
    return mix(mix(source_tap(lo),source_tap(lo+vec2(1,0)),f.x),mix(source_tap(lo+vec2(0,1)),source_tap(lo+vec2(1,1)),f.x),f.y);
}
fn ggx(x:vec2<f32>,r:f32)->vec3<f32> {
    let a=max(r*r,0.0025);let phi=2.0*PI*x.x;
    let c=sqrt((1.0-x.y)/(1.0+(a*a-1.0)*x.y));let s=sqrt(max(0.0,1.0-c*c));
    return vec3(cos(phi)*s,sin(phi)*s,c);
}
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let mode=params.work.x;let face=params.work.y;let mip=params.work.z;let size=params.work.w;
    if any(id.xy>=vec2(size)) {return;}
    let uv=(vec2<f32>(id.xy)+0.5)/f32(size);let n=face_direction(face,uv);
    var value=vec3(0.0);var row=0u;
    if mode==0u { value=equirect(n); }
    if mode==1u {
        let frame=basis(n);
        // Cosine-weighted hemisphere integrates E/pi; normalized weights preserve constants.
        for(var i=0u;i<64u;i++) {
            let x=hammersley(i,64u);let radius=sqrt(x.y);let phi=2.0*PI*x.x;
            let l=frame*vec3(radius*cos(phi),radius*sin(phi),sqrt(1.0-x.y));
            value+=cube_sample(cube,l,256u,0u)/64.0;
        }
    }
    if mode==2u {
        row=256u-(256u>>mip);let r=f32(mip)/7.0;
        let frame=basis(n);var weight=0.0;
        for(var i=0u;i<128u;i++) {
            let h=frame*ggx(hammersley(i,128u),r);let l=normalize(2.0*dot(n,h)*h-n);let nl=max(dot(n,l),0.0);
            // Scale radiance first so even maximum accepted HDR cannot overflow.
            value+=cube_sample(cube,l,256u,0u)*(nl/128.0);weight+=nl/128.0;
        }
        value/=max(weight,1e-8);
    }
    if mode==3u {
        let nv=uv.x;let r=uv.y;let v=vec3(sqrt(1.0-nv*nv),0.0,nv);let a=max(r*r,0.0025);let a2=a*a;
        var ab=vec2(0.0);
        for(var i=0u;i<128u;i++) {
            let h=ggx(hammersley(i,128u),r);let vh=max(dot(v,h),0.0);let l=2.0*vh*h-v;let nl=max(l.z,0.0);
            if nl>0.0 {
                let vis=0.5/max(nl*sqrt(nv*nv*(1.0-a2)+a2)+nv*sqrt(nl*nl*(1.0-a2)+a2),1e-8);
                let factor=4.0*vis*nl*vh/max(h.z,1e-8);let fc=pow(1.0-vh,5.0);
                ab+=vec2((1.0-fc)*factor,fc*factor)/128.0;
            }
        }
        value=vec3(ab,0.0);
    }
    textureStore(output_map,vec2<i32>(id.xy+vec2(select(face*size,0u,mode==3u),row)),vec4(clamp(value,vec3(0.0),vec3(65504.0)),1.0));
}
