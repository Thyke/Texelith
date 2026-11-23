const PI: f32 = 3.141592653589793;
fn face_direction(face:u32, uv:vec2<f32>) -> vec3<f32> {
    let p=uv*2.0-1.0;
    switch face {
        case 0u: {return normalize(vec3(1.0,-p.y,-p.x));}
        case 1u: {return normalize(vec3(-1.0,-p.y,p.x));}
        case 2u: {return normalize(vec3(p.x,1.0,p.y));}
        case 3u: {return normalize(vec3(p.x,-1.0,-p.y));}
        case 4u: {return normalize(vec3(p.x,-p.y,1.0));}
        default: {return normalize(vec3(-p.x,-p.y,-1.0));}
    }
}
// xy = local face UV, z = face index. Identical convention in both passes.
fn face_uv(d:vec3<f32>) -> vec3<f32> {
    let a=abs(d); var p=vec2(0.0); var face=0.0;
    if a.x>=a.y && a.x>=a.z {
        if d.x>=0.0 {face=0.0;p=vec2(-d.z,-d.y)/a.x;} else {face=1.0;p=vec2(d.z,-d.y)/a.x;}
    } else if a.y>=a.z {
        if d.y>=0.0 {face=2.0;p=vec2(d.x,d.z)/a.y;} else {face=3.0;p=vec2(d.x,-d.z)/a.y;}
    } else {
        if d.z>=0.0 {face=4.0;p=vec2(d.x,-d.y)/a.z;} else {face=5.0;p=vec2(-d.x,-d.y)/a.z;}
    }
    return vec3(p*0.5+0.5,face);
}
fn cube_tap(map:texture_2d<f32>, face:u32, p:vec2<i32>, size:u32, row:u32) -> vec3<f32> {
    let q=face_uv(face_direction(face,(vec2<f32>(p)+0.5)/f32(size)));
    let pixel=clamp(vec2<i32>(q.xy*f32(size)),vec2(0),vec2(i32(size)-1));
    return textureLoad(map,pixel+vec2(i32(q.z)*i32(size),i32(row)),0).rgb;
}
fn cube_sample(map:texture_2d<f32>, d:vec3<f32>, size:u32, row:u32) -> vec3<f32> {
    let q=face_uv(d); let p=q.xy*f32(size)-0.5;
    let lo=vec2<i32>(floor(p)); let f=fract(p); let face=u32(q.z);
    return mix(mix(cube_tap(map,face,lo,size,row),cube_tap(map,face,lo+vec2(1,0),size,row),f.x),
        mix(cube_tap(map,face,lo+vec2(0,1),size,row),cube_tap(map,face,lo+vec2(1,1),size,row),f.x),f.y);
}
