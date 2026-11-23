// Same 64-byte layout as NodeUniform: dimensions, parameters, flags, reserved.
struct Params { size: vec2<u32>, source_size: vec2<u32>, parameters: vec4<f32>, offset: u32, encoded: u32, radius: u32, unused: u32, reserved: vec4<u32>, }
@group(0) @binding(0) var input_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;
struct Blocks { invalid: atomic<u32>, padding: array<u32, 3>, words: array<u32>, }
@group(1) @binding(0) var<storage, read_write> blocks: Blocks;

fn normal_at(p: vec2<u32>) -> vec3<f32> {
    var n = textureLoad(input_image, vec2<i32>(p), 0).xyz;
    if params.encoded != 0u { n = n * 2.0 - 1.0; }
    let l2 = dot(n,n);
    if l2 < 1e-20 { return vec3<f32>(0.0,0.0,1.0); }
    return n * inverseSqrt(l2);
}

// Inspect the original full-resolution signed source, before resize/padding.
@compute @workgroup_size(8,8)
fn validate(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let n = textureLoad(input_image, vec2<i32>(id.xy), 0).xyz;
    if n.z < 0.0 || any(n != n) || any(abs(n) > vec3<f32>(1e10)) {
        atomicOr(&blocks.invalid, 1u);
    }
}

// DDS uses floor-halved logical extents. Area overlaps are computed relative to
// each destination footprint, including the last row/column of odd dimensions.
@compute @workgroup_size(8,8)
fn mip(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.size) { return; }
    let start = id.xy * params.source_size;
    let first = start / params.size;
    let remainder = start % params.size;
    var n = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0u; y * params.size.y < remainder.y + params.source_size.y; y++) {
        let wy = min((y+1u)*params.size.y, remainder.y+params.source_size.y) - max(y*params.size.y,remainder.y);
        for (var x = 0u; x * params.size.x < remainder.x + params.source_size.x; x++) {
            let wx = min((x+1u)*params.size.x, remainder.x+params.source_size.x) - max(x*params.size.x,remainder.x);
            let weight = f32(wx)*f32(wy);
            n += normal_at(first+vec2<u32>(x,y)) * weight;
            total += weight;
        }
    }
    n /= total;
    let l2 = dot(n,n);
    if l2 > 1e-20 { n *= inverseSqrt(l2); }
    else { n = vec3<f32>(0.0,0.0,1.0); }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(n,1.0));
}

fn fit(values: array<u32,16>) -> vec2<u32> {
    var lo = 255u;
    var hi = 0u;
    for (var i=0u; i<16u; i++) { lo=min(lo,values[i]); hi=max(hi,values[i]); }
    var best_error = 0xffffffffu;
    var best = vec2<u32>(0u);
    for (var a = u32(max(i32(lo)-i32(params.radius),0)); a <= min(lo+params.radius,255u); a++) {
        for (var b = u32(max(i32(hi)-i32(params.radius),0)); b <= min(hi+params.radius,255u); b++) {
            for (var mode=0u; mode<2u; mode++) {
                let e0=select(a,b,mode==1u);
                let e1=select(b,a,mode==1u);
                // Integer palette in units of 1/35 gives exact rational BC4
                // distances and stable tie-breaking without float contraction.
                var palette: array<u32,8>;
                palette[0]=e0*35u; palette[1]=e1*35u;
                if e0>e1 {
                    for (var j=1u; j<=6u; j++) { palette[j+1u]=((7u-j)*e0+j*e1)*5u; }
                } else {
                    for (var j=1u; j<=4u; j++) { palette[j+1u]=((5u-j)*e0+j*e1)*7u; }
                    palette[6]=0u; palette[7]=8925u;
                }
                var error=0u;
                var bits=vec2<u32>(e0 | (e1<<8u),0u);
                for (var i=0u; i<16u; i++) {
                    var distance=0xffffffffu;
                    var choice=0u;
                    for (var j=0u; j<8u; j++) {
                        let delta=i32(values[i]*35u)-i32(palette[j]);
                        let d=u32(delta*delta);
                        if d<distance { distance=d; choice=j; }
                    }
                    error+=distance;
                    let shift=16u+3u*i;
                    if shift<32u {
                        bits.x |= choice<<shift;
                        if shift>29u { bits.y |= choice>>(32u-shift); }
                    } else { bits.y |= choice<<(shift-32u); }
                }
                if error<best_error { best_error=error; best=bits; }
            }
        }
    }
    return best;
}

@compute @workgroup_size(8,8)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid=(params.size+3u)/4u;
    if any(id.xy>=grid) { return; }
    var r: array<u32,16>;
    var g: array<u32,16>;
    for (var i=0u; i<16u; i++) {
        let p=min(id.xy*4u+vec2<u32>(i%4u,i/4u),params.size-1u);
        let n=normal_at(p);
        if n.z < 0.0 || any(n != n) { atomicOr(&blocks.invalid,1u); }
        r[i]=u32(floor(clamp(n.x*0.5+0.5,0.0,1.0)*255.0+0.5));
        g[i]=u32(floor(clamp(n.y*0.5+0.5,0.0,1.0)*255.0+0.5));
    }
    let rb=fit(r); let gb=fit(g);
    let at=params.offset+(id.y*grid.x+id.x)*4u;
    blocks.words[at]=rb.x; blocks.words[at+1u]=rb.y;
    blocks.words[at+2u]=gb.x; blocks.words[at+3u]=gb.y;
}
