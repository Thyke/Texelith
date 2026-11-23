@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float,write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst = textureDimensions(output);
    if any(id.xy >= dst) { return; }
    let src = textureDimensions(input);
    let ratio = vec2<f32>(src) / vec2<f32>(dst);
    let start = vec2<f32>(id.xy) * ratio;
    let base = vec2<i32>(floor(start));
    let offset = start - vec2<f32>(base);
    var sum = vec4(0.0);
    var weight = 0.0;
    for (var y=0; y<i32(ceil(offset.y+ratio.y)); y++) {
        for (var x=0; x<i32(ceil(offset.x+ratio.x)); x++) {
            let low = max(vec2<f32>(f32(x),f32(y))-offset,vec2(0.0));
            let high = min(vec2<f32>(f32(x+1),f32(y+1))-offset,ratio);
            let overlap = max(high-low,vec2(0.0));
            let w = overlap.x*overlap.y;
            sum += textureLoad(input,clamp(base+vec2(x,y),vec2(0),vec2<i32>(src)-1),0)*w;
            weight += w;
        }
    }
    textureStore(output,id.xy,sum/max(weight,1e-12));
}
