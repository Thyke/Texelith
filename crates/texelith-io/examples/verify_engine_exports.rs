use image::ImageFormat;
use std::{env, fs, path::PathBuf};
use texelith_io::encoders::{PngChannels, PngColorIntent, encode_png8};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("--output") {
        return Err("usage: verify_engine_exports --output DIR".into());
    }
    let out = PathBuf::from(args.next().ok_or("missing output directory")?);
    fs::create_dir_all(&out)?;
    let (width, height) = (4, 4);
    let albedo = [64, 128, 192].repeat(16);
    let normal = [128, 128, 255].repeat(16);
    let smoothness = [0, 0, 0, 255].repeat(16);
    for (name, channels, intent, data) in [
        (
            "albedo.png",
            PngChannels::Rgb,
            PngColorIntent::Srgb,
            albedo.as_slice(),
        ),
        (
            "normal_gl.png",
            PngChannels::Rgb,
            PngColorIntent::Linear,
            normal.as_slice(),
        ),
        (
            "metallic_smoothness.png",
            PngChannels::Rgba,
            PngColorIntent::Linear,
            smoothness.as_slice(),
        ),
    ] {
        let encoded = encode_png8(width, height, channels, intent, data)?;
        let path = out.join(name);
        fs::write(&path, &encoded)?;
        let decoded = image::load_from_memory_with_format(&encoded, ImageFormat::Png)?;
        assert_eq!(decoded.as_bytes(), data, "decoded mismatch in {name}");
        println!(
            "{}: {} bytes, {}x{}, independently decoded exact samples",
            path.display(),
            encoded.len(),
            width,
            height
        );
    }
    let metadata = serde_json::json!({
        "fixture": "texelith-engine-contract-v1", "dimensions": [width, height],
        "engine_runtime_tested": false,
        "files": [
            {"path":"albedo.png", "channels":"RGB", "transfer":"sRGB"},
            {"path":"normal_gl.png", "channels":"RGB", "convention":"OpenGL", "srgb":false},
            {"path":"metallic_smoothness.png", "channels":"RGBA", "metallic":"R", "smoothness":"A", "srgb":false}
        ]
    });
    fs::write(
        out.join("fixture.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!("engine runtime coverage: unverified (synthetic offline fixture only)");
    Ok(())
}
