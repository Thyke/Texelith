fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected image path")?;
    let image = texelith_io::decode_path(path, texelith_io::ColorIntent::PhotoSrgb)?;
    println!(
        "{} x {}; RGBA8: {} bytes ({:.0} MiB)",
        image.width,
        image.height,
        image.rgba8.len(),
        image.rgba8.len() as f64 / 1048576.0
    );
    println!(
        "Photo RGB assumed sRGB; alpha is unassociated and ignored by processing. ICC and EXIF are ignored."
    );
    Ok(())
}
