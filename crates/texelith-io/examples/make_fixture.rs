use image::{ImageBuffer, Rgba};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut size, mut output) = (None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--size" => size = Some(args.next().ok_or("missing size")?.parse::<u32>()?),
            "--output" => output = Some(args.next().ok_or("missing output")?),
            _ => return Err("unknown argument".into()),
        }
    }
    let size = size.ok_or("missing --size")?;
    if size == 0 || size > 4096 {
        return Err("size must be 1..=4096".into());
    }
    let mut hash = 0xcbf29ce484222325u64;
    let img = ImageBuffer::from_fn(size, size, |x, y| {
        let p = [
            (x.wrapping_mul(17) ^ y.wrapping_mul(3)) as u8,
            (y.wrapping_mul(13) ^ (x >> 3)) as u8,
            (x.wrapping_add(y).wrapping_mul(7)) as u8,
            255,
        ];
        for b in p {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
        Rgba(p)
    });
    img.save_with_format(output.ok_or("missing --output")?, image::ImageFormat::Png)?;
    println!("RGBA8 checksum (FNV-1a): {hash:016x}");
    Ok(())
}
