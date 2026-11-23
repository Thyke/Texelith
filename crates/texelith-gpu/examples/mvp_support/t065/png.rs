//! Minimal original PNG encoder for small diagnostic RGBA captures. Uncompressed
//! DEFLATE avoids another runtime/dev dependency; captures stay below 2 MiB.
pub fn save(path: &std::path::Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    if pixels.len() != width as usize * height as usize * 4 {
        return Err("capture extent mismatch".into());
    }
    let mut raw = Vec::new();
    for row in pixels.chunks_exact(width as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let count = raw.len().div_ceil(65535);
    for (i, chunk) in raw.chunks(65535).enumerate() {
        z.push(u8::from(i + 1 == count));
        let len = chunk.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for byte in raw {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::from(width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend([8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, png).map_err(|e| e.to_string())
}
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend(kind);
    out.extend(data);
    let mut crc = !0u32;
    for byte in kind.iter().chain(data) {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320u32.wrapping_mul(crc & 1));
        }
    }
    out.extend_from_slice(&(!crc).to_be_bytes());
}
