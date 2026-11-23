use super::*;
impl Gpu {
    pub fn finish(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    /// Readback belongs exclusively to verification, never the preview renderer.
    pub fn read_display(&self, texture: &wgpu::Texture) -> Result<Vec<u8>> {
        require(
            texture.format() == wgpu::TextureFormat::Rgba8Unorm,
            "display must contain explicit gamma codes in Rgba8Unorm",
        )?;
        require(
            texture
                .usage()
                .contains(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING),
            "display needs sample/readback usages",
        )?;
        let tight = texture.width() * 4;
        let pitch = tight.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T-065 verification readback, outside application ledger"),
            size: u64::from(pitch) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(texture.height()),
                },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.finish()?;
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let bytes = mapped
            .chunks_exact(pitch as usize)
            .flat_map(|row| row[..tight as usize].iter().copied())
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(bytes)
    }
}
