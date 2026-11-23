#[path = "common/gpu_availability.rs"]
mod gpu_availability;

use texelith_gpu::bc5::encode_block;

#[path = "../examples/mvp_support/t081/checks.rs"]
mod checks;

#[test]
fn gpu_bc5_mips_and_live_export_contract() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("T-081 requires native GPU hardware");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("T-081 adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    checks::gpu_checks(device.clone(), queue.clone()).unwrap();
    checks::worker_checks(device, queue).unwrap();
}

// Independent BC4 decoder, including both endpoint modes and little-endian indices.
fn decode(block: &[u8]) -> [f32; 16] {
    let a = f32::from(block[0]);
    let b = f32::from(block[1]);
    let mut table = [a, b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    if block[0] > block[1] {
        for n in 1..7 {
            table[n + 1] = ((7 - n) as f32 * a + n as f32 * b) / 7.0;
        }
    } else {
        for n in 1..5 {
            table[n + 1] = ((5 - n) as f32 * a + n as f32 * b) / 5.0;
        }
        table[7] = 255.0;
    }
    let mut indices = 0u64;
    for (i, byte) in block[2..8].iter().enumerate() {
        indices |= u64::from(*byte) << (i * 8);
    }
    std::array::from_fn(|i| table[((indices >> (3 * i)) & 7) as usize])
}

#[test]
fn independent_decode_constant_and_gradient_blocks() {
    for pixels in [
        [[128, 64]; 16],
        std::array::from_fn(|i| [40 + (i % 4) as u8 * 40, 30 + (i / 4) as u8 * 50]),
    ] {
        let encoded = encode_block(&pixels);
        assert_eq!(encoded, encode_block(&pixels));
        let r = decode(&encoded[..8]);
        let g = decode(&encoded[8..]);
        for i in 0..16 {
            assert!((r[i] - f32::from(pixels[i][0])).abs() <= 10.0);
            assert!((g[i] - f32::from(pixels[i][1])).abs() <= 10.0);
        }
    }
}

#[test]
fn decoder_checks_both_endpoint_orders_and_index_bit_order() {
    let mut encoded = [0u8; 8];
    encoded[0] = 200;
    encoded[1] = 10;
    let indices = (0..16).fold(0u64, |v, i| v | ((i % 8) as u64) << (3 * i));
    encoded[2..].copy_from_slice(&indices.to_le_bytes()[..6]);
    let decoded = decode(&encoded);
    assert_eq!(decoded[0], 200.0);
    assert_eq!(decoded[1], 10.0);
    assert!(decoded[2] > decoded[7]);
    encoded[0] = 10;
    encoded[1] = 200;
    let decoded = decode(&encoded);
    assert_eq!(decoded[6], 0.0);
    assert_eq!(decoded[7], 255.0);
}
