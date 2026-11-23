//! Compile-time production boundary test, intentionally red until T-015 exists.
//!
//! ADR 0003 specifies shared Arc handles, but does not name the lower executor.
//! `GpuExecutor::new` is the proposed T-015 seam, not the T-017 worker API.
//! The implementer may adjust this spelling to the final API, retaining this
//! compile-time check that callers supply both handles without creating a device.
use std::{
    mem::{align_of, offset_of, size_of},
    sync::Arc,
};
use texelith_gpu::{BlurWeights, GpuExecutor, NodeUniform};

#[test]
fn production_executor_accepts_shared_device_and_queue() {
    let _constructor: fn(Arc<wgpu::Device>, Arc<wgpu::Queue>) -> _ = GpuExecutor::new;
}

#[test]
fn production_uniforms_match_the_adr_wire_layout() {
    fn pod<T: bytemuck::Pod + bytemuck::Zeroable>() {}
    pod::<NodeUniform>();
    pod::<BlurWeights>();
    assert_eq!(size_of::<NodeUniform>(), 64);
    assert_eq!(align_of::<NodeUniform>(), 16);
    assert_eq!(offset_of!(NodeUniform, dimensions), 0);
    assert_eq!(offset_of!(NodeUniform, parameters), 16);
    assert_eq!(offset_of!(NodeUniform, flags), 32);
    assert_eq!(offset_of!(NodeUniform, reserved), 48);
    assert_eq!(size_of::<BlurWeights>(), 128);
    assert_eq!(align_of::<BlurWeights>(), 16);
    let uniform = NodeUniform {
        dimensions: [17, 11, 34, 22],
        parameters: [0.25, 0.75, 2.0, 0.0],
        flags: [1, 3, 0, 0],
        reserved: [0; 4],
    };
    let bytes = bytemuck::bytes_of(&uniform);
    assert_eq!(&bytes[..4], &17u32.to_ne_bytes());
    assert_eq!(&bytes[16..20], &0.25f32.to_ne_bytes());
    assert_eq!(&bytes[32..36], &1u32.to_ne_bytes());
    assert_eq!(&bytes[48..], &[0; 16]);
}
