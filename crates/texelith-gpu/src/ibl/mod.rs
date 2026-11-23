//! Deterministic, budgeted environment lighting (ADR 0008).
//! Cubes use six horizontal faces; specular mip rows are stacked vertically.
//! This keeps every byte in the existing 2D allocation ledger. Sampling
//! reprojects bilinear taps across face edges rather than clamping a face.
mod kernel;
use crate::{GpuError, pool::AllocationDesc};
pub use kernel::{IblKernel, PreparedIbl};
use std::sync::Arc;
use texelith_core::Extent;

pub const CUBE_SIZE: u32 = 256;
pub const DIFFUSE_SIZE: u32 = 32;
pub const SPECULAR_SIZE: u32 = 128;
pub const LUT_SIZE: u32 = 128;
pub const SPECULAR_SAMPLES: u32 = 128;
pub const DIFFUSE_SAMPLES: u32 = 64;
pub const MIP_COUNT: u32 = 8;

/// Validated immutable linear RGB pixels. Identity hashes canonical content,
/// dimensions and the fixed algorithm revision, never path or generation.
#[derive(Clone, Debug)]
pub struct Environment {
    extent: Extent,
    pixels: Arc<[[f32; 4]]>,
    key: [u64; 2],
}
impl Environment {
    pub fn new(extent: Extent, rgb: &[[f32; 3]]) -> Result<Self, GpuError> {
        if extent.width == 0
            || extent.height == 0
            || extent.width > 2048
            || extent.height > 1024
            || rgb.len() != extent.width as usize * extent.height as usize
            || rgb
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0 || *v > 65504.0)
        {
            return Err(GpuError(
                "environment requires bounded finite nonnegative linear HDR (<=65504)".into(),
            ));
        }
        let pixels: Arc<[[f32; 4]]> = rgb.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
        let mut key = [0xcbf29ce484222325u64, 0x84222325cbf29ce4u64];
        for byte in extent
            .width
            .to_le_bytes()
            .into_iter()
            .chain(extent.height.to_le_bytes())
            .chain([1])
            .chain(pixels.iter().flat_map(|p| {
                p.iter()
                    .flat_map(|v| (if *v == 0.0 { 0.0 } else { *v }).to_bits().to_le_bytes())
            }))
        {
            key[0] = (key[0] ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            key[1] = (key[1] ^ u64::from(byte)).wrapping_mul(0x9e3779b185ebca87);
        }
        Ok(Self {
            extent,
            pixels,
            key,
        })
    }
    pub fn extent(&self) -> Extent {
        self.extent
    }
    pub fn key(&self) -> [u64; 2] {
        self.key
    }
    pub fn pixels(&self) -> &[[f32; 4]] {
        &self.pixels
    }
    pub fn procedural() -> Self {
        let extent = Extent {
            width: 256,
            height: 128,
        };
        let pixels = (0..128)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    let elevation = ((y as f32 + 0.5) / 128.0 * std::f32::consts::PI).cos();
                    // Extend the soft studio horizon below eye level so the default
                    // downward inspection camera still sees blue atmosphere.
                    let atmosphere = ((elevation + 0.65) / 1.1).clamp(0., 1.).sqrt();
                    let ground = ((-elevation - 0.65) / 0.35).clamp(0., 1.);
                    let longitude =
                        (x as f32 + 0.5) / 256. * std::f32::consts::TAU - std::f32::consts::PI;
                    let horizontal = (1. - elevation * elevation).max(0.).sqrt();
                    let sun_elevation: f32 = -0.12;
                    let sun_longitude: f32 = -1.98;
                    let alignment = (horizontal
                        * (1. - sun_elevation * sun_elevation).sqrt()
                        * (longitude - sun_longitude).cos()
                        + elevation * sun_elevation)
                        .max(0.);
                    let sun = 7. * alignment.powf(4000.) + 0.35 * alignment.powf(48.);
                    let horizon = [0.72, 0.70, 0.62];
                    let zenith = [0.035, 0.17, 0.52];
                    let floor = [0.09, 0.12, 0.16];
                    let warmth = [1., 0.68, 0.34];
                    std::array::from_fn(|channel| {
                        let sky =
                            horizon[channel] * (1. - atmosphere) + zenith[channel] * atmosphere;
                        sky * (1. - ground) + floor[channel] * ground + sun * warmth[channel]
                    })
                })
            })
            .collect::<Vec<_>>();
        Self::new(extent, &pixels).expect("finite procedural sky")
    }
}

/// Output order: converted cube, irradiance E/pi, specular chain, RG BRDF LUT.
/// Irradiance is divided by pi so a unit environment remains exactly unit.
pub fn output_allocations() -> [AllocationDesc; 4] {
    [(1536, 256), (192, 32), (768, 255), (128, 128)].map(|(width, height)| {
        AllocationDesc::Texture {
            extent: Extent { width, height },
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        }
    })
}
/// Fixed, cancellable work units: conversion faces, diffuse faces, specular
/// face/mips, LUT. One completed submission per item before publication.
pub fn passes() -> impl Iterator<Item = (u32, u32, u32, u32)> {
    (0..6)
        .map(|f| (0, f, 0, 256))
        .chain((0..6).map(|f| (1, f, 0, 32)))
        .chain((0..8).flat_map(|m| (0..6).map(move |f| (2, f, m, 128 >> m))))
        .chain(std::iter::once((3, 0, 0, 128)))
}

/// Sampling pair used for fractional roughness, with no integer mip snapping.
pub fn roughness_lod(roughness: f32) -> (u32, u32, f32) {
    let lod = if roughness.is_finite() {
        roughness.clamp(0.0, 1.0) * 7.0
    } else {
        0.0
    };
    (
        lod.floor() as u32,
        (lod.floor() as u32 + 1).min(7),
        lod.fract(),
    )
}
