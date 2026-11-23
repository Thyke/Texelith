//! Original deterministic BC4 fitting primitives for BC5 blocks.
//!
//! The CPU fitter is a reference; `Bc5Kernel` supplies bounded GPU encoding and
//! normal-aware DDS mips to the immutable export worker's format adapter.

mod gpu;
pub use gpu::{Bc5Kernel, Bc5Quality, PreparedBc5, block_mips};

/// Encode two UNORM channels in row-major order as a BC5 block.
pub fn encode_block(pixels: &[[u8; 2]; 16]) -> [u8; 16] {
    let mut result = [0; 16];
    for channel in 0..2 {
        let values = pixels.map(|pixel| pixel[channel]);
        result[channel * 8..channel * 8 + 8].copy_from_slice(&encode_channel(&values));
    }
    result
}

fn encode_channel(values: &[u8; 16]) -> [u8; 8] {
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    // Search a bounded neighbourhood of the extrema. Include both BC4 palette
    // modes; strict comparison gives deterministic endpoint/index tie-breaking.
    let mut best_error = u64::MAX;
    let mut best = [0; 8];
    for a in min.saturating_sub(4)..=min.saturating_add(4) {
        for b in max.saturating_sub(4)..=max.saturating_add(4) {
            for (e0, e1) in [(a, b), (b, a)] {
                let mut palette = [0.0f32; 8];
                palette[0] = f32::from(e0);
                palette[1] = f32::from(e1);
                if e0 > e1 {
                    for i in 1..=6 {
                        palette[i + 1] =
                            ((7 - i) as f32 * palette[0] + i as f32 * palette[1]) / 7.0;
                    }
                } else {
                    for i in 1..=4 {
                        palette[i + 1] =
                            ((5 - i) as f32 * palette[0] + i as f32 * palette[1]) / 5.0;
                    }
                    palette[6] = 0.0;
                    palette[7] = 255.0;
                }
                let mut bits = 0u64;
                let mut error = 0u64;
                for (index, value) in values.iter().enumerate() {
                    let (choice, distance) = palette
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            let delta = (f32::from(*value) - p) * 35.0;
                            (i, (delta * delta).round() as u64)
                        })
                        .min_by_key(|&(_, distance)| distance)
                        .unwrap();
                    error += distance;
                    bits |= (choice as u64) << (3 * index);
                }
                if error < best_error {
                    best_error = error;
                    best[0] = e0;
                    best[1] = e1;
                    best[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
                }
            }
        }
    }
    best
}
