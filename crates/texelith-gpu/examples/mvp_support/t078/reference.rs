//! Original f64 export oracle, shared only by tests and the measurement harness.
//! No production sampling, transfer, normal or quantization helpers are used.
use texelith_core::{
    Boundary, ChannelSource, Component, ExportFormat, ExportPlan, Extent, IntegerDither,
    NormalConvention, OutputPort, OutputRef, SizePolicy, Transfer,
};

#[derive(Clone)]
pub struct Source {
    pub reference: OutputRef,
    pub extent: Extent,
    /// Linear color, signed normal xyz, or scalar in R. Alpha is not source data.
    pub pixels: Vec<[f64; 4]>,
    pub boundary: Boundary,
    pub normal_convention: NormalConvention,
}

pub fn destination(plan: &ExportPlan) -> Extent {
    match plan.size {
        SizePolicy::Keep => plan.source_extent,
        SizePolicy::Resize { width, height } | SizePolicy::Pad { width, height } => {
            Extent { width, height }
        }
    }
}

fn address(index: i64, size: u32, boundary: Boundary) -> u32 {
    match boundary {
        Boundary::Clamp => index.clamp(0, i64::from(size) - 1) as u32,
        Boundary::Repeat => index.rem_euclid(i64::from(size)) as u32,
    }
}

// A shrinking axis integrates the entire output footprint. A growing axis
// interpolates source pixel centers. The product also handles mixed-axis resize.
fn axis(pixel: u32, input: u32, output: u32) -> Vec<(i64, f64)> {
    if input > output {
        let lo = f64::from(pixel) * f64::from(input) / f64::from(output);
        let hi = f64::from(pixel + 1) * f64::from(input) / f64::from(output);
        (lo.floor() as i64..hi.ceil() as i64)
            .map(|i| (i, (hi.min((i + 1) as f64) - lo.max(i as f64)) / (hi - lo)))
            .collect()
    } else {
        let center = (f64::from(pixel) + 0.5) * f64::from(input) / f64::from(output) - 0.5;
        let lo = center.floor();
        vec![
            (lo as i64, 1.0 - (center - lo)),
            (lo as i64 + 1, center - lo),
        ]
    }
}

pub fn sample(source: &Source, plan: &ExportPlan, x: u32, y: u32) -> [f64; 4] {
    if !matches!(plan.size, SizePolicy::Resize { .. }) {
        let boundary = if matches!(plan.size, SizePolicy::Pad { .. }) {
            plan.padding
        } else {
            source.boundary
        };
        return source.pixels[(address(y.into(), source.extent.height, boundary)
            * source.extent.width
            + address(x.into(), source.extent.width, boundary))
            as usize];
    }
    let target = destination(plan);
    let mut sum = [0.0; 4];
    for (sy, wy) in axis(y, source.extent.height, target.height) {
        for (sx, wx) in axis(x, source.extent.width, target.width) {
            let pixel = source.pixels[(address(sy, source.extent.height, source.boundary)
                * source.extent.width
                + address(sx, source.extent.width, source.boundary))
                as usize];
            for c in 0..4 {
                sum[c] += pixel[c] * wx * wy;
            }
        }
    }
    sum
}

pub fn srgb(value: f64) -> f64 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

pub fn quantize(
    value: f64,
    bits: u8,
    clamp: [f64; 2],
    dither: IntegerDither,
    x: u32,
    y: u32,
) -> f64 {
    const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    let max = f64::from((1u32 << bits) - 1);
    let offset = if dither == IntegerDither::Ordered4x4 {
        (f64::from(BAYER[(y % 4) as usize][(x % 4) as usize]) + 0.5) / 16.0 - 0.5
    } else {
        0.0
    };
    ((value.clamp(clamp[0], clamp[1]) * max + offset).clamp(0.0, max) + 0.5).floor() / max
}

pub fn evaluate(plan: &ExportPlan, index: usize, sources: &[Source]) -> Vec<[f64; 4]> {
    let output = &plan.outputs[index];
    let target = destination(plan);
    let mut result = Vec::new();
    for y in 0..target.height {
        for x in 0..target.width {
            let mut pixel = [0.0, 0.0, 0.0, 1.0];
            for (c, channel) in output.channels.iter().enumerate() {
                let mut value = match channel.source {
                    ChannelSource::Constant(value) => f64::from(value),
                    ChannelSource::Map(reference, component) => {
                        if reference.port == OutputPort::Albedo && component == Component::A {
                            1.0
                        } else {
                            let source = sources
                                .iter()
                                .find(|s| s.reference == reference)
                                .expect("oracle source");
                            let mut value = sample(source, plan, x, y);
                            if reference.port == OutputPort::Normal {
                                let length = value[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
                                assert!(length > 1e-12, "normal fixture must be nondegenerate");
                                for v in &mut value[..3] {
                                    *v /= length;
                                }
                                if source.normal_convention != plan.profile.normal_convention {
                                    value[1] = -value[1];
                                }
                                for v in &mut value[..3] {
                                    *v = *v * 0.5 + 0.5;
                                }
                            }
                            value[match component {
                                Component::R => 0,
                                Component::G => 1,
                                Component::B => 2,
                                Component::A => 3,
                            }]
                        }
                    }
                };
                if channel.invert {
                    value = 1.0 - value;
                }
                if channel.transfer == Transfer::Srgb {
                    value = srgb(value);
                }
                if output.format == ExportFormat::Exr32 {
                    value = (value - f64::from(plan.displacement.midlevel))
                        * f64::from(plan.displacement.scale);
                } else {
                    let q = output.quantization.unwrap_or_default();
                    value = quantize(
                        value,
                        output.format.bit_depth(),
                        [q.clamp_min.into(), q.clamp_max.into()],
                        q.dither,
                        x,
                        y,
                    );
                }
                pixel[c] = value;
            }
            result.push(pixel);
        }
    }
    result
}
