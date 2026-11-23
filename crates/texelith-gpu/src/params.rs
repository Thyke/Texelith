use crate::GpuError;
use bytemuck::{Pod, Zeroable};
use texelith_core::{Extent, InputInterpretation, MapKind, NodeParams};

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NodeUniform {
    pub dimensions: [u32; 4],
    pub parameters: [f32; 4],
    pub flags: [u32; 4],
    pub reserved: [u32; 4],
}

impl NodeUniform {
    pub fn new(evaluation: Extent, source: Extent) -> Self {
        Self {
            dimensions: [
                evaluation.width,
                evaluation.height,
                source.width,
                source.height,
            ],
            ..Self::zeroed()
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlurWeights {
    pub slots: [[f32; 4]; 8],
}

const _: () = {
    assert!(std::mem::size_of::<NodeUniform>() == 64);
    assert!(std::mem::align_of::<NodeUniform>() == 16);
    assert!(std::mem::size_of::<BlurWeights>() == 128);
    assert!(std::mem::align_of::<BlurWeights>() == 16);
};

impl BlurWeights {
    /// Packed from -radius to +radius; all unused lanes are zero.
    pub fn gaussian(sigma: f64) -> Result<(Self, u32), GpuError> {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(GpuError("sigma must be finite and nonnegative".into()));
        }
        let mut result = Self::zeroed();
        if sigma == 0.0 {
            result.slots[0][0] = 1.0;
            return Ok((result, 0));
        }
        let radius = (3.0 * sigma).ceil().min(15.0) as u32;
        let mut weights = [0.0f64; 31];
        let count = (radius * 2 + 1) as usize;
        for (i, weight) in weights[..count].iter_mut().enumerate() {
            let distance = (i as f64 - f64::from(radius)) / sigma;
            *weight = (-0.5 * distance * distance).exp();
        }
        let sum: f64 = weights[..count].iter().sum();
        for (i, weight) in weights[..count].iter().enumerate() {
            result.slots[i / 4][i % 4] = (weight / sum) as f32;
        }
        Ok((result, radius))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
}

/// Support passes are executor operations, not additional graph node kinds.
#[derive(Clone, Copy, Debug)]
pub enum Pass {
    ColorSource,
    NormalSource(texelith_core::NormalConvention),
    Source(InputInterpretation),
    Downsample,
    Levels { black: f32, white: f32, gamma: f32 },
    Blur { sigma_source_px: f32, axis: Axis },
    Normal { strength: f32 },
    Display(MapKind),
}

impl Pass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ColorSource => "color_source",
            Self::NormalSource(_) => "normal_source",
            Self::Source(_) => "source",
            Self::Downsample => "downsample",
            Self::Levels { .. } => "levels",
            Self::Blur { .. } => "blur",
            Self::Normal { .. } => "normal",
            Self::Display(_) => "display",
        }
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::ColorSource => 6,
            Self::NormalSource(_) => 7,
            Self::Source(_) => 0,
            Self::Downsample => 1,
            Self::Levels { .. } => 2,
            Self::Blur { .. } => 3,
            Self::Normal { .. } => 4,
            Self::Display(_) => 5,
        }
    }

    /// A scheduler can alias the input for zero blur without allocating/encoding.
    pub const fn is_identity(self) -> bool {
        matches!(
            self,
            Self::Blur {
                sigma_source_px: 0.0,
                ..
            }
        )
    }

    pub const fn input_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Source(_) | Self::ColorSource | Self::NormalSource(_) => {
                wgpu::TextureFormat::Rgba8Unorm
            }
            Self::Display(MapKind::Normal) => wgpu::TextureFormat::Rgba16Float,
            _ => wgpu::TextureFormat::R32Float,
        }
    }

    pub const fn output_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Normal { .. } | Self::ColorSource | Self::NormalSource(_) => {
                wgpu::TextureFormat::Rgba16Float
            }
            Self::Display(_) => wgpu::TextureFormat::Rgba8Unorm,
            _ => wgpu::TextureFormat::R32Float,
        }
    }

    /// Validates graph parameters and computes axis-specific weights once per update.
    pub fn uniforms(
        self,
        evaluation: Extent,
        source: Extent,
    ) -> Result<(NodeUniform, Option<BlurWeights>), GpuError> {
        evaluation.validate()?;
        source.validate()?;
        if evaluation.width > source.width || evaluation.height > source.height {
            return Err(GpuError("evaluation cannot exceed source extent".into()));
        }
        let mut uniform = NodeUniform::new(evaluation, source);
        let mut weights = None;
        match self {
            Self::ColorSource | Self::NormalSource(_) => {
                if evaluation != source {
                    return Err(GpuError(
                        "typed source conversion requires full extent".into(),
                    ));
                }
                uniform.flags[0] = u32::from(matches!(
                    self,
                    Self::NormalSource(texelith_core::NormalConvention::DirectX)
                ));
            }
            Self::Source(interpretation) => {
                if evaluation != source {
                    return Err(GpuError("prepare source before downsampling".into()));
                }
                uniform.flags[0] = u32::from(interpretation == InputInterpretation::LinearHeight);
            }
            Self::Downsample => {}
            Self::Levels {
                black,
                white,
                gamma,
            } => {
                NodeParams::Levels {
                    black,
                    white,
                    gamma,
                }
                .validate()?;
                uniform.parameters = [black, white, gamma, 0.0];
            }
            Self::Blur {
                sigma_source_px,
                axis,
            } => {
                NodeParams::Blur { sigma_source_px }.validate()?;
                let (size, original) = match axis {
                    Axis::X => (evaluation.width, source.width),
                    Axis::Y => (evaluation.height, source.height),
                };
                let sigma = f64::from(sigma_source_px) * f64::from(size) / f64::from(original);
                let (block, radius) = BlurWeights::gaussian(sigma)?;
                uniform.parameters[0] = sigma as f32;
                uniform.flags = [u32::from(axis == Axis::Y), radius, 0, 0];
                weights = Some(block);
            }
            Self::Normal { strength } => {
                NodeParams::NormalFromHeight { strength }.validate()?;
                uniform.parameters[0] = strength;
            }
            Self::Display(kind) => uniform.flags[0] = u32::from(kind == MapKind::Normal),
        }
        Ok((uniform, weights))
    }
}

pub fn workgroups(extent: Extent) -> Result<[u32; 3], GpuError> {
    extent.validate()?;
    Ok([extent.width.div_ceil(8), extent.height.div_ceil(8), 1])
}

/// Integer floor preserves aspect to within one pixel and never enlarges input.
pub fn proxy_extent(source: Extent) -> Result<Extent, GpuError> {
    source.validate()?;
    let longest = source.width.max(source.height);
    if longest <= 1024 {
        return Ok(source);
    }
    Ok(Extent::new(
        (source.width * 1024 / longest).max(1),
        (source.height * 1024 / longest).max(1),
    )?)
}
