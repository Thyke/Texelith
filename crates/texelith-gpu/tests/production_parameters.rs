mod support;

use support::{abi, oracle};
use texelith_core::{Extent, InputInterpretation, MapKind};
use texelith_gpu::{Axis, BlurWeights, Pass, proxy_extent, workgroups};

fn extent(width: u32, height: u32) -> Extent {
    Extent::new(width, height).unwrap()
}

#[test]
fn production_dispatch_and_proxy_match_independent_contract() {
    for e in [
        extent(1, 1),
        extent(1, 4096),
        extent(4096, 1),
        extent(17, 11),
        extent(2049, 1001),
        extent(4096, 4096),
    ] {
        assert_eq!(workgroups(e).unwrap(), abi::workgroups(e).unwrap());
        assert_eq!(proxy_extent(e).unwrap(), oracle::proxy_extent(e));
    }
    for e in [
        Extent {
            width: 0,
            height: 1,
        },
        Extent {
            width: 1,
            height: 4097,
        },
        Extent {
            width: u32::MAX,
            height: u32::MAX,
        },
    ] {
        assert!(workgroups(e).is_err());
        assert!(proxy_extent(e).is_err());
    }
}

#[test]
fn production_gaussians_match_f64_reference_including_padding() {
    for sigma in [0.0, 0.01, 0.25, 1.0, 2.5, 5.0, 10.0] {
        let expected = oracle::gaussian(sigma);
        let (actual, radius) = BlurWeights::gaussian(sigma).unwrap();
        assert_eq!(radius as usize * 2 + 1, expected.len());
        assert_eq!(
            bytemuck::bytes_of(&actual),
            bytemuck::bytes_of(&abi::BlurWeights::new(&expected))
        );
    }
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(BlurWeights::gaussian(invalid).is_err());
    }
}

#[test]
fn production_axis_scaling_and_slot_packing_match_independent_abi() {
    let e = extent(17, 11);
    let source = extent(34, 44);
    for (axis, sigma, flag) in [(Axis::X, 1.0, 0), (Axis::Y, 0.5, 1)] {
        let (uniform, weights) = Pass::Blur {
            sigma_source_px: 2.0,
            axis,
        }
        .uniforms(e, source)
        .unwrap();
        let reference = oracle::gaussian(sigma);
        let mut expected = abi::NodeUniform::new(e, source);
        expected.parameters[0] = sigma as f32;
        expected.flags = [flag, reference.len() as u32 / 2, 0, 0];
        assert_eq!(bytemuck::bytes_of(&uniform), bytemuck::bytes_of(&expected));
        assert_eq!(
            bytemuck::bytes_of(&weights.unwrap()),
            bytemuck::bytes_of(&abi::BlurWeights::new(&reference))
        );
    }
    assert!(
        Pass::Blur {
            sigma_source_px: 0.0,
            axis: Axis::X
        }
        .is_identity()
    );
    assert!(
        !Pass::Blur {
            sigma_source_px: 0.01,
            axis: Axis::X
        }
        .is_identity()
    );
}

#[test]
fn production_passes_pack_flags_parameters_and_formats() {
    let e = extent(17, 11);
    for (pass, params, flags, format) in [
        (
            Pass::Source(InputInterpretation::SrgbPhoto),
            [0.0; 4],
            [0; 4],
            wgpu::TextureFormat::R32Float,
        ),
        (
            Pass::Source(InputInterpretation::LinearHeight),
            [0.0; 4],
            [1, 0, 0, 0],
            wgpu::TextureFormat::R32Float,
        ),
        (
            Pass::Levels {
                black: 0.25,
                white: 0.75,
                gamma: 2.0,
            },
            [0.25, 0.75, 2.0, 0.0],
            [0; 4],
            wgpu::TextureFormat::R32Float,
        ),
        (
            Pass::Normal { strength: 3.0 },
            [3.0, 0.0, 0.0, 0.0],
            [0; 4],
            wgpu::TextureFormat::Rgba16Float,
        ),
        (
            Pass::Display(MapKind::Height),
            [0.0; 4],
            [0; 4],
            wgpu::TextureFormat::Rgba8Unorm,
        ),
        (
            Pass::Display(MapKind::Normal),
            [0.0; 4],
            [1, 0, 0, 0],
            wgpu::TextureFormat::Rgba8Unorm,
        ),
    ] {
        let (uniform, weights) = pass.uniforms(e, e).unwrap();
        assert_eq!(uniform.parameters, params);
        assert_eq!(uniform.flags, flags);
        assert_eq!(uniform.reserved, [0; 4]);
        assert_eq!(pass.output_format(), format);
        assert!(weights.is_none());
    }
}

#[test]
fn production_validation_rejects_bad_parameters_before_gpu_work() {
    let e = extent(17, 11);
    for pass in [
        Pass::Levels {
            black: 0.5,
            white: 0.5,
            gamma: 1.0,
        },
        Pass::Levels {
            black: 0.0,
            white: 1.0,
            gamma: 0.0,
        },
        Pass::Levels {
            black: f32::NAN,
            white: 1.0,
            gamma: 1.0,
        },
        Pass::Normal {
            strength: f32::INFINITY,
        },
        Pass::Normal { strength: -1.0 },
        Pass::Blur {
            sigma_source_px: -1.0,
            axis: Axis::X,
        },
        Pass::Blur {
            sigma_source_px: 5.01,
            axis: Axis::Y,
        },
    ] {
        assert!(pass.uniforms(e, e).is_err());
    }
    assert!(Pass::Downsample.uniforms(extent(18, 11), e).is_err());
    assert!(
        Pass::Source(InputInterpretation::SrgbPhoto)
            .uniforms(extent(8, 5), e)
            .is_err()
    );
}
