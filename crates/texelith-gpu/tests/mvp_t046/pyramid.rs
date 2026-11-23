use super::{
    gpu::{Gpu, with_gpu},
    oracle::{Image, compare},
};
use texelith_core::{Boundary, Extent, InputInterpretation, SourceDescriptor};
use texelith_gpu::{Axis, PyramidPass, pyramid_cache_key, pyramid_extents};

#[test]
fn ceil_half_extents_terminate_once_at_one_by_one_and_validate_before_arithmetic() {
    for (w, h, expected) in [
        (1, 1, vec![(1, 1)]),
        (5, 3, vec![(5, 3), (3, 2), (2, 1), (1, 1)]),
        (1, 7, vec![(1, 7), (1, 4), (1, 2), (1, 1)]),
        (7, 1, vec![(7, 1), (4, 1), (2, 1), (1, 1)]),
    ] {
        let actual = pyramid_extents(Extent::new(w, h).unwrap()).unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|e| (e.width, e.height))
                .collect::<Vec<_>>(),
            expected
        );
    }
    let levels = pyramid_extents(Extent::new(4096, 4096).unwrap()).unwrap();
    assert_eq!(levels.len(), 13, "G0 through G12, including the source");
    assert_eq!(levels.last(), Some(&Extent::new(1, 1).unwrap()));
    for bad in [(0, 1), (1, 0), (4097, 1), (u32::MAX, u32::MAX)] {
        assert!(
            pyramid_extents(Extent {
                width: bad.0,
                height: bad.1
            })
            .is_err()
        );
    }
}

#[test]
fn shared_keys_are_content_based_and_separate_extent_boundary_type_and_revision() {
    let source = SourceDescriptor {
        content_hash: [42; 32],
        extent: Extent::new(5, 3).unwrap(),
        interpretation: InputInterpretation::LinearHeight,
    };
    let key = pyramid_cache_key(&source, Boundary::Repeat, 1).unwrap();
    assert_eq!(
        key,
        pyramid_cache_key(&source.clone(), Boundary::Repeat, 1).unwrap()
    );
    assert_ne!(key, pyramid_cache_key(&source, Boundary::Clamp, 1).unwrap());
    assert_ne!(
        key,
        pyramid_cache_key(&source, Boundary::Repeat, 2).unwrap()
    );
    for changed in [
        SourceDescriptor {
            content_hash: [43; 32],
            ..source.clone()
        },
        SourceDescriptor {
            extent: Extent::new(3, 5).unwrap(),
            ..source.clone()
        },
        SourceDescriptor {
            interpretation: InputInterpretation::SrgbPhoto,
            ..source.clone()
        },
    ] {
        assert_ne!(
            key,
            pyramid_cache_key(&changed, Boundary::Repeat, 1).unwrap()
        );
    }
    let bad = SourceDescriptor {
        extent: Extent {
            width: 0,
            height: 1,
        },
        ..source
    };
    assert!(pyramid_cache_key(&bad, Boundary::Repeat, 1).is_err());
}

fn run(
    gpu: &Gpu,
    operation: PyramidPass,
    input: &wgpu::Texture,
    coarse: Option<&wgpu::Texture>,
    extent: Extent,
    boundary: Boundary,
) -> wgpu::Texture {
    let output = gpu.texture(extent, wgpu::TextureFormat::R32Float);
    let pass = gpu
        .executor
        .prepare_pyramid(operation, input, coarse, &output, boundary)
        .unwrap();
    gpu.dispatch(&pass);
    output
}

#[test]
fn gpu_normalized_filters_preserve_large_signed_finite_constants() {
    with_gpu(|gpu| {
        for (w, h) in [(1, 17), (17, 1), (9, 7), (257, 193)] {
            for value in [2.0_f32.powi(124), -2.0_f32.powi(124)] {
                let image = Image::new(w, h, vec![f64::from(value); (w * h) as usize]);
                let input = gpu.scalar(&image);
                let half = Extent::new(w.div_ceil(2), h.div_ceil(2)).unwrap();
                for boundary in [Boundary::Clamp, Boundary::Repeat] {
                    for op in [
                        PyramidPass::Gaussian(Axis::X),
                        PyramidPass::Gaussian(Axis::Y),
                        PyramidPass::AreaDownsample,
                        PyramidPass::GaussianAreaDownsample,
                    ] {
                        let extent = if matches!(op, PyramidPass::Gaussian(_)) {
                            image.extent
                        } else {
                            half
                        };
                        let result = run(gpu, op, &input, None, extent, boundary);
                        for actual in gpu.read_scalar(&result) {
                            assert!(actual.is_finite(), "{op:?} overflowed {value}");
                            assert!(
                                ((actual - value) / value).abs() <= 2e-6,
                                "{op:?} changed constant {value} to {actual}"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn gpu_primitives_match_hand_calculated_odd_area_and_wrapped_pixel_centers() {
    with_gpu(|gpu| {
        let fine = Image::new(5, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let input = gpu.scalar(&fine);
        let coarse = Image::new(3, 1, vec![0.0, 10.0, 20.0]);
        let coarse_texture = gpu.scalar(&coarse);
        for boundary in [Boundary::Clamp, Boundary::Repeat] {
            let reduced = run(
                gpu,
                PyramidPass::AreaDownsample,
                &input,
                None,
                coarse.extent,
                boundary,
            );
            compare(&gpu.read_scalar(&reduced), &[0.4, 2.0, 3.6], 2e-5);
            let blurred = run(
                gpu,
                PyramidPass::Gaussian(Axis::X),
                &input,
                None,
                fine.extent,
                boundary,
            );
            compare(
                &gpu.read_scalar(&blurred),
                &fine.gaussian(false, boundary).pixels,
                2e-5,
            );
            let expanded = run(
                gpu,
                PyramidPass::BilinearUpsample,
                &coarse_texture,
                None,
                fine.extent,
                boundary,
            );
            compare(
                &gpu.read_scalar(&expanded),
                &coarse.expand(fine.extent, boundary).pixels,
                2e-5,
            );
            let band = run(
                gpu,
                PyramidPass::LaplacianBand,
                &input,
                Some(&coarse_texture),
                fine.extent,
                boundary,
            );
            let removed = run(
                gpu,
                PyramidPass::WeightedReconstruct {
                    band_weight: 1.0,
                    coarse_weight: 0.0,
                },
                &band,
                Some(&coarse_texture),
                fine.extent,
                boundary,
            );
            let expected_band: Vec<f64> = fine
                .pixels
                .iter()
                .zip(coarse.expand(fine.extent, boundary).pixels)
                .map(|(f, c)| f - c)
                .collect();
            compare(&gpu.read_scalar(&removed), &expected_band, 2e-5);
        }
    });
}

#[test]
fn gpu_all_levels_and_laplacian_reconstruction_match_independent_oracles() {
    with_gpu(|gpu| {
        for (w, h, kind) in [
            (1, 1, "constant"),
            (1, 17, "ramp"),
            (17, 1, "ramp"),
            (5, 3, "impulse"),
            (8, 8, "impulse"),
            (9, 7, "constant"),
            (257, 193, "seeded"),
            (31, 17, "ramp"),
        ] {
            let mut state = 42u32;
            let pixels: Vec<f64> = (0..w * h)
                .map(|i| match kind {
                    "constant" => -0.375,
                    "impulse" => {
                        if i == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    "ramp" => f64::from(i) / f64::from((w * h).max(2) - 1) - 0.5,
                    _ => {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        f64::from((state >> 16) as i16) / 32768.0
                    }
                })
                .collect();
            for boundary in [Boundary::Clamp, Boundary::Repeat] {
                let mut refs = vec![Image::new(w, h, pixels.clone())];
                let mut levels = vec![gpu.scalar(&refs[0])];
                while refs.last().unwrap().extent != Extent::new(1, 1).unwrap() {
                    let last = refs.last().unwrap();
                    let horizontal_ref = last.gaussian(false, boundary);
                    let vertical_ref = horizontal_ref.gaussian(true, boundary);
                    let coarse_ref = vertical_ref.reduce();
                    let horizontal = run(
                        gpu,
                        PyramidPass::Gaussian(Axis::X),
                        levels.last().unwrap(),
                        None,
                        last.extent,
                        boundary,
                    );
                    compare(&gpu.read_scalar(&horizontal), &horizontal_ref.pixels, 2e-5);
                    let vertical = run(
                        gpu,
                        PyramidPass::Gaussian(Axis::Y),
                        &horizontal,
                        None,
                        last.extent,
                        boundary,
                    );
                    compare(&gpu.read_scalar(&vertical), &vertical_ref.pixels, 2e-5);
                    let reduced = run(
                        gpu,
                        PyramidPass::AreaDownsample,
                        &vertical,
                        None,
                        coarse_ref.extent,
                        boundary,
                    );
                    compare(&gpu.read_scalar(&reduced), &coarse_ref.pixels, 2e-5);
                    refs.push(coarse_ref);
                    levels.push(reduced);
                }
                assert_eq!(levels.len(), pyramid_extents(refs[0].extent).unwrap().len());
                let mut bands = Vec::new();
                for l in 0..levels.len() - 1 {
                    let fine = &refs[l];
                    let up = refs[l + 1].expand(fine.extent, boundary);
                    let expected: Vec<f64> = fine
                        .pixels
                        .iter()
                        .zip(&up.pixels)
                        .map(|(a, b)| a - b)
                        .collect();
                    let band = run(
                        gpu,
                        PyramidPass::LaplacianBand,
                        &levels[l],
                        Some(&levels[l + 1]),
                        fine.extent,
                        boundary,
                    );
                    compare(&gpu.read_scalar(&band), &expected, 2e-5);
                    // A changed weight must change the result, and signed values
                    // must survive. This catches an unconditional input copy.
                    let changed = run(
                        gpu,
                        PyramidPass::WeightedReconstruct {
                            band_weight: 0.25,
                            coarse_weight: 0.5,
                        },
                        &band,
                        Some(&levels[l + 1]),
                        fine.extent,
                        boundary,
                    );
                    let weighted: Vec<f64> = expected
                        .iter()
                        .zip(up.pixels)
                        .map(|(b, c)| 0.25 * b + 0.5 * c)
                        .collect();
                    compare(&gpu.read_scalar(&changed), &weighted, 2e-5);
                    bands.push(band);
                }
                let mut reconstructed = levels.last().unwrap().clone();
                for l in (0..bands.len()).rev() {
                    reconstructed = run(
                        gpu,
                        PyramidPass::WeightedReconstruct {
                            band_weight: 1.0,
                            coarse_weight: 1.0,
                        },
                        &bands[l],
                        Some(&reconstructed),
                        refs[l].extent,
                        boundary,
                    );
                    compare(&gpu.read_scalar(&reconstructed), &refs[l].pixels, 2e-5);
                }
                let first = gpu.read_scalar(&reconstructed);
                compare(&first, &pixels, 2e-5);
                // Identical dispatches on one backend must be deterministic.
                let again = run(
                    gpu,
                    PyramidPass::Gaussian(Axis::X),
                    &levels[0],
                    None,
                    refs[0].extent,
                    boundary,
                );
                let again2 = run(
                    gpu,
                    PyramidPass::Gaussian(Axis::X),
                    &levels[0],
                    None,
                    refs[0].extent,
                    boundary,
                );
                assert_eq!(gpu.read(&again), gpu.read(&again2));
                if kind == "constant" {
                    for level in &levels {
                        compare(
                            &gpu.read_scalar(level),
                            &vec![-0.375; (level.width() * level.height()) as usize],
                            2e-5,
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn gpu_rejects_aliases_bad_extents_formats_and_nonfinite_weights_before_encoding() {
    with_gpu(|gpu| {
        let fine = gpu.texture(Extent::new(5, 3).unwrap(), wgpu::TextureFormat::R32Float);
        let coarse = gpu.texture(Extent::new(3, 2).unwrap(), wgpu::TextureFormat::R32Float);
        let wrong_size = gpu.texture(Extent::new(2, 2).unwrap(), wgpu::TextureFormat::R32Float);
        let wrong_format =
            gpu.texture(Extent::new(5, 3).unwrap(), wgpu::TextureFormat::Rgba16Float);
        let out = gpu.texture(Extent::new(5, 3).unwrap(), wgpu::TextureFormat::R32Float);
        let prepare = |op, input, extra, output| {
            gpu.executor
                .prepare_pyramid(op, input, extra, output, Boundary::Repeat)
        };
        assert!(prepare(PyramidPass::Gaussian(Axis::X), &fine, None, &fine).is_err());
        assert!(prepare(PyramidPass::AreaDownsample, &fine, None, &wrong_size).is_err());
        assert!(prepare(PyramidPass::BilinearUpsample, &wrong_size, None, &out).is_err());
        assert!(prepare(PyramidPass::Gaussian(Axis::Y), &fine, None, &wrong_format).is_err());
        assert!(prepare(PyramidPass::Gaussian(Axis::Y), &wrong_format, None, &out).is_err());
        assert!(prepare(PyramidPass::LaplacianBand, &fine, None, &out).is_err());
        assert!(prepare(PyramidPass::LaplacianBand, &fine, Some(&wrong_size), &out).is_err());
        assert!(prepare(PyramidPass::LaplacianBand, &fine, Some(&coarse), &fine).is_err());
        let singleton_input =
            gpu.texture(Extent::new(1, 1).unwrap(), wgpu::TextureFormat::R32Float);
        let singleton_output =
            gpu.texture(Extent::new(1, 1).unwrap(), wgpu::TextureFormat::R32Float);
        assert!(
            prepare(
                PyramidPass::LaplacianBand,
                &singleton_input,
                Some(&singleton_output),
                &singleton_output,
            )
            .is_err(),
            "the additional read input must not alias the output either"
        );
        assert!(prepare(PyramidPass::Gaussian(Axis::X), &fine, Some(&coarse), &out).is_err());
        let no_read_usage = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("missing texture binding"),
            size: fine.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let no_write_usage = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("missing storage binding"),
            size: fine.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        assert!(prepare(PyramidPass::Gaussian(Axis::X), &no_read_usage, None, &out).is_err());
        assert!(prepare(PyramidPass::Gaussian(Axis::X), &fine, None, &no_write_usage).is_err());
        for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for op in [
                PyramidPass::WeightedReconstruct {
                    band_weight: weight,
                    coarse_weight: 1.0,
                },
                PyramidPass::WeightedReconstruct {
                    band_weight: 1.0,
                    coarse_weight: weight,
                },
            ] {
                assert!(prepare(op, &fine, Some(&coarse), &out).is_err());
            }
        }
    });
}
