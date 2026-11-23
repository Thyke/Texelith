use super::{
    reference::{self, Combine, Controls, compare, photo, photo_reference},
    support::{GIB, request, with_gpu},
};
use texelith_core::{Extent, InputInterpretation};
use texelith_gpu::{Quality, TypedSourcePixels};

fn pixels(bytes: Vec<u8>) -> TypedSourcePixels {
    TypedSourcePixels::Rgba8 {
        pixels: bytes.into(),
        interpretation: InputInterpretation::SrgbPhoto,
    }
}

#[test]
fn real_worker_matches_cpu_on_constants_ramps_impulses_seeded_npot_and_singleton_axes() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let mut id = 0;
        for (w, h, kind) in [
            (1, 1, 0),
            (17, 13, 0),
            (31, 19, 1),
            (17, 13, 2),
            (257, 193, 3),
            (1, 23, 3),
            (23, 1, 1),
        ] {
            let e = Extent::new(w, h).unwrap();
            let bytes = photo(w, h, kind);
            let source = gpu.install(&worker, e, pixels(bytes.clone()));
            for c in [
                Controls::default(),
                Controls {
                    picked: [0.0; 3],
                    radius: 0.375,
                    feather: 0.0,
                    ..Default::default()
                },
                Controls {
                    picked: [1.0; 3],
                    radius: 0.75,
                    feather: 0.5,
                    threshold: 0.75,
                    ..Default::default()
                },
            ] {
                id += 1;
                compare(
                    &gpu.evaluate(&worker, request(&source, id, c, Quality::Full)),
                    &photo_reference(&bytes, c),
                );
                let selected: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| {
                        reference::selection(
                            [
                                reference::linear_half(p[0]),
                                reference::linear_half(p[1]),
                                reference::linear_half(p[2]),
                            ],
                            c,
                        )
                    })
                    .collect();
                compare(&gpu.photo_selection(e, &bytes, c), &selected);
            }
        }
        assert_eq!(worker.diagnostics().validation_errors, 0);
        assert!(worker.diagnostics().peak_bytes <= GIB);
        worker.join().unwrap();
    });
}

#[test]
fn black_white_primary_extremes_and_zero_radius_select_only_the_exact_picked_center() {
    with_gpu(|gpu| {
        let colors = [
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [255, 0, 0, 1],
            [0, 255, 0, 128],
            [0, 0, 255, 254],
        ];
        let bytes: Vec<_> = colors.into_iter().flatten().collect();
        for (i, picked) in [
            [0., 0., 0.],
            [1., 1., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]
        .into_iter()
        .enumerate()
        {
            let c = Controls {
                picked,
                radius: 0.0,
                feather: 0.0,
                ..Default::default()
            };
            let expected: Vec<_> = (0..5).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
            compare(
                &gpu.photo_selection(Extent::new(5, 1).unwrap(), &bytes, c),
                &expected,
            );
        }
    });
}

#[test]
fn photo_is_decoded_once_picker_is_linear_and_alpha_is_ignored() {
    with_gpu(|gpu| {
        let bytes = vec![
            128, 128, 128, 0, 128, 128, 128, 255, 0, 0, 0, 20, 255, 255, 255, 90,
        ];
        let c = Controls {
            picked: [reference::linear_half(128) as f32; 3],
            radius: 0.0,
            feather: 0.0,
            ..Default::default()
        };
        compare(
            &gpu.photo_selection(Extent::new(4, 1).unwrap(), &bytes, c),
            &[1., 1., 0., 0.],
        );
        let encoded_picker = Controls {
            picked: [128.0 / 255.0; 3],
            ..c
        };
        compare(
            &gpu.photo_selection(Extent::new(4, 1).unwrap(), &bytes, encoded_picker),
            &[0.; 4],
        );
        let red = [128, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0];
        let c = Controls {
            picked: [0.; 3],
            radius: 0.25,
            ..c
        };
        compare(
            &gpu.photo_selection(Extent::new(3, 1).unwrap(), &red, c),
            &[1., 0., 1.],
        );
    });
}

#[test]
fn color_control_edits_recompute_and_identical_cold_requests_are_byte_deterministic() {
    with_gpu(|gpu| {
        let e = Extent::new(256, 1).unwrap();
        let bytes: Vec<_> = (0..=255u8).flat_map(|v| [v, v, v, 255]).collect();
        let base = Controls {
            picked: [0.; 3],
            radius: 0.25,
            feather: 0.5,
            ..Default::default()
        };
        let expected = |c| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| reference::selection([reference::linear_half(p[0]); 3], c))
                .collect::<Vec<_>>()
        };
        let baseline = gpu.photo_selection(e, &bytes, base);
        compare(&baseline, &expected(base));
        let worker = gpu.worker(GIB);
        let source = gpu.install(&worker, e, pixels(bytes.clone()));
        let worker_baseline = gpu.evaluate(&worker, request(&source, 1, base, Quality::Full));
        compare(&worker_baseline, &photo_reference(&bytes, base));
        for (i, c) in [
            Controls {
                picked: [0.5; 3],
                ..base
            },
            Controls {
                radius: 0.5,
                ..base
            },
            Controls {
                threshold: 0.9,
                ..base
            },
            Controls {
                feather: 0.0,
                ..base
            },
        ]
        .into_iter()
        .enumerate()
        {
            let result = gpu.photo_selection(e, &bytes, c);
            compare(&result, &expected(c));
            assert_ne!(
                result, baseline,
                "fixture must exercise edited control: {c:?}"
            );
            let before = worker.diagnostics().submitted;
            compare(
                &gpu.evaluate(&worker, request(&source, i as u64 + 2, c, Quality::Full)),
                &photo_reference(&bytes, c),
            );
            assert!(
                worker.diagnostics().submitted > before,
                "edited controls must invalidate the worker cache: {c:?}"
            );
        }
        let repeated = gpu.photo_selection(e, &bytes, base);
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&repeated),
            bytemuck::cast_slice::<f32, u8>(&baseline)
        );
        // Each fixture creates a fresh kernel and forces an uncached dispatch.
        let cold = gpu.photo_selection(e, &bytes, base);
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&cold),
            bytemuck::cast_slice::<f32, u8>(&baseline)
        );
        worker.join().unwrap();
        let fresh = gpu.worker(GIB);
        let source = gpu.install(&fresh, e, pixels(bytes));
        let cold = gpu.evaluate(&fresh, request(&source, 1, base, Quality::Full));
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&cold),
            bytemuck::cast_slice::<f32, u8>(&worker_baseline)
        );
        fresh.join().unwrap();
    });
}

#[test]
fn worker_absent_mask_uses_constant_one_for_every_mode() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let source = gpu.install(
            &worker,
            Extent::new(2, 1).unwrap(),
            pixels(vec![0, 0, 0, 0, 255, 255, 255, 255]),
        );
        for (i, mode) in [Combine::Replace, Combine::Add, Combine::Subtract]
            .into_iter()
            .enumerate()
        {
            let c = Controls {
                picked: [0.; 3],
                feather: 0.0,
                mode,
                ..Default::default()
            };
            compare(
                &gpu.evaluate(&worker, request(&source, i as u64 + 1, c, Quality::Full)),
                &if matches!(mode, Combine::Subtract) {
                    [0.; 2]
                } else {
                    [1.; 2]
                },
            );
        }
        worker.join().unwrap();
    });
}

#[test]
fn proxy_composes_constant_one_and_returns_the_requested_extent() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        // Every 2x2 cell is constant. Sampling/restriction choice at this exact
        // ratio cannot alter the expected color; no sibling leaf generates input.
        let e = Extent::new(2048, 32).unwrap();
        let bytes: Vec<_> = (0..e.width * e.height)
            .flat_map(|i| {
                if (i % e.width / 16).is_multiple_of(2) {
                    [0, 0, 0, 255]
                } else {
                    [255; 4]
                }
            })
            .collect();
        let source = gpu.install(&worker, e, pixels(bytes));
        let proxy = texelith_gpu::proxy_extent(e).unwrap();
        let c = Controls {
            picked: [0.; 3],
            feather: 0.0,
            ..Default::default()
        };
        let expected = vec![1.0; (proxy.width * proxy.height) as usize];
        compare(
            &gpu.evaluate(&worker, request(&source, 1, c, Quality::Proxy)),
            &expected,
        );
        worker.join().unwrap();
    });
}

#[test]
fn pointwise_selection_commutes_with_wrapped_translation_and_preserves_identical_seam_edges() {
    with_gpu(|gpu| {
        let worker = gpu.worker(GIB);
        let (w, h) = (29, 17);
        let e = Extent::new(w, h).unwrap();
        let mut bytes = photo(w, h, 3);
        for y in 0..h {
            let a = (y * w * 4) as usize;
            let b = ((y * w + w - 1) * 4) as usize;
            let p: [u8; 4] = bytes[a..a + 4].try_into().unwrap();
            bytes[b..b + 4].copy_from_slice(&p);
        }
        let row = bytes[..(w * 4) as usize].to_vec();
        bytes[((h - 1) * w * 4) as usize..].copy_from_slice(&row);
        let shifted: Vec<_> = (0..w * h)
            .flat_map(|i| {
                let j = ((((i / w + 5) % h) * w + (i % w + 11) % w) * 4) as usize;
                bytes[j..j + 4].iter().copied()
            })
            .collect();
        let c = Controls::default();
        let source = gpu.install(&worker, e, pixels(bytes.clone()));
        let original = gpu.evaluate(&worker, request(&source, 1, c, Quality::Full));
        compare(&original, &photo_reference(&bytes, c));
        let original = gpu.photo_selection(e, &bytes, c);
        for y in 0..h {
            assert_eq!(
                original[(y * w) as usize],
                original[(y * w + w - 1) as usize]
            );
        }
        for x in 0..w {
            assert_eq!(original[x as usize], original[((h - 1) * w + x) as usize]);
        }
        let source = gpu.install(&worker, e, pixels(shifted.clone()));
        let translated = gpu.evaluate(&worker, request(&source, 2, c, Quality::Full));
        compare(&translated, &photo_reference(&shifted, c));
        let translated = gpu.photo_selection(e, &shifted, c);
        for i in 0..w * h {
            assert_eq!(
                translated[i as usize],
                original[(((i / w + 5) % h) * w + (i % w + 11) % w) as usize]
            );
        }
        worker.join().unwrap();
    });
}
