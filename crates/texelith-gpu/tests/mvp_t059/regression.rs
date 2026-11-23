use super::{gpu::with_gpu, reference as r, support::*};
use texelith_core::{Boundary, Extent, InputInterpretation, MapKind};
use texelith_gpu::{GpuConfig, GpuWorker, TypedSourcePixels};

#[test]
fn rec709_gray_bin_edges_are_exact_and_do_not_move_histogram_counts() {
    for edge in 0..=r::BINS {
        let gray = edge as f64 / r::BINS as f64;
        assert_eq!(r::luminance([gray; 3]), gray, "gray bin edge {edge}");
    }
    let gray = 221.0 / 1024.0;
    assert_eq!(r::half(r::linear(128)), gray);
    let histogram = r::Histogram::new(&[r::luminance([gray; 3]); 2]);
    assert_eq!(histogram.counts[220], 0);
    assert_eq!(histogram.counts[221], 2);
    // Negative control: the former separately rounded decimal dot product
    // violates gray identity and assigns these same pixels to the wrong bin.
    let rounded_dot = gray * 0.2126 + gray * 0.7152 + gray * 0.0722;
    assert!(rounded_dot < gray);
    assert_eq!(r::bin(rounded_dot), 220);
}

#[test]
fn full_code_range_color_matches_oracle_on_release_suite_fixture() {
    with_gpu(|g| {
        let worker =
            GpuWorker::spawn(g.device.clone(), g.queue.clone(), GpuConfig::default()).unwrap();
        let e = Extent::new(31, 19).unwrap();
        let bytes: Vec<_> = (0..e.width * e.height)
            .flat_map(|i| {
                [
                    (i * 71 % 256) as u8,
                    (i * 17 % 256) as u8,
                    (i * 43 % 256) as u8,
                    13,
                ]
            })
            .collect();
        let rgb: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| std::array::from_fn(|c| r::half(r::linear(p[c]))))
            .collect();
        let source = install(
            &worker,
            e,
            TypedSourcePixels::Rgba8 {
                pixels: bytes.into(),
                interpretation: InputInterpretation::SrgbPhoto,
            },
        );
        let converted = evaluate(g, &worker, request(std::slice::from_ref(&source), 1, None));
        for c in 0..3 {
            compare(
                converted[&MapKind::Albedo].iter().map(|v| v[c]),
                &rgb.iter().map(|v| v[c]).collect::<Vec<_>>(),
                5e-4,
            );
        }
        let mut serial = 1;
        for base in BASES {
            for strength in [0.0, 0.375, 1.0] {
                serial += 1;
                let actual = evaluate(
                    g,
                    &worker,
                    request(
                        std::slice::from_ref(&source),
                        serial,
                        Some(Controls {
                            base,
                            strength,
                            ..Default::default()
                        }),
                    ),
                );
                let expected = r::render(
                    e,
                    e,
                    &rgb,
                    &base.core(),
                    Boundary::Repeat,
                    f64::from(strength),
                    true,
                );
                for c in 0..3 {
                    compare(
                        actual[&MapKind::Albedo].iter().map(|v| v[c]),
                        &expected.iter().map(|v| v[c]).collect::<Vec<_>>(),
                        5e-4,
                    );
                }
            }
        }
        worker.join().unwrap();
    });
}
