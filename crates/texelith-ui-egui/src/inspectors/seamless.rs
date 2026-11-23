//! Shared Seamless controls for both material inspector entry points.
use crate::inspectors::labels::text as label;
use crate::material_ui::labels::text;

pub(crate) fn show(
    ui: &mut egui::Ui,
    v: &mut texelith_core::SeamlessMethod,
    overlap: &[f32; 2],
    grid: &[u32; 2],
    seed: &[u32; 2],
    strength: &[f32; 2],
    responses: &mut Vec<egui::Response>,
) {
    let mut offset = ui.selectable_label(
        matches!(v, texelith_core::SeamlessMethod::OffsetBlend { .. }),
        text("Offset blend"),
    );
    if offset.clicked() {
        *v = texelith_core::SeamlessMethod::OffsetBlend { overlap: 0.25 };
        offset.mark_changed();
    }
    responses.push(offset);
    let mut splat = ui.selectable_label(
        matches!(v, texelith_core::SeamlessMethod::Splat { .. }),
        text("Periodic splats"),
    );
    if splat.clicked() {
        *v = texelith_core::SeamlessMethod::Splat {
            seed: 0,
            grid: 4,
            overlap: 0.25,
        };
        splat.mark_changed();
    }
    responses.push(splat);
    let mut histogram = ui.selectable_label(
        matches!(v, texelith_core::SeamlessMethod::Histogram { .. }),
        label("histogram"),
    );
    if histogram.clicked() {
        *v = texelith_core::SeamlessMethod::Histogram {
            base: texelith_core::SeamlessBase::OffsetBlend { overlap: 0.25 },
            strength: 1.0,
        };
        histogram.mark_changed();
    }
    responses.push(histogram);
    let mut base = match v {
        texelith_core::SeamlessMethod::OffsetBlend { overlap } => {
            texelith_core::SeamlessBase::OffsetBlend { overlap: *overlap }
        }
        texelith_core::SeamlessMethod::Splat {
            seed,
            grid,
            overlap,
        } => texelith_core::SeamlessBase::Splat {
            seed: *seed,
            grid: *grid,
            overlap: *overlap,
        },
        texelith_core::SeamlessMethod::Histogram { base, .. } => base.clone(),
    };
    if let texelith_core::SeamlessMethod::Histogram { strength: s, .. } = v {
        super::rows::float(
            ui,
            &label("histogram-strength"),
            s,
            strength[0]..=strength[1],
            responses,
        );
        ui.horizontal(|ui| {
            let mut offset = ui.selectable_label(
                matches!(base, texelith_core::SeamlessBase::OffsetBlend { .. }),
                text("Offset blend"),
            );
            if offset.clicked() {
                base = texelith_core::SeamlessBase::OffsetBlend { overlap: 0.25 };
                offset.mark_changed();
            }
            responses.push(offset);
            let mut splat = ui.selectable_label(
                matches!(base, texelith_core::SeamlessBase::Splat { .. }),
                text("Periodic splats"),
            );
            if splat.clicked() {
                base = texelith_core::SeamlessBase::Splat {
                    seed: 0,
                    grid: 4,
                    overlap: 0.25,
                };
                splat.mark_changed();
            }
            responses.push(splat);
        });
    }
    match &mut base {
        texelith_core::SeamlessBase::OffsetBlend { overlap: v } => {
            super::rows::float(ui, &text("Overlap"), v, overlap[0]..=overlap[1], responses);
        }
        texelith_core::SeamlessBase::Splat {
            seed: s,
            grid: g,
            overlap: v,
        } => {
            super::rows::uint(ui, &label("seed"), s, seed[0]..=seed[1], responses);
            super::rows::uint(ui, &text("Grid"), g, grid[0]..=grid[1], responses);
            super::rows::float(ui, &text("Overlap"), v, overlap[0]..=overlap[1], responses);
        }
    }
    *v = match v {
        texelith_core::SeamlessMethod::Histogram { strength, .. } => {
            texelith_core::SeamlessMethod::Histogram {
                base,
                strength: *strength,
            }
        }
        _ => match base {
            texelith_core::SeamlessBase::OffsetBlend { overlap } => {
                texelith_core::SeamlessMethod::OffsetBlend { overlap }
            }
            texelith_core::SeamlessBase::Splat {
                seed,
                grid,
                overlap,
            } => texelith_core::SeamlessMethod::Splat {
                seed,
                grid,
                overlap,
            },
        },
    };
}
