//! Fixed label, elastic slider and fixed numeric field shared by inspector sections.
use std::ops::RangeInclusive;

fn numeric(ui: &mut egui::Ui, value: egui::DragValue<'_>) -> egui::Response {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(56., 24.), egui::Sense::hover());
    // add_sized is only a size hint to DragValue's button. Isolate its allocation
    // and clip long numbers in both button and text-edit mode instead of growing the row.
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    child.shrink_clip_rect(rect);
    child.spacing_mut().button_padding.x = 4.;
    let mut response = child.add(value.min_size(rect.size()).clip_text(true));
    response.rect = rect;
    response.interact_rect = response.interact_rect.intersect(rect);
    response
}

fn row(ui: &mut egui::Ui, label: &str, content: impl FnOnce(&mut egui::Ui, f32)) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.;
        let width = ui.available_width();
        let label_width = (width * 0.32).min(96.);
        ui.add_sized([label_width, 24.], egui::Label::new(label).truncate())
            .on_hover_text(label);
        ui.visuals_mut().selection.bg_fill = crate::workspace::studio::AMBER;
        ui.visuals_mut().slider_trailing_fill = true;
        content(ui, (width - label_width - 56. - 12.).max(20.));
    });
}

pub(crate) fn float(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: RangeInclusive<f32>,
    responses: &mut Vec<egui::Response>,
) {
    // Unbounded schemas still accept typed values; dragging covers a working range.
    let working = (*range.start()).max(-10.)..=if *range.end() > 10000. {
        10.
    } else {
        *range.end()
    };
    float_range(ui, label, value, range, working, responses);
}

pub(crate) fn float_range(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: RangeInclusive<f32>,
    slider_range: RangeInclusive<f32>,
    responses: &mut Vec<egui::Response>,
) {
    row(ui, label, |ui, width| {
        ui.spacing_mut().slider_width = width;
        responses.push(
            ui.add(
                egui::Slider::new(value, slider_range.clone())
                    .show_value(false)
                    .clamping(egui::SliderClamping::Never)
                    .logarithmic(*range.start() >= 0. && *slider_range.end() > 100.),
            ),
        );
        responses.push(numeric(
            ui,
            egui::DragValue::new(value)
                .speed(((*slider_range.end() - *slider_range.start()) / 300.).clamp(0.0001, 0.1))
                .range(range)
                .fixed_decimals(if *slider_range.end() <= 0.1 { 3 } else { 2 }),
        ));
    });
}

pub(crate) fn uint(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut u32,
    range: RangeInclusive<u32>,
    responses: &mut Vec<egui::Response>,
) {
    row(ui, label, |ui, width| {
        ui.spacing_mut().slider_width = width;
        responses.push(ui.add(egui::Slider::new(value, range.clone()).show_value(false)));
        responses.push(numeric(
            ui,
            egui::DragValue::new(value).range(range).fixed_decimals(0),
        ));
    });
}

/// Normalize the kernel's source-dependent relief scale into useful artist units.
/// Display that as 1.00, with 55% of the track reserved for subtle relief (0..1).
pub(crate) fn normal_strength(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    unit: f32,
    responses: &mut Vec<egui::Response>,
) {
    row(ui, label, |ui, width| {
        let mut displayed = *value / unit;
        let mut position = if displayed <= 1. {
            displayed * 0.55
        } else {
            0.55 + (displayed - 1.) * 0.45 / 4.
        };
        ui.spacing_mut().slider_width = width;
        let slider = ui.add(
            egui::Slider::new(&mut position, 0.0..=1.0)
                .show_value(false)
                .clamping(egui::SliderClamping::Never),
        );
        if slider.changed() {
            displayed = normal_strength_at(position);
            *value = displayed * unit;
        }
        responses.push(slider);
        let field = numeric(
            ui,
            egui::DragValue::new(&mut displayed)
                .speed(0.01)
                .range(0.0..=f32::MAX.min(f32::MAX * unit))
                .fixed_decimals(2),
        );
        if field.changed() {
            *value = displayed * unit;
        }
        responses.push(field);
    });
}

fn normal_strength_at(position: f32) -> f32 {
    if position <= 0.55 {
        position / 0.55
    } else {
        1. + (position - 0.55) * 4. / 0.45
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_track_has_a_useful_default_and_bounded_sensitivity() {
        assert!((normal_strength_at(0.55) - 1.).abs() < 1e-6);
        assert_eq!(normal_strength_at(0.), 0.);
        assert!((normal_strength_at(1.) - 5.).abs() < 1e-6);
        for i in 0..100 {
            let delta =
                normal_strength_at((i + 1) as f32 / 100.) - normal_strength_at(i as f32 / 100.);
            assert!((0.0..0.09).contains(&delta));
        }
    }
}
