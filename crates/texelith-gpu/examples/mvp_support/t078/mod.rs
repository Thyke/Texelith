mod cases;
mod fixtures;
mod gpu;
mod reference;
#[allow(clippy::duplicate_mod)] // The registry also exposes this validator.
mod report_contract;
mod resource_cases;
mod stress;
mod timing;

use fixtures::*;
use serde_json::{Value, json};
use texelith_core::*;
use texelith_gpu::{ExportPackingKernel, GpuExecutor};

mod oracle {
    use super::*;
    pub fn scalar(port: OutputPort, w: u32, h: u32, values: &[f64]) -> reference::Source {
        assert_eq!(values.len(), (w * h) as usize);
        reference::Source {
            reference: reference(port),
            extent: extent(w, h),
            pixels: values.iter().map(|v| [*v, 0.0, 0.0, 1.0]).collect(),
            boundary: Boundary::Clamp,
            normal_convention: NormalConvention::OpenGl,
        }
    }
}

fn check(checks: &mut Vec<Value>, names: &[&str]) {
    checks.extend(names.iter().map(|name| json!({"name":name,"passed":true})));
}

fn numerics(g: &gpu::Gpu) -> Vec<Value> {
    let mut checks = Vec::new();
    cases::orm_hdrp_mask_gltf_and_urp_pack_exact_distinct_channels_and_constants(g);
    check(
        &mut checks,
        &[
            "orm",
            "hdrp_mask_map",
            "gltf_metallic_roughness",
            "constants_invert",
            "packed_linear",
        ],
    );
    cases::arbitrary_swizzles_use_full_output_ref_and_apply_invert_before_transfer(g);
    check(&mut checks, &["mixed_transfer"]);
    cases::albedo_is_encoded_once_after_linear_resampling_data_and_alpha_stay_linear(g);
    cases::srgb_knee_and_linear_data_are_numerically_independent(g);
    check(&mut checks, &["srgb_once", "source_alpha_ignored"]);
    cases::normals_convert_y_once_select_components_after_conversion_and_renormalize(g);
    check(&mut checks, &["normal_y_once", "normal_renormalization"]);
    cases::png16_preserves_adjacent_height_codes_that_png8_cannot_represent(g);
    check(&mut checks, &["png16_precision"]);
    cases::explicit_integer_clamp_does_not_remap_and_float_displacement_is_never_clamped(g);
    check(&mut checks, &["signed_displacement"]);
    cases::ordered_dither_is_balanced_repeatable_and_integer_only(g);
    check(&mut checks, &["ordered_dither"]);
    cases::area_bilinear_and_mixed_axis_resize_match_oracle_on_npot_singletons_and_impulses(g);
    cases::large_npot_resize_preserves_footprint_precision_on_both_axes(g);
    check(&mut checks, &["large_npot_resize_both_axes"]);
    check(
        &mut checks,
        &[
            "cpu_oracle",
            "npot_257x193",
            "singleton_axes",
            "area_downsample",
            "bilinear_upscale",
        ],
    );
    cases::keep_is_lossless_in_size_and_pad_is_top_left_repeat_or_clamp_with_uv_metadata(g);
    check(&mut checks, &["keep_npot", "pad_repeat_clamp_metadata"]);
    cases::bc5_preparation_requires_four_multiple_but_does_not_force_power_of_two(g);
    check(&mut checks, &["bc5_multiple_of_four"]);
    cases::same_full_resolution_input_and_parameters_produce_identical_content_hashes(g);
    check(&mut checks, &["determinism"]);
    resource_cases::missing_duplicate_proxy_wrong_output_and_alias_sources_are_rejected_before_dispatch(g);
    check(&mut checks, &["full_resolution_sources"]);
    resource_cases::zero_budget_fails_atomically_and_submitted_resources_retire_only_after_fence(g);
    check(&mut checks, &["low_budget", "fence_retirement"]);
    checks
}

fn measured_errors(g: &gpu::Gpu) -> Value {
    let mut scalar: f64 = 0.0;
    let mut integer: f64 = 0.0;
    let values: Vec<_> = (0..257 * 193)
        .map(|i| f64::from((i * 37) % 1024) / 1024.0)
        .collect();
    let source = oracle::scalar(OutputPort::Height, 257, 193, &values);
    let mut p = plan(
        source.extent,
        vec![output(
            "height",
            ExportFormat::Png16,
            vec![map(OutputPort::Height, Component::R)],
        )],
    );
    p.size = SizePolicy::Resize {
        width: 129,
        height: 97,
    };
    let actual = g.run(&p, 0, std::slice::from_ref(&source));
    for (a, b) in actual.iter().zip(reference::evaluate(&p, 0, &[source])) {
        let error = (a[0] - b[0]).abs();
        scalar = scalar.max(error);
        integer = integer.max(((a[0] * 65535.0).round() - (b[0] * 65535.0).round()).abs());
    }
    let normal = gpu::vector(
        OutputPort::Normal,
        2,
        1,
        vec![[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
    );
    let mut p = plan(
        normal.extent,
        vec![output(
            "normal",
            ExportFormat::Png16,
            [Component::R, Component::G, Component::B]
                .map(|c| map(OutputPort::Normal, c))
                .to_vec(),
        )],
    );
    p.size = SizePolicy::Resize {
        width: 1,
        height: 1,
    };
    let n = g.run(&p, 0, &[normal])[0].map(|v| 2.0 * v - 1.0);
    let component = n[0]
        .abs()
        .max((n[1] - 0.5f64.sqrt()).abs())
        .max((n[2] - 0.5f64.sqrt()).abs());
    let length = ((n[..3].iter().map(|v| v * v).sum::<f64>()).sqrt() - 1.0).abs();
    let heights = oracle::scalar(
        OutputPort::Height,
        5,
        1,
        &[0.500001, 0.500002, 0.500003, 0.500004, 0.500005],
    );
    let mut p = plan(
        heights.extent,
        vec![output(
            "displacement",
            ExportFormat::Exr32,
            vec![map(OutputPort::Height, Component::R)],
        )],
    );
    p.displacement.scale = 4.0;
    let displacement = g
        .run(&p, 0, std::slice::from_ref(&heights))
        .iter()
        .zip(reference::evaluate(&p, 0, &[heights]))
        .map(|(a, b)| (a[0] - b[0]).abs())
        .fold(0.0, f64::max);
    json!({"scalar_max_error":scalar,"integer_max_lsb_error":integer,
        "normal_component_max_error":component,"normal_length_max_error":length,"displacement_max_error":displacement})
}

pub(crate) fn run_registered(context: &crate::SuiteContext<'_>) -> Result<Value, String> {
    let scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let g = gpu::Gpu {
        device: context.device.clone(),
        queue: context.queue.clone(),
        executor: pollster::block_on(GpuExecutor::new(
            context.device.clone(),
            context.queue.clone(),
        ))
        .map_err(|e| e.to_string())?,
        kernel: ExportPackingKernel::default(),
    };
    let mut checks = numerics(&g);
    let numerical = measured_errors(&g);
    let resources = stress::run(&g)?;
    check(&mut checks, &["cancellation"]);
    let (timing, maximum) = timing::run(&g, context.extent)?;
    g.wait();
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(error.to_string());
    }
    let manifest = json!({"algorithm_revision":1,"default_output":"orm_rgb8","default_size":"keep",
        "default_transfer":"linear","default_dither":"none","source_quality":"full", "seed":78,
        "size":context.extent.width,"source_pattern":"per-map seeded 10-bit spatial ramp", "maximum_size":"half per axis",
        "boundary":"clamp","quantization":"nearest ties upward; output-anchored Bayer4x4",
        "default_channels":["AO.R","Roughness.R","Metallic.R"],"maximum_channels":["AO.R","Roughness.R","Metallic.R",1]});
    let mut report = json!({"passed":true,"suite":"packing","size":context.extent.width,"seed":78,
        "adapter":{"name":context.info.name,"backend":format!("{:?}",context.info.backend),"driver":context.info.driver},
        "manifest_hash":report_contract::manifest_hash(&manifest)?,"manifest":manifest,"checks":checks,
        "numerical":numerical,"resources":resources,"timing":timing,"maximum_quality":maximum,
        "accounting_scope":"resident full typed source textures, output, padded readback reservation, bounded upload staging, all packing scratch/uniforms; excludes driver allocations",
        "timing_scope":"encode through queue completion, every production packing pass; full-resolution typed sources resident; no cached results, readback outside node timing",
        "cancellation_scope":"explicit scheduler cancellation between submitted horizontal filtering and final packing; export coordinator integration belongs to T-082"});
    if let Err(error) =
        report_contract::validate(&report).and_then(|()| crate::contract::validate_report(&report))
    {
        report["passed"] = json!(false);
        report["failure"] = json!(error);
    }
    Ok(report)
}
