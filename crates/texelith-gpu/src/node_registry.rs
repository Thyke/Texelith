//! Single list consumed by the library, worker and verification example.
//! Kernel/runtime paths are relative to src; suite paths to examples; validator
//! paths to examples/mvp_support.
//! Empty brackets omit runtime registration or a verification suite.
#[doc(hidden)]
#[macro_export]
macro_rules! gpu_nodes {
    ($consumer:ident) => {
        $consumer! {
            builtin {} [builtin, "builtin/runtime.rs"] [];
            height {HeightParameters, PreparedHeight, PreparedMinMaxRemap} [height, "height/runtime.rs"] ["height", "mvp_support/height.rs", []];
            normal_filters {NormalParameters, PreparedNormal} [normal_filters, "normal_filters/runtime.rs"] ["normal", "mvp_support/t048/mod.rs", [normal_contract, "t048/contract.rs"]];
            poisson_operators {PoissonGrid, PoissonOperators, PoissonPass, PreparedPoisson} [] ["poisson_operators", "mvp_support/t049/mod.rs", []];
            roughness {PreparedRoughness} [roughness, "roughness/runtime.rs"] ["roughness", "mvp_support/t053/mod.rs", []];
            horizon_ao {AoParameters, AoResources, HorizonAoKernel, PreparedHorizonAo} [horizon_ao, "horizon_ao/runtime.rs"] ["ao", "mvp_support/t051/mod.rs", []];
            pub seamless_offset {} [seamless_offset, "seamless_offset/runtime.rs"] ["seamless_offset", "mvp_support/t057/mod.rs", []];
            // T-050: bounded height integration and complete-solve suite.
            poisson_solver {PoissonParameters, PoissonSolveReport, PoissonSolver, PoissonStop} [poisson_solver, "poisson_solver/runtime.rs"] ["poisson", "mvp_support/t050/mod.rs", []];
            pub metallic {MetallicKernel} [metallic, "metallic/runtime.rs"] ["metallic", "mvp_support/t054/mod.rs", []]; // T-054
            curvature {CurvatureParameters, PreparedCurvature} [curvature, "curvature/runtime.rs"] ["curvature", "mvp_support/t052/mod.rs", [curvature_contract, "t052/report_contract.rs"]];
            export_packing {ExportPackingKernel, ExportPackingSource, ExportPackingMetadata, PreparedExportPacking} [] ["packing", "mvp_support/t078/mod.rs", [packing_contract, "t078/report_contract.rs"]]; // T-078
            pub seamless_splat {} [seamless_splat, "seamless_splat/runtime.rs"] ["seamless_splat", "mvp_support/t058/mod.rs", [splat_contract, "t058/contract.rs"]]; // T-058
            pub paint_mask {PaintKernel, PreparedPaint} [paint_mask, "paint_mask/runtime.rs"] ["paint", "mvp_support/t056/mod.rs", [paint_contract, "t056/report_contract.rs"]]; // T-056
            pub seam_heatmap {PreparedSeamHeatmap, SeamMetrics} [] ["seam_heatmap", "mvp_support/t060/mod.rs", [seam_heatmap_contract, "t060/report_contract.rs"]]; // T-060
            pub seamless_histogram {} [seamless_histogram, "seamless_histogram/runtime.rs"] ["seamless_histogram", "mvp_support/t059/mod.rs", [histogram_contract, "t059/report_contract.rs"]]; // T-059
        }
    };
}
