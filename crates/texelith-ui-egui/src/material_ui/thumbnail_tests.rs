use crate::gpu_availability;

use super::*;
use crate::material_ui::MAPS;

fn photo_upload() -> TypedSourceUpload {
    let extent = Extent::new(256, 129).unwrap();
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: (0..extent.width * extent.height)
            .flat_map(|i| [(i % 251) as u8, (i / 256) as u8, 127, 255])
            .collect::<Vec<_>>()
            .into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    TypedSourceUpload {
        descriptor: pixels.descriptor(extent).unwrap(),
        pixels,
    }
}

fn settle(h: &mut Harness, c: &mut Controls) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        h.runtime.tick(c);
        assert!(h.runtime.error.is_none(), "{:?}", h.runtime.error);
        assert!(
            h.runtime.viewport.error.is_none(),
            "{:?}",
            h.runtime.viewport.error
        );
        if c.map_thumbnails_current() && !h.runtime.inspectors.busy() {
            break;
        }
        assert!(Instant::now() < deadline, "{}", h.runtime.status);
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn cards_are_drawn(c: &mut Controls, ctx: &egui::Context) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 1000.),
            )),
            ..Default::default()
        },
        |ui| {
            c.show(ui);
        },
    );
    output.textures_delta.clear();
    fn uses(shape: &egui::Shape, texture: egui::TextureId) -> bool {
        match shape {
            egui::Shape::Mesh(mesh) => mesh.texture_id == texture,
            egui::Shape::Vec(shapes) => shapes.iter().any(|s| uses(s, texture)),
            _ => false,
        }
    }
    for map in MAPS {
        let texture = c.workspace.thumbnails[&map]
            .texture
            .as_ref()
            .expect("card texture")
            .id();
        assert!(
            output.shapes.iter().any(|s| uses(&s.shape, texture)),
            "{map:?} card blank"
        );
    }
}

#[test]
fn seven_cards_populate_without_selection_and_keep_images_through_generation_changes() {
    if gpu_availability::skip_if_no_gpu() {
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for mode in [ViewMode::TwoD, ViewMode::ThreeD] {
        let mut h = Harness::new();
        let mut c = Controls::new_material();
        c.action(UiAction::Request(CommandRequest::SetMode { mode }));
        h.decoded(&mut c, photo_upload());
        settle(&mut h, &mut c);
        let initial = c.session().snapshot().unwrap();
        let selected = initial.graph.selected_output_ref();
        assert!(
            MAPS.into_iter()
                .all(|m| c.map_thumbnail_size(m) == Some([128, 65]))
        );
        let ctx = egui::Context::default();
        cards_are_drawn(&mut c, &ctx);
        let old: Vec<_> = MAPS
            .into_iter()
            .map(|m| c.workspace.thumbnails[&m].image.clone())
            .collect();
        let address = ParameterAddress {
            node: initial
                .graph
                .material_nodes()
                .iter()
                .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
                .unwrap()
                .id,
            parameter: ParameterId::Strength,
        };
        c.action(UiAction::DragBegin { address });
        c.action(UiAction::Change {
            address,
            value: ParameterValue::Float(2.0),
        });
        assert!(!c.map_thumbnails_current());
        cards_are_drawn(&mut c, &ctx);
        for (map, image) in MAPS.into_iter().zip(old) {
            assert_eq!(c.workspace.thumbnails[&map].image, image);
        }
        // No auxiliary job may be started while the slider owns compute time.
        for _ in 0..5 {
            h.runtime.tick(&mut c);
            assert!(!h.runtime.inspectors.busy());
        }
        c.action(UiAction::FocusLost);
        settle(&mut h, &mut c);
        assert_eq!(
            c.session().snapshot().unwrap().graph.selected_output_ref(),
            selected
        );
        assert_ne!(
            c.session().snapshot().unwrap().generation,
            initial.generation
        );
        cards_are_drawn(&mut c, &ctx);
        assert!(h.runtime.gpu.diagnostics().validation_errors == 0);
        assert!(h.runtime.gpu.diagnostics().peak_bytes <= 1024 * 1024 * 1024);
    }
}

#[test]
fn thumbnail_requests_bound_color_and_imported_normal_to_proxy_size() {
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearNormal,
    ] {
        let mut c = Controls::new_material();
        c.open_image("request-fixture.png".into());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            panic!()
        };
        c.complete_import(
            token,
            SourceDescriptor {
                content_hash: [7; 32],
                extent: Extent::new(2048, 1024).unwrap(),
                interpretation,
            },
        )
        .unwrap();
        let snapshot = c.session().snapshot().unwrap();
        for map in MAPS
            .into_iter()
            .filter(|m| super::super::super::supported(&snapshot, *m))
        {
            let request = inspector_runtime::thumbnail_request(snapshot.clone(), map, 1).unwrap();
            assert_eq!(request.quality, Quality::Proxy);
            assert_eq!(request.extent, Extent::new(1024, 512).unwrap());
            MaterialPlan::new(&request, 1024 * 1024 * 1024, Default::default()).unwrap();
        }
    }
}
