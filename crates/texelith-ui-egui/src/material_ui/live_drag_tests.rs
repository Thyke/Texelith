use super::*;

fn photo_upload(extent: Extent) -> TypedSourceUpload {
    let pixels = TypedSourcePixels::Rgba8 {
        pixels: (0..extent.width * extent.height)
            .flat_map(|i| {
                let x = i % extent.width;
                let y = i / extent.width;
                [
                    ((x * 17 + y * 31) % 256) as u8,
                    ((x * 13 + y * 7) % 256) as u8,
                    90,
                    255,
                ]
            })
            .collect::<Vec<_>>()
            .into(),
        interpretation: InputInterpretation::SrgbPhoto,
    };
    TypedSourceUpload {
        descriptor: pixels.descriptor(extent).unwrap(),
        pixels,
    }
}

fn change(c: &mut Controls, address: ParameterAddress, value: f32) {
    let value = if address.parameter == ParameterId::SeamlessMethod {
        ParameterValue::SeamlessMethod(SeamlessMethod::OffsetBlend { overlap: value })
    } else if address.parameter == ParameterId::BandWeights {
        ParameterValue::FloatVector(crate::inspectors::key_controls::scale_weights(value))
    } else {
        ParameterValue::Float(value)
    };
    c.action(UiAction::Change { address, value });
    assert!(c.message.is_empty(), "{}", c.message);
}

#[test]
fn live_drag_all_maps_publish_inflight_samples_and_settle_full_after_release() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = Harness::new();
    let mut c = Controls::new_material();
    let extent = Extent::new(1056, 32).unwrap();
    h.decoded(&mut c, photo_upload(extent));
    h.until(&mut c, |r| r.publications > 0);
    c.action(UiAction::Request(CommandRequest::ConfigureMaterial {
        seamless: true,
        painted_mask: false,
    }));
    let history = c.session().history().undo_entries;
    for (map, parameter, values) in [
        (MapKind::Albedo, ParameterId::SeamlessMethod, [0.15, 0.3]),
        (MapKind::Height, ParameterId::BandWeights, [1.5, 3.5]),
        (MapKind::Normal, ParameterId::Strength, [0.01, 0.04]),
        (MapKind::Ao, ParameterId::Bias, [0.01, 0.03]),
        (MapKind::Roughness, ParameterId::Base, [0.2, 0.7]),
        (MapKind::Metallic, ParameterId::Radius, [0.2, 0.4]),
        (MapKind::Edge, ParameterId::EdgeScale, [0.4, 0.8]),
    ] {
        c.action(UiAction::Request(CommandRequest::SelectMap { map }));
        let snapshot = c.session().snapshot().unwrap();
        let address = ParameterAddress {
            node: snapshot
                .graph
                .material_nodes()
                .iter()
                .find(|n| n.params.parameter(parameter).is_ok())
                .unwrap()
                .id,
            parameter,
        };
        let publications = h.runtime.publications;
        c.action(UiAction::DragBegin { address });
        change(&mut c, address, values[0]);
        let generation = c.session().snapshot().unwrap().generation;
        let revision = c.render_revision;
        h.until(&mut c, |r| r.presenting && r.seen == revision);
        // Advance the slider while its previous result is in the presenter.
        change(&mut c, address, values[1]);
        h.until(&mut c, |r| r.publications > publications);
        assert_eq!(
            h.runtime.displayed,
            Some((generation, Quality::Proxy, proxy_extent(extent).unwrap())),
            "{map:?}"
        );
        assert!(c.session().history().active_gesture.is_some());
        assert_ne!(
            h.runtime.displayed.unwrap().0,
            c.session().snapshot().unwrap().generation
        );
        let publications = h.runtime.publications;
        c.action(UiAction::DragRelease);
        h.until(&mut c, |r| r.publications > publications);
        assert_eq!(
            h.runtime.displayed,
            Some((
                c.session().snapshot().unwrap().generation,
                Quality::Full,
                extent
            )),
            "{map:?}"
        );
        assert!(h.runtime.error.is_none(), "{:?}", h.runtime.error);
    }
    assert_eq!(c.session().history().undo_entries, history + 7);
}

#[test]
fn live_drag_proxy_requests_bound_every_supported_source_and_map() {
    for interpretation in [
        InputInterpretation::SrgbPhoto,
        InputInterpretation::LinearHeight,
        InputInterpretation::LinearNormal,
    ] {
        let mut c = Controls::new_material();
        c.open_image("request-fixture.png".into());
        let Effect::LoadImage { token, .. } = c.take_effects().remove(0) else {
            panic!("load")
        };
        let extent = Extent::new(2048, 1024).unwrap();
        c.complete_import(
            token,
            SourceDescriptor {
                content_hash: [7; 32],
                extent,
                interpretation,
            },
        )
        .unwrap();
        let snapshot = c.session().snapshot().unwrap();
        for map in super::super::super::MAPS
            .into_iter()
            .filter(|map| super::super::super::supported(&snapshot, *map))
        {
            for quality in [Quality::Proxy, Quality::Full] {
                let request =
                    super::super::super::request(snapshot.clone(), map, 1, quality).unwrap();
                assert_eq!(request.quality, quality);
                assert_eq!(
                    request.extent,
                    if quality == Quality::Proxy {
                        proxy_extent(extent).unwrap()
                    } else {
                        extent
                    }
                );
                MaterialPlan::new(&request, 1024 * 1024 * 1024, Default::default()).unwrap();
            }
        }
    }
}

#[test]
fn live_drag_3d_and_split_publish_held_gestures_then_latest_full_material() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = Harness::new();
    let mut c = Controls::new_material();
    h.decoded(&mut c, photo_upload(Extent::new(1056, 32).unwrap()));
    h.until(&mut c, |r| r.publications > 0);
    c.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Normal,
    }));
    c.action(UiAction::Request(CommandRequest::SetMode {
        mode: ViewMode::ThreeD,
    }));
    h.runtime.viewport.set_extent(Extent::new(96, 64).unwrap());
    let address = ParameterAddress {
        node: c
            .session()
            .snapshot()
            .unwrap()
            .graph
            .material_nodes()
            .iter()
            .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
            .unwrap()
            .id,
        parameter: ParameterId::Strength,
    };
    for split in [false, true] {
        c.preview.split_view = split;
        let publications = h.runtime.viewport.publications;
        h.until(&mut c, |r| r.viewport.publications > publications);
        h.runtime.viewport.take_ready().unwrap();
        for sample in 0..3 {
            let publications = h.runtime.viewport.publications;
            let requests = h.runtime.viewport.material_requests;
            c.action(UiAction::DragBegin { address });
            change(&mut c, address, 0.01 + sample as f32 * 0.01);
            let generation = c.session().snapshot().unwrap().generation;
            h.until(&mut c, |r| r.viewport.material_requests > requests);
            change(&mut c, address, 0.05 + sample as f32 * 0.01);
            h.until(&mut c, |r| r.viewport.publications > publications);
            let (a, b) = h.runtime.viewport.take_ready().unwrap();
            assert_eq!(a.generation, generation);
            assert!(
                h.runtime
                    .viewport
                    .cached_materials()
                    .any(|m| m.generation == generation)
            );
            assert!(
                h.runtime
                    .viewport
                    .cached_materials()
                    .filter(|m| m.generation == generation)
                    .all(|m| m.quality == Quality::Proxy
                        && m.extent == Extent::new(1024, 31).unwrap())
            );
            assert_eq!(b.is_some(), split);
            if let Some(b) = b {
                assert_eq!(a.generation, b.generation);
            }
            assert!(c.session().history().active_gesture.is_some());
        }
        let publications = h.runtime.viewport.publications;
        c.action(UiAction::DragRelease);
        let generation = c.session().snapshot().unwrap().generation;
        // A bounded in-flight proxy pair may finish first. Wait for the recipe
        // containing the released value, then verify it actually used Full.
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            h.runtime.tick(&mut c);
            if let Some((a, b)) = h.runtime.viewport.take_ready()
                && a.generation == generation
                && h.runtime.viewport.status.contains("Full")
            {
                assert!(
                    h.runtime
                        .viewport
                        .cached_materials()
                        .any(|m| m.generation == generation
                            && m.quality == Quality::Full
                            && m.extent == Extent::new(1056, 32).unwrap())
                );
                assert_eq!(b.is_some(), split);
                if let Some(b) = b {
                    assert_eq!(a.generation, b.generation);
                }
                break;
            }
            assert!(Instant::now() < deadline, "{}", h.runtime.viewport.status);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(h.runtime.viewport.publications > publications);
        assert!(
            h.runtime.viewport.error.is_none(),
            "{:?}",
            h.runtime.viewport.error
        );
    }
}

#[test]
fn live_drag_interruption_rejects_the_pending_2d_sample() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for interruption in 0..6 {
        let mut h = Harness::new();
        let mut c = Controls::new_material();
        h.decoded(&mut c, height_upload(0.5));
        c.action(UiAction::Request(CommandRequest::SelectMap {
            map: MapKind::Normal,
        }));
        h.until(&mut c, |r| r.publications > 0);
        let address = ParameterAddress {
            node: c
                .session()
                .snapshot()
                .unwrap()
                .graph
                .material_nodes()
                .iter()
                .find(|n| matches!(n.params, MaterialNodeParams::NormalFromHeight { .. }))
                .unwrap()
                .id,
            parameter: ParameterId::Strength,
        };
        c.action(UiAction::DragBegin { address });
        change(&mut c, address, 0.02);
        let revision = c.render_revision;
        h.until(&mut c, |r| r.presenting && r.seen == revision);
        let publications = h.runtime.publications;
        let displayed = h.runtime.displayed;
        change(&mut c, address, 0.04);
        match interruption {
            0 => c.action(UiAction::Escape),
            1 => c.action(UiAction::FocusLost),
            2 => c.action(UiAction::Request(CommandRequest::SelectMap {
                map: MapKind::Height,
            })),
            3 => c.action(UiAction::Request(CommandRequest::SetMode {
                mode: ViewMode::ThreeD,
            })),
            4 => {
                h.runtime.heatmap = true;
            }
            5 => c.open_image("missing-interrupt-fixture.png".into()),
            _ => unreachable!(),
        }
        h.until(&mut c, |r| !r.presenting);
        assert_eq!(
            h.runtime.publications, publications,
            "interruption {interruption}"
        );
        assert_eq!(
            h.runtime.displayed, displayed,
            "interruption {interruption}"
        );
    }
}

fn frame(
    h: &mut Harness,
    c: &mut Controls,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
) -> Vec<(egui::TextureId, egui::ImageData)> {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1600., 1000.),
            )),
            focused: true,
            events,
            ..Default::default()
        },
        |ui| {
            c.show(ui);
            h.runtime.tick(c);
            crate::workspace::center_slot(ui, |ui| h.runtime.show(ui, c));
        },
    );
    let updates = output
        .textures_delta
        .set
        .iter()
        .flat_map(|(id, deltas)| deltas.iter().map(|delta| (*id, delta.image.clone())))
        .collect();
    output.textures_delta.clear();
    updates
}

#[test]
fn live_drag_publishes_changing_2d_textures_before_release() {
    if std::env::var("TEXELITH_CI_NO_GPU").as_deref() == Ok("1") {
        eprintln!(
            "SKIP GPU: no GPU adapter (TEXELITH_CI_NO_GPU=1); hardware assertions not run: {:?}",
            std::thread::current().name()
        );
        return;
    }
    let _serial = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut h = Harness::new();
    let mut c = Controls::new_material();
    h.decoded(&mut c, height_upload(0.5));
    h.until(&mut c, |r| r.publications > 0);
    c.preview.split_view = false;
    c.action(UiAction::Request(CommandRequest::SetMode {
        mode: ViewMode::TwoD,
    }));
    c.action(UiAction::Request(CommandRequest::SelectMap {
        map: MapKind::Normal,
    }));
    h.until(&mut c, |r| r.publications > 0);
    let ctx = egui::Context::default();
    for _ in 0..5 {
        frame(&mut h, &mut c, &ctx, vec![]);
    }
    let slider = c
        .inspectors
        .parameter_rects
        .iter()
        .find(|(a, _)| a.parameter == ParameterId::Strength)
        .unwrap()
        .1[0];
    let pos = egui::pos2(slider.left() + 8., slider.center().y);
    frame(
        &mut h,
        &mut c,
        &ctx,
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                pressed: true,
                button: egui::PointerButton::Primary,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let mut publications = h.runtime.publications;
    let mut images = Vec::new();
    // Each poll has a newer input than the request being completed. Waiting for
    // an exact generation would starve the display for the entire gesture.
    for step in 1..=2000 {
        let pos = egui::pos2(
            egui::lerp(slider.x_range(), 0.1 + ((step * 37) % 79) as f32 / 100.),
            slider.center().y,
        );
        let output = frame(&mut h, &mut c, &ctx, vec![egui::Event::PointerMoved(pos)]);
        assert!(c.session().history().active_gesture.is_some());
        if h.runtime.publications > publications {
            assert_eq!(h.runtime.publications, publications + 1);
            let (id, delta) = output
                .iter()
                .find(|(id, _)| Some(*id) == h.runtime.texture.as_ref().map(|t| t.id()))
                .expect("fresh material texture upload in this frame");
            images.push((*id, delta.clone(), h.runtime.displayed.unwrap()));
            publications += 1;
            if images.len() == 4 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(images.len(), 4, "2D must publish throughout a held drag");
    for pair in images.windows(2) {
        assert_ne!(pair[0].0, pair[1].0, "fresh texture generation");
        assert!(pair[0].2.0 < pair[1].2.0, "changing evaluated generations");
        assert_eq!(pair[1].2.1, Quality::Proxy);
    }
    assert!(
        images.iter().any(|image| image.1 != images[0].1),
        "changing pixels"
    );
    let pos = ctx.pointer_latest_pos().unwrap();
    frame(
        &mut h,
        &mut c,
        &ctx,
        vec![egui::Event::PointerButton {
            pos,
            pressed: false,
            button: egui::PointerButton::Primary,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!(c.session().history().active_gesture.is_none());
    assert_eq!(c.render_quality, Quality::Full);
    h.until(&mut c, |r| r.publications > publications);
    assert_eq!(
        h.runtime.displayed.unwrap().0,
        c.session().snapshot().unwrap().generation
    );
    assert_eq!(h.runtime.displayed.unwrap().1, Quality::Full);
    assert_eq!(c.session().history().undo_entries, 1);
}
