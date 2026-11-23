//! T-055 behavioral contract; proposed JSON spellings are documented below.
//! Commands carry `node`, `stroke_id`, and either `brush` or `samples`.
use serde_json::{Value, json};
use texelith_core::*;

const NODE: NodeId = NodeId(104);
const EXTENT: Extent = Extent {
    width: 17,
    height: 13,
};

fn session() -> Session {
    let source = SourceDescriptor {
        content_hash: [44; 32],
        extent: EXTENT,
        interpretation: InputInterpretation::SrgbPhoto,
    };
    let mut dto = MaterialGraphDto::from(material_from_source(source).unwrap());
    dto.graph.push(MaterialNode {
        id: NODE,
        inputs: vec![],
        params: MaterialNodeParams::PaintMask {
            boundary: texelith_core::Boundary::Repeat,
            extent: EXTENT,
            stroke_digest: [0; 32],
            strokes: vec![],
        },
    });
    dto.outputs.insert(
        MapKind::Mask,
        OutputRef {
            node: NODE,
            port: OutputPort::Mask,
        },
    );
    Session::from_snapshot(EvaluationSnapshot {
        generation: 40,
        graph: Graph::try_from(dto).unwrap(),
    })
}
fn point(x: f32, y: f32, pressure: f32) -> Value {
    json!({"uv": [x,y], "pressure": pressure})
}
fn brush(radius: f32) -> Value {
    json!({"radius": radius, "hardness": 0.5, "opacity": 0.75, "mode": "Add"})
}
fn request(id: &str, payload: Value) -> CommandRequest {
    serde_json::from_value(json!({"id": id, "payload": payload}))
        .unwrap_or_else(|e| panic!("{id} must accept the T-055 DTO: {e}"))
}
fn begin(s: &mut Session, id: u64, b: Value, p: Value) -> Result<CommandOutcome, CommandError> {
    s.dispatch(request(
        "mask.begin_stroke",
        json!({"node": NODE, "stroke_id": id, "brush": b, "sample": p}),
    ))
}
fn append(s: &mut Session, id: u64, samples: Vec<Value>) -> Result<CommandOutcome, CommandError> {
    s.dispatch(request(
        "mask.append_stroke",
        json!({"stroke_id": id, "samples": samples}),
    ))
}
fn finish(s: &mut Session, id: u64, commit: bool) -> Result<CommandOutcome, CommandError> {
    s.dispatch(request(
        if commit {
            "mask.commit_stroke"
        } else {
            "mask.cancel_stroke"
        },
        json!({"stroke_id": id}),
    ))
}
fn state(s: &Session) -> (EvaluationSnapshot, HistoryState) {
    (s.snapshot().unwrap(), s.history())
}
fn paint(s: &Session) -> Value {
    let snapshot = s.snapshot().unwrap();
    let p = &snapshot.graph.material_node(NODE).unwrap().params;
    serde_json::to_value(p).unwrap()["PaintMask"].clone()
}
fn draw(s: &mut Session, id: u64) {
    begin(s, id, brush(0.2), point(0.1, 0.5, 1.0)).unwrap();
    append(s, id, vec![point(0.5, 0.5, 1.0)]).unwrap();
    finish(s, id, true).unwrap();
}

#[test]
fn stroke_commit_is_one_insertion_diff_and_undo_redo_restore_content() {
    let mut s = session();
    let original = paint(&s);
    draw(&mut s, 1);
    let committed = paint(&s);
    assert_ne!(committed, original);
    assert_eq!(s.history().undo_entries, 1);
    s.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(paint(&s), original);
    assert_eq!(s.history().redo_entries, 1);
    s.dispatch(CommandRequest::Redo).unwrap();
    assert_eq!(paint(&s), committed);
    assert_eq!(s.history().undo_entries, 1);
}

#[test]
fn escape_cancel_restores_content_and_preserves_redo_focus_loss_commit_clears_it() {
    let mut s = session();
    draw(&mut s, 1);
    s.dispatch(CommandRequest::Undo).unwrap();
    let original = paint(&s);
    begin(&mut s, 2, brush(0.2), point(0.1, 0.5, 1.0)).unwrap();
    append(&mut s, 2, vec![point(0.3, 0.5, 0.5)]).unwrap();
    finish(&mut s, 2, false).unwrap();
    assert_eq!(paint(&s), original);
    assert_eq!(s.history().redo_entries, 1);
    begin(&mut s, 3, brush(0.2), point(0.2, 0.5, 1.0)).unwrap();
    finish(&mut s, 3, true).unwrap(); // frontend sends commit on focus loss
    assert_eq!(s.history().undo_entries, 1);
    assert_eq!(s.history().redo_entries, 0);
    s.dispatch(CommandRequest::Undo).unwrap();
    assert_eq!(paint(&s), original);
}

#[test]
fn sampling_is_bounded_and_independent_of_collinear_event_batching() {
    let mut a = session();
    let mut b = session();
    draw(&mut a, 1);
    begin(&mut b, 1, brush(0.2), point(0.1, 0.5, 1.0)).unwrap();
    append(
        &mut b,
        1,
        vec![
            point(0.2, 0.5, 1.0),
            point(0.3, 0.5, 1.0),
            point(0.4, 0.5, 1.0),
            point(0.5, 0.5, 1.0),
        ],
    )
    .unwrap();
    finish(&mut b, 1, true).unwrap();
    let pa = paint(&a);
    assert_eq!(
        pa,
        paint(&b),
        "equivalent paths must yield stable DTO and digest"
    );
    let strokes = pa["strokes"]
        .as_array()
        .expect("PaintMask must retain vector strokes");
    let points = strokes[0]["points"].as_array().unwrap();
    assert!(points.len() >= 9);
    for pair in points.windows(2) {
        let uv = |p: &Value| (p["uv"][0].as_f64().unwrap(), p["uv"][1].as_f64().unwrap());
        let (x0, y0) = uv(&pair[0]);
        let (x1, y1) = uv(&pair[1]);
        assert!((x1 - x0).hypot(y1 - y0) <= 0.2 / 4.0 + 1e-6);
    }
}

#[test]
fn collinear_event_subdivision_preserves_canonical_rounding() {
    let mut direct = session();
    begin(&mut direct, 1, brush(0.2), point(0.01, 0.5, 1.0)).unwrap();
    append(&mut direct, 1, vec![point(0.6, 0.5, 1.0)]).unwrap();
    finish(&mut direct, 1, true).unwrap();

    for separate_batches in [false, true] {
        let mut subdivided = session();
        begin(&mut subdivided, 1, brush(0.2), point(0.01, 0.5, 1.0)).unwrap();
        if separate_batches {
            append(&mut subdivided, 1, vec![point(0.4, 0.5, 1.0)]).unwrap();
            append(&mut subdivided, 1, vec![point(0.6, 0.5, 1.0)]).unwrap();
        } else {
            append(
                &mut subdivided,
                1,
                vec![point(0.4, 0.5, 1.0), point(0.6, 0.5, 1.0)],
            )
            .unwrap();
        }
        finish(&mut subdivided, 1, true).unwrap();
        assert_eq!(
            paint(&direct),
            paint(&subdivided),
            "subdivision must preserve every sample and the stroke digest"
        );
    }
}

#[test]
fn global_spacing_and_pressure_interpolation_survive_a_turn() {
    let mut direct = session();
    let mut subdivided = session();
    for s in [&mut direct, &mut subdivided] {
        begin(s, 1, brush(0.5), point(0.125, 0.125, 0.25)).unwrap();
    }
    append(
        &mut direct,
        1,
        vec![point(0.34375, 0.125, 0.6875), point(0.34375, 0.375, 0.9375)],
    )
    .unwrap();
    for sample in [
        point(0.25, 0.125, 0.5),
        point(0.34375, 0.125, 0.6875),
        point(0.34375, 0.25, 0.8125),
        point(0.34375, 0.375, 0.9375),
    ] {
        append(&mut subdivided, 1, vec![sample]).unwrap();
    }
    finish(&mut direct, 1, true).unwrap();
    finish(&mut subdivided, 1, true).unwrap();
    let canonical = paint(&direct);
    assert_eq!(canonical, paint(&subdivided));
    let points = canonical["strokes"][0]["points"].as_array().unwrap();
    assert_eq!(points.len(), 9);
    // The corner is 3.5 spacings from the start: retain the half-spacing
    // remainder on the second leg, including interpolated pressure.
    assert_eq!(points[4], point(0.34375, 0.15625, 0.71875));
    assert_eq!(points[8], point(0.34375, 0.375, 0.9375));
}

#[test]
fn invalid_bounds_and_mismatched_ids_are_atomic() {
    for (radius, pressure) in [(0.0, 1.0), (-0.1, 1.0), (0.2, -0.01), (0.2, 1.01)] {
        let mut s = session();
        let before = state(&s);
        assert!(begin(&mut s, 1, brush(radius), point(0.5, 0.5, pressure)).is_err());
        assert_eq!(state(&s), before);
    }
    let mut s = session();
    begin(&mut s, 1, brush(0.2), point(0.5, 0.5, 1.0)).unwrap();
    let before = state(&s);
    assert!(append(&mut s, 2, vec![point(0.6, 0.5, 1.0)]).is_err());
    assert!(finish(&mut s, 2, true).is_err());
    assert_eq!(state(&s), before);
}

#[test]
fn excessive_samples_are_rejected_without_partial_mutation() {
    let mut s = session();
    begin(&mut s, 1, brush(0.2), point(0.0, 0.0, 1.0)).unwrap();
    let before = state(&s);
    let samples: Vec<_> = (0..100_001)
        .map(|i| point((i % 2) as f32, 0.5, 1.0))
        .collect();
    assert!(
        append(&mut s, 1, samples).is_err(),
        "per-stroke cap must prevent unbounded input"
    );
    assert_eq!(state(&s), before);
    finish(&mut s, 1, false).unwrap();
}

fn replay_path(radius: f32, path: &[BrushSample], batch_size: usize) -> (BrushStroke, [u8; 32]) {
    let mut s = session();
    begin(
        &mut s,
        1,
        brush(radius),
        serde_json::to_value(path[0]).unwrap(),
    )
    .unwrap();
    for samples in path[1..].chunks(batch_size) {
        s.dispatch(CommandRequest::AppendStroke {
            stroke_id: StrokeId(1),
            samples: samples.to_vec(),
        })
        .unwrap();
    }
    finish(&mut s, 1, true).unwrap();
    let snapshot = s.snapshot().unwrap();
    let MaterialNodeParams::PaintMask {
        strokes,
        stroke_digest,
        ..
    } = &snapshot.graph.material_node(NODE).unwrap().params
    else {
        panic!("expected vector mask");
    };
    (strokes[0].clone(), *stroke_digest)
}

fn assert_identical_strokes(actual: &(BrushStroke, [u8; 32]), expected: &(BrushStroke, [u8; 32])) {
    assert_eq!(actual, expected, "complete DTO and digest must match");
    for (a, b) in actual.0.points.iter().zip(&expected.0.points) {
        assert_eq!(a.uv.map(f32::to_bits), b.uv.map(f32::to_bits));
        assert_eq!(a.pressure.to_bits(), b.pressure.to_bits());
    }
}

#[test]
fn repeated_reversals_are_identical_after_midpoint_subdivision() {
    let a = BrushSample {
        uv: [0.0, 1.0],
        pressure: 1.0,
    };
    let b = BrushSample {
        uv: [0.1875, 0.5],
        pressure: 1.0,
    };
    let m = BrushSample {
        uv: [0.09375, 0.75],
        pressure: 1.0,
    };
    let mut direct = vec![a];
    let mut subdivided = vec![a];
    for _ in 0..262 {
        direct.extend([b, a]);
        subdivided.extend([m, b, m, a]);
    }
    let expected = replay_path(0.5, &direct, direct.len());
    assert_eq!(expected.0.points.len(), 4479);
    for batch_size in [subdivided.len(), 7] {
        assert_identical_strokes(&replay_path(0.5, &subdivided, batch_size), &expected);
    }
}

#[test]
fn random_dyadic_subdivisions_duplicates_and_batches_preserve_bits() {
    // Dyadic coordinates and interpolated pressures are exactly representable
    // f32 values, so these are equivalent paths, not nearby rounded polylines.
    let mut seed = 0x055a_19cdu32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for case in 0..64 {
        let vertices: Vec<[u32; 3]> = (0..6)
            .map(|_| [next() % 257, next() % 257, next() % 257])
            .collect();
        let sample = |v: [u32; 3], divisor: f32| BrushSample {
            uv: [v[0] as f32 / divisor, v[1] as f32 / divisor],
            pressure: v[2] as f32 / divisor,
        };
        let mut direct = vec![sample(vertices[0], 256.0)];
        let mut subdivided = direct.clone();
        for pair in vertices.windows(2) {
            // Include both forward and reverse traversals without removing the
            // reversal: retracing a painted path still contributes arc length.
            for (a, b) in [(pair[0], pair[1]), (pair[1], pair[0]), (pair[0], pair[1])] {
                direct.push(sample(b, 256.0));
                let mut cuts = vec![16];
                cuts.extend((0..8).map(|_| next() % 17));
                cuts.sort_unstable();
                for t in cuts {
                    let v = std::array::from_fn(|i| a[i] * (16 - t) + b[i] * t);
                    let p = sample(v, 4096.0);
                    subdivided.extend([p, p]);
                }
            }
        }
        let radius = [0.125, 0.2, 0.5, 1.0][case % 4];
        let expected = replay_path(radius, &direct, direct.len());
        for batch_size in [1, 7, subdivided.len()] {
            assert_identical_strokes(&replay_path(radius, &subdivided, batch_size), &expected);
        }
    }
}

#[test]
fn collinear_pressure_breakpoints_are_not_discarded() {
    let path = [
        BrushSample {
            uv: [0.125, 0.5],
            pressure: 0.0,
        },
        BrushSample {
            uv: [0.375, 0.5],
            pressure: 0.25,
        },
        BrushSample {
            uv: [0.625, 0.5],
            pressure: 1.0,
        },
    ];
    let result = replay_path(0.5, &path, 1);
    assert_eq!(result.0.points.len(), 9);
    assert_eq!(result.0.points[4], path[1]);
    assert_eq!(result.0.points[2].pressure, 0.125);
    assert_eq!(result.0.points[6].pressure, 0.625);
    assert_ne!(result.1, replay_path(0.5, &[path[0], path[2]], 1).1);
}

#[test]
fn resampling_limit_failure_preserves_builder_and_allows_a_shorter_append() {
    let mut s = session();
    begin(&mut s, 1, brush(0.00001), point(0.0, 0.0, 1.0)).unwrap();
    let before = state(&s);
    assert!(append(&mut s, 1, vec![point(1.0, 0.0, 1.0)]).is_err());
    assert_eq!(state(&s), before);
    append(&mut s, 1, vec![point(0.00002, 0.0, 1.0)]).unwrap();
    finish(&mut s, 1, true).unwrap();
    let expected = replay_path(
        0.00001,
        &[
            BrushSample {
                uv: [0.0, 0.0],
                pressure: 1.0,
            },
            BrushSample {
                uv: [0.00002, 0.0],
                pressure: 1.0,
            },
        ],
        1,
    );
    let actual = paint(&s);
    assert_eq!(
        actual["strokes"][0],
        serde_json::to_value(expected.0).unwrap()
    );
    assert_eq!(actual["stroke_digest"], json!(expected.1));
}

#[test]
fn full_diagonal_and_stationary_pressure_changes_remain_bounded() {
    let a = BrushSample {
        uv: [0.0, 0.0],
        pressure: 0.0,
    };
    let b = BrushSample {
        uv: [1.0, 1.0],
        pressure: 0.0,
    };
    let c = BrushSample { pressure: 1.0, ..b };
    let d = BrushSample { pressure: 1.0, ..a };
    let path = [a, b, c, d];
    let expected = replay_path(1.0, &path, 1);
    assert_identical_strokes(&replay_path(1.0, &[a, a, b, b, c, c, d, d], 7), &expected);
    assert_eq!(expected.0.points.len(), 24);
    assert_eq!(expected.0.points[11].pressure, 0.0);
    assert_eq!(expected.0.points[12].pressure, 1.0);
    assert_eq!(expected.0.points.last(), Some(&d));
    expected.0.validate().unwrap();
}

#[test]
fn maximum_input_batch_is_canonicalized_before_dense_sampling() {
    let path: Vec<_> = (0..100_000)
        .map(|i| BrushSample {
            uv: [i as f32 / 131_072.0, 0.5],
            pressure: 1.0,
        })
        .collect();
    let expected = replay_path(0.0001, &[path[0], path[path.len() - 1]], 1);
    assert!(expected.0.points.len() > 60_000);
    assert_identical_strokes(&replay_path(0.0001, &path, path.len()), &expected);
}
