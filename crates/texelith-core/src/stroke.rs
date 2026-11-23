//! Portable vector brush data and a bounded, arc-length sampler (revision 2).
//!
//! Commands quantize UV and pressure to a 2^-48 grid (nearest, ties up), merge
//! forward collinear events with linear pressure, then sample at radius/8.
//! Arc lengths use integer square roots in 2^-63 UV units; spacing is rounded
//! to that same grid. Turns, reversals and pressure breakpoints remain intact.

use crate::{CoreError, MaskMode};
use serde::{Deserialize, Serialize};

pub const MAX_BRUSH_POINTS: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrokeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Brush {
    /// Radius in normalized source UV units, in (0, 1].
    pub radius: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub mode: MaskMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrushSample {
    pub uv: [f32; 2],
    pub pressure: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrushStroke {
    pub id: StrokeId,
    pub brush: Brush,
    /// Replay in this order; no raster data is retained by core or history.
    pub points: Vec<BrushSample>,
}

fn invalid(message: &'static str) -> CoreError {
    CoreError::InvalidMaterial(message)
}

fn unit(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn zero(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

impl Brush {
    pub fn validate(&self) -> Result<(), CoreError> {
        if !unit(self.radius) || self.radius == 0.0 {
            return Err(invalid("brush radius must be in (0, 1] UV units"));
        }
        if !unit(self.hardness) || !unit(self.opacity) {
            return Err(invalid("brush hardness and opacity must be in [0, 1]"));
        }
        Ok(())
    }

    fn canonical(self) -> Self {
        Self {
            hardness: zero(self.hardness),
            opacity: zero(self.opacity),
            ..self
        }
    }
}

impl BrushSample {
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.uv.into_iter().all(unit) || !unit(self.pressure) {
            return Err(invalid(
                "brush UV and pressure must be finite and in [0, 1]",
            ));
        }
        Ok(())
    }
}

impl BrushStroke {
    pub fn validate(&self) -> Result<(), CoreError> {
        self.brush.validate()?;
        if self.points.is_empty() || self.points.len() > MAX_BRUSH_POINTS {
            return Err(invalid(
                "stroke requires 1..=100000 points; finish and restart",
            ));
        }
        for point in &self.points {
            point.validate()?;
        }
        for pair in self.points.windows(2) {
            if distance(pair[0], pair[1]) > f64::from(self.brush.radius) / 4.0 {
                return Err(invalid("stroke spacing exceeds radius/4"));
            }
        }
        Ok(())
    }
}

/// Versioned BLAKE3 identity of ordered strokes. IDs identify edits, not pixels.
/// Empty data retains the original zero digest; scalar bits normalize signed zero.
pub fn stroke_digest(strokes: &[BrushStroke]) -> Result<[u8; 32], CoreError> {
    if strokes.is_empty() {
        return Ok([0; 32]);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"texelith.brush-strokes\0");
    hash.update(&2u32.to_le_bytes());
    hash.update(&(strokes.len() as u64).to_le_bytes());
    let mut total = 0usize;
    for stroke in strokes {
        stroke.validate()?;
        total = total
            .checked_add(stroke.points.len())
            .ok_or_else(|| invalid("brush point limit exceeded"))?;
        if total > MAX_BRUSH_POINTS {
            return Err(invalid("brush point limit exceeded; finish and restart"));
        }
        for value in [
            stroke.brush.radius,
            stroke.brush.hardness,
            stroke.brush.opacity,
        ] {
            hash.update(&zero(value).to_bits().to_le_bytes());
        }
        hash.update(&[match stroke.brush.mode {
            MaskMode::Replace => 0,
            MaskMode::Add => 1,
            MaskMode::Subtract => 2,
        }]);
        hash.update(&(stroke.points.len() as u64).to_le_bytes());
        for point in &stroke.points {
            for value in [point.uv[0], point.uv[1], point.pressure] {
                hash.update(&zero(value).to_bits().to_le_bytes());
            }
        }
    }
    Ok(*hash.finalize().as_bytes())
}

fn distance(a: BrushSample, b: BrushSample) -> f64 {
    let dx = f64::from(b.uv[0]) - f64::from(a.uv[0]);
    let dy = f64::from(b.uv[1]) - f64::from(a.uv[1]);
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Debug)]
pub(crate) struct StrokeBuilder {
    pub(crate) stroke: BrushStroke,
    start: FixedSample,
    last: FixedSample,
    /// Finalized arc length before the current collinear run, in 2^-63 units.
    distance: u128,
    /// Replay prefix before this run; extending it must replace provisional samples.
    prefix_points: usize,
    input_count: usize,
}

const SAMPLE_SCALE: u64 = 1 << 48;
const SPACING_SCALE: u64 = 1 << 60; // radius/8 in 2^-63 arc-length units

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FixedSample([u64; 3]);

impl FixedSample {
    fn new(sample: BrushSample) -> Self {
        Self(
            [sample.uv[0], sample.uv[1], sample.pressure]
                .map(|value| (f64::from(value) * SAMPLE_SCALE as f64).round() as u64),
        )
    }

    fn replay(self) -> BrushSample {
        let [u, v, pressure] = self
            .0
            .map(|value| (value as f64 / SAMPLE_SCALE as f64) as f32);
        BrushSample {
            uv: [u, v],
            pressure,
        }
    }

    fn delta(self, end: Self) -> [i128; 3] {
        std::array::from_fn(|i| i128::from(end.0[i]) - i128::from(self.0[i]))
    }

    fn length(self, end: Self) -> u128 {
        let [dx, dy, _] = self.delta(end);
        // Coordinates have 48 fractional bits. The extra 30 squared bits give
        // lengths 63 fractional bits. The radicand is at most 2^127.
        (((dx * dx + dy * dy) as u128) << 30).isqrt()
    }

    fn extends(self, middle: Self, end: Self) -> bool {
        let a = self.delta(middle);
        let b = middle.delta(end);
        // Positive UV dot product preserves reversals and stationary pressure
        // changes. Cross products also require pressure to stay linear in UV.
        a[0] * b[0] + a[1] * b[1] > 0
            && a[0] * b[1] == a[1] * b[0]
            && a[0] * b[2] == a[2] * b[0]
            && a[1] * b[2] == a[2] * b[1]
    }

    fn interpolate(self, end: Self, offset: u128, length: u128) -> BrushSample {
        // Weighted nonnegative integers avoid cancellation and direction-specific
        // tie rounding. Products fit in 113 bits even across the full UV diagonal.
        Self(std::array::from_fn(|i| {
            ((u128::from(self.0[i]) * (length - offset)
                + u128::from(end.0[i]) * offset
                + length / 2)
                / length) as u64
        }))
        .replay()
    }
}

impl StrokeBuilder {
    pub(crate) fn new(id: StrokeId, brush: Brush, sample: BrushSample) -> Result<Self, CoreError> {
        brush.validate()?;
        sample.validate()?;
        let sample = FixedSample::new(sample);
        Ok(Self {
            stroke: BrushStroke {
                id,
                brush: brush.canonical(),
                points: vec![sample.replay()],
            },
            start: sample,
            last: sample,
            distance: 0,
            prefix_points: 1,
            input_count: 1,
        })
    }

    pub(crate) fn append(&mut self, samples: &[BrushSample]) -> Result<(), CoreError> {
        if samples.len() > MAX_BRUSH_POINTS - self.input_count {
            return Err(invalid("stroke input limit exceeded; finish and restart"));
        }
        for sample in samples {
            sample.validate()?;
        }
        // Half the maximum spacing leaves room for f32 replay rounding. Only
        // conversion at the input/output boundary uses floats, never arc phase.
        let spacing = (f64::from(self.stroke.brush.radius) * SPACING_SCALE as f64).round() as u128;
        let mut pending = false;
        for sample in samples {
            let sample = FixedSample::new(*sample);
            if sample == self.last {
                continue;
            }
            if !self.start.extends(self.last, sample) {
                if pending {
                    self.sample_run(spacing)?;
                }
                self.distance += self.start.length(self.last);
                self.prefix_points = self.stroke.points.len();
                self.start = self.last;
            }
            self.last = sample;
            pending = true;
        }
        if pending {
            self.sample_run(spacing)?;
        }
        self.input_count += samples.len();
        Ok(())
    }

    fn sample_run(&mut self, spacing: u128) -> Result<(), CoreError> {
        // Recompute only the open run from its canonical endpoints, once per
        // run/batch, not per subdivision. Replace provisional samples from prior
        // append requests so their temporary endpoints cannot affect replay.
        self.stroke.points.truncate(self.prefix_points);
        let length = self.start.length(self.last);
        let end = self.distance + length;
        if length > 0 {
            if spacing == 0 || end / spacing >= MAX_BRUSH_POINTS as u128 {
                return Err(invalid(
                    "resampled stroke limit exceeded; finish and restart",
                ));
            }
            while self.stroke.points.len() as u128 * spacing <= end {
                let offset = self.stroke.points.len() as u128 * spacing - self.distance;
                self.stroke
                    .points
                    .push(self.start.interpolate(self.last, offset, length));
            }
        }
        Ok(())
    }

    pub(crate) fn preview(&self) -> Result<BrushStroke, CoreError> {
        let mut stroke = self.stroke.clone();
        let last = self.last.replay();
        if stroke.points.last() != Some(&last) {
            stroke.points.push(last);
        }
        stroke.validate()?;
        Ok(stroke)
    }

    pub(crate) fn byte_count(&self) -> usize {
        size_of::<Self>() + self.stroke.points.len() * size_of::<BrushSample>()
    }
}
