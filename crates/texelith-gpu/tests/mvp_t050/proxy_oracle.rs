//! Independent f64 box integration of decoded unit normals, in pixel coordinates.
use super::oracle::Grid;

pub fn restrict(bytes: &[u8], source: Grid, target: Grid) -> Vec<[f64; 3]> {
    let mut result = Vec::with_capacity(target.len());
    for y in 0..target.h {
        for x in 0..target.w {
            let left = x as f64 * source.w as f64 / target.w as f64;
            let right = (x + 1) as f64 * source.w as f64 / target.w as f64;
            let top = y as f64 * source.h as f64 / target.h as f64;
            let bottom = (y + 1) as f64 * source.h as f64 / target.h as f64;
            let mut sum = [0.0; 3];
            for sy in top.floor() as usize..bottom.ceil() as usize {
                for sx in left.floor() as usize..right.ceil() as usize {
                    let weight = (right.min((sx + 1) as f64) - left.max(sx as f64))
                        * (bottom.min((sy + 1) as f64) - top.max(sy as f64));
                    if weight <= 0.0 {
                        continue;
                    }
                    let offset = 3 * (sy * source.w + sx);
                    let n: [f64; 3] =
                        std::array::from_fn(|i| f64::from(bytes[offset + i]) / 127.5 - 1.0);
                    let length = n.iter().map(|v| v * v).sum::<f64>().sqrt();
                    for i in 0..3 {
                        sum[i] += weight * n[i] / length;
                    }
                }
            }
            let length = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
            result.push(if length > 1e-10 {
                sum.map(|v| v / length)
            } else {
                [0.0, 0.0, 1.0]
            });
        }
    }
    result
}

pub fn analytic_rgb(source: Grid) -> Vec<u8> {
    let height = super::oracle::analytic(source);
    super::oracle::normals(&source.gradient(&height), false)
        .iter()
        .flat_map(|n| n.map(|v| ((v * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8))
        .collect()
}

#[test]
fn area_oracle_preserves_constants_and_cancelling_normals() {
    let source = Grid {
        w: 3,
        h: 1,
        repeat: true,
    };
    let target = Grid {
        w: 2,
        h: 1,
        repeat: true,
    };
    let actual = restrict(&[255, 128, 255].repeat(3), source, target);
    assert_eq!(actual[0], actual[1]);
    let n = actual[0];
    assert!((n[0] - n[2]).abs() < 1e-14);
    let source = Grid { w: 2, ..source };
    let target = Grid { w: 1, ..target };
    assert_eq!(
        restrict(&[255, 128, 255, 0, 127, 0], source, target),
        vec![[0.0, 0.0, 1.0]]
    );
}
