pub(super) type Matrix = [[f64; 4]; 4];
pub(super) const IDENTITY: Matrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
pub(super) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}
pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
pub(super) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(super) fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let largest = a.into_iter().map(f64::abs).fold(0.0, f64::max);
    if largest == 0.0 || !a.iter().all(|x| x.is_finite()) {
        return None;
    }
    let a = a.map(|x| x / largest);
    Some(scale(a, 1.0 / dot(a, a).sqrt()))
}
pub(super) fn multiply(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|col| {
        std::array::from_fn(|row| (0..4).map(|k| a[k][row] * b[col][k]).sum())
    })
}
pub(super) fn position(m: Matrix, p: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r])
}
pub(super) fn cofactors(m: Matrix) -> ([[f64; 3]; 3], f64) {
    let columns: [[f64; 3]; 3] = std::array::from_fn(|i| [m[i][0], m[i][1], m[i][2]]);
    let cof = [
        cross(columns[1], columns[2]),
        cross(columns[2], columns[0]),
        cross(columns[0], columns[1]),
    ];
    (cof, dot(columns[0], cof[0]))
}
pub(super) fn normal(cof: [[f64; 3]; 3], det: f64, n: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| (cof[0][r] * n[0] + cof[1][r] * n[1] + cof[2][r] * n[2]) / det)
}
