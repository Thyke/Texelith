//! Independent f64 reference, deliberately separate from the production helpers.
pub fn transfer(x: f64) -> u8 {
    let x = x.clamp(0., 1.);
    let code = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1. / 2.4) - 0.055
    };
    (code * 255.).round() as u8
}
pub fn decode(x: u8) -> f64 {
    let x = f64::from(x) / 255.;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
pub fn tone(x: f64) -> f64 {
    (x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0., 1.)
}
pub fn shade(color: [f64; 3], r: f64, m: f64, nv: f64, nl: f64, nh: f64, vh: f64) -> [f64; 3] {
    if nv <= 0. || nl <= 0. {
        return [0.; 3];
    }
    let a = r * r;
    let a = a.max(0.0025);
    let d = a * a / (std::f64::consts::PI * (nh * nh * (a * a - 1.) + 1.).powi(2));
    let smith = |x: f64| 2. * x / (x + (a * a + (1. - a * a) * x * x).sqrt());
    color.map(|c| {
        let f0 = 0.04 + (c - 0.04) * m;
        let f = f0 + (1. - f0) * (1. - vh).powi(5);
        // Correlated Smith, independently expressed as lambda terms.
        let lambda_v = 1. / smith(nv) - 1.;
        let lambda_l = 1. / smith(nl) - 1.;
        let g = 1. / (1. + lambda_v + lambda_l);
        ((1. - f) * (1. - m) * c / std::f64::consts::PI + d * g * f / (4. * nv * nl)) * nl
    })
}
pub fn normal(rgb: [u8; 3]) -> [f64; 3] {
    let n = rgb.map(|x| f64::from(x) / 127.5 - 1.);
    let l = n.iter().map(|x| x * x).sum::<f64>().sqrt();
    n.map(|x| x / l)
}
