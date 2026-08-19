//! Fourier transforms, ported from `xmris.processing`.
//!
//! `spec_to_fid` reproduces `xmris.processing.fid.to_fid`: inverse-shift the
//! frequency domain so DC sits at index 0, then inverse transform with
//! orthonormal (`1/sqrt(N)`) scaling. rustfft applies no normalisation of its
//! own, so the factor is explicit here.

use num_complex::Complex;
use rustfft::FftPlanner;

/// Move the zero-frequency component from the centre to index 0.
pub fn ifftshift(v: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = v.len();
    let split = n / 2; // ceil(n/2) elements move to the front
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&v[split..]);
    out.extend_from_slice(&v[..split]);
    out
}

/// Move the zero-frequency component from index 0 to the centre.
pub fn fftshift(v: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = v.len();
    let split = n - n / 2;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&v[split..]);
    out.extend_from_slice(&v[..split]);
    out
}

fn scaled(mut buf: Vec<Complex<f32>>, inverse: bool) -> Vec<Complex<f32>> {
    let n = buf.len();
    let mut planner = FftPlanner::<f32>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    fft.process(&mut buf);
    let norm = 1.0 / (n as f32).sqrt();
    for c in buf.iter_mut() {
        *c *= norm;
    }
    buf
}

/// Frequency domain (DC-centred) to time domain.
pub fn spec_to_fid(spec: &[Complex<f32>]) -> Vec<Complex<f32>> {
    scaled(ifftshift(spec), true)
}

/// Time domain to frequency domain (DC-centred). Exact inverse of `spec_to_fid`.
pub fn fid_to_spec(fid: &[Complex<f32>]) -> Vec<Complex<f32>> {
    fftshift(&scaled(fid.to_vec(), false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn approx_eq(a: &[Complex<f32>], b: &[Complex<f32>], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).norm() < tol,
                "index {i}: {x:?} vs {y:?} (tol {tol})"
            );
        }
    }

    #[test]
    fn ifftshift_inverts_fftshift_even_length() {
        let v: Vec<Complex<f32>> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
        approx_eq(&ifftshift(&fftshift(&v)), &v, 1e-6);
    }

    #[test]
    fn ifftshift_inverts_fftshift_odd_length() {
        let v: Vec<Complex<f32>> = (0..7).map(|i| Complex::new(i as f32, 0.0)).collect();
        approx_eq(&ifftshift(&fftshift(&v)), &v, 1e-6);
    }

    #[test]
    fn fftshift_moves_dc_to_centre() {
        let mut v = vec![Complex::new(0.0f32, 0.0); 8];
        v[0] = Complex::new(1.0, 0.0);
        assert_eq!(fftshift(&v)[4], Complex::new(1.0, 0.0));
    }

    #[test]
    fn spec_to_fid_round_trips() {
        let n = 64;
        let spec: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();
        approx_eq(&fid_to_spec(&spec_to_fid(&spec)), &spec, 1e-5);
    }

    #[test]
    fn transform_is_orthonormal() {
        // Energy is preserved under an ortho-normalised transform.
        let n = 32;
        let spec: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new(i as f32, (i as f32) * 0.5))
            .collect();
        let e_in: f32 = spec.iter().map(|c| c.norm_sqr()).sum();
        let e_out: f32 = spec_to_fid(&spec).iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (e_in - e_out).abs() / e_in < 1e-4,
            "energy {e_in} -> {e_out}"
        );
    }

    #[test]
    fn single_bin_spectrum_becomes_a_pure_tone() {
        // A delta one bin above centre must produce a FID whose phase
        // advances by exactly 2*pi/n per sample.
        let n = 32usize;
        let mut spec = vec![Complex::new(0.0f32, 0.0); n];
        spec[n / 2 + 1] = Complex::new(1.0, 0.0);
        let fid = spec_to_fid(&spec);
        let dphi = (fid[1] * fid[0].conj()).arg();
        assert!(
            (dphi - 2.0 * PI / n as f32).abs() < 1e-4,
            "phase step {dphi}"
        );
    }
}
