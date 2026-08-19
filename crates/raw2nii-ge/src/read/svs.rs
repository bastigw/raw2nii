//! Single-voxel reader.
//!
//! `/fid` is MATLAB `(rows, samples)` and already time-domain, so there is no
//! Fourier transform here. Critically (spec §4.1) there is also **no chop
//! correction and no conjugation**: fidall applies both during
//! reconstruction, and repeating them would corrupt the data.

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use raw2nii_core::dataset::{DimTag, Metadata, MrsDataset};
use raw2nii_core::error::{Raw2NiiError, Result};
use serde_json::json;

use crate::header::geometry::{build_affine, svs_localisation};
use crate::header::GeHeader;
use crate::mat::MatFile;

pub fn read_svs(m: &MatFile, h: &GeHeader) -> Result<MrsDataset> {
    let fid = m
        .complex_array("/fid", 2)
        .map_err(|e| Raw2NiiError::MissingData(format!("/fid: {e}")))?;

    let shape = fid.shape().to_vec();
    if shape.len() != 2 {
        return Err(Raw2NiiError::DimensionMismatch {
            expected: vec![0, 0],
            actual: shape,
        });
    }
    let (rows, samples) = (shape[0], shape[1]);

    // (rows, samples) -> (1, 1, 1, samples, rows)
    let mut data = ArrayD::<Complex<f32>>::zeros(IxDyn(&[1, 1, 1, samples, rows]));
    for r in 0..rows {
        for s in 0..samples {
            data[[0, 0, 0, s, r]] = fid[[r, s]];
        }
    }

    let bw = m
        .scalar_f64("/par/bw")
        .map_err(|e| Raw2NiiError::MissingMetadata(format!("/par/bw: {e}")))?;
    if bw <= 0.0 {
        return Err(Raw2NiiError::MissingMetadata(format!(
            "/par/bw must be positive, got {bw}"
        )));
    }

    let f0_hz = m
        .scalar_f64("/par/f0")
        .map_err(|e| Raw2NiiError::MissingMetadata(format!("/par/f0: {e}")))?;

    let loc = svs_localisation(h);
    let affine = build_affine(h, loc.extents_mm, [1, 1, 1]);

    let (nucleus, nucleus_warning) = h.nucleus_name();
    let mut warnings = loc.warnings;
    warnings.extend(nucleus_warning);

    let mut extra = serde_json::Map::new();
    extra.insert("Manufacturer".to_string(), json!("GE"));
    extra.insert("PulseSequenceFile".to_string(), json!(h.psdname));
    extra.insert("SpectralWidth".to_string(), json!(bw));
    if let Some(dt) = h.scan_datetime_iso() {
        extra.insert("ConversionTime".to_string(), json!(dt));
    }

    Ok(MrsDataset {
        data,
        tags: [Some(DimTag::Dyn), None, None],
        affine,
        dwell_time_s: 1.0 / bw,
        meta: Metadata {
            spectrometer_frequency_mhz: vec![f0_hz / 1e6],
            resonant_nucleus: vec![nucleus],
            extra,
            warnings,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples::sample_mat;

    macro_rules! read_sample {
        ($name:expr) => {{
            let p = match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            };
            let m = MatFile::open(&p).unwrap();
            let h = GeHeader::from_mat(&m).unwrap();
            read_svs(&m, &h).unwrap()
        }};
    }

    #[test]
    fn produces_nifti_mrs_shape() {
        let ds = read_sample!("MRS_2H");
        assert_eq!(ds.data.shape(), &[1, 1, 1, 2048, 64]);
        assert_eq!(ds.tags[0], Some(DimTag::Dyn));
        assert!(ds.validate().is_ok());
    }

    #[test]
    fn dwell_time_is_the_inverse_bandwidth() {
        let ds = read_sample!("MRS_2H");
        assert!((ds.dwell_time_s - 1.0 / 5000.0).abs() < 1e-12);
    }

    #[test]
    fn required_metadata_is_populated() {
        let ds = read_sample!("MRS_2H");
        assert_eq!(ds.meta.resonant_nucleus, vec!["2H".to_string()]);
        assert!((ds.meta.spectrometer_frequency_mhz[0] - 19.5934).abs() < 1e-3);
    }

    #[test]
    fn does_not_rechop_already_dechopped_data() {
        // Spec §4.1: MRS_2H has data_collect_type == 0, so MNUtils' rule would
        // say "chopped", but fidall already de-chopped. Every transient must
        // keep the same sign at t = 0.
        let ds = read_sample!("MRS_2H");
        let n_negative = (0..64)
            .filter(|&r| ds.data[[0, 0, 0, 0, r]].re < 0.0)
            .count();
        assert_eq!(n_negative, 0, "sign alternation implies chop was applied");
    }

    #[test]
    fn unlocalised_dimensions_use_the_spec_default() {
        // MRS_2H: user14 == 91, so all three dimensions are unlocalised.
        let ds = read_sample!("MRS_2H");
        for c in 0..3 {
            let norm = (ds.affine[0][c].powi(2)
                + ds.affine[1][c].powi(2)
                + ds.affine[2][c].powi(2))
            .sqrt();
            assert!(
                (norm - 10000.0).abs() < 1e-6,
                "column {c} extent {norm}, expected the 10 m default"
            );
        }
    }

    #[test]
    fn slab_uses_its_prescribed_thickness() {
        let ds = read_sample!("MRS_2H_slab");
        let z = (ds.affine[0][2].powi(2) + ds.affine[1][2].powi(2) + ds.affine[2][2].powi(2))
            .sqrt();
        assert!((z - 80.0).abs() < 1e-6, "slab thickness {z}");
    }

    #[test]
    fn conjugation_is_not_applied() {
        // Spec §4.1: the dominant peak of MRS_2H_slab sits at +53.71 Hz and
        // the FID must rotate counter-clockwise there, with a ratio above 3.
        let ds = read_sample!("MRS_2H_slab");
        let n = 2048usize;
        let dt = ds.dwell_time_s;
        let x = 53.71f64;
        let mut ccw = num_complex::Complex::<f64>::new(0.0, 0.0);
        let mut cw = num_complex::Complex::<f64>::new(0.0, 0.0);
        for t in 0..n {
            let v = ds.data[[0, 0, 0, t, 0]];
            let v = num_complex::Complex::new(v.re as f64, v.im as f64);
            let phase = 2.0 * std::f64::consts::PI * x * t as f64 * dt;
            ccw += v * num_complex::Complex::new(0.0, -phase).exp();
            cw += v * num_complex::Complex::new(0.0, phase).exp();
        }
        assert!(
            ccw.norm() / cw.norm() > 3.0,
            "ccw/cw = {}, expected > 3 (unconjugated)",
            ccw.norm() / cw.norm()
        );
    }
}
