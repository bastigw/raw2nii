//! Spectroscopic imaging reader.
//!
//! `/spec` is frequency-domain and NIfTI-MRS requires time-domain data, so
//! every voxel spectrum is inverse transformed with the xmris recipe. The grid
//! is whatever is actually stored — the zero-filled size `zf`, never the
//! acquired size `nn` (spec §2.2).

use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use raw2nii_core::dataset::{Metadata, MrsDataset};
use raw2nii_core::error::{Raw2NiiError, Result};
use raw2nii_core::fft::spec_to_fid;
use serde_json::json;

use crate::header::geometry::{build_affine, mrsi_localisation};
use crate::header::GeHeader;
use crate::mat::MatFile;

pub fn read_mrsi(m: &MatFile, h: &GeHeader) -> Result<MrsDataset> {
    let spec = m
        .complex_array("/spec", 4)
        .map_err(|e| Raw2NiiError::MissingData(format!("/spec: {e}")))?;

    let shape = spec.shape().to_vec();
    let (nspec, nx, ny, nz) = (shape[0], shape[1], shape[2], shape[3]);

    // The stored grid is authoritative. `nn` records what was acquired and is
    // read only to report zero-filling.
    let acquired = m.vec_f64("/nn").ok();

    let hz = m
        .vec_f64("/hz")
        .map_err(|e| Raw2NiiError::MissingMetadata(format!("/hz: {e}")))?;
    if hz.len() < 2 {
        return Err(Raw2NiiError::MissingMetadata(
            "/hz needs at least two points to derive a dwell time".to_string(),
        ));
    }
    let df = (hz[1] - hz[0]).abs();
    if df <= 0.0 {
        return Err(Raw2NiiError::MissingMetadata(format!(
            "/hz spacing must be positive, got {df}"
        )));
    }
    let dwell = 1.0 / (nspec as f64 * df);

    let mut data = ArrayD::<Complex<f32>>::zeros(IxDyn(&[nx, ny, nz, nspec]));
    let mut column = vec![Complex::<f32>::new(0.0, 0.0); nspec];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                for (t, slot) in column.iter_mut().enumerate() {
                    *slot = spec[[t, x, y, z]];
                }
                for (t, v) in spec_to_fid(&column).into_iter().enumerate() {
                    data[[x, y, z, t]] = v;
                }
            }
        }
    }

    let loc = mrsi_localisation(h, [nx, ny, nz]);
    let affine = build_affine(h, loc.extents_mm, [nx, ny, nz]);

    let (nucleus, nucleus_warning) = h.nucleus_name();
    let mut warnings = loc.warnings;
    warnings.extend(nucleus_warning);

    let f0_hz = m
        .scalar_f64("/par/synthesizer_frequency")
        .or_else(|_| m.scalar_f64("/h/rdb_hdr/ps_mps_freq"))
        .map_err(|e| Raw2NiiError::MissingMetadata(format!("centre frequency: {e}")))?;

    let mut processing = vec![json!({
        "Method": "Fourier transform",
        "Details": "frequency to time domain: ifftshift, ifft (ortho)"
    })];
    if let Some(nn) = &acquired {
        let stored = [nspec as f64, nx as f64, ny as f64, nz as f64];
        if nn.len() >= 4 && nn[..4] != stored {
            processing.push(json!({
                "Method": "zero-fill",
                "Details": format!(
                    "acquired {:?} reconstructed to {:?}",
                    &nn[..4], stored
                )
            }));
        }
    }

    let mut extra = serde_json::Map::new();
    extra.insert("Manufacturer".to_string(), json!("GE"));
    extra.insert("PulseSequenceFile".to_string(), json!(h.psdname));
    extra.insert("SpectralWidth".to_string(), json!(1.0 / dwell));
    extra.insert("ProcessingApplied".to_string(), json!(processing));
    if let Some(dt) = h.scan_datetime_iso() {
        extra.insert("ConversionTime".to_string(), json!(dt));
    }

    Ok(MrsDataset {
        data,
        tags: [None, None, None],
        affine,
        dwell_time_s: dwell,
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
            read_mrsi(&m, &h).unwrap()
        }};
    }

    #[test]
    fn uses_the_zero_filled_grid_not_the_acquired_one() {
        // MRSI_2H acquired 10^3 (nn) and stores 16^3 (zf).
        let ds = read_sample!("MRSI_2H");
        assert_eq!(ds.data.shape(), &[16, 16, 16, 700]);
        assert!(ds.validate().is_ok());
    }

    #[test]
    fn handles_2d_mrsi_with_a_single_slice() {
        let ds = read_sample!("MRSI_13C");
        assert_eq!(ds.data.shape(), &[8, 8, 1, 544]);
    }

    #[test]
    fn has_no_dimension_tags() {
        let ds = read_sample!("MRSI_13C");
        assert_eq!(ds.tags, [None, None, None]);
    }

    #[test]
    fn voxel_size_is_dfov_over_grid() {
        // MRSI_2H: dfov 200 over a 16 grid -> 12.5 mm.
        let ds = read_sample!("MRSI_2H");
        let x = (ds.affine[0][0].powi(2) + ds.affine[1][0].powi(2) + ds.affine[2][0].powi(2))
            .sqrt();
        assert!((x - 12.5).abs() < 1e-6, "voxel extent {x}");
    }

    #[test]
    fn thirteen_c_voxel_size_uses_its_own_dfov() {
        // MRSI_13C: dfov 300 over an 8 grid -> 37.5 mm.
        let ds = read_sample!("MRSI_13C");
        let x = (ds.affine[0][0].powi(2) + ds.affine[1][0].powi(2) + ds.affine[2][0].powi(2))
            .sqrt();
        assert!((x - 37.5).abs() < 1e-6, "voxel extent {x}");
    }

    #[test]
    fn records_zero_filling_and_the_ignored_user14() {
        let ds = read_sample!("MRSI_2H");
        assert!(
            ds.meta.warnings.iter().any(|w| w.contains("user14")),
            "the ignored user14 value must be recorded"
        );
        let applied = ds.meta.extra.get("ProcessingApplied").unwrap();
        assert!(applied.to_string().contains("zero-fill"));
    }

    #[test]
    fn output_is_time_domain_starting_at_maximum_signal() {
        // A FID peaks at t = 0; a spectrum does not.
        let ds = read_sample!("MRSI_13C");
        let first = ds.data[[4, 4, 0, 0]].norm();
        let late = ds.data[[4, 4, 0, 500]].norm();
        assert!(
            first > late,
            "expected a decaying FID, got |t0| = {first}, |t500| = {late}"
        );
    }
}
