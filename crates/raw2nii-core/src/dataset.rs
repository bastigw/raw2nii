//! The interchange contract. Every backend produces one of these; the NIfTI
//! writer consumes nothing else.

use ndarray::ArrayD;
use num_complex::Complex;

use crate::error::{Raw2NiiError, Result};

/// Tags for NIfTI-MRS dimensions 5, 6 and 7 (spec §2.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimTag {
    Coil,
    Dyn,
    PhaseCycle,
    Edit,
    Meas,
    Isis,
    MetCycle,
}

impl DimTag {
    pub fn as_str(self) -> &'static str {
        match self {
            DimTag::Coil => "DIM_COIL",
            DimTag::Dyn => "DIM_DYN",
            DimTag::PhaseCycle => "DIM_PHASE_CYCLE",
            DimTag::Edit => "DIM_EDIT",
            DimTag::Meas => "DIM_MEAS",
            DimTag::Isis => "DIM_ISIS",
            DimTag::MetCycle => "DIM_METCYCLE",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Metadata {
    /// Spec §2.3.1 required key, in MHz.
    pub spectrometer_frequency_mhz: Vec<f64>,
    /// Spec §2.3.1 required key, e.g. "2H".
    pub resonant_nucleus: Vec<String>,
    /// Optional keys, written verbatim into the JSON extension.
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Non-fatal problems encountered while reading.
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MrsDataset {
    /// Axes are (x, y, z, t, [dim5, dim6, dim7]).
    pub data: ArrayD<Complex<f32>>,
    pub tags: [Option<DimTag>; 3],
    pub affine: [[f64; 4]; 4],
    pub dwell_time_s: f64,
    pub meta: Metadata,
}

pub fn identity_affine() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

impl MrsDataset {
    /// Enforce the invariants the writer relies on.
    pub fn validate(&self) -> Result<()> {
        let n = self.data.ndim();
        if !(4..=7).contains(&n) {
            return Err(Raw2NiiError::DimensionMismatch {
                expected: vec![4],
                actual: self.data.shape().to_vec(),
            });
        }
        if self.meta.resonant_nucleus.is_empty() {
            return Err(Raw2NiiError::MissingMetadata(
                "ResonantNucleus".to_string(),
            ));
        }
        if self.meta.spectrometer_frequency_mhz.is_empty() {
            return Err(Raw2NiiError::MissingMetadata(
                "SpectrometerFrequency".to_string(),
            ));
        }
        if !(self.dwell_time_s > 0.0) {
            return Err(Raw2NiiError::MissingMetadata(
                "dwell time must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn svs_like(shape: &[usize]) -> MrsDataset {
        MrsDataset {
            data: ArrayD::from_elem(IxDyn(shape), Complex::new(1.0, 0.0)),
            tags: [Some(DimTag::Dyn), None, None],
            affine: identity_affine(),
            dwell_time_s: 2e-4,
            meta: Metadata {
                spectrometer_frequency_mhz: vec![19.5934],
                resonant_nucleus: vec!["2H".to_string()],
                extra: serde_json::Map::new(),
                warnings: vec![],
            },
        }
    }

    #[test]
    fn dim_tags_render_spec_names() {
        assert_eq!(DimTag::Dyn.as_str(), "DIM_DYN");
        assert_eq!(DimTag::Coil.as_str(), "DIM_COIL");
    }

    #[test]
    fn accepts_valid_svs_dataset() {
        assert!(svs_like(&[1, 1, 1, 2048, 64]).validate().is_ok());
    }

    #[test]
    fn accepts_valid_mrsi_dataset() {
        let mut ds = svs_like(&[16, 16, 16, 700]);
        ds.tags = [None, None, None];
        assert!(ds.validate().is_ok());
    }

    #[test]
    fn rejects_fewer_than_four_dimensions() {
        let ds = svs_like(&[1, 1, 2048]);
        assert!(matches!(
            ds.validate(),
            Err(Raw2NiiError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_missing_required_metadata() {
        let mut ds = svs_like(&[1, 1, 1, 2048, 64]);
        ds.meta.resonant_nucleus.clear();
        assert!(matches!(
            ds.validate(),
            Err(Raw2NiiError::MissingMetadata(_))
        ));
    }

    #[test]
    fn rejects_non_positive_dwell_time() {
        let mut ds = svs_like(&[1, 1, 1, 2048, 64]);
        ds.dwell_time_s = 0.0;
        assert!(matches!(
            ds.validate(),
            Err(Raw2NiiError::MissingMetadata(_))
        ));
    }
}
