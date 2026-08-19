//! GE-specific backends. Currently the fidall `.mat` v7.3 container.

pub mod flavor;
pub mod header;
pub mod mat;
pub mod read;
pub mod samples;

use std::path::Path;

use raw2nii_core::backend::{Backend, Confidence};
use raw2nii_core::dataset::MrsDataset;
use raw2nii_core::error::{Raw2NiiError, Result};

use flavor::Flavor;
use header::geometry::UNLOCALISED_MM;
use header::GeHeader;
use mat::MatFile;

pub struct GeMatBackend;

impl Backend for GeMatBackend {
    fn name(&self) -> &'static str {
        "ge-fidall-mat"
    }

    fn probe(&self, path: &Path) -> Confidence {
        if path.extension().and_then(|s| s.to_str()) != Some("mat") {
            return Confidence::No;
        }
        match MatFile::open(path) {
            Ok(m) if m.has("/h") && flavor::detect(&m).is_some() => Confidence::Yes,
            Ok(_) => Confidence::Maybe,
            Err(_) => Confidence::No,
        }
    }

    fn convert(&self, path: &Path) -> Result<Vec<MrsDataset>> {
        let m = MatFile::open(path)
            .map_err(|e| Raw2NiiError::Backend(format!("{}: {e}", path.display())))?;
        let h = GeHeader::from_mat(&m)
            .map_err(|e| Raw2NiiError::MissingMetadata(format!("/h: {e}")))?;

        let ds = match flavor::detect(&m) {
            Some(Flavor::Svs) => read::read_svs(&m, &h)?,
            Some(Flavor::Mrsi) => read::read_mrsi(&m, &h)?,
            None => {
                return Err(Raw2NiiError::UnsupportedFormat(format!(
                    "{}: neither /fid+/par/samples nor /spec+/nn+/dim present",
                    path.display()
                )))
            }
        };
        Ok(vec![ds])
    }
}

/// Output filename stem, spec §7.1:
/// `exam{ex_no}_series{se_no:02}_{nucleus}_{type}`.
///
/// `unlocalised` applies to SVS only; `n_slices` applies to MRSI only.
pub fn output_stem(h: &GeHeader, flavor: Flavor, unlocalised: bool, n_slices: usize) -> String {
    let (nucleus, _) = h.nucleus_name();
    let kind = match flavor {
        Flavor::Svs if unlocalised => "svs-unloc",
        Flavor::Svs => "svs-slab",
        Flavor::Mrsi if n_slices <= 1 => "mrsi2d",
        Flavor::Mrsi => "mrsi3d",
    };
    format!(
        "exam{:05}_series{:02}_{}_{}",
        h.exam_number, h.series_number, nucleus, kind
    )
}

/// Whether an SVS dataset came out unlocalised, for naming purposes.
pub fn is_unlocalised(ds: &MrsDataset) -> bool {
    let z = (ds.affine[0][2].powi(2) + ds.affine[1][2].powi(2) + ds.affine[2][2].powi(2)).sqrt();
    (z - UNLOCALISED_MM).abs() < 1e-6
}

#[cfg(test)]
mod backend_tests {
    use super::*;
    use crate::samples::sample_mat;
    use raw2nii_core::backend::{Backend, Confidence, Registry};

    macro_rules! path {
        ($name:expr) => {
            match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            }
        };
    }

    #[test]
    fn claims_a_fidall_mat() {
        let b = GeMatBackend;
        assert_eq!(b.probe(&path!("MRS_2H")), Confidence::Yes);
    }

    #[test]
    fn declines_a_directory_without_a_mat() {
        // BS_prescan_13C ships no .mat, only ScanArchive .h5 files.
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/datasets/BS_prescan_13C");
        if !dir.exists() {
            eprintln!("SKIP: tests/datasets absent");
            return;
        }
        let h5 = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|s| s.to_str()) == Some("h5"))
            .expect("the prescan directory contains .h5 files");
        assert_eq!(GeMatBackend.probe(&h5), Confidence::No);
    }

    #[test]
    fn converts_svs_end_to_end() {
        let out = GeMatBackend.convert(&path!("MRS_2H")).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data.shape(), &[1, 1, 1, 2048, 64]);
    }

    #[test]
    fn converts_mrsi_end_to_end() {
        let out = GeMatBackend.convert(&path!("MRSI_13C")).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data.shape(), &[8, 8, 1, 544]);
    }

    #[test]
    fn registry_selects_the_highest_confidence_backend() {
        let reg = Registry::default().with_backend(Box::new(GeMatBackend));
        let b = reg.select(&path!("MRS_2H")).expect("a backend must claim it");
        assert_eq!(b.name(), "ge-fidall-mat");
    }

    #[test]
    fn registry_finds_a_backend_by_name() {
        let reg = Registry::default().with_backend(Box::new(GeMatBackend));
        assert!(reg.by_name("ge-fidall-mat").is_some());
        assert!(reg.by_name("siemens-twix").is_none());
    }

    #[test]
    fn unlocalised_svs_stem() {
        let m = crate::mat::MatFile::open(&path!("MRS_2H")).unwrap();
        let h = crate::header::GeHeader::from_mat(&m).unwrap();
        // n_slices is ignored for SVS.
        assert_eq!(
            output_stem(&h, crate::flavor::Flavor::Svs, true, 1),
            "exam20000_series06_2H_svs-unloc"
        );
    }

    #[test]
    fn slab_svs_stem() {
        let m = crate::mat::MatFile::open(&path!("MRS_2H_slab")).unwrap();
        let h = crate::header::GeHeader::from_mat(&m).unwrap();
        assert_eq!(
            output_stem(&h, crate::flavor::Flavor::Svs, false, 1),
            "exam20000_series07_2H_svs-slab"
        );
    }

    #[test]
    fn mrsi_stems_distinguish_2d_from_3d() {
        // MRSI_13C has a single slice; MRSI_2H has 16.
        let m = crate::mat::MatFile::open(&path!("MRSI_13C")).unwrap();
        let h = crate::header::GeHeader::from_mat(&m).unwrap();
        assert_eq!(
            output_stem(&h, crate::flavor::Flavor::Mrsi, false, 1),
            "exam04874_series10_13C_mrsi2d"
        );

        let m = crate::mat::MatFile::open(&path!("MRSI_2H")).unwrap();
        let h = crate::header::GeHeader::from_mat(&m).unwrap();
        assert_eq!(
            output_stem(&h, crate::flavor::Flavor::Mrsi, false, 16),
            "exam15732_series05_2H_mrsi3d"
        );
    }
}
