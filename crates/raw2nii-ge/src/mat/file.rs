//! MATLAB v7.3 (HDF5) reading.
//!
//! This module knows HDF5 and MATLAB storage conventions. It knows nothing
//! about MRS. Two conventions matter:
//!
//! 1. MATLAB stores array dimensions reversed relative to HDF5, so a dataset
//!    an HDF5 tool reports as `(2048, 64)` is a MATLAB `64 x 2048` array.
//! 2. MATLAB strings are `uint16` arrays of character codes.

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum MatError {
    #[error("cannot open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: hdf5_metno::Error,
    },
    #[error("dataset not found: {0}")]
    NotFound(String),
    #[error("dataset {path} is not readable as {want}: {source}")]
    Type {
        path: String,
        want: &'static str,
        #[source]
        source: hdf5_metno::Error,
    },
    #[error("dataset {path} has {n} elements, expected exactly one")]
    NotScalar { path: String, n: usize },
}

pub struct MatFile {
    file: hdf5_metno::File,
}

impl MatFile {
    pub fn open(path: &Path) -> Result<Self, MatError> {
        let file = hdf5_metno::File::open(path).map_err(|source| MatError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { file })
    }

    fn dataset(&self, path: &str) -> Result<hdf5_metno::Dataset, MatError> {
        self.file
            .dataset(path)
            .map_err(|_| MatError::NotFound(path.to_string()))
    }

    pub fn has(&self, path: &str) -> bool {
        self.file.dataset(path).is_ok() || self.file.group(path).is_ok()
    }

    /// MATLAB-order shape, i.e. the HDF5 shape reversed.
    pub fn matlab_shape(&self, path: &str) -> Result<Vec<usize>, MatError> {
        let mut shape = self.dataset(path)?.shape();
        shape.reverse();
        Ok(shape)
    }

    pub fn vec_f64(&self, path: &str) -> Result<Vec<f64>, MatError> {
        let ds = self.dataset(path)?;
        ds.read_raw::<f64>().map_err(|source| MatError::Type {
            path: path.to_string(),
            want: "f64",
            source,
        })
    }

    pub fn scalar_f64(&self, path: &str) -> Result<f64, MatError> {
        let v = self.vec_f64(path)?;
        match v.len() {
            1 => Ok(v[0]),
            n => Err(MatError::NotScalar {
                path: path.to_string(),
                n,
            }),
        }
    }

    /// MATLAB character arrays are stored as `uint16` character codes.
    pub fn string(&self, path: &str) -> Result<String, MatError> {
        let ds = self.dataset(path)?;
        let codes = ds.read_raw::<u16>().map_err(|source| MatError::Type {
            path: path.to_string(),
            want: "u16 character codes",
            source,
        })?;
        Ok(codes
            .into_iter()
            .take_while(|&c| c != 0)
            .filter_map(|c| char::from_u32(c as u32))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples::sample_mat;

    macro_rules! sample {
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
    fn reads_scalar_parameters() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        assert_eq!(f.scalar_f64("/par/samples").unwrap(), 2048.0);
        assert_eq!(f.scalar_f64("/par/rows").unwrap(), 64.0);
        assert_eq!(f.scalar_f64("/par/bw").unwrap(), 5000.0);
        assert_eq!(f.scalar_f64("/h/exam/ex_no").unwrap(), 20000.0);
        assert_eq!(f.scalar_f64("/h/series/se_no").unwrap(), 6.0);
        assert_eq!(f.scalar_f64("/h/image/specnuc").unwrap(), 2.0);
    }

    #[test]
    fn reverses_matlab_dimensions() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        // h5dump reports (2048, 64); MATLAB order is rows x samples.
        assert_eq!(f.matlab_shape("/fid").unwrap(), vec![64, 2048]);
    }

    #[test]
    fn reverses_dimensions_for_mrsi() {
        let f = MatFile::open(&sample!("MRSI_2H")).unwrap();
        assert_eq!(f.matlab_shape("/spec").unwrap(), vec![700, 16, 16, 16]);
        assert_eq!(
            f.vec_f64("/zf").unwrap(),
            vec![700.0, 16.0, 16.0, 16.0, 1.0, 1.0]
        );
    }

    #[test]
    fn reads_matlab_string() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        assert_eq!(f.string("/h/image/psdname").unwrap(), "fidall2");
    }

    #[test]
    fn missing_path_is_an_error() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        assert!(matches!(
            f.scalar_f64("/par/no_such_field"),
            Err(MatError::NotFound(_))
        ));
    }

    #[test]
    fn has_reports_presence() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        assert!(f.has("/fid"));
        assert!(!f.has("/nn"));
    }
}
