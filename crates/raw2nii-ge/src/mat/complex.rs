//! Complex array reading.
//!
//! MATLAB v7.3 writes complex data as an HDF5 compound type with `real` and
//! `imag` members. HDF5 hands us elements in C order over the HDF5 shape,
//! which is the reverse of the MATLAB shape — so reading into an array of
//! the reversed shape and then reversing the axes yields MATLAB order
//! without moving any data twice.

use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex;

use super::{MatError, MatFile};

#[derive(hdf5_metno::H5Type, Clone, Copy, Debug)]
#[repr(C)]
struct C32 {
    real: f32,
    imag: f32,
}

impl MatFile {
    /// Read a complex dataset in MATLAB dimension order, right-padded with
    /// trailing singleton dimensions until it has at least `min_ndim` axes.
    pub fn complex_array(
        &self,
        path: &str,
        min_ndim: usize,
    ) -> Result<ArrayD<Complex<f32>>, MatError> {
        let ds = self
            .file_ref()
            .dataset(path)
            .map_err(|_| MatError::NotFound(path.to_string()))?;

        let hdf5_shape = ds.shape();
        let raw = ds.read_raw::<C32>().map_err(|source| MatError::Type {
            path: path.to_string(),
            want: "complex compound {real, imag}",
            source,
        })?;

        let data: Vec<Complex<f32>> = raw
            .into_iter()
            .map(|c| Complex::new(c.real, c.imag))
            .collect();

        // Build over the HDF5 shape (C order), then reverse axes to get
        // MATLAB order. A well-formed .mat always gives a matching element
        // count; a corrupt or truncated file might not, so this is a proper
        // error rather than a panic on untrusted input.
        let arr = ArrayD::from_shape_vec(IxDyn(&hdf5_shape), data).map_err(|source| {
            MatError::Shape {
                path: path.to_string(),
                source,
            }
        })?;
        let mut arr = arr.reversed_axes();

        while arr.ndim() < min_ndim {
            let new_axis = arr.ndim();
            arr = arr.insert_axis(Axis(new_axis));
        }

        Ok(arr.as_standard_layout().to_owned())
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
    fn reads_svs_fid_in_matlab_order() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        let a = f.complex_array("/fid", 2).unwrap();
        assert_eq!(a.shape(), &[64, 2048]);
        // First transient, first sample: real positive, imag negative.
        assert!(a[[0, 0]].re > 0.0);
        assert!(a[[0, 0]].im < 0.0);
    }

    #[test]
    fn reads_mrsi_spec_in_matlab_order() {
        let f = MatFile::open(&sample!("MRSI_2H")).unwrap();
        let a = f.complex_array("/spec", 4).unwrap();
        assert_eq!(a.shape(), &[700, 16, 16, 16]);
    }

    #[test]
    fn right_pads_dropped_trailing_singletons() {
        // MRSI_13C /spec is stored 3-D (544 x 8 x 8); nn says the fourth
        // dimension exists with length 1.
        let f = MatFile::open(&sample!("MRSI_13C")).unwrap();
        let a = f.complex_array("/spec", 4).unwrap();
        assert_eq!(a.shape(), &[544, 8, 8, 1]);
    }

    #[test]
    fn element_count_is_preserved() {
        let f = MatFile::open(&sample!("MRS_2H")).unwrap();
        let a = f.complex_array("/fid", 2).unwrap();
        assert_eq!(a.len(), 64 * 2048);
    }
}
