//! Maps `raw2nii_core::Raw2NiiError` onto a small Python exception
//! hierarchy so callers can catch broadly (`Raw2NiiError`) or narrowly
//! (e.g. `UnsupportedFormatError`).

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};

create_exception!(raw2nii, Raw2NiiError, PyException);
create_exception!(raw2nii, UnsupportedFormatError, Raw2NiiError);
create_exception!(raw2nii, MissingDataError, Raw2NiiError);
create_exception!(raw2nii, MissingMetadataError, Raw2NiiError);
create_exception!(raw2nii, GeometryError, Raw2NiiError);
create_exception!(raw2nii, DimensionMismatchError, Raw2NiiError);
create_exception!(raw2nii, BackendError, Raw2NiiError);
create_exception!(raw2nii, IoError, Raw2NiiError);

pub fn to_pyerr(err: raw2nii_core::Raw2NiiError) -> PyErr {
    use raw2nii_core::Raw2NiiError as E;
    let msg = err.to_string();
    match err {
        E::UnsupportedFormat(_) => UnsupportedFormatError::new_err(msg),
        E::MissingData(_) => MissingDataError::new_err(msg),
        E::MissingMetadata(_) => MissingMetadataError::new_err(msg),
        E::Geometry(_) => GeometryError::new_err(msg),
        E::DimensionMismatch { .. } => DimensionMismatchError::new_err(msg),
        E::Backend(_) => BackendError::new_err(msg),
        E::Io(_) => IoError::new_err(msg),
    }
}
