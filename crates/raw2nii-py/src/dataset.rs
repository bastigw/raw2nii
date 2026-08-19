//! The Python-visible `Dataset` type: a thin wrapper around
//! `raw2nii_core::MrsDataset` that hands `data` to Python as a zero-copy
//! numpy array and everything else as plain Python values.

use ndarray::ArrayD;
use num_complex::Complex32;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use raw2nii_core::dataset::{DimTag, Metadata, MrsDataset};

fn dim_tag_from_str(s: &str) -> PyResult<DimTag> {
    Ok(match s {
        "DIM_COIL" => DimTag::Coil,
        "DIM_DYN" => DimTag::Dyn,
        "DIM_PHASE_CYCLE" => DimTag::PhaseCycle,
        "DIM_EDIT" => DimTag::Edit,
        "DIM_MEAS" => DimTag::Meas,
        "DIM_ISIS" => DimTag::Isis,
        "DIM_METCYCLE" => DimTag::MetCycle,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown dimension tag {other:?}"
            )))
        }
    })
}

#[pyclass(name = "Dataset", module = "raw2nii")]
pub struct PyDataset {
    data: Py<PyArrayDyn<Complex32>>,
    #[pyo3(get, set)]
    tags: [Option<String>; 3],
    #[pyo3(get, set)]
    affine: [[f64; 4]; 4],
    #[pyo3(get, set)]
    dwell_time_s: f64,
    #[pyo3(get, set)]
    spectrometer_frequency_mhz: Vec<f64>,
    #[pyo3(get, set)]
    resonant_nucleus: Vec<String>,
    extra: serde_json::Map<String, serde_json::Value>,
    #[pyo3(get)]
    warnings: Vec<String>,
}

impl PyDataset {
    /// Moves `ds.data` into a numpy array without copying: `ArrayD`'s
    /// backing `Vec` is handed straight to numpy's allocator.
    pub fn from_dataset(py: Python<'_>, ds: MrsDataset) -> PyResult<Self> {
        let MrsDataset {
            data,
            tags,
            affine,
            dwell_time_s,
            meta,
        } = ds;

        let array = PyArrayDyn::from_owned_array_bound(py, data);
        Ok(Self {
            data: array.unbind(),
            tags: tags.map(|t| t.map(|t| t.as_str().to_string())),
            affine,
            dwell_time_s,
            spectrometer_frequency_mhz: meta.spectrometer_frequency_mhz,
            resonant_nucleus: meta.resonant_nucleus,
            extra: meta.extra,
            warnings: meta.warnings,
        })
    }

    /// Copies the numpy array back into an owned `ArrayD` (unavoidable:
    /// Python retains ownership of the buffer) and reassembles an
    /// `MrsDataset` for the writer.
    pub fn to_dataset(&self, py: Python<'_>) -> PyResult<MrsDataset> {
        let data: ArrayD<Complex32> = self.data.bind(py).to_owned_array();
        let tags = [
            self.tags[0].as_deref().map(dim_tag_from_str).transpose()?,
            self.tags[1].as_deref().map(dim_tag_from_str).transpose()?,
            self.tags[2].as_deref().map(dim_tag_from_str).transpose()?,
        ];
        Ok(MrsDataset {
            data,
            tags,
            affine: self.affine,
            dwell_time_s: self.dwell_time_s,
            meta: Metadata {
                spectrometer_frequency_mhz: self.spectrometer_frequency_mhz.clone(),
                resonant_nucleus: self.resonant_nucleus.clone(),
                extra: self.extra.clone(),
                warnings: self.warnings.clone(),
            },
        })
    }
}

#[pymethods]
impl PyDataset {
    /// Complex64 (numpy dtype, i.e. two float32s) array with axes
    /// `(x, y, z, t, [dim5, dim6, dim7])`.
    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyArrayDyn<Complex32>> {
        self.data.clone_ref(py)
    }

    /// Arbitrary extra NIfTI-MRS JSON-extension keys, as a plain dict.
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.extra)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[setter]
    fn set_extra(&mut self, value: Bound<'_, PyDict>) -> PyResult<()> {
        let json: serde_json::Value = pythonize::depythonize(value.as_any())?;
        self.extra = match json {
            serde_json::Value::Object(map) => map,
            _ => return Err(PyValueError::new_err("extra must be a dict")),
        };
        Ok(())
    }

    /// Write this dataset to a NIfTI-MRS file. `path` should end in
    /// `.nii` or `.nii.gz`; the latter is gzip-compressed at
    /// `compress_level` (0-9).
    #[pyo3(signature = (path, compress_level=6))]
    fn write(&self, py: Python<'_>, path: std::path::PathBuf, compress_level: u32) -> PyResult<()> {
        let ds = self.to_dataset(py)?;
        raw2nii_core::write::write_file(&ds, &path, compress_level).map_err(crate::error::to_pyerr)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let shape = self.data.bind(py).shape().to_vec();
        format!(
            "Dataset(shape={shape:?}, nucleus={:?}, dwell_time_s={})",
            self.resonant_nucleus, self.dwell_time_s
        )
    }
}
