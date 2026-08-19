//! Python bindings for raw2nii: `read()` a vendor raw file into `Dataset`
//! objects (data as a zero-copy numpy array), `write()` one back out, or
//! `convert()` a file straight to disk using the same naming rules as the
//! CLI.

// pyo3's `create_exception!` macro expands to code gated on a `gil-refs`
// cfg that this crate never sets; harmless, but noisy under `-D warnings`.
#![allow(unexpected_cfgs)]
// pyo3's `#[pyfunction]`/`#[pymethods]` expansion emits a `?`-driven `PyErr`
// conversion clippy can't see is a no-op at the macro-generated call site.
#![allow(clippy::useless_conversion)]

mod dataset;
mod error;

use std::path::{Path, PathBuf};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use raw2nii_convert::{convert_one, Outcome, OutputFormat};
use raw2nii_core::backend::Registry;
use raw2nii_core::dataset::MrsDataset;
use raw2nii_ge::{is_unlocalised, output_stem, GeMatBackend};

use dataset::PyDataset;
use error::{
    to_pyerr, BackendError, DimensionMismatchError, GeometryError, IoError, MissingDataError,
    MissingMetadataError, Raw2NiiError, UnsupportedFormatError,
};

fn registry() -> Registry {
    Registry::default().with_backend(Box::new(GeMatBackend))
}

/// Re-derive the spec §7.1 output stem for one converted dataset. Kept in
/// lockstep with `raw2nii-cli`'s `convert::stem_for` — both read the same
/// GE-specific naming inputs back off the source file.
fn stem_for(input: &Path, ds: &MrsDataset) -> PyResult<String> {
    use raw2nii_ge::flavor::{detect, Flavor};
    use raw2nii_ge::header::GeHeader;
    use raw2nii_ge::mat::MatFile;

    let m = MatFile::open(input).map_err(|e| BackendError::new_err(e.to_string()))?;
    let h = GeHeader::from_mat(&m).map_err(|e| BackendError::new_err(e.to_string()))?;
    let flavor = detect(&m).ok_or_else(|| {
        UnsupportedFormatError::new_err(format!(
            "{}: unrecognised fidall .mat layout",
            input.display()
        ))
    })?;

    let n_slices = match flavor {
        Flavor::Svs => 1,
        Flavor::Mrsi => ds.data.shape()[2],
    };
    Ok(output_stem(&h, flavor, is_unlocalised(ds), n_slices))
}

/// Read a vendor raw file, returning one `Dataset` per FID it contains
/// (almost always one).
#[pyfunction]
fn read(py: Python<'_>, path: PathBuf) -> PyResult<Vec<Py<PyDataset>>> {
    let registry = registry();
    let backend = registry.select(&path).ok_or_else(|| {
        UnsupportedFormatError::new_err(format!("no backend can read {}", path.display()))
    })?;
    let datasets = backend.convert(&path).map_err(to_pyerr)?;
    datasets
        .into_iter()
        .map(|ds| Py::new(py, PyDataset::from_dataset(py, ds)?))
        .collect()
}

/// Convert a vendor raw file straight to a NIfTI-MRS file on disk,
/// returning the path(s) written. `format` is `"nii-gz"` (default) or
/// `"nii"`.
#[pyfunction]
#[pyo3(signature = (path, output_dir=None, format="nii-gz", compress_level=6, overwrite=false))]
fn convert(
    path: PathBuf,
    output_dir: Option<PathBuf>,
    format: &str,
    compress_level: u32,
    overwrite: bool,
) -> PyResult<Vec<String>> {
    let extension = match format {
        "nii-gz" => "nii.gz",
        "nii" => "nii",
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown format {other:?}, expected \"nii-gz\" or \"nii\""
            )))
        }
    };

    let registry = registry();
    let backend = registry.select(&path).ok_or_else(|| {
        UnsupportedFormatError::new_err(format!("no backend can read {}", path.display()))
    })?;
    let datasets = backend.convert(&path).map_err(to_pyerr)?;

    let dir = output_dir
        .or_else(|| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    let mut written = Vec::with_capacity(datasets.len());
    for ds in &datasets {
        let stem = stem_for(&path, ds)?;
        let out = dir.join(format!("{stem}.{extension}"));

        if out.exists() && !overwrite {
            return Err(IoError::new_err(format!("{} exists", out.display())));
        }
        for w in &ds.meta.warnings {
            tracing::warn!("{}: {w}", path.display());
        }

        std::fs::create_dir_all(&dir).map_err(|e| IoError::new_err(e.to_string()))?;
        raw2nii_core::write::write_file(ds, &out, compress_level).map_err(to_pyerr)?;
        written.push(out.to_string_lossy().into_owned());
    }

    Ok(written)
}

fn parse_format(format: &str) -> PyResult<OutputFormat> {
    match format {
        "nii-gz" => Ok(OutputFormat::NiiGz),
        "nii" => Ok(OutputFormat::Nii),
        other => Err(PyValueError::new_err(format!(
            "unknown format {other:?}, expected \"nii-gz\" or \"nii\""
        ))),
    }
}

fn outcome_to_dict(py: Python<'_>, outcome: &Outcome) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    match outcome {
        Outcome::Written { input, output } => {
            d.set_item("status", "written")?;
            d.set_item("input", input.to_string_lossy().into_owned())?;
            d.set_item("output", output.to_string_lossy().into_owned())?;
        }
        Outcome::Skipped { input, reason } => {
            d.set_item("status", "skipped")?;
            d.set_item("input", input.to_string_lossy().into_owned())?;
            d.set_item("reason", reason.clone())?;
        }
        Outcome::Failed { input, error } => {
            d.set_item("status", "failed")?;
            d.set_item("input", input.to_string_lossy().into_owned())?;
            d.set_item("error", error.clone())?;
        }
    }
    Ok(d.into())
}

/// Convert many files, optionally in parallel, exactly like the CLI's
/// `convert` subcommand: same naming rules, same dry-run/overwrite/format
/// semantics. Returns one status dict per input, in input order. Used by
/// the `raw2nii` console script; `convert()` above remains the simple
/// single-file library entry point.
#[pyfunction]
#[pyo3(signature = (inputs, output_dir=None, overwrite=false, compress_level=6, jobs=None, dry_run=false, format="nii-gz"))]
#[allow(clippy::too_many_arguments)]
fn _convert_many(
    py: Python<'_>,
    inputs: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    overwrite: bool,
    compress_level: u32,
    jobs: Option<usize>,
    dry_run: bool,
    format: &str,
) -> PyResult<Vec<Py<PyDict>>> {
    let fmt = parse_format(format)?;
    let registry = registry();

    let outcomes: Vec<Outcome> = py.allow_threads(|| {
        let run = || {
            inputs
                .par_iter()
                .map(|input| {
                    convert_one(
                        &registry,
                        input,
                        output_dir.as_deref(),
                        overwrite,
                        compress_level,
                        fmt,
                        !dry_run,
                    )
                })
                .collect::<Vec<_>>()
        };
        match jobs {
            Some(n) => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("thread pool builds with a caller-supplied thread count")
                .install(run),
            None => run(),
        }
    });

    outcomes.iter().map(|o| outcome_to_dict(py, o)).collect()
}

/// Build a verified `tar.zst` snapshot of `source_dir` (excluding
/// `exclude`, normally the output directory) at `archive_path`. Returns
/// whether the archive verified against the source files on disk.
#[pyfunction]
fn _build_and_verify_archive(
    source_dir: PathBuf,
    archive_path: PathBuf,
    exclude: PathBuf,
) -> PyResult<bool> {
    raw2nii_convert::archive::build_archive(&source_dir, &archive_path, &exclude)
        .map_err(|e| IoError::new_err(e.to_string()))?;
    raw2nii_convert::archive::verify_archive(&archive_path, &source_dir, &exclude)
        .map_err(|e| IoError::new_err(e.to_string()))
}

#[pymodule]
fn raw2nii(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(read, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(convert, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_convert_many, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_build_and_verify_archive, m)?)?;
    m.add_class::<PyDataset>()?;

    m.add("Raw2NiiError", py.get_type_bound::<Raw2NiiError>())?;
    m.add(
        "UnsupportedFormatError",
        py.get_type_bound::<UnsupportedFormatError>(),
    )?;
    m.add("MissingDataError", py.get_type_bound::<MissingDataError>())?;
    m.add(
        "MissingMetadataError",
        py.get_type_bound::<MissingMetadataError>(),
    )?;
    m.add("GeometryError", py.get_type_bound::<GeometryError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type_bound::<DimensionMismatchError>(),
    )?;
    m.add("BackendError", py.get_type_bound::<BackendError>())?;
    m.add("IoError", py.get_type_bound::<IoError>())?;
    Ok(())
}
