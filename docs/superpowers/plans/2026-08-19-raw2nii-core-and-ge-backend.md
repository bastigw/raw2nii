# raw2nii Core + GE `.mat` Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert GE fidall `.mat` files (SVS and MRSI) into conformant NIfTI-MRS v0.11 files, via a Rust library plus a minimal `raw2nii convert` CLI.

**Architecture:** A Cargo workspace. `raw2nii-core` owns the `MrsDataset` interchange type, the `Backend` trait, the FFT, the JSON header-extension builder and the NIfTI-2 writer. `raw2nii-ge` owns everything GE-specific: MATLAB v7.3 reading, header field access, flavor detection and the two readers. `raw2nii-cli` is a thin clap binary. Vendor knowledge never crosses into core; the writer only ever sees `MrsDataset`.

**Tech Stack:** Rust 2021, `hdf5-metno` (static libhdf5), `ndarray`, `num-complex`, `rustfft`, `serde_json`, `flate2` (zlib-ng), `thiserror`, `tracing`, `clap`.

**Spec:** `docs/superpowers/specs/2026-08-19-raw2nii-ge-mat-to-nifti-mrs-design.md`

## Global Constraints

- Rust edition 2021, minimum toolchain 1.75.
- `raw2nii-core` must not print, must not call `std::process::exit`, and must return `Result<_, Raw2NiiError>`. Only `raw2nii-cli` prints.
- `hdf5-metno` uses the `static` feature so libhdf5 is vendored — no system HDF5 dependency.
- NIfTI-MRS conformance target: **v0.11**, so `intent_name` is exactly `"mrs_v0_11"`.
- NIfTI-**2** format: `sizeof_hdr = 540`, magic `"n+2\0\r\n\x1a\n"`.
- Header extension `ecode = 44`, total extension size a positive multiple of 16.
- Data written as `DT_COMPLEX` (datatype code `32`, `bitpix = 64`), i.e. `Complex<f32>`.
- `xyzt_units = 10` (NIFTI_UNITS_MM `2` | NIFTI_UNITS_SEC `8`). Dwell time in `pixdim[4]`, in seconds.
- Unlocalised spatial dimensions use `pixdim = 10000.0` mm (spec §2.2).
- The `.mat` backend must **never** apply chop correction, must **never** read `rdb_hdr/data_collect_type`, and must **never** conjugate `/fid`. Established empirically in spec §4.1.
- MRSI geometry must **not** consult `rdb_hdr/user14` (spec §6.2 rule 3), but must log the value it ignored.
- FOV is `image/dfov`. Grid size is the actual stored dataspace (equals `zf`), never `nn`.
- `tests/datasets/` is gitignored and may be absent. Every test touching it must skip cleanly, never fail.
- No test may depend on `*_raw_fids.h5` — those files are not scanner output and have been removed.

---

## File Structure

| File | Responsibility |
|---|---|
| `Cargo.toml` | workspace manifest |
| `crates/raw2nii-core/src/error.rs` | `Raw2NiiError`, `Result` alias |
| `crates/raw2nii-core/src/dataset.rs` | `MrsDataset`, `DimTag`, `Metadata` |
| `crates/raw2nii-core/src/backend.rs` | `Backend` trait, `Confidence`, `Registry` |
| `crates/raw2nii-core/src/fft.rs` | `ifftshift`, `ifft_ortho`, `spec_to_fid` |
| `crates/raw2nii-core/src/geom.rs` | `Affine`, corner-to-srow algebra |
| `crates/raw2nii-core/src/meta/json.rs` | JSON header-extension builder |
| `crates/raw2nii-core/src/write/nifti.rs` | NIfTI-2 serialiser + gzip |
| `crates/raw2nii-ge/src/mat/file.rs` | MAT v7.3 open, path resolution, dim reversal |
| `crates/raw2nii-ge/src/mat/complex.rs` | compound `{real,imag}` → `Complex<f32>` |
| `crates/raw2nii-ge/src/header/fields.rs` | typed `/h` accessors + `FIELD_MAP` |
| `crates/raw2nii-ge/src/header/geometry.rs` | GE affine + localisation rules |
| `crates/raw2nii-ge/src/flavor.rs` | SVS vs MRSI detection |
| `crates/raw2nii-ge/src/read/svs.rs` | SVS reader |
| `crates/raw2nii-ge/src/read/mrsi.rs` | MRSI reader |
| `crates/raw2nii-ge/src/lib.rs` | `GeMatBackend` implementing `Backend` |
| `crates/raw2nii-cli/src/main.rs` | clap binary, `convert` subcommand |
| `crates/raw2nii-ge/tests/samples.rs` | sample-data helper, skips when absent |

---

### Task 1: Workspace skeleton and sample-data test helper

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`
- Create: `crates/raw2nii-core/Cargo.toml`, `crates/raw2nii-core/src/lib.rs`
- Create: `crates/raw2nii-ge/Cargo.toml`, `crates/raw2nii-ge/src/lib.rs`
- Create: `crates/raw2nii-ge/src/samples.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `raw2nii_ge::samples::sample_mat(name: &str) -> Option<std::path::PathBuf>` — returns the `ScanArchive*.mat` inside `tests/datasets/<name>/`, or `None` when the data is absent.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-ge/src/samples.rs`:

```rust
//! Locating the sample datasets. They are gitignored and may be absent,
//! so every consumer must handle `None` by skipping, never by failing.

use std::path::PathBuf;

/// Repository root, derived from this crate's manifest directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Path to the `ScanArchive*.mat` inside `tests/datasets/<name>/`.
pub fn sample_mat(name: &str) -> Option<PathBuf> {
    let dir = repo_root().join("tests").join("datasets").join(name);
    let entries = std::fs::read_dir(dir).ok()?;
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_svs_sample_or_skips() {
        match sample_mat("MRS_2H") {
            Some(p) => assert!(p.exists(), "returned path must exist: {p:?}"),
            None => eprintln!("SKIP: tests/datasets absent"),
        }
    }

    #[test]
    fn unknown_dataset_is_none() {
        assert!(sample_mat("NoSuchDataset").is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge`
Expected: FAIL — the workspace and crates do not exist yet (`error: could not find Cargo.toml`).

- [ ] **Step 3: Write minimal implementation**

Root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/raw2nii-core", "crates/raw2nii-ge", "crates/raw2nii-cli"]

[workspace.package]
edition = "2021"
rust-version = "1.75"
license = "MIT"

[workspace.dependencies]
ndarray = "0.16"
num-complex = "0.4"
thiserror = "2"
tracing = "0.1"
serde_json = "1"
```

`rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.75"
```

`crates/raw2nii-core/Cargo.toml`:

```toml
[package]
name = "raw2nii-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
ndarray.workspace = true
num-complex.workspace = true
thiserror.workspace = true
tracing.workspace = true
serde_json.workspace = true
```

`crates/raw2nii-core/src/lib.rs`:

```rust
//! Vendor-neutral core: the dataset contract, the backend seam, and the
//! NIfTI-MRS writer. This crate never prints and never exits.
```

`crates/raw2nii-ge/Cargo.toml`:

```toml
[package]
name = "raw2nii-ge"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
raw2nii-core = { path = "../raw2nii-core" }
ndarray.workspace = true
num-complex.workspace = true
thiserror.workspace = true
tracing.workspace = true
```

`crates/raw2nii-ge/src/lib.rs`:

```rust
//! GE-specific backends. Currently the fidall `.mat` v7.3 container.

pub mod samples;
```

Create `crates/raw2nii-cli/Cargo.toml` with a `[[bin]]` placeholder and `crates/raw2nii-cli/src/main.rs` containing `fn main() {}` so the workspace builds.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge`
Expected: PASS — 2 tests. `finds_svs_sample_or_skips` prints `SKIP` if the datasets are absent, which is a pass.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml rust-toolchain.toml crates/
git commit -m "feat: cargo workspace skeleton and sample-data helper"
```

---

### Task 2: MAT v7.3 reader — scalars, strings, dimension reversal

**Files:**
- Create: `crates/raw2nii-ge/src/mat/mod.rs`, `crates/raw2nii-ge/src/mat/file.rs`
- Modify: `crates/raw2nii-ge/Cargo.toml` (add `hdf5-metno`), `crates/raw2nii-ge/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MatFile::open(path: &Path) -> Result<MatFile, MatError>`
  - `MatFile::has(&self, path: &str) -> bool`
  - `MatFile::scalar_f64(&self, path: &str) -> Result<f64, MatError>`
  - `MatFile::vec_f64(&self, path: &str) -> Result<Vec<f64>, MatError>`
  - `MatFile::string(&self, path: &str) -> Result<String, MatError>`
  - `MatFile::matlab_shape(&self, path: &str) -> Result<Vec<usize>, MatError>`
  - `MatError` (enum, `thiserror`)

**Context:** MATLAB v7.3 is HDF5 with dimensions stored reversed. `h5dump` reporting `(2048, 64)` means the MATLAB array is `64 x 2048`. MATLAB strings are stored as `uint16` character-code arrays.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/mat/file.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge mat::file`
Expected: FAIL — `MatFile` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/raw2nii-ge/Cargo.toml`:

```toml
hdf5-metno = { version = "0.10", features = ["static"] }
```

Create `crates/raw2nii-ge/src/mat/mod.rs`:

```rust
pub mod file;

pub use file::{MatError, MatFile};
```

Create `crates/raw2nii-ge/src/mat/file.rs`:

```rust
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
```

Add `pub mod mat;` to `crates/raw2nii-ge/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge mat::file`
Expected: PASS — 6 tests.

Note: the first build compiles libhdf5 from source and takes several minutes. This is expected and happens once.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: MAT v7.3 reader with MATLAB dimension reversal"
```

---

### Task 3: MAT complex array reading

**Files:**
- Create: `crates/raw2nii-ge/src/mat/complex.rs`
- Modify: `crates/raw2nii-ge/src/mat/mod.rs`

**Interfaces:**
- Consumes: `MatFile`, `MatError` from Task 2.
- Produces: `MatFile::complex_array(&self, path: &str) -> Result<ArrayD<Complex<f32>>, MatError>` — returns the array in **MATLAB dimension order**, right-padded with trailing singleton dimensions to `min_ndim`.

**Context:** MATLAB v7.3 stores complex data as an HDF5 compound type with `real` and `imag` members. MATLAB drops trailing singleton dimensions, so `MRSI_13C` `/spec` is 3-D where the size vector says 4-D — the reader right-pads.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/mat/complex.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge mat::complex`
Expected: FAIL — `complex_array` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/mat/complex.rs`:

```rust
//! Complex array reading.
//!
//! MATLAB v7.3 writes complex data as an HDF5 compound type with `real` and
//! `imag` members. HDF5 hands us elements in C order over the HDF5 shape,
//! which is the reverse of the MATLAB shape — so reading into an array of
//! the reversed shape and then reversing the axes yields MATLAB order
//! without moving any data twice.

use ndarray::{ArrayD, IxDyn};
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
        // MATLAB order.
        let arr = ArrayD::from_shape_vec(IxDyn(&hdf5_shape), data).expect("shape matches length");
        let mut arr = arr.reversed_axes();

        while arr.ndim() < min_ndim {
            let mut shape = arr.shape().to_vec();
            shape.push(1);
            arr = arr
                .into_shape_with_order(IxDyn(&shape))
                .expect("appending a unit axis preserves length");
        }

        Ok(arr.as_standard_layout().to_owned())
    }
}
```

Add to `crates/raw2nii-ge/src/mat/file.rs`, inside `impl MatFile`:

```rust
    /// Internal accessor so sibling modules can reach the underlying file.
    pub(crate) fn file_ref(&self) -> &hdf5_metno::File {
        &self.file
    }
```

Add `pub mod complex;` to `crates/raw2nii-ge/src/mat/mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge mat::complex`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: complex array reading with trailing-singleton padding"
```

---

### Task 4: Core error type and `MrsDataset`

**Files:**
- Create: `crates/raw2nii-core/src/error.rs`, `crates/raw2nii-core/src/dataset.rs`
- Modify: `crates/raw2nii-core/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Raw2NiiError` with variants `UnsupportedFormat(String)`, `MissingData(String)`, `MissingMetadata(String)`, `Geometry(String)`, `DimensionMismatch { expected: Vec<usize>, actual: Vec<usize> }`, `Backend(String)`, `Io(std::io::Error)`
  - `type Result<T> = std::result::Result<T, Raw2NiiError>`
  - `DimTag` enum with `as_str()`
  - `Metadata { spectrometer_frequency_mhz: Vec<f64>, resonant_nucleus: Vec<String>, extra: serde_json::Map<String, serde_json::Value>, warnings: Vec<String> }`
  - `MrsDataset { data: ArrayD<Complex<f32>>, tags: [Option<DimTag>; 3], affine: [[f64; 4]; 4], dwell_time_s: f64, meta: Metadata }`
  - `MrsDataset::validate(&self) -> Result<()>`

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-core/src/dataset.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-core dataset`
Expected: FAIL — `MrsDataset` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-core/src/error.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Raw2NiiError {
    #[error("no backend can read {0}")]
    UnsupportedFormat(String),
    #[error("required data missing: {0}")]
    MissingData(String),
    #[error("required metadata missing: {0}")]
    MissingMetadata(String),
    #[error("geometry error: {0}")]
    Geometry(String),
    #[error("dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Raw2NiiError>;
```

Create `crates/raw2nii-core/src/dataset.rs`:

```rust
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
```

Set `crates/raw2nii-core/src/lib.rs` to:

```rust
//! Vendor-neutral core: the dataset contract, the backend seam, and the
//! NIfTI-MRS writer. This crate never prints and never exits.

pub mod dataset;
pub mod error;

pub use dataset::{identity_affine, DimTag, Metadata, MrsDataset};
pub use error::{Raw2NiiError, Result};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-core dataset`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-core/
git commit -m "feat: core error type and MrsDataset contract"
```

---

### Task 5: FFT — `ifftshift` and orthonormal inverse transform

**Files:**
- Create: `crates/raw2nii-core/src/fft.rs`
- Modify: `crates/raw2nii-core/Cargo.toml`, `crates/raw2nii-core/src/lib.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ifftshift(v: &[Complex<f32>]) -> Vec<Complex<f32>>`
  - `fftshift(v: &[Complex<f32>]) -> Vec<Complex<f32>>`
  - `spec_to_fid(spec: &[Complex<f32>]) -> Vec<Complex<f32>>` — `ifftshift` then inverse DFT with `1/sqrt(N)` normalisation
  - `fid_to_spec(fid: &[Complex<f32>]) -> Vec<Complex<f32>>` — forward transform then `fftshift`, the exact inverse of `spec_to_fid`

**Context:** Ported from `xmris.processing.fid.to_fid`: `ifftshift` → `ifftn(norm="ortho")` → time coordinates. `rustfft` does not normalise, so the `1/sqrt(N)` factor is applied explicitly.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-core/src/fft.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-core fft`
Expected: FAIL — `ifftshift` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/raw2nii-core/Cargo.toml`:

```toml
rustfft = "6"
```

Create `crates/raw2nii-core/src/fft.rs`:

```rust
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
```

Add `pub mod fft;` to `crates/raw2nii-core/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-core fft`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-core/
git commit -m "feat: orthonormal FFT ported from xmris"
```

---

### Task 6: GE header field access

**Files:**
- Create: `crates/raw2nii-ge/src/header/mod.rs`, `crates/raw2nii-ge/src/header/fields.rs`
- Modify: `crates/raw2nii-ge/src/lib.rs`

**Interfaces:**
- Consumes: `MatFile` from Task 2.
- Produces:
  - `GeHeader::from_mat(&MatFile) -> Result<GeHeader, MatError>`
  - Fields: `exam_number: i64`, `series_number: i64`, `specnuc: i64`, `psdname: String`, `dfov: f64`, `slthick: f64`, `user14: f64`, `norm: [f64; 3]`, `tlhc: [f64; 3]`, `ctr: [f64; 3]`, `scan_date: String`, `scan_time: String`
  - `GeHeader::nucleus_name(&self) -> (String, Option<String>)` — the NIfTI-MRS nucleus string plus an optional warning
  - `GeHeader::scan_datetime_iso(&self) -> Option<String>`

**Context:** `scan_date` is `MM/DD/YY` with a 1900 year offset — `125` means 2025. `specnuc` gives the nucleus: 1 → `1H`, 2 → `2H`, 13 → `13C`, 19 → `19F`, 23 → `23NA`, 31 → `31P`. Unrecognised codes produce a warning and a raw-code string, never an error.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/header/fields.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::MatFile;
    use crate::samples::sample_mat;

    macro_rules! header {
        ($name:expr) => {{
            let p = match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            };
            GeHeader::from_mat(&MatFile::open(&p).unwrap()).unwrap()
        }};
    }

    #[test]
    fn reads_identifiers() {
        let h = header!("MRS_2H");
        assert_eq!(h.exam_number, 20000);
        assert_eq!(h.series_number, 6);
        assert_eq!(h.specnuc, 2);
        assert_eq!(h.psdname, "fidall2");
    }

    #[test]
    fn reads_geometry_fields() {
        let h = header!("MRS_2H_slab");
        assert_eq!(h.dfov, 300.0);
        assert_eq!(h.slthick, 80.0);
        assert_eq!(h.user14, 1.0);
        // The slab is tilted.
        assert!((h.norm[1] - (-0.266223)).abs() < 1e-4);
        assert!((h.norm[2] - 0.963911).abs() < 1e-4);
    }

    #[test]
    fn maps_deuterium() {
        let h = header!("MRS_2H");
        let (name, warn) = h.nucleus_name();
        assert_eq!(name, "2H");
        assert!(warn.is_none());
    }

    #[test]
    fn maps_carbon13() {
        let h = header!("MRSI_13C");
        let (name, warn) = h.nucleus_name();
        assert_eq!(name, "13C");
        assert!(warn.is_none());
        assert_eq!(h.exam_number, 4874);
        assert_eq!(h.series_number, 10);
    }

    #[test]
    fn unknown_nucleus_code_warns_but_does_not_fail() {
        let (name, warn) = nucleus_from_code(77);
        assert_eq!(name, "77");
        assert!(warn.is_some(), "an unknown code must produce a warning");
    }

    #[test]
    fn parses_scan_datetime_with_1900_offset() {
        // scan_date "11/25/125" means 2025-11-25.
        assert_eq!(
            parse_scan_datetime("11/25/125", "10:31").unwrap(),
            "2025-11-25T10:31:00"
        );
    }

    #[test]
    fn reads_datetime_from_sample() {
        let h = header!("MRS_2H");
        assert_eq!(h.scan_datetime_iso().unwrap(), "2025-11-25T10:31:00");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge header`
Expected: FAIL — `GeHeader` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/header/mod.rs`:

```rust
pub mod fields;

pub use fields::GeHeader;
```

Create `crates/raw2nii-ge/src/header/fields.rs`:

```rust
//! Typed access to the GE `/h` structs.
//!
//! These structs are identical across GE containers — ScanArchive, P-file and
//! the fidall `.mat` all carry them — so this module is the reuse point for
//! any future GE backend.

use crate::mat::{MatError, MatFile};

/// Map a GE `image/specnuc` code to a NIfTI-MRS nucleus string.
///
/// Returns the name plus an optional warning for unrecognised codes. An
/// unknown code is never fatal: the raw code is used and flagged.
pub fn nucleus_from_code(code: i64) -> (String, Option<String>) {
    let name = match code {
        1 => "1H",
        2 => "2H",
        3 => "3HE",
        7 => "7LI",
        13 => "13C",
        19 => "19F",
        23 => "23NA",
        31 => "31P",
        129 => "129XE",
        _ => {
            return (
                code.to_string(),
                Some(format!(
                    "unrecognised image/specnuc code {code}; \
                     ResonantNucleus written as the raw code"
                )),
            )
        }
    };
    (name.to_string(), None)
}

/// GE stores the scan date as `MM/DD/YY` with a 1900 year offset, so `125`
/// means 2025.
pub fn parse_scan_datetime(date: &str, time: &str) -> Option<String> {
    let mut d = date.split('/');
    let month: u32 = d.next()?.trim().parse().ok()?;
    let day: u32 = d.next()?.trim().parse().ok()?;
    let year_offset: i32 = d.next()?.trim().parse().ok()?;
    let year = 1900 + year_offset;

    let mut t = time.split(':');
    let hour: u32 = t.next()?.trim().parse().ok()?;
    let minute: u32 = t.next()?.trim().parse().ok()?;

    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:00"
    ))
}

#[derive(Debug, Clone)]
pub struct GeHeader {
    pub exam_number: i64,
    pub series_number: i64,
    pub specnuc: i64,
    pub psdname: String,
    pub dfov: f64,
    pub slthick: f64,
    pub user14: f64,
    pub norm: [f64; 3],
    pub tlhc: [f64; 3],
    pub ctr: [f64; 3],
    pub scan_date: String,
    pub scan_time: String,
}

impl GeHeader {
    pub fn from_mat(m: &MatFile) -> Result<Self, MatError> {
        let triple = |a: &str, b: &str, c: &str| -> Result<[f64; 3], MatError> {
            Ok([m.scalar_f64(a)?, m.scalar_f64(b)?, m.scalar_f64(c)?])
        };

        Ok(Self {
            exam_number: m.scalar_f64("/h/exam/ex_no")? as i64,
            series_number: m.scalar_f64("/h/series/se_no")? as i64,
            specnuc: m.scalar_f64("/h/image/specnuc")? as i64,
            psdname: m.string("/h/image/psdname").unwrap_or_default(),
            dfov: m.scalar_f64("/h/image/dfov")?,
            slthick: m.scalar_f64("/h/image/slthick")?,
            user14: m.scalar_f64("/h/rdb_hdr/user14").unwrap_or(f64::NAN),
            norm: triple(
                "/h/image/norm_R",
                "/h/image/norm_A",
                "/h/image/norm_S",
            )?,
            tlhc: triple(
                "/h/image/tlhc_R",
                "/h/image/tlhc_A",
                "/h/image/tlhc_S",
            )?,
            ctr: triple("/h/image/ctr_R", "/h/image/ctr_A", "/h/image/ctr_S")?,
            scan_date: m.string("/h/rdb_hdr/scan_date").unwrap_or_default(),
            scan_time: m.string("/h/rdb_hdr/scan_time").unwrap_or_default(),
        })
    }

    pub fn nucleus_name(&self) -> (String, Option<String>) {
        nucleus_from_code(self.specnuc)
    }

    pub fn scan_datetime_iso(&self) -> Option<String> {
        parse_scan_datetime(&self.scan_date, &self.scan_time)
    }
}
```

Add `pub mod header;` to `crates/raw2nii-ge/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge header`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: typed GE header access with nucleus and datetime mapping"
```

---

### Task 7: GE geometry — affine and localisation rules

**Files:**
- Create: `crates/raw2nii-ge/src/header/geometry.rs`
- Modify: `crates/raw2nii-ge/src/header/mod.rs`

**Interfaces:**
- Consumes: `GeHeader` from Task 6.
- Produces:
  - `Localisation { extents_mm: [f64; 3], warnings: Vec<String> }`
  - `svs_localisation(h: &GeHeader) -> Localisation`
  - `mrsi_localisation(h: &GeHeader, grid: [usize; 3]) -> Localisation`
  - `build_affine(h: &GeHeader, extents_mm: [f64; 3]) -> [[f64; 4]; 4]`
  - `pub const UNLOCALISED_MM: f64 = 10000.0`

**Context (spec §6.2):** SVS `roilenx`/`roileny` are 0 on every sample, so in-plane is always unlocalised. For the excitation dimension, `user14 == 91` under a `fidall*` psd means an unlocalised pulse. MRSI never consults `user14` but must log the value it ignored.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/header/geometry.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn header(psdname: &str, user14: f64, slthick: f64, dfov: f64) -> GeHeader {
        GeHeader {
            exam_number: 1,
            series_number: 1,
            specnuc: 2,
            psdname: psdname.to_string(),
            dfov,
            slthick,
            user14,
            norm: [0.0, 0.0, 1.0],
            tlhc: [100.0, 100.0, 0.0],
            ctr: [0.0, 0.0, 0.0],
            scan_date: String::new(),
            scan_time: String::new(),
        }
    }

    #[test]
    fn svs_in_plane_is_always_unlocalised() {
        let loc = svs_localisation(&header("fidall2", 1.0, 80.0, 300.0));
        assert_eq!(loc.extents_mm[0], UNLOCALISED_MM);
        assert_eq!(loc.extents_mm[1], UNLOCALISED_MM);
    }

    #[test]
    fn svs_slab_uses_slthick() {
        // MRS_2H_slab: user14 == 1, slthick == 80.
        let loc = svs_localisation(&header("fidall2", 1.0, 80.0, 300.0));
        assert_eq!(loc.extents_mm[2], 80.0);
    }

    #[test]
    fn svs_unlocalised_pulse_overrides_slthick() {
        // MRS_2H: user14 == 91, slthick == 40 but the pulse is non-selective.
        let loc = svs_localisation(&header("fidall2", 91.0, 40.0, 300.0));
        assert_eq!(loc.extents_mm[2], UNLOCALISED_MM);
        assert!(
            loc.warnings.iter().any(|w| w.contains("91")),
            "must report the pulse that triggered the rule"
        );
    }

    #[test]
    fn unknown_psd_falls_back_to_slthick_with_a_warning() {
        // MRS_2H_TE_60 uses psd `echocsi`, where user14 means something else.
        let loc = svs_localisation(&header("echocsi", 1.0, 40.0, 300.0));
        assert_eq!(loc.extents_mm[2], 40.0);
        assert!(loc.warnings.iter().any(|w| w.contains("echocsi")));
    }

    #[test]
    fn mrsi_uses_dfov_over_grid_and_ignores_user14() {
        // MRSI_2H: user14 == 91, dfov 200, stored grid 16^3 -> 12.5 mm.
        let loc = mrsi_localisation(&header("fidall2", 91.0, 20.0, 200.0), [16, 16, 16]);
        assert_eq!(loc.extents_mm[0], 12.5);
        assert_eq!(loc.extents_mm[1], 12.5);
        assert_eq!(loc.extents_mm[2], 12.5);
        assert!(
            loc.warnings.iter().any(|w| w.contains("user14")),
            "the ignored user14 value must be logged"
        );
    }

    #[test]
    fn mrsi_2d_uses_slthick_for_the_single_slice() {
        // MRSI_13C: dfov 300, 8x8 grid -> 37.5 mm in plane, slthick 15 in z.
        let loc = mrsi_localisation(&header("fidall2", 1.0, 15.0, 300.0), [8, 8, 1]);
        assert_eq!(loc.extents_mm[0], 37.5);
        assert_eq!(loc.extents_mm[1], 37.5);
        assert_eq!(loc.extents_mm[2], 15.0);
    }

    #[test]
    fn affine_scales_rows_by_extent() {
        let h = header("fidall2", 1.0, 80.0, 300.0);
        let a = build_affine(&h, [10.0, 10.0, 80.0]);
        let col0 = (a[0][0].powi(2) + a[1][0].powi(2) + a[2][0].powi(2)).sqrt();
        let col2 = (a[0][2].powi(2) + a[1][2].powi(2) + a[2][2].powi(2)).sqrt();
        assert!((col0 - 10.0).abs() < 1e-6, "column norm {col0}");
        assert!((col2 - 80.0).abs() < 1e-6, "column norm {col2}");
        assert_eq!(a[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn affine_translation_comes_from_the_centre() {
        let mut h = header("fidall2", 1.0, 80.0, 300.0);
        h.ctr = [1.0, 2.0, 3.0];
        let a = build_affine(&h, [10.0, 10.0, 10.0]);
        // NIfTI is LPS-negated relative to GE's RAS-style centre.
        assert!((a[0][3] - (-1.0)).abs() < 1e-6);
        assert!((a[1][3] - (-2.0)).abs() < 1e-6);
        assert!((a[2][3] - 3.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge geometry`
Expected: FAIL — `svs_localisation` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/header/geometry.rs`:

```rust
//! GE geometry: voxel extents and the affine.
//!
//! Spec §6.2. Two rules carry non-obvious history:
//!
//! * `user14 == 91` under a `fidall*` psd marks a non-selective excitation
//!   pulse, which makes the excitation dimension unlocalised regardless of
//!   `slthick`. `MRS_2H` has `slthick = 40` yet is genuinely unlocalised.
//! * MRSI never consults `user14` — it is assumed to be a legacy SVS field.
//!   That assumption is unverified, so the ignored value is logged.

use super::fields::GeHeader;

/// Spec §2.2: unlocalised dimensions take a 10 m pixdim.
pub const UNLOCALISED_MM: f64 = 10000.0;

/// The `rdb_hdr/user14` value marking a non-selective excitation pulse.
const NONSELECTIVE_PULSE: f64 = 91.0;

#[derive(Debug, Clone)]
pub struct Localisation {
    pub extents_mm: [f64; 3],
    pub warnings: Vec<String>,
}

fn is_fidall(psdname: &str) -> bool {
    psdname.starts_with("fidall")
}

pub fn svs_localisation(h: &GeHeader) -> Localisation {
    let mut warnings = Vec::new();

    // roilenx and roileny are 0 on every sample: SVS is never localised
    // in plane by these sequences.
    let x = UNLOCALISED_MM;
    let y = UNLOCALISED_MM;

    let z = if !is_fidall(&h.psdname) {
        warnings.push(format!(
            "psd '{}' is not a fidall variant; the user14 localisation rule \
             does not apply, falling back to slthick = {}",
            h.psdname, h.slthick
        ));
        h.slthick
    } else if h.user14 == NONSELECTIVE_PULSE {
        warnings.push(format!(
            "rdb_hdr/user14 = {} indicates a non-selective pulse; \
             unlocalised spectroscopy expected, slthick = {} ignored",
            NONSELECTIVE_PULSE, h.slthick
        ));
        UNLOCALISED_MM
    } else {
        h.slthick
    };

    Localisation {
        extents_mm: [x, y, z],
        warnings,
    }
}

pub fn mrsi_localisation(h: &GeHeader, grid: [usize; 3]) -> Localisation {
    let mut warnings = Vec::new();

    if h.user14 == NONSELECTIVE_PULSE {
        warnings.push(format!(
            "rdb_hdr/user14 = {} ignored for MRSI: assumed a legacy SVS field, \
             spatial dimensions come from the encoding and dfov",
            NONSELECTIVE_PULSE
        ));
    }

    let extent = |n: usize, fallback: f64| -> f64 {
        if n > 1 {
            h.dfov / n as f64
        } else {
            fallback
        }
    };

    Localisation {
        extents_mm: [
            extent(grid[0], h.dfov),
            extent(grid[1], h.dfov),
            extent(grid[2], h.slthick),
        ],
        warnings,
    }
}

/// Build a 4x4 affine mapping voxel indices to scanner coordinates.
///
/// GE reports coordinates in an RAS-style frame (R, A, S); NIfTI uses LPS-like
/// qform/sform conventions with x and y negated.
pub fn build_affine(h: &GeHeader, extents_mm: [f64; 3]) -> [[f64; 4]; 4] {
    let normal = normalise(h.norm);
    let (col0, col1) = orthogonal_basis(normal);

    let mut a = [[0.0f64; 4]; 4];
    for (r, axis) in [col0, col1, normal].iter().enumerate() {
        // r indexes the output column, axis is the direction it advances in.
        for c in 0..3 {
            a[c][r] = axis[c] * extents_mm[r];
        }
    }

    // Negate R and A to reach NIfTI's convention.
    for c in 0..3 {
        a[0][c] = -a[0][c];
        a[1][c] = -a[1][c];
    }

    a[0][3] = -h.ctr[0];
    a[1][3] = -h.ctr[1];
    a[2][3] = h.ctr[2];
    a[3] = [0.0, 0.0, 0.0, 1.0];
    a
}

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Two unit vectors completing a right-handed basis with `n`.
fn orthogonal_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick whichever cardinal axis is least aligned with n, so the cross
    // product is well conditioned.
    let seed = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalise(cross(seed, n));
    let v = normalise(cross(n, u));
    (u, v)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
```

Add `pub mod geometry;` to `crates/raw2nii-ge/src/header/mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge geometry`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: GE geometry with unlocalised-pulse and MRSI rules"
```

---

### Task 8: JSON header extension builder

**Files:**
- Create: `crates/raw2nii-core/src/meta/mod.rs`, `crates/raw2nii-core/src/meta/json.rs`
- Modify: `crates/raw2nii-core/src/lib.rs`

**Interfaces:**
- Consumes: `MrsDataset`, `DimTag`, `Metadata` from Task 4.
- Produces: `build_extension(ds: &MrsDataset) -> Vec<u8>` — the complete NIfTI header extension: `esize` (i32 LE), `ecode = 44` (i32 LE), then UTF-8 JSON, zero-padded so the total is a multiple of 16.

**Context (spec §2.3):** `SpectrometerFrequency` and `ResonantNucleus` are required and must be JSON **arrays** even with one element. `dim_5`/`dim_6`/`dim_7` carry the dimension tags.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-core/src/meta/json.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{identity_affine, DimTag, Metadata, MrsDataset};
    use ndarray::{ArrayD, IxDyn};
    use num_complex::Complex;

    fn dataset(shape: &[usize], tags: [Option<DimTag>; 3]) -> MrsDataset {
        MrsDataset {
            data: ArrayD::from_elem(IxDyn(shape), Complex::new(0.0, 0.0)),
            tags,
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

    fn json_of(ext: &[u8]) -> serde_json::Value {
        let text = std::str::from_utf8(&ext[8..]).unwrap();
        serde_json::from_str(text.trim_end_matches('\0')).unwrap()
    }

    #[test]
    fn extension_size_is_a_multiple_of_sixteen() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        assert_eq!(ext.len() % 16, 0, "extension length {}", ext.len());
    }

    #[test]
    fn esize_matches_actual_length_and_ecode_is_44() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let esize = i32::from_le_bytes(ext[0..4].try_into().unwrap());
        let ecode = i32::from_le_bytes(ext[4..8].try_into().unwrap());
        assert_eq!(esize as usize, ext.len());
        assert_eq!(ecode, 44);
    }

    #[test]
    fn required_keys_are_arrays_even_when_single() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let v = json_of(&ext);
        assert!(v["SpectrometerFrequency"].is_array());
        assert!(v["ResonantNucleus"].is_array());
        assert_eq!(v["ResonantNucleus"][0], "2H");
        assert_eq!(v["SpectrometerFrequency"][0], 19.5934);
    }

    #[test]
    fn dimension_tags_are_written() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let v = json_of(&ext);
        assert_eq!(v["dim_5"], "DIM_DYN");
        assert!(v.get("dim_6").is_none());
    }

    #[test]
    fn optional_and_warning_fields_pass_through() {
        let mut ds = dataset(&[16, 16, 16, 700], [None, None, None]);
        ds.meta
            .extra
            .insert("EchoTime".to_string(), serde_json::json!(0.035));
        ds.meta.warnings.push("stale header field".to_string());
        let v = json_of(&build_extension(&ds));
        assert_eq!(v["EchoTime"], 0.035);
        assert_eq!(v["ConversionWarnings"][0], "stale header field");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-core meta`
Expected: FAIL — `build_extension` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-core/src/meta/mod.rs`:

```rust
pub mod json;

pub use json::build_extension;
```

Create `crates/raw2nii-core/src/meta/json.rs`:

```rust
//! The NIfTI-MRS JSON header extension (spec §2.3).
//!
//! Layout: `esize` (i32 LE), `ecode` = 44 (i32 LE), then the UTF-8 JSON,
//! zero-padded so the whole extension is a positive multiple of 16 bytes.

use serde_json::{json, Map, Value};

use crate::dataset::MrsDataset;

/// `NIfTI_ECODE_MRS`.
const ECODE_MRS: i32 = 44;

pub fn build_extension(ds: &MrsDataset) -> Vec<u8> {
    let mut obj: Map<String, Value> = Map::new();

    obj.insert(
        "SpectrometerFrequency".to_string(),
        json!(ds.meta.spectrometer_frequency_mhz),
    );
    obj.insert(
        "ResonantNucleus".to_string(),
        json!(ds.meta.resonant_nucleus),
    );

    for (i, tag) in ds.tags.iter().enumerate() {
        if let Some(t) = tag {
            obj.insert(format!("dim_{}", i + 5), json!(t.as_str()));
        }
    }

    for (k, v) in &ds.meta.extra {
        obj.insert(k.clone(), v.clone());
    }

    if !ds.meta.warnings.is_empty() {
        obj.insert(
            "ConversionWarnings".to_string(),
            json!(ds.meta.warnings),
        );
    }

    let text = serde_json::to_string(&Value::Object(obj)).expect("metadata is serialisable");

    let mut ext = Vec::with_capacity(text.len() + 24);
    ext.extend_from_slice(&0i32.to_le_bytes()); // esize placeholder
    ext.extend_from_slice(&ECODE_MRS.to_le_bytes());
    ext.extend_from_slice(text.as_bytes());
    while ext.len() % 16 != 0 {
        ext.push(0);
    }

    let esize = ext.len() as i32;
    ext[0..4].copy_from_slice(&esize.to_le_bytes());
    ext
}
```

Add `pub mod meta;` to `crates/raw2nii-core/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-core meta`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-core/
git commit -m "feat: NIfTI-MRS JSON header extension builder"
```

---

### Task 9: NIfTI-2 writer

**Files:**
- Create: `crates/raw2nii-core/src/write/mod.rs`, `crates/raw2nii-core/src/write/nifti.rs`
- Modify: `crates/raw2nii-core/Cargo.toml`, `crates/raw2nii-core/src/lib.rs`

**Interfaces:**
- Consumes: `MrsDataset` (Task 4), `build_extension` (Task 8).
- Produces:
  - `serialise(ds: &MrsDataset) -> Result<Vec<u8>>` — a complete uncompressed NIfTI-2 file
  - `write_file(ds: &MrsDataset, path: &Path, gzip_level: u32) -> Result<()>` — gzips when the path ends in `.gz`, writes to a temp file and renames atomically

**Context:** NIfTI-2 header is exactly 540 bytes in the field order below, followed by a 4-byte extension flag, the extension, then the data.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-core/src/write/nifti.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{identity_affine, DimTag, Metadata, MrsDataset};
    use ndarray::{ArrayD, IxDyn};
    use num_complex::Complex;

    fn svs() -> MrsDataset {
        MrsDataset {
            data: ArrayD::from_elem(IxDyn(&[1, 1, 1, 8, 2]), Complex::new(1.5, -2.5)),
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

    fn i32_at(b: &[u8], off: usize) -> i32 {
        i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }
    fn i16_at(b: &[u8], off: usize) -> i16 {
        i16::from_le_bytes(b[off..off + 2].try_into().unwrap())
    }
    fn i64_at(b: &[u8], off: usize) -> i64 {
        i64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }
    fn f64_at(b: &[u8], off: usize) -> f64 {
        f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }

    #[test]
    fn header_is_540_bytes_with_nifti2_magic() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i32_at(&b, 0), 540);
        assert_eq!(&b[4..12], b"n+2\0\r\n\x1a\n");
    }

    #[test]
    fn datatype_is_complex64() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i16_at(&b, 12), 32, "DT_COMPLEX");
        assert_eq!(i16_at(&b, 14), 64, "bitpix");
    }

    #[test]
    fn dimensions_are_written_in_nifti_order() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i64_at(&b, 16), 5, "dim[0]");
        assert_eq!(i64_at(&b, 24), 1);
        assert_eq!(i64_at(&b, 32), 1);
        assert_eq!(i64_at(&b, 40), 1);
        assert_eq!(i64_at(&b, 48), 8);
        assert_eq!(i64_at(&b, 56), 2);
    }

    #[test]
    fn dwell_time_is_in_pixdim_four() {
        let b = serialise(&svs()).unwrap();
        // pixdim starts at 104; pixdim[4] is the fifth element.
        assert!((f64_at(&b, 104 + 4 * 8) - 2e-4).abs() < 1e-12);
    }

    #[test]
    fn intent_name_declares_the_standard_version() {
        let b = serialise(&svs()).unwrap();
        let name = &b[508..524];
        let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
        assert_eq!(std::str::from_utf8(&name[..end]).unwrap(), "mrs_v0_11");
    }

    #[test]
    fn xyzt_units_are_mm_and_seconds() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i32_at(&b, 500), 10, "NIFTI_UNITS_MM | NIFTI_UNITS_SEC");
    }

    #[test]
    fn qform_and_sform_are_populated() {
        let b = serialise(&svs()).unwrap();
        assert_ne!(i32_at(&b, 344), 0, "qform_code");
        assert_ne!(i32_at(&b, 348), 0, "sform_code");
    }

    #[test]
    fn extension_flag_is_set_and_vox_offset_points_past_it() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(b[540], 1, "extension flag");
        let vox_offset = i64_at(&b, 168);
        let esize = i32_at(&b, 544) as i64;
        assert_eq!(vox_offset, 544 + esize);
    }

    #[test]
    fn data_is_written_as_interleaved_f32_pairs() {
        let b = serialise(&svs()).unwrap();
        let vox_offset = i64_at(&b, 168) as usize;
        let n = 1 * 1 * 1 * 8 * 2;
        assert_eq!(b.len(), vox_offset + n * 8);
        let re = f32::from_le_bytes(b[vox_offset..vox_offset + 4].try_into().unwrap());
        let im = f32::from_le_bytes(b[vox_offset + 4..vox_offset + 8].try_into().unwrap());
        assert_eq!(re, 1.5);
        assert_eq!(im, -2.5);
    }

    #[test]
    fn rejects_an_invalid_dataset() {
        let mut ds = svs();
        ds.meta.resonant_nucleus.clear();
        assert!(serialise(&ds).is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-core write`
Expected: FAIL — `serialise` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/raw2nii-core/Cargo.toml`:

```toml
flate2 = { version = "1", features = ["zlib-ng"], default-features = false }
```

Create `crates/raw2nii-core/src/write/mod.rs`:

```rust
pub mod nifti;

pub use nifti::{serialise, write_file};
```

Create `crates/raw2nii-core/src/write/nifti.rs`:

```rust
//! NIfTI-2 serialisation.
//!
//! The header is exactly 540 bytes, little-endian, followed by a 4-byte
//! extension flag, the MRS header extension, then the complex data.

use std::io::Write;
use std::path::Path;

use crate::dataset::MrsDataset;
use crate::error::Result;
use crate::meta::build_extension;

const HEADER_SIZE: usize = 540;
const DT_COMPLEX: i16 = 32;
const BITPIX: i16 = 64;
/// NIFTI_UNITS_MM (2) | NIFTI_UNITS_SEC (8).
const XYZT_UNITS: i32 = 10;
const INTENT_NAME: &[u8] = b"mrs_v0_11";

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(HEADER_SIZE),
        }
    }
    fn i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    /// Fixed-width, zero-padded, always NUL-terminated by the padding.
    fn fixed(&mut self, v: &[u8], width: usize) {
        let n = v.len().min(width);
        self.buf.extend_from_slice(&v[..n]);
        self.buf.extend(std::iter::repeat(0u8).take(width - n));
    }
}

pub fn serialise(ds: &MrsDataset) -> Result<Vec<u8>> {
    ds.validate()?;

    let ext = build_extension(ds);
    let vox_offset = (HEADER_SIZE + 4 + ext.len()) as i64;

    let shape = ds.data.shape();
    let mut dim = [1i64; 8];
    dim[0] = shape.len() as i64;
    for (i, &s) in shape.iter().enumerate() {
        dim[i + 1] = s as i64;
    }

    // pixdim[1..3] are spatial extents taken from the affine column norms;
    // pixdim[4] is the dwell time.
    let mut pixdim = [1.0f64; 8];
    pixdim[0] = 1.0;
    for c in 0..3 {
        pixdim[c + 1] = (ds.affine[0][c].powi(2)
            + ds.affine[1][c].powi(2)
            + ds.affine[2][c].powi(2))
        .sqrt();
    }
    pixdim[4] = ds.dwell_time_s;

    let mut w = Writer::new();
    w.i32(HEADER_SIZE as i32); // 0   sizeof_hdr
    w.bytes(b"n+2\0\r\n\x1a\n"); // 4   magic
    w.i16(DT_COMPLEX); // 12  datatype
    w.i16(BITPIX); // 14  bitpix
    for d in dim {
        w.i64(d);
    } // 16  dim[8]
    w.f64(0.0); // 80  intent_p1
    w.f64(0.0); // 88  intent_p2
    w.f64(0.0); // 96  intent_p3
    for p in pixdim {
        w.f64(p);
    } // 104 pixdim[8]
    w.i64(vox_offset); // 168 vox_offset
    w.f64(1.0); // 176 scl_slope
    w.f64(0.0); // 184 scl_inter
    w.f64(0.0); // 192 cal_max
    w.f64(0.0); // 200 cal_min
    w.f64(0.0); // 208 slice_duration
    w.f64(0.0); // 216 toffset
    w.i64(0); // 224 slice_start
    w.i64(0); // 232 slice_end
    w.fixed(b"raw2nii NIfTI-MRS", 80); // 240 descrip
    w.fixed(b"", 24); // 320 aux_file
    w.i32(1); // 344 qform_code (scanner anat)
    w.i32(1); // 348 sform_code
    w.f64(0.0); // 352 quatern_b
    w.f64(0.0); // 360 quatern_c
    w.f64(0.0); // 368 quatern_d
    w.f64(ds.affine[0][3]); // 376 qoffset_x
    w.f64(ds.affine[1][3]); // 384 qoffset_y
    w.f64(ds.affine[2][3]); // 392 qoffset_z
    for r in 0..3 {
        for c in 0..4 {
            w.f64(ds.affine[r][c]);
        }
    } // 400 srow_x/y/z
    w.i32(0); // 496 slice_code
    w.i32(XYZT_UNITS); // 500 xyzt_units
    w.i32(0); // 504 intent_code
    w.fixed(INTENT_NAME, 16); // 508 intent_name
    w.u8(0); // 524 dim_info
    w.fixed(b"", 15); // 525 unused_str

    debug_assert_eq!(w.buf.len(), HEADER_SIZE);

    let mut out = w.buf;
    out.extend_from_slice(&[1u8, 0, 0, 0]); // extension flag
    out.extend_from_slice(&ext);

    // NIfTI expects the first dimension to vary fastest. ndarray's standard
    // layout has the last axis varying fastest, so reverse the axes before
    // flattening.
    let flat = ds.data.view().reversed_axes();
    out.reserve(ds.data.len() * 8);
    for c in flat.iter() {
        out.extend_from_slice(&c.re.to_le_bytes());
        out.extend_from_slice(&c.im.to_le_bytes());
    }

    Ok(out)
}

/// Write to `path`, gzipping when it ends in `.gz`. Writes to a sibling
/// temporary file and renames, so an aborted run never leaves a partial file.
pub fn write_file(ds: &MrsDataset, path: &Path, gzip_level: u32) -> Result<()> {
    let bytes = serialise(ds)?;
    let tmp = path.with_extension("partial");

    {
        let file = std::fs::File::create(&tmp)?;
        let mut sink: Box<dyn Write> = if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("gz"))
        {
            Box::new(flate2::write::GzEncoder::new(
                file,
                flate2::Compression::new(gzip_level),
            ))
        } else {
            Box::new(file)
        };
        sink.write_all(&bytes)?;
        sink.flush()?;
    }

    std::fs::rename(&tmp, path)?;
    Ok(())
}
```

Add `pub mod write;` to `crates/raw2nii-core/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-core write`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-core/
git commit -m "feat: NIfTI-2 writer with MRS extension and atomic output"
```

---

### Task 10: Flavor detection

**Files:**
- Create: `crates/raw2nii-ge/src/flavor.rs`
- Modify: `crates/raw2nii-ge/src/lib.rs`

**Interfaces:**
- Consumes: `MatFile` from Task 2.
- Produces: `Flavor` enum (`Svs`, `Mrsi`) and `detect(m: &MatFile) -> Option<Flavor>`.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/flavor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::MatFile;
    use crate::samples::sample_mat;

    macro_rules! detect_sample {
        ($name:expr) => {{
            let p = match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            };
            detect(&MatFile::open(&p).unwrap())
        }};
    }

    #[test]
    fn detects_svs() {
        assert_eq!(detect_sample!("MRS_2H"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_slab"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_TE_60"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_TI_400"), Some(Flavor::Svs));
    }

    #[test]
    fn detects_mrsi() {
        assert_eq!(detect_sample!("MRSI_13C"), Some(Flavor::Mrsi));
        assert_eq!(detect_sample!("MRSI_2H"), Some(Flavor::Mrsi));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge flavor`
Expected: FAIL — `detect` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/flavor.rs`:

```rust
//! SVS versus MRSI detection.
//!
//! This is a fidall-`.mat` distinction, not a universal one, so it stays
//! private to this backend.

use crate::mat::MatFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Svs,
    Mrsi,
}

pub fn detect(m: &MatFile) -> Option<Flavor> {
    if m.has("/fid") && m.has("/par/samples") {
        return Some(Flavor::Svs);
    }
    if m.has("/spec") && m.has("/nn") && m.has("/dim") {
        return Some(Flavor::Mrsi);
    }
    None
}
```

Add `pub mod flavor;` to `crates/raw2nii-ge/src/lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge flavor`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: SVS/MRSI flavor detection"
```

---

### Task 11: SVS reader

**Files:**
- Create: `crates/raw2nii-ge/src/read/mod.rs`, `crates/raw2nii-ge/src/read/svs.rs`
- Modify: `crates/raw2nii-ge/src/lib.rs`

**Interfaces:**
- Consumes: `MatFile` (Tasks 2–3), `GeHeader` (Task 6), `svs_localisation`/`build_affine` (Task 7), `MrsDataset`/`DimTag`/`Metadata` (Task 4).
- Produces: `read_svs(m: &MatFile, h: &GeHeader) -> Result<MrsDataset>`.

**Context:** `/fid` is MATLAB `(rows, samples)` and already time-domain. Output is `(1, 1, 1, samples, rows)` with `dim_5 = DIM_DYN`. **No chop, no conjugation** — spec §4.1.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/read/svs.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge read::svs`
Expected: FAIL — `read_svs` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/read/mod.rs`:

```rust
pub mod svs;

pub use svs::read_svs;
```

Create `crates/raw2nii-ge/src/read/svs.rs`:

```rust
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
    let affine = build_affine(h, loc.extents_mm);

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
```

Add `pub mod read;` to `crates/raw2nii-ge/src/lib.rs`, and add `serde_json.workspace = true` plus `num-complex.workspace = true` to `crates/raw2nii-ge/Cargo.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge read::svs`
Expected: PASS — 7 tests. `conjugation_is_not_applied` reproduces the spec §4.1 measurement (ratio ≈ 5.0).

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: SVS reader, no chop and no conjugation per spec 4.1"
```

---

### Task 12: MRSI reader

**Files:**
- Create: `crates/raw2nii-ge/src/read/mrsi.rs`
- Modify: `crates/raw2nii-ge/src/read/mod.rs`

**Interfaces:**
- Consumes: `MatFile`, `GeHeader`, `mrsi_localisation`/`build_affine`, `spec_to_fid` (Task 5).
- Produces: `read_mrsi(m: &MatFile, h: &GeHeader) -> Result<MrsDataset>`.

**Context:** `/spec` is MATLAB `(nspec, nx, ny, nz)` and frequency-domain. NIfTI-MRS requires time domain, so each voxel's spectrum goes through `spec_to_fid`. Output is `(nx, ny, nz, nspec)`, `dim[0] = 4`, no dimension tags. The grid is the stored (zero-filled) size, never `nn`.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/read/mrsi.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge read::mrsi`
Expected: FAIL — `read_mrsi` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-ge/src/read/mrsi.rs`:

```rust
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
    let affine = build_affine(h, loc.extents_mm);

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
```

Add `pub mod mrsi;` and `pub use mrsi::read_mrsi;` to `crates/raw2nii-ge/src/read/mod.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge read::mrsi`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/
git commit -m "feat: MRSI reader with spectrum-to-FID transform"
```

---

### Task 13: Backend trait, registry and `GeMatBackend`

**Files:**
- Create: `crates/raw2nii-core/src/backend.rs`
- Modify: `crates/raw2nii-core/src/lib.rs`, `crates/raw2nii-ge/src/lib.rs`

**Interfaces:**
- Consumes: `MrsDataset`, `Result`, `read_svs`, `read_mrsi`, `detect`.
- Produces:
  - `Confidence { No, Maybe, Yes }`
  - `trait Backend { fn name(&self) -> &'static str; fn probe(&self, path: &Path) -> Confidence; fn convert(&self, path: &Path) -> Result<Vec<MrsDataset>>; }`
  - `Registry::with_backend(Box<dyn Backend>) -> Registry`, `Registry::select(&self, path: &Path) -> Option<&dyn Backend>`, `Registry::by_name(&self, name: &str) -> Option<&dyn Backend>`
  - `raw2nii_ge::GeMatBackend` implementing `Backend`, `name() == "ge-fidall-mat"`
  - `raw2nii_ge::output_stem(h: &GeHeader, flavor: Flavor, unlocalised: bool, n_slices: usize) -> String` — `unlocalised` only affects SVS, `n_slices` only affects MRSI (1 slice ⇒ `mrsi2d`, more ⇒ `mrsi3d`)
  - `raw2nii_ge::is_unlocalised(ds: &MrsDataset) -> bool`

**Context:** Naming is `exam{ex_no}_series{se_no:02}_{nucleus}_{type}` (spec §7.1), e.g. `exam20000_series06_2H_svs-unloc`.

- [ ] **Step 1: Write the failing test**

Append to `crates/raw2nii-ge/src/lib.rs`:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge backend_tests`
Expected: FAIL — `GeMatBackend` is not defined.

- [ ] **Step 3: Write minimal implementation**

Create `crates/raw2nii-core/src/backend.rs`:

```rust
//! The extension seam.
//!
//! Abstraction lives at the outer boundary only: a backend takes a path and
//! produces `MrsDataset` values. There is deliberately no shared intermediate
//! representation between vendors.

use std::path::Path;

use crate::dataset::MrsDataset;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    No,
    Maybe,
    Yes,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Cheap: extension, magic bytes, a few key probes. Never a full parse.
    fn probe(&self, path: &Path) -> Confidence;
    fn convert(&self, path: &Path) -> Result<Vec<MrsDataset>>;
}

#[derive(Default)]
pub struct Registry {
    backends: Vec<Box<dyn Backend>>,
}

impl Registry {
    pub fn with_backend(mut self, b: Box<dyn Backend>) -> Self {
        self.backends.push(b);
        self
    }

    pub fn select(&self, path: &Path) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .map(|b| (b.probe(path), b))
            .filter(|(c, _)| *c > Confidence::No)
            .max_by_key(|(c, _)| *c)
            .map(|(_, b)| b.as_ref())
    }

    pub fn by_name(&self, name: &str) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
}
```

Add `pub mod backend;` to `crates/raw2nii-core/src/lib.rs`.

Replace the body of `crates/raw2nii-ge/src/lib.rs` with:

```rust
//! GE backends. Currently the fidall `.mat` v7.3 container.

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
        "exam{}_series{:02}_{}_{}",
        h.exam_number, h.series_number, nucleus, kind
    )
}

/// Whether an SVS dataset came out unlocalised, for naming purposes.
pub fn is_unlocalised(ds: &MrsDataset) -> bool {
    let z = (ds.affine[0][2].powi(2) + ds.affine[1][2].powi(2) + ds.affine[2][2].powi(2)).sqrt();
    (z - UNLOCALISED_MM).abs() < 1e-6
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-ge backend_tests`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/
git commit -m "feat: backend trait, registry and GE fidall .mat backend"
```

---

### Task 14: Minimal `convert` CLI

**Files:**
- Modify: `crates/raw2nii-cli/Cargo.toml`, `crates/raw2nii-cli/src/main.rs`

**Interfaces:**
- Consumes: `Registry`, `GeMatBackend`, `output_stem`, `is_unlocalised`, `write_file`.
- Produces: the `raw2nii` binary with `raw2nii convert <PATH> [-o DIR] [--overwrite] [--compress-level N] [-v]`.

**Context:** Directory recursion, archiving, deletion and parallelism are Plan 2. This task delivers single-file and simple-directory conversion so the pipeline is usable end to end.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-cli/tests/cli.rs`:

```rust
use std::path::PathBuf;
use std::process::Command;

fn sample(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/datasets")
        .join(name);
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_raw2nii"))
}

#[test]
fn converts_a_single_file_to_the_named_output() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let out = std::env::temp_dir().join("raw2nii-cli-test-svs");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    let status = Command::new(bin())
        .args(["convert"])
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());

    let expected = out.join("exam20000_series06_2H_svs-unloc.nii.gz");
    assert!(expected.exists(), "missing {expected:?}");
    assert!(std::fs::metadata(&expected).unwrap().len() > 1000);
}

#[test]
fn refuses_to_overwrite_without_the_flag() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let out = std::env::temp_dir().join("raw2nii-cli-test-overwrite");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    for _ in 0..2 {
        let status = Command::new(bin())
            .args(["convert"])
            .arg(&input)
            .arg("-o")
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "a skip is not a failure");
    }
}

#[test]
fn unreadable_input_exits_non_zero() {
    let bogus = std::env::temp_dir().join("raw2nii-not-a-real-file.mat");
    let _ = std::fs::remove_file(&bogus);
    let status = Command::new(bin())
        .args(["convert"])
        .arg(&bogus)
        .status()
        .unwrap();
    assert!(!status.success());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli`
Expected: FAIL — the binary has no `convert` subcommand, so the first test's status is non-zero.

- [ ] **Step 3: Write minimal implementation**

Set `crates/raw2nii-cli/Cargo.toml`:

```toml
[package]
name = "raw2nii-cli"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[[bin]]
name = "raw2nii"
path = "src/main.rs"

[dependencies]
raw2nii-core = { path = "../raw2nii-core" }
raw2nii-ge = { path = "../raw2nii-ge" }
clap = { version = "4", features = ["derive"] }
tracing.workspace = true
tracing-subscriber = "0.3"
```

Set `crates/raw2nii-cli/src/main.rs`:

```rust
//! The `raw2nii` binary. This is the only crate that prints or exits.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use raw2nii_core::backend::Registry;
use raw2nii_core::write::write_file;
use raw2nii_ge::{is_unlocalised, output_stem, GeMatBackend};

#[derive(Parser)]
#[command(name = "raw2nii", version, about = "Convert GE MRS data to NIfTI-MRS")]
struct Cli {
    /// Increase verbosity: -v for info, -vv for debug.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert one file, or every convertible file in a directory.
    Convert {
        /// A file or a directory.
        path: PathBuf,
        /// Output directory. Defaults to alongside each input.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Replace existing output instead of skipping it.
        #[arg(long)]
        overwrite: bool,
        /// gzip level, 0 to 9.
        #[arg(long, default_value_t = 4)]
        compress_level: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    let level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        _ => tracing::Level::DEBUG,
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    let Command::Convert {
        path,
        output,
        overwrite,
        compress_level,
    } = cli.command;

    let inputs = match discover(&path) {
        Ok(v) if v.is_empty() => {
            eprintln!("error: no convertible files found in {}", path.display());
            std::process::exit(1);
        }
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let registry = Registry::default().with_backend(Box::new(GeMatBackend));
    let mut failures = 0usize;

    for input in inputs {
        match convert_one(
            &registry,
            &input,
            output.as_deref(),
            overwrite,
            compress_level,
        ) {
            Ok(Some(p)) => println!("{} -> {}", input.display(), p.display()),
            Ok(None) => tracing::warn!("{}: output exists, skipping", input.display()),
            Err(e) => {
                eprintln!("error: {}: {e}", input.display());
                failures += 1;
            }
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
}

fn discover(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("mat") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn convert_one(
    registry: &Registry,
    input: &Path,
    output_dir: Option<&Path>,
    overwrite: bool,
    compress_level: u32,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let backend = registry
        .select(input)
        .ok_or_else(|| format!("no backend can read {}", input.display()))?;

    let datasets = backend.convert(input)?;
    let dir = output_dir
        .map(Path::to_path_buf)
        .or_else(|| input.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;

    let mut written = None;
    for ds in &datasets {
        let stem = stem_for(input, ds)?;
        let out = dir.join(format!("{stem}.nii.gz"));
        if out.exists() && !overwrite {
            return Ok(None);
        }
        for w in &ds.meta.warnings {
            tracing::warn!("{}: {w}", input.display());
        }
        write_file(ds, &out, compress_level)?;
        written = Some(out);
    }
    Ok(written)
}

/// Re-derive the naming inputs from the file, then apply the §7.1 scheme.
fn stem_for(
    input: &Path,
    ds: &raw2nii_core::dataset::MrsDataset,
) -> Result<String, Box<dyn std::error::Error>> {
    use raw2nii_ge::flavor::{detect, Flavor};
    use raw2nii_ge::header::GeHeader;
    use raw2nii_ge::mat::MatFile;

    let m = MatFile::open(input)?;
    let h = GeHeader::from_mat(&m)?;
    let flavor = detect(&m).ok_or("unrecognised fidall .mat layout")?;

    // n_slices is only consulted for MRSI; SVS ignores it.
    let n_slices = match flavor {
        Flavor::Svs => 1,
        Flavor::Mrsi => ds.data.shape()[2],
    };
    Ok(output_stem(&h, flavor, is_unlocalised(ds), n_slices))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "feat: raw2nii convert CLI"
```

---

### Task 15: Conformance and golden-file tests

**Files:**
- Create: `crates/raw2nii-ge/tests/conformance.rs`, `crates/raw2nii-ge/tests/golden.rs`
- Create: `tests/goldens/.gitkeep`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `GeMatBackend`, `serialise`.
- Produces: nothing consumed by later tasks.

**Context (spec §9):** Conformance asserts the standard's hard requirements against every sample. Goldens catch unintended changes; `RAW2NII_BLESS=1` regenerates them. Goldens are small enough to commit even though the source datasets are not.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-ge/tests/conformance.rs`:

```rust
//! Spec conformance, asserted against every shipped dataset.

use std::path::PathBuf;

use raw2nii_core::backend::Backend;
use raw2nii_core::write::serialise;
use raw2nii_ge::GeMatBackend;

const DATASETS: &[&str] = &[
    "MRS_2H",
    "MRS_2H_slab",
    "MRS_2H_TE_60",
    "MRS_2H_TI_400",
    "MRSI_13C",
    "MRSI_2H",
];

fn sample(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/datasets")
        .join(name);
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
}

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn i16_at(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}
fn i64_at(b: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn f64_at(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

#[test]
fn every_dataset_produces_a_conformant_file() {
    let mut checked = 0;
    for name in DATASETS {
        let Some(path) = sample(name) else { continue };
        let datasets = GeMatBackend
            .convert(&path)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        for ds in &datasets {
            let b = serialise(ds).unwrap_or_else(|e| panic!("{name}: {e}"));

            assert_eq!(i32_at(&b, 0), 540, "{name}: sizeof_hdr");
            assert_eq!(&b[4..12], b"n+2\0\r\n\x1a\n", "{name}: magic");

            let datatype = i16_at(&b, 12);
            assert!(
                matches!(datatype, 32 | 1792 | 2048),
                "{name}: datatype {datatype} is not a complex type"
            );

            let ndim = i64_at(&b, 16);
            assert!((4..=7).contains(&ndim), "{name}: dim[0] = {ndim}");

            let dwell = f64_at(&b, 104 + 4 * 8);
            assert!(dwell > 0.0, "{name}: pixdim[4] = {dwell}");

            assert_eq!(i32_at(&b, 500), 10, "{name}: xyzt_units");
            assert_ne!(i32_at(&b, 344), 0, "{name}: qform_code");
            assert_ne!(i32_at(&b, 348), 0, "{name}: sform_code");

            let intent = &b[508..524];
            let end = intent.iter().position(|&c| c == 0).unwrap_or(intent.len());
            assert_eq!(
                std::str::from_utf8(&intent[..end]).unwrap(),
                "mrs_v0_11",
                "{name}: intent_name"
            );

            assert_eq!(b[540], 1, "{name}: extension flag");
            let esize = i32_at(&b, 544);
            assert!(esize > 0 && esize % 16 == 0, "{name}: esize {esize}");
            assert_eq!(i32_at(&b, 548), 44, "{name}: ecode");

            let json_bytes = &b[552..(544 + esize as usize)];
            let text = std::str::from_utf8(json_bytes)
                .unwrap()
                .trim_end_matches('\0');
            let v: serde_json::Value = serde_json::from_str(text)
                .unwrap_or_else(|e| panic!("{name}: extension is not valid JSON: {e}"));
            assert!(
                v["SpectrometerFrequency"].is_array(),
                "{name}: SpectrometerFrequency must be an array"
            );
            assert!(
                v["ResonantNucleus"].is_array(),
                "{name}: ResonantNucleus must be an array"
            );

            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("SKIP: tests/datasets absent");
    }
}

#[test]
fn the_prescan_directory_is_declined() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/datasets/BS_prescan_13C");
    if !dir.exists() {
        eprintln!("SKIP: tests/datasets absent");
        return;
    }
    let mats: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mat"))
        .collect();
    assert!(
        mats.is_empty(),
        "BS_prescan_13C is the negative case and must ship no .mat"
    );
}
```

Create `crates/raw2nii-ge/tests/golden.rs`:

```rust
//! Golden-file tests. Set `RAW2NII_BLESS=1` to regenerate.
//!
//! Goldens are the serialised, uncompressed NIfTI bytes hashed to keep the
//! committed artefacts small while still catching any byte-level change.

use std::path::PathBuf;

use raw2nii_core::backend::Backend;
use raw2nii_core::write::serialise;
use raw2nii_ge::GeMatBackend;

const DATASETS: &[&str] = &[
    "MRS_2H",
    "MRS_2H_slab",
    "MRS_2H_TE_60",
    "MRS_2H_TI_400",
    "MRSI_13C",
    "MRSI_2H",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sample(name: &str) -> Option<PathBuf> {
    let dir = repo_root().join("tests/datasets").join(name);
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
}

/// FNV-1a, so the test has no hashing dependency.
fn digest(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[test]
fn output_matches_the_goldens() {
    let bless = std::env::var("RAW2NII_BLESS").is_ok();
    let dir = repo_root().join("tests/goldens");
    std::fs::create_dir_all(&dir).unwrap();

    let mut checked = 0;
    for name in DATASETS {
        let Some(path) = sample(name) else { continue };
        let datasets = GeMatBackend.convert(&path).unwrap();
        let bytes = serialise(&datasets[0]).unwrap();
        let actual = format!("{}  {} bytes", digest(&bytes), bytes.len());

        let golden = dir.join(format!("{name}.txt"));
        if bless {
            std::fs::write(&golden, &actual).unwrap();
            continue;
        }
        let expected = match std::fs::read_to_string(&golden) {
            Ok(s) => s,
            Err(_) => panic!(
                "no golden for {name}. Run: RAW2NII_BLESS=1 cargo test -p raw2nii-ge --test golden"
            ),
        };
        assert_eq!(actual, expected.trim(), "{name}: output changed");
        checked += 1;
    }
    if checked == 0 && !bless {
        eprintln!("SKIP: tests/datasets absent");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-ge --test golden`
Expected: FAIL with "no golden for MRS_2H" when the datasets are present. (With the datasets absent it prints SKIP and passes — that is correct behaviour.)

- [ ] **Step 3: Write minimal implementation**

Goldens are generated, not hand-written:

```bash
RAW2NII_BLESS=1 cargo test -p raw2nii-ge --test golden
```

Then ensure they are tracked. Add to `.gitignore`, immediately after the `/tests/datasets/` entry:

```
# Goldens are small text digests and ARE tracked, unlike the datasets above.
!/tests/goldens/
```

Create `tests/goldens/.gitkeep` as an empty file so the directory exists on a fresh clone.

Inspect the generated files before committing — each should contain one line of the form `<16 hex chars>  <N> bytes`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --workspace`
Expected: PASS — every test across all three crates. With the datasets absent, the data-dependent tests print SKIP and pass.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-ge/tests/ tests/goldens/ .gitignore
git commit -m "test: NIfTI-MRS conformance and golden-file regression tests"
```

---

## Definition of Done

- [ ] `cargo test --workspace` passes with the sample datasets present.
- [ ] `cargo test --workspace` passes with `tests/datasets/` absent, printing SKIP for data-dependent tests.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- [ ] `raw2nii convert tests/datasets/MRS_2H -o /tmp/out` writes `exam20000_series06_2H_svs-unloc.nii.gz`.
- [ ] All six `.mat` datasets convert and pass conformance.
- [ ] `raw2nii-core` contains no `println!`, no `eprintln!` outside `#[cfg(test)]`, and no `process::exit`.

## Follow-on Plans

- **Plan 2 — CLI operations:** recursive discovery, rayon parallelism, `--archive` (tar.zst of the parent directory, output excluded), `--delete` (gated on archive verification), `--dry-run`, `--format`, `--json-log`.
- **Plan 3 — Python bindings:** `raw2nii-py` via pyo3/maturin, zero-copy `ds.data` as a numpy array, `py.allow_threads`, the exception hierarchy, `abi3-py39` wheels with vendored libhdf5.
- **Performance (spec §11):** the criterion benchmark over the sample set, with a regression guard. Deferred deliberately — the spec commits to measure-then-assert, and there is nothing meaningful to measure until Plan 2 adds parallel discovery and configurable compression.
