//! Converting one discovered input into zero or one written file.

use std::path::{Path, PathBuf};

use raw2nii_core::backend::Registry;
use raw2nii_core::dataset::MrsDataset;
use raw2nii_core::write::write_file;
use raw2nii_ge::{is_unlocalised, output_stem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    #[value(name = "nii-gz")]
    NiiGz,
    #[value(name = "nii")]
    Nii,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::NiiGz => "nii.gz",
            OutputFormat::Nii => "nii",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::NiiGz => f.write_str("nii-gz"),
            OutputFormat::Nii => f.write_str("nii"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Written { input: PathBuf, output: PathBuf },
    Skipped { input: PathBuf, reason: String },
    Failed { input: PathBuf, error: String },
}

impl Outcome {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn input(&self) -> &Path {
        match self {
            Outcome::Written { input, .. }
            | Outcome::Skipped { input, .. }
            | Outcome::Failed { input, .. } => input,
        }
    }

    pub fn is_failure(&self) -> bool {
        matches!(self, Outcome::Failed { .. })
    }
}

/// Re-derive the naming inputs from the file, then apply the spec §7.1 scheme.
fn stem_for(input: &Path, ds: &MrsDataset) -> Result<String, String> {
    use raw2nii_ge::flavor::{detect, Flavor};
    use raw2nii_ge::header::GeHeader;
    use raw2nii_ge::mat::MatFile;

    let m = MatFile::open(input).map_err(|e| e.to_string())?;
    let h = GeHeader::from_mat(&m).map_err(|e| e.to_string())?;
    let flavor = detect(&m).ok_or("unrecognised fidall .mat layout")?;

    let n_slices = match flavor {
        Flavor::Svs => 1,
        Flavor::Mrsi => ds.data.shape()[2],
    };
    Ok(output_stem(&h, flavor, is_unlocalised(ds), n_slices))
}

/// Convert one input. `write` gates the actual file write, so a dry run can
/// reuse the exact same conversion and naming logic and still write nothing.
#[allow(clippy::too_many_arguments)]
pub fn convert_one(
    registry: &Registry,
    input: &Path,
    output_dir: Option<&Path>,
    overwrite: bool,
    compress_level: u32,
    format: OutputFormat,
    write: bool,
) -> Outcome {
    let input = input.to_path_buf();

    let backend = match registry.select(&input) {
        Some(b) => b,
        None => {
            return Outcome::Failed {
                error: format!("no backend can read {}", input.display()),
                input,
            }
        }
    };

    let datasets = match backend.convert(&input) {
        Ok(d) => d,
        Err(e) => {
            return Outcome::Failed {
                input,
                error: e.to_string(),
            }
        }
    };

    let dir = output_dir
        .map(Path::to_path_buf)
        .or_else(|| input.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    let mut last = None;
    for ds in &datasets {
        let stem = match stem_for(&input, ds) {
            Ok(s) => s,
            Err(e) => return Outcome::Failed { input, error: e },
        };
        let out = dir.join(format!("{stem}.{}", format.extension()));

        if out.exists() && !overwrite {
            return Outcome::Skipped {
                input,
                reason: format!("{} exists", out.display()),
            };
        }
        for w in &ds.meta.warnings {
            tracing::warn!("{}: {w}", input.display());
        }
        if write {
            if let Err(e) = std::fs::create_dir_all(&dir)
                .and_then(|_| write_file(ds, &out, compress_level).map_err(std::io::Error::other))
            {
                return Outcome::Failed {
                    input,
                    error: e.to_string(),
                };
            }
        }
        last = Some(out);
    }

    match last {
        Some(output) => Outcome::Written { input, output },
        None => Outcome::Skipped {
            input,
            reason: "backend produced no datasets".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raw2nii_ge::GeMatBackend;

    macro_rules! sample {
        ($name:expr) => {
            match raw2nii_ge::samples::sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            }
        };
    }

    #[test]
    fn writes_and_reports_the_output_path() {
        let input = sample!("MRS_2H");
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcome = convert_one(
            &registry,
            &input,
            Some(dir.path()),
            false,
            4,
            OutputFormat::NiiGz,
            true,
        );
        match outcome {
            Outcome::Written { output, .. } => {
                assert!(output.exists());
                assert!(output.to_string_lossy().ends_with(".nii.gz"));
            }
            other => panic!("expected Written, got {other:?}"),
        }
    }

    #[test]
    fn nii_format_writes_uncompressed_extension() {
        let input = sample!("MRS_2H");
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcome = convert_one(
            &registry,
            &input,
            Some(dir.path()),
            false,
            4,
            OutputFormat::Nii,
            true,
        );
        match outcome {
            Outcome::Written { output, .. } => {
                assert!(output.to_string_lossy().ends_with(".nii"));
                assert!(!output.to_string_lossy().ends_with(".nii.gz"));
            }
            other => panic!("expected Written, got {other:?}"),
        }
    }

    #[test]
    fn dry_run_reports_without_writing() {
        let input = sample!("MRS_2H");
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcome = convert_one(
            &registry,
            &input,
            Some(dir.path()),
            false,
            4,
            OutputFormat::NiiGz,
            false,
        );
        match outcome {
            Outcome::Written { output, .. } => assert!(!output.exists(), "dry run must not write"),
            other => panic!("expected Written (dry), got {other:?}"),
        }
    }

    #[test]
    fn existing_output_is_skipped_without_overwrite() {
        let input = sample!("MRS_2H");
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let first = convert_one(
            &registry,
            &input,
            Some(dir.path()),
            false,
            4,
            OutputFormat::NiiGz,
            true,
        );
        assert!(matches!(first, Outcome::Written { .. }));

        let second = convert_one(
            &registry,
            &input,
            Some(dir.path()),
            false,
            4,
            OutputFormat::NiiGz,
            true,
        );
        assert!(matches!(second, Outcome::Skipped { .. }));
    }

    #[test]
    fn an_unreadable_input_fails_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("not_a_mat_file.mat");
        std::fs::write(&bogus, b"not hdf5").unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcome = convert_one(
            &registry,
            &bogus,
            Some(dir.path()),
            false,
            4,
            OutputFormat::NiiGz,
            true,
        );
        assert!(outcome.is_failure());
    }

    #[test]
    fn parallel_conversion_preserves_input_order() {
        use rayon::prelude::*;

        let a = sample!("MRS_2H");
        let b = sample!("MRSI_2H");
        let inputs = vec![a.clone(), b.clone()];
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcomes: Vec<Outcome> = inputs
            .par_iter()
            .map(|input| {
                convert_one(
                    &registry,
                    input,
                    Some(dir.path()),
                    false,
                    4,
                    OutputFormat::NiiGz,
                    true,
                )
            })
            .collect();

        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].input(), a.as_path());
        assert_eq!(outcomes[1].input(), b.as_path());
        assert!(outcomes
            .iter()
            .all(|o| matches!(o, Outcome::Written { .. })));
    }
}
