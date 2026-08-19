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
