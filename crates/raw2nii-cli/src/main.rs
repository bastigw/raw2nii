//! The `raw2nii` binary. This is the only crate that prints or exits.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use raw2nii_convert::{archive, convert, discover, report};
use raw2nii_core::backend::Registry;
use raw2nii_ge::GeMatBackend;

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
        /// Worker threads for parallel conversion. Defaults to available parallelism.
        #[arg(short, long)]
        jobs: Option<usize>,
        /// Show what would be converted without writing any files.
        #[arg(long)]
        dry_run: bool,
        /// Output container: gzip-compressed (default) or plain.
        #[arg(long, value_enum, default_value_t = convert::OutputFormat::NiiGz)]
        format: convert::OutputFormat,
        /// Emit machine-readable JSON Lines instead of human-readable output.
        #[arg(long)]
        json_log: bool,
        /// Write a verified tar.zst snapshot of the input directory (output excluded) here.
        #[arg(long)]
        archive: Option<PathBuf>,
        /// After a verified --archive, delete the original input files. Requires --archive.
        #[arg(long, requires = "archive")]
        delete: bool,
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
        .with_writer(std::io::stderr)
        .init();

    let Command::Convert {
        path,
        output,
        overwrite,
        compress_level,
        jobs,
        dry_run,
        format,
        json_log,
        archive,
        delete,
    } = cli.command;

    let inputs = match discover::discover(&path) {
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

    if let Some(jobs) = jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .expect("thread pool is built exactly once, at startup");
    }

    let registry = Registry::default().with_backend(Box::new(GeMatBackend));
    let mut failures = 0usize;

    let outcomes: Vec<convert::Outcome> = inputs
        .par_iter()
        .map(|input| {
            convert::convert_one(
                &registry,
                input,
                output.as_deref(),
                overwrite,
                compress_level,
                format,
                !dry_run,
            )
        })
        .collect();

    if json_log {
        for outcome in &outcomes {
            println!("{}", report::render_json_line(outcome));
            if outcome.is_failure() {
                failures += 1;
            }
        }
        println!("{}", report::render_summary_json(&outcomes));
    } else {
        for outcome in &outcomes {
            match outcome {
                convert::Outcome::Written { input, output } => {
                    println!("{} -> {}", input.display(), output.display())
                }
                convert::Outcome::Skipped { input, reason } => {
                    tracing::warn!("{}: {reason}", input.display())
                }
                convert::Outcome::Failed { input, error } => {
                    eprintln!("error: {}: {error}", input.display());
                    failures += 1;
                }
            }
        }
    }

    if let Some(archive_path) = archive {
        if failures > 0 {
            eprintln!("error: not archiving: {failures} file(s) failed to convert");
        } else if !path.is_dir() {
            eprintln!(
                "error: --archive requires a directory input, got a file: {}",
                path.display()
            );
            failures += 1;
        } else {
            let exclude = output
                .clone()
                .unwrap_or_else(|| path.join(".raw2nii-no-exclude"));
            match archive::build_archive(&path, &archive_path, &exclude)
                .and_then(|_| archive::verify_archive(&archive_path, &path, &exclude))
            {
                Ok(true) => {
                    println!("archived {} -> {}", path.display(), archive_path.display());
                    if delete {
                        for outcome in &outcomes {
                            if let convert::Outcome::Written { input, .. } = outcome {
                                if let Err(e) = std::fs::remove_file(input) {
                                    eprintln!("error: could not delete {}: {e}", input.display());
                                    failures += 1;
                                }
                            }
                        }
                    }
                }
                Ok(false) => {
                    eprintln!("error: archive verification failed, originals were not deleted");
                    failures += 1;
                }
                Err(e) => {
                    eprintln!("error: archiving {}: {e}", path.display());
                    failures += 1;
                }
            }
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
}
