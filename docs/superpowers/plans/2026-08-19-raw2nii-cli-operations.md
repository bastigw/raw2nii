# raw2nii CLI Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the minimal `raw2nii convert` CLI (single-threaded, always writes, no source-archiving) into a batch tool: parallel conversion, `--dry-run`, `--format`, `--json-log`, and an `--archive`/`--delete` pair that lets a user replace a raw dataset directory with a verified `.tar.zst` before removing the originals.

**Architecture:** `raw2nii-cli` grows from one 169-line `main.rs` into five focused modules — `discover` (filesystem walk), `convert` (one input → one outcome), `report` (outcome rendering, human or JSON), `archive` (tar.zst build + verify), and `main` (clap parsing and wiring only). `raw2nii-core` and `raw2nii-ge` are untouched: every new capability is CLI-only orchestration around the existing `Backend`/`Registry`/`write_file` contract.

**Tech Stack:** adds `rayon` (parallel conversion), `tar` + `zstd` (archiving), `serde` + `serde_json` (JSON-lines log), `tempfile` (dev-dependency, CLI-local filesystem tests).

**Spec:** `docs/superpowers/plans/2026-08-19-raw2nii-core-and-ge-backend.md` (§ Follow-on Plans, Plan 2) — this plan has no separate design doc; the four bullet points there (`--archive`, `--delete`, `--dry-run`, `--format`, `--json-log`, rayon parallelism, recursive discovery already exists) are the full spec.

## Global Constraints

- Rust edition 2021 (workspace), toolchain per `rust-toolchain.toml` (`stable`).
- Only `raw2nii-cli` may print or call `std::process::exit`; `raw2nii-core` and `raw2nii-ge` are untouched by this plan.
- New third-party crates are added once to `[workspace.dependencies]` in the root `Cargo.toml`, then referenced with `.workspace = true` from `crates/raw2nii-cli/Cargo.toml` — follow the existing pattern (`ndarray`, `thiserror`, etc.).
- `tests/datasets/` is gitignored and may be absent. Every test touching it must skip cleanly (`eprintln!("SKIP: ...")` + `return`), never fail. Reuse `raw2nii_ge::samples::sample_mat`.
- Filesystem tests that do **not** need real scanner data (discovery, archiving) use `tempfile::tempdir()` so they run identically with or without `tests/datasets/`.
- `--delete` must never run implicitly. It is refused unless `--archive` is also given and the archive verified successfully in the same invocation.
- Conversion order in output must stay deterministic (sorted input order), even when converted in parallel.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/raw2nii-cli/Cargo.toml` | add `rayon`, `tar`, `zstd`, `serde`, `serde_json`; add `tempfile` as `[dev-dependencies]` |
| `crates/raw2nii-cli/src/discover.rs` | recursive `.mat` discovery under a file or directory (extracted from `main.rs`) |
| `crates/raw2nii-cli/src/convert.rs` | `Outcome` enum, `convert_one`, `stem_for`, `OutputFormat` (extracted + extended from `main.rs`) |
| `crates/raw2nii-cli/src/report.rs` | render a `&[Outcome]` as human text or JSON-lines |
| `crates/raw2nii-cli/src/archive.rs` | `build_archive`, `verify_archive`, `delete_originals` |
| `crates/raw2nii-cli/src/main.rs` | clap CLI surface, thread-pool sizing, wiring the modules above |

---

### Task 1: Extract `discover` into its own module

**Files:**
- Create: `crates/raw2nii-cli/src/discover.rs`
- Modify: `crates/raw2nii-cli/src/main.rs` (remove the inline `discover` fn, add `mod discover;`, call `discover::discover`)
- Modify: `crates/raw2nii-cli/Cargo.toml` (add `tempfile` dev-dependency)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn discover(path: &Path) -> std::io::Result<Vec<PathBuf>>` — sorted, recursive `.mat` file list; a file argument returns itself unchecked.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-cli/src/discover.rs`:

```rust
//! Filesystem discovery: which files are candidate conversion inputs.

use std::path::{Path, PathBuf};

/// All `.mat` files under `path`, sorted. If `path` is itself a file, it is
/// returned unchecked — the backend registry decides whether it can read it.
pub fn discover(path: &Path) -> std::io::Result<Vec<PathBuf>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_argument_is_returned_unchecked() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("archive.mat");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(discover(&f).unwrap(), vec![f]);
    }

    #[test]
    fn finds_mat_files_recursively_and_sorts_them() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("b.mat"), b"").unwrap();
        std::fs::write(dir.path().join("a.txt"), b"").unwrap();
        std::fs::write(dir.path().join("sub/a.mat"), b"").unwrap();

        let found = discover(dir.path()).unwrap();
        assert_eq!(
            found,
            vec![dir.path().join("b.mat"), dir.path().join("sub/a.mat")]
        );
    }

    #[test]
    fn an_empty_directory_yields_no_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(discover(dir.path()).unwrap().is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli discover`
Expected: FAIL — `tempfile` is not a dependency yet, so the crate does not compile.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/raw2nii-cli/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

Add `mod discover;` near the top of `crates/raw2nii-cli/src/main.rs`, delete the existing inline `fn discover(...)`, and change its one call site from `discover(&path)` to `discover::discover(&path)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli discover`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "refactor: extract file discovery into its own module"
```

---

### Task 2: Extract `convert` into its own module, with an `Outcome` type

**Files:**
- Create: `crates/raw2nii-cli/src/convert.rs`
- Modify: `crates/raw2nii-cli/src/main.rs` (remove inline `convert_one`/`stem_for`, add `mod convert;`, adapt the call site)

**Interfaces:**
- Consumes: `raw2nii_core::backend::Registry`, `raw2nii_core::write::write_file` (existing), `discover::discover` (Task 1).
- Produces:
  - `pub enum Outcome { Written { input: PathBuf, output: PathBuf }, Skipped { input: PathBuf, reason: String }, Failed { input: PathBuf, error: String } }`
  - `pub fn convert_one(registry: &Registry, input: &Path, output_dir: Option<&Path>, overwrite: bool, compress_level: u32, format: OutputFormat) -> Outcome` — never panics, never returns `Err`; all failure modes become `Outcome::Failed`.
  - `pub enum OutputFormat { NiiGz, Nii }` with `impl OutputFormat { pub fn extension(self) -> &'static str }` returning `"nii.gz"` / `"nii"`.

**Context:** `convert_one` currently returns `Result<Option<PathBuf>, Box<dyn Error>>` and the caller in `main` prints and tallies failures inline. Folding success/skip/failure into one `Outcome` enum is what Tasks 3–5 need: rayon collects a `Vec<Outcome>` in input order, `report` renders it two ways, and dry-run/archive both need to inspect it after the fact instead of re-deriving state from side effects.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-cli/src/convert.rs`:

```rust
//! Converting one discovered input into zero or one written file.

use std::path::{Path, PathBuf};

use raw2nii_core::backend::Registry;
use raw2nii_core::dataset::MrsDataset;
use raw2nii_core::write::write_file;
use raw2nii_ge::{is_unlocalised, output_stem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    NiiGz,
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

#[derive(Debug, Clone)]
pub enum Outcome {
    Written { input: PathBuf, output: PathBuf },
    Skipped { input: PathBuf, reason: String },
    Failed { input: PathBuf, error: String },
}

impl Outcome {
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
        Err(e) => return Outcome::Failed { input, error: e.to_string() },
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
            if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| {
                write_file(ds, &out, compress_level).map_err(std::io::Error::other)
            }) {
                return Outcome::Failed { input, error: e.to_string() };
            }
        }
        last = Some(out);
    }

    match last {
        Some(output) => Outcome::Written { input, output },
        None => Outcome::Skipped { input, reason: "backend produced no datasets".to_string() },
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

        let outcome = convert_one(&registry, &input, Some(dir.path()), false, 4, OutputFormat::NiiGz, true);
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

        let outcome = convert_one(&registry, &input, Some(dir.path()), false, 4, OutputFormat::Nii, true);
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

        let outcome = convert_one(&registry, &input, Some(dir.path()), false, 4, OutputFormat::NiiGz, false);
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

        let first = convert_one(&registry, &input, Some(dir.path()), false, 4, OutputFormat::NiiGz, true);
        assert!(matches!(first, Outcome::Written { .. }));

        let second = convert_one(&registry, &input, Some(dir.path()), false, 4, OutputFormat::NiiGz, true);
        assert!(matches!(second, Outcome::Skipped { .. }));
    }

    #[test]
    fn an_unreadable_input_fails_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("not_a_mat_file.mat");
        std::fs::write(&bogus, b"not hdf5").unwrap();
        let registry = Registry::default().with_backend(Box::new(GeMatBackend));

        let outcome = convert_one(&registry, &bogus, Some(dir.path()), false, 4, OutputFormat::NiiGz, true);
        assert!(outcome.is_failure());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli convert::`
Expected: FAIL — `convert` module does not exist, and `clap::ValueEnum` derive is unused elsewhere so this is a clean new-symbol failure.

- [ ] **Step 3: Write minimal implementation**

Add `mod convert;` to `crates/raw2nii-cli/src/main.rs`. Delete the inline `convert_one` and `stem_for` functions and their `use raw2nii_ge::{is_unlocalised, output_stem};` import. Update the call site inside the `for input in inputs` loop:

```rust
    for input in inputs {
        let outcome = convert::convert_one(
            &registry,
            &input,
            output.as_deref(),
            overwrite,
            compress_level,
            convert::OutputFormat::NiiGz,
            true,
        );
        match &outcome {
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
```

(`OutputFormat::NiiGz` and `write: true` are hard-coded here; Task 3 threads them through from clap, Task 4 wires up `--dry-run`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli`
Expected: PASS — all `convert::` and `discover::` tests, plus the CLI still builds.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "refactor: extract conversion into an Outcome-returning module"
```

---

### Task 3: Parallel conversion with rayon, deterministic order

**Files:**
- Modify: `crates/raw2nii-cli/Cargo.toml` (add `rayon`), `crates/raw2nii-cli/src/main.rs`

**Interfaces:**
- Consumes: `discover::discover`, `convert::{convert_one, Outcome, OutputFormat}` (Tasks 1–2).
- Produces: `main` converts every discovered input in parallel and collects `Vec<Outcome>` in the same order `discover` returned (i.e. sorted, not completion order); a `--jobs N` flag caps the thread pool.

**Context:** `Registry` holds `Box<dyn Backend>` where `Backend: Send + Sync` is already required (`crates/raw2nii-core/src/backend.rs:19`), so sharing one `Registry` across threads via a plain reference is sound. `rayon`'s `par_iter().map(f).collect::<Vec<_>>()` preserves input order regardless of which thread finishes first — no explicit sorting needed.

- [ ] **Step 1: Write the failing test**

Add to `crates/raw2nii-cli/src/convert.rs`, inside `mod tests`:

```rust
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
                convert_one(&registry, input, Some(dir.path()), false, 4, OutputFormat::NiiGz, true)
            })
            .collect();

        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].input(), a.as_path());
        assert_eq!(outcomes[1].input(), b.as_path());
        assert!(outcomes.iter().all(|o| matches!(o, Outcome::Written { .. })));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli convert::tests::parallel_conversion_preserves_input_order`
Expected: FAIL — `rayon` is not a dependency, so `use rayon::prelude::*` does not resolve.

- [ ] **Step 3: Write minimal implementation**

Add to the root `Cargo.toml` `[workspace.dependencies]`:

```toml
rayon = "1"
```

Add to `crates/raw2nii-cli/Cargo.toml` `[dependencies]`:

```toml
rayon.workspace = true
```

In `crates/raw2nii-cli/src/main.rs`, add a `--jobs` flag to the `Convert` variant:

```rust
        /// Worker threads for parallel conversion. Defaults to available parallelism.
        #[arg(short, long)]
        jobs: Option<usize>,
```

Destructure it alongside the existing fields, then before the conversion loop:

```rust
    if let Some(jobs) = jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .expect("thread pool is built exactly once, at startup");
    }
```

Replace the sequential `for input in inputs { ... }` loop with a parallel map that collects, then a sequential report pass (order is preserved by `collect`, so reporting stays deterministic even though conversion is not):

```rust
    use rayon::prelude::*;
    let outcomes: Vec<convert::Outcome> = inputs
        .par_iter()
        .map(|input| {
            convert::convert_one(
                &registry,
                input,
                output.as_deref(),
                overwrite,
                compress_level,
                convert::OutputFormat::NiiGz,
                true,
            )
        })
        .collect();

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/raw2nii-cli/
git commit -m "feat: convert inputs in parallel with rayon"
```

---

### Task 4: `--dry-run` and `--format`

**Files:**
- Modify: `crates/raw2nii-cli/src/main.rs`

**Interfaces:**
- Consumes: `convert::{convert_one, OutputFormat}` (Tasks 2–3).
- Produces: `raw2nii convert --dry-run <path>` reports every planned output without writing; `raw2nii convert --format nii <path>` writes uncompressed `.nii` instead of `.nii.gz`.

- [ ] **Step 1: Write the failing test**

Add an integration test, `crates/raw2nii-cli/tests/dry_run.rs`:

```rust
//! End-to-end CLI behaviour for --dry-run and --format.

use std::path::PathBuf;
use std::process::Command;

fn sample(name: &str) -> Option<PathBuf> {
    raw2nii_ge::samples::sample_mat(name)
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_raw2nii"))
}

#[test]
fn dry_run_reports_but_does_not_write() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    let output = Command::new(bin())
        .args(["convert", "--dry-run", "-o"])
        .arg(dir.path())
        .arg(&input)
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "dry run must write nothing");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("exam20000_series06_2H_svs-unloc.nii.gz"), "{stdout}");
}

#[test]
fn format_nii_writes_uncompressed() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    let status = Command::new(bin())
        .args(["convert", "--format", "nii", "-o"])
        .arg(dir.path())
        .arg(&input)
        .status()
        .unwrap();

    assert!(status.success());
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, vec!["exam20000_series06_2H_svs-unloc.nii".to_string()]);
}
```

Add `tempfile = "3"` and (already present from Task 1) to `[dev-dependencies]`, and add `raw2nii-ge` to `[dev-dependencies]` if it is only a normal dependency today — check `crates/raw2nii-cli/Cargo.toml`; it is already a `[dependencies]` entry, so no change needed there.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli --test dry_run`
Expected: FAIL — `--dry-run` and `--format` are not recognised clap arguments (clap exits non-zero with a usage error).

- [ ] **Step 3: Write minimal implementation**

In `crates/raw2nii-cli/src/main.rs`, add to the `Convert` variant:

```rust
        /// Show what would be converted without writing any files.
        #[arg(long)]
        dry_run: bool,
        /// Output container: gzip-compressed (default) or plain.
        #[arg(long, value_enum, default_value_t = convert::OutputFormat::NiiGz)]
        format: convert::OutputFormat,
```

`OutputFormat` needs `Default`-like CLI ergonomics; add `#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]` (already present from Task 2) and implement `Display` so `default_value_t` can print it, with an explicit match so the printed and parsed spellings are guaranteed to match:

```rust
impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::NiiGz => f.write_str("nii-gz"),
            OutputFormat::Nii => f.write_str("nii"),
        }
    }
}
```

and give each variant an explicit clap name so `--format nii` (not `--format nii-gz`/`niiGz`) is the CLI spelling:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    #[value(name = "nii-gz")]
    NiiGz,
    #[value(name = "nii")]
    Nii,
}
```

Destructure `dry_run` and `format` out of `Command::Convert { .. }`, and pass them into the parallel map from Task 3:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli --test dry_run`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "feat: --dry-run and --format nii/nii-gz"
```

---

### Task 5: `--json-log`

**Files:**
- Create: `crates/raw2nii-cli/src/report.rs`
- Modify: `crates/raw2nii-cli/src/main.rs`, `crates/raw2nii-cli/Cargo.toml`

**Interfaces:**
- Consumes: `convert::Outcome` (Task 2).
- Produces:
  - `pub fn render_json_line(o: &Outcome) -> String` — one compact JSON object per line, e.g. `{"status":"written","input":"...","output":"..."}` / `{"status":"skipped","input":"...","reason":"..."}` / `{"status":"failed","input":"...","error":"..."}`.
  - `pub fn render_summary_json(outcomes: &[Outcome]) -> String` — one JSON object: `{"written":N,"skipped":N,"failed":N}`.
  - `raw2nii convert --json-log <path>` prints one `render_json_line` per outcome to stdout instead of the human-readable lines, followed by one `render_summary_json` line.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-cli/src/report.rs`:

```rust
//! Rendering conversion outcomes for humans or for machine consumption.

use serde_json::json;

use crate::convert::Outcome;

pub fn render_json_line(o: &Outcome) -> String {
    let value = match o {
        Outcome::Written { input, output } => json!({
            "status": "written",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
        }),
        Outcome::Skipped { input, reason } => json!({
            "status": "skipped",
            "input": input.display().to_string(),
            "reason": reason,
        }),
        Outcome::Failed { input, error } => json!({
            "status": "failed",
            "input": input.display().to_string(),
            "error": error,
        }),
    };
    value.to_string()
}

pub fn render_summary_json(outcomes: &[Outcome]) -> String {
    let written = outcomes.iter().filter(|o| matches!(o, Outcome::Written { .. })).count();
    let skipped = outcomes.iter().filter(|o| matches!(o, Outcome::Skipped { .. })).count();
    let failed = outcomes.iter().filter(|o| o.is_failure()).count();
    json!({ "written": written, "skipped": skipped, "failed": failed }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn renders_a_written_outcome() {
        let o = Outcome::Written {
            input: PathBuf::from("in.mat"),
            output: PathBuf::from("out.nii.gz"),
        };
        let line = render_json_line(&o);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "written");
        assert_eq!(v["output"], "out.nii.gz");
    }

    #[test]
    fn renders_a_failed_outcome() {
        let o = Outcome::Failed {
            input: PathBuf::from("in.mat"),
            error: "boom".to_string(),
        };
        let line = render_json_line(&o);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"], "boom");
    }

    #[test]
    fn summary_counts_each_status() {
        let outcomes = vec![
            Outcome::Written { input: PathBuf::from("a"), output: PathBuf::from("a.nii.gz") },
            Outcome::Skipped { input: PathBuf::from("b"), reason: "exists".to_string() },
            Outcome::Failed { input: PathBuf::from("c"), error: "boom".to_string() },
            Outcome::Failed { input: PathBuf::from("d"), error: "boom".to_string() },
        ];
        let v: serde_json::Value = serde_json::from_str(&render_summary_json(&outcomes)).unwrap();
        assert_eq!(v["written"], 1);
        assert_eq!(v["skipped"], 1);
        assert_eq!(v["failed"], 2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli report::`
Expected: FAIL — `serde_json` (already a workspace dep used by `raw2nii-core`, but not yet by `raw2nii-cli`) is not in scope, and the `report` module does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `crates/raw2nii-cli/Cargo.toml` `[dependencies]`:

```toml
serde_json.workspace = true
```

Add `mod report;` and `mod convert;` (already added in Task 2) to `crates/raw2nii-cli/src/main.rs`. Add a `--json-log` flag to `Convert`:

```rust
        /// Emit machine-readable JSON Lines instead of human-readable output.
        #[arg(long)]
        json_log: bool,
```

Replace the reporting loop (from Task 4) with a branch on `json_log`:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "feat: --json-log for machine-readable conversion output"
```

---

### Task 6: `archive` module — build and verify a `.tar.zst`

**Files:**
- Create: `crates/raw2nii-cli/src/archive.rs`
- Modify: `crates/raw2nii-cli/Cargo.toml` (add `tar`, `zstd`)

**Interfaces:**
- Consumes: nothing (pure filesystem module; `main.rs` wires it up in Task 7).
- Produces:
  - `pub fn build_archive(source_dir: &Path, archive_path: &Path, exclude: &Path) -> std::io::Result<()>` — writes a `.tar.zst` of every file under `source_dir`, skipping anything under `exclude` (typically the output directory) and skipping `archive_path` itself if it happens to sit inside `source_dir`.
  - `pub fn verify_archive(archive_path: &Path, source_dir: &Path, exclude: &Path) -> std::io::Result<bool>` — re-reads the archive and returns `true` only if every non-excluded file under `source_dir` appears in the archive with the same byte length.

**Context:** `exclude` exists because "the parent directory, output excluded" (spec follow-on bullet) means: when `-o` points inside the same tree being archived, the freshly written `.nii.gz` files must not end up inside their own source archive.

- [ ] **Step 1: Write the failing test**

Create `crates/raw2nii-cli/src/archive.rs`:

```rust
//! Building and verifying a `.tar.zst` snapshot of a source directory,
//! ahead of `--delete` removing the originals.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Files under `dir`, relative to `dir`, skipping anything under `exclude`
/// or equal to `skip_path`.
fn collect_files(dir: &Path, exclude: &Path, skip_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let p = entry?.path();
            if p == skip_path || p.starts_with(exclude) {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn build_archive(source_dir: &Path, archive_path: &Path, exclude: &Path) -> std::io::Result<()> {
    let files = collect_files(source_dir, exclude, archive_path)?;

    let file = File::create(archive_path)?;
    let encoder = zstd::stream::write::Encoder::new(file, 0)?.auto_finish();
    let mut builder = tar::Builder::new(encoder);
    for f in &files {
        let rel = f.strip_prefix(source_dir).expect("collect_files yields children of source_dir");
        builder.append_path_with_name(f, rel)?;
    }
    builder.into_inner()?.flush()
}

pub fn verify_archive(archive_path: &Path, source_dir: &Path, exclude: &Path) -> std::io::Result<bool> {
    let expected = collect_files(source_dir, exclude, archive_path)?;

    let file = File::open(archive_path)?;
    let decoder = zstd::stream::read::Decoder::new(file)?;
    let mut archive = tar::Archive::new(decoder);

    let mut sizes = std::collections::HashMap::new();
    for entry in archive.entries()? {
        let entry = entry?;
        let path = entry.path()?.into_owned();
        sizes.insert(path, entry.header().size()?);
    }

    for f in &expected {
        let rel = f.strip_prefix(source_dir).expect("collect_files yields children of source_dir");
        let on_disk = std::fs::metadata(f)?.len();
        match sizes.get(rel) {
            Some(&archived) if archived == on_disk => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, contents: &[u8]) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, contents).unwrap();
    }

    #[test]
    fn round_trips_a_small_tree() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");
        write(src.path(), "sub/b.mat", b"world!!");

        let out = tempfile::tempdir().unwrap();
        let archive_path = out.path().join("snapshot.tar.zst");
        let exclude = out.path().join("nonexistent-output-dir");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        assert!(verify_archive(&archive_path, src.path(), &exclude).unwrap());
    }

    #[test]
    fn excludes_the_output_directory() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");
        write(src.path(), "out/a.nii.gz", b"converted bytes");

        let archive_path = src.path().join("out").join("snapshot.tar.zst");
        let exclude = src.path().join("out");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        assert!(verify_archive(&archive_path, src.path(), &exclude).unwrap());

        // The excluded file must genuinely be absent from the archive.
        let file = File::open(&archive_path).unwrap();
        let decoder = zstd::stream::read::Decoder::new(file).unwrap();
        let mut archive = tar::Archive::new(decoder);
        let names: Vec<String> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(!names.iter().any(|n| n.contains("a.nii.gz")));
    }

    #[test]
    fn verify_fails_after_a_source_file_is_modified() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");

        let out = tempfile::tempdir().unwrap();
        let archive_path = out.path().join("snapshot.tar.zst");
        let exclude = out.path().join("no-such-dir");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        write(src.path(), "a.mat", b"hello, but now longer");

        assert!(!verify_archive(&archive_path, src.path(), &exclude).unwrap());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli archive::`
Expected: FAIL — `tar` and `zstd` are not dependencies yet.

- [ ] **Step 3: Write minimal implementation**

Add to the root `Cargo.toml` `[workspace.dependencies]`:

```toml
tar = "0.4"
zstd = "0.13"
```

Add to `crates/raw2nii-cli/Cargo.toml` `[dependencies]`:

```toml
tar.workspace = true
zstd.workspace = true
```

Add `mod archive;` to `crates/raw2nii-cli/src/main.rs` (wiring happens in Task 7).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli archive::`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/raw2nii-cli/
git commit -m "feat: tar.zst archive build and byte-size verification"
```

---

### Task 7: Wire `--archive` and `--delete` into the CLI

**Files:**
- Modify: `crates/raw2nii-cli/src/main.rs`

**Interfaces:**
- Consumes: `archive::{build_archive, verify_archive}` (Task 6), `outcomes: Vec<convert::Outcome>` (Task 3).
- Produces: `raw2nii convert --archive <path.tar.zst> <dir>` archives `<dir>` (excluding the output directory) after every conversion succeeds; `raw2nii convert --archive <path> --delete <dir>` additionally removes the original input files once the archive is verified. `--delete` without `--archive` is a clap-level usage error, not a runtime one.

**Context:** Archiving and deletion only make sense for a directory input with no failures — a single-file input has nothing worth snapshotting, and deleting after a partial failure would destroy data whose conversion never happened. Both restrictions are enforced explicitly, with a clear message, rather than silently doing something narrower than what was asked.

- [ ] **Step 1: Write the failing test**

Add to `crates/raw2nii-cli/tests/dry_run.rs` (the file created in Task 4):

```rust
#[test]
fn delete_without_archive_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(bin())
        .args(["convert", "--delete"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--delete") && stderr.contains("--archive"), "{stderr}");
}

#[test]
fn archive_then_delete_removes_originals_but_keeps_outputs() {
    let Some(sample_dir) = raw2nii_ge::samples::sample_mat("MRS_2H").and_then(|p| p.parent().map(|d| d.to_path_buf())) else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };

    // Work on a throwaway copy: --delete is destructive.
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("MRS_2H");
    copy_dir(&sample_dir, &src);

    let out_dir = work.path().join("out");
    let archive_path = work.path().join("MRS_2H.tar.zst");

    let status = Command::new(bin())
        .args(["convert", "-o"])
        .arg(&out_dir)
        .arg("--archive")
        .arg(&archive_path)
        .arg("--delete")
        .arg(&src)
        .status()
        .unwrap();

    assert!(status.success());
    assert!(archive_path.exists(), "archive must be written");
    assert!(
        std::fs::read_dir(&out_dir).unwrap().next().is_some(),
        "converted output must survive --delete"
    );
    let mat_files: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("mat"))
        .collect();
    assert!(mat_files.is_empty(), "--delete must remove the original .mat files");
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), &dest).unwrap();
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p raw2nii-cli --test dry_run archive_then_delete`
Expected: FAIL — `--archive`/`--delete` are not recognised clap arguments.

- [ ] **Step 3: Write minimal implementation**

Add to the `Convert` variant in `crates/raw2nii-cli/src/main.rs`:

```rust
        /// Write a verified tar.zst snapshot of the input directory (output excluded) here.
        #[arg(long)]
        archive: Option<PathBuf>,
        /// After a verified --archive, delete the original input files. Requires --archive.
        #[arg(long, requires = "archive")]
        delete: bool,
```

clap's `requires = "archive"` turns the missing-archive case into the usage error the test checks for, so no runtime check is needed for that half.

After the conversion + reporting block, before the final `if failures > 0` check:

```rust
    if let Some(archive_path) = archive {
        if failures > 0 {
            eprintln!("error: not archiving: {failures} file(s) failed to convert");
        } else if !path.is_dir() {
            eprintln!("error: --archive requires a directory input, got a file: {}", path.display());
            failures += 1;
        } else {
            let exclude = output.clone().unwrap_or_else(|| path.clone());
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
```

Note the `exclude` default: when `-o` is not given, outputs land next to each input inside `path` itself, so excluding `path` from its own archive would exclude everything. In that no-`-o` case there is nothing separate to exclude — `build_archive`/`verify_archive` are still correct because `exclude` only ever removes paths that are literally under it, and `path == exclude` degenerates to "archive nothing", which is wrong. Handle it explicitly: when `output` is `None`, pass a non-existent sentinel path as `exclude` instead of `path`:

```rust
            let exclude = output.clone().unwrap_or_else(|| path.join(".raw2nii-no-exclude"));
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p raw2nii-cli`
Expected: PASS — all unit and integration tests, including `archive_then_delete_removes_originals_but_keeps_outputs`.

- [ ] **Step 5: Commit**

```bash
git add crates/raw2nii-cli/
git commit -m "feat: --archive/--delete, gated on verified tar.zst snapshots"
```

---

## Definition of Done

- [ ] `cargo test --workspace` passes with `tests/datasets/` present.
- [ ] `cargo test --workspace` passes with `tests/datasets/` absent (data-dependent tests print SKIP).
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- [ ] `raw2nii convert --dry-run tests/datasets/MRS_2H` prints the planned output path and writes nothing.
- [ ] `raw2nii convert --format nii -o /tmp/out tests/datasets/MRS_2H` writes `exam20000_series06_2H_svs-unloc.nii` (not `.nii.gz`).
- [ ] `raw2nii convert --json-log tests/datasets` prints one JSON object per file plus a summary line, and every line parses as JSON.
- [ ] `raw2nii convert --delete tests/datasets` (no `--archive`) exits non-zero with a usage error naming both flags.
- [ ] On a throwaway copy: `raw2nii convert --archive out.tar.zst --delete <dir>` leaves the converted outputs in place, removes the original `.mat` files, and the archive verifies.
- [ ] `raw2nii-core` and `raw2nii-ge` are unmodified by this plan (`git diff --stat` on `main` shows changes confined to `Cargo.toml` and `crates/raw2nii-cli/`).

## Follow-on Plans

- **Plan 3 — Python bindings:** unchanged from Plan 1's follow-on list — `raw2nii-py` via pyo3/maturin.
- **Performance (spec §11):** now unblocked — Task 3 gives rayon-parallel conversion something to benchmark against the sequential baseline.
