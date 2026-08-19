# raw2nii

Convert GE fidall `.mat` MRS acquisitions (SVS and MRSI) into conformant
[NIfTI-MRS v0.11](specification.md) files. Rust workspace: a vendor-neutral
core, a GE backend, and a `raw2nii` CLI.

## Install / build

Requires the Rust toolchain in `rust-toolchain.toml` (currently `stable`) —
install via [rustup](https://rustup.rs) if you don't have it.

```bash
cargo build --release
```

The binary is `target/release/raw2nii`. `raw2nii-ge` vendors libhdf5
(`hdf5-metno`'s `static` feature), so there is no system HDF5 dependency —
the first build compiles it from source and takes a few minutes.

## Usage

```bash
raw2nii convert <path> [options]
```

`<path>` is either a single `.mat` file or a directory, searched recursively
for `.mat` files. Every convertible file is converted in parallel; output is
reported in the same order the files were discovered, regardless of which
finished first.

| Flag | Default | Effect |
|---|---|---|
| `-o, --output <dir>` | alongside each input | Output directory for every converted file. |
| `--overwrite` | off | Replace an existing output instead of skipping it. |
| `--compress-level <0-9>` | `4` | gzip level for `.nii.gz` output. |
| `--format <nii-gz\|nii>` | `nii-gz` | Write gzip-compressed or plain `.nii`. |
| `-j, --jobs <N>` | available parallelism | Worker threads for conversion. |
| `--dry-run` | off | Convert in memory and report planned output paths; write nothing. |
| `--json-log` | off | Emit one JSON object per file on stdout, followed by a summary line, instead of human-readable text. |
| `--archive <path.tar.zst>` | — | After every file converts successfully, write a verified `.tar.zst` snapshot of the input directory (the output directory, if separate, is excluded). Requires a directory input. |
| `--delete` | off | Remove the original input files once `--archive` has verified the snapshot. Refused unless `--archive` is also given. |
| `-v, -vv` | warnings only | Increase log verbosity (info / debug), on stderr. |

Output filenames follow spec §7.1: `exam{N}_series{N}_{nucleus}_{type}.{ext}`,
e.g. `exam20000_series06_2H_svs-unloc.nii.gz`.

### Examples

```bash
# Convert one file next to itself
raw2nii convert scan.mat

# Convert a whole study tree into a separate output directory
raw2nii convert -o ./converted ./raw_study

# See what would happen without writing anything
raw2nii convert --dry-run ./raw_study

# Snapshot the raw data as a verified archive, then remove the originals
raw2nii convert -o ./converted --archive ./raw_study.tar.zst --delete ./raw_study
```

`--delete` is destructive: it only runs after `--archive` has re-read the
`.tar.zst` and confirmed every original file (outside the output directory)
is present with matching byte size. If verification fails, nothing is
deleted.

## Architecture

A Cargo workspace of three crates:

| Crate | Responsibility |
|---|---|
| `raw2nii-core` | The `MrsDataset` interchange type, the `Backend` trait and `Registry`, the FFT, the NIfTI-MRS JSON header-extension builder, and the NIfTI-2 writer. Never prints, never calls `process::exit`. |
| `raw2nii-ge` | Everything GE-specific: MATLAB v7.3 (`.mat`) reading, typed header access, SVS/MRSI flavor detection, and the two readers. Implements `Backend` as `GeMatBackend`. |
| `raw2nii-cli` | The `raw2nii` binary: argument parsing, parallel conversion, reporting, and archiving. The only crate that prints or exits. |

Vendor knowledge never crosses into `raw2nii-core` — the writer only ever
sees an `MrsDataset`. Adding a new scanner vendor means adding a new crate
that implements `Backend`, registering it in `raw2nii-cli`'s `Registry`, and
touching nothing else.

Within `raw2nii-cli`: `discover` walks the filesystem for `.mat` files,
`convert` turns one input into an `Outcome` (`Written` / `Skipped` /
`Failed`) without ever panicking, `report` renders a batch of outcomes as
either human-readable lines or JSON Lines, and `archive` builds and verifies
the `--archive` snapshot. `main` only parses arguments and wires these
together.

See `specification.md` for the NIfTI-MRS format itself, and
`docs/superpowers/plans/` for the implementation plans this codebase was
built from (each documents its own design rationale and test-driven task
breakdown).

## Development

```bash
cargo test --workspace              # runs cleanly with or without sample data
cargo clippy --workspace --all-targets -- -D warnings
```

Real scanner sample datasets live in `tests/datasets/` (gitignored — not
distributed with the repo). Every test that needs them checks for their
presence via `raw2nii_ge::samples::sample_mat` and prints `SKIP: tests/datasets absent`
rather than failing when they're not there. `tests/goldens/` holds small,
tracked digest files used by the golden-file regression test; regenerate
them with `RAW2NII_BLESS=1 cargo test -p raw2nii-ge --test golden` after an
intentional output change.

## Status

GE fidall `.mat` (SVS and MRSI) is the only supported input format today.
Planned follow-on work (see `docs/superpowers/plans/`): Python bindings
(`raw2nii-py` via pyo3/maturin) and a performance benchmark suite.
