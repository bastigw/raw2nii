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

A Cargo workspace of five crates:

| Crate | Responsibility |
|---|---|
| `raw2nii-core` | The `MrsDataset` interchange type, the `Backend` trait and `Registry`, the FFT, the NIfTI-MRS JSON header-extension builder, and the NIfTI-2 writer. Never prints, never calls `process::exit`. |
| `raw2nii-ge` | Everything GE-specific: MATLAB v7.3 (`.mat`) reading, typed header access, SVS/MRSI flavor detection, and the two readers. Implements `Backend` as `GeMatBackend`. |
| `raw2nii-convert` | Shared conversion, discovery, archive, and reporting logic used by both the `raw2nii-cli` binary and the `raw2nii-py` Python bindings, so the two front ends can't drift apart on naming or archive rules. |
| `raw2nii-cli` | The `raw2nii` binary: argument parsing and wiring `raw2nii-convert` together. The only crate that prints or exits. |
| `raw2nii-py` | Python bindings (pyo3/maturin): the `raw2nii` module plus the `raw2nii` console-script CLI, both built on `raw2nii-convert`. |

Vendor knowledge never crosses into `raw2nii-core` — the writer only ever
sees an `MrsDataset`. Adding a new scanner vendor means adding a new crate
that implements `Backend`, registering it in the `Registry`, and touching
nothing else.

Within `raw2nii-convert`: `discover` walks the filesystem for `.mat` files,
`convert` turns one input into an `Outcome` (`Written` / `Skipped` /
`Failed`) without ever panicking, `report` renders a batch of outcomes as
either human-readable lines or JSON Lines, and `archive` builds and verifies
the `--archive` snapshot. `raw2nii-cli`'s `main` only parses arguments and
wires these together; `raw2nii-py` wraps the same `convert_one` and
`archive` functions for its console script.

See `specification.md` for the NIfTI-MRS format itself, and
`docs/superpowers/plans/` for the implementation plans this codebase was
built from (each documents its own design rationale and test-driven task
breakdown).

## Python bindings

`crates/raw2nii-py` exposes the core conversion API as a Python module
(`raw2nii`) via [pyo3](https://pyo3.rs)/[maturin](https://www.maturin.rs), plus
a `raw2nii` console-script CLI.

### Install as a uv tool (recommended)

```bash
uv tool install "raw2nii @ git+https://github.com/bastigw/raw2nii#subdirectory=crates/raw2nii-py"
# or, from a local checkout:
uv tool install crates/raw2nii-py
# or, once published: uv tool install raw2nii
```

This builds the extension with maturin under the hood and puts a `raw2nii`
command on your `PATH`:

```bash
raw2nii scan.mat                          # convert next to the input
raw2nii -o out --format nii-gz *.mat      # convert a batch into out/
raw2nii -r ./study                        # recurse into a directory
raw2nii -j 4 --dry-run -r ./study         # parallel dry run
raw2nii -o out --json-log ./study         # machine-readable output
raw2nii -o out --archive study.tar.zst --delete ./study
raw2nii --help
```

By default `raw2nii` prints a progress preamble, a colorized line per
converted file (green = written, yellow = skipped, red = failed), and a
colorized summary with elapsed time. Most of this status/progress output
(the preamble, skip/fail lines, and summary) goes to stderr; only the
per-file `input -> output` success lines and the archive-success message go
to stdout. Color is decided independently per stream, based on whether that
stream is a terminal; pass `--no-color` or set `NO_COLOR=1` to disable it
entirely (e.g. when piping output to a file or another program). `-v`/`--verbose`
additionally lists every discovered input file before conversion starts.
This detailed/colorized output behavior is specific to the `raw2nii` uv-tool
console script; the native `raw2nii-cli` binary has no color and no summary
line.

The console script mirrors the native CLI's option set (`-j`/`--jobs`,
`--dry-run`, `--json-log`, `--archive`/`--delete`, `-v`/`--verbose`) on top of
the same shared conversion, discovery, and archive code in
`raw2nii-convert`, plus `-r`/`--recursive` and multi-input support that the
native CLI doesn't have.

### Use as a library

Build a wheel or install into the active virtualenv in editable mode:

```bash
pip install maturin
cd crates/raw2nii-py
maturin develop --release   # or `maturin build --release` for a wheel
```

```python
import raw2nii

# One call: read, name, and write straight to disk.
paths = raw2nii.convert("scan.mat", output_dir="out", format="nii-gz")

# Or work with the dataset in memory first.
ds = raw2nii.read("scan.mat")[0]
ds.data          # complex64 numpy array, axes (x, y, z, t, [dim5, dim6, dim7]), zero-copy
ds.affine         # 4x4 nested list
ds.dwell_time_s
ds.resonant_nucleus
ds.extra          # dict of extra NIfTI-MRS JSON-extension keys
ds.write("scan.nii.gz")
```

Errors raise `raw2nii.Raw2NiiError` or one of its subclasses
(`UnsupportedFormatError`, `MissingDataError`, `MissingMetadataError`,
`GeometryError`, `DimensionMismatchError`, `BackendError`, `IoError`),
mirroring `raw2nii_core::Raw2NiiError`.

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
Planned follow-on work (see `docs/superpowers/plans/`): a performance
benchmark suite.
