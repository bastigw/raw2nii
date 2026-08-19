# raw2nii — GE MRS to NIfTI-MRS converter

Design document. 2026-08-19.

## 1. Purpose

Convert GE MR spectroscopy data into the NIfTI-MRS standard (v0.11, `specification.md`
in the repository root) with a fast, dependency-free CLI and a thin Python binding.

The immediate input is the `.mat` output of the `fidall` reconstruction pipeline. The
architecture admits further containers (GE ScanArchive, P-file, other vendors) without
restructuring.

### Goals

- Convert fidall `.mat` files (SVS and MRSI) to conformant NIfTI-MRS.
- Single static binary, no MATLAB, no Python at runtime.
- Optional archiving and deletion of source data after successful conversion.
- Python wrapper giving in-memory access to converted data.

### Non-goals

- Reading GE ScanArchive `.h5` or P-files. The seam exists; the backends do not.
- Other vendors. Same.
- Any analysis, fitting, or processing beyond what conformance requires.
- Reimplementing the fidall reconstruction.

## 2. Input data

Seven sample datasets live in `tests/datasets/`. All `.mat` files are MATLAB v7.3, which
is HDF5, and therefore readable without MATLAB.

| dataset          | flavor | `.mat` | notes                                 |
| ---------------- | ------ | ------ | ------------------------------------- |
| `MRS_2H`         | SVS    | yes    | unlocalised                           |
| `MRS_2H_slab`    | SVS    | yes    | 80 mm slab, tilted                    |
| `MRS_2H_TE_60`   | SVS    | yes    | psd `echocsi`, not `fidall2`          |
| `MRS_2H_TI_400`  | SVS    | yes    | unlocalised                           |
| `MRSI_13C`       | MRSI   | yes    | 2D, 8x8                               |
| `MRSI_2H`        | MRSI   | yes    | 3D spiral, acquired 10^3, stored 16^3 |
| `BS_prescan_13C` | —      | **no** | prescan only; used as a negative test |

### 2.1 The two flavors

|        | SVS                                                    | MRSI                                                               |
| ------ | ------------------------------------------------------ | ------------------------------------------------------------------ |
| data   | `/fid` and `/spec`                                     | `/spec` only                                                       |
| params | rich `/par` (samples, rows, bw, f0, echo_time, …)      | thin `/par`, plus top-level `nn`, `zf`, `dim`, `nx/ny/nz/nt/nspec` |
| header | `/h` (GE `rdb_hdr`, `image`, `series`, `exam` structs) | same                                                               |

Detection: `/fid` and `/par/samples` present implies SVS. `/spec`, `/nn` and `/dim`
present implies MRSI. Neither implies the backend declines with a diagnostic listing the
top-level variables found.

### 2.2 Dimension conventions

MATLAB v7.3 stores array dimensions reversed relative to HDF5. Shapes reported by
`h5dump` are therefore backwards. Verified against the scalar size vectors:

| dataset            | `h5dump` shape | MATLAB shape   | corroborated by                   |
| ------------------ | -------------- | -------------- | --------------------------------- |
| `MRS_2H` `/fid`    | (2048, 64)     | (64, 2048)     | `par.rows=64`, `par.samples=2048` |
| `MRSI_13C` `/spec` | (8, 8, 544)    | (544, 8, 8)    | `nn = [544,8,8,1,1,1]`            |
| `MRSI_2H` `/spec`  | (16,16,16,700) | (700,16,16,16) | `zf = [700,16,16,16,1,1]`         |

`nn` is the acquired size vector `[nspec, nx, ny, nz, nt, nc]`. `zf` is the reconstructed
(zero-filled) size. **The grid size comes from the actual dataspace, which equals `zf`,
never from `nn`.** MRSI_2H acquired 10^3 and stores 16^3; the zero-filled grid is the
correct output grid.

MATLAB drops trailing singleton dimensions, so `MRSI_13C` `/spec` is 3-D rather than 4-D.
The reader right-pads the size vector to 6 elements before any other processing.

Complex data is stored as an HDF5 compound type with `real` and `imag` members.

## 3. Architecture

A Cargo workspace. The output type `MrsDataset` is the contract every backend produces
and the only type the NIfTI writer sees.

```
crates/
├── raw2nii-core/
│   ├── dataset.rs        MrsDataset — the output contract
│   ├── backend.rs        trait Backend, Confidence, Registry
│   ├── fft.rs            ifftshift + ifft(ortho)
│   ├── meta/             FieldMap machinery, JSON extension builder
│   ├── geom.rs           vendor-neutral affine algebra
│   └── write/nifti.rs    NIfTI-2 writer, ecode 44
├── raw2nii-ge/           feature = "ge"
│   ├── header/           rdb_hdr/image/series/exam structs, FIELD_MAP, GE affine
│   ├── mat/              container: fidall .mat v7.3   (this version)
│   ├── scanarchive/      container: ScanArchive .h5    (future)
│   └── pfile/            container: P*.7               (future)
├── raw2nii-cli/          clap, rayon, tracing, archive/delete
└── raw2nii-py/           pyo3 + maturin
```

### 3.1 The extension seam

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Cheap: extension, magic bytes, a few key probes. No full parse.
    fn probe(&self, path: &Path) -> Confidence;   // No | Maybe | Yes
    fn convert(&self, path: &Path, opts: &Opts) -> Result<Vec<MrsDataset>>;
}
```

`Vec<MrsDataset>` because one container may hold several datasets. The registry selects
the highest-confidence backend; `--format <name>` overrides.

Abstraction is at the outer boundary only. There is deliberately no common intermediate
representation across vendors: a Siemens twix file has nothing resembling a MATLAB struct
tree, and forcing one would buy nothing. Below `MrsDataset`, vendors share only the FFT,
the JSON builder, and the NIfTI writer — which is genuinely all they have in common.

The real reuse axis within a vendor is the header, not the container. ScanArchive `.h5`,
P-files and the fidall `.mat` all carry the same GE `rdb_hdr`/`image`/`series` structs, so
`raw2nii-ge/header/` is written once and every future GE container reuses it. Only byte
extraction differs.

Adding a backend later: new crate, `impl Backend`, register it, add a sample directory.
No change to core, the writer, or existing backends.

SVS-versus-MRSI is a fidall-`.mat` distinction, not a universal one, and is therefore a
private detail inside `raw2nii-ge/mat/`.

### 3.2 Data flow

```
.mat ──probe──> Backend ──convert──> MrsDataset {
                                       data:   Array<Complex<f32>, IxDyn>  (x,y,z,t,…)
                                       tags:   [Option<DimTag>; 3]         (dim_5..7)
                                       affine: Affine4x4
                                       dwell:  f64
                                       meta:   Metadata
                                     } ──write──> .nii.gz
```

### 3.3 Boundaries

- `mat/` knows HDF5 and nothing about MRS. Testable against the sample files alone.
- `geom` and `header/` expose pure functions. No IO.
- `write/nifti.rs` is `(&MrsDataset) -> Vec<u8>`. No IO, no flavor knowledge.
- `raw2nii-core` returns `Result<_, Raw2NiiError>`, never prints, never exits. This is
  what keeps `raw2nii-py` a shim rather than a reimplementation.

## 4. Readers

### 4.1 SVS

`/fid` is already time-domain; no Fourier transform. Output shape `(1,1,1,samples,rows)`
with `dim_5 = "DIM_DYN"`.

#### Chop: settled, do not apply

The `.mat` backend **must not** apply chop correction, and **must not** consult
`rdb_hdr/data_collect_type`. Established empirically rather than assumed.

MNUtils' rule (`chopped = data_collect_type % 2 == 0`) is correct, but it applies to FIDs
read from the raw ScanArchive, not to the `.mat`. A raw FID array read out of the archive
via MATLAB showed a textbook chop pattern — the sign of the real part at t=0 across the
first 40 acquisitions read `-+-+-+-+…` — on a series with `data_collect_type = 0`, exactly
as the rule predicts.

The `.mat` `/fid` arrays show no such alternation:

| dataset | `data_collect_type` | rule says | sign of real part at t=0, per transient |
|---|---|---|---|
| `MRS_2H` (unlocalised) | 0 | chopped | all 64 positive |
| `MRS_2H_TI_400` (unlocalised) | 0 | chopped | all 32 positive |
| `MRS_2H_slab` | 0 | chopped | all positive bar one low-signal transient |
| `MRS_2H_TE_60` | 1 | not chopped | all 32 positive |

The two unlocalised datasets are the decisive case: `data_collect_type = 0` means the raw
data was chopped, yet `/fid` has no alternation. fidall de-chopped during reconstruction,
as its own log states ("Data chopped", "Apodising"). Applying MNUtils' rule to the `.mat`
would **re-chop** the data and cancel the average.

`data_collect_type` becomes relevant again only for a future ScanArchive backend.

A regression test asserts that no sign alternation across transients exists in the output
for all four SVS datasets.

#### Conjugation: settled, do not apply

`/fid` is written through unchanged. fidall already establishes the convention that
Appendix A requires.

Appendix A follows Levitt: with x real, y imaginary and z time, data from a nucleus of
positive gyromagnetic ratio — 2H and 13C both qualify — must be stored so that a positive
relative frequency appears as a positive, counter-clockwise rotation.

Measured by locating the dominant peak in fidall's own `/spec`, reading its offset from
`/hz`, and correlating `/fid` against `exp(+i·2π·Δf·t)` and `exp(-i·2π·Δf·t)`:

| dataset | peak offset | ccw/cw ratio | result |
|---|---|---|---|
| `MRS_2H_slab` | +53.71 Hz | **5.04** | counter-clockwise |
| `MRS_2H_TI_400` | +4.88 Hz | 1.17 | counter-clockwise |
| `MRS_2H_TE_60` | +2.44 Hz | 1.30 | counter-clockwise |
| `MRS_2H` | +0.00 Hz | 1.00 | on-resonance; carries no information |

`MRS_2H_slab` is the only high-leverage case, its dominant peak being well off-resonance
while the next strongest is 0.35 of it. The other two agree weakly because their peaks sit
near the carrier. `MRS_2H` is on-resonance, so its 1.00 ratio is a tie, not evidence.

A component at a positive frequency on fidall's `/hz` axis therefore rotates as
`exp(+i·2π·Δf·t)`, which is what Appendix A demands. No conjugation.

Recorded assumption: this treats fidall's `/hz` as a true absolute-frequency scale in
Levitt's sense. Were that axis itself sign-flipped, the conclusion would invert. Nothing
in the sample data contradicts it.

MNUtils applies `np.conj` — like its chop correction, that targets raw ScanArchive FIDs,
not the `.mat`.

A test asserts the counter-clockwise result on `MRS_2H_slab` with a ratio above 3, so a
future fidall change that flips the convention fails the suite rather than silently
producing mirrored spectra.

### 4.2 MRSI

`/spec` is frequency-domain, and NIfTI-MRS requires time-domain data, so an inverse
transform is mandatory. Port `xmris.processing.fid.to_fid` exactly:

1. `ifftshift` along the spectral axis;
2. `ifftn` with `norm="ortho"` (rustfft plus an explicit `1/sqrt(N)`);
3. time coordinates `t = arange(N) * dt`, `dt = 1 / (N * df)`, `df` from `/hz`,
   cross-checked against `par.sample_frequency`.

Output shape `(nx, ny, nz, nspec)`, `dim[0] = 4`. Zero-filling is recorded in the JSON
`ProcessingApplied` array.

## 5. Metadata

Specification §2.3.1 requires exactly two JSON keys. Dwell time is not a JSON key; it
lives in `pixdim[4]`.

| key                     | source                                             | note                                                |
| ----------------------- | -------------------------------------------------- | --------------------------------------------------- |
| `SpectrometerFrequency` | `par.f0` / `rdb_hdr.ps_mps_freq`, converted to MHz | array, even when single-element                     |
| `ResonantNucleus`       | `image/specnuc`                                    | code-to-string table, e.g. 2 → `"2H"`, 13 → `"13C"` |

`image/specnuc` is authoritative for the nucleus. It reads 2 and 13 across the samples and
agrees with the prescan text file's "Nucleus = 13". `par.nucleus` reads 50 on the 2H files
and is **not** used.

The `specnuc` code-to-string table is built from evidence and covered by tests. An
unrecognised code is a warning, not an error: the raw code is written and flagged.

All other fields — `EchoTime`, `RepetitionTime`, `TxOffset`, `Manufacturer`,
`ProtocolName`, `PatientID`, scan date and time, and so on — are optional, come from the
same const `FIELD_MAP` in `raw2nii-ge/header/`, and a missing optional field produces a
warning, never a failure. Adding a field is one table row, and a "which required keys are
missing" report falls out of the same table.

## 6. Geometry

Orientation comes from `image/norm_{R,A,S}`; position from `image/tlhc_{R,A,S}`,
`trhc`, `brhc` and `ctr_{R,A,S}`. These are the same fields `spec2nii` uses for GE, and
they are complete in the `.mat`, so **no DICOM-derived NIfTI is required** — unlike
`MNUtils.MRSISeries.create_MRSI_affine`, which rescales an existing NIfTI affine.

### 6.1 Field of view

**FOV is `image/dfov`.** The grid is the stored, zero-filled size. Voxel size is
`dfov / grid`.

MRSI_2H gives 200/16 = 12.5 mm; MRSI_13C gives 300/8 = 37.5 mm. The `/wfn` string
inside MRSI_2H names a waveform file containing `fov140`, and `pixsize_X * dim_X` equals
100 there, but neither is used: `dfov` is authoritative and the zero-filled output grid is
correct. `/wfn` is not parsed.

Note for implementers: `pixsize_X * dim_X` does not equal `dfov` on **any** sample (150 vs
300, 600 vs 300, 100 vs 200). Those `image` fields appear inherited from a localizer.
The mismatch is surfaced once per run as a warning, not per file, and is never used for
geometry.

### 6.2 Localisation and unlocalised dimensions

Specification §2.2 sets `pixdim[1..3]` to 10000 mm for unlocalised dimensions. Position is
still prescribed from the header in every case, since even unlocalised data is placed
somewhere.

`rdb_hdr/roilenx` and `roileny` are 0 on all four SVS samples, so those dimensions are
unlocalised and get 10000 mm.

For the excitation dimension the pulse type decides, read from `rdb_hdr/user14`:

| dataset         | psd         | `rdb.user14` | result                                                     |
| --------------- | ----------- | ------------ | ---------------------------------------------------------- |
| `MRS_2H`        | fidall2     | 91           | unlocalised, despite `slthick=40`                          |
| `MRS_2H_TI_400` | fidall2     | 91           | unlocalised                                                |
| `MRSI_2H`       | fidall2     | 91           | non-selective excitation; z still localised by 3D encoding |
| `MRS_2H_slab`   | fidall2     | 1            | localised, `slthick=80`                                    |
| `MRSI_13C`      | fidall2     | 1            | localised, `slthick=15`                                    |
| `MRS_2H_TE_60`  | **echocsi** | 1            | unknown psd; user CV meaning differs                       |

Rules:

1. `user14 == 91` implies unlocalised excitation: 10000 mm in the excitation dimension,
   plus a debug message naming the pulse and stating that unlocalised data is expected.
2. Otherwise `image/slthick` gives the thickness. `rdb_hdr/roilenz` is not used for
   thickness — it reads 200 against `slthick=40`, and 85 against `slthick=80`.
3. **The rule is not applied to MRSI at all.** `user14` is assumed to be a legacy SVS
   field that MRSI sequences do not use, so an MRSI dataset derives all three spatial
   dimensions from its encoding and `dfov` regardless of `user14`. MRSI_2H is the case in
   point: `user14 == 91`, yet its z dimension is localised by the 3D phase encode and
   receives real geometry.

   This is a **working assumption, not a verified fact** — see §12. It is confined to one
   predicate so that revisiting it is a single-line change, and MRSI geometry logs the
   `user14` value it ignored, so a wrong assumption is visible in the logs rather than
   silent.
4. The rule is scoped to `psdname` matching `fidall*`. An unrecognised psd produces a
   warning and falls back to `slthick`. `MRS_2H_TE_60` is the explicit test case for this
   path.

## 7. CLI

```
raw2nii convert <PATH>... [OPTIONS]

  <PATH>                   file or directory; directories are searched recursively

  -o, --output <DIR>       output directory (default: alongside each input)
      --format <BACKEND>   override probe, e.g. ge-fidall-mat
      --overwrite          replace existing output (default: skip with a warning)
      --dry-run            report intended actions, change nothing
  -j, --jobs <N>           parallel conversions (default: number of CPUs)
      --compress-level <N> gzip level 0-9 (default 4)
  -v, -vv                  warn -> info -> debug
      --json-log           machine-readable events

      --archive [<DIR>]    tar.zst the source folder after successful conversion
      --delete             remove archived sources; requires --archive
```

### 7.1 Output naming

```
exam{ex_no}_series{se_no:02d}_{nucleus}_{type}.nii.gz
```

| dataset       | output                                   |
| ------------- | ---------------------------------------- |
| `MRS_2H`      | `exam20000_series06_2H_svs-unloc.nii.gz` |
| `MRS_2H_slab` | `exam20000_series07_2H_svs-slab.nii.gz`  |
| `MRSI_13C`    | `exam04874_series10_13C_mrsi2d.nii.gz`   |
| `MRSI_2H`     | `exam15732_series05_2H_mrsi3d.nii.gz`    |

Sources: `exam/ex_no`, `series/se_no`, `image/specnuc`, and the detected flavor combined
with localisation state (SVS) or `dim` (MRSI). The series number is zero-padded to two
digits so `series06` sorts before `series25`, matching MNUtils' `Series{id:02d}`.

Scan date and time are **not** in the filename but are written into the JSON header
extension. `rdb_hdr/scan_date` is `MM/DD/YY` with a 1900 year offset (`125` means 2025)
and `rdb_hdr/scan_time` is `HH:MM`; both cross-check against the ScanArchive filename
timestamps.

Exam plus series is unique, so collisions are governed by `--overwrite`.

### 7.2 Archiving

The archive unit is **the parent directory of the converted `.mat`**. The sample data is
flat (`tests/datasets/MRS_2H/`) rather than the `Exam*/Series<N>/` layout MNUtils assumes,
so keying off a `Series<N>` directory name would fail on real inputs.

- The archive is written outside the directory being archived — by default a sibling
  `<folder>.tar.zst` — never inside it.
- Generated `.nii.gz` files are excluded; they are output, not source.
- Several `.mat` files in one directory produce **one** archive, written once, after all
  of them convert.
- zstd level 3, parallel across directories.

### 7.3 Deletion

Deletion is the only irreversible action and is gated accordingly.

1. `--delete` without `--archive` is a usage error, not a warning.
2. Deletion runs only after every conversion in that directory succeeded, the archive was
   written, and the archive was reopened and its member list verified against what was put
   in. A verification failure keeps everything and exits non-zero.
3. Only files that are verified members of the archive are deleted. Anything that appeared
   during the run is left alone.
4. `--dry-run` prints the exact deletion list.

### 7.4 Concurrency

rayon across discovered inputs, one task per `.mat`. Archive and delete phases join
per-directory, so a directory is never archived while a conversion inside it is still
running. Failure of one file never aborts the run: errors accumulate and are reported at
the end. The exit code is 0 on full success and 1 if any conversion failed.

## 8. Errors

One `Raw2NiiError` enum in core, via `thiserror`. The distinction that matters is
refusing versus degrading.

Hard errors, no output written:

- No backend claims the file, or `--format` names a backend that does not claim it
- `/h` missing or unreadable, or required data (`/fid`, `/spec`) absent
- A required §2.3.1 key underivable (`specnuc` or `f0` absent)
- Dimension contradiction, for example `zf` disagreeing with the actual dataspace
- Archive verification failure

Warnings, output still written and the warning recorded in the JSON:

- Optional metadata absent
- Unknown psd for the `user14` localisation rule; falls back to `slthick`
- Unknown `specnuc` code; the raw code is written and flagged
- `pixsize * dim != dfov`, reported once per run

Output is written to a temporary file and atomically renamed, so an aborted run never
leaves a partial `.nii.gz`.

## 9. Testing

1. **`mat/` unit tests** — dimension reversal, compound-complex decoding, `#refs#`
   dereferencing, trailing-singleton padding. Asserted against the real sample files.
2. **Pure-function tests** — `geom` against hand-computed affines from the sample
   `tlhc`/`norm` values; `fft` round-trip `to_spectrum ∘ to_fid == identity` to 1e-6, plus
   a fixture generated from xmris so the Rust inverse transform is proven equivalent to
   the Python it is ported from.
3. **Golden-file tests** — convert all six `.mat` datasets and assert byte-identical
   `.nii.gz` against committed goldens, regenerated with `cargo test -- --bless`.
4. **Conformance tests** — assert `intent_name == "mrs_v0_11"`, a complex datatype of
   64 bits or more, ecode 44, the header extension padded to a multiple of 16 bytes,
   `SpectrometerFrequency` and `ResonantNucleus` present as arrays, `pixdim[4]` set, and
   qform/sform populated. Run against every golden.

`BS_prescan_13C` has no `.mat` and serves as the "backend correctly declines" case.

Chop is settled (§4.1) and is covered by a regression test asserting that no sign
alternation across transients survives into the output, for all four SVS datasets.

Conjugation is settled (§4.1) and is covered by a test asserting a counter-clockwise
rotation ratio above 3 on `MRS_2H_slab`.

One open empirical question lands as a test that locks the answer: the `specnuc`
code-to-string table beyond the observed values.

Note on test data: `*_raw_fids.h5` files are **not** scanner output — they were generated
manually through MATLAB — and are deliberately absent. No test may depend on them.

## 10. Python wrapper

```python
import raw2nii

results = raw2nii.convert("Exam20000/", output="derivatives/", jobs=8)
# -> [Result(input=..., output=..., warnings=[...]), ...]

ds = raw2nii.read("ScanArchive_..._103147349.mat")
ds.data        # np.ndarray, complex64, shape (1,1,1,2048,64)
ds.affine      # np.ndarray (4,4)
ds.dwell_time  # float, seconds
ds.dim_tags    # ('DIM_DYN', None, None)
ds.metadata    # dict, the JSON header extension verbatim
ds.to_nifti("out.nii.gz")
```

`ds.data` is zero-copy: `ndarray` becomes a `numpy::PyArray` over the buffer pyo3 already
owns. No serialisation, no temporary file. This is the reason for the workspace layout
rather than a subprocess wrapper — `MNUtils.MRSSeries` can drop its MATLAB-engine
dependency and read FIDs directly.

Heavy calls are wrapped in `py.allow_threads`, so `convert()` releases the GIL and
`jobs=N` yields real parallelism from Python.

Errors map to an exception hierarchy — `Raw2NiiError` base, with
`UnsupportedFormatError`, `MissingMetadataError`, `GeometryError` and
`ArchiveVerificationError` — so callers catch precisely rather than matching strings.
Warnings go through Python's `warnings` module, not stderr.

Build: `maturin`, `abi3-py39` (one wheel per platform rather than per Python version),
`cibuildwheel` for Linux, macOS and Windows. `hdf5-metno` with the `static` feature
vendors libhdf5 into the extension module, so wheels install without a system HDF5 —
matching the single-static-binary property of the CLI.

## 11. Performance

Rust is chosen for a static binary with no MATLAB or Python startup and for a clean
in-process Python binding. It is not chosen because the arithmetic is expensive.

Work distribution on the largest sample (MRSI_2H, 23 MB `.mat`, 16^3 x 700 complex,
183 MB in memory):

| stage                      | cost                     | note                                       |
| -------------------------- | ------------------------ | ------------------------------------------ |
| HDF5 read and decompress   | IO-bound                 | dominant on a cold cache                   |
| compound to `Complex<f32>` | one pass                 | interleaved copy, unavoidable              |
| inverse FFT                | 4096 FIDs x 700 points   | rustfft, planner reused, rayon over voxels |
| gzip output                | dominant on a warm cache | see below                                  |

For inputs of a few hundred megabytes the bottleneck is gzip, not the mathematics. Hence
`flate2` with the `zlib-ng` backend, default level 4 (level 9 costs roughly three times
the time for a few percent of size), `--compress-level 0..9`, and uncompressed `.nii` when
the output path has no `.gz` suffix.

Peak memory is about twice the array size during the inverse transform. If a future
dataset outgrows memory, the fix is chunking over voxels, which the current shape already
permits. That is not built now.

Performance targets are measure-then-assert: a criterion benchmark over the sample set
with a regression guard, rather than numbers promised in advance.

## 12. Decisions and open questions

Decided:

- Input is the fidall `.mat`; ScanArchive and P-file parsing are out of scope.
- SVS, MRSI and wash-in (a wash-in series is an MRS series) are in scope.
- Path argument accepts a file or a directory.
- Archive unit is the whole source folder as `tar.zst`.
- Unlocalised dimensions use the specification default of 10000 mm; position is always
  prescribed from the header.
- Processing follows MNUtils where MNUtils operates on comparable data, with the
  spectrum-to-FID conversion ported from `xmris.processing`. MNUtils' chop correction is
  the documented exception: it targets raw ScanArchive FIDs, and the `.mat` is already
  de-chopped (§4.1).
- Workspace layout is core plus CLI plus pyo3.
- FOV is `dfov`; the output grid is the zero-filled grid.
- Filenames are `exam{n}_series{nn}_{nucleus}_{type}`, without a date.

- Chop correction is **not** applied to the `.mat` `/fid`, and `data_collect_type` is not
  consulted by this backend. Established empirically (§4.1).

- Conjugation is **not** applied. fidall already satisfies Appendix A. Established
  empirically (§4.1).

Assumed, pending confirmation, and deliberately isolated so each is cheap to revisit:

- `rdb_hdr/user14` is a legacy SVS field that MRSI does not use, so the unlocalised-pulse
  rule is not applied to MRSI (§6.2). Not fully verified. Confined to one predicate, and
  the ignored value is logged.
- fidall's `/hz` is a true absolute-frequency scale in Levitt's sense, which the
  conjugation conclusion rests on (§4.1).

Open, each with an assigned implementation task:

- The `image/specnuc` code-to-string table beyond the observed 2 and 13.
- `user14` semantics under psds other than `fidall*`; currently a warning plus fallback.
