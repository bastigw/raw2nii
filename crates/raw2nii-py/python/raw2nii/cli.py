"""``raw2nii`` console script: convert vendor raw files to NIfTI-MRS.

argparse wrapper around the Rust conversion, discovery, and archive logic
shared with the ``raw2nii`` binary (via ``raw2nii-convert``), installed as a
console-script entry point so the package works as a ``uv tool``
(``uv tool install raw2nii`` then ``raw2nii scan.mat``). Mirrors the native
CLI's option set (``-j``/``--jobs``, ``--dry-run``, ``--json-log``,
``--archive``/``--delete``, ``-v``/``--verbose``) plus ``-r``/``--recursive``,
``--no-color``, and multi-input support, which the native CLI doesn't have.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import _build_and_verify_archive, _convert_many, _output


def _discover(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*.mat" if recursive else "*.mat"
    return sorted(root.glob(pattern))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raw2nii",
        description="Convert vendor MRS raw data to NIfTI-MRS.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="input .mat file(s) or directories to scan for .mat files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="directory to write converted files to (default: alongside each input)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="when an input is a directory, scan it recursively for .mat files",
    )
    parser.add_argument(
        "--format",
        choices=("nii-gz", "nii"),
        default="nii-gz",
        help="output format (default: nii-gz)",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=6,
        help="gzip compression level, 0-9 (default: 6, nii-gz only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output files",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="worker threads for parallel conversion (default: available parallelism)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be converted without writing any files",
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="emit machine-readable JSON Lines instead of human-readable output",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "write a verified tar.zst snapshot of the input directory "
            "(output excluded) here; requires a single directory input"
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="after a verified --archive, delete the original input files (requires --archive)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity: -v for info, -vv for debug",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable colored output (also honors the NO_COLOR env var)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    use_color = _output.color_enabled(sys.stdout, args.no_color) and not args.json_log

    def err(text: str, color: str) -> None:
        print(_output.colorize(text, color, use_color), file=sys.stderr)

    if args.delete and not args.archive:
        err("raw2nii: --delete requires --archive", "red")
        return 1
    if args.archive and (len(args.inputs) != 1 or not args.inputs[0].is_dir()):
        err("raw2nii: --archive requires a single directory input", "red")
        return 1

    inputs: list[Path] = []
    for raw_input in args.inputs:
        if not raw_input.exists():
            err(f"raw2nii: {raw_input}: no such file or directory", "red")
            return 1
        inputs.extend(_discover(raw_input, args.recursive))

    if not inputs:
        err("raw2nii: no .mat files found", "red")
        return 1

    if not args.json_log:
        jobs_note = f" with {args.jobs} worker(s)" if args.jobs else ""
        err(f"raw2nii: converting {len(inputs)} file(s){jobs_note}", "cyan")
        if args.verbose:
            for i, p in enumerate(inputs, start=1):
                print(_output.progress_line(i, len(inputs), str(p), use_color), file=sys.stderr)

    start = time.perf_counter()
    outcomes = _convert_many(
        inputs=[str(p) for p in inputs],
        output_dir=str(args.output_dir) if args.output_dir else None,
        overwrite=args.overwrite,
        compress_level=args.compress_level,
        jobs=args.jobs,
        dry_run=args.dry_run,
        format=args.format,
    )
    elapsed = time.perf_counter() - start

    failures = 0
    if args.json_log:
        written = skipped = failed = 0
        for outcome in outcomes:
            print(json.dumps(outcome))
            if outcome["status"] == "written":
                written += 1
            elif outcome["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
        failures = failed
        print(json.dumps({"written": written, "skipped": skipped, "failed": failed}))
    else:
        written = skipped = 0
        for outcome in outcomes:
            status = outcome["status"]
            if status == "written":
                written += 1
                print(_output.colorize(f"{outcome['input']} -> {outcome['output']}", "green", use_color))
            elif status == "skipped":
                skipped += 1
                err(f"raw2nii: {outcome['input']}: {outcome['reason']}", "yellow")
            else:
                err(f"raw2nii: {outcome['input']}: {outcome['error']}", "red")
                failures += 1
        summary_color = "red" if failures else "green"
        err(
            f"raw2nii: {written} written, {skipped} skipped, {failures} failed ({elapsed:.2f}s)",
            summary_color,
        )

    if args.archive:
        if failures > 0:
            err(f"raw2nii: not archiving: {failures} file(s) failed to convert", "red")
        else:
            source_dir = args.inputs[0]
            exclude = args.output_dir or (source_dir / ".raw2nii-no-exclude")
            try:
                verified = _build_and_verify_archive(str(source_dir), str(args.archive), str(exclude))
            except Exception as exc:  # noqa: BLE001 - surfaced verbatim like the native CLI
                err(f"raw2nii: archiving {source_dir}: {exc}", "red")
                failures += 1
                verified = False

            if verified:
                print(_output.colorize(f"archived {source_dir} -> {args.archive}", "green", use_color))
                if args.delete:
                    for outcome in outcomes:
                        if outcome["status"] == "written":
                            try:
                                Path(outcome["input"]).unlink()
                            except OSError as exc:
                                err(f"raw2nii: could not delete {outcome['input']}: {exc}", "red")
                                failures += 1
            elif not failures:
                err("raw2nii: archive verification failed, originals were not deleted", "red")
                failures += 1

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
