"""``raw2nii`` console script: convert vendor raw files to NIfTI-MRS.

Thin argparse wrapper around :func:`raw2nii.convert`, installed as a
console-script entry point so the package works as a ``uv tool``
(``uv tool install raw2nii`` then ``raw2nii scan.mat``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import Raw2NiiError, convert


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    inputs: list[Path] = []
    for raw_input in args.inputs:
        if not raw_input.exists():
            print(f"raw2nii: {raw_input}: no such file or directory", file=sys.stderr)
            return 1
        inputs.extend(_discover(raw_input, args.recursive))

    if not inputs:
        print("raw2nii: no .mat files found", file=sys.stderr)
        return 1

    exit_code = 0
    for path in inputs:
        try:
            written = convert(
                str(path),
                output_dir=str(args.output_dir) if args.output_dir else None,
                format=args.format,
                compress_level=args.compress_level,
                overwrite=args.overwrite,
            )
        except Raw2NiiError as exc:
            print(f"raw2nii: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        for out in written:
            print(out)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
