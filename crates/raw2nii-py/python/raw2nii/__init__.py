"""Vendor MRS raw data to NIfTI-MRS conversion."""

from .raw2nii import (
    BackendError,
    Dataset,
    DimensionMismatchError,
    GeometryError,
    IoError,
    MissingDataError,
    MissingMetadataError,
    Raw2NiiError,
    UnsupportedFormatError,
    _build_and_verify_archive,
    _convert_many,
    convert,
    read,
)

__all__ = [
    "read",
    "convert",
    "Dataset",
    "Raw2NiiError",
    "UnsupportedFormatError",
    "MissingDataError",
    "MissingMetadataError",
    "GeometryError",
    "DimensionMismatchError",
    "BackendError",
    "IoError",
]
