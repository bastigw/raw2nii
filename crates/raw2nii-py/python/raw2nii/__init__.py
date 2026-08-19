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
