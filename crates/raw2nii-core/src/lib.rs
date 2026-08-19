//! Vendor-neutral core: the dataset contract, the backend seam, and the
//! NIfTI-MRS writer. This crate never prints and never exits.

pub mod dataset;
pub mod error;
pub mod fft;

pub use dataset::{identity_affine, DimTag, Metadata, MrsDataset};
pub use error::{Raw2NiiError, Result};
