//! Vendor-neutral core: the dataset contract, the backend seam, and the
//! NIfTI-MRS writer. This crate never prints and never exits.

pub mod backend;
pub mod dataset;
pub mod error;
pub mod fft;
pub mod meta;
pub mod write;

pub use dataset::{identity_affine, DimTag, Metadata, MrsDataset};
pub use error::{Raw2NiiError, Result};
