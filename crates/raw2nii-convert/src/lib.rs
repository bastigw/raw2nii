//! Shared conversion, discovery, archiving, and reporting logic used by both
//! the `raw2nii` CLI binary and the `raw2nii` Python bindings, so the two
//! front ends can't drift apart on naming, output-format, or archive rules.

pub mod archive;
pub mod convert;
pub mod discover;
pub mod report;

pub use convert::{convert_one, Outcome, OutputFormat};
pub use discover::discover;
