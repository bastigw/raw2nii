//! The extension seam.
//!
//! Abstraction lives at the outer boundary only: a backend takes a path and
//! produces `MrsDataset` values. There is deliberately no shared intermediate
//! representation between vendors.

use std::path::Path;

use crate::dataset::MrsDataset;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    No,
    Maybe,
    Yes,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Cheap: extension, magic bytes, a few key probes. Never a full parse.
    fn probe(&self, path: &Path) -> Confidence;
    fn convert(&self, path: &Path) -> Result<Vec<MrsDataset>>;
}

#[derive(Default)]
pub struct Registry {
    backends: Vec<Box<dyn Backend>>,
}

impl Registry {
    pub fn with_backend(mut self, b: Box<dyn Backend>) -> Self {
        self.backends.push(b);
        self
    }

    pub fn select(&self, path: &Path) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .map(|b| (b.probe(path), b))
            .filter(|(c, _)| *c > Confidence::No)
            .max_by_key(|(c, _)| *c)
            .map(|(_, b)| b.as_ref())
    }

    pub fn by_name(&self, name: &str) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
}
