#[derive(Debug, thiserror::Error)]
pub enum Raw2NiiError {
    #[error("no backend can read {0}")]
    UnsupportedFormat(String),
    #[error("required data missing: {0}")]
    MissingData(String),
    #[error("required metadata missing: {0}")]
    MissingMetadata(String),
    #[error("geometry error: {0}")]
    Geometry(String),
    #[error("dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Raw2NiiError>;
