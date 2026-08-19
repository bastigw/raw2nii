//! Locating the sample datasets. They are gitignored and may be absent,
//! so every consumer must handle `None` by skipping, never by failing.

use std::path::PathBuf;

/// Repository root, derived from this crate's manifest directory.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Path to the `ScanArchive*.mat` inside `tests/datasets/<name>/`.
pub fn sample_mat(name: &str) -> Option<PathBuf> {
    let dir = repo_root().join("tests").join("datasets").join(name);
    let entries = std::fs::read_dir(dir).ok()?;
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_svs_sample_or_skips() {
        match sample_mat("MRS_2H") {
            Some(p) => assert!(p.exists(), "returned path must exist: {p:?}"),
            None => eprintln!("SKIP: tests/datasets absent"),
        }
    }

    #[test]
    fn unknown_dataset_is_none() {
        assert!(sample_mat("NoSuchDataset").is_none());
    }
}
