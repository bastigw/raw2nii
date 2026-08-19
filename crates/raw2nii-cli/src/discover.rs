//! Filesystem discovery: which files are candidate conversion inputs.

use std::path::{Path, PathBuf};

/// All `.mat` files under `path`, sorted. If `path` is itself a file, it is
/// returned unchecked — the backend registry decides whether it can read it.
pub fn discover(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("mat") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_argument_is_returned_unchecked() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("archive.mat");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(discover(&f).unwrap(), vec![f]);
    }

    #[test]
    fn finds_mat_files_recursively_and_sorts_them() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("b.mat"), b"").unwrap();
        std::fs::write(dir.path().join("a.txt"), b"").unwrap();
        std::fs::write(dir.path().join("sub/a.mat"), b"").unwrap();

        let found = discover(dir.path()).unwrap();
        assert_eq!(
            found,
            vec![dir.path().join("b.mat"), dir.path().join("sub/a.mat")]
        );
    }

    #[test]
    fn an_empty_directory_yields_no_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(discover(dir.path()).unwrap().is_empty());
    }
}
