//! Building and verifying a `.tar.zst` snapshot of a source directory,
//! ahead of `--delete` removing the originals.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Files under `dir`, relative to `dir`, skipping anything under `exclude`
/// or equal to `skip_path`.
fn collect_files(dir: &Path, exclude: &Path, skip_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let p = entry?.path();
            if p == skip_path || p.starts_with(exclude) {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn build_archive(
    source_dir: &Path,
    archive_path: &Path,
    exclude: &Path,
) -> std::io::Result<()> {
    let files = collect_files(source_dir, exclude, archive_path)?;

    let file = File::create(archive_path)?;
    let encoder = zstd::stream::write::Encoder::new(file, 0)?.auto_finish();
    let mut builder = tar::Builder::new(encoder);
    for f in &files {
        let rel = f
            .strip_prefix(source_dir)
            .expect("collect_files yields children of source_dir");
        builder.append_path_with_name(f, rel)?;
    }
    builder.into_inner()?.flush()
}

pub fn verify_archive(
    archive_path: &Path,
    source_dir: &Path,
    exclude: &Path,
) -> std::io::Result<bool> {
    let expected = collect_files(source_dir, exclude, archive_path)?;

    let file = File::open(archive_path)?;
    let decoder = zstd::stream::read::Decoder::new(file)?;
    let mut archive = tar::Archive::new(decoder);

    let mut sizes = std::collections::HashMap::new();
    for entry in archive.entries()? {
        let entry = entry?;
        let path = entry.path()?.into_owned();
        sizes.insert(path, entry.header().size()?);
    }

    for f in &expected {
        let rel = f
            .strip_prefix(source_dir)
            .expect("collect_files yields children of source_dir");
        let on_disk = std::fs::metadata(f)?.len();
        match sizes.get(rel) {
            Some(&archived) if archived == on_disk => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, rel: &str, contents: &[u8]) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, contents).unwrap();
    }

    #[test]
    fn round_trips_a_small_tree() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");
        write(src.path(), "sub/b.mat", b"world!!");

        let out = tempfile::tempdir().unwrap();
        let archive_path = out.path().join("snapshot.tar.zst");
        let exclude = out.path().join("nonexistent-output-dir");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        assert!(verify_archive(&archive_path, src.path(), &exclude).unwrap());
    }

    #[test]
    fn excludes_the_output_directory() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");
        write(src.path(), "out/a.nii.gz", b"converted bytes");

        let archive_path = src.path().join("out").join("snapshot.tar.zst");
        let exclude = src.path().join("out");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        assert!(verify_archive(&archive_path, src.path(), &exclude).unwrap());

        // The excluded file must genuinely be absent from the archive.
        let file = File::open(&archive_path).unwrap();
        let decoder = zstd::stream::read::Decoder::new(file).unwrap();
        let mut archive = tar::Archive::new(decoder);
        let names: Vec<String> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(!names.iter().any(|n| n.contains("a.nii.gz")));
    }

    #[test]
    fn verify_fails_after_a_source_file_is_modified() {
        let src = tempfile::tempdir().unwrap();
        write(src.path(), "a.mat", b"hello");

        let out = tempfile::tempdir().unwrap();
        let archive_path = out.path().join("snapshot.tar.zst");
        let exclude = out.path().join("no-such-dir");

        build_archive(src.path(), &archive_path, &exclude).unwrap();
        write(src.path(), "a.mat", b"hello, but now longer");

        assert!(!verify_archive(&archive_path, src.path(), &exclude).unwrap());
    }
}
