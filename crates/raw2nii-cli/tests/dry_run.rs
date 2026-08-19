//! End-to-end CLI behaviour for --dry-run, --format, and --archive/--delete.

use std::path::PathBuf;
use std::process::Command;

fn sample(name: &str) -> Option<PathBuf> {
    raw2nii_ge::samples::sample_mat(name)
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_raw2nii"))
}

#[test]
fn dry_run_reports_but_does_not_write() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    let output = Command::new(bin())
        .args(["convert", "--dry-run", "-o"])
        .arg(dir.path())
        .arg(&input)
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "dry run must write nothing"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("exam20000_series06_2H_svs-unloc.nii.gz"),
        "{stdout}"
    );
}

#[test]
fn format_nii_writes_uncompressed() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    let status = Command::new(bin())
        .args(["convert", "--format", "nii", "-o"])
        .arg(dir.path())
        .arg(&input)
        .status()
        .unwrap();

    assert!(status.success());
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["exam20000_series06_2H_svs-unloc.nii".to_string()]
    );
}

#[test]
fn json_log_emits_parseable_lines() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    let output = Command::new(bin())
        .args(["convert", "--json-log", "-o"])
        .arg(dir.path())
        .arg(&input)
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2, "one result line plus one summary line: {stdout}");
    for line in &lines {
        let _: serde_json::Value = serde_json::from_str(line).expect("each line must be JSON");
    }
    let summary: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(summary["written"], 1);
}

#[test]
fn delete_without_archive_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(bin())
        .args(["convert", "--delete"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--delete") && stderr.contains("--archive"),
        "{stderr}"
    );
}

#[test]
fn archive_then_delete_removes_originals_but_keeps_outputs() {
    let Some(sample_dir) = sample("MRS_2H").and_then(|p| p.parent().map(|d| d.to_path_buf()))
    else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };

    // Work on a throwaway copy: --delete is destructive.
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("MRS_2H");
    copy_dir(&sample_dir, &src);

    let out_dir = work.path().join("out");
    let archive_path = work.path().join("MRS_2H.tar.zst");

    let status = Command::new(bin())
        .args(["convert", "-o"])
        .arg(&out_dir)
        .arg("--archive")
        .arg(&archive_path)
        .arg("--delete")
        .arg(&src)
        .status()
        .unwrap();

    assert!(status.success());
    assert!(archive_path.exists(), "archive must be written");
    assert!(
        std::fs::read_dir(&out_dir).unwrap().next().is_some(),
        "converted output must survive --delete"
    );
    let mat_files: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("mat"))
        .collect();
    assert!(
        mat_files.is_empty(),
        "--delete must remove the original .mat files"
    );
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), &dest).unwrap();
        }
    }
}
