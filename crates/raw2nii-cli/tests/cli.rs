use std::path::PathBuf;
use std::process::Command;

fn sample(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/datasets")
        .join(name);
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("mat")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("ScanArchive"))
        })
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_raw2nii"))
}

#[test]
fn converts_a_single_file_to_the_named_output() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let out = std::env::temp_dir().join("raw2nii-cli-test-svs");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    let status = Command::new(bin())
        .args(["convert"])
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());

    let expected = out.join("exam20000_series06_2H_svs-unloc.nii.gz");
    assert!(expected.exists(), "missing {expected:?}");
    assert!(std::fs::metadata(&expected).unwrap().len() > 1000);
}

#[test]
fn refuses_to_overwrite_without_the_flag() {
    let Some(input) = sample("MRS_2H") else {
        eprintln!("SKIP: tests/datasets absent");
        return;
    };
    let out = std::env::temp_dir().join("raw2nii-cli-test-overwrite");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    for _ in 0..2 {
        let status = Command::new(bin())
            .args(["convert"])
            .arg(&input)
            .arg("-o")
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "a skip is not a failure");
    }
}

#[test]
fn unreadable_input_exits_non_zero() {
    let bogus = std::env::temp_dir().join("raw2nii-not-a-real-file.mat");
    let _ = std::fs::remove_file(&bogus);
    let status = Command::new(bin())
        .args(["convert"])
        .arg(&bogus)
        .status()
        .unwrap();
    assert!(!status.success());
}
