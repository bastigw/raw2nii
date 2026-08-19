//! Golden-file tests. Set `RAW2NII_BLESS=1` to regenerate.
//!
//! Goldens are the serialised, uncompressed NIfTI bytes hashed to keep the
//! committed artefacts small while still catching any byte-level change.

use std::path::PathBuf;

use raw2nii_core::backend::Backend;
use raw2nii_core::write::serialise;
use raw2nii_ge::samples::sample_mat;
use raw2nii_ge::GeMatBackend;

const DATASETS: &[&str] = &[
    "MRS_2H",
    "MRS_2H_slab",
    "MRS_2H_TE_60",
    "MRS_2H_TI_400",
    "MRSI_13C",
    "MRSI_2H",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// FNV-1a, so the test has no hashing dependency.
fn digest(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[test]
fn output_matches_the_goldens() {
    let bless = std::env::var("RAW2NII_BLESS").is_ok();
    let dir = repo_root().join("tests/goldens");
    std::fs::create_dir_all(&dir).unwrap();

    let mut checked = 0;
    for name in DATASETS {
        let Some(path) = sample_mat(name) else {
            continue;
        };
        let datasets = GeMatBackend.convert(&path).unwrap();
        let bytes = serialise(&datasets[0]).unwrap();
        let actual = format!("{}  {} bytes", digest(&bytes), bytes.len());

        let golden = dir.join(format!("{name}.txt"));
        if bless {
            std::fs::write(&golden, &actual).unwrap();
            continue;
        }
        let expected = match std::fs::read_to_string(&golden) {
            Ok(s) => s,
            Err(_) => panic!(
                "no golden for {name}. Run: RAW2NII_BLESS=1 cargo test -p raw2nii-ge --test golden"
            ),
        };
        assert_eq!(actual, expected.trim(), "{name}: output changed");
        checked += 1;
    }
    if checked == 0 && !bless {
        eprintln!("SKIP: tests/datasets absent");
    }
}
