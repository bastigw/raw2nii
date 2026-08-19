//! Spec conformance, asserted against every shipped dataset.

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

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn i16_at(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}
fn i64_at(b: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn f64_at(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

#[test]
fn every_dataset_produces_a_conformant_file() {
    let mut checked = 0;
    for name in DATASETS {
        let Some(path) = sample_mat(name) else {
            continue;
        };
        let datasets = GeMatBackend
            .convert(&path)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        for ds in &datasets {
            let b = serialise(ds).unwrap_or_else(|e| panic!("{name}: {e}"));

            assert_eq!(i32_at(&b, 0), 540, "{name}: sizeof_hdr");
            assert_eq!(&b[4..12], b"n+2\0\r\n\x1a\n", "{name}: magic");

            let datatype = i16_at(&b, 12);
            assert!(
                matches!(datatype, 32 | 1792 | 2048),
                "{name}: datatype {datatype} is not a complex type"
            );

            let ndim = i64_at(&b, 16);
            assert!((4..=7).contains(&ndim), "{name}: dim[0] = {ndim}");

            let dwell = f64_at(&b, 104 + 4 * 8);
            assert!(dwell > 0.0, "{name}: pixdim[4] = {dwell}");

            assert_eq!(i32_at(&b, 500), 10, "{name}: xyzt_units");
            assert_ne!(i32_at(&b, 344), 0, "{name}: qform_code");
            assert_ne!(i32_at(&b, 348), 0, "{name}: sform_code");

            let intent = &b[508..524];
            let end = intent.iter().position(|&c| c == 0).unwrap_or(intent.len());
            assert_eq!(
                std::str::from_utf8(&intent[..end]).unwrap(),
                "mrs_v0_11",
                "{name}: intent_name"
            );

            assert_eq!(b[540], 1, "{name}: extension flag");
            let esize = i32_at(&b, 544);
            assert!(esize > 0 && esize % 16 == 0, "{name}: esize {esize}");
            assert_eq!(i32_at(&b, 548), 44, "{name}: ecode");

            let json_bytes = &b[552..(544 + esize as usize)];
            let text = std::str::from_utf8(json_bytes)
                .unwrap()
                .trim_end_matches('\0');
            let v: serde_json::Value = serde_json::from_str(text)
                .unwrap_or_else(|e| panic!("{name}: extension is not valid JSON: {e}"));
            assert!(
                v["SpectrometerFrequency"].is_array(),
                "{name}: SpectrometerFrequency must be an array"
            );
            assert!(
                v["ResonantNucleus"].is_array(),
                "{name}: ResonantNucleus must be an array"
            );

            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("SKIP: tests/datasets absent");
    }
}

#[test]
fn the_prescan_directory_is_declined() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/datasets/BS_prescan_13C");
    if !dir.exists() {
        eprintln!("SKIP: tests/datasets absent");
        return;
    }
    let mats: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mat"))
        .collect();
    assert!(
        mats.is_empty(),
        "BS_prescan_13C is the negative case and must ship no .mat"
    );
}
