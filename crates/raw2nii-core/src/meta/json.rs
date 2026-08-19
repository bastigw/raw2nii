//! The NIfTI-MRS JSON header extension (spec §2.3).
//!
//! Layout: `esize` (i32 LE), `ecode` = 44 (i32 LE), then the UTF-8 JSON,
//! zero-padded so the whole extension is a positive multiple of 16 bytes.

use serde_json::{json, Map, Value};

use crate::dataset::MrsDataset;

/// `NIfTI_ECODE_MRS`.
const ECODE_MRS: i32 = 44;

pub fn build_extension(ds: &MrsDataset) -> Vec<u8> {
    let mut obj: Map<String, Value> = Map::new();

    obj.insert(
        "SpectrometerFrequency".to_string(),
        json!(ds.meta.spectrometer_frequency_mhz),
    );
    obj.insert(
        "ResonantNucleus".to_string(),
        json!(ds.meta.resonant_nucleus),
    );

    for (i, tag) in ds.tags.iter().enumerate() {
        if let Some(t) = tag {
            obj.insert(format!("dim_{}", i + 5), json!(t.as_str()));
        }
    }

    for (k, v) in &ds.meta.extra {
        obj.insert(k.clone(), v.clone());
    }

    if !ds.meta.warnings.is_empty() {
        obj.insert(
            "ConversionWarnings".to_string(),
            json!(ds.meta.warnings),
        );
    }

    let text = serde_json::to_string(&Value::Object(obj)).expect("metadata is serialisable");

    let mut ext = Vec::with_capacity(text.len() + 24);
    ext.extend_from_slice(&0i32.to_le_bytes()); // esize placeholder
    ext.extend_from_slice(&ECODE_MRS.to_le_bytes());
    ext.extend_from_slice(text.as_bytes());
    while ext.len() % 16 != 0 {
        ext.push(0);
    }

    let esize = ext.len() as i32;
    ext[0..4].copy_from_slice(&esize.to_le_bytes());
    ext
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{identity_affine, DimTag, Metadata, MrsDataset};
    use ndarray::{ArrayD, IxDyn};
    use num_complex::Complex;

    fn dataset(shape: &[usize], tags: [Option<DimTag>; 3]) -> MrsDataset {
        MrsDataset {
            data: ArrayD::from_elem(IxDyn(shape), Complex::new(0.0, 0.0)),
            tags,
            affine: identity_affine(),
            dwell_time_s: 2e-4,
            meta: Metadata {
                spectrometer_frequency_mhz: vec![19.5934],
                resonant_nucleus: vec!["2H".to_string()],
                extra: serde_json::Map::new(),
                warnings: vec![],
            },
        }
    }

    fn json_of(ext: &[u8]) -> serde_json::Value {
        let text = std::str::from_utf8(&ext[8..]).unwrap();
        serde_json::from_str(text.trim_end_matches('\0')).unwrap()
    }

    #[test]
    fn extension_size_is_a_multiple_of_sixteen() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        assert_eq!(ext.len() % 16, 0, "extension length {}", ext.len());
    }

    #[test]
    fn esize_matches_actual_length_and_ecode_is_44() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let esize = i32::from_le_bytes(ext[0..4].try_into().unwrap());
        let ecode = i32::from_le_bytes(ext[4..8].try_into().unwrap());
        assert_eq!(esize as usize, ext.len());
        assert_eq!(ecode, 44);
    }

    #[test]
    fn required_keys_are_arrays_even_when_single() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let v = json_of(&ext);
        assert!(v["SpectrometerFrequency"].is_array());
        assert!(v["ResonantNucleus"].is_array());
        assert_eq!(v["ResonantNucleus"][0], "2H");
        assert_eq!(v["SpectrometerFrequency"][0], 19.5934);
    }

    #[test]
    fn dimension_tags_are_written() {
        let ext = build_extension(&dataset(&[1, 1, 1, 2048, 64], [Some(DimTag::Dyn), None, None]));
        let v = json_of(&ext);
        assert_eq!(v["dim_5"], "DIM_DYN");
        assert!(v.get("dim_6").is_none());
    }

    #[test]
    fn optional_and_warning_fields_pass_through() {
        let mut ds = dataset(&[16, 16, 16, 700], [None, None, None]);
        ds.meta
            .extra
            .insert("EchoTime".to_string(), serde_json::json!(0.035));
        ds.meta.warnings.push("stale header field".to_string());
        let v = json_of(&build_extension(&ds));
        assert_eq!(v["EchoTime"], 0.035);
        assert_eq!(v["ConversionWarnings"][0], "stale header field");
    }
}
