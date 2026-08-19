//! Typed access to the GE `/h` structs.
//!
//! These structs are identical across GE containers — ScanArchive, P-file and
//! the fidall `.mat` all carry them — so this module is the reuse point for
//! any future GE backend.

use crate::mat::{MatError, MatFile};

/// Map a GE `image/specnuc` code to a NIfTI-MRS nucleus string.
///
/// Returns the name plus an optional warning for unrecognised codes. An
/// unknown code is never fatal: the raw code is used and flagged.
pub fn nucleus_from_code(code: i64) -> (String, Option<String>) {
    let name = match code {
        1 => "1H",
        2 => "2H",
        3 => "3HE",
        7 => "7LI",
        13 => "13C",
        19 => "19F",
        23 => "23NA",
        31 => "31P",
        129 => "129XE",
        _ => {
            return (
                code.to_string(),
                Some(format!(
                    "unrecognised image/specnuc code {code}; \
                     ResonantNucleus written as the raw code"
                )),
            )
        }
    };
    (name.to_string(), None)
}

/// GE stores the scan date as `MM/DD/YY` with a 1900 year offset, so `125`
/// means 2025.
pub fn parse_scan_datetime(date: &str, time: &str) -> Option<String> {
    let mut d = date.split('/');
    let month: u32 = d.next()?.trim().parse().ok()?;
    let day: u32 = d.next()?.trim().parse().ok()?;
    let year_offset: i32 = d.next()?.trim().parse().ok()?;
    let year = 1900 + year_offset;

    let mut t = time.split(':');
    let hour: u32 = t.next()?.trim().parse().ok()?;
    let minute: u32 = t.next()?.trim().parse().ok()?;

    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:00"
    ))
}

#[derive(Debug, Clone)]
pub struct GeHeader {
    pub exam_number: i64,
    pub series_number: i64,
    pub specnuc: i64,
    pub psdname: String,
    pub dfov: f64,
    pub slthick: f64,
    pub user14: f64,
    pub norm: [f64; 3],
    pub tlhc: [f64; 3],
    pub ctr: [f64; 3],
    pub scan_date: String,
    pub scan_time: String,
}

impl GeHeader {
    pub fn from_mat(m: &MatFile) -> Result<Self, MatError> {
        let triple = |a: &str, b: &str, c: &str| -> Result<[f64; 3], MatError> {
            Ok([m.scalar_f64(a)?, m.scalar_f64(b)?, m.scalar_f64(c)?])
        };

        Ok(Self {
            exam_number: m.scalar_f64("/h/exam/ex_no")? as i64,
            series_number: m.scalar_f64("/h/series/se_no")? as i64,
            specnuc: m.scalar_f64("/h/image/specnuc")? as i64,
            psdname: m.string("/h/image/psdname").unwrap_or_default(),
            dfov: m.scalar_f64("/h/image/dfov")?,
            slthick: m.scalar_f64("/h/image/slthick")?,
            user14: m.scalar_f64("/h/rdb_hdr/user14").unwrap_or(f64::NAN),
            norm: triple(
                "/h/image/norm_R",
                "/h/image/norm_A",
                "/h/image/norm_S",
            )?,
            tlhc: triple(
                "/h/image/tlhc_R",
                "/h/image/tlhc_A",
                "/h/image/tlhc_S",
            )?,
            ctr: triple("/h/image/ctr_R", "/h/image/ctr_A", "/h/image/ctr_S")?,
            scan_date: m.string("/h/rdb_hdr/scan_date").unwrap_or_default(),
            scan_time: m.string("/h/rdb_hdr/scan_time").unwrap_or_default(),
        })
    }

    pub fn nucleus_name(&self) -> (String, Option<String>) {
        nucleus_from_code(self.specnuc)
    }

    pub fn scan_datetime_iso(&self) -> Option<String> {
        parse_scan_datetime(&self.scan_date, &self.scan_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::MatFile;
    use crate::samples::sample_mat;

    macro_rules! header {
        ($name:expr) => {{
            let p = match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            };
            GeHeader::from_mat(&MatFile::open(&p).unwrap()).unwrap()
        }};
    }

    #[test]
    fn reads_identifiers() {
        let h = header!("MRS_2H");
        assert_eq!(h.exam_number, 20000);
        assert_eq!(h.series_number, 6);
        assert_eq!(h.specnuc, 2);
        assert_eq!(h.psdname, "fidall2");
    }

    #[test]
    fn reads_geometry_fields() {
        let h = header!("MRS_2H_slab");
        assert_eq!(h.dfov, 300.0);
        assert_eq!(h.slthick, 80.0);
        assert_eq!(h.user14, 1.0);
        // The slab is tilted.
        assert!((h.norm[1] - (-0.266223)).abs() < 1e-4);
        assert!((h.norm[2] - 0.963911).abs() < 1e-4);
    }

    #[test]
    fn maps_deuterium() {
        let h = header!("MRS_2H");
        let (name, warn) = h.nucleus_name();
        assert_eq!(name, "2H");
        assert!(warn.is_none());
    }

    #[test]
    fn maps_carbon13() {
        let h = header!("MRSI_13C");
        let (name, warn) = h.nucleus_name();
        assert_eq!(name, "13C");
        assert!(warn.is_none());
        assert_eq!(h.exam_number, 4874);
        assert_eq!(h.series_number, 10);
    }

    #[test]
    fn unknown_nucleus_code_warns_but_does_not_fail() {
        let (name, warn) = nucleus_from_code(77);
        assert_eq!(name, "77");
        assert!(warn.is_some(), "an unknown code must produce a warning");
    }

    #[test]
    fn parses_scan_datetime_with_1900_offset() {
        // scan_date "11/25/125" means 2025-11-25.
        assert_eq!(
            parse_scan_datetime("11/25/125", "10:31").unwrap(),
            "2025-11-25T10:31:00"
        );
    }

    #[test]
    fn reads_datetime_from_sample() {
        let h = header!("MRS_2H");
        assert_eq!(h.scan_datetime_iso().unwrap(), "2025-11-25T10:31:00");
    }
}
