//! SVS versus MRSI detection.
//!
//! This is a fidall-`.mat` distinction, not a universal one, so it stays
//! private to this backend.

use crate::mat::MatFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Svs,
    Mrsi,
}

pub fn detect(m: &MatFile) -> Option<Flavor> {
    if m.has("/fid") && m.has("/par/samples") {
        return Some(Flavor::Svs);
    }
    if m.has("/spec") && m.has("/nn") && m.has("/dim") {
        return Some(Flavor::Mrsi);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::MatFile;
    use crate::samples::sample_mat;

    macro_rules! detect_sample {
        ($name:expr) => {{
            let p = match sample_mat($name) {
                Some(p) => p,
                None => {
                    eprintln!("SKIP: tests/datasets absent");
                    return;
                }
            };
            detect(&MatFile::open(&p).unwrap())
        }};
    }

    #[test]
    fn detects_svs() {
        assert_eq!(detect_sample!("MRS_2H"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_slab"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_TE_60"), Some(Flavor::Svs));
        assert_eq!(detect_sample!("MRS_2H_TI_400"), Some(Flavor::Svs));
    }

    #[test]
    fn detects_mrsi() {
        assert_eq!(detect_sample!("MRSI_13C"), Some(Flavor::Mrsi));
        assert_eq!(detect_sample!("MRSI_2H"), Some(Flavor::Mrsi));
    }
}
