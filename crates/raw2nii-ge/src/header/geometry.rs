//! GE geometry: voxel extents and the affine.
//!
//! Spec §6.2. Two rules carry non-obvious history:
//!
//! * `user14 == 91` under a `fidall*` psd marks a non-selective excitation
//!   pulse, which makes the excitation dimension unlocalised regardless of
//!   `slthick`. `MRS_2H` has `slthick = 40` yet is genuinely unlocalised.
//! * MRSI never consults `user14` — it is assumed to be a legacy SVS field.
//!   That assumption is unverified, so the ignored value is logged.

use super::fields::GeHeader;

/// Spec §2.2: unlocalised dimensions take a 10 m pixdim.
pub const UNLOCALISED_MM: f64 = 10000.0;

/// The `rdb_hdr/user14` value marking a non-selective excitation pulse.
const NONSELECTIVE_PULSE: f64 = 91.0;

#[derive(Debug, Clone)]
pub struct Localisation {
    pub extents_mm: [f64; 3],
    pub warnings: Vec<String>,
}

fn is_fidall(psdname: &str) -> bool {
    psdname.starts_with("fidall")
}

pub fn svs_localisation(h: &GeHeader) -> Localisation {
    let mut warnings = Vec::new();

    // roilenx and roileny are 0 on every sample: SVS is never localised
    // in plane by these sequences.
    let x = UNLOCALISED_MM;
    let y = UNLOCALISED_MM;

    let z = if !is_fidall(&h.psdname) {
        warnings.push(format!(
            "psd '{}' is not a fidall variant; the user14 localisation rule \
             does not apply, falling back to slthick = {}",
            h.psdname, h.slthick
        ));
        h.slthick
    } else if h.user14 == NONSELECTIVE_PULSE {
        warnings.push(format!(
            "rdb_hdr/user14 = {} indicates a non-selective pulse; \
             unlocalised spectroscopy expected, slthick = {} ignored",
            NONSELECTIVE_PULSE, h.slthick
        ));
        UNLOCALISED_MM
    } else {
        h.slthick
    };

    Localisation {
        extents_mm: [x, y, z],
        warnings,
    }
}

pub fn mrsi_localisation(h: &GeHeader, grid: [usize; 3]) -> Localisation {
    let mut warnings = Vec::new();

    if h.user14 == NONSELECTIVE_PULSE {
        warnings.push(format!(
            "rdb_hdr/user14 = {} ignored for MRSI: assumed a legacy SVS field, \
             spatial dimensions come from the encoding and dfov",
            NONSELECTIVE_PULSE
        ));
    }

    let extent = |n: usize, fallback: f64| -> f64 {
        if n > 1 {
            h.dfov / n as f64
        } else {
            fallback
        }
    };

    Localisation {
        extents_mm: [
            extent(grid[0], h.dfov),
            extent(grid[1], h.dfov),
            extent(grid[2], h.slthick),
        ],
        warnings,
    }
}

/// Build a 4x4 affine mapping voxel indices to scanner coordinates.
///
/// GE reports coordinates in an RAS-style frame (R, A, S); NIfTI uses LPS-like
/// qform/sform conventions with x and y negated.
pub fn build_affine(h: &GeHeader, extents_mm: [f64; 3]) -> [[f64; 4]; 4] {
    let normal = normalise(h.norm);
    let (col0, col1) = orthogonal_basis(normal);

    let mut a = [[0.0f64; 4]; 4];
    for (r, axis) in [col0, col1, normal].iter().enumerate() {
        // r indexes the output column, axis is the direction it advances in.
        for c in 0..3 {
            a[c][r] = axis[c] * extents_mm[r];
        }
    }

    // Negate R and A to reach NIfTI's convention.
    for c in 0..3 {
        a[0][c] = -a[0][c];
        a[1][c] = -a[1][c];
    }

    a[0][3] = -h.ctr[0];
    a[1][3] = -h.ctr[1];
    a[2][3] = h.ctr[2];
    a[3] = [0.0, 0.0, 0.0, 1.0];
    a
}

fn normalise(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Two unit vectors completing a right-handed basis with `n`.
fn orthogonal_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Pick whichever cardinal axis is least aligned with n, so the cross
    // product is well conditioned.
    let seed = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalise(cross(seed, n));
    let v = normalise(cross(n, u));
    (u, v)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(psdname: &str, user14: f64, slthick: f64, dfov: f64) -> GeHeader {
        GeHeader {
            exam_number: 1,
            series_number: 1,
            specnuc: 2,
            psdname: psdname.to_string(),
            dfov,
            slthick,
            user14,
            norm: [0.0, 0.0, 1.0],
            tlhc: [100.0, 100.0, 0.0],
            ctr: [0.0, 0.0, 0.0],
            scan_date: String::new(),
            scan_time: String::new(),
        }
    }

    #[test]
    fn svs_in_plane_is_always_unlocalised() {
        let loc = svs_localisation(&header("fidall2", 1.0, 80.0, 300.0));
        assert_eq!(loc.extents_mm[0], UNLOCALISED_MM);
        assert_eq!(loc.extents_mm[1], UNLOCALISED_MM);
    }

    #[test]
    fn svs_slab_uses_slthick() {
        // MRS_2H_slab: user14 == 1, slthick == 80.
        let loc = svs_localisation(&header("fidall2", 1.0, 80.0, 300.0));
        assert_eq!(loc.extents_mm[2], 80.0);
    }

    #[test]
    fn svs_unlocalised_pulse_overrides_slthick() {
        // MRS_2H: user14 == 91, slthick == 40 but the pulse is non-selective.
        let loc = svs_localisation(&header("fidall2", 91.0, 40.0, 300.0));
        assert_eq!(loc.extents_mm[2], UNLOCALISED_MM);
        assert!(
            loc.warnings.iter().any(|w| w.contains("91")),
            "must report the pulse that triggered the rule"
        );
    }

    #[test]
    fn unknown_psd_falls_back_to_slthick_with_a_warning() {
        // MRS_2H_TE_60 uses psd `echocsi`, where user14 means something else.
        let loc = svs_localisation(&header("echocsi", 1.0, 40.0, 300.0));
        assert_eq!(loc.extents_mm[2], 40.0);
        assert!(loc.warnings.iter().any(|w| w.contains("echocsi")));
    }

    #[test]
    fn mrsi_uses_dfov_over_grid_and_ignores_user14() {
        // MRSI_2H: user14 == 91, dfov 200, stored grid 16^3 -> 12.5 mm.
        let loc = mrsi_localisation(&header("fidall2", 91.0, 20.0, 200.0), [16, 16, 16]);
        assert_eq!(loc.extents_mm[0], 12.5);
        assert_eq!(loc.extents_mm[1], 12.5);
        assert_eq!(loc.extents_mm[2], 12.5);
        assert!(
            loc.warnings.iter().any(|w| w.contains("user14")),
            "the ignored user14 value must be logged"
        );
    }

    #[test]
    fn mrsi_2d_uses_slthick_for_the_single_slice() {
        // MRSI_13C: dfov 300, 8x8 grid -> 37.5 mm in plane, slthick 15 in z.
        let loc = mrsi_localisation(&header("fidall2", 1.0, 15.0, 300.0), [8, 8, 1]);
        assert_eq!(loc.extents_mm[0], 37.5);
        assert_eq!(loc.extents_mm[1], 37.5);
        assert_eq!(loc.extents_mm[2], 15.0);
    }

    #[test]
    fn affine_scales_rows_by_extent() {
        let h = header("fidall2", 1.0, 80.0, 300.0);
        let a = build_affine(&h, [10.0, 10.0, 80.0]);
        let col0 = (a[0][0].powi(2) + a[1][0].powi(2) + a[2][0].powi(2)).sqrt();
        let col2 = (a[0][2].powi(2) + a[1][2].powi(2) + a[2][2].powi(2)).sqrt();
        assert!((col0 - 10.0).abs() < 1e-6, "column norm {col0}");
        assert!((col2 - 80.0).abs() < 1e-6, "column norm {col2}");
        assert_eq!(a[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn affine_translation_comes_from_the_centre() {
        let mut h = header("fidall2", 1.0, 80.0, 300.0);
        h.ctr = [1.0, 2.0, 3.0];
        let a = build_affine(&h, [10.0, 10.0, 10.0]);
        // NIfTI is LPS-negated relative to GE's RAS-style centre.
        assert!((a[0][3] - (-1.0)).abs() < 1e-6);
        assert!((a[1][3] - (-2.0)).abs() < 1e-6);
        assert!((a[2][3] - 3.0).abs() < 1e-6);
    }
}
