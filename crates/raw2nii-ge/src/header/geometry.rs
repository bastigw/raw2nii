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
///
/// The translation is the world position of voxel index `(0, 0, 0)` — the
/// array corner — not `h.ctr`, which is the volume's geometric centre. For a
/// grid of `n` voxels along a direction, the centre sits at index
/// `(n - 1) / 2`, so the corner is offset from the centre by that many voxel
/// extents, back along that direction. For a single-voxel grid (SVS) this
/// offset is zero and the corner coincides with the centre.
pub fn build_affine(h: &GeHeader, extents_mm: [f64; 3], grid: [usize; 3]) -> [[f64; 4]; 4] {
    let normal = normalise(h.norm);
    let col0 = normalise(subtract(h.trhc, h.tlhc));
    let col1 = normalise(subtract(h.brhc, h.trhc));
    let axes = [col0, col1, normal];

    let mut a = [[0.0f64; 4]; 4];
    for (r, axis) in axes.iter().enumerate() {
        // r indexes the output column, axis is the direction it advances in.
        for c in 0..3 {
            a[c][r] = axis[c] * extents_mm[r];
        }
    }

    // Negate R and A to reach NIfTI's convention.
    for x in a[0].iter_mut().take(3) {
        *x = -*x;
    }
    for x in a[1].iter_mut().take(3) {
        *x = -*x;
    }

    let mut corner = h.ctr;
    for (r, axis) in axes.iter().enumerate() {
        let half_extent = extents_mm[r] * (grid[r] as f64 - 1.0) / 2.0;
        for c in 0..3 {
            corner[c] -= axis[c] * half_extent;
        }
    }

    a[0][3] = -corner[0];
    a[1][3] = -corner[1];
    a[2][3] = corner[2];
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

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
            // Chosen so trhc - tlhc normalises to [0,-1,0] and brhc - trhc
            // normalises to [1,0,0] -- the same col0/col1 the old
            // orthogonal_basis([0,0,1]) construction produced, so the affine
            // tests below (written against that basis) still hold under the
            // corner-derived construction.
            tlhc: [100.0, 100.0, 0.0],
            trhc: [100.0, 0.0, 0.0],
            brhc: [200.0, 0.0, 0.0],
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
        let a = build_affine(&h, [10.0, 10.0, 80.0], [1, 1, 1]);
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
        // grid = [1, 1, 1]: a single voxel, so the corner coincides with the
        // centre and this is unaffected by the corner-offset fix below.
        let a = build_affine(&h, [10.0, 10.0, 10.0], [1, 1, 1]);
        // NIfTI is LPS-negated relative to GE's RAS-style centre.
        assert!((a[0][3] - (-1.0)).abs() < 1e-6);
        assert!((a[1][3] - (-2.0)).abs() < 1e-6);
        assert!((a[2][3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn affine_translation_offsets_to_the_corner_voxel_for_multi_voxel_grids() {
        // norm = [0,0,1] (the default in the `header()` test helper) means
        // normal = [0,0,1] (pure S), and the corner triple in `header()` is
        // chosen so col0 = [0,-1,0], col1 = [1,0,0] (see the comment there).
        // A 4x4x4 grid with 10mm voxels offsets the
        // corner from the centre by (4-1)/2 * 10 = 15mm along each of those
        // three directions: -15 along col0 = +15mm on the A (world Y) axis
        // before negation, +15mm along col1 = -15mm on the R (world X) axis
        // before negation, and -15mm along S (world Z, not negated).
        let mut h = header("fidall2", 1.0, 80.0, 300.0);
        h.ctr = [0.0, 0.0, 0.0];
        let a = build_affine(&h, [10.0, 10.0, 10.0], [4, 4, 4]);
        // (grid - 1) / 2 * extent = 1.5 * 10 = 15mm offset from centre to
        // corner along each grid axis's real direction, R and A negated
        // relative to S per the LPS convention.
        assert!((a[0][3] - 15.0).abs() < 1e-6, "R translation: {}", a[0][3]);
        assert!(
            (a[1][3] - (-15.0)).abs() < 1e-6,
            "A translation: {}",
            a[1][3]
        );
        assert!(
            (a[2][3] - (-15.0)).abs() < 1e-6,
            "S translation: {}",
            a[2][3]
        );
    }

    #[test]
    fn mrsi_extents_are_identical_regardless_of_user14() {
        let selective = mrsi_localisation(&header("fidall2", 1.0, 15.0, 300.0), [8, 8, 4]);
        let nonselective = mrsi_localisation(&header("fidall2", 91.0, 15.0, 300.0), [8, 8, 4]);
        assert_eq!(selective.extents_mm, nonselective.extents_mm);
        assert!(!selective.warnings.iter().any(|w| w.contains("user14")));
        assert!(nonselective.warnings.iter().any(|w| w.contains("user14")));
    }

    #[test]
    fn svs_psd_check_gates_before_the_user14_check() {
        // Non-fidall psd with user14 == 91: the psd-not-recognised branch
        // must win, falling back to slthick rather than treating this as
        // the non-selective-pulse case.
        let loc = svs_localisation(&header("echocsi", 91.0, 40.0, 300.0));
        assert_eq!(loc.extents_mm[2], 40.0);
        assert!(loc.warnings.iter().any(|w| w.contains("echocsi")));
    }

    #[test]
    fn build_affine_derives_in_plane_axes_from_corners_not_an_arbitrary_seed() {
        // Regression guard for Finding 1: the in-plane axes must come from
        // the trhc/tlhc/brhc corner triple, not from an arbitrary cardinal
        // seed crossed with the slice normal. MRS_2H_slab is a genuinely
        // tilted acquisition (h.norm ~= [0, -0.266, 0.964]), so the old
        // orthogonal_basis(normal) construction and the corner-derived one
        // disagree -- a fixture aligned with a cardinal axis would not
        // distinguish them.
        let p = match crate::samples::sample_mat("MRS_2H_slab") {
            Some(p) => p,
            None => {
                eprintln!("SKIP: tests/datasets absent");
                return;
            }
        };
        let m = crate::mat::MatFile::open(&p).unwrap();
        let h = GeHeader::from_mat(&m).unwrap();

        // Sanity check: this really is a tilted acquisition, not axis-aligned.
        assert!((h.norm[1] - (-0.266223)).abs() < 1e-3);
        assert!((h.norm[2] - 0.963911).abs() < 1e-3);

        let expected_col0 = normalise(subtract(h.trhc, h.tlhc));
        let expected_col1 = normalise(subtract(h.brhc, h.trhc));
        let expected_normal = normalise(h.norm);

        // The old, buggy construction: an arbitrary cardinal seed crossed
        // with the slice normal. This must NOT match the corner-derived
        // basis on a tilted acquisition -- if it does, the fix regressed.
        let seed = if expected_normal[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }
        let old_col0 = normalise(cross(seed, expected_normal));

        let diff = (0..3)
            .map(|i| (old_col0[i] - expected_col0[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 0.1,
            "expected the corner-derived col0 to differ substantially from \
             the old orthogonal_basis seed-derived col0 on a tilted \
             acquisition; got old={old_col0:?} new={expected_col0:?}"
        );

        // The affine actually built by build_affine must match the
        // corner-derived basis (via its column norms and orientation),
        // scaled by the given extents.
        let extents = [10.0, 20.0, 30.0];
        let a = build_affine(&h, extents, [1, 1, 1]);

        // Reconstruct the unit column directions the affine encodes (undoing
        // the extent scaling and the R/A negation) and compare against the
        // independently-derived expected axes.
        let mut got_col0 = [
            a[0][0] / extents[0],
            a[1][0] / extents[0],
            a[2][0] / extents[0],
        ];
        let mut got_col1 = [
            a[0][1] / extents[1],
            a[1][1] / extents[1],
            a[2][1] / extents[1],
        ];
        // Undo the R/A negation applied in build_affine. The negation loops
        // over world-coordinate rows (R, A), not per-axis columns, so it
        // applies to every axis's R and A component -- including normal's.
        got_col0[0] = -got_col0[0];
        got_col0[1] = -got_col0[1];
        got_col1[0] = -got_col1[0];
        got_col1[1] = -got_col1[1];
        let mut got_col2 = [a[0][2] / extents[2], a[1][2] / extents[2], a[2][2] / extents[2]];
        got_col2[0] = -got_col2[0];
        got_col2[1] = -got_col2[1];

        for i in 0..3 {
            assert!(
                (got_col0[i] - expected_col0[i]).abs() < 1e-9,
                "col0[{i}]: got {}, expected {}",
                got_col0[i],
                expected_col0[i]
            );
            assert!(
                (got_col1[i] - expected_col1[i]).abs() < 1e-9,
                "col1[{i}]: got {}, expected {}",
                got_col1[i],
                expected_col1[i]
            );
            assert!(
                (got_col2[i] - expected_normal[i]).abs() < 1e-9,
                "normal[{i}]: got {}, expected {}",
                got_col2[i],
                expected_normal[i]
            );
        }

        // Sanity: the corner-derived basis is unit-length and mutually
        // orthogonal, as any valid rotation basis must be.
        let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let len = |u: [f64; 3]| dot(u, u).sqrt();
        assert!((len(expected_col0) - 1.0).abs() < 1e-9);
        assert!((len(expected_col1) - 1.0).abs() < 1e-9);
        assert!((len(expected_normal) - 1.0).abs() < 1e-9);
        assert!(dot(expected_col0, expected_col1).abs() < 1e-4);
        assert!(dot(expected_col0, expected_normal).abs() < 1e-4);
        assert!(dot(expected_col1, expected_normal).abs() < 1e-4);
    }
}
