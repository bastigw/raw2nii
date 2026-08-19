//! NIfTI-2 serialisation.
//!
//! The header is exactly 540 bytes, little-endian, followed by a 4-byte
//! extension flag, the MRS header extension, then the complex data.

use std::io::Write;
use std::path::Path;

use crate::dataset::MrsDataset;
use crate::error::Result;
use crate::meta::build_extension;

const HEADER_SIZE: usize = 540;
const DT_COMPLEX: i16 = 32;
const BITPIX: i16 = 64;
/// NIFTI_UNITS_MM (2) | NIFTI_UNITS_SEC (8).
const XYZT_UNITS: i32 = 10;
const INTENT_NAME: &[u8] = b"mrs_v0_11";

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(HEADER_SIZE),
        }
    }
    fn i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    /// Fixed-width, zero-padded, always NUL-terminated by the padding.
    fn fixed(&mut self, v: &[u8], width: usize) {
        let n = v.len().min(width);
        self.buf.extend_from_slice(&v[..n]);
        self.buf.extend(std::iter::repeat(0u8).take(width - n));
    }
}

pub fn serialise(ds: &MrsDataset) -> Result<Vec<u8>> {
    ds.validate()?;

    let ext = build_extension(ds);
    let vox_offset = (HEADER_SIZE + 4 + ext.len()) as i64;

    let shape = ds.data.shape();
    let mut dim = [1i64; 8];
    dim[0] = shape.len() as i64;
    for (i, &s) in shape.iter().enumerate() {
        dim[i + 1] = s as i64;
    }

    // pixdim[1..3] are spatial extents taken from the affine column norms;
    // pixdim[4] is the dwell time.
    let mut pixdim = [1.0f64; 8];
    pixdim[0] = 1.0;
    for c in 0..3 {
        pixdim[c + 1] = (ds.affine[0][c].powi(2)
            + ds.affine[1][c].powi(2)
            + ds.affine[2][c].powi(2))
        .sqrt();
    }
    pixdim[4] = ds.dwell_time_s;

    let mut w = Writer::new();
    w.i32(HEADER_SIZE as i32); // 0   sizeof_hdr
    w.bytes(b"n+2\0\r\n\x1a\n"); // 4   magic
    w.i16(DT_COMPLEX); // 12  datatype
    w.i16(BITPIX); // 14  bitpix
    for d in dim {
        w.i64(d);
    } // 16  dim[8]
    w.f64(0.0); // 80  intent_p1
    w.f64(0.0); // 88  intent_p2
    w.f64(0.0); // 96  intent_p3
    for p in pixdim {
        w.f64(p);
    } // 104 pixdim[8]
    w.i64(vox_offset); // 168 vox_offset
    w.f64(1.0); // 176 scl_slope
    w.f64(0.0); // 184 scl_inter
    w.f64(0.0); // 192 cal_max
    w.f64(0.0); // 200 cal_min
    w.f64(0.0); // 208 slice_duration
    w.f64(0.0); // 216 toffset
    w.i64(0); // 224 slice_start
    w.i64(0); // 232 slice_end
    w.fixed(b"raw2nii NIfTI-MRS", 80); // 240 descrip
    w.fixed(b"", 24); // 320 aux_file
    // qform_code = 0: no qform. quatern_b/c/d are hardcoded to 0.0 below,
    // which decodes to an identity rotation -- correct only for axis-aligned
    // acquisitions. A nonzero qform_code alongside that fake identity
    // quaternion would contradict the real, possibly-oblique affine in
    // srow_x/y/z, and some readers (e.g. FSL) prefer a nonzero qform. sform
    // is the sole, authoritative geometry; see srow_x/y/z below.
    w.i32(0); // 344 qform_code
    w.i32(1); // 348 sform_code
    w.f64(0.0); // 352 quatern_b
    w.f64(0.0); // 360 quatern_c
    w.f64(0.0); // 368 quatern_d
    w.f64(ds.affine[0][3]); // 376 qoffset_x
    w.f64(ds.affine[1][3]); // 384 qoffset_y
    w.f64(ds.affine[2][3]); // 392 qoffset_z
    for r in 0..3 {
        for c in 0..4 {
            w.f64(ds.affine[r][c]);
        }
    } // 400 srow_x/y/z
    w.i32(0); // 496 slice_code
    w.i32(XYZT_UNITS); // 500 xyzt_units
    w.i32(0); // 504 intent_code
    w.fixed(INTENT_NAME, 16); // 508 intent_name
    w.u8(0); // 524 dim_info
    w.fixed(b"", 15); // 525 unused_str

    debug_assert_eq!(w.buf.len(), HEADER_SIZE);

    let mut out = w.buf;
    out.extend_from_slice(&[1u8, 0, 0, 0]); // extension flag
    out.extend_from_slice(&ext);

    // NIfTI expects the first dimension to vary fastest. ndarray's standard
    // layout has the last axis varying fastest, so reverse the axes before
    // flattening.
    let flat = ds.data.view().reversed_axes();
    out.reserve(ds.data.len() * 8);
    for c in flat.iter() {
        out.extend_from_slice(&c.re.to_le_bytes());
        out.extend_from_slice(&c.im.to_le_bytes());
    }

    Ok(out)
}

/// Write to `path`, gzipping when it ends in `.gz`. Writes to a sibling
/// temporary file and renames, so an aborted run never leaves a partial file.
pub fn write_file(ds: &MrsDataset, path: &Path, gzip_level: u32) -> Result<()> {
    let bytes = serialise(ds)?;
    let tmp = path.with_extension("partial");

    let write_result = (|| -> std::io::Result<()> {
        let file = std::fs::File::create(&tmp)?;
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("gz"))
        {
            let mut encoder =
                flate2::write::GzEncoder::new(file, flate2::Compression::new(gzip_level));
            encoder.write_all(&bytes)?;
            // `.finish()` writes the gzip trailer (CRC32 + ISIZE); relying on
            // Drop would silently discard any error from that final write.
            encoder.finish()?;
        } else {
            let mut file = file;
            file.write_all(&bytes)?;
            file.flush()?;
        }
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }

    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{identity_affine, DimTag, Metadata, MrsDataset};
    use ndarray::{ArrayD, IxDyn};
    use num_complex::Complex;

    fn svs() -> MrsDataset {
        MrsDataset {
            data: ArrayD::from_elem(IxDyn(&[1, 1, 1, 8, 2]), Complex::new(1.5, -2.5)),
            tags: [Some(DimTag::Dyn), None, None],
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
    fn header_is_540_bytes_with_nifti2_magic() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i32_at(&b, 0), 540);
        assert_eq!(&b[4..12], b"n+2\0\r\n\x1a\n");
    }

    #[test]
    fn datatype_is_complex64() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i16_at(&b, 12), 32, "DT_COMPLEX");
        assert_eq!(i16_at(&b, 14), 64, "bitpix");
    }

    #[test]
    fn dimensions_are_written_in_nifti_order() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i64_at(&b, 16), 5, "dim[0]");
        assert_eq!(i64_at(&b, 24), 1);
        assert_eq!(i64_at(&b, 32), 1);
        assert_eq!(i64_at(&b, 40), 1);
        assert_eq!(i64_at(&b, 48), 8);
        assert_eq!(i64_at(&b, 56), 2);
    }

    #[test]
    fn dwell_time_is_in_pixdim_four() {
        let b = serialise(&svs()).unwrap();
        // pixdim starts at 104; pixdim[4] is the fifth element.
        assert!((f64_at(&b, 104 + 4 * 8) - 2e-4).abs() < 1e-12);
    }

    #[test]
    fn intent_name_declares_the_standard_version() {
        let b = serialise(&svs()).unwrap();
        let name = &b[508..524];
        let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
        assert_eq!(std::str::from_utf8(&name[..end]).unwrap(), "mrs_v0_11");
    }

    #[test]
    fn xyzt_units_are_mm_and_seconds() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(i32_at(&b, 500), 10, "NIFTI_UNITS_MM | NIFTI_UNITS_SEC");
    }

    #[test]
    fn qform_is_disabled_and_sform_is_authoritative() {
        // Finding 2: quatern_b/c/d are hardcoded to an identity rotation, so
        // qform_code must be 0 ("no qform") to avoid contradicting the real,
        // possibly-oblique affine carried in srow_x/y/z under sform_code.
        let b = serialise(&svs()).unwrap();
        assert_eq!(i32_at(&b, 344), 0, "qform_code");
        assert_eq!(i32_at(&b, 348), 1, "sform_code");
    }

    #[test]
    fn extension_flag_is_set_and_vox_offset_points_past_it() {
        let b = serialise(&svs()).unwrap();
        assert_eq!(b[540], 1, "extension flag");
        let vox_offset = i64_at(&b, 168);
        let esize = i32_at(&b, 544) as i64;
        assert_eq!(vox_offset, 544 + esize);
    }

    #[test]
    fn data_is_written_as_interleaved_f32_pairs() {
        let ds = svs();
        let b = serialise(&ds).unwrap();
        let vox_offset = i64_at(&b, 168) as usize;
        let n: usize = ds.data.shape().iter().product();
        assert_eq!(b.len(), vox_offset + n * 8);
        let re = f32::from_le_bytes(b[vox_offset..vox_offset + 4].try_into().unwrap());
        let im = f32::from_le_bytes(b[vox_offset + 4..vox_offset + 8].try_into().unwrap());
        assert_eq!(re, 1.5);
        assert_eq!(im, -2.5);
    }

    #[test]
    fn rejects_an_invalid_dataset() {
        let mut ds = svs();
        ds.meta.resonant_nucleus.clear();
        assert!(serialise(&ds).is_err());
    }

    #[test]
    fn dim1_varies_fastest_in_the_byte_stream() {
        // Distinct values along the t axis (shape[3]) so a wrong or missing
        // axis reversal would produce a byte stream that fails this check,
        // unlike a uniform fixture where any ordering looks identical.
        let mut ds = svs();
        ds.data =
            ArrayD::from_shape_fn(IxDyn(&[1, 1, 1, 4, 1]), |idx| Complex::new(idx[3] as f32, 0.0));

        let b = serialise(&ds).unwrap();
        let vox_offset = i64_at(&b, 168) as usize;
        for i in 0..4 {
            let off = vox_offset + i * 8;
            let re = f32::from_le_bytes(b[off..off + 4].try_into().unwrap());
            assert_eq!(re, i as f32, "sample {i} out of order");
        }
    }

    #[test]
    fn dim4_varies_faster_than_dim5_in_the_byte_stream() {
        // Shape [1,1,1,3,2] has two non-singleton axes of *different* sizes
        // (t=3, r=2). A wrong or missing axis reversal changes not just the
        // order but the effective traversal shape, unlike the [1,1,1,4,1]
        // fixture above (which has only one non-singleton axis, so no
        // permutation of size-1 axes can change its byte order).
        let mut data = ArrayD::<Complex<f32>>::zeros(IxDyn(&[1, 1, 1, 3, 2]));
        for t in 0..3 {
            for r in 0..2 {
                data[[0, 0, 0, t, r]] = Complex::new((10 * t + r) as f32, 0.0);
            }
        }
        let ds = MrsDataset {
            data,
            tags: [Some(DimTag::Dyn), None, None],
            affine: identity_affine(),
            dwell_time_s: 2e-4,
            meta: Metadata {
                spectrometer_frequency_mhz: vec![19.5934],
                resonant_nucleus: vec!["2H".to_string()],
                extra: serde_json::Map::new(),
                warnings: vec![],
            },
        };
        let b = serialise(&ds).unwrap();
        let vox_offset = i64_at(&b, 168) as usize;
        let expected = [0.0f32, 10.0, 20.0, 1.0, 11.0, 21.0];
        for (i, &want) in expected.iter().enumerate() {
            let off = vox_offset + i * 8; // 8 bytes per complex sample (f32 re + f32 im)
            let got = f32::from_le_bytes(b[off..off + 4].try_into().unwrap());
            assert_eq!(got, want, "sample {i} at byte offset {off}");
        }
    }

    #[test]
    fn gzip_output_round_trips_to_the_uncompressed_bytes() {
        use std::io::Read as _;

        let ds = svs();
        let uncompressed = serialise(&ds).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "raw2nii_write_test_{}_{}.nii.gz",
            std::process::id(),
            "gzip_round_trip"
        ));
        write_file(&ds, &path, 6).unwrap();

        let compressed = std::fs::read(&path).unwrap();
        let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        std::fs::remove_file(&path).ok();

        assert_eq!(decompressed, uncompressed);
    }

    #[test]
    fn write_file_writes_uncompressed_bytes_for_non_gz_paths() {
        let ds = svs();
        let expected = serialise(&ds).unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "raw2nii_write_test_{}_{}.nii",
            std::process::id(),
            "plain"
        ));
        write_file(&ds, &path, 6).unwrap();

        let actual = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(actual, expected);
    }
}
