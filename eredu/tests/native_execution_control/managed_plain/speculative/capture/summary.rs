//! The same finite scalar summary for target/draft rows and controlled delivery.
use super::*;
use eredu_core::capture::{CapturePayload, CaptureRecord};

pub(super) fn compare(row: &[f32], records: &[CaptureRecord]) {
    let CapturePayload::Summary(summary) = records[1].payload.as_ref().unwrap() else {
        panic!("finite summary")
    };
    assert_eq!(summary.elements, row.len() as u64);
    assert_eq!(summary.finite, row.len() as u64);
    assert_eq!(
        (
            summary.non_finite,
            summary.nan,
            summary.positive_infinity,
            summary.negative_infinity
        ),
        (0, 0, 0, 0)
    );
    let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / row.len() as f64;
    let rms = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / row.len() as f64).sqrt();
    let min = f64::from(row.iter().copied().fold(f32::INFINITY, f32::min));
    let max = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    for (actual, expected) in [
        (summary.min, min),
        (summary.max, max),
        (summary.mean, mean),
        (summary.rms, rms),
    ] {
        assert!((actual.unwrap() - expected).abs() < 2e-5 + 2e-5 * expected.abs());
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_speculative_summary_matches_full_rows() {
    compare_modes(CaptureKind::Summary);
}
