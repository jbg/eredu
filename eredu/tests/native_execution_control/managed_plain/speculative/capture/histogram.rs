//! Fixed-edge counts from actual speculative full rows, through all shared drivers.
use super::*;
use eredu_core::capture::{CaptureHistogram, CapturePayload, CaptureRecord};

pub(super) const EDGES: [f32; 5] = [-1.0, -0.125, 0.0, 0.125, 1.0];

fn payload(records: &[CaptureRecord]) -> &CaptureHistogram {
    let CapturePayload::Histogram(histogram) = records[1].payload.as_ref().unwrap() else {
        panic!("fixed-edge histogram")
    };
    histogram
}

pub(super) fn compare(row: &[f32], records: &[CaptureRecord]) {
    let actual = payload(records);
    let mut counts = [0u64; EDGES.len() - 1];
    let (mut below, mut above, mut non_finite) = (0u64, 0u64, 0u64);
    for &value in row {
        if !value.is_finite() {
            non_finite += 1;
        } else if value < EDGES[0] {
            below += 1;
        } else if value > EDGES[EDGES.len() - 1] {
            above += 1;
        } else {
            let bin = EDGES
                .windows(2)
                .enumerate()
                .find_map(|(index, edges)| {
                    (value >= edges[0]
                        && (value < edges[1] || index == counts.len() - 1 && value <= edges[1]))
                        .then_some(index)
                })
                .expect("every finite in-range value has an exact bin");
            counts[bin] += 1;
        }
    }
    assert_eq!(actual.edges, EDGES);
    assert_eq!(actual.counts, counts);
    assert_eq!(
        (actual.below, actual.above, actual.non_finite),
        (below, above, non_finite)
    );
    assert_eq!(
        actual.counts.iter().sum::<u64>() + below + above + non_finite,
        row.len() as u64
    );
}

pub(super) fn encoded(records: &[CaptureRecord]) -> serde_json::Value {
    let histogram = payload(records);
    serde_json::json!({"edges":histogram.edges,"counts":histogram.counts,
        "below":histogram.below,"above":histogram.above,"non_finite":histogram.non_finite})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_speculative_histogram_matches_full_rows() {
    compare_modes(CaptureKind::Histogram);
}
