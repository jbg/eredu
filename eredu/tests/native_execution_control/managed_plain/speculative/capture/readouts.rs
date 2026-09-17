//! Both funded transforms agree with the same captured unmodified numerical row.
use super::*;
use eredu_core::capture::{
    CandidateLogitsSource, CandidateScoreStage, CapturePayload, CaptureRecord,
};

pub(super) fn compare(row: &[f32], records: &[CaptureRecord]) {
    let CapturePayload::TokenScores(scored) = records[1].payload.as_ref().unwrap() else {
        panic!("ordered scores")
    };
    let CapturePayload::Candidates(candidates) = records[2].payload.as_ref().unwrap() else {
        panic!("sorted candidates")
    };
    assert_eq!(scored.vocabulary as usize, row.len());
    assert_eq!(scored.stage, CandidateScoreStage::RawLogitsBeforeSampling);
    assert_eq!(
        candidates.stage,
        CandidateScoreStage::RawLogitsBeforeSampling
    );
    assert_eq!(scored.source, CandidateLogitsSource::Original);
    assert_eq!(candidates.source, CandidateLogitsSource::Original);
    let domain = Some(eredu_core::capture::CandidateDomain {
        allowed_tokens: row.len() as u64, vocabulary: row.len() as u64, constrained: false,
    });
    assert_eq!(scored.domain, domain);
    assert_eq!(candidates.domain, domain);
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let partition = maximum
        + row
            .iter()
            .map(|&v| (f64::from(v) - maximum).exp())
            .sum::<f64>()
            .ln();
    assert!((scored.log_partition - partition).abs() < 2e-5);
    assert_eq!(scored.scores.len(), 3);
    for (actual, id) in scored.scores.iter().zip([0u32, 17, 63]) {
        let value = row[id as usize];
        assert_eq!(actual.target.token_id, id);
        assert!((actual.target.score - value).abs() < 2e-5);
        assert!(actual.target.allowed);
        assert!((actual.log_probability - (f64::from(value) - partition)).abs() < 3e-5);
        assert_eq!(
            actual.rank,
            1 + row.iter().filter(|&&other| other > value).count() as u64
        );
        let alternative = actual.strongest_alternative.as_ref().unwrap();
        let expected = row
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != id as usize)
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .unwrap();
        assert_eq!(alternative.token_id as usize, expected.0);
        assert!((alternative.score - expected.1).abs() < 2e-5);
        assert!(alternative.allowed);
    }
    let mut ids: Vec<_> = (0..row.len()).collect();
    ids.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap().then(a.cmp(&b)));
    assert_eq!(candidates.candidates.len(), 3);
    for (actual, expected) in candidates.candidates.iter().zip(ids.into_iter().rev()) {
        assert_eq!(actual.token_id as usize, expected);
        assert!((actual.score - row[expected]).abs() < 2e-5);
        assert!(actual.allowed);
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_speculative_scores_and_candidates_match_full_rows() {
    compare_modes(CaptureKind::Readouts);
}
