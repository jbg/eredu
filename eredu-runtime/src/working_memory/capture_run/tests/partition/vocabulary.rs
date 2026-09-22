//! Complete distribution metadata uses the exact original requested identities.
use super::*;
use crate::capture::partition::PartitionCaptureRecordEncoding;

pub(super) fn vocabulary_source(scores: bool, prompt: u64) -> SharedCapturePlan {
    super::super::candidates::source_with_transform(
        prompt,
        if scores {
            CaptureTransform::TokenScores {
                token_ids: vec![3, 1],
            }
        } else {
            CaptureTransform::TopCandidates { count: 2 }
        },
        CaptureLimitPolicy::Fail,
        u64::MAX,
    )
}
pub(super) fn vocabulary_record(
    mut record: CaptureRecord,
    scores: bool,
    rows: u64,
    domain: bool,
) -> CaptureRecord {
    let summary = domain.then_some(CandidateDomain {
        allowed_tokens: 2,
        vocabulary: 4,
        constrained: true,
    });
    let candidate = |id: u32| CaptureCandidate {
        token_id: id,
        score: id as f32,
        allowed: !domain || id == 0 || id == 3,
    };
    record.source_shape = Some(vec![1, rows, 4]);
    record.selected_shape = record.source_shape.clone();
    record.source_dtype = Some(TensorDtype::F16);
    record.outcome = CaptureOutcome::Captured;
    record.payload = Some(if scores {
        let partition = (0..4).map(|id| ((id as f64) - 3.0).exp()).sum::<f64>().ln() + 3.0;
        CapturePayload::TokenScores(CaptureTokenScores {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            vocabulary: 4,
            log_partition: partition,
            scores: [3, 1]
                .into_iter()
                .map(|id| CaptureTokenScore {
                    target: candidate(id),
                    log_probability: f64::from(id) - partition,
                    rank: 4 - u64::from(id),
                    strongest_alternative: Some(candidate(if id == 3 { 2 } else { 3 })),
                })
                .collect(),
            domain: summary,
        })
    } else {
        CapturePayload::Candidates(CaptureCandidates {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: CandidateLogitsSource::Original,
            candidates: vec![candidate(3), candidate(2)],
            domain: summary,
        })
    });
    record.charged = record.charged.checked_add(native_usage()).unwrap();
    record
}
#[test]
fn partition_vocabulary_reader_preserves_requested_ids_domains_and_failed_prefix_custody() {
    for scores in [false, true] {
        for case in [
            "valid",
            "unknown-domain",
            "id",
            "source",
            "domain",
            "short",
            "syntax",
            "funding",
        ] {
            let source = vocabulary_source(scores, 3);
            let h = plan(&source).initialization_peak_bytes();
            let pool = capture_test_ledger(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let mut bank = run
                .prepare_capture_run(&reservation, plan(&source))
                .unwrap();
            let mut frame = bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            let context = context(&source, 0);
            let record = vocabulary_record(
                frame.records()[0].clone(),
                scores,
                3,
                case != "unknown-domain",
            );
            let (funding, used, refuse, retired) = funding();
            let encoded = PartitionCaptureRecordEncoding::new(
                &context,
                "retained-receipt",
                3,
                PartitionCaptureCombination::Disjoint,
                &record,
                64 << 10,
            )
            .encode(&funding)
            .unwrap();
            let text = std::str::from_utf8(&encoded).unwrap();
            let bytes = match case {
                "id" => text
                    .replacen("\"token_id\":3", "\"token_id\":2", 1)
                    .into_bytes(),
                "source" => text
                    .replace("\"source\":\"original\"", "\"source\":\"effective\"")
                    .into_bytes(),
                "domain" => text
                    .replace("\"allowed_tokens\":2", "\"allowed_tokens\":5")
                    .into_bytes(),
                "short" => text
                    .replace(
                        if scores {
                            "\"rank\":1"
                        } else {
                            "\"allowed\":true"
                        },
                        "\"unrecognized\":1",
                    )
                    .into_bytes(),
                "syntax" => text.as_bytes()[..text.len() - 5].to_vec(),
                _ => text.as_bytes().to_vec(),
            };
            if case == "funding" {
                refuse.store(true, Ordering::SeqCst);
            }
            let valid = case == "valid" || case == "unknown-domain";
            let error = if scores {
                let claim = frame.take_token_scores(0).unwrap();
                let result = claim.decode_partition_receipt(
                    &bytes,
                    expected(&context, record.charged),
                    &funding,
                );
                assert!(frame.take_token_scores(0).is_err());
                if valid {
                    let result = result.unwrap();
                    assert_eq!(
                        Some(&CapturePayload::TokenScores(result.observation().clone())),
                        record.payload.as_ref()
                    );
                    frame
                        .record_token_scores(result, TensorDtype::F16, native_usage())
                        .unwrap();
                    None
                } else {
                    Some(result.unwrap_err())
                }
            } else {
                let claim = frame.take_candidates(0).unwrap();
                let result = claim.decode_partition_receipt(
                    &bytes,
                    expected(&context, record.charged),
                    &funding,
                );
                assert!(frame.take_candidates(0).is_err());
                if valid {
                    let result = result.unwrap();
                    assert_eq!(
                        Some(&CapturePayload::Candidates(result.observation().clone())),
                        record.payload.as_ref()
                    );
                    frame
                        .record_candidates(result, TensorDtype::F16, native_usage())
                        .unwrap();
                    None
                } else {
                    Some(result.unwrap_err())
                }
            };
            assert!(used.load(Ordering::SeqCst) > 0);
            drop(encoded);
            drop(frame);
            drop(bank);
            drop(run);
            drop(reservation);
            drop(funding);
            if let Some(error) = error {
                assert_eq!(pool.payload_used_bytes().unwrap(), h);
                assert!(!retired.load(Ordering::SeqCst));
                drop(error);
            }
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            assert!(retired.load(Ordering::SeqCst));
        }
    }
}
