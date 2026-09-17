use super::*;
use crate::*;
fn selected(class: SpeculativeStrategyClass) -> SelectedSpeculativeRealization {
    let id = |s: &str| SpeculativeIdentity::new(s).unwrap();
    let target = id("target");
    let strategy = id("prediction");
    let capture = SpeculativeCaptureSchema::new(
        id("capture"),
        [
            SpeculativeCaptureEntry::new(id("hidden"), vec![1, 2, 8], id("rank"), id("seam"))
                .unwrap(),
        ],
    )
    .unwrap();
    let state = SpeculativeStateCacheIdentityIngredients::new(
        target.clone(),
        strategy.clone(),
        None,
        None,
        id("artifact"),
        id("format"),
        id("placement"),
        0,
        id("text"),
        vec![id("target-state"), id("prediction-state")],
    )
    .unwrap();
    let requirements = SpeculativeRealizationRequirements::new(
        target.clone(),
        SpeculativeStrategyRequirements::embedded(
            class,
            strategy.clone(),
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap(),
        capture.clone(),
        SpeculativeMechanismRequirements::new([]),
        state,
    )
    .unwrap();
    let request =
        SpeculativeSelectionRequest::new(SpeculativePlacementRequest::Single, capture.clone())
            .with_architecture_proof(SpeculativeArchitectureCompatibilityProof::new(
                target,
                strategy,
                capture.identity().clone(),
            ));
    select_speculative_realization(
        &requirements,
        &request,
        &SpeculativeMechanismCapabilities::new(
            requirements.mechanisms().mechanisms().iter().copied(),
        ),
    )
    .unwrap()
}
fn plan(
    selected: &SelectedSpeculativeRealization,
    chunk: u64,
    aligned: bool,
) -> EmbeddedSchedulePlan<'_> {
    let shape = if selected.requirements().strategy().class()
        == SpeculativeStrategyClass::EmbeddedSequential
    {
        EmbeddedPredictionShape::Sequential {
            depth: NonZeroUsize::new(5).unwrap(),
        }
    } else {
        EmbeddedPredictionShape::Fused {
            depth: NonZeroUsize::new(3).unwrap(),
            maximum_proposals: NonZeroUsize::new(8).unwrap(),
        }
    };
    let prefill = PrefillControlPlan::new(
        eredu_core::InferenceGeometry {
            batch_size: 1,
            input_positions: 7,
            cached_positions: 11,
            max_output_tokens: 5,
            prefill_chunk_positions: chunk,
            output: OutputDemand::LastPosition,
        },
        true,
    )
    .unwrap();
    let config = SpeculativeConfig {
        max_tokens: 5,
        max_draft_tokens: 2,
        ..Default::default()
    };
    EmbeddedSchedulePlan::new(
        selected,
        shape,
        if aligned {
            PredictionPrefillAlignment::Aligned
        } else {
            PredictionPrefillAlignment::NextToken
        },
        29,
        prefill,
        23,
        &config,
        SpeculativeSchedulerOptions {
            lookahead_blocks: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
#[test]
fn shared_chunks_keep_prediction_pairing_and_physical_depth_separate_from_request_caps() {
    for class in [
        SpeculativeStrategyClass::EmbeddedSequential,
        SpeculativeStrategyClass::EmbeddedFused,
    ] {
        let selected = selected(class);
        for aligned in [false, true] {
            for chunk in [1, 3, 7] {
                let plan = plan(&selected, chunk, aligned);
                assert_eq!(plan.geometry().proposal_capacity(), 2);
                assert!(plan.shape().depth().get() > plan.geometry().proposal_capacity());
                let mut hidden_tokens = Vec::new();
                let mut projection = Vec::new();
                let mut seeds = 0;
                for index in 0..plan.prefill.span_count() {
                    let (target, prediction) = plan.prefill_invocations(index).unwrap();
                    projection.push(target.target_output().unwrap());
                    if let Some(prediction) = prediction {
                        let span = prediction.prefill_span().unwrap();
                        assert_eq!(span.seed_start, 29 + hidden_tokens.len() as u64);
                        for row in 0..span.sequence {
                            hidden_tokens.push((span.hidden_start + row, span.token_start + row));
                        }
                        seeds += 1;
                    }
                }
                let expected: Vec<_> = if aligned {
                    (0..7).map(|n| (n, n)).collect()
                } else {
                    (0..6).map(|n| (n, n + 1)).collect()
                };
                assert_eq!(hidden_tokens, expected);
                assert_eq!(
                    plan.attempts(EmbeddedOccurrenceKind::PredictionPrefill),
                    seeds
                );
                assert_eq!(projection.pop(), Some(OutputDemand::LastPosition));
                assert!(projection.iter().all(|d| *d == OutputDemand::StateOnly));
                if class == SpeculativeStrategyClass::EmbeddedFused {
                    let call = plan.decode_invocation(Phase::FusedProposal, 18, 2).unwrap();
                    assert_eq!(call.source_positions(), 1);
                    assert_eq!(call.positions(), 2);
                    assert_eq!(plan.attempts(EmbeddedOccurrenceKind::FusedProposal), 8);
                } else {
                    assert_eq!(
                        plan.attempts(EmbeddedOccurrenceKind::SequentialProposal),
                        16
                    );
                }
            }
        }
    }
}
#[test]
fn failed_and_discarded_claims_spend_attempts_and_restored_futures_add_without_refund() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let mut cursor = plan(&selected, 3, false).into_cursor();
    let call = cursor
        .plan()
        .decode_invocation(Phase::Verification, 18, 3)
        .unwrap();
    for ordinal in 0..4 {
        let claim = cursor.claim(call).unwrap();
        assert_eq!(claim.ordinal(), ordinal);
        drop(claim); // A native error or discarded branch does not refund it.
    }
    assert!(matches!(
        cursor.claim(call),
        Err(EmbeddedOccurrenceError::Exhausted)
    ));
    assert!(cursor
        .plan()
        .decode_invocation(Phase::Proposal { depth: 2 }, 18, 1)
        .is_err());
    assert!(cursor
        .plan()
        .decode_invocation(Phase::Verification, 22, 3)
        .is_err());
    let continuation = cursor
        .continuation(2, SpeculativeRequestStatus::ReadyToDraft)
        .unwrap();
    let stale = cursor
        .continuation(2, SpeculativeRequestStatus::ReadyToDraft)
        .unwrap();
    assert!(continuation.next_slots() > continuation.previous_slots());
    let prefill = cursor
        .plan()
        .attempts(EmbeddedOccurrenceKind::TargetPrefill);
    cursor.install_continuation(continuation).unwrap();
    assert!(cursor.install_continuation(stale).is_err());
    assert_eq!(cursor.attempted(), 4);
    assert_eq!(
        cursor
            .plan()
            .attempts(EmbeddedOccurrenceKind::TargetPrefill),
        prefill
    );
    assert_eq!(cursor.claim(call).unwrap().ordinal(), 4);
    let mut visited = 0;
    cursor
        .plan()
        .visit_candidates(EmbeddedOccurrenceKind::Verification, |candidate| {
            assert!(cursor.plan().contains(candidate));
            visited += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(visited > 1);
}
