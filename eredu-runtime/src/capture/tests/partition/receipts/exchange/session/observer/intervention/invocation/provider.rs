use super::*;
use crate::capture::{
    CaptureBackendProvider, PartitionCaptureBackendProvider, SpeculativeCaptureObserver,
    SpeculativeCaptureScope,
};
use crate::inspection::{with_speculative_activation, SpeculativeActivationObserver};
use eredu_core::speculative::{SpeculativeActivationOrigin, SpeculativeActivationPhase as Phase};

struct Provider;
impl CaptureBackendProvider for Provider {
    type Tensor = Value;
    type Error = std::io::Error;
    type Backend<'a> = Backend;
    fn backend(&mut self) -> Backend {
        Backend::default()
    }
}
struct Bootstrap(Transport);
impl ConsensusTransport for Bootstrap {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.0.participant_count()
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let submitted = self.0.submit_all_gather_words(words)?;
        submitted.completion.wait()?;
        self.0.resolve_all_gather_words(submitted.output)
    }
}
fn partition_error(
    error: PartitionCaptureObserverError<std::io::Error>,
) -> CaptureExecutionError<std::io::Error> {
    match error {
        PartitionCaptureObserverError::Capture(error) => error,
        error => CaptureExecutionError::Backend(std::io::Error::other(error)),
    }
}

#[test]
fn speculative_collector_binds_live_partition_provider_and_preserves_scoped_replay() {
    let all = world(5);
    let hooks = world(4);
    let setup = world(5);
    let results = std::thread::scope(|scope| {
        (0..5)
            .map(|rank| {
                let all = Arc::clone(&all);
                let hooks = Arc::clone(&hooks);
                let setup = Arc::clone(&setup);
                scope.spawn(move || {
                    let identity = crate::establish_communication_session(
                        &Bootstrap(transport(setup, rank, Fault::None)),
                        &crate::CommunicationManifest::new(5, rank, vec![], vec![]).unwrap(),
                        Some([rank as u8 + 1; 32]),
                    )
                    .unwrap()
                    .identity();
                    let identity = PartitionCaptureIdentity::for_session(
                        "artifact-exact".into(),
                        "retained-prediction-execution".into(),
                        identity,
                        Some("effective-overlay".into()),
                    )
                    .unwrap();
                    let members = vec![0, 2, 3, 4];
                    let transport = Arc::new(HookTransport {
                        transport: transport(all, rank, Fault::None),
                        hook: members
                            .iter()
                            .position(|r| *r == rank)
                            .map(|r| transport(hooks, r, Fault::None)),
                        members,
                    });
                    let layout = Arc::new(layout());
                    let (capture, plan) = plans_at(
                        rank,
                        InterventionEvidence::Preview { max_elements: 100 },
                        false,
                        Some(CaptureInvocationBounds {
                            batch: 1,
                            max_sequence: 5,
                            max_context: None,
                            max_predictions: 8,
                        }),
                    );
                    let catalog = discovery(&capture);
                    let mut session = CaptureSession::new(capture);
                    session
                        .enable_interventions(plan, Arc::new(Estimates))
                        .unwrap();
                    let provider = PartitionCaptureBackendProvider::new(
                        Provider,
                        transport,
                        Arc::clone(&layout),
                        identity,
                        PartitionCaptureReceiptLimits {
                            max_producers: 5,
                            max_fragments: 64,
                            max_record_bytes: 1 << 16,
                        },
                        estimate,
                        partition_error,
                    );
                    let mut observer = SpeculativeCaptureObserver::new(
                        session,
                        provider,
                        |error: &CaptureExecutionError<std::io::Error>| error.to_string(),
                        eredu_core::SpeculativeRequestId::new(9),
                        vec![SpeculativeCaptureScope::Prediction { depth: 0 }],
                        vec![
                            SpeculativeCaptureScope::Prediction { depth: 0 },
                            SpeculativeCaptureScope::Prediction { depth: 1 },
                        ],
                    )
                    .unwrap();
                    let saved = observer.checkpoint(&catalog).unwrap();
                    let mut results = Vec::new();
                    let mut epoch = DistributedCommitEpoch::FIRST;
                    let mut spent = CaptureUsage::default();
                    for (index, (phase, rows, keep, scale, abort)) in [
                        (Phase::PredictionPrefill, 5, true, true, false),
                        (Phase::Proposal { depth: 1 }, 1, false, true, false),
                        (Phase::Proposal { depth: 0 }, 1, true, false, false),
                        (Phase::PredictionReplay, 5, true, true, false),
                        (Phase::PredictionReplay, 5, true, true, true),
                        (Phase::PredictionReplay, 5, true, true, false),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if matches!(index, 3 | 5) {
                            observer.restore(&saved).unwrap();
                        }
                        observer.set_activation_origin(Some(SpeculativeActivationOrigin {
                            request: eredu_core::SpeculativeRequestId::new(9),
                            committed_tokens: 3,
                            prediction: 4,
                            prefix_digest: [7; 32],
                            optimistic: false,
                        }));
                        let result = with_speculative_activation(
                            Some(&mut observer),
                            phase,
                            rows as usize,
                            |observer| {
                                let observer = observer.unwrap();
                                observer.prepare_transaction(
                                    epoch,
                                    if phase == Phase::PredictionPrefill {
                                        crate::ExpertPass::Prefill
                                    } else {
                                        crate::ExpertPass::Decode
                                    },
                                )?;
                                observer.coordinate_transaction(epoch)?;
                                if let Some((_, map)) = layout.0.iter().find(|(r, _)| *r == rank) {
                                    let global = value(
                                        vec![rows, 20],
                                        (0..rows * 20).map(|i| i as f32 * 0.25 - 3.).collect(),
                                    );
                                    let input = local_value(&global, map);
                                    observer.observe("block.output", &input)?;
                                    let output = observer.intervene("block.output", &input)?;
                                    let mut expected = data(&global).to_vec();
                                    for (i, value) in expected.iter_mut().enumerate() {
                                        if keep && i < 20 && ![2, 9, 17].contains(&(i % 20)) {
                                            *value = 0.;
                                        }
                                        if scale {
                                            *value *= 2.;
                                        }
                                    }
                                    assert_eq!(
                                        data(output.as_ref().unwrap_or(&input)),
                                        data(&local_value(&value(vec![rows, 20], expected), map))
                                    );
                                    assert_eq!(data(&input), data(&local_value(&global, map)));
                                }
                                observer.complete_transaction(epoch)?;
                                observer.finish_transaction(epoch, !abort);
                                if abort {
                                    Err("final commit aborted".to_string())
                                } else {
                                    Ok(())
                                }
                            },
                        );
                        assert_eq!(result.is_ok(), !abort);
                        observer.set_activation_origin(None);
                        let record = observer.take_activation_capture().unwrap();
                        assert_eq!(record.invocation, index as u64);
                        assert_eq!(record.completed, !abort);
                        assert_eq!(record.captures.as_step().invocation.unwrap().sequence, rows);
                        assert_eq!(
                            record.captures.as_step().outcome,
                            if abort {
                                CaptureStepOutcome::Aborted
                            } else {
                                CaptureStepOutcome::Committed
                            }
                        );
                        assert_eq!(record.captures.as_step().records[0].payload.is_some(), keep && !abort);
                        for (operation, active) in
                            record.captures.as_step().interventions.iter().zip([keep, scale])
                        {
                            if !abort {
                                assert_eq!(
                                    operation.outcome,
                                    if active {
                                        InterventionOutcome::Applied
                                    } else {
                                        InterventionOutcome::Inactive
                                    }
                                );
                            }
                            assert!(operation
                                .evidence
                                .iter()
                                .all(|evidence| evidence.payload.is_some() == (active && !abort)));
                        }
                        assert!(record.captures.as_step().cumulative_usage.host_bytes > spent.host_bytes);
                        spent = record.captures.as_step().cumulative_usage;
                        results.push(record);
                        epoch = epoch.next().unwrap();
                    }
                    for index in [3, 5] {
                        assert_eq!(
                            results[0].captures.as_step().records[0].payload,
                            results[index].captures.as_step().records[0].payload
                        );
                    }
                    results
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    for rank in &results[1..] {
        for (actual, expected) in rank.iter().zip(&results[0]) {
            assert_eq!(actual.captures.as_step().records, expected.captures.as_step().records);
            assert_eq!(actual.captures.as_step().partitions, expected.captures.as_step().partitions);
            assert_eq!(
                actual.captures.as_step().cumulative_usage,
                expected.captures.as_step().cumulative_usage
            );
        }
    }
}
