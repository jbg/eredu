use super::*;
use crate::capture::{
    CaptureBackendProvider, PartitionCaptureBackendProvider, SpeculativeCaptureObserver,
};
use crate::inspection::{with_speculative_activation, SpeculativeActivationObserver};
use eredu_core::speculative::{
    SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativeCaptureScope,
    SpeculativeControlError,
};

struct Provider {
    fail: bool,
}
impl CaptureBackendProvider for Provider {
    type Tensor = Value;
    type Error = std::io::Error;
    type Backend<'a> = Backend;
    fn backend(&mut self) -> Backend {
        Backend {
            fail: self.fail,
            ..Default::default()
        }
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

#[test]
fn speculative_partition_provider_retains_sparse_edits_and_native_failures() {
    let mut spent = None;
    for fault in [None, Some("native"), Some("missing")] {
        let all = world(9);
        let active = world(8);
        let setup = world(9);
        let outputs = std::thread::scope(|scope| {
            (0..9)
                .map(|rank| {
                    let all = Arc::clone(&all);
                    let active = Arc::clone(&active);
                    let setup = Arc::clone(&setup);
                    scope.spawn(move || {
                        let identity = crate::establish_communication_session(
                            &Bootstrap(transport(setup, rank, Fault::None)),
                            &crate::CommunicationManifest::new(9, rank, vec![], vec![]).unwrap(),
                            Some([rank as u8 + 1; 32]),
                        )
                        .unwrap()
                        .identity();
                        let identity = PartitionCaptureIdentity::for_session(
                            "artifact-exact".into(),
                            "prediction-execution".into(),
                            identity,
                            None,
                        )
                        .unwrap();
                        let transport = Arc::new(HookTransport {
                            transport: transport(all, rank, Fault::None),
                            hook: (rank < 8).then(|| transport(active, rank, Fault::None)),
                            members: (0..8).collect(),
                        });
                        let layout = Arc::new(SparseLayout::new());
                        let bounds = CaptureInvocationBounds {
                            batch: 1,
                            max_sequence: 3,
                            max_context: None,
                            max_predictions: 1,
                        };
                        let (capture, plan) = admission_at(rank, false, Some(bounds));
                        let mut session = CaptureSession::new(capture);
                        session
                            .enable_interventions(plan, Arc::new(Estimates))
                            .unwrap();
                        let provider = PartitionCaptureBackendProvider::new(
                            Provider {
                                fail: fault == Some("native") && rank == 2,
                            },
                            Arc::clone(&transport),
                            Arc::clone(&layout),
                            identity,
                            PartitionCaptureReceiptLimits {
                                max_producers: 9,
                                ..LIMITS
                            },
                            estimate,
                            |error: PartitionCaptureObserverError<std::io::Error>| match error {
                                PartitionCaptureObserverError::Capture(error) => error,
                                error => {
                                    CaptureExecutionError::Backend(std::io::Error::other(error))
                                }
                            },
                        );
                        let mut collector = SpeculativeCaptureObserver::new(
                            session,
                            provider,
                            |error: &CaptureExecutionError<std::io::Error>| error.to_string(),
                            eredu_core::SpeculativeRequestId::new(7),
                            vec![],
                            vec![SpeculativeCaptureScope::Prediction { depth: 0 }; 2],
                        )
                        .unwrap();
                        collector.set_activation_origin(Some(SpeculativeActivationOrigin {
                            request: eredu_core::SpeculativeRequestId::new(7),
                            committed_tokens: 0,
                            prediction: 0,
                            prefix_digest: [7; 32],
                            optimistic: false,
                        }));
                        let result = with_speculative_activation(
                            Some(&mut collector),
                            SpeculativeActivationPhase::PredictionPrefill,
                            3,
                            |observer| {
                                let observer = observer.unwrap();
                                let epoch = DistributedCommitEpoch::FIRST;
                                observer.prepare_transaction(epoch, crate::ExpertPass::Prefill)?;
                                observer.coordinate_transaction(epoch)?;
                                let local = if rank < 8 {
                                    invoke(
                                        observer.routed_unit_observer("experts")?.unwrap(),
                                        &layout.0[rank],
                                        false,
                                        if rank == 2 { fault } else { None },
                                    )
                                    .map_err(|error| error.to_string())
                                } else {
                                    Ok(())
                                };
                                assert_eq!(local.is_ok(), rank == 8 || fault.is_none());
                                let delivery = observer.complete_transaction(epoch);
                                observer.finish_transaction(epoch, fault.is_none());
                                local.and(delivery)
                            },
                        );
                        assert_eq!(result.is_ok(), fault.is_none());
                        let failure = collector.take_activation_error();
                        assert_eq!(failure.is_some(), fault.is_some());
                        if rank == 2 && fault == Some("native") {
                            use std::error::Error as _;
                            let Some(SpeculativeControlError::Backend(error)) = failure else {
                                panic!("native sparse source lost")
                            };
                            let mut source = error.source();
                            let mut original = false;
                            while let Some(error) = source {
                                original |= error.is::<std::io::Error>();
                                source = error.source();
                            }
                            assert!(
                                original,
                                "the whole borrowed provider must retain the native cause"
                            );
                        }
                        let record = collector.take_activation_capture().unwrap();
                        assert_eq!(record.completed, fault.is_none());
                        assert_eq!(
                            record.captures.as_step().invocation,
                            Some(CaptureInvocationShape {
                                batch: 1,
                                sequence: 3,
                                context: None
                            })
                        );
                        assert!(
                            record.captures.as_step().records.is_empty()
                                && record.captures.as_step().partitions.is_empty()
                        );
                        for (index, operation) in record.captures.as_step().interventions.iter().enumerate() {
                            if fault.is_none() {
                                assert_eq!(operation.outcome, InterventionOutcome::Applied);
                                let receipt = operation.routed_units.unwrap();
                                assert_eq!(
                                    (receipt.source_tokens, receipt.completed_tokens),
                                    (3, 3)
                                );
                                assert_eq!(
                                    receipt.affected_values,
                                    if index == 0 { 8 } else { 28 }
                                );
                            } else {
                                assert!(!matches!(
                                    operation.outcome,
                                    InterventionOutcome::Applied | InterventionOutcome::Unmatched
                                ));
                            }
                        }
                        record.captures.as_step().cumulative_usage
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(outputs.iter().all(|usage| *usage == outputs[0]));
        if let Some(spent) = spent {
            assert_eq!(outputs[0], spent, "failure does not refund sparse work");
        } else {
            spent = Some(outputs[0]);
        }
    }
}
