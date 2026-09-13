use super::*;
use eredu_core::{consensus::ConsensusTransport, BoundedCompletionOutcome, Submission};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Barrier,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Decode,
    Short,
    Decision,
    Wait,
    Deadline,
    Submit,
    CandidateFlags,
}
struct World {
    frames: Mutex<Vec<Vec<u32>>>,
    barrier: Barrier,
}
struct Transport {
    rank: usize,
    world: Arc<World>,
    calls: AtomicUsize,
    resolves: AtomicUsize,
    fenced: AtomicBool,
    fault: Fault,
}
impl Transport {
    fn gather(&self, words: &[u32]) -> Vec<u32> {
        self.world.frames.lock().unwrap()[self.rank] = words.to_vec();
        self.world.barrier.wait();
        let result = self
            .world
            .frames
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        self.world.barrier.wait();
        result
    }
}
struct Done(Fault);
impl Completion for Done {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        panic!("only bounded waiting is admitted")
    }
}
impl BoundedCompletion for Done {
    fn supports_cancellation(mode: CompletionCancellationMode) -> bool {
        mode == CompletionCancellationMode::QuarantineUntilComplete
    }
    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        match self.0 {
            Fault::Wait => Err(std::io::Error::other("original completion cause")),
            Fault::Deadline => Ok(BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: policy.cancellation(),
            }),
            _ => Ok(BoundedCompletionOutcome::Completed),
        }
    }
}
impl ConsensusTransport for Transport {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        2
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        assert_eq!(
            self.calls.load(Ordering::SeqCst),
            0,
            "unbounded transport is only used by setup"
        );
        Ok(self.gather(words))
    }
}
impl BoundedConsensusTransport for Transport {
    type Completion = Done;
    type GatherOutput = (usize, Vec<u32>);
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Done>, Self::Error> {
        assert!(!self.fenced.load(Ordering::SeqCst));
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut gathered = self.gather(words);
        if self.fault == Fault::Submit && call == 0 {
            return Err(std::io::Error::other("original submission cause"));
        }
        if self.fault == Fault::Short && call == 0 {
            gathered.pop();
        }
        if self.fault == Fault::Decision && call == 0 {
            // Corrupt only the peer disposition: local echo still matches.
            gathered[(1 - self.rank) * TEXT_PREPARATION_WORDS + 7] = 1;
        }
        if self.fault == Fault::CandidateFlags && call == 2 {
            gathered[(1 - self.rank) * TEXT_PREPARATION_WORDS + 9] |= 1 << 31;
        }
        Ok(Submission {
            output: (call, gathered),
            completion: Done(if call == 0 { self.fault } else { Fault::None }),
        })
    }
    fn resolve_all_gather_words(
        &self,
        (call, gathered): Self::GatherOutput,
    ) -> Result<Vec<u32>, Self::Error> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        if self.fault == Fault::Decode && call == 0 {
            return Err(std::io::Error::other("original resolution cause"));
        }
        Ok(gathered)
    }
}
impl TextPreparationTransport for Transport {
    fn preparation_rank(&self) -> usize {
        self.rank
    }
    fn ensure_preparation_active(&self) -> Result<(), BackendFailure> {
        if self.fenced.load(Ordering::SeqCst) {
            Err(BackendFailure::from_error(std::io::Error::other(
                "fenced native owner",
            )))
        } else {
            Ok(())
        }
    }
    fn fail_preparation(&self, _: &TextPreparationAgreementError) {
        self.fenced.store(true, Ordering::SeqCst);
    }
}

fn ranks<R: Send>(
    fault: Fault,
    run: impl Fn(usize, &Transport, &TextPreparationCoordinator) -> R + Sync,
) -> Vec<R> {
    let world = Arc::new(World {
        frames: Mutex::new(vec![vec![], vec![]]),
        barrier: Barrier::new(2),
    });
    std::thread::scope(|scope| {
        (0..2)
            .map(|rank| {
                let world = Arc::clone(&world);
                let run = &run;
                scope.spawn(move || {
                    let transport = Transport {
                        rank,
                        world,
                        calls: AtomicUsize::new(0),
                        resolves: AtomicUsize::new(0),
                        fenced: AtomicBool::new(false),
                        fault: if rank == 1 { fault } else { Fault::None },
                    };
                    let manifest =
                        crate::CommunicationManifest::new(2, rank, vec![], vec![]).unwrap();
                    let identity = crate::establish_communication_session(
                        &transport,
                        &manifest,
                        Some([rank as u8 + 1; 32]),
                    )
                    .unwrap()
                    .identity();
                    let wait = BoundedCompletionWait::new(
                        std::time::Duration::from_secs(2),
                        CompletionCancellationMode::QuarantineUntilComplete,
                    )
                    .unwrap();
                    let owner = TextPreparationCoordinator::new(
                        identity,
                        TextPreparationUsage {
                            attempts: 0,
                            retained_bytes: 1024,
                            host_bytes: 256,
                        },
                        wait,
                    )
                    .unwrap();
                    run(rank, &transport, &owner)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}

#[test]
fn cold_failure_and_cancellation_are_agreed_before_corrected_retry() {
    ranks(Fault::None, |rank, transport, owner| {
        for stage in [
            TextPreparationStage::Request,
            TextPreparationStage::Prompt,
            TextPreparationStage::Sampling,
            TextPreparationStage::Instrumentation,
            TextPreparationStage::Delivery,
        ] {
            let failed = owner
                .agree(
                    transport,
                    stage,
                    if rank == 1 {
                        TextPreparationStatus::Failed
                    } else {
                        TextPreparationStatus::Ready
                    },
                )
                .unwrap();
            assert_eq!(failed, TextPreparationOutcome::Rejected { rank: 1 });
            let cancelled = owner
                .agree(
                    transport,
                    stage,
                    if rank == 0 {
                        TextPreparationStatus::Cancelled
                    } else {
                        TextPreparationStatus::Ready
                    },
                )
                .unwrap();
            assert_eq!(cancelled, TextPreparationOutcome::Cancelled);
            assert_eq!(
                owner
                    .agree(transport, stage, TextPreparationStatus::Ready)
                    .unwrap(),
                TextPreparationOutcome::Ready
            );
        }
        let usage = owner.usage().unwrap();
        assert_eq!(usage.attempts, 15);
        assert_eq!(
            usage.retained_bytes,
            15 * owner.per_attempt().retained_bytes
        );
        assert_eq!(usage.host_bytes, 15 * owner.per_attempt().host_bytes);
        assert_eq!(transport.calls.load(Ordering::SeqCst), 30);
        assert!(!transport.fenced.load(Ordering::SeqCst));
    });
}

#[test]
fn failure_takes_precedence_over_peer_cancellation() {
    ranks(Fault::None, |rank, transport, owner| {
        assert_eq!(
            owner
                .agree(
                    transport,
                    TextPreparationStage::Request,
                    if rank == 0 {
                        TextPreparationStatus::Cancelled
                    } else {
                        TextPreparationStatus::Failed
                    }
                )
                .unwrap(),
            TextPreparationOutcome::Rejected { rank: 1 }
        );
    });
}

#[test]
fn stage_and_attempt_mismatch_fence_without_advancement() {
    for attempt in [false, true] {
        ranks(Fault::None, |rank, transport, owner| {
            if attempt && rank == 1 {
                owner.state.lock().unwrap().usage.attempts = 1;
            }
            let stage = if !attempt && rank == 1 {
                TextPreparationStage::Sampling
            } else {
                TextPreparationStage::Prompt
            };
            assert!(matches!(
                owner.agree(transport, stage, TextPreparationStatus::Ready),
                Err(TextPreparationAgreementError::Protocol(_))
            ));
            assert!(transport.fenced.load(Ordering::SeqCst));
            let calls = transport.calls.load(Ordering::SeqCst);
            assert!(matches!(
                owner.agree(transport, stage, TextPreparationStatus::Ready),
                Err(TextPreparationAgreementError::Fenced)
            ));
            assert_eq!(transport.calls.load(Ordering::SeqCst), calls);
        });
    }
}

#[test]
fn local_decode_or_disposition_disagreement_cannot_release_ready_peers() {
    for fault in [
        Fault::Decode,
        Fault::Short,
        Fault::Decision,
        Fault::Wait,
        Fault::Deadline,
        Fault::Submit,
    ] {
        ranks(fault, |rank, transport, owner| {
            let error = owner
                .agree(
                    transport,
                    TextPreparationStage::Prompt,
                    TextPreparationStatus::Ready,
                )
                .unwrap_err();
            assert!(
                transport.fenced.load(Ordering::SeqCst),
                "{fault:?}, {error}"
            );
            assert_eq!(owner.usage().unwrap(), owner.per_attempt());
            assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
            if rank == 1 && matches!(fault, Fault::Decode | Fault::Wait | Fault::Submit) {
                let mut cause: &dyn std::error::Error = &error;
                while let Some(source) = cause.source() {
                    cause = source;
                }
                assert!(cause.to_string().starts_with("original "));
            }
        });
    }
}

#[test]
fn exhausted_accounting_and_unsupported_completion_submit_nothing() {
    for unsupported in [false, true] {
        ranks(Fault::None, |_, transport, owner| {
            let alternate;
            let owner = if unsupported {
                alternate = TextPreparationCoordinator::new(
                    owner.identity,
                    TextPreparationUsage::default(),
                    BoundedCompletionWait::new(
                        std::time::Duration::from_secs(1),
                        CompletionCancellationMode::NativeCancel,
                    )
                    .unwrap(),
                )
                .unwrap();
                &alternate
            } else {
                owner.state.lock().unwrap().usage.attempts = u64::MAX;
                owner
            };
            let error = owner
                .agree(
                    transport,
                    TextPreparationStage::Request,
                    TextPreparationStatus::Ready,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                TextPreparationAgreementError::Overflow
                    | TextPreparationAgreementError::Admission(_)
            ));
            assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
            assert_eq!(transport.resolves.load(Ordering::SeqCst), 0);
            assert!(transport.fenced.load(Ordering::SeqCst));
        });
    }
}

#[test]
fn repeated_setup_identity_cannot_attach_to_another_preparation_owner() {
    ranks(Fault::None, |rank, transport, owner| {
        let manifest = crate::CommunicationManifest::new(2, rank, vec![], vec![]).unwrap();
        let other = crate::establish_communication_session(
            transport,
            &manifest,
            Some([rank as u8 + 17; 32]),
        )
        .unwrap()
        .identity();
        assert_ne!(other, owner.identity);
        let alternate =
            TextPreparationCoordinator::new(other, TextPreparationUsage::default(), owner.wait)
                .unwrap();
        let selected = if rank == 0 { owner } else { &alternate };
        assert!(matches!(
            selected.agree(
                transport,
                TextPreparationStage::Request,
                TextPreparationStatus::Ready
            ),
            Err(TextPreparationAgreementError::Protocol(_))
        ));
        assert!(transport.fenced.load(Ordering::SeqCst));
    });
}

fn schedule_state(index: usize) -> eredu_core::SpeculativeScheduleState {
    eredu_core::SpeculativeScheduleState {
        request: eredu_core::SpeculativeRequestId::new(index),
        status: eredu_core::SpeculativeRequestStatus::TargetVerificationInFlight,
        cancellation_requested: false,
        verification_complete: false,
        verification_deadline_expired: false,
        optimistic_eligible: false,
    }
}

#[test]
fn speculative_completion_and_cancellation_resolve_per_lane_before_advancement() {
    ranks(Fault::None, |rank, transport, owner| {
        let mut first = schedule_state(0);
        first.verification_complete = rank == 0;
        first.cancellation_requested = rank == 1;
        first.verification_deadline_expired = rank == 0;
        first.optimistic_eligible = rank == 1;
        let mut second = schedule_state(1);
        second.verification_complete = true;
        second.optimistic_eligible = true;
        let resolved = owner
            .coordinate_speculative_step(transport, vec![first, second])
            .unwrap();
        assert!(!resolved[0].verification_complete);
        assert!(resolved[0].cancellation_requested);
        assert!(resolved[0].verification_deadline_expired);
        assert!(!resolved[0].optimistic_eligible);
        assert!(resolved[1].verification_complete);
        assert!(resolved[1].optimistic_eligible);
        assert!(!resolved[1].cancellation_requested);
        assert!(!resolved[1].verification_deadline_expired);
        assert_eq!(owner.usage().unwrap().attempts, 3);
        let mut ready = schedule_state(0);
        ready.verification_complete = true;
        assert!(
            owner
                .coordinate_speculative_step(transport, vec![ready])
                .unwrap()[0]
                .verification_complete
        );
        assert_eq!(owner.usage().unwrap().attempts, 5);
        owner
            .agree(
                transport,
                TextPreparationStage::Delivery,
                TextPreparationStatus::Ready,
            )
            .unwrap();
        assert_eq!(owner.usage().unwrap().attempts, 6);
        assert_eq!(
            owner.usage().unwrap().retained_bytes,
            6 * owner.per_attempt().retained_bytes
        );
        assert_eq!(
            owner.usage().unwrap().host_bytes,
            6 * owner.per_attempt().host_bytes
        );
    });
}

#[test]
fn speculative_table_and_lifecycle_mismatches_fence_all_participants() {
    for mismatch in 0..3 {
        ranks(Fault::None, |rank, transport, owner| {
            let mut states = vec![schedule_state(0), schedule_state(1)];
            if rank == 1 {
                match mismatch {
                    0 => {
                        states.pop();
                    }
                    1 => states[0].status = eredu_core::SpeculativeRequestStatus::ReadyToDraft,
                    _ => states[0].request = eredu_core::SpeculativeRequestId::new(7),
                }
            }
            assert!(owner
                .coordinate_speculative_step(transport, states)
                .is_err());
            assert!(transport.fenced.load(Ordering::SeqCst));
            assert!(matches!(
                owner.coordinate_speculative_step(transport, vec![]),
                Err(TextPreparationAgreementError::Fenced)
            ));
        });
    }
}

#[test]
fn speculative_scheduler_exchange_failures_are_confirmed_before_return() {
    for fault in [
        Fault::Decode,
        Fault::Short,
        Fault::Wait,
        Fault::Deadline,
        Fault::Submit,
        Fault::CandidateFlags,
    ] {
        ranks(fault, |_, transport, owner| {
            assert!(owner
                .coordinate_speculative_step(transport, vec![schedule_state(0)])
                .is_err());
            assert!(transport.fenced.load(Ordering::SeqCst));
            assert_eq!(
                transport.calls.load(Ordering::SeqCst),
                if fault == Fault::CandidateFlags { 4 } else { 2 }
            );
            assert_eq!(
                owner.usage().unwrap().attempts,
                if fault == Fault::CandidateFlags { 2 } else { 1 }
            );
        });
    }
}
