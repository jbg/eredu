use super::*;
mod funding;
use eredu_core::{
    consensus::ConsensusTransport, BoundedCompletionOutcome, CompletionCancellationMode, Submission,
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar,
    },
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Short,
    Echo,
    Completion,
    Deadline,
    Resolve,
}
#[derive(Default)]
struct Frames {
    rounds: BTreeMap<usize, Vec<Option<Vec<u32>>>>,
    terminal: bool,
}
struct World {
    frames: Mutex<Frames>,
    changed: Condvar,
    size: usize,
}
struct Transport {
    rank: usize,
    world: Arc<World>,
    next: AtomicUsize,
    calls: AtomicUsize,
    resolves: AtomicUsize,
    fenced: AtomicBool,
    fault: Fault,
    setup: RefCell<Option<CommunicationSessionIdentity>>,
}
impl Transport {
    fn gather(&self, words: &[u32]) -> Result<Vec<u32>, std::io::Error> {
        let round = self.next.fetch_add(1, Ordering::SeqCst);
        let mut frames = self.world.frames.lock().unwrap();
        frames
            .rounds
            .entry(round)
            .or_insert_with(|| vec![None; self.world.size])[self.rank] = Some(words.to_vec());
        self.world.changed.notify_all();
        loop {
            if frames.terminal {
                return Err(std::io::Error::other("shared terminal transport"));
            }
            let row = &frames.rounds[&round];
            if row.iter().all(Option::is_some) {
                return Ok(row
                    .iter()
                    .flat_map(|v| v.as_ref().unwrap().iter().copied())
                    .collect());
            }
            let (next, timeout) = self
                .world
                .changed
                .wait_timeout(frames, Duration::from_secs(2))
                .unwrap();
            frames = next;
            if timeout.timed_out() {
                return Err(std::io::Error::other("test bounded collective timed out"));
            }
        }
    }
}
struct Done(Fault);
impl Completion for Done {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.0 != Fault::Deadline)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        panic!("unbounded completion is prohibited")
    }
}
impl BoundedCompletion for Done {
    fn supports_cancellation(mode: CompletionCancellationMode) -> bool {
        mode == CompletionCancellationMode::QuarantineUntilComplete
    }
    fn wait_bounded(
        self,
        wait: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        match self.0 {
            Fault::Completion => Err(std::io::Error::other(
                "original parameter completion failure",
            )),
            Fault::Deadline => Ok(BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: wait.cancellation(),
            }),
            _ => Ok(BoundedCompletionOutcome::Completed),
        }
    }
}
impl ConsensusTransport for Transport {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.world.size
    }
    fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
        assert_eq!(
            self.calls.load(Ordering::SeqCst),
            0,
            "only setup may use unbounded transport"
        );
        self.gather(words)
    }
}
impl BoundedConsensusTransport for Transport {
    type Completion = Done;
    type GatherOutput = (usize, Vec<u32>);
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Done>, Self::Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let mut gathered = self.gather(words)?;
        if call == 0 {
            match self.fault {
                Fault::Short => {
                    gathered.pop();
                }
                Fault::Echo => gathered[self.rank * words.len()] ^= 1,
                _ => {}
            }
        }
        Ok(Submission {
            output: (call, gathered),
            completion: Done(if call == 0 { self.fault } else { Fault::None }),
        })
    }
    fn resolve_all_gather_words(
        &self,
        (call, output): Self::GatherOutput,
    ) -> Result<Vec<u32>, Self::Error> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        if call == 0 && self.fault == Fault::Resolve {
            return Err(std::io::Error::other(
                "original parameter resolution failure",
            ));
        }
        Ok(output)
    }
}
impl ParameterOperationTransport for Transport {
    fn parameter_rank(&self) -> usize {
        self.rank
    }
    fn parameter_setup(&self) -> CommunicationSessionIdentity {
        self.setup.borrow().unwrap()
    }
    fn parameter_wait(&self) -> Result<BoundedCompletionWait, ParameterError> {
        Ok(BoundedCompletionWait::new(
            Duration::from_secs(2),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn estimate_parameter_gather(&self, words: usize) -> Result<CaptureUsage, ParameterError> {
        Ok(CaptureUsage {
            retained_bytes: 128 + words as u64 * self.world.size as u64 * 8,
            host_bytes: words as u64 * self.world.size as u64 * 4,
            ..Default::default()
        })
    }
    fn ensure_parameter_active(&self) -> Result<(), BackendFailure> {
        if self.fenced.load(Ordering::SeqCst) {
            Err(BackendFailure::from_error(std::io::Error::other(
                "fenced native owner",
            )))
        } else {
            Ok(())
        }
    }
    fn fail_parameter_operation(&self, _: &Error) {
        self.fenced.store(true, Ordering::SeqCst);
        self.world.frames.lock().unwrap().terminal = true;
        self.world.changed.notify_all();
    }
}
fn ranks(fault: Fault, run: impl Fn(usize, &Transport, &ParameterOperationCoordinator) + Sync) {
    let world = Arc::new(World {
        frames: Mutex::new(Frames::default()),
        changed: Condvar::new(),
        size: 3,
    });
    std::thread::scope(|scope| {
        for rank in 0..world.size {
            let world = Arc::clone(&world);
            let run = &run;
            scope.spawn(move || {
                let transport = Transport {
                    rank,
                    world,
                    next: AtomicUsize::new(0),
                    calls: AtomicUsize::new(0),
                    resolves: AtomicUsize::new(0),
                    fenced: AtomicBool::new(false),
                    fault: if rank == 1 { fault } else { Fault::None },
                    setup: RefCell::new(None),
                };
                let manifest = crate::CommunicationManifest::new(3, rank, vec![], vec![]).unwrap();
                let setup = crate::establish_communication_session(
                    &transport,
                    &manifest,
                    Some([rank as u8 + 1; 32]),
                )
                .unwrap()
                .identity();
                transport.setup.replace(Some(setup));
                let owner = ParameterOperationCoordinator::new(&transport).unwrap();
                run(rank, &transport, &owner);
            });
        }
    });
}
#[derive(Default)]
struct Budget {
    total: CaptureUsage,
    deny: bool,
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        if self.deny {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: true,
            });
        }
        self.total = self.total.checked_add(cost)?;
        Ok(None)
    }
}
fn binding() -> ParameterOperationBinding {
    ParameterOperationBinding::new("released-source", "retained-execution", "shared-branch", 3)
        .unwrap()
}
fn original(rank: usize) -> Vec<f32> {
    vec![0.5 + rank as f32, -2.25 - rank as f32]
}

#[test]
fn reads_agree_asymmetric_limits_failed_sources_and_delivery_before_retry() {
    ranks(Fault::None, |rank, transport, owner| {
        let mut budget = Budget::default();
        for failure in 0..5 {
            budget.deny = failure == 0 && rank == 1;
            let produced = RefCell::new(false);
            let result = owner.read(
                transport,
                binding(),
                &[19; 32],
                ParameterOperationKind::Query,
                if failure == 1 && rank == 1 {
                    Err(ParameterError::Invalid(
                        "local metadata preparation failed before a layout existed".into(),
                    ))
                } else {
                    Ok(ParameterReadPreparation::new((), [41; 32], 4))
                },
                &mut budget,
                |_| {
                    *produced.borrow_mut() = true;
                    if failure == 2 && rank == 1 {
                        return Err(ParameterError::Invalid("original producer failure".into()));
                    }
                    Ok(original(rank).into_iter().map(f32::to_bits).collect())
                },
                |rows| {
                    if failure == 3 && rank == 1 {
                        return Err(ParameterError::Invalid("original assembly failure".into()));
                    }
                    let actual = rows
                        .iter()
                        .flatten()
                        .copied()
                        .map(f32::from_bits)
                        .collect::<Vec<_>>();
                    assert_eq!(actual, (0..3).flat_map(original).collect::<Vec<_>>());
                    Ok(actual)
                },
            );
            if failure < 4 {
                assert!(completed_rejection(result.as_ref().unwrap_err()));
                if failure <= 1 {
                    assert!(!*produced.borrow());
                }
                assert!(!transport.fenced.load(Ordering::SeqCst));
            } else {
                assert_eq!(result.unwrap().len(), 6);
            }
        }
        assert_eq!(owner.usage().unwrap().attempts, 5);
        assert_eq!(
            owner.usage().unwrap().reserved,
            owner.per_operation().checked_mul(5).unwrap()
        );
        assert!(budget.total.host_bytes > 0);
    });
}

#[test]
fn transactions_prepare_every_rank_before_publication_and_restore_after_partial_failure() {
    let prepared_count = AtomicUsize::new(0);
    ranks(Fault::None, |rank, transport, owner| {
        let values = RefCell::new(original(rank));
        for failure in 0..3 {
            let published = RefCell::new(false);
            let result = owner.transaction(
                transport,
                binding(),
                &[23; 32],
                ParameterOperationKind::Activation,
                Ok(()),
                |_| {
                    if failure == 0 && rank == 1 {
                        return Err(ParameterError::Invalid(
                            "replacement preparation rejected".into(),
                        ));
                    }
                    prepared_count.fetch_add(1, Ordering::SeqCst);
                    Ok((values.borrow().clone(), vec![11.0 + rank as f32, -7.0]))
                },
                |(_, replacement)| {
                    assert!(prepared_count.load(Ordering::SeqCst) >= 5);
                    *published.borrow_mut() = true;
                    *values.borrow_mut() = replacement.clone();
                    if failure == 1 && rank == 1 {
                        Err(ParameterError::Invalid(
                            "publication rejected after local mutation".into(),
                        ))
                    } else {
                        Ok(())
                    }
                },
                |(original, _)| {
                    *values.borrow_mut() = original.clone();
                    Ok(())
                },
            );
            if failure < 2 {
                assert!(completed_rejection(result.as_ref().unwrap_err()));
                assert_eq!(*values.borrow(), original(rank));
                assert_eq!(*published.borrow(), failure == 1);
            } else {
                assert!(result.is_ok());
                assert_eq!(*values.borrow(), [11.0 + rank as f32, -7.0]);
            }
        }
        assert_eq!(owner.usage().unwrap().attempts, 3);
        assert!(!transport.fenced.load(Ordering::SeqCst));
    });
}

#[test]
fn failed_restoration_fences_all_owners_and_prevents_later_work() {
    ranks(Fault::None, |rank, transport, owner| {
        let result = owner.transaction(
            transport,
            binding(),
            &[29; 32],
            ParameterOperationKind::Removal,
            Ok(()),
            |_| Ok(()),
            |_| {
                if rank == 0 {
                    Err(ParameterError::Invalid("publish".into()))
                } else {
                    Ok(())
                }
            },
            |_| {
                if rank == 1 {
                    Err(ParameterError::Invalid("restore".into()))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert!(transport.fenced.load(Ordering::SeqCst));
        let before = transport.calls.load(Ordering::SeqCst);
        let mut budget = Budget::default();
        assert!(owner
            .read(
                transport,
                binding(),
                &[29; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [41; 32], 0)),
                &mut budget,
                |_| panic!("fenced producer"),
                |_| Ok(())
            )
            .is_err());
        assert_eq!(transport.calls.load(Ordering::SeqCst), before);
    });
}

#[test]
fn binding_intent_and_payload_bound_mismatch_fence_without_parameter_work() {
    for mismatch in 0..4 {
        ranks(Fault::None, |rank, transport, owner| {
            let binding = if mismatch == 0 && rank == 1 {
                ParameterOperationBinding::new(
                    "released-source",
                    "retained-execution",
                    "foreign-branch",
                    3,
                )
                .unwrap()
            } else {
                binding()
            };
            let intent = if mismatch == 1 && rank == 1 {
                [31; 32]
            } else {
                [32; 32]
            };
            let width = if mismatch == 2 && rank == 1 { 2 } else { 3 };
            assert!(owner
                .read(
                    transport,
                    binding,
                    &intent,
                    ParameterOperationKind::Query,
                    Ok(ParameterReadPreparation::new(
                        (),
                        if mismatch == 3 && rank == 1 {
                            [99; 32]
                        } else {
                            [41; 32]
                        },
                        width
                    )),
                    &mut Budget::default(),
                    |_| panic!("mismatched operation cannot produce"),
                    |_| Ok(())
                )
                .is_err());
            assert!(transport.fenced.load(Ordering::SeqCst));
        });
    }
}

#[test]
fn loaded_models_have_distinct_monotone_identity_and_mixed_models_cannot_submit() {
    ranks(Fault::None, |rank, transport, owner| {
        let first = owner.register_model().unwrap();
        let second = owner.register_model().unwrap();
        assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
        assert_eq!(owner.usage().unwrap().attempts, 0);
        let first_binding = first.binding("source", "execution", 0).unwrap();
        let second_binding = second.binding("source", "execution", 0).unwrap();
        assert_ne!(first_binding, second_binding);
        assert_ne!(
            first_binding,
            first.binding("source", "execution", 1).unwrap()
        );
        assert_ne!(
            first_binding,
            first.binding("other-source", "execution", 0).unwrap()
        );
        // Corresponding model instances agree across ranks, even when all loaded
        // artifact/execution/version facts are otherwise identical.
        for binding in [first_binding, second_binding, first_binding] {
            owner
                .read(
                    transport,
                    binding,
                    &[71; 32],
                    ParameterOperationKind::Query,
                    Ok(ParameterReadPreparation::new((), [72; 32], 1)),
                    &mut Budget::default(),
                    |_| Ok(vec![rank as u32]),
                    |rows| {
                        assert_eq!(rows, [vec![0], vec![1], vec![2]]);
                        Ok(())
                    },
                )
                .unwrap();
        }
        assert_eq!(owner.usage().unwrap().attempts, 3);
        drop(second);
        let third = owner.register_model().unwrap();
        assert_ne!(
            third.binding("source", "execution", 0).unwrap(),
            second_binding
        );
        assert!(owner
            .read(
                transport,
                if rank == 1 {
                    second_binding
                } else {
                    first_binding
                },
                &[71; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [72; 32], 1)),
                &mut Budget::default(),
                |_| panic!("mixed loaded models cannot read"),
                |_| Ok(()),
            )
            .is_err());
        assert!(transport.fenced.load(Ordering::SeqCst));
        assert!(owner.register_model().is_err());
    });
}

#[test]
fn malformed_frames_completion_and_resolution_errors_never_release_results() {
    for fault in [
        Fault::Short,
        Fault::Echo,
        Fault::Completion,
        Fault::Deadline,
        Fault::Resolve,
    ] {
        ranks(fault, |rank, transport, owner| {
            let result = owner.read(
                transport,
                binding(),
                &[37; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [41; 32], 4)),
                &mut Budget::default(),
                |_| panic!("failed admission cannot produce"),
                |_| Ok(()),
            );
            assert!(result.is_err(), "{fault:?}");
            assert!(transport.fenced.load(Ordering::SeqCst));
            assert_eq!(owner.usage().unwrap().attempts, 1);
            if rank == 1 && matches!(fault, Fault::Completion | Fault::Deadline) {
                assert_eq!(transport.resolves.load(Ordering::SeqCst), 0);
            }
        });
    }
}

#[test]
fn abandoned_or_reentered_operation_fences_shared_work_without_refunding_attempts() {
    for reenter in [false, true] {
        ranks(Fault::None, |rank, transport, owner| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                owner.read(
                    transport,
                    binding(),
                    &[47; 32],
                    ParameterOperationKind::Query,
                    Ok(ParameterReadPreparation::new((), [48; 32], 2)),
                    &mut Budget::default(),
                    |_| {
                        if rank == 1 {
                            if !reenter {
                                panic!("injected preparation unwind");
                            }
                            owner.read(
                                transport,
                                binding(),
                                &[49; 32],
                                ParameterOperationKind::Query,
                                Ok(ParameterReadPreparation::new((), [50; 32], 0)),
                                &mut Budget::default(),
                                |_| panic!("reentered source"),
                                |_| Ok(()),
                            )?;
                        }
                        Ok(vec![1, 2])
                    },
                    |_| -> Result<(), ParameterError> {
                        panic!("abandoned operation cannot deliver")
                    },
                )
            }));
            assert!(result.is_err() || result.unwrap().is_err());
            assert!(transport.fenced.load(Ordering::SeqCst));
            // An unwind poisons authority, while its charged work remains visible.
            let usage = owner.usage().unwrap();
            assert_eq!(usage.attempts, 1);
            assert_eq!(usage.reserved, owner.per_operation());
            assert!(
                matches!(owner.register_model(), Err(ParameterError::Coordination(error))
                if matches!(*error, Error::Fenced))
            );
            let calls = transport.calls.load(Ordering::SeqCst);
            let retry = owner.read(
                transport,
                binding(),
                &[47; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [48; 32], 0)),
                &mut Budget::default(),
                |_| panic!("terminal owner cannot retry"),
                |_| Ok(()),
            );
            assert!(matches!(retry, Err(ParameterError::Coordination(error))
                if matches!(*error, Error::Fenced)));
            assert_eq!(owner.usage().unwrap(), usage);
            assert_eq!(transport.calls.load(Ordering::SeqCst), calls);
        });
    }
}

#[test]
fn catalogue_binding_reports_failed_hashing_but_cannot_authorize_values_or_edits() {
    ranks(Fault::None, |rank, transport, owner| {
        let model = owner.register_model().unwrap();
        let binding = model.catalog_binding(0);
        let local = if rank == 1 {
            Err(ParameterError::Invalid(
                "injected source identity failure".into(),
            ))
        } else {
            Ok(ParameterReadPreparation::new((), [0x62; 32], 1))
        };
        assert!(owner
            .read(
                transport,
                binding,
                &[0x63; 32],
                ParameterOperationKind::Catalog,
                local,
                &mut Budget::default(),
                |_| panic!("failed metadata cannot produce"),
                |_| Ok(())
            )
            .is_err());
        assert!(owner
            .read(
                transport,
                binding,
                &[0x64; 32],
                ParameterOperationKind::Query,
                Ok(ParameterReadPreparation::new((), [0x62; 32], 1)),
                &mut Budget::default(),
                |_| panic!("catalogue binding cannot read weights"),
                |_| Ok(())
            )
            .is_err());
        assert!(owner
            .transaction(
                transport,
                binding,
                &[0x65; 32],
                ParameterOperationKind::Activation,
                Ok(()),
                |_| -> Result<(), ParameterError> {
                    panic!("catalogue binding cannot prepare edits")
                },
                |_| Ok(()),
                |_| Ok(())
            )
            .is_err());
        owner
            .read(
                transport,
                binding,
                &[0x63; 32],
                ParameterOperationKind::Catalog,
                Ok(ParameterReadPreparation::new((), [0x62; 32], 1)),
                &mut Budget::default(),
                |_| Ok(vec![rank as u32]),
                |rows| {
                    assert_eq!(rows, [vec![0], vec![1], vec![2]]);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(owner.usage().unwrap().attempts, 4);
        assert!(!transport.fenced.load(Ordering::SeqCst));
    });
}
