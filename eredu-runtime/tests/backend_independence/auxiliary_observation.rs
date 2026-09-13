use super::*;
use eredu_runtime::ActivationObserver;

thread_local! {
    static CHECKPOINT_FAILURE_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn checkpoint_failure() -> bool {
    CHECKPOINT_FAILURE_AFTER.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            true
        }
        Some(count) => {
            remaining.set(Some(count - 1));
            false
        }
        None => false,
    })
}

struct AuxiliaryOperation(bool);
impl
    eredu_runtime::PredictionTargetOperation<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for AuxiliaryOperation
{
    type Output = i32;
    fn apply(
        self,
        _: &mut OrdinaryTextFixture,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: Option<&()>,
        _: &(),
    ) -> Result<i32, Error> {
        state.as_mut()[0].0 += 10;
        if self.0 {
            Err(Error::backend("second prediction operation failed"))
        } else {
            Ok(state.as_ref()[0].0)
        }
    }
}

#[derive(Default)]
struct Work {
    calls: usize,
    completions: usize,
}

fn invoke(
    session: &mut ReferencePartitionedSession,
    observer: &mut TransactionProbe,
    work: &mut Work,
    fail_execution: bool,
    fail_completion: bool,
) -> Result<Vec<i32>, Error> {
    session.with_prediction_observation(
        eredu_runtime::ExpertPass::Decode,
        &(),
        observer,
        work,
        |session, work, observer| {
            let mut outputs = Vec::new();
            for ordinal in 0..2 {
                work.calls += 1;
                let value = session
                    .apply_prediction_target_operation(
                        AuxiliaryOperation(fail_execution && ordinal == 1),
                        &(),
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                observer.observe("prediction.unit", &FakeTensor(vec![value]))?;
                outputs.push(value);
            }
            Ok(outputs)
        },
        |_, work, outputs| {
            assert_eq!(outputs, &[10, 20]);
            work.completions += 1;
            PARTITION_COMMIT_TRACE.with(|trace| {
                trace
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .push("complete");
            });
            if fail_completion {
                Err(Error::backend_source(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "auxiliary completion sentinel",
                )))
            } else {
                Ok(())
            }
        },
        |error| Error::backend(error.to_string()),
    )
}

#[test]
fn auxiliary_phase_spans_all_depths_and_agrees_failures_before_publication() {
    for fault in [
        "none",
        "preparation",
        "coordination",
        "checkpoint",
        "second_checkpoint",
        "execution",
        "completion",
        "delivery",
        "participation",
    ] {
        let coordinator = Arc::new(ConcurrentCheckpointCoordinator::new());
        let results = std::thread::scope(|scope| {
            (0..2)
                .map(|rank| {
                    let coordinator = Arc::clone(&coordinator);
                    scope.spawn(move || {
                        PARTITION_COMMIT_TRACE.with(|slot| {
                            *slot.borrow_mut() = Some(Rc::new(RefCell::new(Vec::new())));
                        });
                        let (mut session, _, _) = partitioned_agreement_session(
                            rank,
                            if rank == 0 && fault == "checkpoint" {
                                LocalPartitionFailure::Checkpoint
                            } else {
                                LocalPartitionFailure::None
                            },
                            TestPhaseAgreement::Concurrent(ConcurrentCheckpointAgreement {
                                rank,
                                coordinator,
                            }),
                        );
                        let mut observer = TransactionProbe {
                            active: !(rank == 1 && fault == "participation"),
                            fault: if rank == 0 {
                                match fault {
                                    "preparation" => ObservationFault::Preparation,
                                    "coordination" => ObservationFault::Coordination,
                                    "delivery" => ObservationFault::Delivery,
                                    _ => ObservationFault::None,
                                }
                            } else {
                                ObservationFault::None
                            },
                            events: Vec::new(),
                        };
                        if rank == 0 && fault == "second_checkpoint" {
                            CHECKPOINT_FAILURE_AFTER.with(|remaining| remaining.set(Some(2)));
                        }
                        let mut work = Work::default();
                        let result = invoke(
                            &mut session,
                            &mut observer,
                            &mut work,
                            rank == 0 && fault == "execution",
                            rank == 0 && fault == "completion",
                        );
                        if rank == 0 && fault == "completion" {
                            let mut cause: Option<&(dyn std::error::Error + 'static)> =
                                Some(result.as_ref().unwrap_err());
                            let mut retained = false;
                            while let Some(error) = cause {
                                retained |=
                                    error.downcast_ref::<std::io::Error>().is_some_and(|error| {
                                        error.kind() == std::io::ErrorKind::PermissionDenied
                                    });
                                cause = error.source();
                            }
                            assert!(retained, "completion must retain its original cause");
                        }
                        let state = session.report().unwrap().state_report().to_vec();
                        let commit = session.report().unwrap().distributed_commit();
                        PARTITION_COMMIT_TRACE.with(|slot| *slot.borrow_mut() = None);
                        (result.is_ok(), state, commit, work, observer.events)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        let calls = coordinator.calls.lock().unwrap();
        let phases = |rank| {
            calls
                .iter()
                .filter_map(|(owner, phase, _)| (*owner == rank).then_some(*phase))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            phases(0),
            phases(1),
            "every rank follows the same protocol: {fault}"
        );
        for (rank, (success, state, commit, work, events)) in results.iter().enumerate() {
            assert_eq!(*success, fault == "none", "fault={fault}, rank={rank}");
            assert_eq!(state, if *success { &[20] } else { &[0] });
            assert_eq!(commit.unwrap().epoch().value(), 1);
            assert_eq!(
                matches!(
                    commit.unwrap(),
                    eredu_core::DistributedCommitOutcome::Committed { .. }
                ),
                *success
            );
            let before_work = matches!(
                fault,
                "preparation" | "coordination" | "checkpoint" | "participation"
            );
            assert_eq!(work.calls, if before_work { 0 } else { 2 });
            assert_eq!(
                work.completions,
                usize::from(!before_work && !matches!(fault, "execution" | "second_checkpoint"))
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|(event, _)| *event == "delivery")
                    .count(),
                usize::from(matches!(fault, "none" | "delivery"))
            );
            if fault != "participation" || rank == 0 {
                assert_eq!(
                    events.last(),
                    Some(&(if *success { "committed" } else { "discarded" }, 1))
                );
            }
        }
    }
}

#[test]
fn auxiliary_delivery_obeys_aborted_and_indeterminate_final_decisions() {
    for decision in [
        FinalDecisionResult::Aborted,
        FinalDecisionResult::Indeterminate(DistributedCommitPhase::DecisionCompletion),
    ] {
        PARTITION_COMMIT_TRACE.with(|slot| {
            *slot.borrow_mut() = Some(Rc::new(RefCell::new(Vec::new())));
        });
        let (mut session, _, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Final {
                result: decision,
                commits: Rc::new(Cell::new(0)),
            },
        );
        let mut observer = TransactionProbe {
            active: true,
            fault: ObservationFault::None,
            events: Vec::new(),
        };
        let mut work = Work::default();
        assert!(invoke(&mut session, &mut observer, &mut work, false, false).is_err());
        assert_eq!(
            observer.events,
            [
                ("prepare", 1),
                ("coordinate", 1),
                ("delivery", 1),
                ("discarded", 1)
            ]
        );
        let uncertain = matches!(decision, FinalDecisionResult::Indeterminate(_));
        assert_eq!(
            session.report().unwrap().state_report(),
            if uncertain { &[20] } else { &[0] }
        );
        if uncertain {
            assert!(invoke(&mut session, &mut observer, &mut work, false, false).is_err());
            assert_eq!(
                work.calls, 2,
                "indeterminate work must fence subsequent mutation"
            );
        }
        PARTITION_COMMIT_TRACE.with(|slot| *slot.borrow_mut() = None);
    }
}
