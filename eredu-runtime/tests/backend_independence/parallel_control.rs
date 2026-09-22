//! Shared lifecycle failure must fence before either a phase or final decision can bypass its loan.
use super::*;
use eredu_runtime::replicated_session::ParallelControlEvent;
use eredu_runtime::replicated_session::{
    ParallelControlCallbackVisitor, SessionTransactionControlOccurrence,
};

#[derive(Default)]
struct CallbackTypes {
    session: Vec<usize>,
    group: Vec<usize>,
}
impl ParallelControlCallbackVisitor<FakeBackend> for CallbackTypes {
    type Error = std::convert::Infallible;
    fn visit<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(Option<(&(), &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        assert!(std::mem::size_of::<F>() > 0);
        assert!(std::mem::size_of::<Result<T, E>>() > 0);
        self.session.push(occurrence.ordinal());
        Ok(())
    }
    fn visit_group<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(
            Option<&<FakeBackend as eredu_runtime::CommunicationBackend>::CommunicationGroup>,
        ) -> Result<T, E>,
    {
        assert!(std::mem::size_of::<F>() > 0);
        assert!(std::mem::size_of::<Result<T, E>>() > 0);
        self.group.push(occurrence.ordinal());
        Ok(())
    }
}

struct Trace {
    reject: ParallelControlEvent,
    events: Vec<ParallelControlEvent>,
}
thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}
struct ResetTrace;
impl Drop for ResetTrace {
    fn drop(&mut self) {
        TRACE.with(|trace| *trace.borrow_mut() = None);
    }
}

pub(super) fn with_loan<T, E, F>(
    event: ParallelControlEvent,
    run: F,
) -> Result<Result<T, E>, eredu_core::BackendFailure>
where
    F: FnOnce(Option<(&(), &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
{
    let reject = TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        if let Some(trace) = trace.as_mut() {
            trace.events.push(event);
            trace.reject == event
        } else {
            false
        }
    });
    if reject {
        return Err(eredu_core::BackendFailure::from_error(
            std::io::Error::other("injected request control admission failure"),
        ));
    }
    Ok(run(None))
}

#[test]
fn rejected_control_loan_fences_phase_and_final_decision_without_retry_or_rollback() {
    for reject in [
        ParallelControlEvent::Phase(DistributedExecutionPhase::StateCheckpoint),
        ParallelControlEvent::Commit,
    ] {
        let commits = Rc::new(Cell::new(0));
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Final {
                result: FinalDecisionResult::Committed,
                commits: Rc::clone(&commits),
            },
        );
        TRACE.with(|trace| {
            *trace.borrow_mut() = Some(Trace {
                reject,
                events: Vec::new(),
            })
        });
        let _reset = ResetTrace;
        let error = session.decode(&FakeTensor(vec![3]), &()).unwrap_err();
        assert!(matches!(
            error,
            eredu_runtime::ReplicatedTextSessionError::ParallelControl(_)
        ));
        assert_eq!(
            commits.get(),
            0,
            "loan failure must suppress final decision"
        );
        let forwards = counters.snapshot().forward_calls;
        let report = session.report().unwrap();
        if reject == ParallelControlEvent::Commit {
            assert!(forwards > 0);
            assert_eq!(
                report.state_report(),
                &[1],
                "unknown commit may not roll state back"
            );
            assert!(matches!(
                report.distributed_commit(),
                Some(DistributedCommitOutcome::Indeterminate {
                    phase: DistributedCommitPhase::DecisionSubmission,
                    ..
                })
            ));
        } else {
            assert_eq!(forwards, 0, "early control refusal must prevent execution");
            assert_eq!(report.state_report(), &[0]);
        }
        let attempts = TRACE.with(|trace| trace.borrow().as_ref().unwrap().events.len());
        assert!(session.decode(&FakeTensor(vec![4]), &()).is_err());
        assert_eq!(counters.snapshot().forward_calls, forwards);
        TRACE.with(|trace| {
            let trace = trace.borrow();
            let trace = trace.as_ref().unwrap();
            assert_eq!(
                trace.events.len(),
                attempts,
                "fenced retries may not issue a new control loan"
            );
            assert_eq!(trace.events.last(), Some(&reject));
        });
    }
}
#[test]
fn transaction_control_source_matches_actual_output_driver_and_failed_prefixes() {
    for reject in [
        ParallelControlEvent::Phase(DistributedExecutionPhase::StateCheckpoint),
        ParallelControlEvent::Commit,
        ParallelControlEvent::Phase(DistributedExecutionPhase::PredictionExtensionExecution),
    ] {
        let commits = Rc::new(Cell::new(0));
        let (mut session, _, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Final {
                result: FinalDecisionResult::Committed,
                commits: Rc::clone(&commits),
            },
        );
        let plan = session.transaction_control_plan(
            eredu_core::OutputDemand::Sequence,
            false,
            false,
            false,
        );
        let mut types = CallbackTypes {
            session: Vec::new(),
            group: Vec::new(),
        };
        assert!(
            session
                .visit_transaction_control_callbacks(&plan, &mut types)
                .unwrap()
        );
        assert_eq!(types.session, types.group);
        assert_eq!(types.session.len(), plan.occurrences().count());
        TRACE.with(|trace| {
            *trace.borrow_mut() = Some(Trace {
                reject,
                events: Vec::new(),
            })
        });
        let _reset = ResetTrace;
        let result = session.sequence_logits_with_completion(
            &FakeTensor(vec![3]),
            eredu_runtime::ExpertPass::Decode,
            &(),
            |_, _, _| Ok(()),
        );
        let mut cursor = plan.cursor();
        TRACE.with(|trace| {
            let trace = trace.borrow();
            for event in &trace.as_ref().unwrap().events {
                cursor.claim(*event).unwrap();
            }
        });
        cursor.finish_transaction(result.is_ok()).unwrap();
        assert!(cursor.is_complete());
        assert_eq!(commits.get(), usize::from(result.is_ok()));
        assert!(cursor.claim(ParallelControlEvent::Commit).is_err());
    }
}

#[test]
fn cache_control_source_matches_save_load_and_fenced_failure_prefixes() {
    use eredu_runtime::replicated_session::SessionCacheControlOperation as Operation;
    let options = eredu_core::cache::PromptCacheOptions::new(None, true).unwrap();
    for operation in [Operation::Save, Operation::Load] {
        let phases: &[DistributedExecutionPhase] = match operation {
            Operation::Save => &[
                DistributedExecutionPhase::PromptCacheSavePreflight,
                DistributedExecutionPhase::PromptCacheSavePreparation,
                DistributedExecutionPhase::PromptCacheSavePublication,
            ],
            Operation::Load => &[
                DistributedExecutionPhase::PromptCacheLoadPreflight,
                DistributedExecutionPhase::PromptCacheLoadPreparation,
            ],
        };
        // Each possible loan refusal, full success, and a local preparation
        // failure exercise the same actual driver and retained control source.
        for case in 0..phases.len() + 2 {
            let local_failure = case == phases.len() + 1;
            let rejected = phases.get(case).copied();
            let (mut session, descriptor, store, calls, _) = partitioned_cache_control_session(
                0,
                DistributedExecutionPhase::PredictionExtensionExecution,
                None,
                !local_failure,
            );
            if !local_failure {
                session
                    .save_prompt_cache(
                        std::path::Path::new("unused"),
                        descriptor.clone(),
                        if operation == Operation::Save {
                            &[1]
                        } else {
                            &[3]
                        },
                        &options,
                        &(),
                    )
                    .unwrap();
            }
            if operation == Operation::Load {
                session.decode(&FakeTensor(vec![4]), &()).unwrap();
                assert_eq!(session.report().unwrap().state_report(), &[1]);
            }
            let original_store = reference_prompt_cache_snapshot(&store);
            calls.borrow_mut().clear();
            let plan = session.cache_control_plan(operation);
            assert_eq!(plan.operation(), operation);
            assert_eq!(
                plan.occurrences()
                    .map(|row| row.event())
                    .collect::<Vec<_>>(),
                phases
                    .iter()
                    .copied()
                    .map(ParallelControlEvent::Phase)
                    .collect::<Vec<_>>()
            );
            let mut types = CallbackTypes::default();
            assert!(
                session
                    .visit_cache_control_callbacks(&plan, &mut types)
                    .unwrap()
            );
            assert_eq!(types.session, (0..phases.len()).collect::<Vec<_>>());
            assert_eq!(types.session, types.group);
            TRACE.with(|trace| {
                *trace.borrow_mut() = Some(Trace {
                    reject: ParallelControlEvent::Phase(
                        rejected.unwrap_or(DistributedExecutionPhase::PredictionExtensionExecution),
                    ),
                    events: Vec::new(),
                })
            });
            let _reset = ResetTrace;
            let execute = |session: &mut ReferencePartitionedSession| match operation {
                Operation::Save => session.save_prompt_cache_distributed(
                    std::path::Path::new("unused"),
                    descriptor.clone(),
                    &[3],
                    &options,
                    &(),
                ),
                Operation::Load => session.load_prompt_cache_distributed(
                    std::path::Path::new("unused"),
                    &descriptor,
                    &[3],
                    &(),
                ),
            };
            let result = execute(&mut session);
            let success = rejected.is_none() && !local_failure;
            assert_eq!(result.is_ok(), success);
            let events = TRACE.with(|trace| trace.borrow().as_ref().unwrap().events.clone());
            let mut cursor = plan.cursor();
            for event in &events {
                cursor.claim(*event).unwrap();
            }
            cursor.finish(result.is_ok()).unwrap();
            assert!(cursor.is_complete());
            assert_eq!(
                events.len(),
                if local_failure {
                    2
                } else {
                    (case + 1).min(phases.len())
                }
            );
            if success {
                assert!(result.unwrap().is_some());
                match operation {
                    Operation::Save => {
                        assert_ne!(reference_prompt_cache_snapshot(&store), original_store)
                    }
                    Operation::Load => assert_eq!(session.report().unwrap().state_report(), &[0]),
                }
            } else {
                assert_eq!(reference_prompt_cache_snapshot(&store), original_store);
                if operation == Operation::Load {
                    assert_eq!(session.report().unwrap().state_report(), &[1]);
                }
                assert!(execute(&mut session).is_err());
                TRACE.with(|trace| assert_eq!(trace.borrow().as_ref().unwrap().events, events));
            }
        }
    }
}

thread_local! {
    static MODEL_TRACE: RefCell<Option<Vec<(CollectiveGroupId, DistributedExecutionPhase)>>> = const { RefCell::new(None) };
}
pub(super) fn record_model_control(group: CollectiveGroupId, phase: DistributedExecutionPhase) {
    MODEL_TRACE.with(|trace| {
        if let Some(trace) = trace.borrow_mut().as_mut() {
            trace.push((group, phase));
        }
    });
}
struct ResetModelTrace;
impl Drop for ResetModelTrace {
    fn drop(&mut self) {
        MODEL_TRACE.with(|trace| *trace.borrow_mut() = None);
    }
}

#[test]
fn model_control_source_records_the_actual_partition_pass_separately_from_transaction_phases() {
    use eredu_nn::workspace::*;
    use eredu_runtime::replicated_session::SessionModelControlPlan;
    #[derive(Debug)]
    struct Controls;
    impl WorkspaceMechanisms for Controls {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::CommunicationControl(_)
            ));
            assert!(operation.inputs.is_empty() && operation.outputs.is_empty());
            Ok(Some(WorkspaceOperationBound {
                outputs: Vec::new(),
                scratch_bytes: 0,
                assumptions: "descriptive model control has no tensor storage".into(),
            }))
        }
    }
    #[derive(Debug)]
    struct Account(Arc<AtomicUsize>);
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    for failed in [false, true] {
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            if failed {
                LocalPartitionFailure::Execution
            } else {
                LocalPartitionFailure::None
            },
            TestPhaseAgreement::Final {
                result: FinalDecisionResult::Committed,
                commits: Rc::new(Cell::new(0)),
            },
        );
        MODEL_TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
        let _reset = ResetModelTrace;
        assert_eq!(session.decode(&FakeTensor(vec![3]), &()).is_err(), failed);
        assert!(counters.snapshot().forward_calls > 0);
        let actual = MODEL_TRACE.with(|trace| trace.borrow_mut().take().unwrap());
        assert_eq!(
            actual,
            [(
                CollectiveGroupId::new(41),
                DistributedExecutionPhase::Execution
            )]
        );
        let context = WorkspaceContext::new(Controls);
        for (group, phase) in actual {
            assert_eq!(phase, DistributedExecutionPhase::Execution);
            context
                .record_model_control(WorkspaceModelControl {
                    group,
                    phase: WorkspaceModelControlPhase::Execution,
                })
                .unwrap();
        }
        let report = context.report(&[]).unwrap();
        assert_eq!(report.tensor_buffers.total_bytes, Some(0));
        let retired = Arc::new(AtomicUsize::new(0));
        let funding = HostMetadataFunding::new(Account(retired.clone())).unwrap();
        let plan = SessionModelControlPlan::from_operations(&report.operations, &funding).unwrap();
        let mut callbacks = CallbackTypes::default();
        assert!(
            session
                .visit_model_control_callbacks(&plan, &mut callbacks)
                .unwrap()
        );
        assert_eq!(callbacks.session, [0]);
        assert_eq!(callbacks.group, [0]);
        let declaration = plan.occurrences()[0].declaration();
        let alias = plan.clone();
        let mut rejected = plan.clone().into_cursor();
        assert!(rejected.finish(true).is_err());
        assert!(rejected.claim(declaration).is_err());
        drop(rejected);
        let mut cursor = plan.into_cursor();
        drop(funding);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(cursor.claim(declaration).unwrap(), 0);
        cursor.finish(!failed).unwrap();
        assert!(cursor.claim(declaration).is_err());
        assert!(cursor.finish(true).is_err());
        drop(cursor);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
