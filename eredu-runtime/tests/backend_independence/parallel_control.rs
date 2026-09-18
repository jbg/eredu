//! Shared lifecycle failure must fence before either a phase or final decision can bypass its loan.
use super::*;
use eredu_runtime::replicated_session::ParallelControlEvent;

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
