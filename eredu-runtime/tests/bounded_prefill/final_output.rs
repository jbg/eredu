//! Physical Sequence observations do not retain old scores into the next span.
use super::*;
use std::rc::Weak;

struct Scores {
    values: Vec<f64>,
    chunk: PrefillChunk,
    status: Arc<AtomicU8>,
    retired: Rc<RefCell<Vec<PrefillChunk>>>,
}
impl Drop for Scores {
    fn drop(&mut self) {
        assert_eq!(
            self.status.load(Ordering::SeqCst),
            1,
            "score drop precedes next native work and follows its completion"
        );
        self.retired.borrow_mut().push(self.chunk.clone());
    }
}
struct Tracked {
    inner: RecurrentExecutor,
    owners: Vec<Weak<Scores>>,
    retired: Rc<RefCell<Vec<PrefillChunk>>>,
    fail_at: Option<u64>,
    cancel_after_first: bool,
}
impl Tracked {
    fn new() -> Self {
        Self {
            inner: RecurrentExecutor::new(),
            owners: vec![],
            retired: Default::default(),
            fail_at: None,
            cancel_after_first: false,
        }
    }
}
impl PrefillExecutor for Tracked {
    type Output = Rc<Scores>;
    type Completion = NativeCompletion;
    type Error = std::io::Error;
    fn agree_cancellation_at(
        &mut self,
        _: eredu_runtime::prefill::PrefillBoundary,
        cancellation: &GenerationCancellationToken,
        _: InferenceRequest,
    ) -> Result<bool, Self::Error> {
        Ok(cancellation.is_cancelled()
            || (self.cancel_after_first && !self.inner.submitted.is_empty()))
    }
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        reservation: InferenceRequest,
    ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error> {
        assert!(
            self.owners.iter().all(|owner| owner.upgrade().is_none()),
            "prior scores survived into next input/model submission"
        );
        self.inner.status.store(0, Ordering::SeqCst);
        self.inner.fail_submission = self.fail_at == Some(chunk.input.start);
        let submitted = self.inner.submit_chunk(chunk, reservation)?;
        let output = submitted.output.map(|values| {
            let value = Rc::new(Scores {
                values,
                chunk: chunk.clone(),
                status: self.inner.status.clone(),
                retired: self.retired.clone(),
            });
            self.owners.push(Rc::downgrade(&value));
            value
        });
        Ok(Submission {
            output,
            completion: submitted.completion,
        })
    }
}

#[test]
fn final_only_run_retires_settled_sequence_scores_before_next_submission() {
    for chunk_size in [3, 4] {
        for demand in [
            OutputDemand::Sequence,
            OutputDemand::LastPosition,
            OutputDemand::StateOnly,
        ] {
            let g = geometry(chunk_size, demand);
            let pool = memory::host_ledger(10000, 100).unwrap();
            let execution = InferenceExecutionIdentity::default();
            let reservation = pool.reserve(&execution, &admission(g)).unwrap();
            let mut driver = PrefillDriver::new(
                &execution,
                reservation,
                g,
                GenerationCancellationToken::new(),
            )
            .unwrap();
            let mut executor = Tracked::new();
            let (outcome, output) = driver.run_final(&mut executor).unwrap();
            assert_eq!(outcome, PrefillOutcome::Complete);
            assert_eq!(
                executor
                    .inner
                    .submitted
                    .iter()
                    .map(|chunk| chunk.input.clone())
                    .collect::<Vec<_>>(),
                if chunk_size == 3 {
                    vec![0..3, 3..6, 6..7]
                } else {
                    vec![0..4, 4..7]
                }
            );
            let (mut reference, all_scores) = run(7, demand, false);
            assert!((executor.inner.state - reference.state).abs() < 1e-12);
            if demand == OutputDemand::StateOnly {
                assert!(output.is_none());
                assert!(executor.owners.is_empty());
                assert!(executor.inner.projected.is_empty());
            } else {
                let score = output.as_ref().unwrap();
                let final_start = if chunk_size == 3 { 6 } else { 4 };
                assert_eq!(score.chunk.input, final_start..7);
                let retained = if demand == OutputDemand::Sequence {
                    7 - final_start as usize
                } else {
                    1
                };
                assert_eq!(
                    score.values.as_slice(),
                    &all_scores[all_scores.len() - retained..]
                );
                assert_eq!(
                    executor
                        .owners
                        .iter()
                        .filter(|owner| owner.upgrade().is_some())
                        .count(),
                    1
                );
                assert_eq!(
                    executor.retired.borrow().len(),
                    if demand == OutputDemand::Sequence {
                        executor.inner.submitted.len() - 1
                    } else {
                        0
                    }
                );
            }
            for token in [0.2, 0.7, 0.4] {
                assert!((executor.inner.decode(token) - reference.decode(token)).abs() < 1e-12);
            }
            drop(output);
            assert!(executor
                .owners
                .iter()
                .all(|owner| owner.upgrade().is_none()));
            assert_eq!(executor.retired.borrow().len(), executor.owners.len());
            drop(driver);
            assert_eq!(pool.funded_used_bytes().unwrap(), 100);
        }
    }
}

#[test]
fn final_only_run_cancellation_and_later_failure_leave_no_prior_scores() {
    for cancel in [false, true] {
        let g = geometry(3, OutputDemand::Sequence);
        let pool = memory::host_ledger(10000, 100).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let reservation = pool.reserve(&execution, &admission(g)).unwrap();
        let mut driver = PrefillDriver::new(
            &execution,
            reservation,
            g,
            GenerationCancellationToken::new(),
        )
        .unwrap();
        let mut executor = Tracked::new();
        executor.cancel_after_first = cancel;
        executor.fail_at = (!cancel).then_some(3);
        let result = driver.run_final(&mut executor);
        if cancel {
            let (outcome, output) = result.unwrap();
            assert_eq!(outcome, PrefillOutcome::Cancelled);
            assert!(output.is_none());
        } else {
            assert!(matches!(result, Err(PrefillError::Submission(_))));
            assert!(matches!(
                driver.step(&mut executor),
                Err(PrefillError::Failed)
            ));
        }
        assert_eq!(executor.inner.submitted.len(), 1);
        assert_eq!(executor.retired.borrow().len(), 1);
        assert!(executor
            .owners
            .iter()
            .all(|owner| owner.upgrade().is_none()));
        drop(driver);
        assert_eq!(pool.funded_used_bytes().unwrap(), 100);
    }
}
