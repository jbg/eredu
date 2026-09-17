use super::*;

struct Boundaries {
    inner: RecurrentExecutor,
    observed: Vec<PrefillBoundary>,
    cancel_at: Option<PrefillBoundary>,
}
impl Boundaries {
    fn new() -> Self {
        Self {
            inner: RecurrentExecutor::new(),
            observed: vec![],
            cancel_at: None,
        }
    }
}
impl PrefillExecutor for Boundaries {
    type Output = Vec<f64>;
    type Completion = NativeCompletion;
    type Error = std::io::Error;
    fn agree_cancellation_at(
        &mut self,
        boundary: PrefillBoundary,
        cancellation: &GenerationCancellationToken,
        _reservation: InferenceRequest,
    ) -> Result<bool, Self::Error> {
        self.observed.push(boundary);
        if self.cancel_at == Some(boundary) {
            cancellation.cancel();
        }
        Ok(cancellation.is_cancelled())
    }
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        reservation: InferenceRequest,
    ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error> {
        self.inner.submit_chunk(chunk, reservation)
    }
}

#[test]
fn shared_driver_boundaries_are_completion_gated_and_run_step_equivalent() {
    let mut g = geometry(2, OutputDemand::LastPosition);
    g.input_positions = 5;
    let execution = InferenceExecutionIdentity::default();
    let mut results = vec![];
    for controlled in [false, true] {
        let request = InferenceRequest::without_memory_budget(&execution, g).unwrap();
        let mut driver =
            PrefillDriver::new(&execution, request, g, GenerationCancellationToken::new()).unwrap();
        let mut executor = Boundaries::new();
        let mut scores = vec![];
        if controlled {
            assert!(matches!(
                driver.step(&mut executor).unwrap(),
                PrefillProgress::Pending
            ));
            assert!(matches!(
                driver.step(&mut executor).unwrap(),
                PrefillProgress::Pending
            ));
            assert_eq!(
                executor.observed,
                [PrefillBoundary::Before { input_position: 0 }]
            );
            assert_eq!(executor.inner.submitted.len(), 1);
            executor.inner.status.store(1, Ordering::SeqCst);
            loop {
                match driver.step(&mut executor).unwrap() {
                    PrefillProgress::Chunk { output, .. } => {
                        scores.extend(output.into_iter().flatten())
                    }
                    PrefillProgress::Complete => break,
                    other => panic!("unexpected progress {other:?}"),
                }
            }
        } else {
            assert_eq!(
                driver
                    .run(&mut executor, |_, output| scores
                        .extend(output.into_iter().flatten()))
                    .unwrap(),
                PrefillOutcome::Complete,
            );
        }
        assert_eq!(
            executor.observed,
            [
                PrefillBoundary::Before { input_position: 0 },
                PrefillBoundary::After { input_position: 2 },
                PrefillBoundary::Before { input_position: 2 },
                PrefillBoundary::After { input_position: 4 },
                PrefillBoundary::Before { input_position: 4 },
                PrefillBoundary::After { input_position: 5 },
            ]
        );
        assert_eq!(
            executor
                .inner
                .submitted
                .iter()
                .map(|c| c.input.end - c.input.start)
                .collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert_eq!(executor.inner.projected, [1]);
        assert_eq!(scores.len(), 1);
        assert_ne!(scores[0], 0.0);
        results.push((scores, executor.inner.state));
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn shared_driver_cancelled_boundary_does_not_reissue_or_publish_completed_scores() {
    for boundary in [
        PrefillBoundary::Before { input_position: 0 },
        PrefillBoundary::After { input_position: 2 },
    ] {
        let mut g = geometry(2, OutputDemand::Sequence);
        g.input_positions = 5;
        let execution = InferenceExecutionIdentity::default();
        let request = InferenceRequest::without_memory_budget(&execution, g).unwrap();
        let mut driver =
            PrefillDriver::new(&execution, request, g, GenerationCancellationToken::new()).unwrap();
        let mut executor = Boundaries::new();
        executor.cancel_at = Some(boundary);
        assert_eq!(
            driver
                .run(&mut executor, |_, _| panic!("cancelled output published"))
                .unwrap(),
            PrefillOutcome::Cancelled
        );
        let completed = if matches!(boundary, PrefillBoundary::After { .. }) {
            2
        } else {
            0
        };
        assert_eq!(driver.completed_positions(), completed);
        assert_eq!(executor.inner.submitted.len(), usize::from(completed != 0));
        assert_eq!(executor.observed.last(), Some(&boundary));
        let calls = executor.observed.len();
        assert!(matches!(
            driver.step(&mut executor).unwrap(),
            PrefillProgress::Cancelled
        ));
        assert_eq!(executor.observed.len(), calls);
    }
}

#[test]
fn scope_geometry_distinguishes_inner_admission_and_final_readout_without_overflow() {
    let mut g = geometry(2, OutputDemand::LastPosition);
    g.input_positions = 5;
    g.cached_positions = 7;
    for retained in [false, true] {
        let plan = PrefillControlPlan::new(g, retained).unwrap();
        assert_eq!(plan.span_count(), 3);
        assert_eq!(plan.scope_count(), if retained { 17 } else { 14 });
        assert_eq!(plan.role(0), Some(PrefillControlRole::SourcePreparation));
        assert_eq!(
            plan.role(plan.scope_count() - 1),
            Some(PrefillControlRole::FinalIndex)
        );
        assert_eq!(plan.role(plan.scope_count()), None);
        for (start, end) in [(0, 2), (2, 4), (4, 5)] {
            assert_eq!(
                plan.boundary_role(PrefillBoundary::Before {
                    input_position: start
                }),
                Some(PrefillControlRole::Span {
                    phase: PrefillSpanControlPhase::PreBoundary,
                    input_start: start,
                    input_end: end,
                    position: 7 + start,
                    output: g.output.for_chunk(end == 5),
                })
            );
            assert_eq!(
                plan.boundary_role(PrefillBoundary::After {
                    input_position: end
                }),
                Some(PrefillControlRole::Span {
                    phase: PrefillSpanControlPhase::PostBoundary,
                    input_start: start,
                    input_end: end,
                    position: 7 + start,
                    output: g.output.for_chunk(end == 5),
                })
            );
        }
        assert_eq!(
            plan.boundary_role(PrefillBoundary::After { input_position: 0 }),
            None
        );
        assert_eq!(
            plan.boundary_role(PrefillBoundary::Before { input_position: 1 }),
            None
        );
        assert_eq!(
            plan.boundary_role(PrefillBoundary::Before { input_position: 5 }),
            None
        );
        let state = PrefillControlPlan::new(
            InferenceGeometry {
                output: OutputDemand::StateOnly,
                ..g
            },
            retained,
        )
        .unwrap();
        assert_eq!(state.scope_count(), plan.scope_count() - 1);
        assert!(matches!(
            state.role(state.scope_count() - 1),
            Some(PrefillControlRole::Span {
                phase: PrefillSpanControlPhase::PostBoundary,
                ..
            })
        ));
    }
    let enormous = InferenceGeometry {
        cached_positions: 0,
        input_positions: u64::MAX,
        prefill_chunk_positions: 1,
        max_output_tokens: 0,
        ..g
    };
    assert!(matches!(
        PrefillControlPlan::new(enormous, true),
        Err(WorkingMemoryError::Overflow)
    ));
    let one = PrefillControlPlan::new(
        InferenceGeometry {
            prefill_chunk_positions: u64::MAX,
            ..enormous
        },
        true,
    )
    .unwrap();
    assert_eq!(one.scope_count(), 7);
    assert!(matches!(
        one.role(5),
        Some(PrefillControlRole::Span {
            input_end: u64::MAX,
            ..
        })
    ));
}
