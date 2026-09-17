// Exact protocol proof through materialized target/extension, the shared driver,
// and ordinary registration. Numerical parity lives in native fixtures.
#[derive(Clone)]
struct ReferenceCapturedCase {
    chunk: Option<std::num::NonZeroU64>,
    span_supported: bool,
    sequence: bool,
    prompt_positions: usize,
    target_projection_rows: Rc<RefCell<Vec<Vec<i32>>>>,
    cancel_after: Option<eredu_core::speculative::SpeculativeActivationPhase>,
    cancellation: GenerationCancellationToken,
    invocations: Rc<
        RefCell<
            Vec<(
                eredu_core::speculative::SpeculativeActivationPhase,
                usize,
                Option<eredu_core::speculative::SpeculativePrefillSpan>,
                bool,
            )>,
        >,
    >,
}
impl ReferenceCapturedCase {
    fn new(
        chunk: Option<u64>,
        cancel_after: Option<eredu_core::speculative::SpeculativeActivationPhase>,
    ) -> Self {
        Self {
            chunk: chunk.and_then(std::num::NonZeroU64::new),
            span_supported: true,
            sequence: false,
            prompt_positions: 5,
            target_projection_rows: Default::default(),
            cancel_after,
            cancellation: GenerationCancellationToken::new(),
            invocations: Default::default(),
        }
    }
}
struct ReferenceSpanObserver {
    case: ReferenceCapturedCase,
    span: Option<eredu_core::speculative::SpeculativePrefillSpan>,
    active: Option<usize>,
}
impl ReferenceSpanObserver {
    fn new(case: ReferenceCapturedCase) -> Self {
        Self {
            case,
            span: None,
            active: None,
        }
    }
}
impl eredu_runtime::ActivationObserver<ReferenceTensor, Error> for ReferenceSpanObserver {
    fn supports_prefill_spans(&self) -> bool {
        self.case.span_supported
    }
    fn requires_sequence_readout(&self) -> bool {
        self.case.sequence
    }
    fn observe(&mut self, _: &str, _: &ReferenceTensor) -> Result<(), Error> {
        Ok(())
    }
}
impl eredu_runtime::inspection::SpeculativeActivationObserver<ReferenceTensor, Error>
    for ReferenceSpanObserver
{
    fn set_prefill_reduction_geometry(
        &mut self,
        _: eredu_core::speculative::SpeculativePrefillReductionGeometry,
    ) {
        assert!(
            self.case.span_supported,
            "whole callback receives no reduction geometry"
        );
    }
    fn complete_prefill_reductions(&mut self) -> Result<(), Error> {
        assert!(
            self.case.span_supported,
            "whole callback receives no reduction completion"
        );
        Ok(())
    }
    fn finish_prefill_reductions(&mut self, _: bool) {
        assert!(
            self.case.span_supported || self.case.chunk.is_some(),
            "whole callback receives no reduction cleanup"
        );
    }
    fn set_prefill_span(&mut self, span: Option<eredu_core::speculative::SpeculativePrefillSpan>) {
        assert!(
            self.case.span_supported,
            "whole-invocation observer receives no span hooks"
        );
        assert!(self.active.is_none());
        self.span = span;
    }
    fn begin_activation_invocation(
        &mut self,
        phase: eredu_core::speculative::SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), Error> {
        assert!(self.active.is_none());
        if let Some(span) = self.span {
            assert!(span.validate(phase, sequence));
        }
        if phase == eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill {
            clear_reference_trace();
        }
        let mut entries = self.case.invocations.borrow_mut();
        self.active = Some(entries.len());
        entries.push((phase, sequence, self.span, false));
        Ok(())
    }
    fn complete_activation_invocation(&mut self) -> Result<(), Error> {
        assert!(self.active.is_some());
        Ok(())
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        let index = self
            .active
            .take()
            .expect("every entered invocation closes once");
        let phase = {
            let mut entries = self.case.invocations.borrow_mut();
            entries[index].3 = success;
            entries[index].0
        };
        if phase == eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill {
            self.case.target_projection_rows.borrow_mut().push(
                reference_trace()
                    .linear_outputs
                    .iter()
                    .filter(|(id, _)| matches!(id.as_str(), "lm_head.weight" | "head.weight"))
                    .map(|(_, shape)| shape[1])
                    .collect(),
            );
        }
        if success && self.case.cancel_after == Some(phase) {
            self.case.cancellation.cancel();
        }
    }
}

#[test]
fn materialized_captured_spans_preserve_shifted_rows_and_registration_cancellation_counts() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(captured_spans_and_registration_cancellation)
        .unwrap()
        .join()
        .unwrap();
}

fn captured_spans_and_registration_cancellation() {
    use eredu_core::speculative::SpeculativeActivationPhase as Phase;
    for (mut config, shifted) in [
        (production_deepseek_v3_prediction_config(), true),
        (production_deepseek_v4_dspark_config(), false),
    ] {
        // The actual selected target permits the 513-row default-span boundary
        // and its following decode steps; the small original fixture used128.
        config["max_position_embeddings"] = 1024.into();
        let full = ReferenceCapturedCase::new(None, None);
        let baseline =
            run_reference_embedded_production_with_prefill(&config, false, Some(full.clone()))
                .unwrap();
        let target = full
            .invocations
            .borrow()
            .iter()
            .filter(|e| e.0 == Phase::TargetPrefill)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(target.len(), 1);
        assert_eq!(target[0].1, 5);
        assert_eq!(target[0].2.unwrap().input_start, 0);
        assert_eq!(*full.target_projection_rows.borrow(), vec![vec![1]]);
        let mut whole = ReferenceCapturedCase::new(None, None);
        whole.span_supported = false;
        whole.sequence = true;
        let observed =
            run_reference_embedded_production_with_prefill(&config, false, Some(whole.clone()))
                .unwrap();
        assert_eq!(observed.tokens, baseline.tokens);
        assert_eq!(observed.accepted, baseline.accepted);
        assert_eq!(observed.target_tokens, baseline.target_tokens);
        assert!(whole
            .invocations
            .borrow()
            .iter()
            .all(|entry| entry.2.is_none() && entry.3));
        assert_eq!(*whole.target_projection_rows.borrow(), vec![vec![5]]);
        let mut incompatible = ReferenceCapturedCase::new(Some(2), None);
        incompatible.span_supported = false;
        assert!(run_reference_embedded_production_with_prefill(
            &config,
            false,
            Some(incompatible.clone())
        )
        .is_err());
        assert!(incompatible.invocations.borrow().is_empty());
        assert!(incompatible.target_projection_rows.borrow().is_empty());
        let mut long = ReferenceCapturedCase::new(None, None);
        long.prompt_positions = 513;
        let long_result =
            run_reference_embedded_production_with_prefill(&config, false, Some(long.clone()))
                .unwrap();
        assert_eq!(long_result.tokens.len(), 13);
        let entries = long.invocations.borrow();
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.0 == Phase::TargetPrefill)
                .map(|e| e.1)
                .collect::<Vec<_>>(),
            [512, 1]
        );
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.0 == Phase::PredictionPrefill)
                .map(|e| e.1)
                .collect::<Vec<_>>(),
            if shifted { vec![511, 1] } else { vec![512, 1] }
        );
        assert_eq!(
            *long.target_projection_rows.borrow(),
            vec![Vec::<i32>::new(), vec![1]]
        );
        drop(entries);
        let cancelled = ReferenceCapturedCase::new(None, None);
        cancelled.cancellation.cancel();
        let early =
            run_reference_embedded_production_with_prefill(&config, false, Some(cancelled.clone()))
                .unwrap();
        assert!(early.tokens.is_empty());
        assert_eq!(early.target_tokens, 0);
        assert_eq!(early.publications, 0);
        assert!(cancelled.invocations.borrow().is_empty());
        assert!(cancelled.target_projection_rows.borrow().is_empty());
        let spans = ReferenceCapturedCase::new(Some(2), None);
        let result =
            run_reference_embedded_production_with_prefill(&config, false, Some(spans.clone()))
                .unwrap();
        assert_eq!(result.tokens, baseline.tokens);
        assert_eq!(result.tokens.len(), 13);
        assert_eq!(result.target_tokens, baseline.target_tokens);
        assert_eq!(result.accepted, baseline.accepted);
        assert!(result.accepted.len() >= 3);
        let entries = spans.invocations.borrow();
        assert!(entries.iter().all(|e| e.3));
        let target = entries
            .iter()
            .filter(|e| e.0 == Phase::TargetPrefill)
            .collect::<Vec<_>>();
        assert_eq!(target.iter().map(|e| e.1).collect::<Vec<_>>(), [2, 2, 1]);
        assert_eq!(
            target
                .iter()
                .map(|e| e.2.unwrap().input_start)
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );
        let seed = entries
            .iter()
            .filter(|e| e.0 == Phase::PredictionPrefill)
            .collect::<Vec<_>>();
        assert_eq!(
            seed.iter().map(|e| e.1).collect::<Vec<_>>(),
            if shifted {
                vec![1, 2, 1]
            } else {
                vec![2, 2, 1]
            }
        );
        for entry in seed {
            let span = entry.2.unwrap();
            assert_eq!(span.token_start - span.hidden_start, u64::from(shifted));
            assert_eq!(span.seed_start, span.hidden_start);
        }
        // Whole-sequence verification stays wider than one generated position.
        assert!(entries
            .iter()
            .any(|e| e.0 == Phase::Verification && e.1 > 1 && e.2.is_none()));
        drop(entries);
        for (chunk, positions, completed) in [(Some(2), 5, 2), (None, 513, 512)] {
            for phase in [Phase::TargetPrefill, Phase::PredictionPrefill] {
                let mut cancelled = ReferenceCapturedCase::new(chunk, Some(phase));
                cancelled.prompt_positions = positions;
                let result = run_reference_embedded_production_with_prefill(
                    &config,
                    false,
                    Some(cancelled.clone()),
                )
                .unwrap();
                assert!(result.tokens.is_empty());
                assert!(result.accepted.is_empty());
                assert_eq!(result.publications, 0);
                assert_eq!(
                    result.target_tokens, completed,
                    "completed target work is not refunded by cancellation"
                );
                let entries = cancelled.invocations.borrow();
                assert_eq!(
                    entries
                        .iter()
                        .filter(|e| e.0 == Phase::TargetPrefill)
                        .count(),
                    1
                );
                assert!(entries
                    .iter()
                    .all(|e| matches!(e.0, Phase::TargetPrefill | Phase::PredictionPrefill)));
                assert!(
                    entries.iter().all(|e| e.3),
                    "the in-flight chunk settles before agreed cancellation"
                );
            }
        }
    }
}
