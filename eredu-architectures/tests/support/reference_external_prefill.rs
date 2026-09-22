// Production architecture/assistant protocol with shape-only operators. Native
// Gemma and the numeric raw-context test cover values separately.
#[derive(Clone)]
struct ReferenceExternalCase {
    chunk: Option<std::num::NonZeroU64>,
    cancellation: GenerationCancellationToken,
    cancel_after_span: bool,
    span_supported: bool,
    context_supported: bool,
    sequence: bool,
    prompt_positions: usize,
    trace: Rc<RefCell<ReferenceExternalSpanTrace>>,
}
#[derive(Default)]
struct ReferenceExternalSpanTrace {
    spans: Vec<(u64, u64)>,
    contexts: Vec<u64>,
    finished: Vec<bool>,
    linear_at_finish: Vec<(String, Vec<i32>)>,
    whole_values: Vec<(String, Vec<i32>)>,
    whole_complete: bool,
}
impl ReferenceExternalCase {
    fn new(chunk: Option<u64>, cancel_after_span: bool) -> Self {
        Self {
            chunk: chunk.and_then(std::num::NonZeroU64::new),
            cancellation: GenerationCancellationToken::new(),
            cancel_after_span,
            span_supported: true,
            context_supported: true,
            sequence: false,
            prompt_positions: 5,
            trace: Default::default(),
        }
    }
}
struct ReferenceExternalSpanObserver {
    case: ReferenceExternalCase,
    start: Option<u64>,
}
impl eredu_runtime::ActivationObserver<ReferenceTensor, Error> for ReferenceExternalSpanObserver {
    fn supports_prefill_spans(&self) -> bool {
        self.case.span_supported
    }
    fn supports_prefill_context(&self) -> bool {
        self.case.context_supported
    }
    fn requires_sequence_readout(&self) -> bool {
        self.case.sequence
    }
    fn begin_prefill_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Error> {
        assert!(self.case.span_supported);
        let span = (chunk.input.start, chunk.input.end);
        let mut trace = self.case.trace.borrow_mut();
        if trace.spans.last() != Some(&span) {
            trace.spans.push(span);
        }
        self.start = Some(chunk.input.start);
        Ok(())
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), Error> {
        assert!(self.case.context_supported);
        self.case.trace.borrow_mut().contexts.push(frontier);
        self.start = None;
        Ok(())
    }
    fn observe(&mut self, path: &str, value: &ReferenceTensor) -> Result<(), Error> {
        if !self.case.span_supported {
            let mut trace = self.case.trace.borrow_mut();
            if !trace.whole_complete {
                trace
                    .whole_values
                    .push((path.into(), value.shape().to_vec()));
                if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                    trace.linear_at_finish = reference_trace().linear_outputs;
                    trace.whole_complete = true;
                }
            }
        }
        if self.start == Some(0) && self.case.cancel_after_span {
            self.case.cancellation.cancel();
        }
        Ok(())
    }
    fn finish_prefill(&mut self, committed: bool) {
        let mut trace = self.case.trace.borrow_mut();
        trace.finished.push(committed);
        trace.linear_at_finish = reference_trace().linear_outputs;
        self.start = None;
    }
}

struct ReferenceExternalSpan<'a, A> {
    request: &'a ExternalPredictionCaptureRequest,
    receiver: &'a mut (dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
        ReferenceTensor,
        Error,
    > + 'a),
    original: ReferenceExternalSpanSource<'a>,
    origin: eredu_core::speculative::SpeculativeActivationOrigin,
    _architecture: std::marker::PhantomData<fn() -> A>,
}
impl<A, P, O>
    eredu_runtime::replicated_session::PrefillSpanOperation<
        PreparedCompositeArchitecture<A>,
        ReferenceBackend,
        ReferenceReplicatedMechanisms,
        eredu_runtime::DirectReplicatedTextExecution,
        P,
        O,
    > for ReferenceExternalSpan<'_, A>
where
    A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    P: eredu_runtime::replicated_session::PreparedPrefillSource<
        PreparedCompositeArchitecture<A>,
        ReferenceBackend,
        ReferenceState,
    >,
    O: eredu_runtime::ActivationObserver<ReferenceTensor, Error> + ?Sized,
{
    fn has_speculative_span_authority(&self) -> bool {
        self.original
            .issuer
            .validate_pool(&self.original.pool)
            .is_ok()
    }
    fn score_layout(&self) -> eredu_runtime::replicated_session::PrefillScoreLayout {
        eredu_runtime::replicated_session::PrefillScoreLayout::SelectedPositions
    }

    fn execute<'s>(
        &mut self,
        session: &mut ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            ReferenceBackend,
            ReferenceReplicatedMechanisms,
        >,
        _source: &'s P,
        _prepared: Option<&'s P::Chunk>,
        input: Result<
            <PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<
                ReferenceBackend,
                ReferenceState,
            >>::Input<'s>,
            eredu_runtime::ReplicatedTextSessionError<
                Error,
                eredu_runtime::ResidentUnitWindowError,
                Error,
            >,
        >,
        _identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: &(),
        _observer: &mut O,
    ) -> Result<
        (
            Option<ReferenceTensor>,
            Option<eredu_core::DistributedCommitOutcome>,
        ),
        eredu_runtime::ReplicatedTextSessionError<
            Error,
            eredu_runtime::ResidentUnitWindowError,
            Error,
        >,
    > {
        let role = self
            .original
            .reserve(chunk, self.origin)
            .map_err(eredu_runtime::ReplicatedTextSessionError::Mechanism)?;
        let _role = ReferenceSpanRoleScope::enter(role);
        let preparation = (|| {
            self.receiver.prepare(chunk)?;
            let paths = A::external_prediction_capture_paths(self.request)?
                .ok_or_else(|| Error::backend("reference assistant capture selection differs"))?;
            Ok(ExactReferenceCaptureObserver::new(paths))
        })();
        let mut observer = session.agree_prediction_prefill_preparation(preparation, context)?;
        let captured = observer.values.clone();
        let request = self.request;
        let (mut scores, capture, frontier, target_commit) = session.prefill_capture_span(
            input,
            chunk.output,
            context,
            &mut observer,
            |forward| {
                let values = {
                    let loan = captured.borrow();
                    loan.iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| {
                            value.ok_or_else(|| {
                                Error::backend(format!(
                                    "reference target missed capture path {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                A::external_prediction_capture(request, forward, values)?
                    .ok_or_else(|| Error::backend("reference target omitted selected capture"))
            },
        )?;
        let consumed = self.receiver.consume(chunk, frontier, &mut scores, capture);
        session.agree_prediction_prefill_preparation(consumed, context)?;
        Ok((scores, target_commit))
    }
}

#[test]
fn materialized_external_assistants_use_shared_spans_and_preserve_lazy_context() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            for (config, artifact, dflash) in [
                (
                    production_gemma_target_config(),
                    reference_gemma_assistant_artifact(),
                    false,
                ),
                (
                    production_muse_target_config(),
                    reference_muse_assistant_artifact(),
                    true,
                ),
            ] {
                clear_reference_trace();
                let default = ReferenceExternalCase::new(None, false);
                let baseline =
                    run_reference_external_spans(&config, artifact.path(), Some(default.clone()))
                        .unwrap();
                {
                    let trace = default.trace.borrow();
                    assert_eq!(trace.spans, [(0, 5)], "None must enter the shared driver");
                    assert_eq!(trace.finished, [true]);
                    let heads = trace
                        .linear_at_finish
                        .iter()
                        .filter(|(name, _)| name.ends_with("lm_head.weight"))
                        .collect::<Vec<_>>();
                    assert_eq!(heads.len(), 1, "one actual materialized target head call");
                    assert_eq!(heads[0].1[1], 1, "selection precedes vocabulary projection");
                    assert_eq!(trace.contexts, if dflash { vec![] } else { vec![5] });
                }
                // Existing whole-invocation callbacks keep their ordered full
                // hidden/context/logit values, without receiving new span hooks.
                clear_reference_trace();
                let mut whole = ReferenceExternalCase::new(None, false);
                whole.span_supported = false;
                whole.context_supported = false;
                whole.sequence = true;
                let full =
                    run_reference_external_spans(&config, artifact.path(), Some(whole.clone()))
                        .unwrap();
                assert_eq!(full.tokens, baseline.tokens);
                assert_eq!(full.accepted, baseline.accepted);
                assert_eq!(full.target_tokens, baseline.target_tokens);
                {
                    let trace = whole.trace.borrow();
                    assert!(
                        trace.spans.is_empty()
                            && trace.contexts.is_empty()
                            && trace.finished.is_empty()
                    );
                    assert!(trace.whole_complete);
                    assert_eq!(
                        trace.whole_values.last().unwrap().0,
                        eredu_core::MODEL_LOGITS_OBSERVATION_PATH
                    );
                    assert_eq!(trace.whole_values.last().unwrap().1[1], 5);
                    let heads = trace
                        .linear_at_finish
                        .iter()
                        .filter(|(name, _)| name.ends_with("lm_head.weight"))
                        .collect::<Vec<_>>();
                    assert_eq!(heads.len(), 1);
                    assert_eq!(heads[0].1[1], 5);
                }
                clear_reference_trace();
                let before = ReferenceExternalCase::new(None, false);
                before.cancellation.cancel();
                let stopped =
                    run_reference_external_spans(&config, artifact.path(), Some(before.clone()))
                        .unwrap();
                assert!(stopped.tokens.is_empty());
                assert_eq!(stopped.target_tokens, 0);
                assert_eq!(stopped.publications, 0);
                assert!(before.trace.borrow().spans.is_empty());
                assert!(!reference_trace()
                    .linear_outputs
                    .iter()
                    .any(|(name, _)| name.ends_with("lm_head.weight")));
                if dflash {
                    clear_reference_trace();
                    let mut long = ReferenceExternalCase::new(None, false);
                    long.prompt_positions = 513;
                    let result =
                        run_reference_external_spans(&config, artifact.path(), Some(long.clone()))
                            .unwrap();
                    assert_eq!(result.tokens.len(), 13);
                    let trace = long.trace.borrow();
                    assert_eq!(trace.spans, [(0, 512), (512, 513)]);
                    let heads = trace
                        .linear_at_finish
                        .iter()
                        .filter(|(name, _)| name.ends_with("lm_head.weight"))
                        .collect::<Vec<_>>();
                    assert_eq!(heads.len(), 1);
                    assert_eq!(heads[0].1[1], 1);
                }
                clear_reference_trace();
                let spans = ReferenceExternalCase::new(Some(2), false);
                let result =
                    run_reference_external_spans(&config, artifact.path(), Some(spans.clone()))
                        .unwrap();
                assert_eq!(result.tokens, baseline.tokens);
                assert_eq!(result.tokens.len(), 13);
                assert_eq!(result.accepted, baseline.accepted);
                assert!(result.accepted.len() >= 3);
                assert_eq!(result.target_tokens, baseline.target_tokens);
                {
                    let trace = spans.trace.borrow();
                    assert_eq!(trace.spans, [(0, 2), (2, 4), (4, 5)]);
                    assert_eq!(trace.finished, [true]);
                    if dflash {
                        assert!(trace.contexts.is_empty());
                        assert!(
                            !trace
                                .linear_at_finish
                                .iter()
                                .any(|(name, _)| name == "encoder.fc.weight"),
                            "raw context is not encoded during prefill"
                        );
                        assert!(
                            reference_trace()
                                .linear_outputs
                                .iter()
                                .any(|(name, _)| name == "encoder.fc.weight"),
                            "first proposal encodes the pending context"
                        );
                    } else {
                        assert_eq!(trace.contexts, [2, 4, 5]);
                    }
                }
                let cancelled = ReferenceExternalCase::new(Some(2), true);
                let result =
                    run_reference_external_spans(&config, artifact.path(), Some(cancelled.clone()))
                        .unwrap();
                assert!(result.tokens.is_empty());
                assert!(result.accepted.is_empty());
                assert_eq!(result.publications, 0);
                assert_eq!(result.target_tokens, 2);
                assert_eq!(cancelled.trace.borrow().spans, [(0, 2)]);
                assert_eq!(cancelled.trace.borrow().finished, [false]);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn external_span_completion_failure_retires_roots_before_any_sampling_publication() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let artifact = reference_gemma_assistant_artifact();
            let spans = ReferenceExternalCase::new(Some(2), false);
            let (result, evidence) =
                with_reference_completion_control(ReferenceCompletionMode::FailWait, || {
                    run_reference_external_spans(
                        &production_gemma_target_config(),
                        artifact.path(),
                        Some(spans.clone()),
                    )
                });
            let error =
                result.expect_err("the first capture/seed completion must fail the prefill");
            assert!(error.contains("injected reference exact-completion failure"));
            assert_eq!(evidence.submissions, 1);
            assert_eq!(evidence.waits, 1);
            assert_eq!(evidence.failures, 1);
            assert!(evidence.retained_at_wait > 0);
            assert_eq!(evidence.drops, 1);
            assert_eq!(evidence.released_resources, evidence.retained_at_wait);
            assert_eq!(evidence.publications, 0);
            assert!(evidence.exact_restores >= 1);
            assert_eq!(spans.trace.borrow().finished, [false]);
        })
        .unwrap()
        .join()
        .unwrap();
}

// This shape-only backend has no native tensor allocation. Its descriptive
// report and occurrence records use the actual funded host metadata worker.
#[derive(Debug)]
struct ReferenceShapeOnlyFacts;
impl eredu_nn::workspace::WorkspaceMechanisms for ReferenceShapeOnlyFacts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn operation_bound(
        &self,
        _: &eredu_nn::workspace::WorkspaceOperation,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl eredu_nn::workspace::WorkspaceFactMechanisms for ReferenceShapeOnlyFacts {
    type Error = std::convert::Infallible;
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::memory_fixture::topology())
    }
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}
struct ReferenceExternalSpanSource<'a> {
    pool: eredu_runtime::working_memory::MemoryLedger,
    issuer: eredu_runtime::working_memory::OriginalSpeculativeRequest,
    cursor: eredu_runtime::speculative::external_occurrence::ExternalOccurrenceCursor<'a>,
    metadata: eredu_nn::workspace::WorkspaceContext,
    placement: std::sync::Arc<eredu_core::MemoryPlacement>,
}
impl ReferenceExternalSpanSource<'_> {
    fn reserve(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        origin: eredu_core::speculative::SpeculativeActivationOrigin,
    ) -> Result<eredu_runtime::working_memory::OriginalExternalSpeculativeRole, Error> {
        use eredu_runtime::{
            speculative::external_occurrence::ExternalInvocationKind, working_memory::*,
        };
        let width = chunk.input.end - chunk.input.start;
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: chunk.position,
            input_positions: width,
            max_output_tokens: 0,
            prefill_chunk_positions: width,
            output: chunk.output,
        };
        let span = eredu_core::speculative::SpeculativePrefillSpan {
            prompt_tokens: self.cursor.plan().prefill_geometry().input_positions,
            input_start: chunk.input.start,
            input_end: chunk.input.end,
            position: chunk.position,
            hidden_start: chunk.input.start,
            token_start: chunk.input.start,
            sequence: width,
            seed_start: chunk.position,
        };
        let invocation = self
            .cursor
            .plan()
            .invocation(
                ExternalInvocationKind::TargetPrefill,
                geometry,
                Some(span),
                origin,
            )
            .map_err(Error::backend)?;
        let claim = self.cursor.claim(invocation).map_err(Error::backend)?;
        let report = quote_inference_workspace_with_context(geometry, &self.metadata, |_| {
            self.metadata.begin_state_span([])?;
            self.metadata.finish_report(&[])
        })
        .map_err(Error::backend)?;
        let requirements = SpeculativeInvocationRequirements::new(
            report.span_workspace_plan(),
            Some(0),
            Some(0),
            Some(0),
            Some(
                u64::try_from(
                    std::mem::size_of::<ReferenceExternalCompletion>()
                        + std::mem::size_of::<ReferenceSpanRoleScope>(),
                )
                .map_err(Error::backend)?,
            ),
            self.placement.clone(),
        )
        .map_err(Error::backend)?;
        self.issuer
            .reserve_external_role(claim, requirements)
            .map_err(Error::backend)
    }
}

thread_local! {
    static REFERENCE_SPAN_ROLE: RefCell<Option<eredu_runtime::working_memory::OriginalExternalSpeculativeRole>> = const { RefCell::new(None) };
}
struct ReferenceSpanRoleScope(
    Option<eredu_runtime::working_memory::OriginalExternalSpeculativeRole>,
);
impl ReferenceSpanRoleScope {
    fn enter(role: eredu_runtime::working_memory::OriginalExternalSpeculativeRole) -> Self {
        Self(REFERENCE_SPAN_ROLE.with(|slot| slot.replace(Some(role))))
    }
}
impl Drop for ReferenceSpanRoleScope {
    fn drop(&mut self) {
        REFERENCE_SPAN_ROLE.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}
