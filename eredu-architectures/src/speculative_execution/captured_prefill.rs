//! Materialized embedded seed consumption inside the shared guarded span.
use super::*;
use eredu_core::{GenerationCancellationToken, SpeculativePrefillOutcome};
use eredu_nn::{Error as NeuralError, NeuralBackend, Tensor};
use eredu_runtime::replicated_session::{
    PrefillSourceOutcome, PrefillSpanOperation, ReplicatedTextSessionError,
};

/// Selected initial scores and one target seed; verification keeps its full output.
pub struct PrefillValues<T, L> {
    pub(super) logits: L,
    pub(super) capture: EmbeddedPredictionTensor<T>,
    pub(super) evaluated: usize,
}

pub(super) fn legacy<S: EmbeddedPredictionStrategy<M> + ?Sized, M: SpeculativeTensorMechanisms>(
    strategy: &mut S,
    input: S::Input,
    cache: &mut S::TargetCache,
    observers: &mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    context: M::Context<'_>,
) -> Result<PrefillValues<M::Tensor, M::Logits>, M::Error> {
    let mut output = strategy.prefill_target(input, cache, context, observers.internal())?;
    output.capture = observers.tensor::<M>(EMBEDDED_TARGET_CAPTURE_PATH, output.capture, None, context)?;
    let count = M::sequence_len(&output.tokens)?;
    let scores = M::sequence_len(&output.logits)?;
    let captures = M::sequence_len(&output.capture)?;
    if count == 0 {
        return Err(M::empty_prediction_input());
    }
    if scores != count || captures != count {
        return Err(M::invalid_prediction_output(scores, captures, count, None));
    }
    let tokens = output.tokens.clone();
    strategy.seed_prediction_cache(&output, &tokens, cache, context, observers.internal())?;
    Ok(PrefillValues {
        logits: M::logits_row_with_source(&output.logits, count - 1, strategy.target_logit_evidence(cache), context)?,
        capture: M::tensor_row_with_source(&output.capture,count-1,strategy.target_logit_evidence(cache),context)?,
        evaluated: count,
    })
}

// Kept outside the target-state exchange and final score-selection result.
// The callback is host-only and cannot publish a successful partial prefill.
struct LogicalPrefill<'a, T, L, E> {
    observers: &'a mut EmbeddedPredictionObservers<T, L, E>,
    success: bool,
    span_protocol: bool,
}
impl<T, L, E> Drop for LogicalPrefill<'_, T, L, E> {
    fn drop(&mut self) {
        if self.span_protocol {
            if let Some(observer) = self.observers.internal() {
                observer.finish_prefill_reductions(self.success);
            }
        }
    }
}

struct CapturedSeed<'a, 'c, A, B, S, SM, D, P, N, M, H>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    SM::PolicyError: std::fmt::Display,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = NeuralError>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    phase: &'a mut H,
    extension: &'a mut P,
    lane: PredictionPhaseState<'a, P::LaneState>,
    target_evidence: PredictionPhaseEvidence<'a, S>,
    selected: &'a eredu_runtime::SelectedSpeculativeRealization,
    binding: Option<&'a eredu_runtime::SpeculativeIdentity>,
    observers: &'a mut EmbeddedPredictionObservers<B::Tensor, M::Logits, M::Error>,
    context: M::Context<'c>,
    carry: Option<EmbeddedPredictionTensor<B::Tensor>>,
    next: u64,
    prompt_tokens: u64,
    seed_base: Option<u64>,
    span_protocol: bool,
    _types: PhantomData<fn() -> (A, S, SM, D, N)>,
}

impl<A, B, S, SM, D, P, I, N, M, H>
    ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M, H>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B> + 'static,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = NeuralError>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>
        + ReplicatedPredictionNative<A, B, S, M>,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    I: ReplicatedPredictionInput<A, B, S, M::Error, Input = N::Input>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    H: ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>,
{
    pub(super) fn prefill_captured_spans<'c>(
        &mut self,
        input: I::Input,
        cache: &mut EmbeddedPredictionCache<S, P::LaneState>,
        observers: &mut EmbeddedPredictionObservers<B::Tensor, M::Logits, M::Error>,
        cancellation: &GenerationCancellationToken,
        context: M::Context<'c>,
    ) -> Result<SpeculativePrefillOutcome<PrefillValues<B::Tensor, M::Logits>>, M::Error> {
        M::validate_outer_tensor_observer(observers.tensors.is_some(), context)?;
        let compatible = observers.supports_tensor_prefill_spans()
            && observers
                .internal
                .as_ref()
                .is_none_or(|observer| observer.supports_prefill_spans());
        // An ordinary callback without span semantics keeps its complete logical
        // invocation. The shared driver still selects its declared score demand.
        let span_protocol = I::requested_chunks(&input).is_some() || compatible;
        let mut logical = LogicalPrefill {
            observers,
            success: false,
            span_protocol,
        };
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            ..
        } = self;
        let result = N::with_prefill_source(lowerer, input, context, |prepared, context| {
            let tensor_context=N::target_context(context);
            let mut prepared = Some(prepared.and_then(|prepared| {
                cache
                    .bind_prepared_input(prepared.identity())
                    .map_err(|error| match error {
                        EmbeddedPredictionCacheError::Prepared(cause) => N::session_failure(cause),
                        other => N::session_cause_with_context(other,context),
                    })?;
                let shape = prepared.shape().map_err(|cause|N::session_cause_with_context(cause,context))?;
                Ok((prepared, shape))
            }));
            let shape = prepared
                .as_ref()
                .and_then(|p| p.as_ref().ok())
                .map(|(_, shape)| *shape);
            let chunk = prepared
                .as_ref()
                .and_then(|p| p.as_ref().ok())
                .and_then(|(p, _)| p.chunk_positions());
            let whole_input = prepared
                .as_ref()
                .and_then(|p| p.as_ref().ok())
                .is_some_and(|(p, _)| p.requires_whole_input());
            let chunk = if chunk.is_none() && (!compatible || whole_input) {
                shape.and_then(|shape| std::num::NonZeroU64::new(shape[1]))
            } else {
                chunk
            };
            let session_error = N::prepare_session_cause(context, Some(0))?;
            let mut target = cache
                .take_target()
                .ok_or_else(|| N::session_cause_with_context(EmbeddedPredictionCacheError::TargetStateActive,context))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut target, tensor_context)
            {
                cache.restore_target(target);
                return Err(N::session_cause_with_context(error,context));
            }
            let (target_evidence, prediction_evidence) = match &mut cache.prepared {
                Some(prepared) => (
                    Some((&prepared.target, &mut prepared.target_evidence)),
                    Some((&prepared.prediction, &mut prepared.prediction_evidence)),
                ),
                None => (None, None),
            };
            let mut seed = CapturedSeed::<A, B, S, SM, D, P, N, M, H> {
                phase: &mut self.phase,
                extension,
                lane: PredictionPhaseState::new(&mut cache.prediction, prediction_evidence),
                target_evidence: PredictionPhaseEvidence::new(target_evidence),
                selected,
                binding: cache.prepared_input.as_ref(),
                observers: &mut *logical.observers,
                context,
                carry: None,
                next: 0,
                prompt_tokens: shape.map_or(0, |shape| shape[1]),
                seed_base: None,
                span_protocol,
                _types: PhantomData,
            };
            let demand = if seed
                .observers
                .internal
                .as_ref()
                .is_some_and(|observer| observer.requires_sequence_readout())
            {
                eredu_core::OutputDemand::Sequence
            } else {
                eredu_core::OutputDemand::LastPosition
            };
            let result = session.try_prefill_unbudgeted_source_with_operation(
                shape,
                chunk,
                demand,
                |geometry| {
                    if !compatible && span_protocol {
                        return Err(N::neural_cause_with_context(
                            eredu_core::speculative::SpeculativeControlError::Unsupported(
                                "observer does not support explicit prefill spans",
                            ),context));
                    }
                    prepared
                        .take()
                        .expect("source factory called once")
                        .map_err(|cause|N::neural_cause_with_context(cause,context))?
                        .0
                        .into_source(geometry)
                },
                cancellation,
                tensor_context,
                &mut eredu_runtime::NoopObserver,
                &mut seed,
            );
            let carry = seed.carry.take();
            drop(seed);
            let restored =
                match session.exchange_prediction_target_state(&mut target, tensor_context) {
                    Ok(()) => Ok(()),
                    Err(error) => match session.recover_prediction_target_state_after_failure(&mut target) {
                        Ok(()) => Err(N::session_cause_with_context(error,context)),
                        Err(recovery) => Err(N::session_cause_with_context(TargetStateRecoveryFailure { exchange:error,recovery },context)),
                    }

                };
            cache.restore_target(target);
            let result = result.map_err(session_error);
            let progress = match (result, restored) {
                (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
                (Ok(value), Ok(())) => value,
            };
            let evaluated =
                usize::try_from(progress.completed_positions).map_err(|cause|N::session_cause_with_context(cause,context))?;
            match progress.outcome {
                PrefillSourceOutcome::Unavailable => {
                    if let Some(Err(error)) = prepared.take() {
                        return Err(error);
                    }
                    Err(M::observation_error(
                        "selected embedded prefill source is unavailable",
                    ))
                }
                PrefillSourceOutcome::Cancelled => Ok(SpeculativePrefillOutcome::Cancelled {
                    evaluated_tokens: evaluated,
                }),
                PrefillSourceOutcome::Complete(scores) => {
                    cache
                        .retain_capture_generation(N::generation)
                        .map_err(|cause|N::session_cause_with_context(cause,context))?;
                    Ok(SpeculativePrefillOutcome::Complete(PrefillValues {
                        logits: M::selected_prefill_logits_with_source(scores.ok_or_else(|| {
                            M::observation_error("selected target prefill omitted final scores")
                        })?, cache.target_logit_evidence(), context)?,
                        capture: carry.ok_or_else(M::empty_prediction_input)?,
                        evaluated,
                    }))
                }
            }
        });
        logical.success = matches!(&result, Ok(SpeculativePrefillOutcome::Complete(_)));
        result
    }
}

impl<A, B, S, SM, D, P, N, M, H, Source, O> PrefillSpanOperation<A, B, SM, D, Source, O>
    for &mut CapturedSeed<'_, '_, A, B, S, SM, D, P, N, M, H>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B> + 'static,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = NeuralError>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>
        + ReplicatedPredictionNative<A, B, S, M>,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    Source: PredictionPrefillSource<A, B, S>,
    O: eredu_runtime::ActivationObserver<B::Tensor, NeuralError> + ?Sized,
    H: ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>,
{
    fn score_layout(&self) -> eredu_runtime::replicated_session::PrefillScoreLayout {
        M::prefill_score_layout(self.context)
    }

    fn prepare(
        &mut self,
        source: &Source,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Source::Chunk, NeuralError> {
        N::prepare_prefill_chunk(source, chunk, self.context)
    }

    fn execute<'s>(
        &mut self,
        session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        source: &'s Source,
        prepared: Option<&'s Source::Chunk>,
        input: Result<
            A::Input<'s>,
            ReplicatedTextSessionError<NeuralError, SM::PolicyError, SM::Error>,
        >,
        _identity: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<
        (
            Option<B::Tensor>,
            Option<eredu_core::DistributedCommitOutcome>,
        ),
        ReplicatedTextSessionError<NeuralError, SM::PolicyError, SM::Error>,
    > {
        M::with_tensor_source(
            prepared.and_then(|chunk|source.completed_token_packet(chunk)).and_then(|packet|packet.evidence()),
            self.context,
            |execution_context| N::validate_captured_span(execution_context, || {
            let span = eredu_core::speculative::SpeculativePrefillSpan {
                prompt_tokens: self.prompt_tokens,
                input_start: chunk.input.start,
                input_end: chunk.input.end,
                position: chunk.position,
                hidden_start: chunk.input.start,
                token_start: chunk.input.start,
                sequence: chunk.input.end - chunk.input.start,
                seed_start: chunk.position,
            };
            let preparation = if self.next == chunk.input.start {
                Ok(())
            } else {
                Err(N::neural_cause_with_context(PredictionContractFailure("captured prefill source did not advance contiguously"),execution_context))
            };
            session
                .agree_prediction_prefill_preparation(preparation, context)
                .map_err(|e| N::session_cause_with_context(e,execution_context))?;
            if self.span_protocol {
                if let Some(observer) = self.observers.internal() {
                    observer.set_prefill_reduction_geometry(
                        eredu_core::speculative::SpeculativePrefillReductionGeometry {
                            target_sequence: self.prompt_tokens,
                            prediction_sequence: u64::try_from(
                                self.extension.prefill_sequence_len(
                                    usize::try_from(self.prompt_tokens)
                                        .map_err(|cause|N::session_cause_with_context(cause,execution_context))?,
                                ),
                            )
                            .map_err(|cause|N::session_cause_with_context(cause,execution_context))?,
                        },
                    );
                }
            }
            let published = span_neural::<A, B, S, SM, D, N, M, _>(
                session,
                self.observers.internal(),
                SpeculativeActivationPhase::TargetPrefill,
                span,
                self.span_protocol,
                context,
                execution_context,
                |session, observer| {
                    self.phase.run_target(
                        session, self.target_evidence.reborrow(),
                        prepared.map(|prepared| source.tokens(prepared)),
                        SpeculativeActivationPhase::TargetPrefill,
                        chunk.output, Some(span), execution_context, observer,
                        |session, observer, phase_context| {
                    match observer {
                        Some(observer) => {
                            session.prefill_prediction_span(input, chunk.output, context, observer)
                        }
                        None => session.prefill_prediction_span(
                            input,
                            chunk.output,
                            context,
                            &mut eredu_runtime::NoopObserver,
                        ),
                    }
                    .map_err(|error| {
                        N::session_cause_with_context(error,phase_context)
                    })
                        },
                    )
                },
            )?;
            let generation = published.generation();
            let (scores, mut capture, target_commit) = published.into_parts();
            let prepared_seed = (|| {
                let binding = self.binding.ok_or_else(|| {
                    M::observation_error("captured prefill lost the exact input binding")
                })?;
                let lane_identity = self.selected.lane_identity_ref(binding, generation);
                self.extension
                    .validate_capture(self.selected, &lane_identity, N::shape(&capture))
                    .map_err(|cause|N::session_cause_with_context(cause,execution_context))?;
                capture = self.observers.tensor::<M>(
                    EMBEDDED_TARGET_CAPTURE_PATH, capture,
                    self.span_protocol.then_some(chunk), execution_context,
                )?;
                self.extension
                    .validate_capture(self.selected, &lane_identity, N::shape(&capture))
                    .map_err(|cause|N::session_cause_with_context(cause,execution_context))?;
                let tokens = prepared
                    .map(|prepared| source.tokens(prepared))
                    .ok_or_else(|| {
                        M::observation_error("published target has no prepared chunk")
                    })?;
                let width = usize::try_from(span.sequence).map_err(|cause|N::session_cause_with_context(cause,execution_context))?;
                if M::sequence_len(&capture)? != width || M::sequence_len(tokens)? != width {
                    return Err(M::invalid_prediction_output(
                        width,
                        M::sequence_len(&capture)?,
                        M::sequence_len(tokens)?,
                        Some(width),
                    ));
                }
                let capture=M::retain_tensor_packet(capture,self.target_evidence.evidence().cloned(),execution_context)?;
                let token_packet=M::prefill_token_packet(tokens,
                    prepared.and_then(|prepared|source.completed_token_packet(prepared)),execution_context)?;
                let alignment = self.extension.prefill_alignment();
                let shifted = alignment == eredu_core::speculative::PredictionPrefillAlignment::NextToken;
                let frontier = self.extension.prefill_frontier(self.lane.state_mut()).map_err(|cause|N::session_cause_with_context(cause,execution_context))?;
                let base = *self.seed_base.get_or_insert(frontier);
                let seed_geometry = alignment.seed(span, base).ok_or_else(|| {
                    M::observation_error("prediction seed span geometry overflow or mismatch")
                })?;
                if frontier != seed_geometry.frontier_before() {
                    return Err(M::observation_error(
                        "prediction seed frontier differs from completed spans",
                    ));
                }
                let skip = usize::try_from(seed_geometry.token_skip()).map_err(|cause|N::session_cause_with_context(cause,execution_context))?;
                let after = seed_geometry.frontier_after();
                let carry = M::tensor_row_packet(&capture,width-1,execution_context)?;
                let seed = if let Some(seed_span) = seed_geometry.invocation() {
                    let (hidden, next) = if shifted {
                        let hidden = match self.carry.as_ref() {
                            Some(previous) if width == 1 => previous.clone(),
                            previous => {
                                let prefix = M::tensor_prefix_packet(&capture,width-1,execution_context)?;
                                match previous {
                                    Some(previous) => M::tensor_concatenate_packet(previous,&prefix,execution_context,
                                        |left,right|B::Tensor::concatenate(&[left.clone(),right.clone()],1,context)
                                            .map_err(|cause|N::session_cause_with_context(cause,execution_context)))?,
                                    None => prefix,
                                }
                            }
                        };
                        (hidden, M::token_range_packet(&token_packet,skip,width,execution_context)?)
                    } else {
                        (capture.clone(),token_packet.clone())
                    };
                    Some((hidden, next, seed_span))
                } else {
                    None
                };
                Ok::<_, M::Error>((capture, carry, seed, after))
            })();
            let (capture, carry, seed, expected_frontier) = session
                .agree_prediction_prefill_preparation(
                    prepared_seed.map_err(|cause|N::neural_cause_with_context(cause,execution_context)),
                    context,
                )
                .map_err(|error| {
                    N::session_cause_with_context(error,execution_context)
                })?;
            if let Some((hidden, next, seed_span)) = seed {
                M::with_tensor_source(capture.evidence(),execution_context,|execution_context|
                M::with_tensor_source(hidden.evidence(),execution_context,|execution_context|
                M::with_tensor_source(next.evidence(),execution_context,|execution_context|
                M::with_tensor_source(carry.evidence(),execution_context,|execution_context|{
                let extension = &mut *self.extension;
                let lane = self.lane.reborrow();
                let phase = &mut *self.phase;
                let full_context = execution_context;
                let equation = PredictionEquation::Prefill {
                    target_capture: &*capture, hidden: &*hidden, tokens: &*next,
                };
                let execute_seed =
                    |session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
                     observer: Option<
                        &mut dyn eredu_runtime::ActivationObserver<B::Tensor, NeuralError>,
                    >| {
                        phase.run(
                            session, extension, lane, eredu_runtime::ExpertPass::Prefill,
                            &equation, Some(PredictionCompletionPoint::CapturedSeed), Some(seed_span), full_context, observer,
                            |session, extension, lane, observer, phase_context| {
                        let mut noop = eredu_runtime::NoopObserver;
                        let observed = observer.is_some();
                        let observer = observer.unwrap_or(&mut noop);
                        session.with_prediction_observation(
                            eredu_runtime::ExpertPass::Prefill,
                            context,
                            observer,
                            &mut (extension, lane),
                            |session, (extension, lane), observer| {
                                execute_prediction_equation::<A, B, S, N, P, _>(
                                    equation,
                                    extension,
                                    &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                                        session,
                                        context,
                                        execution_context:phase_context,
                                        _native: PhantomData,
                                    },
                                    lane,
                                    observed.then_some(observer),
                                    |token| N::token(token, phase_context),
                                ).map(|output| {
                                    let PredictionEquationOutput::StateOnly = output else {
                                        unreachable!("captured prefill equation result")
                                    };
                                })
                            },
                            |_, (extension, lane), _| {
                                N::complete_prediction_state(
                                    &**extension, lane, &[&*carry], PredictionCompletionPoint::CapturedSeed,
                                    PredictionCompletionSources::default(), phase_context,
                                )
                            },
                            |error| {
                                N::session_cause_with_context(error,execution_context)
                            },
                        )
                            },
                        )
                    };
                span_neural::<A, B, S, SM, D, N, M, _>(
                    session,
                    self.observers.internal(),
                    SpeculativeActivationPhase::PredictionPrefill,
                    seed_span,
                    self.span_protocol,
                    context,
                    execution_context,
                    execute_seed,
                )?;
                Ok(())
                }))))?;
            } else {
                let (lane, evidence) = self.lane.completion_sources();
                let completion = N::complete_prediction_state(
                    &*self.extension, lane, &[&*carry], PredictionCompletionPoint::CapturedCarry,
                    PredictionCompletionSources {state: evidence, outputs: carry.evidence()},
                    execution_context,
                ).map_err(|cause|N::neural_cause_with_context(cause,execution_context));
                session
                    .agree_prediction_prefill_preparation(completion, context)
                    .map_err(|error| {
                        N::session_cause_with_context(error,execution_context)
                    })?;
            }
            let frontier = self
                .extension
                .prefill_frontier(self.lane.state_mut())
                .and_then(|actual| {
                    if actual == expected_frontier {
                        Ok(())
                    } else {
                        Err(N::neural_cause_with_context(PredictionContractFailure("prediction seed did not publish the complete span frontier"),execution_context))
                    }
                });
            session
                .agree_prediction_prefill_preparation(frontier, context)
                .map_err(|error| {
                    N::session_cause_with_context(error,execution_context)
                })?;
            if self.span_protocol && chunk.input.end == self.prompt_tokens {
                let complete = self
                    .observers
                    .internal()
                    .map_or(Ok(()), |observer| observer.complete_prefill_reductions());
                session
                    .agree_prediction_prefill_preparation(
                        complete.map_err(|cause|N::neural_cause_with_context(cause,execution_context)),
                        context,
                    )
                    .map_err(|error| {
                        N::session_cause_with_context(error,execution_context)
                    })?;
            }
            self.carry = Some(carry);
            self.next = chunk.input.end;
            Ok((scores, target_commit))
        }))
        .map_err(|error| {
            ReplicatedTextSessionError::Architecture(N::neural_cause_with_context(error,self.context))
        })
    }
}

fn span_neural<A, B, S, SM, D, N, M, R>(
    session: &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
    observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    phase: SpeculativeActivationPhase,
    span: eredu_core::speculative::SpeculativePrefillSpan,
    span_protocol: bool,
    context: &<B::Tensor as Tensor>::Context,
    execution_context:M::Context<'_>,
    operation: impl FnOnce(
        &mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, NeuralError>>,
    ) -> Result<R, M::Error>,
) -> Result<R, M::Error>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    S: eredu_runtime::RuntimeState<B>,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = NeuralError>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    N: ReplicatedPredictionNative<A, B, S, M>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    let Some(observer) = observer else {
        session
            .agree_prediction_prefill_preparation(Ok(()), context)
            .map_err(|error| N::session_cause_with_context(error,execution_context))?;
        let result = operation(session, None);
        return session
            .agree_prediction_prefill_preparation(
                result.map_err(|cause|N::neural_cause_with_context(cause,execution_context)),
                context,
            )
            .map_err(|error| N::session_cause_with_context(error,execution_context));
    };
    struct Invocation<'a, T, E> {
        observer: &'a mut dyn SpeculativeActivationObserver<T, E>,
        success: bool,
        span_protocol: bool,
    }
    impl<T, E> Drop for Invocation<'_, T, E> {
        fn drop(&mut self) {
            self.observer.finish_activation_invocation(self.success);
            if self.span_protocol {
                self.observer.set_prefill_span(None);
            }
        }
    }
    let mut invocation = Invocation {
        observer,
        success: false,
        span_protocol,
    };
    if span_protocol {
        invocation.observer.set_prefill_span(Some(span));
    }
    let admission = invocation.observer.begin_activation_invocation(
        phase,
        usize::try_from(span.sequence).map_err(|cause|N::session_cause_with_context(cause,execution_context))?,
    );
    session
        .agree_prediction_prefill_preparation(
            admission.map_err(|cause|N::neural_cause_with_context(cause,execution_context)),
            context,
        )
        .map_err(|error| N::session_cause_with_context(error,execution_context))?;
    let mut operation = Some(operation);
    let mut output = None;
    let result = invocation
        .observer
        .with_activation_observer(&mut |observer| {
            let mut bridge = eredu_runtime::inspection::ObserverErrorBridge::new(
                observer,
                |cause|N::session_cause_with_context(cause,execution_context),
                |error: &M::Error| N::neural_observer_error(error,execution_context),
            );
            let result =
                operation.take().expect("one admitted forward")(session, Some(&mut bridge));
            output = Some(bridge.resolve(result)?);
            Ok(())
        });
    let completed = result
        .and_then(|()| invocation.observer.complete_activation_invocation())
        .and_then(|()| {
            output.ok_or_else(|| {
                M::observation_error("observer did not execute the admitted forward")
            })
        });
    let output = session
        .agree_prediction_prefill_preparation(
            completed.map_err(|cause|N::neural_cause_with_context(cause,execution_context)),
            context,
        )
        .map_err(|error| N::session_cause_with_context(error,execution_context))?;
    invocation.success = true;
    Ok(output)
}
