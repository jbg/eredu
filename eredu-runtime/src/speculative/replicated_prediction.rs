//! Shared orchestration for a typed replicated target and its paired prediction equations.

use std::marker::PhantomData;

use eredu_core::speculative::{
    GenerationExecutorTypes, SpeculativeActivationPhase, SpeculativeControlError,
};

use crate::inspection::SpeculativeActivationObserver;

use super::{
    validate_prediction_output, EmbeddedPredictionCache, EmbeddedPredictionCacheError,
    EmbeddedPredictionDraftCache, EmbeddedPredictionObservers, EmbeddedPredictionOutput,
    EmbeddedPredictionStrategy, PredictionTargetLifecycle, ReplicatedPredictionInput,
    ReplicatedPredictionNative, SpeculativeTensorMechanisms,
};

mod observation;

/// Portable embedded strategy over one typed replicated session and paired extension.
pub struct ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M>
where
    B: eredu_core::SubmissionBackend<
        Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    >,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: super::MaterializedPredictionExecutor<A, B, N>,
    N: super::PredictionExtensionMaterializer<B>,
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    session: &'a mut crate::ReplicatedTextSession<A, B, SM, D>,
    extension: &'a mut P,
    selected: &'a crate::SelectedSpeculativeRealization,
    input: I,
    cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    _native: PhantomData<fn() -> (S, N, M)>,
}

impl<'a, A, B, S, SM, D, P, I, N, M>
    ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: super::MaterializedPredictionExecutor<A, B, N>,
    N: super::PredictionExtensionMaterializer<B>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    /// Completes the statically checked pairing before any executor erasure.
    pub fn new(
        session: &'a mut crate::ReplicatedTextSession<A, B, SM, D>,
        extension: &'a mut P,
        selected: &'a crate::SelectedSpeculativeRealization,
        input: I,
        cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Self {
        Self {
            session,
            extension,
            selected,
            input,
            cache_context,
            _native: PhantomData,
        }
    }

    /// Keeps one auxiliary invocation in the ordinary session transaction. The
    /// unobserved path retains its existing execution and validation behavior.
    fn prediction_phase<R>(
        session: &mut crate::ReplicatedTextSession<A, B, SM, D>,
        extension: &mut P,
        lane: &mut P::LaneState,
        pass: crate::ExpertPass,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, A::Error>>,
        execute: impl FnOnce(
            &mut P,
            &mut ReplicatedPredictionInvoker<'_, A, B, S, SM, D, N, M>,
            &mut P::LaneState,
            Option<&mut dyn crate::ActivationObserver<B::Tensor, A::Error>>,
        ) -> Result<R, M::Error>,
        complete: impl FnOnce(&P, &mut P::LaneState, &R) -> Result<(), eredu_core::BackendFailure>,
    ) -> Result<R, M::Error>
    where
        N: ReplicatedPredictionNative<B, S, M>,
    {
        N::validate(|| {
            let Some(observer) = observer else {
                return execute(
                    extension,
                    &mut ReplicatedPredictionInvoker {
                        session,
                        context,
                        _native: PhantomData,
                    },
                    lane,
                    None,
                );
            };
            session.with_prediction_observation(
                pass,
                context,
                observer,
                &mut (&mut *extension, lane),
                |session, (extension, lane), observer| {
                    execute(
                        extension,
                        &mut ReplicatedPredictionInvoker {
                            session,
                            context,
                            _native: PhantomData,
                        },
                        lane,
                        Some(observer),
                    )
                },
                |_, (extension, lane), output| {
                    complete(extension, lane, output).map_err(N::session_failure)
                },
                |error| N::session_failure(eredu_core::BackendFailure::from_error(error)),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_prediction_rows(
        session: &mut crate::ReplicatedTextSession<A, B, SM, D>,
        extension: &mut P,
        capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut P::LaneState,
        sequence: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(), M::Error>
    where
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        N: ReplicatedPredictionNative<B, S, M>,
    {
        observation::neural(
            observer,
            SpeculativeActivationPhase::PredictionPrefill,
            sequence,
            N::session_error,
            |observer| {
                Self::prediction_phase(
                    session,
                    extension,
                    lane,
                    crate::ExpertPass::Prefill,
                    context,
                    observer,
                    |extension, invoker, lane, observer| {
                        extension.prefill_observed::<S, _>(
                            invoker, capture, hidden, tokens, lane, observer,
                        )
                    },
                    |extension, lane, _| extension.complete_state(lane, &[], context),
                )
            },
        )
    }

    /// Realizes one typed target/prediction lane before final executor erasure.
    pub fn new_cache(&mut self) -> Result<EmbeddedPredictionCache<S, P::LaneState>, M::Error>
    where
        N: ReplicatedPredictionNative<B, S, M>,
    {
        let state = self
            .session
            .prepare_prediction_target_state(self.cache_context)
            .map_err(N::session_error)?;
        Ok(EmbeddedPredictionCache::new(
            state,
            self.extension.new_state(),
        ))
    }
}

struct ReplicatedPredictionInvoker<'a, A, B, S, SM, D, N, M>
where
    B: eredu_core::SubmissionBackend<
        Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    >,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    session: &'a mut crate::ReplicatedTextSession<A, B, SM, D>,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    _native: PhantomData<fn() -> (N, M)>,
}

impl<A, B, S, SM, D, N, M> super::PredictionOperationInvoker<A, B, S>
    for ReplicatedPredictionInvoker<'_, A, B, S, SM, D, N, M>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    N: ReplicatedPredictionNative<B, S, M>,
{
    type Error = M::Error;

    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Self::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
    {
        self.session
            .apply_prediction_target_operation(operation, self.context)
            .map_err(|error| N::session_failure(eredu_core::BackendFailure::from_error(error)))
    }

    fn invalid(message: String) -> Self::Error {
        N::session_error(message)
    }
}

impl<A, B, S, SM, D, P, I, N, M> EmbeddedPredictionStrategy<M>
    for ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B> + 'static,
    A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: super::MaterializedPredictionExecutor<A, B, N>,
    N: super::PredictionExtensionMaterializer<B> + ReplicatedPredictionNative<B, S, M>,
    I: ReplicatedPredictionInput<A, B, S, M::Error>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    type Input = I::Input;
    type TargetCache = EmbeddedPredictionCache<S, P::LaneState>;
    type PredictionCache = EmbeddedPredictionDraftCache<P::LaneState>;
    type Telemetry = N::Telemetry;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.session.trim_parameter_conversions().map(|target| {
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target,
                external_drafter: None,
            }
        })
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        self.session
            .parameter_conversion_retention()
            .map(|target| {
                Some(eredu_core::residency::ExecutionConversionRetentionReport {
                    target,
                    external_drafter: None,
                })
            })
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::TargetCache,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        let Some(target) = cache.target() else {
            return Ok(None);
        };
        let Some(parameters) = self
            .session
            .parameter_memory_observation()
            .map_err(eredu_core::BackendFailure::from_error)?
        else {
            return Ok(None);
        };
        let Some(prediction) = self
            .extension
            .memory_observation(cache.prediction(), additional)
        else {
            return Ok(None);
        };
        let (current_state_bytes, peak_state_bytes) =
            N::TargetState::memory_bounds(target, additional);
        Ok(Some(
            eredu_core::speculative::SpeculativeContinuationObservation {
                target: eredu_core::speculative::SpeculativeModelMemoryObservation {
                    parameter_conversions: parameters.parameter_conversions.clone(),
                    current_positions: N::TargetState::position(target)?,
                    current_state_bytes,
                    peak_state_bytes,
                    parameters: parameters.parameters,
                    available: parameters.available,
                    allocator_cache_limit: parameters.allocator_cache_limit,
                },
                draft: None,
                embedded: Some(eredu_core::speculative::EmbeddedContinuationObservation {
                    prediction,
                    retained_feature_bytes: None,
                }),
                parameter_conversions: parameters.parameter_conversions,
                seed_bytes: None,
            },
        ))
    }

    fn control_target_estimate(
        &self,
        cache: &Self::TargetCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        cache.snapshot_estimate(
            N::TargetState::estimate(cache.target()?),
            self.extension.snapshot_estimate(cache.prediction()),
        )
    }

    fn control_prediction_estimate(
        &self,
        cache: &Self::PredictionCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        cache.snapshot_estimate(self.extension.snapshot_estimate(cache.prediction()))
    }

    fn control_target_snapshot<'a>(
        &self,
        cache: &Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<Option<Self::TargetCache>, SpeculativeControlError> {
        let Some(target) = cache.target() else {
            return Ok(None);
        };
        let context = N::target_context(context);
        let Some(target) =
            N::TargetState::snapshot(target, context).map_err(SpeculativeControlError::backend)?
        else {
            return Ok(None);
        };
        let Some(prediction) = self
            .extension
            .snapshot(cache.prediction(), context)
            .map_err(SpeculativeControlError::Backend)?
        else {
            return Ok(None);
        };
        Ok(Some(cache.with_snapshot_storage(target, prediction)))
    }

    fn control_prediction_snapshot<'a>(
        &self,
        cache: &Self::PredictionCache,
        context: M::Context<'a>,
    ) -> Result<Option<Self::PredictionCache>, SpeculativeControlError> {
        let Some(prediction) = self
            .extension
            .snapshot(cache.prediction(), N::target_context(context))
            .map_err(SpeculativeControlError::Backend)?
        else {
            return Ok(None);
        };
        Ok(Some(cache.with_snapshot_storage(prediction)))
    }

    fn supports_internal_observations(&self) -> bool {
        self.extension.activation_execution(self.selected).is_some()
    }

    fn proposal_capacity(&self) -> usize {
        self.selected
            .requirements()
            .strategy()
            .proposal_capacity()
            .get()
    }

    fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, M::Error> {
        cache
            .checkpoint(N::TargetState::checkpoint)
            .map_err(|error| N::session_failure(eredu_core::BackendFailure::from_error(error)))
    }

    fn take_telemetry(&mut self) -> Result<Self::Telemetry, M::Error> {
        N::take_telemetry()
    }

    fn take_verification_telemetry(
        &mut self,
        _output: &mut EmbeddedPredictionOutput<B::Tensor>,
    ) -> Result<Self::Telemetry, M::Error> {
        N::take_telemetry()
    }

    fn prefill_with_prediction<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observers: &mut EmbeddedPredictionObservers<B::Tensor, M::Logits, M::Error>,
    ) -> Result<(EmbeddedPredictionOutput<B::Tensor>, usize), M::Error> {
        let tensor_context = N::target_context(context);
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            ..
        } = self;
        let maximum_chunk_tokens = extension
            .maximum_prefill_chunk_tokens(selected)
            .unwrap_or(usize::MAX);
        let mut last = None;
        let mut tail = None;
        let mut total_tokens = 0usize;
        lowerer.with_prefill_chunks(input, maximum_chunk_tokens, tensor_context, |prepared, tokens, identity| {
            let sequence = M::sequence_len(&tokens)?;
            cache
                .bind_prepared_input(identity)
                .map_err(N::session_error)?;
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_error(EmbeddedPredictionCacheError::TargetStateActive))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_error(error));
            }
            let result = observation::neural(observers.internal(), SpeculativeActivationPhase::TargetPrefill,
                sequence, N::session_error, |observer| N::validate(|| {
                    match observer {
                        Some(observer) => session.prefill_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.prefill_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture, forward)| (EmbeddedPredictionOutput::new(logits, capture, tokens), forward))
                    .map_err(N::session_error)
                }));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => session
                    .recover_prediction_target_state_after_failure(&mut lane)
                    .map_err(|recovery| {
                        N::session_error(format!(
                            "prediction target state exchange failed: {error}; local ownership recovery failed: {recovery}"
                        ))
                    })
                    .and(Err(N::session_error(error))),
            };
            cache.restore_target(lane);
            let (mut output, forward) = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity(selected, N::TargetState::position)
                .map_err(N::session_error)?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(N::session_error)?;
            cache
                .retain_capture_generation(N::TargetState::position)
                .map_err(N::session_error)?;
            output.capture = observers.target_capture(output.capture())?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(N::session_error)?;
            validate_prediction_output::<M>(&output, Some(sequence))?;
            if sequence == 0 {
                return Err(M::empty_prediction_input());
            }
            // Sequential prediction pairs hidden[t] with original token[t + 1].
            // Keep only the previous chunk's final row to bridge that shifted pair.
            if let Some(previous) = tail.take() {
                let first = M::token_range(output.tokens(), 0, 1, context)?;
                Self::prefill_prediction_rows(session, extension, &previous, &previous,
                    &first, cache.prediction_mut(), 1, tensor_context, observers.internal())?;
            }
            let prediction_sequence = extension.prefill_sequence_len(sequence);
            if prediction_sequence != 0 {
                let hidden = M::tensor_prefix(output.capture(), sequence - 1, context)?;
                let next = M::token_range(output.tokens(), 1, sequence, context)?;
                Self::prefill_prediction_rows(session, extension, output.capture(), &hidden,
                    &next, cache.prediction_mut(), prediction_sequence, tensor_context,
                    observers.internal())?;
            }
            if maximum_chunk_tokens != usize::MAX {
                tail = Some(M::tensor_row(output.capture(), sequence - 1, context)?);
            }
            total_tokens = total_tokens.checked_add(sequence)
                .ok_or_else(|| N::session_error("prediction prefill token count overflow"))?;
            last = Some(output);
            Ok(forward)
        })?;
        Ok((last.ok_or_else(M::empty_prediction_input)?, total_tokens))
    }

    fn verify_target<'a>(
        &mut self,
        tokens: &B::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<B::Tensor>, M::Error> {
        let sequence = M::sequence_len(tokens)?;
        let tensor_context = N::target_context(context);
        let retained = tokens.clone();
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            ..
        } = self;
        lowerer.with_decode(tokens, tensor_context, |prepared| {
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_error(EmbeddedPredictionCacheError::TargetStateActive))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_error(error));
            }
            let result = observation::neural(observer, phase, sequence, N::session_error,
                |observer| N::validate(|| {
                    match observer {
                        Some(observer) => session.decode_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.decode_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture)| EmbeddedPredictionOutput::new(logits, capture, retained))
                    .map_err(N::session_error)
                }));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => session
                    .recover_prediction_target_state_after_failure(&mut lane)
                    .map_err(|recovery| {
                        N::session_error(format!(
                            "prediction target state exchange failed: {error}; local ownership recovery failed: {recovery}"
                        ))
                    })
                    .and(Err(N::session_error(error))),
            };
            cache.restore_target(lane);
            let output = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity(selected, N::TargetState::position)
                .map_err(N::session_error)?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(N::session_error)?;
            cache
                .retain_capture_generation(N::TargetState::position)
                .map_err(N::session_error)?;
            Ok(output)
        })
    }

    fn prediction_cache(&self, cache: &Self::TargetCache) -> Self::PredictionCache {
        cache.prediction_fork()
    }

    fn commit_prediction_cache(
        &self,
        cache: &mut Self::TargetCache,
        prediction: &Self::PredictionCache,
    ) -> Result<(), M::Error> {
        cache.commit_prediction(prediction);
        Ok(())
    }

    fn restore_target_checkpoint<'a>(
        cache: &mut Self::TargetCache,
        checkpoint: &Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<(), M::Error> {
        cache
            .restore(checkpoint, |state, previous| {
                N::TargetState::restore(state, previous, N::target_context(context))
                    .map_err(N::session_failure)
            })
            .map_err(N::session_error)
    }

    fn sequential_logits<'a>(
        &mut self,
        capture: &B::Tensor,
        last_token: u32,
        depth: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(M::Logits, B::Tensor), M::Error> {
        let tensor_context = N::target_context(context);
        let token = N::token(last_token, tensor_context)?;
        observation::neural(
            observer,
            SpeculativeActivationPhase::Proposal { depth },
            1,
            N::session_error,
            |observer| {
                Self::prediction_phase(
                    self.session,
                    self.extension,
                    cache.prediction_mut(),
                    crate::ExpertPass::Decode,
                    tensor_context,
                    observer,
                    |extension, invoker, lane, observer| {
                        let observed = observer.is_some();
                        let (logits, hidden) = extension.logits_observed::<S, _>(
                            invoker, capture, &token, depth, lane, observer,
                        )?;
                        let row = M::logits_row(&logits, 0, context)?;
                        Ok((row, hidden, observed.then_some(logits)))
                    },
                    |extension, lane, (_, hidden, logits)| {
                        let logits = logits.as_ref().expect("observed output retains its logits");
                        extension.complete_state(lane, &[logits, hidden], tensor_context)
                    },
                )
                .map(|(row, hidden, _)| (row, hidden))
            },
        )
    }

    fn fused_logits<'a>(
        &mut self,
        capture: &B::Tensor,
        last_token: u32,
        capacity: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<Option<B::Tensor>, M::Error> {
        let lane = cache
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(capture))
            .map_err(N::session_error)?;
        if self.selected.requirements().strategy().class()
            == crate::SpeculativeStrategyClass::EmbeddedSequential
        {
            // Sequential proposals still validate the retained lane and capture
            // above; no fused invocation exists to instrument for this strategy.
            return Ok(None);
        }
        let tensor_context = N::target_context(context);
        let token = N::token(last_token, tensor_context)?;
        if observer.is_some() {
            // Retain all temporary proposal cache members through the same
            // completion/commit protocol as sequential prediction phases. The
            // persistent accepted-context lane is never modified by proposals.
            let mut proposal = cache.prediction().clone();
            return observation::neural(
                observer,
                SpeculativeActivationPhase::FusedProposal,
                capacity,
                N::session_error,
                |observer| {
                    Self::prediction_phase(
                        self.session,
                        self.extension,
                        &mut proposal,
                        crate::ExpertPass::Decode,
                        tensor_context,
                        observer,
                        |extension, invoker, lane, observer| {
                            extension.fused_logits_observed::<S, _>(
                                invoker, &token, capacity, lane, observer,
                            )
                        },
                        |extension, lane, logits| match logits {
                            Some(logits) => {
                                extension.complete_state(lane, &[logits], tensor_context)
                            }
                            None => extension.complete_state(lane, &[], tensor_context),
                        },
                    )
                },
            );
        }
        N::validate(|| {
            self.extension.fused_logits::<S, _>(
                &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                    session: self.session,
                    context: tensor_context,
                    _native: PhantomData,
                },
                &token,
                capacity,
                cache.prediction(),
            )
        })
    }

    fn advance_prediction_cache<'a>(
        &mut self,
        captures: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(), M::Error> {
        let lane = cache
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(captures))
            .map_err(N::session_error)?;
        let tensor_context = N::target_context(context);
        observation::neural(
            observer,
            SpeculativeActivationPhase::PredictionReplay,
            M::sequence_len(tokens)?,
            N::session_error,
            |observer| {
                Self::prediction_phase(
                    self.session,
                    self.extension,
                    cache.prediction_mut(),
                    crate::ExpertPass::Decode,
                    tensor_context,
                    observer,
                    |extension, invoker, lane, observer| {
                        extension
                            .advance_observed::<S, _>(invoker, captures, tokens, lane, observer)
                    },
                    |extension, lane, _| extension.complete_state(lane, &[], tensor_context),
                )
            },
        )
    }
}

impl<A, B, S, SM, D, P, I, N, M>
    crate::speculative::EmbeddedPredictionCacheFactory<N::ExecutorTypes, M>
    for ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: crate::RuntimeState<B> + 'static,
    A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    D: crate::ReplicatedTextExecutionStrategy<A, B, S, SM::ResidentPolicy, SM::BoundedPolicy>,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: super::MaterializedPredictionExecutor<A, B, N>,
    N: super::PredictionExtensionMaterializer<B> + ReplicatedPredictionNative<B, S, M>,
    I: ReplicatedPredictionInput<A, B, S, M::Error>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    fn new_cache<'a>(&mut self, _context: M::Context<'a>) -> Result<Self::TargetCache, M::Error>
    where
        Self: 'a,
    {
        ReplicatedMaterializedPredictionStrategy::new_cache(self)
    }

    fn bind_context<'a>(
        context: <N::ExecutorTypes as GenerationExecutorTypes>::Context<'a>,
    ) -> M::Context<'a>
    where
        Self: 'a,
    {
        context
    }
}
