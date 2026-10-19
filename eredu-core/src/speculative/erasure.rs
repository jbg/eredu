//! Final erasure of a fully constructed generation executor; family construction
//! and native parameter binding remain with their declaring owners.
use super::*;

/// Backend-facing value types fixed at final generation-executor binding.
/// Family construction remains typed. This bundle fixes input, score, context,
/// completion and telemetry values while cache, source state and verification
/// payloads retain their typed ownership behind the final executor boundary.
pub trait GenerationExecutorTypes: 'static {
    /// Backend-owned model input.
    type Input;
    /// Opaque logits consumed by backend sampling.
    type Logits: Clone;
    /// Selected execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact native completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Structured backend failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Constructs the stable failure for an internally inconsistent erased value.
    fn erased_type_mismatch(value: &'static str) -> Self::Error;
}

trait CloneAny: std::any::Any {
    fn clone_any(&self) -> Box<dyn CloneAny>;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

impl<T: Clone + 'static> CloneAny for T {
    fn clone_any(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// Opaque erased target cache for a completed embedded executor.
pub struct DynGenerationCache(Box<dyn std::any::Any>);
/// Opaque erased proposal seed state.
pub struct DynGenerationTargetState(Box<dyn std::any::Any>);
/// Opaque erased, discardable prediction branch.
pub struct DynGenerationDraftState(Box<dyn CloneAny>);
/// Opaque erased target-cache checkpoint.
pub struct DynGenerationCheckpoint(Box<dyn std::any::Any>);
/// Opaque erased verification payload.
pub struct DynGenerationVerification(Box<dyn std::any::Any>);

impl Clone for DynGenerationDraftState {
    fn clone(&self) -> Self {
        Self(self.0.as_ref().clone_any())
    }
}

/// An executor which can realize a fresh lane cache before scheduling.
pub trait GenerationExecutorCacheFactory<T: GenerationExecutorTypes>: SpeculativeExecutor {
    /// Realizes one independent authoritative target/prediction lane.
    fn new_cache<'a>(&mut self, context: Self::Context<'a>) -> Result<Self::Cache, Self::Error>
    where
        Self: 'a;

    /// Binds the backend's fixed scheduler context to this exact typed executor.
    fn bind_context<'a>(context: T::Context<'a>) -> Self::Context<'a>
    where
        Self: 'a;
}

/// Object-safe final binding of a fully constructed generation executor.
/// The blanket implementation preserves typed source policies, mode admission,
/// state ownership, exact completions and transactional verification. It does
/// not select a family, reopen artifacts or construct model parameters.
pub trait ErasedGenerationExecutor<T: GenerationExecutorTypes> {
    /// Verification evidence selected by the retained source.
    fn verification_policy(&self) -> SpeculativeVerificationPolicy;
    /// Updates only request-local state from retained source declarations.
    fn configure_decoding_mode(
        &mut self,
        state: &mut DynGenerationTargetState,
        mode: crate::decoding::DecodingMode,
        temperature: f32,
    ) -> Result<(), GenerationError>;
    /// Ordinary complete-canvas refinement, retaining position score semantics.
    fn refine_block<'a>(
        &mut self,
        state: &mut DynGenerationTargetState,
        logits: &T::Logits,
        committed: &[u32],
        policy: Option<crate::refinement::RefinementPolicy>,
        temperature: f32,
        context: T::Context<'a>,
    ) -> Result<Option<RefinementProgress<T::Logits>>, T::Error>
    where
        Self: 'a;
    /// Exact mode facts from the already paired typed executor.
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities;
    /// Preserves the typed score-row storage bound through executor erasure.
    fn logits_snapshot_estimate(
        &self,
        logits: &T::Logits,
    ) -> Option<crate::execution_control::SnapshotEstimate>;
    /// Preserves mode agreement across the completed executor's type erasure.
    fn coordinate_decoding_control<'a>(
        &mut self,
        local: crate::decoding::DecodingCoordination,
        context: T::Context<'a>,
    ) -> Result<crate::decoding::DecodingCoordination, crate::BackendFailure>
    where
        Self: 'a;
    /// Preserves preparation agreement through executable erasure.
    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<crate::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<crate::SpeculativeScheduleState>, crate::BackendFailure>
    where
        Self: 'a;
    /// Preserves preparation agreement through executable erasure.
    fn agree_text_preparation<'a>(
        &mut self,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure>
    where
        Self: 'a;
    /// Installs validated prospective internal edits through the typed collector.
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError>;
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        crate::residency::ExecutionConversionRetentionTrimReport,
        crate::residency::ParameterConversionTrimError,
    >;

    /// Reads shared retention budgets without touching a lane or native work.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Option<crate::residency::ExecutionConversionRetentionReport>, crate::BackendFailure>;

    /// Reads one settled lane without polling, copying, or evaluating native work.
    fn continuation_memory_observation(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
        additional: u64,
    ) -> Result<Option<crate::speculative::SpeculativeContinuationObservation>, crate::BackendFailure>;
    /// Complete durable snapshot bound for the erased canonical lane.
    fn control_snapshot_estimate(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
    ) -> Option<crate::execution_control::SnapshotEstimate>;
    /// Copies complete typed state without erasing a native failure's source.
    fn control_snapshot<'a>(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<(DynGenerationCheckpoint, DynGenerationTargetState)>, SpeculativeControlError>
    where
        Self: 'a;
    /// Restores all typed state atomically after copying succeeds.
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut DynGenerationCache,
        saved: &DynGenerationCheckpoint,
        state: &DynGenerationTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<DynGenerationTargetState>, SpeculativeControlError>
    where
        Self: 'a;
    /// Installs one request's admitted collector for the borrowed executor scope.
    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: crate::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a;
    /// Whether the installed internal observer requires scheduler provenance.
    fn requires_activation_origin(&self) -> bool;
    /// Forwards the shared scheduler's bounded operation scope.
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>);
    /// Moves one already charged internal record through erasure.
    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture>;
    /// Drains the original portable capture failure.
    fn take_activation_error(&mut self) -> Option<crate::speculative::SpeculativeControlError>;
    /// Maximum proposals supported by the retained realization.
    fn max_proposals(&self) -> usize;
    /// Enables optional component telemetry.
    fn set_telemetry_enabled(&mut self, enabled: bool);
    /// Whether component telemetry is available.
    fn supports_telemetry(&self) -> bool;
    /// Drains component telemetry.
    fn take_telemetry(&mut self) -> Result<T::Telemetry, T::Error>;
    /// Drains telemetry retained in one verification payload.
    fn take_verification_telemetry(
        &mut self,
        output: &mut DynGenerationVerification,
    ) -> Result<T::Telemetry, T::Error>;
    /// Whether an exact cloned branch may be promoted.
    fn supports_exact_optimistic_promotion(&self) -> bool;
    /// Realizes one lane cache.
    fn new_cache<'a>(&mut self, context: T::Context<'a>) -> Result<DynGenerationCache, T::Error>
    where
        Self: 'a;
    /// Prefills one lane.
    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynGenerationCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynGenerationTargetState, T::Logits>, T::Error>
    where
        Self: 'a;
    /// Forks a proposal branch.
    fn begin_proposal<'a>(
        &mut self,
        state: &DynGenerationTargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: T::Context<'a>,
    ) -> Result<DynGenerationDraftState, T::Error>
    where
        Self: 'a;
    /// Advances a proposal branch and produces logits.
    fn proposal_step<'a>(
        &mut self,
        state: &mut DynGenerationDraftState,
        last_token: u32,
        context: T::Context<'a>,
    ) -> Result<crate::speculative::SpeculativeProposalStep<T::Logits>, T::Error>
    where
        Self: 'a;
    /// Captures an exact target-cache checkpoint.
    fn checkpoint(&self, cache: &DynGenerationCache) -> Result<DynGenerationCheckpoint, T::Error>;
    /// Restores an exact checkpoint.
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut DynGenerationCache,
        checkpoint: &DynGenerationCheckpoint,
        context: T::Context<'a>,
    ) -> Result<(), T::Error>
    where
        Self: 'a;
    /// Submits target verification.
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut DynGenerationCache,
        context: T::Context<'a>,
    ) -> Result<Submission<DynGenerationVerification, T::Completion>, T::Error>
    where
        Self: 'a;
    /// Selects one retained verification-logits row.
    fn verification_logits<'a>(
        &self,
        output: &DynGenerationVerification,
        index: usize,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error>
    where
        Self: 'a;
    /// Commits an exact verified prefix.
    #[allow(clippy::too_many_arguments)]
    fn commit_verification<'a>(
        &mut self,
        output: DynGenerationVerification,
        draft_state: DynGenerationDraftState,
        cache: &mut DynGenerationCache,
        checkpoint: &DynGenerationCheckpoint,
        verified_inputs: usize,
        context: T::Context<'a>,
    ) -> Result<SpeculativeCommit<DynGenerationTargetState>, T::Error>
    where
        Self: 'a;
}

fn erased_ref<'a, T: 'static, E: GenerationExecutorTypes>(
    value: &'a Box<dyn std::any::Any>,
    name: &'static str,
) -> Result<&'a T, E::Error> {
    value
        .downcast_ref::<T>()
        .ok_or_else(|| E::erased_type_mismatch(name))
}

fn erased_mut<'a, T: 'static, E: GenerationExecutorTypes>(
    value: &'a mut Box<dyn std::any::Any>,
    name: &'static str,
) -> Result<&'a mut T, E::Error> {
    value
        .downcast_mut::<T>()
        .ok_or_else(|| E::erased_type_mismatch(name))
}

impl<T, E> ErasedGenerationExecutor<T> for E
where
    T: GenerationExecutorTypes,
    E: GenerationExecutorCacheFactory<
        T,
        Input = T::Input,
        Logits = T::Logits,
        Completion = T::Completion,
        Telemetry = T::Telemetry,
        Error = T::Error,
    >,
    E::Cache: 'static,
    E::TargetState: 'static,
    E::DraftState: 'static,
    E::CacheCheckpoint: 'static,
    E::Verification: 'static,
{
    fn verification_policy(&self) -> SpeculativeVerificationPolicy {
        SpeculativeExecutor::verification_policy(self)
    }
    fn configure_decoding_mode(
        &mut self,
        state: &mut DynGenerationTargetState,
        mode: crate::decoding::DecodingMode,
        temperature: f32,
    ) -> Result<(), GenerationError> {
        let state =
            state
                .0
                .downcast_mut::<E::TargetState>()
                .ok_or(GenerationError::DecodingMode(
                    crate::decoding::DecodingModeError::Unsupported(
                        "generation state belongs to another executor",
                    ),
                ))?;
        SpeculativeExecutor::configure_decoding_mode(self, state, mode, temperature)
    }
    fn refine_block<'a>(
        &mut self,
        state: &mut DynGenerationTargetState,
        logits: &T::Logits,
        committed: &[u32],
        policy: Option<crate::refinement::RefinementPolicy>,
        temperature: f32,
        context: T::Context<'a>,
    ) -> Result<Option<RefinementProgress<T::Logits>>, T::Error>
    where
        Self: 'a,
    {
        let state = erased_mut::<E::TargetState, T>(&mut state.0, "generation refinement state")?;
        SpeculativeExecutor::refine_block(
            self,
            state,
            logits,
            committed,
            policy,
            temperature,
            E::bind_context(context),
        )
    }
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
        SpeculativeExecutor::decoding_capabilities(self)
    }
    fn logits_snapshot_estimate(
        &self,
        logits: &T::Logits,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        SpeculativeExecutor::logits_snapshot_estimate(self, logits)
    }
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        SpeculativeExecutor::readmit_activation_interventions(self, plan)
    }
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        crate::residency::ExecutionConversionRetentionTrimReport,
        crate::residency::ParameterConversionTrimError,
    > {
        SpeculativeExecutor::trim_parameter_conversions(self)
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Option<crate::residency::ExecutionConversionRetentionReport>, crate::BackendFailure>
    {
        SpeculativeExecutor::parameter_conversion_retention(self)
    }

    fn continuation_memory_observation(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
        additional: u64,
    ) -> Result<Option<crate::speculative::SpeculativeContinuationObservation>, crate::BackendFailure>
    {
        SpeculativeExecutor::continuation_memory_observation(
            self,
            cache.0.downcast_ref().expect("paired embedded cache"),
            state
                .0
                .downcast_ref()
                .expect("paired embedded target state"),
            additional,
        )
    }

    fn control_snapshot_estimate(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        SpeculativeExecutor::control_snapshot_estimate(
            self,
            erased_ref::<E::Cache, T>(&cache.0, "snapshot cache").ok()?,
            erased_ref::<E::TargetState, T>(&state.0, "snapshot seed").ok()?,
        )
    }
    fn control_snapshot<'a>(
        &self,
        cache: &DynGenerationCache,
        state: &DynGenerationTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<(DynGenerationCheckpoint, DynGenerationTargetState)>, SpeculativeControlError>
    where
        Self: 'a,
    {
        let cache = erased_ref::<E::Cache, T>(&cache.0, "snapshot cache")
            .map_err(SpeculativeControlError::backend)?;
        let state = erased_ref::<E::TargetState, T>(&state.0, "snapshot seed")
            .map_err(SpeculativeControlError::backend)?;
        SpeculativeExecutor::control_snapshot(
            self,
            cache,
            state,
            <E as GenerationExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|pair| {
            pair.map(|(cache, state)| {
                (
                    DynGenerationCheckpoint(Box::new(cache)),
                    DynGenerationTargetState(Box::new(state)),
                )
            })
        })
    }
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut DynGenerationCache,
        saved: &DynGenerationCheckpoint,
        state: &DynGenerationTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<DynGenerationTargetState>, SpeculativeControlError>
    where
        Self: 'a,
    {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "snapshot cache")
            .map_err(SpeculativeControlError::backend)?;
        let saved = erased_ref::<E::CacheCheckpoint, T>(&saved.0, "snapshot checkpoint")
            .map_err(SpeculativeControlError::backend)?;
        let state = erased_ref::<E::TargetState, T>(&state.0, "snapshot seed")
            .map_err(SpeculativeControlError::backend)?;
        SpeculativeExecutor::restore_control_snapshot(
            self,
            cache,
            saved,
            state,
            <E as GenerationExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|state| state.map(|state| DynGenerationTargetState(Box::new(state))))
    }
    fn coordinate_decoding_control<'a>(
        &mut self,
        local: crate::decoding::DecodingCoordination,
        context: T::Context<'a>,
    ) -> Result<crate::decoding::DecodingCoordination, crate::BackendFailure>
    where
        Self: 'a,
    {
        SpeculativeExecutor::coordinate_decoding_control(
            self,
            local,
            <E as GenerationExecutorCacheFactory<T>>::bind_context(context),
        )
    }

    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<crate::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<crate::SpeculativeScheduleState>, crate::BackendFailure>
    where
        Self: 'a,
    {
        SpeculativeExecutor::coordinate_speculative_step(
            self,
            local,
            <E as GenerationExecutorCacheFactory<T>>::bind_context(context),
        )
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure>
    where
        Self: 'a,
    {
        SpeculativeExecutor::agree_text_preparation(
            self,
            stage,
            status,
            <E as GenerationExecutorCacheFactory<T>>::bind_context(context),
        )
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: crate::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::configure_activation_capture(self, plan, request, context)
    }

    fn requires_activation_origin(&self) -> bool {
        SpeculativeExecutor::requires_activation_origin(self)
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        SpeculativeExecutor::set_activation_origin(self, origin);
    }

    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        SpeculativeExecutor::take_activation_capture(self)
    }

    fn take_activation_error(&mut self) -> Option<crate::speculative::SpeculativeControlError> {
        SpeculativeExecutor::take_activation_error(self)
    }

    fn max_proposals(&self) -> usize {
        SpeculativeExecutor::max_proposals(self)
    }

    fn set_telemetry_enabled(&mut self, enabled: bool) {
        SpeculativeExecutor::set_telemetry_enabled(self, enabled);
    }

    fn supports_telemetry(&self) -> bool {
        SpeculativeExecutor::supports_telemetry(self)
    }

    fn take_telemetry(&mut self) -> Result<T::Telemetry, T::Error> {
        SpeculativeExecutor::take_telemetry(self)
    }

    fn take_verification_telemetry(
        &mut self,
        output: &mut DynGenerationVerification,
    ) -> Result<T::Telemetry, T::Error> {
        let output = erased_mut::<E::Verification, T>(&mut output.0, "verification telemetry")?;
        SpeculativeExecutor::take_verification_telemetry(self, output)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeExecutor::supports_exact_optimistic_promotion(self)
    }

    fn new_cache<'a>(&mut self, context: T::Context<'a>) -> Result<DynGenerationCache, T::Error>
    where
        Self: 'a,
    {
        GenerationExecutorCacheFactory::<T>::new_cache(self, E::bind_context(context))
            .map(|cache| DynGenerationCache(Box::new(cache)))
    }

    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynGenerationCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynGenerationTargetState, T::Logits>, T::Error>
    where
        Self: 'a,
    {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::prefill(self, input, cache, context).map(|prefill| {
            let (logits, state, target_tokens) = prefill.into_parts();
            SpeculativePrefill::new(
                logits,
                DynGenerationTargetState(Box::new(state)),
                target_tokens,
            )
        })
    }

    fn begin_proposal<'a>(
        &mut self,
        state: &DynGenerationTargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: T::Context<'a>,
    ) -> Result<DynGenerationDraftState, T::Error>
    where
        Self: 'a,
    {
        let state = erased_ref::<E::TargetState, T>(&state.0, "target state")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::begin_proposal(self, state, last_token, proposal_capacity, context)
            .map(|state| DynGenerationDraftState(Box::new(state)))
    }

    fn proposal_step<'a>(
        &mut self,
        state: &mut DynGenerationDraftState,
        last_token: u32,
        context: T::Context<'a>,
    ) -> Result<crate::speculative::SpeculativeProposalStep<T::Logits>, T::Error>
    where
        Self: 'a,
    {
        let state = state
            .0
            .as_any_mut()
            .downcast_mut::<E::DraftState>()
            .ok_or_else(|| T::erased_type_mismatch("draft state"))?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::proposal_step(self, state, last_token, context)
    }

    fn checkpoint(&self, cache: &DynGenerationCache) -> Result<DynGenerationCheckpoint, T::Error> {
        let cache = erased_ref::<E::Cache, T>(&cache.0, "target cache")?;
        SpeculativeExecutor::checkpoint(self, cache)
            .map(|checkpoint| DynGenerationCheckpoint(Box::new(checkpoint)))
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut DynGenerationCache,
        checkpoint: &DynGenerationCheckpoint,
        context: T::Context<'a>,
    ) -> Result<(), T::Error>
    where
        Self: 'a,
    {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let checkpoint = erased_ref::<E::CacheCheckpoint, T>(&checkpoint.0, "target checkpoint")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::restore_checkpoint(self, cache, checkpoint, context)
    }

    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut DynGenerationCache,
        context: T::Context<'a>,
    ) -> Result<Submission<DynGenerationVerification, T::Completion>, T::Error>
    where
        Self: 'a,
    {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::submit_verification(self, input_tokens, cache, context).map(
            |submission| Submission {
                output: DynGenerationVerification(Box::new(submission.output)),
                completion: submission.completion,
            },
        )
    }

    fn verification_logits<'a>(
        &self,
        output: &DynGenerationVerification,
        index: usize,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error>
    where
        Self: 'a,
    {
        let output = erased_ref::<E::Verification, T>(&output.0, "verification")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::verification_logits(self, output, index, context)
    }

    fn commit_verification<'a>(
        &mut self,
        output: DynGenerationVerification,
        draft_state: DynGenerationDraftState,
        cache: &mut DynGenerationCache,
        checkpoint: &DynGenerationCheckpoint,
        verified_inputs: usize,
        context: T::Context<'a>,
    ) -> Result<SpeculativeCommit<DynGenerationTargetState>, T::Error>
    where
        Self: 'a,
    {
        let output = output
            .0
            .downcast::<E::Verification>()
            .map_err(|_| T::erased_type_mismatch("verification"))?;
        let draft_state = draft_state
            .0
            .into_any()
            .downcast::<E::DraftState>()
            .map_err(|_| T::erased_type_mismatch("draft state"))?;
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let checkpoint = erased_ref::<E::CacheCheckpoint, T>(&checkpoint.0, "target checkpoint")?;
        let context = <E as GenerationExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::commit_verification(
            self,
            *output,
            *draft_state,
            cache,
            checkpoint,
            verified_inputs,
            context,
        )
        .map(|commit| {
            let (state, replayed) = commit.into_parts();
            SpeculativeCommit::new(DynGenerationTargetState(Box::new(state)), replayed)
        })
    }
}

/// Concrete `SpeculativeExecutor` view over an architecture-erased retained executor.
pub struct DynGenerationExecutor<'a, T: GenerationExecutorTypes> {
    inner: &'a mut dyn ErasedGenerationExecutor<T>,
}

impl<'a, T: GenerationExecutorTypes> DynGenerationExecutor<'a, T> {
    /// Lends one already retained executor to backend scheduling.
    pub fn new(inner: &'a mut dyn ErasedGenerationExecutor<T>) -> Self {
        Self { inner }
    }

    /// Realizes one independent lane cache from the retained executor.
    pub fn new_cache<'b>(&mut self, context: T::Context<'b>) -> Result<DynGenerationCache, T::Error>
    where
        Self: 'b,
    {
        self.inner.new_cache(context)
    }
}

impl<T: GenerationExecutorTypes> SpeculativeExecutor for DynGenerationExecutor<'_, T> {
    fn verification_policy(&self) -> SpeculativeVerificationPolicy {
        self.inner.verification_policy()
    }
    fn configure_decoding_mode(
        &mut self,
        state: &mut Self::TargetState,
        mode: crate::decoding::DecodingMode,
        temperature: f32,
    ) -> Result<(), GenerationError> {
        self.inner.configure_decoding_mode(state, mode, temperature)
    }
    fn refine_block<'a>(
        &mut self,
        state: &mut Self::TargetState,
        logits: &Self::Logits,
        committed: &[u32],
        policy: Option<crate::refinement::RefinementPolicy>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<Option<RefinementProgress<Self::Logits>>, Self::Error>
    where
        Self: 'a,
    {
        self.inner
            .refine_block(state, logits, committed, policy, temperature, context)
    }
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
        self.inner.decoding_capabilities()
    }
    fn logits_snapshot_estimate(
        &self,
        logits: &Self::Logits,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        self.inner.logits_snapshot_estimate(logits)
    }
    fn coordinate_decoding_control<'a>(
        &mut self,
        local: crate::decoding::DecodingCoordination,
        context: Self::Context<'a>,
    ) -> Result<crate::decoding::DecodingCoordination, crate::BackendFailure>
    where
        Self: 'a,
    {
        self.inner.coordinate_decoding_control(local, context)
    }
    type Input = T::Input;
    type Cache = DynGenerationCache;
    type TargetState = DynGenerationTargetState;
    type DraftState = DynGenerationDraftState;
    type CacheCheckpoint = DynGenerationCheckpoint;
    type Verification = DynGenerationVerification;
    type Logits = T::Logits;
    type Context<'a>
        = T::Context<'a>
    where
        Self: 'a;
    type Completion = T::Completion;
    type Telemetry = T::Telemetry;
    type Error = T::Error;

    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.inner.readmit_activation_interventions(plan)
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        crate::residency::ExecutionConversionRetentionTrimReport,
        crate::residency::ParameterConversionTrimError,
    > {
        self.inner.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Option<crate::residency::ExecutionConversionRetentionReport>, crate::BackendFailure>
    {
        self.inner.parameter_conversion_retention()
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        additional: u64,
    ) -> Result<Option<crate::speculative::SpeculativeContinuationObservation>, crate::BackendFailure>
    {
        self.inner
            .continuation_memory_observation(cache, state, additional)
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        self.inner.control_snapshot_estimate(cache, state)
    }
    fn control_snapshot<'a>(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError>
    where
        Self: 'a,
    {
        self.inner.control_snapshot(cache, state, context)
    }
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError>
    where
        Self: 'a,
    {
        self.inner
            .restore_control_snapshot(cache, saved, state, context)
    }

    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<crate::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<crate::SpeculativeScheduleState>, crate::BackendFailure>
    where
        Self: 'a,
    {
        self.inner.coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, crate::BackendFailure>
    where
        Self: 'a,
    {
        self.inner.agree_text_preparation(stage, status, context)
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: crate::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
        self.inner
            .configure_activation_capture(plan, request, context)
    }

    fn requires_activation_origin(&self) -> bool {
        self.inner.requires_activation_origin()
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.inner.set_activation_origin(origin);
    }

    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.inner.take_activation_capture()
    }

    fn take_activation_error(&mut self) -> Option<crate::speculative::SpeculativeControlError> {
        self.inner.take_activation_error()
    }

    fn max_proposals(&self) -> usize {
        self.inner.max_proposals()
    }
    fn set_telemetry_enabled(&mut self, enabled: bool) {
        self.inner.set_telemetry_enabled(enabled);
    }
    fn supports_telemetry(&self) -> bool {
        self.inner.supports_telemetry()
    }
    fn take_telemetry(&mut self) -> Result<Self::Telemetry, Self::Error> {
        self.inner.take_telemetry()
    }
    fn take_verification_telemetry(
        &mut self,
        output: &mut Self::Verification,
    ) -> Result<Self::Telemetry, Self::Error> {
        self.inner.take_verification_telemetry(output)
    }
    fn supports_exact_optimistic_promotion(&self) -> bool {
        self.inner.supports_exact_optimistic_promotion()
    }
    fn prefill<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'a,
    {
        self.inner.prefill(input, cache, context)
    }
    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a,
    {
        self.inner
            .begin_proposal(state, last_token, proposal_capacity, context)
    }
    fn proposal_step<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<crate::speculative::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'a,
    {
        self.inner.proposal_step(state, last_token, context)
    }
    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        self.inner.checkpoint(cache)
    }
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        self.inner.restore_checkpoint(cache, checkpoint, context)
    }
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'a,
    {
        self.inner.submit_verification(input_tokens, cache, context)
    }
    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'a,
    {
        self.inner.verification_logits(output, index, context)
    }
    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'a,
    {
        self.inner.commit_verification(
            output,
            draft_state,
            cache,
            checkpoint,
            verified_inputs,
            context,
        )
    }
}
