//! Whole-session MLX speculative generation capability.

mod activation_source;
pub(crate) use activation_source::observer as original_activation_observer;
mod original;
pub(crate) mod prefill_input;
pub(super) use prefill_input::OriginalEmbeddedPrefillInput;

use eredu_architectures::speculative_execution::{
    DynEmbeddedExecutor, EmbeddedExecutorTypes, EmbeddedPredictionOutput,
    ReplicatedPredictionInput, ReplicatedPredictionNative, SpeculativeTensorMechanisms,
};
use eredu_core::{
    generation::{GenerationCancellationToken, SemanticEvent, SpeculativeConfig},
    ModelRuntime, PreparedSpeculativeLane, SemanticState, SpeculativeCallbackPublisher,
    SpeculativeCapability, SpeculativeDraft, SpeculativeExecutor, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeOutputRuntime, SpeculativeSampling,
    SpeculativeSemanticConstraint, SpeculativeTokenFilterController,
};
use eredu_runtime::{ConstrainedSampler, SpeculativeSampler};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use super::{
    session::MlxTextSampler,
    speculative::{
        scheduler::{component_timing_enabled, MlxSpeculativeRuntime},
        MlxAssistantPreparationVisitor, MlxDrafter, MlxSpeculativeSampling, MlxSpeculativeSeed,
        SpeculativeExecutionStreams,
    },
    MlxBackend, MlxModelInput,
};
use crate::backend::error::Error;
use crate::backend::managed_memory::NativeMemoryOwner;
use crate::backend::nn::{shared::MlxNeuralBackend, tensor::TokenValidationScope};
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::backend::runtime::media::input;
use crate::MlxTensor;

mod embedded_error;
pub(super) mod embedded_logits;
mod embedded_tensors;
use super::speculative::IndependentLogits;

pub(crate) struct MlxEmbeddedPredictionMechanisms;

pub(crate) struct MlxEmbeddedExecutorTypes;

pub(crate) trait MlxEmbeddedExecutorContinuation {
    fn construction_controls(&self, _bytes: Option<usize>) -> Result<(), Error> {
        Ok(())
    }
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynEmbeddedExecutor<'_, MlxEmbeddedExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>;
}

impl EmbeddedExecutorTypes for MlxEmbeddedExecutorTypes {
    type Input = MlxModelInput;
    type Logits = IndependentLogits;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::TypedSpeculativeCompletion;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
    type Error = Error;
    fn take_retained_failure(error: Error) -> Result<eredu_core::BackendFailure, Error> {
        error.take_retained_backend_failure()
    }

    fn request_context<'a>(
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error>
    where
        Self: 'a,
    {
        context.request_context(request)
    }
    fn driver_buffer<V>(
        capacity: usize,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<V>, Error> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::driver_buffer(capacity, context)
    }
    fn driver_buffer_bytes<V>(capacity: usize) -> Option<usize> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::driver_buffer_bytes::<V>(capacity)
    }
    fn driver_host_metadata(
        bytes: Option<usize>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::HostPreparationAuthority, Error> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::driver_host_metadata(bytes, context)
    }
    fn driver_identity(
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeRequestIdentity, Error> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::driver_identity(context)
    }
    fn copy_sequence(
        source: eredu_core::SpeculativeSequenceRef<'_>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Error>> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::copy_sequence(source, context)
    }
    fn sequence_copy_bytes(source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        type Shared = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::AutoregressiveMechanisms;
        Shared::sequence_copy_bytes(source)
    }
    fn coordinate_retained_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        eredu_core::BackendFailure,
    > {
        context.coordinate_speculative_step(local)
    }
    fn prepare_control_continuation(
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
        context: Self::Context<'_>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        let Some((sources, environment)) = context.original_numerical() else {
            return Ok(());
        };
        let result = (|| {
            sources.validate_environment(environment)?;
            if context.embedded_invocation().is_some() {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            let selected = context.original_embedded().ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
            selected.prepare_continuation(committed, status)
        })();
        result.map_err(|cause| {
            eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                sources.retain_error(cause),
                Error::take_retained_backend_failure,
            )
        })
    }

    fn erased_type_mismatch(value: &'static str) -> Self::Error {
        eredu_architectures::speculative_execution::EmbeddedPredictionContractError::ErasedValue {
            value,
        }
        .into()
    }
}

pub(crate) struct MlxTextPredictionInput;

impl<A, S> ReplicatedPredictionInput<A, MlxNeuralBackend, S, Error> for MlxTextPredictionInput
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    type Input = MlxModelInput;

    type Prefill = eredu_architectures::speculative_execution::TextPredictionPrefill<MlxTensor>;

    fn with_prefill_source<R>(
        &mut self,
        input: Self::Input,
        _context: &Stream,
        operation: impl FnOnce(Result<Self::Prefill, Error>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let prepared = (|| {
                // Validate all parts before selecting any payload. Retain the
                // original segments; never concatenate the complete prompt.
                input::validate(input)?;
                let tokens = input
                    .parts
                    .iter()
                    .map(|part| match (part.modality(), part.payload()) {
                        (eredu_core::InputModality::Text, input::InputPayload::TokenIds(value)) => {
                            Ok(MlxTensor::from_array(value.clone()))
                        }
                        _ => Err(Error::Exception(Exception::custom(
                            "embedded text prefill requires token-ID input",
                        ))),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let identity = input.shared_cache_identity().cloned().or_else(|| {
                    input
                        .cache_identity()
                        .cloned()
                        .map(eredu_runtime::SharedPreparedInputCacheIdentity::new)
                });
                Ok(
                    eredu_architectures::speculative_execution::TextPredictionPrefill::new(
                        tokens,
                        identity,
                        input.prefill_chunk_positions(),
                    ),
                )
            })();
            operation(prepared)
        })
    }

    fn with_prefill_source_with_metadata<R>(
        &mut self,
        input: Self::Input,
        prepared: eredu_runtime::input::PreparedModelInputOwner<MlxTensor>,
        _context: &Stream,
        metadata: &eredu_nn::workspace::WorkspaceContext,
        operation: impl FnOnce(Result<Self::Prefill, Error>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let source=(|| {
                let funding=metadata.metadata_funding().ok_or_else(||Error::Neural(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()))?;
                // The full-context wrapper authenticated these exact original B
                // parts. This aliases the existing cache owner and no raw ID data.
                eredu_architectures::speculative_execution::TextPredictionPrefill::from_prepared_owner_with_metadata(
                    prepared, input.shared_cache_identity().cloned(), input.prefill_chunk_positions(), &funding,
                ).map_err(Error::Neural)
            })();
            operation(source)
        })
    }

    fn requested_chunks(input: &Self::Input) -> Option<std::num::NonZeroU64> {
        input.with_borrowed(|input| input.prefill_chunk_positions())
    }

    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        context: &Stream,
        operation: impl for<'a> FnOnce(
            A::Input<'a>,
            MlxTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let tokens = input::text_token_ids(input, context).map(MlxTensor::from_array)?;
            let prepared_tokens = tokens.clone();
            operation(
                A::text_input(&prepared_tokens, None),
                tokens,
                input.cache_identity(),
            )
        })
    }

    fn with_decode<R>(
        &mut self,
        tokens: &MlxTensor,
        _context: &Stream,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        operation(A::text_input(tokens, None))
    }
}

impl<A, S> ReplicatedPredictionNative<A, MlxNeuralBackend, S, MlxEmbeddedPredictionMechanisms>
    for super::replicated_text::MlxEmbeddedPredictionMaterializer
where
    S: super::replicated_text::MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    type Input = MlxModelInput;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
    type ExecutorTypes = MlxEmbeddedExecutorTypes;

    fn prefill_schedule_authority(
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority, Error> {
        context
            .original_embedded()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?
            .prefill_schedule()
    }

    fn executor_context<'a>(
        context: <Self::ExecutorTypes as EmbeddedExecutorTypes>::Context<'a>,
    ) -> <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Context<'a> {
        context
    }

    fn target_context<'a>(
        context: <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> &'a Stream {
        context.target()
    }

    fn with_prefill_source<I, P, R>(
        lowerer: &mut I,
        input: MlxModelInput,
        context: SpeculativeExecutionStreams<'_>,
        operation: impl for<'source> FnOnce(
            Result<P, <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Error>,
            SpeculativeExecutionStreams<'source>,
        ) -> Result<
            R,
            <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Error,
        >,
    ) -> Result<R, <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Error>
    where
        I: ReplicatedPredictionInput<
            A,
            MlxNeuralBackend,
            S,
            <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Error,
            Input = MlxModelInput,
            Prefill = P,
        >,
    {
        prefill_input::with_source::<A, S, I, R>(lowerer, input, context, operation)
    }

    fn prepare_prefill_chunk<P>(
        source: &P,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<P::Chunk, eredu_nn::Error>
    where
        P: eredu_architectures::speculative_execution::PredictionPrefillSource<
            A,
            MlxNeuralBackend,
            S,
        >,
    {
        prefill_input::prepare_chunk::<A, S, P>(source, chunk, context)
    }

    fn prediction_snapshot_context<'a>(
        context: <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> <Self as eredu_architectures::prediction_extension::PredictionExtensionMaterializer<
        MlxNeuralBackend,
    >>::SnapshotContext<'a>
    where
        Self: eredu_architectures::prediction_extension::PredictionExtensionMaterializer<
            MlxNeuralBackend,
        >,
        MlxNeuralBackend: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
    {
        context
    }

    fn uses_prepared_cache(context: SpeculativeExecutionStreams<'_>) -> bool {
        context.original_numerical().is_some()
    }
    fn prepared_cache_metadata(
        bytes: Option<usize>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_core::HostPreparationAuthority, eredu_core::BackendFailure> {
        super::replicated_text::cache_metadata(bytes, context)
    }
    fn prepared_cache_error<E: std::error::Error + Send + Sync + 'static>(
        cause: E,
        context: SpeculativeExecutionStreams<'_>,
    ) -> eredu_core::BackendFailure {
        super::replicated_text::cache_error(cause, context)
    }
    fn prepared_cache<P>(
        source: &S,
        extension: &P,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionCache<S, P::LaneState>,
        eredu_core::BackendFailure,
    >
    where
        S: 'static,
        P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            Self,
        >,
    {
        super::replicated_text::prepare_cache::<A, P, S>(source, extension, selected, context)
    }

    fn checkpoint(state: &S) -> Result<S, Error> {
        state.deep_checkpoint().map_err(Error::from)
    }

    fn control_state_estimate(
        state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = state.original_isolated_snapshot_estimate()?;
        // A new isolated copy adds one independent neutral authority in addition
        // to any request/storage owners already retained by its source state.
        let authority =
            std::mem::size_of::<eredu_runtime::working_memory::WorkingMemoryUnquotedLease>() as u64;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(authority)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(authority)?;
        Some(estimate)
    }

    fn control_state_snapshot<'a>(
        state: &S,
        context: SpeculativeExecutionStreams<'a>,
    ) -> Result<Option<S>, eredu_core::speculative::SpeculativeControlError> {
        copy_control_state(state, context)
    }

    fn restore(state: &mut S, checkpoint: &S, context: &Stream) -> Result<(), Error> {
        state
            .restore_checkpoint(checkpoint, context)
            .map_err(Error::from)
    }

    fn generation(state: &S) -> Result<u64, Error> {
        u64::try_from(state.offset())
            .map_err(|_| Error::InvalidOperation("target capture generation is negative"))
    }

    fn token(token: u32, context: SpeculativeExecutionStreams<'_>) -> Result<MlxTensor, Error> {
        if let Some(active) = context.embedded_invocation() {
            return active.token(token, context.target());
        }
        if let Some((sources, _)) = context.original_numerical() {
            return Err(sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let token = i32::try_from(token)
            .map_err(|_| Error::InvalidOperation("prediction token exceeds int32 storage"))?;
        Ok(MlxTensor::from_array(Array::from_slice(&[token], &[1, 1])))
    }

    fn shape(tensor: &MlxTensor) -> &[i32] {
        tensor.as_array().shape()
    }

    fn complete_prediction_state<P>(
        extension: &P,
        state: &mut P::LaneState,
        outputs: &[&MlxTensor],
        point: eredu_architectures::speculative_execution::PredictionCompletionPoint,
        sources: eredu_architectures::speculative_execution::PredictionCompletionSources<'_>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(), Error>
    where
        P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            Self,
        >,
    {
        if context.original_numerical().is_some() {
            return super::replicated_text::complete_original_prediction_state::<A, P>(
                extension, state, outputs, point, sources, context,
            );
        }
        extension
            .complete_state(state, outputs, context.target())
            .map_err(Error::StorageSource)
    }

    fn validate_with_context<T>(
        context: SpeculativeExecutionStreams<'_>,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        match context.embedded_invocation() {
            Some(active) => {
                active.validate_scope(context.target())?;
                operation()
            }
            None => <Self as ReplicatedPredictionNative<
                A,
                MlxNeuralBackend,
                S,
                MlxEmbeddedPredictionMechanisms,
            >>::validate(operation),
        }
    }

    fn validate_captured_span<T>(
        context: SpeculativeExecutionStreams<'_>,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        if let Some(source) = context.original_embedded() {
            let (_, environment) = context.original_numerical().ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
            source
                .numerical_sources()
                .validate_environment(environment)?;
            // Each actual target/seed invocation installs its exact prepared
            // collector. No ordinary collector may enclose those native scopes.
            operation()
        } else {
            <Self as ReplicatedPredictionNative<
                A,
                MlxNeuralBackend,
                S,
                MlxEmbeddedPredictionMechanisms,
            >>::validate(operation)
        }
    }

    fn validate<T>(operation: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        let scope = TokenValidationScope::begin()?;
        let output = operation()?;
        let validations = scope.finish();
        if !validations.is_empty() {
            async_eval_with_event(validations.arrays())?.synchronize()?;
            validations.validate_completed()?;
        }
        Ok(output)
    }

    fn session_error(error: impl std::fmt::Display) -> Error {
        Error::Exception(Exception::custom(error.to_string()))
    }

    fn session_failure(error: eredu_core::BackendFailure) -> Error {
        Error::StorageSource(error)
    }
    fn session_cause_with_context<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Error {
        embedded_error::session_cause(error, context)
    }
    fn prepare_session_cause<'context: 'context, E: std::error::Error + Send + Sync + 'static>(
        context: SpeculativeExecutionStreams<'context>,
        additional_controls: Option<usize>,
    ) -> Result<impl FnOnce(E) -> Error, Error> {
        embedded_error::prepare_session_cause(context, additional_controls)
    }
    fn session_arguments_with_context(
        arguments: std::fmt::Arguments<'_>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Error {
        embedded_error::session_arguments(arguments, context)
    }
    fn neural_cause_with_context<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        context: SpeculativeExecutionStreams<'_>,
    ) -> eredu_nn::Error {
        embedded_error::neural_cause(error, context)
    }
    fn neural_observer_error(
        error: &Error,
        context: SpeculativeExecutionStreams<'_>,
    ) -> eredu_nn::Error {
        embedded_error::neural_observer(error, context)
    }

    fn take_telemetry() -> Result<Self::Telemetry, Error> {
        Ok(Self::Telemetry::default())
    }
}

/// Copies state under independent domain authority, including empty host state.
pub(super) fn copy_control_state<S: super::replicated_text::MlxStateMechanisms>(
    state: &S,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<Option<S>, eredu_core::speculative::SpeculativeControlError> {
    if !state.supports_isolated_snapshot() {
        return Ok(None);
    }
    let memory = NativeMemoryOwner::acquire(&context.memory_ledger())
        .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
    let copy = crate::backend::submission_recovery::detached_retained(memory.clone(), || {
        super::speculative::state_snapshot::settle(state.retained_arrays())?;
        let mut copy = state.isolated_snapshot(context.target())?;
        copy.inference_retention_mut()
            .retain_unquoted(&memory.unquoted_lease()?);
        super::speculative::state_snapshot::settle(copy.retained_arrays())?;
        for array in copy.retained_arrays() {
            memory.retain_array(array)?;
        }
        Ok(copy)
    })
    .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
    Ok(Some(copy))
}

impl SpeculativeTensorMechanisms for MlxEmbeddedPredictionMechanisms {
    type Tensor = MlxTensor;
    type Logits = IndependentLogits;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::TypedSpeculativeCompletion;
    type Error = Error;
    fn take_retained_failure(error: Error) -> Result<eredu_core::BackendFailure, Error> {
        error.take_retained_backend_failure()
    }

    fn observation_error(message: &'static str) -> Self::Error {
        Error::InvalidOperation(message)
    }

    fn control_tensor_estimate(
        value: &MlxTensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::speculative::state_snapshot::estimate_array(value.as_array())
    }

    fn control_tensor_snapshot<'a>(
        value: &MlxTensor,
        context: Self::Context<'a>,
    ) -> Result<Option<MlxTensor>, eredu_core::speculative::SpeculativeControlError> {
        let memory = NativeMemoryOwner::acquire(&context.memory_ledger())
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        crate::backend::submission_recovery::detached_retained(memory.clone(), || {
            let array =
                super::speculative::state_snapshot::copy_array(value.as_array(), context.target())?;
            memory.retain_array(&array)?;
            Ok(Some(MlxTensor::from_array(array)))
        })
        .map_err(eredu_core::speculative::SpeculativeControlError::backend)
    }

    fn coordinate_speculative_step<'a>(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        context.coordinate_speculative_step(local)
    }

    fn agree_text_preparation<'a>(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        context.agree_text_preparation(stage, status)
    }

    fn activation_observer<'a>(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<
        Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Error>>>,
        eredu_core::speculative::SpeculativeControlError,
    > {
        if context.original_numerical().is_some() {
            return activation_source::observer(plan, request, context);
        }
        let transport = super::speculative::capture_error_transport::RetainedCaptureTransport::new(
            |error: &eredu_runtime::capture::CaptureExecutionError<Error>| {
                Error::Exception(Exception::custom(error.to_string()))
            },
        );
        match context.capture_binding() {
            Some(binding) => {
                binding.observer_with_error(plan, request, context.target(), transport)
            }
            None => super::session::bounded_capture::speculative_capture_with_error(
                plan,
                request,
                context.target(),
                transport,
            ),
        }
    }

    fn empty_prediction_input() -> Self::Error {
        Error::InvalidOperation("embedded prediction input must contain at least one token")
    }

    fn fused_prediction_exhausted() -> Self::Error {
        Error::InvalidOperation("fused embedded prediction proposal block is exhausted")
    }

    fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error {
        eredu_architectures::speculative_execution::EmbeddedPredictionContractError::Commit {
            verified,
            available,
        }
        .into()
    }

    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Self::Error {
        eredu_architectures::speculative_execution::EmbeddedPredictionContractError::Output {
            logits,
            capture,
            tokens,
            expected,
        }
        .into()
    }

    fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error {
        eredu_architectures::speculative_execution::EmbeddedPredictionContractError::FusedCapacity {
            requested,
            available,
        }
        .into()
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        usize::try_from(value.as_array().dim(1))
            .map_err(|_| Error::InvalidOperation("prediction sequence length exceeds usize"))
    }

    fn validate_outer_tensor_observer<'a>(
        has_caller: bool,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        embedded_logits::validate_observer(has_caller, context)
    }

    fn observe_outer_tensor<'a>(
        value: Self::Tensor,
        path: &str,
        chunk: Option<&eredu_runtime::prefill::PrefillChunk>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Self::Tensor, Self::Error>>,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        embedded_logits::observe_owned(value, path, chunk, observer, context)
    }

    fn selected_prefill_logits(value: Self::Tensor) -> Result<Self::Logits, Self::Error> {
        Ok(IndependentLogits::Ordinary(value.into_array()))
    }

    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        if let Some(active) = context.embedded_invocation() {
            return active.logits_row(value.as_array(), row, context.target());
        }
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        embedded_logits::row(value.as_array(), row, context.target())
            .map(IndependentLogits::Ordinary)
    }

    fn prefill_score_layout<'a>(
        context: Self::Context<'a>,
    ) -> eredu_runtime::replicated_session::PrefillScoreLayout {
        use eredu_runtime::replicated_session::PrefillScoreLayout;
        if context.original_numerical().is_some() {
            PrefillScoreLayout::SelectedPositions
        } else {
            PrefillScoreLayout::FinalScores
        }
    }

    fn selected_prefill_logits_with_source<'a>(
        value: Self::Tensor,
        source: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        embedded_logits::selected(value, source, context)
    }
    fn logits_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        source: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        embedded_logits::source_row(value, row, source, context, false)
    }
    fn fused_logits_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        source: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        embedded_logits::source_row(value, row, source, context, true)
    }
    fn retain_logit_block<'a>(
        value: Self::Tensor,
        source: Option<eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionLogitBlock<Self::Tensor>,
        Self::Error,
    > {
        embedded_logits::retain_block(value, source, context)
    }

    fn tensor_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        source: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::tensor_row(value, row, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::capture_range(
            value,
            row,
            row.checked_add(1)
                .ok_or(Error::InvalidOperation("prediction row overflow"))?,
            source,
            None,
            context,
        )
    }
    fn tensor_prefix_with_source<'a>(
        value: &Self::Tensor,
        end: usize,
        source: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::tensor_prefix(value, end, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::capture_range(value, 0, end, source, None, context)
    }
    fn tensor_row_packet<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::tensor_row(value, row, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::capture_range(
            value,
            row,
            row.checked_add(1)
                .ok_or(Error::InvalidOperation("prediction row overflow"))?,
            value.evidence(),
            Some(value),
            context,
        )
    }
    fn tensor_prefix_packet<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::tensor_prefix(value, end, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::capture_range(value, 0, end, value.evidence(), Some(value), context)
    }
    fn tensor_concatenate_packet<'a>(
        left: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        right: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        context: Self::Context<'a>,
        ordinary: impl FnOnce(&Self::Tensor, &Self::Tensor) -> Result<Self::Tensor, Self::Error>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return ordinary(left, right).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::concatenate(left, right, context, std::mem::size_of_val(&ordinary))
    }
    fn prefill_token_packet<'a>(
        value: &Self::Tensor,
        prepared: Option<
            &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        >,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Ok(prepared.cloned().unwrap_or_else(|| {
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary(
                    value.clone(),
                )
            }));
        }
        let prepared = prepared.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        embedded_tensors::share(prepared, context)
    }
    fn control_tensor_packet_estimate(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let Some(evidence) = value.evidence() else {
            return Self::control_tensor_estimate(value);
        };
        if super::speculative::completed_tensor_source(evidence).is_none()
            && super::speculative::registered_tensor_source(evidence).is_none()
        {
            return None;
        }
        let bytes = std::mem::size_of_val(value) as u64;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }
    fn control_tensor_packet_snapshot<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<
        Option<eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>>,
        eredu_core::speculative::SpeculativeControlError,
    > {
        if context.original_numerical().is_none() {
            return Self::control_tensor_snapshot(value, context).map(|value| {
                value.map(
                    eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
                )
            });
        }
        embedded_tensors::share(value, context)
            .map(Some)
            .map_err(|cause| {
                eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                    cause,
                    Self::take_retained_failure,
                )
            })
    }

    fn tensor_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let row = i32::try_from(row)
            .map_err(|_| Error::InvalidOperation("prediction row exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., row..row + 1, ..), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn tensor_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let end = i32::try_from(end)
            .map_err(|_| Error::InvalidOperation("prediction prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end, ..), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn token_range<'a>(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let start = i32::try_from(start)
            .map_err(|_| Error::InvalidOperation("prediction token start exceeds i32"))?;
        let end = i32::try_from(end)
            .map_err(|_| Error::InvalidOperation("prediction token end exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., start..end), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let end = i32::try_from(end)
            .map_err(|_| Error::InvalidOperation("prediction token prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn target_tokens_packet<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::target_tokens(tokens, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::tokens(tokens, context)
    }
    fn token_range_packet<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        start: usize,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return Self::token_range(value, start, end, context).map(
                eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary,
            );
        }
        embedded_tensors::token_range(value, start, end, context)
    }
    fn with_tensor_sources<R>(
        sources: &[&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
        context: Self::Context<'_>,
        run: impl for<'scope> FnOnce(Self::Context<'scope>) -> Result<R, Self::Error>,
    ) -> Result<R, Self::Error> {
        if context.original_numerical().is_none() {
            return run(context);
        }
        embedded_tensors::with_sources(sources, context, run)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let width = i32::try_from(tokens.len())
            .map_err(|_| Error::InvalidOperation("prediction token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if context.crosses_devices() {
            value = value.copy(context.target())?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn fused_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        embedded_logits::row(value.as_array(), row, context.draft())
            .map(IndependentLogits::Ordinary)
    }

    fn submit_verification_completion_with_source<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        let Some((sources, environment)) = context.original_numerical() else {
            return Self::submit_verification_completion(output, inputs, context);
        };
        if context.embedded_invocation().is_some() || environment.stream() != context.target() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        super::speculative::TypedSpeculativeCompletion::completed_embedded(
            output.logits().as_array(),
            output.capture().as_array(),
            evidence.ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?,
            sources,
            environment,
        )
    }

    fn submit_verification_completion<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        super::speculative::MlxSpeculativeCompletion::submit([
            output.logits().as_array(),
            output.capture().as_array(),
            output.tokens().as_array(),
            inputs.as_array(),
        ])
        .map(super::speculative::TypedSpeculativeCompletion::new)
        .map_err(Error::from)
    }
}

impl<'world> SpeculativeGenerationBackend for MlxBackend<'world> {
    type Drafter = MlxDrafter;

    fn speculative_capability(runtime: &ModelRuntime<Self>) -> SpeculativeCapability {
        runtime.session().speculative_capability()
    }

    fn supports_speculative_prefill_chunking(
        runtime: &ModelRuntime<Self>,
        drafting: &SpeculativeDraft<'_, Self::Drafter>,
    ) -> bool {
        supports_prefill_chunking(runtime.session(), drafting)
    }

    fn speculative_activation_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        runtime.session().speculative_activation_discovery()
    }

    fn validate_speculative_activations(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        plan.validate(&Self::speculative_activation_discovery(runtime)?)?;
        eredu_runtime::intervention::preflight(
            plan.captures(),
            plan.interventions(),
            &super::session::intervention::NativeInterventionEstimator,
        )
    }

    fn validate_speculative_capture(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        super::speculative::validate_control_capture(plan)?;
        <Self as eredu_core::TextGenerationBackend>::validate_text_capture(runtime, plan)
    }

    fn speculative_intervention_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        let mut discovery =
            <Self as eredu_core::TextGenerationBackend>::intervention_discovery(runtime)?;
        discovery
            .points
            .retain(|point| point.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
        Ok(discovery)
    }
    fn validate_speculative_interventions(
        runtime: &ModelRuntime<Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        Self::validate_speculative_capture(runtime, capture)?;
        let discovery = Self::speculative_intervention_discovery(runtime)?;
        eredu_runtime::intervention::validate_session(
            capture,
            plan,
            &discovery,
            &super::session::intervention::NativeInterventionEstimator,
        )
    }

    fn with_speculative_execution<C, V>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        MlxSpeculativeSession::new(runtime).with_execution(request, visitor)
    }
}

struct MlxSpeculativeSession<'runtime, 'world> {
    runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>,
}

struct MlxSpeculativeLaneRuntime<'a, C> {
    input: MlxModelInput,
    config: eredu_core::SpeculativeConfiguration,
    prng_key: Option<MlxSpeculativeSeed>,
    sampler: MlxPreparedSampler<C>,
    semantic: eredu_core::SemanticStateOwner,
    cancellation: GenerationCancellationToken,
    on_event: eredu_core::SpeculativeEventCallback<'a>,
    memory_owner: Option<NativeMemoryOwner>,
}

type MlxPreparedSampler<C> = ConstrainedSampler<MlxTextSampler, C>;

fn validate_lane_proposal_capacity(
    config: &SpeculativeConfig,
    proposal_capacity: usize,
) -> Result<(), Error> {
    if config.max_draft_tokens > proposal_capacity {
        return Err(Error::Speculative(format!(
            "lane requests {} draft tokens, but the selected speculative realization admits at most {proposal_capacity}",
            config.max_draft_tokens
        )));
    }
    Ok(())
}

fn supports_prefill_chunking(
    session: &super::session::MlxModelSession,
    drafting: &SpeculativeDraft<'_, MlxDrafter>,
) -> bool {
    match drafting {
        SpeculativeDraft::Embedded => true,
        SpeculativeDraft::External(drafter) => {
            drafter.is_autoregressive() || session.supports_external_capture_spans()
        }
        _ => false,
    }
}

fn validate_speculative_inference_policy(
    generation: eredu_core::TextGenerationConfig,
) -> Result<(), Error> {
    let policy = generation.inference_policy();
    policy
        .validate(generation.sampling().max_new_tokens)
        .map_err(|error| Error::Other(Box::new(error)))?;
    Ok(())
}

fn validate_speculative_prefill_policy(
    generation: eredu_core::TextGenerationConfig,
    selected_support: bool,
) -> Result<(), Error> {
    validate_speculative_inference_policy(generation.clone())?;
    if generation
        .inference_policy()
        .prefill_chunk_positions
        .is_some()
        && !selected_support
    {
        return Err(Error::Other(Box::new(
            eredu_core::CapabilityError::InvalidConfiguration {
                field: "speculative_inference_policy",
                detail: "selected target does not provide shared capture-span publication".into(),
            },
        )));
    }
    Ok(())
}

struct MlxExternalBatchRunner<'target, 'lane, C, V> {
    target: &'target mut (dyn super::replicated_text::ErasedReplicatedTextExecutable + 'target),
    lanes: eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>,
    streams: SpeculativeExecutionStreams<'target>,
    visitor: V,
    capture: eredu_architectures::composite_execution::ExternalPredictionCaptureRequest,
    selected: eredu_runtime::SelectedSpeculativeRealization,
}

struct MlxExternalBatchVisitor<'target, 'lane, C, V> {
    runner: MlxExternalBatchRunner<'target, 'lane, C, V>,
}

impl<'target, 'lane, C, V>
    eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>
    for MlxExternalBatchVisitor<'target, 'lane, C, V>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    type Output = Result<SpeculativeGenerationBatchOutput, Error>;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        assistant: &mut super::speculative::MlxExternalAssistant<A>,
    ) -> Self::Output {
        let runner = self.runner;
        let target = runner.target.external_prediction_mut().ok_or_else(|| {
            Error::ArchitectureModel(
                "selected target has no external-assistant prediction capability".into(),
            )
        })?;
        let mut caches = (0..runner.lanes.len())
            .map(|_| {
                target
                    .prepare_external_prediction_target_cache()
                    .map(|cache| {
                        super::speculative::external::MlxExternalPredictionCache::new(
                            cache,
                            runner.selected.clone(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        A::visit_executor::<
            crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
            _,
        >(
            target,
            assistant,
            runner.capture,
            MlxExternalExecutorRunner {
                lanes: runner.lanes,
                caches: &mut caches,
                streams: runner.streams,
                visitor: runner.visitor,
            },
        )
    }
}

struct MlxExternalExecutorRunner<'cache, 'lane, 'streams, C, V> {
    lanes: eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>,
    caches: &'cache mut [super::speculative::external::MlxExternalPredictionCache],
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: V,
}

impl<'cache, 'lane, 'streams, A, C, V>
    eredu_architectures::ExternalAssistantExecutorVisitor<
        A,
        crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
    > for MlxExternalExecutorRunner<'cache, 'lane, 'streams, C, V>
where
    A: eredu_architectures::ExternalAssistantArchitecture,
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    type Output = Result<SpeculativeGenerationBatchOutput, Error>;

    fn execute<'run, E>(self, backend: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: SpeculativeExecutor<
                Input = MlxModelInput,
                Cache = super::speculative::external::MlxExternalPredictionCache,
                Logits = IndependentLogits,
                Context<'run> = SpeculativeExecutionStreams<'run>,
                Completion = super::speculative::TypedSpeculativeCompletion,
                Telemetry = crate::composition::mlx::speculative::scheduler::SpeculativeComponentTimings,
                Error = Error,
            > + 'run,
    {
        run_speculative_batch(
            backend,
            self.lanes,
            self.caches,
            Ok,
            self.streams,
            self.visitor,
        )
    }
}

fn run_speculative_batch<'run, 'lane, 'streams, B, C, S>(
    backend: &'run mut B,
    lanes: eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'lane, C>>,
    caches: &'run mut [B::Cache],
    wrap_sampler: impl Fn(MlxPreparedSampler<C>) -> Result<S, Exception>,
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: impl SpeculativeGenerationVisitor,
) -> Result<SpeculativeGenerationBatchOutput, Error>
where
    'lane: 'run,
    'streams: 'run,
    B: MlxSpeculativeRuntime<'run>,
    B::Logits: super::speculative::LogitsSource,
    B::Error:
        From<Exception> + From<<B::Logits as super::speculative::LogitsSource>::OriginalError>,
    C: SpeculativeTokenFilterController + 'run,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'run,
{
    let topology = streams.topology();
    let component_timings_collected = component_timing_enabled() && backend.supports_telemetry();
    let prepared = (|| {
        if caches.len() != lanes.len() {
            return Err(match streams.original_numerical() {
                Some((sources, _)) => sources.retain_startup_error(BatchCardinality {
                    caches: caches.len(),
                    lanes: lanes.len(),
                }),
                None => Error::Exception(Exception::custom(format!(
                    "speculative cache has {} lanes but the request has {} lanes",
                    caches.len(),
                    lanes.len(),
                ))),
            });
        }
        let mut prepared = backend
            .driver_buffer(lanes.len(), streams)
            .map_err(|cause| match B::take_retained_failure(cause) {
                Ok(cause) => Error::StorageSource(cause),
                Err(cause) => match streams.original_numerical() {
                    Some((sources, _)) => sources.retain_startup_error(cause),
                    None => Error::Exception(Exception::from_source(cause)),
                },
            })?;
        for (index, (lane, cache)) in lanes.into_iter().zip(caches.iter_mut()).enumerate() {
            let streams = streams.request_context(eredu_core::SpeculativeRequestId::new(index))?;
            let MlxSpeculativeLaneRuntime {
                input,
                config,
                prng_key,
                sampler,
                semantic,
                cancellation,
                on_event,
                memory_owner,
            } = lane;
            let sampling = MlxSpeculativeSampling::prepare(
                wrap_sampler(sampler)?,
                streams,
                memory_owner.as_ref(),
            )?
            .with_error::<B::Error>()
            .with_logits::<B::Logits>();
            let randomness =
                <MlxSpeculativeSampling<S, B::Error, B::Logits> as SpeculativeSampling>::initialize_randomness(
                    prng_key,
                    config.temperature,
                    streams,
                ).map_err(|cause| match B::take_retained_failure(cause) {
                    Ok(cause) => Error::StorageSource(cause),
                    Err(cause) => match streams.original_numerical() {
                        Some((sources, _)) => sources.retain_startup_error(cause),
                        None => Error::Exception(Exception::from_source(cause)),
                    },
                })?;
            let sequence = super::speculative::autoregressive::sequence::new(
                config.max_tokens,
                &config.eos_token_ids,
                streams,
            )?;
            prepared
                .try_push(PreparedSpeculativeLane::new(
                    cache,
                    input,
                    config,
                    SpeculativeOutputRuntime::new(
                        sampling,
                        sequence,
                        SpeculativeSemanticConstraint::semantic(semantic),
                        prepare_speculative_callback(on_event, streams)?,
                        cancellation,
                    ),
                    randomness,
                ))
                .map_err(|cause| match streams.original_numerical() {
                    Some((sources, _)) => sources.retain_startup_error(cause),
                    None => Error::Exception(Exception::from_source(cause)),
                })?;
        }
        Ok(prepared)
    })();
    let prepared = streams
        .finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Delivery,
            prepared,
            |cause| match streams.original_numerical() {
                Some(_) => Error::StorageSource(cause),
                None => Error::Exception(Exception::from_source(cause)),
            },
        )
        .map_err(|cause| batch_preparation_error(cause, streams))?;
    visitor
        .run(
            backend,
            prepared,
            topology,
            streams.is_split(),
            component_timings_collected,
            streams,
        )
        .map_err(|cause| match streams.original_numerical() {
            Some((sources, _)) => match cause {
                eredu_core::SpeculativeDriverError::Backend(cause) => {
                    match B::take_retained_failure(cause) {
                        Ok(cause) => Error::StorageSource(cause),
                        Err(cause) => sources.retain_startup_error(
                            eredu_core::SpeculativeDriverError::Backend(cause),
                        ),
                    }
                }
                eredu_core::SpeculativeDriverError::Preparation(cause) => {
                    Error::StorageSource(cause)
                }
                cause => sources.retain_startup_error(cause),
            },
            None => Error::Exception(Exception::from_source(Exception::from_source(cause))),
        })
}

// Original failures retain their existing paid cause. Ordinary conversion keeps
// an outer Exception wrapper; sequence storage does not alter its policy.
fn batch_preparation_error(cause: Error, streams: SpeculativeExecutionStreams<'_>) -> Error {
    match streams.original_numerical() {
        Some((sources, _)) => sources.retain_error(cause),
        None => match cause {
            Error::Exception(cause) => Error::Exception(Exception::from_source(cause)),
            cause => Error::Exception(Exception::from_source(cause)),
        },
    }
}

#[derive(Debug, thiserror::Error)]
#[error("speculative cache has {caches} lanes but the request has {lanes} lanes")]
struct BatchCardinality {
    caches: usize,
    lanes: usize,
}

impl<'runtime, 'world> MlxSpeculativeSession<'runtime, 'world> {
    fn new(runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>) -> Self {
        Self { runtime }
    }

    fn prepare_mlx_speculative_sampling<C>(
        generation: eredu_core::TextGenerationConfig,
        constraint: C,
        memory: &NativeMemoryOwner,
    ) -> Result<(Option<MlxSpeculativeSeed>, MlxPreparedSampler<C>), Error>
    where
        C: SpeculativeTokenFilterController,
    {
        validate_speculative_inference_policy(generation.clone())?;
        let resolved = generation.sampling();
        let sampler = MlxTextSampler::from_config(generation.clone())
            .map_err(|error| Error::Exception(Exception::from_source(error)))?;
        let prng_key = (resolved.temperature != 0.0)
            .then(|| {
                let key = safemlx::random::key(generation.seed())?;
                memory.retain_array(&key)?;
                Ok::<_, Error>(
                    MlxSpeculativeSampling::<MlxPreparedSampler<C>>::seed_from_array(key)
                        .with_memory_owner(memory),
                )
            })
            .transpose()?;
        Ok((prng_key, ConstrainedSampler::new(sampler, constraint)))
    }

    fn prepare_speculative_batch_lanes<'a, C>(
        lanes: eredu_core::SpeculativeBuffer<SpeculativeGenerationLane<'a, MlxBackend<'world>, C>>,
        proposal_capacity: usize,
        memory: &NativeMemoryOwner,
    ) -> Result<eredu_core::SpeculativeBuffer<MlxSpeculativeLaneRuntime<'a, C>>, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        for lane in &lanes {
            validate_speculative_inference_policy(lane.generation().clone())?;
            validate_lane_proposal_capacity(lane.config(), proposal_capacity)?;
        }
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for mut lane in lanes {
            let mut prompt = lane.take_prompt();
            let generation = lane.take_generation();
            if let Some(chunk) = generation.inference_policy().prefill_chunk_positions {
                prompt = prompt.with_prefill_chunk_positions(chunk);
            }
            let config = lane.take_config();
            let constraint = lane.take_constraint();
            let semantic = lane.take_semantic();
            let cancellation = lane.take_cancellation();
            let on_event = lane.take_on_event();
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, constraint, memory)?;
            prepared_lanes.push(MlxSpeculativeLaneRuntime {
                input: prompt,
                config,
                prng_key,
                sampler,
                semantic,
                cancellation,
                on_event,
                memory_owner: Some(memory.clone()),
            });
        }
        Ok(prepared_lanes.into())
    }

    fn with_execution<C, V>(
        &mut self,
        mut request: SpeculativeGenerationBatchRequest<'_, MlxBackend<'world>, MlxDrafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let drafting = request.take_drafting();
        let lanes = request.take_lanes();
        original::run_request(self.runtime, drafting, lanes, visitor)
    }

    fn generate_speculative_batch_with_external_draft<C, V>(
        &mut self,
        drafter: &mut MlxDrafter,
        lanes: eredu_core::SpeculativeBuffer<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
        memory: &NativeMemoryOwner,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        self.runtime.session().ensure_no_submission_in_flight()?;
        let proposal_capacity = drafter
            .selected()
            .requirements()
            .strategy()
            .proposal_capacity()
            .get();
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        if drafter.is_autoregressive() {
            let topology = drafter.topology();
            let coordination = self.runtime.session().speculative_partition_binding();
            return self.runtime.session_mut().with_model_operation(|model| {
                let streams =
                    SpeculativeExecutionStreams::bind(&target_stream, &draft_stream, topology)?
                        .with_capture_binding(coordination.as_ref())
                        .with_memory_owner(memory);
                let prepared_lanes =
                    Self::prepare_speculative_batch_lanes(lanes, proposal_capacity, memory)?;
                drafter.with_autoregressive(|draft| {
                    let mut executor =
                        eredu_runtime::speculative::autoregressive::AutoregressiveExecutor::<
                            super::speculative::autoregressive::MlxAutoregressiveMechanisms,
                        >::new(
                            model,
                            draft.executable_mut(),
                            std::num::NonZeroUsize::new(proposal_capacity)
                                .expect("selected nonzero capacity"),
                        );
                    let mut caches = (0..prepared_lanes.len())
                        .map(|_| executor.new_cache(streams))
                        .collect::<Result<Vec<_>, _>>()?;
                    run_speculative_batch(
                        &mut executor,
                        prepared_lanes,
                        &mut caches,
                        Ok,
                        streams,
                        visitor,
                    )
                })
            });
        }
        let capture = drafter.capture().clone();
        let selected = drafter.selected().clone();
        self.runtime.session_mut().with_model_operation(|model| {
            let streams = SpeculativeExecutionStreams::bind(
                &target_stream,
                &draft_stream,
                drafter.topology(),
            )?
            .with_memory_owner(memory);
            let prepared_lanes =
                Self::prepare_speculative_batch_lanes(lanes, proposal_capacity, memory)?;
            drafter.visit(MlxExternalBatchVisitor {
                runner: MlxExternalBatchRunner {
                    target: model.erased_mut(),
                    lanes: prepared_lanes,
                    streams,
                    visitor,
                    capture,
                    selected,
                },
            })
        })
    }

    fn generate_speculative_batch_with_embedded_draft<C, V>(
        &mut self,
        lanes: eredu_core::SpeculativeBuffer<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
        memory: &NativeMemoryOwner,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        self.runtime.session().ensure_no_submission_in_flight()?;
        let stream = self.runtime.backend().stream().clone();
        let capture = self.runtime.session().speculative_partition_binding();
        let streams = SpeculativeExecutionStreams::single(&stream)
            .with_capture_binding(capture.as_ref())
            .with_memory_owner(memory);
        let mut continuation = MlxEmbeddedBatchContinuation {
            lanes,
            streams,
            visitor: Some(visitor),
        };
        self.runtime.session_mut().with_model_operation(|model| {
            model
                .erased_mut()
                .with_embedded_prediction(&mut continuation)
                .ok_or_else(|| {
                    Error::Speculative(
                        "neutral target has no installed prediction-extension contract".into(),
                    )
                })?
        })
    }
}

struct MlxEmbeddedBatchContinuation<'lane, 'world, 'streams, C, V>
where
    C: SpeculativeTokenFilterController,
{
    lanes: eredu_core::SpeculativeBuffer<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>,
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: Option<V>,
}

impl<C, V> MlxEmbeddedExecutorContinuation for MlxEmbeddedBatchContinuation<'_, '_, '_, C, V>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynEmbeddedExecutor<'_, MlxEmbeddedExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error> {
        let proposal_capacity = selected.requirements().strategy().proposal_capacity().get();
        let lanes = std::mem::take(&mut self.lanes);
        // Cache construction may enter architecture-owned preparation
        // collectives. Every rank must first accept its sampler/lane policy.
        let prepared_lanes = self.streams.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Sampling,
            MlxSpeculativeSession::prepare_speculative_batch_lanes(
                lanes,
                proposal_capacity,
                self.streams
                    .memory_owner()
                    .expect("prepared execution retains its domain owner"),
            ),
            |error| Error::Exception(Exception::from_source(error)),
        )?;
        let caches = (0..prepared_lanes.len())
            .map(|_| executor.new_cache())
            .collect::<Result<Vec<_>, _>>();
        let mut caches = self.streams.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Sampling,
            caches,
            Error::StorageSource,
        )?;
        run_speculative_batch(
            executor,
            prepared_lanes,
            &mut caches,
            Ok,
            self.streams,
            self.visitor
                .take()
                .expect("embedded executor continuation is invoked once"),
        )
    }
}

#[cfg(test)]
mod mechanism_tests;
#[cfg(test)]
mod memory_tests;

fn prepare_speculative_callback<'a>(
    callback: eredu_core::SpeculativeEventCallback<'a>,
    streams: SpeculativeExecutionStreams<'_>,
) -> Result<SpeculativeCallbackPublisher<'a>, Error> {
    let Some((sources, _)) = streams.original_numerical() else {
        return Ok(SpeculativeCallbackPublisher::semantic_callback(callback));
    };
    use eredu_core::HostPreparationAuthority;
    use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<eredu_core::SpeculativeEventCallback<'a>>(),
        size_of::<Result<SpeculativeCallbackPublisher<'a>, Error>>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<HostMetadataFunding>(),
        size_of::<Option<usize>>(),
        SpeculativeCallbackPublisher::prepared_control_bytes()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    let host = HostPreparationAuthority::retain(sources.metadata_funding().clone());
    Ok(SpeculativeCallbackPublisher::semantic_prepared_callback(
        callback, host,
    ))
}
