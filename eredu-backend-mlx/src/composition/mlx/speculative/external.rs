//! Family-neutral MLX mechanisms for architecture-owned external assistants.
mod logits;
mod state;
use super::{Error, IndependentLogits, LogitsSource, TypedSpeculativeCompletion};
fn ordinary_error(message: impl Into<String>) -> Error {
    Exception::custom(message).into()
}

use eredu_architectures::{
    composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    },
    external_assistant::{
        ExternalAssistantCache, ExternalAssistantExecutionMechanisms,
        ExternalAssistantTensorPlacement, ExternalAssistantTransfer,
    },
    ExternalAssistantArchitecture,
};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use crate::{
    backend::{
        managed_memory::NativeMemoryOwner,
        nn::shared::MlxNeuralBackend,
        runtime::{cache::kv::ConcatKeyValueCache, media::input::ModelInput},
        submission_recovery,
    },
    composition::mlx::{
        replicated_text::{ErasedExternalPredictionExecutable, MlxPredictionTargetState},
        speculative::{
            scheduler::SpeculativeComponentTimings, MlxExternalAssistant, MlxSpeculativeCompletion,
            SpeculativeExecutionStreams,
        },
        MlxModelInput,
    },
    MlxTensor,
};

/// The single MLX mechanism adapter used by every architecture-owned external lifecycle.
pub(crate) struct MlxExternalAssistantMechanisms;

/// Architecture-owned semantic envelope over opaque native target storage.
pub(crate) type MlxExternalPredictionCache = ExternalAssistantCache<MlxPredictionTargetState>;

impl<A> ExternalAssistantExecutionMechanisms<A> for MlxExternalAssistantMechanisms
where
    A: ExternalAssistantArchitecture,
{
    type NeuralBackend = MlxNeuralBackend;
    type AttentionCache = ConcatKeyValueCache;
    type Target = dyn ErasedExternalPredictionExecutable;
    type Assistant = MlxExternalAssistant<A>;
    type Input = MlxModelInput;
    type NativeCache = MlxPredictionTargetState;
    type NativeCacheCheckpoint = MlxPredictionTargetState;
    type Tensor = MlxTensor;
    type Logits = IndependentLogits;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = TypedSpeculativeCompletion;
    type Telemetry = SpeculativeComponentTimings;
    type Error = Error;

    fn request_context<'a>(
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error> {
        context.request_context(request)
    }
    fn coordinate_speculative_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        eredu_core::BackendFailure,
    > {
        context.coordinate_speculative_step(local)
    }
    fn driver_identity(
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeRequestIdentity, Self::Error> {
        super::autoregressive::host_containers::identity(context)
    }
    fn copy_sequence(
        source: eredu_core::SpeculativeSequenceRef<'_>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Self::Error>>
    {
        super::autoregressive::sequence::copy(source, context)
    }
    fn sequence_copy_bytes(source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        super::autoregressive::sequence::copy_bytes(source)
    }
    fn take_retained_failure(
        error: Self::Error,
    ) -> Result<eredu_core::BackendFailure, Self::Error> {
        error.take_retained_backend_failure()
    }
    fn requires_activation_origin() -> bool {
        true
    }
    fn invocation_context<'a>(
        context: Self::Context<'a>,
        origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) -> Result<Self::Context<'a>, Self::Error> {
        context.with_external_origin(origin).map_err(Error::from)
    }
    fn prepare_control_continuation<'a>(
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        match context.original_external() {
            Some(source) => source
                .prepare_continuation(committed, status)
                .map_err(Error::from),
            None => Ok(()),
        }
    }

    fn source_context<'a, 'scope>(
        context: Self::Context<'a>,
        sources: &'scope [&'scope eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    ) -> Result<Self::Context<'scope>, Self::Error>
    where
        'a: 'scope,
    {
        if context.original_external().is_some() {
            context
                .with_external_tensor_sources(sources)
                .map_err(Error::from)
        } else {
            Ok(context)
        }
    }

    fn target_tokens_with_source<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,
        Self::Error,
    > {
        if let Some((sources, environment)) = context.original_numerical() {
            let preparation = context.original_cache_preparation().ok_or_else(|| {
                Error::from(sources.retain_startup_error(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ))
            })?;
            super::prepared_prediction_token_ids(tokens, sources, environment, preparation)
                .map_err(Error::from)
        } else {
            <Self as ExternalAssistantExecutionMechanisms<A>>::target_tokens(tokens, context)
                .map(eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary)
        }
    }
    fn transfer_with_source<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,
        Self::Error,
    > {
        use eredu_architectures::speculative_execution::EmbeddedPredictionTensor as Packet;
        if let Some((sources, _)) = context.original_numerical() {
            let funding = sources.metadata_funding();
            let controls = [
                std::mem::size_of::<Self::Context<'_>>(),
                std::mem::size_of::<ExternalAssistantTransfer>(),
                std::mem::size_of::<Packet<MlxTensor>>(),
                std::mem::size_of::<Result<Packet<MlxTensor>, Self::Error>>(),
                std::mem::size_of::<
                    eredu_architectures::external_assistant::ExternalOperationResult<MlxTensor>,
                >(),
                std::mem::size_of::<(
                    &Packet<MlxTensor>,
                    ExternalAssistantTransfer,
                    Self::Context<'_>,
                )>(),
            ];
            funding
                .reserve_metadata(
                    controls
                        .into_iter()
                        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                        .ok_or(Error::WorkspacePlanning(
                            eredu_nn::workspace::HostMetadataFundingError::Overflow,
                        ))?,
                )
                .map_err(Error::WorkspacePlanning)?;
            if context.original_external().is_none() {
                return Err(
                    <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context),
                );
            }
            match context.topology() {
                eredu_core::SpeculativeExecutionTopology::Single
                | eredu_core::SpeculativeExecutionTopology::SameDeviceSplit => {
                    let placement = match direction {
                        ExternalAssistantTransfer::TargetToDraft => {
                            eredu_core::SamplingPlacement::Draft
                        }
                        ExternalAssistantTransfer::DraftToTarget => {
                            eredu_core::SamplingPlacement::Target
                        }
                    };
                    let evidence = value.evidence().ok_or_else(|| {
                        <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context)
                    })?;
                    super::tensor_sources::input_array_environment(
                        evidence,
                        value.as_array(),
                        context,
                        placement,
                        funding,
                    )?;
                    // The immutable packet and completed source already own all
                    // native custody. Aliasing does not make another Array handle.
                    return Ok(value.clone());
                }
                eredu_core::SpeculativeExecutionTopology::CrossDeviceSplit => {
                    let transferred = state::transfer(value, value.evidence(), direction, context)?;
                    let evidence = transferred.evidence.ok_or_else(|| {
                        <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context)
                    })?;
                    let host = crate::composition::mlx::replicated_text::cache_metadata(
                        Packet::<MlxTensor>::retained_control_bytes(),
                        context,
                    )
                    .map_err(Error::StorageSource)?;
                    return Ok(Packet::from_prepared(transferred.output, evidence, host));
                }
                _ => {
                    return Err(
                        <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context),
                    )
                }
            }
        }
        <Self as ExternalAssistantExecutionMechanisms<A>>::transfer(value, direction, context)
            .map(eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary)
    }

    fn assistant_operation_with_evidence<
        I: eredu_architectures::external_assistant::invocation::ExternalAssistantOperation<A>,
    >(
        assistant: &mut Self::Assistant,
        arguments: I::Arguments<'_, Self::Tensor>,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalOperationResult<I::Output<Self::Tensor>>,
        Self::Error,
    > {
        if context.original_external().is_some() {
            return super::assistant::execute_original_assistant::<A, I>(
                assistant, arguments, context,
            )
            .map_err(Error::from);
        }
        <Self as ExternalAssistantExecutionMechanisms<A>>::assistant_operation::<I>(
            assistant, arguments, context,
        )
        .map(eredu_architectures::external_assistant::ExternalOperationResult::ordinary)
    }

    fn control_copy_tensor_with_source(
        value: &Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'_>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalOperationResult<Self::Tensor>,
        Self::Error,
    > {
        if context.original_external().is_some() {
            state::copy(value, evidence, placement, context)
        } else {
            <Self as ExternalAssistantExecutionMechanisms<A>>::control_copy_tensor(
                value, placement, context,
            )
            .map(eredu_architectures::external_assistant::ExternalOperationResult::ordinary)
        }
    }

    fn tensor_range_with_source<'a>(
        value: &Self::Tensor,
        axis: u8,
        start: usize,
        end: usize,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalOperationResult<Self::Tensor>,
        Self::Error,
    > {
        state::range(value, axis, start, end, evidence, placement, context)
    }
    fn tensor_alias_with_source<'a>(
        value: &Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalOperationResult<Self::Tensor>,
        Self::Error,
    > {
        state::alias(value, evidence, placement, context)
    }
    fn token_prefix_with_source<'a>(
        value: &eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_some() {
            let evidence = value.evidence().ok_or_else(|| {
                <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context)
            })?;
            return super::sampling::numerical::tensor_axis_range_at(
                value,
                1,
                0,
                end,
                true,
                evidence,
                context,
                eredu_core::SamplingPlacement::Target,
            );
        }
        <Self as ExternalAssistantExecutionMechanisms<A>>::token_prefix(value, end, context)
            .map(eredu_architectures::speculative_execution::EmbeddedPredictionTensor::ordinary)
    }
    fn transfer_tensor_with_source<'a>(
        value: &Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalOperationResult<Self::Tensor>,
        Self::Error,
    > {
        if context.original_numerical().is_none() {
            return <Self as ExternalAssistantExecutionMechanisms<A>>::transfer(
                value, direction, context,
            )
            .map(eredu_architectures::external_assistant::ExternalOperationResult::ordinary);
        }
        state::transfer(value, evidence, direction, context)
    }
    fn join_tensor_sources<'a>(
        visit: impl FnMut(&mut dyn FnMut(&Self::Tensor)),
        evidence: &[&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
        context: Self::Context<'a>,
    ) -> Result<
        Option<eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        Self::Error,
    > {
        state::join(visit, evidence, context)
    }

    fn join_tensor_sources_at<'a>(
        visit: impl FnMut(&mut dyn FnMut(&Self::Tensor)),
        evidence: &[&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<
        Option<eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        Self::Error,
    > {
        state::join_at(visit, evidence, placement, context)
    }

    fn config(assistant: &Self::Assistant) -> &A::Config {
        &assistant.config
    }

    fn module(assistant: &mut Self::Assistant) -> &mut A::Module<Self::NeuralBackend> {
        &mut assistant.module.inner
    }

    fn neural_error(error: eredu_nn::Error) -> Self::Error {
        Error::from(error)
    }

    fn error(message: String) -> Self::Error {
        ordinary_error(message)
    }

    fn with_prepared_input_cache_identity<R>(
        input: &Self::Input,
        run: impl FnOnce(&eredu_runtime::PreparedInputCacheIdentity) -> R,
    ) -> Result<R, Self::Error> {
        input.cache_identity().map(run).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))
    }
    fn identity_text(
        arguments: std::fmt::Arguments<'_>,
        context: Self::Context<'_>,
    ) -> Result<eredu_runtime::SpeculativeIdentity, Self::Error> {
        if let Some((sources, _)) = context.original_numerical() {
            sources
                .metadata_funding()
                .reserve_metadata(std::mem::size_of::<(
                    std::fmt::Arguments<'_>,
                    String,
                    eredu_runtime::SpeculativeIdentity,
                    Result<eredu_runtime::SpeculativeIdentity, Self::Error>,
                )>())
                .map_err(Error::WorkspacePlanning)?;
            let value = sources
                .metadata_funding()
                .metadata_string(arguments)
                .map_err(Error::Neural)?;
            eredu_runtime::SpeculativeIdentity::new(value)
                .map_err(|cause| sources.retain_startup_error(cause))
        } else {
            eredu_runtime::SpeculativeIdentity::new(arguments.to_string())
                .map_err(|cause| ordinary_error(cause.to_string()))
        }
    }
    fn checkpoint_native_with_context(
        cache: &Self::NativeCache,
        context: Self::Context<'_>,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        if context.original_external().is_some() {
            original_cache_copy(cache, context)
        } else {
            cache.deep_clone().map_err(Error::from)
        }
    }

    fn prepared_input_cache_identity(
        input: &Self::Input,
    ) -> Result<eredu_runtime::PreparedInputCacheIdentity, Self::Error> {
        input.cache_identity().cloned().ok_or_else(|| {
            ordinary_error(
                "external speculative input is missing its prepared-input cache identity",
            )
        })
    }

    fn capture_shape_matches(
        value: &Self::Tensor,
        entry: &eredu_runtime::SpeculativeCaptureEntry,
        context: Self::Context<'_>,
    ) -> Result<bool, Self::Error> {
        if let Some((sources, _)) = context.original_numerical() {
            sources
                .metadata_funding()
                .reserve_metadata(std::mem::size_of::<(
                    &Self::Tensor,
                    &eredu_runtime::SpeculativeCaptureEntry,
                    &[i32],
                    bool,
                    Result<bool, Self::Error>,
                )>())
                .map_err(Error::WorkspacePlanning)?;
        }
        let shape = value.as_array().shape();
        Ok(entry.matches_dimensions(shape.len(), |i| {
            shape.get(i).and_then(|n| usize::try_from(*n).ok())
        }))
    }
    fn capture_contract_error(
        error: eredu_runtime::SpeculativeCaptureError,
        context: Self::Context<'_>,
    ) -> Self::Error {
        match context.original_numerical() {
            Some((sources, _)) => sources.retain_startup_error(error),
            None => ordinary_error(error.to_string()),
        }
    }

    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error> {
        value
            .as_array()
            .shape()
            .iter()
            .map(|extent| {
                usize::try_from(*extent).map_err(|_| {
                    ordinary_error("external target capture has a negative tensor extent")
                })
            })
            .collect()
    }

    fn prefill_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        input.with_borrowed(|input: ModelInput<'_>| {
            target.prefill_external_prediction_target(input, request, cache)
        })
    }

    fn prefill_chunk_positions(input: &Self::Input) -> Option<std::num::NonZeroU64> {
        input.with_borrowed(|input| input.prefill_chunk_positions())
    }
    fn prefill_target_spans_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
            Self::Tensor,
            Self::Error,
        >,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_runtime::replicated_session::PrefillSourceProgress<Option<Self::Tensor>>,
        Self::Error,
    > {
        target.prefill_external_prediction_spans(
            input,
            request,
            cache,
            receiver,
            cancellation,
            context,
        )
    }
    fn supports_prefill_observation(assistant: &Self::Assistant, context_values: bool) -> bool {
        assistant.observers.supports_prefill(context_values)
    }
    fn prefill_output_demand(assistant: &Self::Assistant) -> eredu_core::OutputDemand {
        assistant.observers.prefill_output_demand()
    }
    fn begin_prefill_chunk(
        assistant: &mut Self::Assistant,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Self::Error> {
        assistant
            .observers
            .begin_prefill_chunk(chunk)
            .map_err(Error::from)
    }
    fn begin_prefill_context(
        assistant: &mut Self::Assistant,
        frontier: u64,
    ) -> Result<(), Self::Error> {
        assistant
            .observers
            .begin_prefill_context(frontier)
            .map_err(Error::from)
    }
    fn finish_prefill(assistant: &mut Self::Assistant, committed: bool) {
        assistant.observers.finish_prefill(committed);
    }

    fn verify_target_with_evidence<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::external_assistant::ExternalTargetResult<Self::Tensor>,
        Self::Error,
    > {
        target.verify_external_prediction_target_with_evidence(tokens, request, cache, context)
    }

    fn verify_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        target.verify_external_prediction_target(tokens, request, cache)
    }

    fn control_cache_estimate(
        cache: &Self::NativeCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        cache.control_estimate()
    }
    fn control_tensor_bytes(tensor: &Self::Tensor) -> Option<u64> {
        (tensor.as_array().nbytes() as u64)
            .checked_mul(2)?
            .checked_add(4096)
    }
    fn control_checkpoint<'a>(
        cache: &Self::NativeCache,
        context: Self::Context<'a>,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        if context.original_external().is_some() {
            original_cache_copy(cache, context)
        } else {
            cache
                .control_copy(context.target(), &context.memory_ledger())
                .map_err(Error::from)
        }
    }
    fn control_restore<'a>(
        cache: &mut Self::NativeCache,
        saved: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        let mut replacement = if context.original_external().is_some() {
            original_cache_copy(saved, context)?
        } else {
            saved.control_copy(context.target(), &context.memory_ledger())?
        };
        replacement.inherit_retention(cache)?;
        *cache = replacement;
        Ok(())
    }
    fn control_copy_tensor<'a>(
        tensor: &Self::Tensor,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if let Some((sources, _)) = context.original_numerical() {
            return Err(sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ));
        }
        let owner = NativeMemoryOwner::acquire(&context.memory_ledger()).map_err(Error::from)?;
        let stream = match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        };
        submission_recovery::detached_retained(owner.clone(), || {
            let copy = super::state_snapshot::copy_array(tensor.as_array(), stream)?;
            owner.retain_array(&copy)?;
            Ok(MlxTensor::from_array(copy))
        })
        .map_err(Error::from)
    }

    fn checkpoint_native(
        cache: &Self::NativeCache,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        cache.deep_clone().map_err(Error::from)
    }

    fn restore_checkpoint_native<'a>(
        cache: &mut Self::NativeCache,
        checkpoint: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        if context.original_external().is_some() {
            let mut replacement = original_cache_copy(checkpoint, context)?;
            replacement.inherit_retention(cache)?;
            *cache = replacement;
            Ok(())
        } else {
            cache
                .restore(checkpoint, context.target())
                .map_err(Error::from)
        }
    }

    fn native_cache_len(cache: &Self::NativeCache) -> Result<i32, Self::Error> {
        i32::try_from(cache.generation()?)
            .map_err(|_| ordinary_error("external target cache frontier exceeds i32"))
    }

    fn observe_tensor(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        if !assistant.observers.has_tensor_observer() {
            return Ok(value);
        }
        assistant
            .observers
            .observe_tensor(path, &value)
            .map_err(Error::from)
    }

    fn observe_logits(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        if !assistant.observers.has_logits_observer() {
            return Ok(value);
        }
        let value = value.ordinary().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        assistant
            .observers
            .observe_logits(path, value)
            .map(IndependentLogits::Ordinary)
            .map_err(Error::from)
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        usize::try_from(value.as_array().dim(1))
            .map_err(|_| ordinary_error("external assistant sequence length exceeds usize"))
    }

    fn sequence_row<'a>(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| ordinary_error("external assistant row exceeds i32"))?;
        let stream = match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        };
        let array = if retain_dimension {
            value
                .as_array()
                .try_index_device((.., row..row + 1, ..), stream)?
        } else {
            value.as_array().try_index_device((.., row, ..), stream)?
        };
        Ok(MlxTensor::from_array(array))
    }

    fn into_logits_with_source<'a>(
        value: Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        logits::selected(value, evidence, context)
    }

    fn into_logits_with_source_at<'a>(
        value: Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        logits::selected_at(value, evidence, placement, context)
    }

    fn logits_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        logits::row(value, row, evidence, placement, context)
    }

    fn into_logits(value: Self::Tensor) -> Self::Logits {
        IndependentLogits::Ordinary(value.into_array())
    }

    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let length = value.as_array().dim(1);
        if length <= maximum {
            Ok(value.clone())
        } else {
            value
                .as_array()
                .try_index_device((.., length - maximum.., ..), context.target())
                .map(MlxTensor::from_array)
                .map_err(Error::from)
        }
    }

    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let retained = value.as_array().dim(-2).min(cache_len);
        value
            .as_array()
            .try_index_device((.., .., ..retained, ..), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let end = i32::try_from(end)
            .map_err(|_| ordinary_error("external assistant prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end), context.target())
            .map(MlxTensor::from_array)
            .map_err(Error::from)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| ordinary_error("external assistant token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if context.crosses_devices() {
            value = value.copy(context.target())?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn transfer<'a>(
        value: &Self::Tensor,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if !context.is_split() {
            return Ok(value.clone());
        }
        if !context.crosses_devices() {
            match direction {
                ExternalAssistantTransfer::TargetToDraft => {
                    let _completion = context.wait_for_target_outputs([value.as_array()])?;
                }
                ExternalAssistantTransfer::DraftToTarget => {
                    let _completion = context.wait_for_draft_outputs([value.as_array()])?;
                }
            }
            return Ok(value.clone());
        }
        async_eval_with_event([value.as_array()])?.synchronize()?;
        let destination = match direction {
            ExternalAssistantTransfer::TargetToDraft => context.draft(),
            ExternalAssistantTransfer::DraftToTarget => context.target(),
        };
        let copied = value.as_array().copy(destination)?;
        async_eval_with_event([&copied])?.synchronize()?;
        Ok(MlxTensor::from_array(copied))
    }

    fn target_operation_with_source<'a>(
        target: &mut Self::Target,
        operation: ExternalPredictionTargetOperation<'_, Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_architectures::speculative_execution::EmbeddedPredictionTensor<Self::Tensor>,
        Self::Error,
    > {
        target.apply_external_prediction_target_operation_with_source(operation, context)
    }

    fn target_operation<'a>(
        target: &mut Self::Target,
        operation: ExternalPredictionTargetOperation<'_, Self::Tensor>,
        _context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        target.apply_external_prediction_target_operation(operation)
    }

    fn neural_context<'a>(
        context: Self::Context<'a>,
        placement: ExternalAssistantTensorPlacement,
    ) -> &'a Stream {
        match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        }
    }

    fn observe_borrowed_tensor<'a>(
        assistant: &mut Self::Assistant,
        path: &str,
        value: &Self::Tensor,
        evidence: Option<&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if context.original_numerical().is_some() {
            if assistant.observers.has_tensor_observer() {
                return Err(
                    <Self as ExternalAssistantExecutionMechanisms<A>>::state_refusal(context),
                );
            }
            return state::alias(
                value,
                evidence,
                ExternalAssistantTensorPlacement::Target,
                context,
            )
            .map(|value| value.output);
        }
        <Self as ExternalAssistantExecutionMechanisms<A>>::observe_tensor(
            assistant,
            path,
            value.clone(),
        )
    }
    fn state_host_metadata<'a>(
        bytes: Option<usize>,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        crate::composition::mlx::replicated_text::cache_metadata(bytes, context)
            .map_err(Error::StorageSource)
    }
    fn state_buffer<'a, T>(
        capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::SpeculativeBuffer<T>, Self::Error> {
        super::autoregressive::host_containers::buffer(capacity, context)
    }
    fn state_vector<T>(capacity: usize, context: Self::Context<'_>) -> Result<Vec<T>, Self::Error> {
        if let Some((sources, _)) = context.original_numerical() {
            let funding = sources.metadata_funding();
            (|| {
                funding
                    .reserve_metadata(std::mem::size_of::<(Vec<T>, Result<Vec<T>, Error>, usize)>())
                    .map_err(Error::WorkspacePlanning)?;
                funding.metadata_vec(capacity).map_err(Error::Neural)
            })()
            .map_err(|cause| sources.retain_error(cause))
        } else {
            Ok(Vec::with_capacity(capacity))
        }
    }
    fn state_buffer_bytes<T>(capacity: usize) -> Option<usize> {
        super::autoregressive::host_containers::buffer_bytes::<T>(capacity)
    }
    fn state_dimension<'a>(
        value: &Self::Tensor,
        axis: usize,
        context: Self::Context<'a>,
    ) -> Result<usize, Self::Error> {
        state::dimension(value, axis, context)
    }
    fn state_refusal<'a>(context: Self::Context<'a>) -> Self::Error {
        let cause = eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch;
        match context.original_numerical() {
            Some((sources, _)) => sources.retain_startup_error(cause),
            None => Error::PrefillControl(cause),
        }
    }
    fn submit_completion_with_sources<'a, 'c, I>(
        values: I,
        evidence: &[&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
        context: Self::Context<'c>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
        I::IntoIter: Clone,
    {
        let Some((sources, environment)) = context.original_numerical() else {
            return MlxSpeculativeCompletion::submit(values.into_iter().map(MlxTensor::as_array))
                .map(TypedSpeculativeCompletion::new)
                .map_err(Error::from);
        };
        let parts = [
            std::mem::size_of::<I>(),
            std::mem::size_of::<I::IntoIter>(),
            std::mem::size_of::<Result<Self::Completion, Error>>(),
            std::mem::size_of::<Self::Context<'_>>(),
        ];
        sources
            .metadata_funding()
            .reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(
                        eredu_nn::workspace::HostMetadataFundingError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let values = values.into_iter();
        // The repeatable source traversal validates each exact completed backing.
        let evidence = state::join(
            |f| {
                for value in values.clone() {
                    f(value)
                }
            },
            evidence,
            context,
        )?
        .ok_or_else(|| {
            sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )
        })?;
        TypedSpeculativeCompletion::completed_external(evidence, sources, environment)
    }

    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a,
    {
        MlxSpeculativeCompletion::submit(values.into_iter().map(MlxTensor::as_array))
            .map(TypedSpeculativeCompletion::new)
            .map_err(Error::from)
    }
}

// Same isolated source/copy worker used by existing original target snapshots.
// The current request supplies both its exact native environment and H custody.
fn original_cache_copy(
    cache: &MlxPredictionTargetState,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<MlxPredictionTargetState, Error> {
    let (sources, environment) = context.original_numerical().ok_or(Error::PrefillControl(
        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
    ))?;
    sources.validate_environment(environment)?;
    let funding = sources.metadata_funding();
    funding
        .reserve_metadata(std::mem::size_of::<(
            &MlxPredictionTargetState,
            SpeculativeExecutionStreams<'_>,
            MlxPredictionTargetState,
            Result<MlxPredictionTargetState, Error>,
        )>())
        .map_err(Error::WorkspacePlanning)?;
    let (roots, mechanisms) = sources.numerical_prerequisites();
    cache
        .copy_original(
            environment,
            roots,
            mechanisms,
            funding,
            sources.request().limits().clone(),
        )
        .map_err(|cause| sources.retain_error(cause))
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
