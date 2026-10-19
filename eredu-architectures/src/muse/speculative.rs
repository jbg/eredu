//! Shared architecture-owned speculative lifecycle for Muse assistants.

use std::marker::PhantomData;

use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

use super::{
    assistant::integration::{MuseAssistantConfig, MuseAssistantProposal},
    DFlashContext,
};
use crate::{
    composite_execution::ExternalPredictionTargetCapture,
    external_assistant::{
        ExternalAssistantCache, ExternalAssistantCacheCheckpoint,
        ExternalAssistantExecutionMechanisms, ExternalAssistantTensorPlacement,
        ExternalAssistantTransfer, MuseAssistantArchitecture,
        EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
        EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,
    },
};

/// Architecture adapter from family-neutral backend mechanisms to DFlash lifecycle mechanisms.
pub struct ArchitectureExternalMechanisms<M>(PhantomData<fn() -> M>);

impl<M> ExternalMechanisms for ArchitectureExternalMechanisms<M>
where
    M: ExternalAssistantExecutionMechanisms<MuseAssistantArchitecture> + 'static,
{
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        M::coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure> {
        M::agree_text_preparation(stage, status, context)
    }

    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        M::coordinate_decoding_control(local, context)
    }

    type Target = M::Target;
    type Assistant = M::Assistant;
    type Capture = crate::composite_execution::ExternalPredictionCaptureRequest;
    type Input = M::Input;
    type Cache = ExternalAssistantCache<M::NativeCache>;
    type CacheCheckpoint = ExternalAssistantCacheCheckpoint<M::NativeCacheCheckpoint>;
    type Tensor = M::Tensor;
    type Proposal = MuseAssistantProposal<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn control_logits_estimate(
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::control_logits_estimate(logits)
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        M::trim_parameter_conversions(target, assistant)
    }

    fn parameter_conversion_retention(
        target: &Self::Target,
        assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        M::parameter_conversion_retention(target, assistant)
    }

    fn target_memory_observation(
        target: &Self::Target,
        cache: &Self::Cache,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        M::target_memory_observation(target, cache.native(), additional)
    }

    fn assistant_memory_observation(
        assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        M::assistant_memory_observation(assistant)
    }

    fn tensor_backing_bytes(
        tensors: &[&Self::Tensor],
    ) -> Result<Option<u64>, eredu_core::BackendFailure> {
        M::tensor_backing_bytes(tensors)
    }

    fn proposal_layout(assistant: &Self::Assistant) -> super::assistant::ProposalLayout {
        M::config(assistant).proposal_layout()
    }

    fn max_proposals(assistant: &Self::Assistant) -> usize {
        M::config(assistant).proposal_capacity()
    }

    fn mask_token(assistant: &Self::Assistant) -> u32 {
        M::config(assistant).mask_token_id()
    }

    fn sliding_window(assistant: &Self::Assistant) -> i32 {
        M::config(assistant).sliding_window()
    }

    fn empty_input() -> Self::Error {
        M::error("Muse assistant input is empty".into())
    }

    fn invalid_capacity(maximum: usize) -> Self::Error {
        M::error(format!(
            "Muse assistant proposal capacity must be between 1 and {maximum}"
        ))
    }

    fn missing_context() -> Self::Error {
        M::error("DFlash proposal has neither pending nor committed context".into())
    }

    fn block_exhausted() -> Self::Error {
        M::error("Muse assistant block is exhausted".into())
    }

    fn state_mismatch() -> Self::Error {
        M::error("Muse assistant verification/checkpoint state mismatch".into())
    }

    fn invalid_output(expected: usize, logits: usize, captures: &[usize]) -> Self::Error {
        M::error(format!(
            "Muse target output lengths disagree: expected={expected}, logits={logits}, captures={captures:?}"
        ))
    }

    fn observe_logits(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        M::observe_logits(assistant, path, value)
    }

    fn observe_output(
        assistant: &mut Self::Assistant,
        _request: &Self::Capture,
        output: &mut ExternalTargetOutput<Self::Tensor>,
        logits_path: &str,
    ) -> Result<(), Self::Error> {
        output.logits = M::observe_tensor(assistant, logits_path, output.logits.clone())?;
        Ok(())
    }

    fn validate_output_capture(
        cache: &Self::Cache,
        output: &ExternalTargetOutput<Self::Tensor>,
    ) -> Result<(), Self::Error> {
        validate_output_capture::<M>(cache, output)
    }

    fn bind_prepared_input(
        input: &Self::Input,
        cache: &mut Self::Cache,
    ) -> Result<(), Self::Error> {
        cache
            .bind_prepared_input_cache_identity(&M::prepared_input_cache_identity(input)?)
            .map_err(M::error)
    }

    fn prefill_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
        let (logits, mut capture) =
            M::prefill_target_native(target, request, input, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant, cache, &mut capture)?;
        target_output::<M>(logits, capture)
    }

    fn verify_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        tokens: &Self::Tensor,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
        let (logits, mut capture) =
            M::verify_target_native(target, request, tokens, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant, cache, &mut capture)?;
        target_output::<M>(logits, capture)
    }

    fn control_cache_estimate(
        cache: &Self::Cache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let native = M::control_cache_estimate(cache.native())?;
        let host = cache.control_metadata_bytes()?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(host)?,
            copy_bytes: native.copy_bytes.checked_add(host)?,
        })
    }
    fn control_tensor_bytes(tensor: &Self::Tensor) -> Option<u64> {
        M::control_tensor_bytes(tensor)
    }
    fn control_copy_tensor<'a>(
        tensor: &Self::Tensor,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::control_copy_tensor(tensor, placement, context)
    }
    fn control_checkpoint<'a>(
        cache: &Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Self::CacheCheckpoint, Self::Error> {
        M::control_checkpoint(cache.native(), context).map(|native| cache.checkpoint(native))
    }
    fn control_restore<'a>(
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        M::control_restore(cache.native_mut(), saved.native(), context)?;
        cache.restore_semantics(saved);
        Ok(())
    }

    fn checkpoint(cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        M::checkpoint_native(cache.native()).map(|native| cache.checkpoint(native))
    }

    fn restore_checkpoint<'a>(
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        M::restore_checkpoint_native(cache.native_mut(), checkpoint.native(), context)?;
        cache.restore_semantics(checkpoint);
        Ok(())
    }

    fn cache_len(cache: &Self::Cache) -> Result<i32, Self::Error> {
        cache.frontier().map_err(M::error)
    }

    fn checkpoint_len(checkpoint: &Self::CacheCheckpoint) -> Result<i32, Self::Error> {
        checkpoint.frontier().map_err(M::error)
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        M::sequence_len(value)
    }

    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::sequence_row(
            value,
            row,
            false,
            ExternalAssistantTensorPlacement::Target,
            context,
        )
        .map(M::into_logits)
    }

    fn proposal_logits_row<'a>(
        assistant: &mut Self::Assistant,
        value: &Self::Proposal,
        row: usize,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let result = M::module(assistant)
            .proposal_row(
                value,
                row,
                last_token,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)?;
        if let Some(confidence) = result.confidence {
            M::observe_tensor(assistant, "external_assistant.confidence", confidence)?;
        }
        Ok(M::into_logits(result.logits))
    }

    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::sequence_suffix(value, maximum, context)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::token_prefix(value, end, context)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::target_tokens(tokens, context)
    }

    fn target_to_draft<'a>(
        value: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::transfer(value, ExternalAssistantTransfer::TargetToDraft, context)
    }

    fn assemble_target_states<'a>(
        assistant: &mut Self::Assistant,
        states: &[Self::Tensor],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::module(assistant)
            .assemble_target_states(
                states,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)
    }

    fn update_context<'a>(
        assistant: &mut Self::Assistant,
        previous: Option<DFlashContext<Self::Tensor>>,
        pending: &Self::Tensor,
        absolute_end: i32,
        context: Self::Context<'a>,
    ) -> Result<DFlashContext<Self::Tensor>, Self::Error> {
        M::module(assistant)
            .update_context(
                previous,
                pending,
                absolute_end,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)
    }

    fn target_embeddings<'a>(
        target: &mut Self::Target,
        assistant: &Self::Assistant,
        ids: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let ids = M::target_tokens(ids, context)?;
        let embeddings = M::target_operation(
            target,
            M::config(assistant).embedding_operation(&ids),
            context,
        )?;
        M::transfer(
            &embeddings,
            ExternalAssistantTransfer::TargetToDraft,
            context,
        )
    }

    fn proposal_states<'a>(
        assistant: &mut Self::Assistant,
        embeddings: &Self::Tensor,
        committed: &DFlashContext<Self::Tensor>,
        absolute_end: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let requirements = M::module(assistant)
            .definition()
            .tensor_parallel_requirements()
            .map_err(M::neural_error)?;
        let parallel = requirements
            .as_ref()
            .map(|requirements| M::assistant_parallel_context(context, requirements))
            .transpose()?
            .flatten();
        M::module(assistant)
            .proposal_states_selected(
                embeddings,
                committed,
                absolute_end,
                parallel,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)
    }

    fn prepare_proposal<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        states: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Proposal, Self::Error> {
        let vocabulary = match M::config(assistant) {
            MuseAssistantConfig::Spark(c) => i32::try_from(c.released().vocabulary_size())
                .map_err(|_| M::error("Muse vocabulary exceeds i32".into()))?,
            MuseAssistantConfig::Glimmer(_) => 0,
        };
        // Transfer only when the architecture declares a target-owned head.
        let projected = if M::module(assistant).target_projection(states).is_some() {
            let target_states =
                M::transfer(states, ExternalAssistantTransfer::DraftToTarget, context)?;
            let operation = M::module(assistant)
                .target_projection(&target_states)
                .ok_or_else(|| M::error("Muse target projection contract changed".into()))?;
            let projected = M::target_operation(target, operation, context)?;
            Some(M::transfer(
                &projected,
                ExternalAssistantTransfer::TargetToDraft,
                context,
            )?)
        } else {
            None
        };
        M::module(assistant)
            .prepare_proposal(
                states,
                projected.as_ref(),
                vocabulary,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)
    }

    fn submit_verification<'a>(
        output: &ExternalTargetOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        M::submit_completion(
            std::iter::once(&output.logits)
                .chain(output.target_states.iter())
                .chain(std::iter::once(inputs)),
        )
    }
}

fn observe_capture<M>(
    assistant: &mut M::Assistant,
    cache: &ExternalAssistantCache<M::NativeCache>,
    capture: &mut ExternalPredictionTargetCapture<M::Tensor>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<MuseAssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::Muse { target_states } = capture else {
        return Err(M::error(
            "Muse target returned a different assistant capture".into(),
        ));
    };
    let paths = cache.capture_paths();
    if paths.len() != target_states.len() {
        return Err(M::error(format!(
            "Muse target capture path count {} differs from state count {}",
            paths.len(),
            target_states.len()
        )));
    }
    for (path, state) in paths.into_iter().zip(target_states) {
        *state = M::observe_tensor(assistant, path, state.clone())?;
    }
    Ok(())
}

fn validate_output_capture<M>(
    cache: &ExternalAssistantCache<M::NativeCache>,
    output: &ExternalTargetOutput<M::Tensor>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<MuseAssistantArchitecture>,
{
    let shapes = output
        .target_states
        .iter()
        .map(M::tensor_shape)
        .collect::<Result<Vec<_>, _>>()?;
    cache.validate_capture_shapes(&shapes).map_err(M::error)
}

fn target_output<M>(
    logits: M::Tensor,
    capture: ExternalPredictionTargetCapture<M::Tensor>,
) -> Result<ExternalTargetOutput<M::Tensor>, M::Error>
where
    M: ExternalAssistantExecutionMechanisms<MuseAssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::Muse { target_states } = capture else {
        return Err(M::error(
            "Muse target returned a different assistant capture".into(),
        ));
    };
    Ok(ExternalTargetOutput {
        logits,
        target_states,
    })
}

/// Exact target output and ordered DFlash capture.
pub struct ExternalTargetOutput<T> {
    /// Target vocabulary logits.
    pub logits: T,
    /// Target layer states in the order proven by assistant compatibility.
    pub target_states: Vec<T>,
}

/// Committed target and DFlash context state.
#[derive(Clone)]
pub struct ExternalTargetState<T: Clone> {
    pending_context: Option<T>,
    draft_context: Option<DFlashContext<T>>,
    cache_len: i32,
}

impl<T: Clone> ExternalTargetState<T> {
    fn map<E>(
        &self,
        mut f: impl FnMut(&T, ExternalAssistantTensorPlacement) -> Result<T, E>,
    ) -> Result<Self, E> {
        let pending_context = self
            .pending_context
            .as_ref()
            .map(|t| f(t, ExternalAssistantTensorPlacement::Target))
            .transpose()?;
        let draft_context = self
            .draft_context
            .as_ref()
            .map(|c| -> Result<_, E> {
                Ok(DFlashContext {
                    encoded: f(&c.encoded, ExternalAssistantTensorPlacement::Draft)?,
                    layers: c
                        .layers
                        .iter()
                        .map(|l| {
                            Ok(super::DFlashLayerContext {
                                keys: f(&l.keys, ExternalAssistantTensorPlacement::Draft)?,
                                values: f(&l.values, ExternalAssistantTensorPlacement::Draft)?,
                            })
                        })
                        .collect::<Result<_, E>>()?,
                    start: c.start,
                    end: c.end,
                })
            })
            .transpose()?;
        Ok(Self {
            pending_context,
            draft_context,
            cache_len: self.cache_len,
        })
    }
    fn projected_context_tensors(&self) -> Vec<&T> {
        let mut tensors = Vec::new();
        if let Some(context) = &self.draft_context {
            tensors.push(&context.encoded);
            for layer in &context.layers {
                tensors.extend([&layer.keys, &layer.values]);
            }
        }
        tensors
    }

    fn tensor_bytes(&self, mut f: impl FnMut(&T) -> Option<u64>) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>() as u64;
        if let Some(t) = &self.pending_context {
            bytes = bytes.checked_add(f(t)?)?;
        }
        if let Some(c) = &self.draft_context {
            bytes = bytes.checked_add(f(&c.encoded)?)?;
            for l in &c.layers {
                bytes = bytes
                    .checked_add(f(&l.keys)?)?
                    .checked_add(f(&l.values)?)?
                    .checked_add(256)?;
            }
        }
        Some(bytes)
    }
}

/// Private proposal payload and committed context, isolated from target publication.
#[derive(Clone)]
pub struct ExternalDraftState<T: Clone, P: Clone> {
    proposal: Option<P>,
    cursor: usize,
    proposal_capacity: usize,
    draft_context: DFlashContext<T>,
    cache_len: i32,
}

/// Mechanisms consumed by the shared architecture-owned Muse assistant lifecycle.
pub trait ExternalMechanisms: 'static {
    /// Resolves cancellation and completion across the retained target session.
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Agrees host preparation before any participant enters the next phase.
    fn agree_text_preparation(
        _stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure> {
        use eredu_core::run_preparation::{TextPreparationOutcome as O, TextPreparationStatus as S};
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }

    /// Agrees explicit commands without executing model work or restoring credits.
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Complete score-row backing allowance for a settled mode handoff.
    fn control_logits_estimate(
        _logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Materialized ordinary target.
    type Target: ?Sized;
    /// Materialized DFlash assistant.
    type Assistant: ?Sized;
    /// Architecture-proven ordered capture request.
    type Capture;
    /// Prepared target input.
    type Input;
    /// Complete target cache.
    type Cache;
    /// Exact target checkpoint.
    type CacheCheckpoint;
    /// Retained native tensor.
    type Tensor: Clone;
    /// Immutable prepared neural proposal data, retained on draft placement.
    type Proposal: Clone;
    /// Native sampling logits.
    type Logits: Clone;
    /// Selected target/draft assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact target completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        _target: &mut Self::Target,
        _assistant: &mut Self::Assistant,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads participant budgets without touching caches or submitted work.
    fn parameter_conversion_retention(
        _target: &Self::Target,
        _assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Reads the installed target frontier, native capacity and parameter owners.
    fn target_memory_observation(
        _target: &Self::Target,
        _cache: &Self::Cache,
        _additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Reads the current assistant parameter owner without acquiring any unit.
    fn assistant_memory_observation(
        _assistant: &Self::Assistant,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeParameterMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Complete available backing upper allowance for one borrowed tensor survey.
    /// Unknown or foreign backing stays unknown; observation never evaluates work.
    fn tensor_backing_bytes(
        _tensors: &[&Self::Tensor],
    ) -> Result<Option<u64>, eredu_core::BackendFailure> {
        Ok(None)
    }

    /// Known bound for an isolated reusable target-cache checkpoint.
    fn control_cache_estimate(
        _cache: &Self::Cache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Complete conservative logical tensor-copy bound.
    fn control_tensor_bytes(_tensor: &Self::Tensor) -> Option<u64> {
        None
    }
    /// Isolated native checkpoint, available only with a known control estimate.
    fn control_checkpoint<'a>(
        cache: &Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Self::CacheCheckpoint, Self::Error> {
        let _ = context;
        Self::checkpoint(cache)
    }
    /// Atomic isolated native restoration; never consumes saved state.
    fn control_restore<'a>(
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        Self::restore_checkpoint(cache, saved, context)
    }
    /// Copies one immutable seed tensor after budget admission.
    fn control_copy_tensor<'a>(
        tensor: &Self::Tensor,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (placement, context);
        Ok(tensor.clone())
    }

    /// Meaning of the anchor position in the neural proposal input.
    fn proposal_layout(assistant: &Self::Assistant) -> super::assistant::ProposalLayout;
    /// Released assistant sampled-slot capacity.
    fn max_proposals(assistant: &Self::Assistant) -> usize;
    /// Assistant mask token.
    fn mask_token(assistant: &Self::Assistant) -> u32;
    /// Assistant retained context window.
    fn sliding_window(assistant: &Self::Assistant) -> i32;
    /// Stable empty-input failure.
    fn empty_input() -> Self::Error;
    /// Stable invalid-capacity failure.
    fn invalid_capacity(maximum: usize) -> Self::Error;
    /// Stable missing-context failure.
    fn missing_context() -> Self::Error;
    /// Stable exhausted-block failure.
    fn block_exhausted() -> Self::Error;
    /// Stable verification/checkpoint mismatch failure.
    fn state_mismatch() -> Self::Error;
    /// Stable target-output geometry failure.
    fn invalid_output(expected: usize, logits: usize, captures: &[usize]) -> Self::Error;

    /// Observes and optionally replaces one proposal logits row.
    fn observe_logits(
        _assistant: &mut Self::Assistant,
        _path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(value)
    }

    /// Observes and optionally replaces one complete target output and its capture.
    fn observe_output(
        _assistant: &mut Self::Assistant,
        _request: &Self::Capture,
        _output: &mut ExternalTargetOutput<Self::Tensor>,
        _logits_path: &str,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Validates the intervened capture against its selected realization.
    fn validate_output_capture(
        _cache: &Self::Cache,
        _output: &ExternalTargetOutput<Self::Tensor>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Binds the exact prepared-input identity before target execution.
    fn bind_prepared_input(input: &Self::Input, cache: &mut Self::Cache)
        -> Result<(), Self::Error>;

    /// Runs target prefill and returns ordered captured states.
    fn prefill_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error>;
    /// Runs target verification and returns ordered captured states.
    fn verify_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        tokens: &Self::Tensor,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error>;
    /// Captures a fallible target checkpoint.
    fn checkpoint(cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error>;
    /// Restores a target checkpoint.
    fn restore_checkpoint<'a>(
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>;
    /// Current committed target frontier.
    fn cache_len(cache: &Self::Cache) -> Result<i32, Self::Error>;
    /// Committed frontier retained by a checkpoint.
    fn checkpoint_len(checkpoint: &Self::CacheCheckpoint) -> Result<i32, Self::Error>;
    /// Sequence width of logits or tokens.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;
    /// Selects one logits row.
    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Computes one conditional proposal distribution from the actual sampled predecessor.
    fn proposal_logits_row<'a>(
        assistant: &mut Self::Assistant,
        value: &Self::Proposal,
        row: usize,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Retains a suffix along the target-state sequence axis.
    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Selects a target-token prefix.
    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Constructs target token ids.
    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Moves or orders target output for draft execution.
    fn target_to_draft<'a>(
        value: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Validates and concatenates ordered target states.
    fn assemble_target_states<'a>(
        assistant: &mut Self::Assistant,
        states: &[Self::Tensor],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Encodes newly committed target context.
    fn update_context<'a>(
        assistant: &mut Self::Assistant,
        previous: Option<DFlashContext<Self::Tensor>>,
        pending: &Self::Tensor,
        absolute_end: i32,
        context: Self::Context<'a>,
    ) -> Result<DFlashContext<Self::Tensor>, Self::Error>;
    /// Applies the target-owned embedding and returns it on draft placement.
    fn target_embeddings<'a>(
        target: &mut Self::Target,
        assistant: &Self::Assistant,
        ids: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Runs the neutral proposal body, returning sampled-position states on draft placement.
    fn proposal_states<'a>(
        assistant: &mut Self::Assistant,
        embeddings: &Self::Tensor,
        committed: &DFlashContext<Self::Tensor>,
        absolute_end: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Prepares draft-local scoring data through the selected own/shared head policy.
    fn prepare_proposal<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        states: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Proposal, Self::Error>;
    /// Submits logits and every ordered captured target state.
    fn submit_verification<'a>(
        output: &ExternalTargetOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error>;
}

/// Shared Muse assistant proposal, verification and commitment executor.
pub struct ExternalExecutor<'a, M: ExternalMechanisms> {
    target: &'a mut M::Target,
    assistant: &'a mut M::Assistant,
    capture: M::Capture,
    _mechanisms: PhantomData<fn() -> M>,
}

impl<'a, M: ExternalMechanisms> ExternalExecutor<'a, M> {
    /// Binds materialized target and assistant objects to the neutral lifecycle.
    pub const fn new(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: M::Capture,
    ) -> Self {
        Self {
            target,
            assistant,
            capture,
            _mechanisms: PhantomData,
        }
    }

    fn target_state<'context>(
        &mut self,
        output: &ExternalTargetOutput<M::Tensor>,
        draft_context: Option<DFlashContext<M::Tensor>>,
        cache_len: i32,
        context: M::Context<'context>,
    ) -> Result<ExternalTargetState<M::Tensor>, M::Error> {
        let assembled = M::assemble_target_states(self.assistant, &output.target_states, context)?;
        let pending = M::sequence_suffix(&assembled, M::sliding_window(self.assistant), context)?;
        Ok(ExternalTargetState {
            pending_context: Some(pending),
            draft_context,
            cache_len,
        })
    }

    fn validate_output(
        output: &ExternalTargetOutput<M::Tensor>,
        expected: usize,
    ) -> Result<(), M::Error> {
        let logits = M::sequence_len(&output.logits)?;
        let captures = output
            .target_states
            .iter()
            .map(M::sequence_len)
            .collect::<Result<Vec<_>, _>>()?;
        if logits != expected || captures.iter().any(|&capture| capture != expected) {
            return Err(M::invalid_output(expected, logits, &captures));
        }
        Ok(())
    }
}

fn block_token_ids(
    layout: super::assistant::ProposalLayout,
    anchor: u32,
    mask: u32,
    proposal_capacity: usize,
) -> Option<Vec<u32>> {
    let positions = match layout {
        super::assistant::ProposalLayout::SampleFromAnchor => proposal_capacity,
        super::assistant::ProposalLayout::AnchorThenMasks => proposal_capacity.checked_add(1)?,
    };
    if positions == 0 {
        return None;
    }
    let mut ids = vec![mask; positions];
    ids[0] = anchor;
    Some(ids)
}

impl<M: ExternalMechanisms> SpeculativeExecutor for ExternalExecutor<'_, M> {
    fn coordinate_speculative_step<'context>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'context>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>
    where Self: 'context,
    {
        M::coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'context>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    where Self: 'context,
    {
        M::agree_text_preparation(stage, status, context)
    }

    fn coordinate_decoding_control<'context>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>
    where Self: 'context,
    {
        M::coordinate_decoding_control(local, context)
    }

    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        eredu_core::decoding::DecodingCapabilities::causal_drafting(self.max_proposals())
            .with_completion_support(
                M::Completion::supports_cancellation(
                    eredu_core::CompletionCancellationMode::NativeCancel,
                ),
                M::Completion::supports_cancellation(
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                ),
            )
    }

    fn logits_snapshot_estimate(
        &self,
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::control_logits_estimate(logits)
    }

    type Input = M::Input;
    type Cache = M::Cache;
    type TargetState = ExternalTargetState<M::Tensor>;
    type DraftState = ExternalDraftState<M::Tensor, M::Proposal>;
    type CacheCheckpoint = M::CacheCheckpoint;
    type Verification =
        eredu_runtime::speculative::CausalVerification<ExternalTargetOutput<M::Tensor>, M::Tensor>;
    type Logits = M::Logits;
    type Context<'a>
        = M::Context<'a>
    where
        Self: 'a;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        M::trim_parameter_conversions(self.target, self.assistant)
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        M::parameter_conversion_retention(self.target, self.assistant)
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        let (Some(target), Some(assistant)) = (
            M::target_memory_observation(self.target, cache, additional_positions)?,
            M::assistant_memory_observation(self.assistant)?,
        ) else {
            return Ok(None);
        };
        let positions =
            u64::try_from(state.cache_len).map_err(eredu_core::BackendFailure::from_error)?;
        if positions != target.current_positions {
            return Err(eredu_core::BackendFailure::from_error(
                eredu_nn::Error::backend("conditioned context and target cache frontiers disagree"),
            ));
        }
        // Pending captures still live on target placement; projected context
        // lives on the assistant. The shared seed allowance covers every execution
        // pool, including source backing before a cross-device transfer.
        let seed_bytes =
            M::tensor_backing_bytes(&state.pending_context.iter().collect::<Vec<_>>())?;
        let backing = M::tensor_backing_bytes(&state.projected_context_tensors())?;
        let current = backing.and_then(|bytes| bytes.checked_add(state.tensor_bytes(|_| Some(0))?));
        Ok(Some(
            eredu_core::speculative::SpeculativeContinuationObservation {
                target,
                draft: Some(eredu_core::speculative::SpeculativeModelMemoryObservation {
                    current_positions: positions,
                    current_state_bytes: current,
                    // Current backing cannot predict the native allocator's future
                    // buffer capacity. Runtime adds the declared logical envelope.
                    peak_state_bytes: (additional_positions == 0).then_some(current).flatten(),
                    parameters: assistant.parameters,
                    available: assistant.available,
                    allocator_cache_limit: assistant.allocator_cache_limit,
                    parameter_conversions: assistant.parameter_conversions,
                }),
                embedded: None,
                parameter_conversions: None,
                // No independent proposal block survives this canonical boundary.
                // Unknown lazy capture backing remains unknown without evaluation.
                seed_bytes,
            },
        ))
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let native = M::control_cache_estimate(cache)?;
        let seed = state.tensor_bytes(M::control_tensor_bytes)?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: native.retained_bytes.checked_add(seed)?,
            copy_bytes: native.copy_bytes.checked_add(seed)?,
        })
    }
    fn control_snapshot<'a>(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<
        Option<(Self::CacheCheckpoint, Self::TargetState)>,
        eredu_core::speculative::SpeculativeControlError,
    >
    where
        Self: 'a,
    {
        if self.control_snapshot_estimate(cache, state).is_none() {
            return Ok(None);
        }
        let state = state
            .map(|t, placement| M::control_copy_tensor(t, placement, context))
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some((
            M::control_checkpoint(cache, context)
                .map_err(eredu_core::speculative::SpeculativeControlError::backend)?,
            state,
        )))
    }
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'a,
    {
        let state = state
            .map(|t, placement| M::control_copy_tensor(t, placement, context))
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        M::control_restore(cache, saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(state))
    }

    fn max_proposals(&self) -> usize {
        M::max_proposals(self.assistant)
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        M::bind_prepared_input(&input, cache)?;
        let checkpoint = M::checkpoint(cache)?;
        let result = (|| {
            let mut output = M::prefill_target(
                self.target,
                self.assistant,
                &self.capture,
                input,
                cache,
                context,
            )?;
            M::observe_output(
                self.assistant,
                &self.capture,
                &mut output,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
            )?;
            M::validate_output_capture(cache, &output)?;
            let sequence = M::sequence_len(&output.logits)?;
            if sequence == 0 {
                return Err(M::empty_input());
            }
            Self::validate_output(&output, sequence)?;
            Ok(SpeculativePrefill::new(
                M::logits_row(&output.logits, sequence - 1, context)?,
                self.target_state(&output, None, M::cache_len(cache)?, context)?,
                sequence,
            ))
        })();
        match result {
            Ok(prefill) => Ok(prefill),
            Err(error) => {
                M::restore_checkpoint(cache, &checkpoint, context)?;
                Err(error)
            }
        }
    }

    fn begin_proposal<'context>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: M::Context<'context>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'context,
    {
        let maximum = self.max_proposals();
        if proposal_capacity > maximum {
            return Err(M::invalid_capacity(maximum));
        }
        let pending = state
            .pending_context
            .as_ref()
            .map(|pending| M::target_to_draft(pending, context))
            .transpose()?;
        let draft_context = match pending {
            Some(pending) => M::update_context(
                self.assistant,
                state.draft_context.clone(),
                &pending,
                state.cache_len,
                context,
            )?,
            None => state.draft_context.clone().ok_or_else(M::missing_context)?,
        };
        if proposal_capacity == 0 {
            return Ok(ExternalDraftState {
                proposal: None,
                cursor: 0,
                proposal_capacity,
                draft_context,
                cache_len: state.cache_len,
            });
        }
        let ids = block_token_ids(
            M::proposal_layout(self.assistant),
            last_token,
            M::mask_token(self.assistant),
            proposal_capacity,
        )
        .ok_or_else(|| M::invalid_capacity(maximum))?;
        let embeddings = M::target_embeddings(self.target, self.assistant, &ids, context)?;
        let states = M::proposal_states(
            self.assistant,
            &embeddings,
            &draft_context,
            state.cache_len,
            context,
        )?;
        let proposal = M::prepare_proposal(self.target, self.assistant, &states, context)?;
        Ok(ExternalDraftState {
            proposal: Some(proposal),
            cursor: 0,
            proposal_capacity,
            draft_context,
            cache_len: state.cache_len,
        })
    }

    fn proposal_step<'context>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: M::Context<'context>,
    ) -> Result<eredu_core::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        if state.cursor >= state.proposal_capacity {
            return Err(M::block_exhausted());
        }
        let row = state.cursor;
        let logits = M::proposal_logits_row(
            self.assistant,
            state
                .proposal
                .as_ref()
                .ok_or_else(|| M::invalid_capacity(0))?,
            row,
            last_token,
            context,
        )?;
        let logits = M::observe_logits(
            self.assistant,
            EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
            logits,
        )?;
        state.cursor += 1;
        Ok(eredu_core::SpeculativeProposalStep::CausalLogits(logits))
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        M::checkpoint(cache)
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
        M::restore_checkpoint(cache, checkpoint, context)
    }

    fn submit_verification<'context>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: M::Context<'context>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'context,
    {
        let inputs = M::target_tokens(input_tokens, context)?;
        let mut output = M::verify_target(
            self.target,
            self.assistant,
            &self.capture,
            &inputs,
            cache,
            context,
        )?;
        M::observe_output(
            self.assistant,
            &self.capture,
            &mut output,
            EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,
        )?;
        M::validate_output_capture(cache, &output)?;
        Self::validate_output(&output, input_tokens.len())?;
        let completion = M::submit_verification(&output, &inputs, context)?;
        Ok(Submission {
            output: eredu_runtime::speculative::CausalVerification::new(output, inputs),
            completion,
        })
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
        M::logits_row(&output.output().logits, index, context)
    }

    fn commit_verification<'context>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: M::Context<'context>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'context,
    {
        let (verified, inputs) = output.into_parts();
        let input_len = M::sequence_len(&inputs)?;
        let checkpoint_offset = M::checkpoint_len(checkpoint)?;
        if verified_inputs > input_len
            || draft_state.cache_len != checkpoint_offset
            || draft_state.draft_context.end != checkpoint_offset
        {
            return Err(M::state_mismatch());
        }
        if verified_inputs == 0 {
            M::restore_checkpoint(cache, checkpoint, context)?;
            return Ok(SpeculativeCommit::new(
                ExternalTargetState {
                    pending_context: None,
                    draft_context: Some(draft_state.draft_context),
                    cache_len: checkpoint_offset,
                },
                0,
            ));
        }
        let (retained, replayed_tokens) = if verified_inputs == input_len {
            (verified, 0)
        } else {
            M::restore_checkpoint(cache, checkpoint, context)?;
            let inputs = M::token_prefix(&inputs, verified_inputs, context)?;
            let mut replayed = M::verify_target(
                self.target,
                self.assistant,
                &self.capture,
                &inputs,
                cache,
                context,
            )?;
            M::observe_output(
                self.assistant,
                &self.capture,
                &mut replayed,
                EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,
            )?;
            M::validate_output_capture(cache, &replayed)?;
            Self::validate_output(&replayed, verified_inputs)?;
            (replayed, verified_inputs)
        };
        Ok(SpeculativeCommit::new(
            self.target_state(
                &retained,
                Some(draft_state.draft_context),
                M::cache_len(cache)?,
                context,
            )?,
            replayed_tokens,
        ))
    }
}
