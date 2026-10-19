//! Architecture-owned speculative lifecycle for a Gemma 4 external assistant.

use std::marker::PhantomData;

use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

use super::{AssistantState, SharedAttentionStates};
use crate::{
    composite_execution::ExternalPredictionTargetCapture,
    external_assistant::{
        ExternalAssistantCache, ExternalAssistantCacheCheckpoint,
        ExternalAssistantExecutionMechanisms, ExternalAssistantTensorPlacement,
        ExternalAssistantTransfer, Gemma4AssistantArchitecture,
        EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
        EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,
    },
};

/// Architecture adapter from family-neutral backend mechanisms to Gemma lifecycle mechanisms.
pub struct ArchitectureExternalMechanisms<M>(PhantomData<fn() -> M>);

impl<M> ExternalMechanisms for ArchitectureExternalMechanisms<M>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture> + 'static,
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
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
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

    fn max_proposals(assistant: &Self::Assistant) -> usize {
        M::config(assistant).block_size.saturating_sub(1)
    }

    fn empty_input() -> Self::Error {
        M::error("Gemma 4 speculative input must contain at least one token".into())
    }

    fn invalid_commit(verified: usize, available: usize) -> Self::Error {
        M::error(format!(
            "cannot commit {verified} verified Gemma inputs from a block of {available}"
        ))
    }

    fn invalid_output(expected: usize, logits: usize, hidden: usize) -> Self::Error {
        M::error(format!(
            "Gemma target output lengths disagree: expected={expected}, logits={logits}, hidden={hidden}"
        ))
    }

    fn observe_tensor(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        M::observe_tensor(assistant, path, value)
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

    fn hidden_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::sequence_row(
            value,
            row,
            true,
            ExternalAssistantTensorPlacement::Target,
            context,
        )
    }

    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::shared_prefix(value, cache_len, context)
    }

    fn target_to_draft<'a>(
        value: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::transfer(value, ExternalAssistantTransfer::TargetToDraft, context)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::target_tokens(tokens, context)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::token_prefix(value, end, context)
    }

    fn proposal_embedding<'a>(
        target: &mut Self::Target,
        token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let ids = M::target_tokens(&[token], context)?;
        let embedding = M::target_operation(
            target,
            crate::composite_execution::ExternalPredictionTargetOperation::TokenEmbeddings(&ids),
            context,
        )?;
        M::transfer(
            &embedding,
            ExternalAssistantTransfer::TargetToDraft,
            context,
        )
    }

    fn draft_step<'a>(
        assistant: &mut Self::Assistant,
        embedding: &Self::Tensor,
        state: &mut AssistantState<Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let module = M::module(assistant);
        let logits = module
            .draft_step::<M::AttentionCache>(
                embedding,
                state,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map(M::into_logits)
            .map_err(M::neural_error)?;
        M::observe_logits(
            assistant,
            EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
            logits,
        )
    }

    fn submit_verification<'a>(
        output: &ExternalTargetOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        M::submit_completion(
            [&output.logits, &output.hidden, inputs].into_iter().chain(
                output
                    .shared_kv
                    .values()
                    .flat_map(|(keys, values)| [keys, values]),
            ),
        )
    }
}

fn observe_capture<M>(
    assistant: &mut M::Assistant,
    cache: &ExternalAssistantCache<M::NativeCache>,
    capture: &mut ExternalPredictionTargetCapture<M::Tensor>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = capture else {
        return Err(M::error(
            "Gemma target returned a different assistant capture".into(),
        ));
    };
    let paths = cache.capture_paths();
    if paths.len() != 1 + shared_kv.len() * 2 {
        return Err(M::error(format!(
            "Gemma target capture path count {} differs from tensor count {}",
            paths.len(),
            1 + shared_kv.len() * 2
        )));
    }
    *hidden = M::observe_tensor(assistant, paths[0], hidden.clone())?;
    let (shared_paths, remainder) = paths[1..].as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    for ((_, keys, values), paths) in shared_kv.iter_mut().zip(shared_paths) {
        *keys = M::observe_tensor(assistant, paths[0], keys.clone())?;
        *values = M::observe_tensor(assistant, paths[1], values.clone())?;
    }
    Ok(())
}

fn validate_output_capture<M>(
    cache: &ExternalAssistantCache<M::NativeCache>,
    output: &ExternalTargetOutput<M::Tensor>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    let mut shapes = Vec::with_capacity(1 + output.shared_kv.len() * 2);
    shapes.push(M::tensor_shape(&output.hidden)?);
    for (keys, values) in output.shared_kv.values() {
        shapes.push(M::tensor_shape(keys)?);
        shapes.push(M::tensor_shape(values)?);
    }
    cache.validate_capture_shapes(&shapes).map_err(M::error)
}

fn target_output<M>(
    logits: M::Tensor,
    capture: ExternalPredictionTargetCapture<M::Tensor>,
) -> Result<ExternalTargetOutput<M::Tensor>, M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = capture else {
        return Err(M::error(
            "Gemma 4 target returned a different assistant capture".into(),
        ));
    };
    Ok(ExternalTargetOutput {
        logits,
        hidden,
        shared_kv: shared_kv
            .into_iter()
            .map(|(policy, keys, values)| (policy, (keys, values)))
            .collect(),
    })
}

/// Exact ordinary-target output consumed by the Gemma 4 assistant lifecycle.
pub struct ExternalTargetOutput<T> {
    /// Target vocabulary logits.
    pub logits: T,
    /// Target-width hidden capture used to seed the assistant.
    pub hidden: T,
    /// Shared target K/V captures keyed by attention policy.
    pub shared_kv: SharedAttentionStates<T>,
}

/// Committed Gemma 4 target state from which one private assistant round starts.
#[derive(Debug, Clone)]
pub struct ExternalTargetState<T> {
    hidden: T,
    shared_kv: SharedAttentionStates<T>,
    cache_len: i32,
}

impl<T: Clone> ExternalTargetState<T> {
    fn map<E>(&self, mut f: impl FnMut(&T) -> Result<T, E>) -> Result<Self, E> {
        Ok(Self {
            hidden: f(&self.hidden)?,
            shared_kv: self
                .shared_kv
                .iter()
                .map(|(key, (k, v))| Ok((*key, (f(k)?, f(v)?))))
                .collect::<Result<_, E>>()?,
            cache_len: self.cache_len,
        })
    }
    fn tensor_bytes(&self, mut f: impl FnMut(&T) -> Option<u64>) -> Option<u64> {
        let mut bytes = (std::mem::size_of::<Self>() as u64).checked_add(f(&self.hidden)?)?;
        for (k, v) in self.shared_kv.values() {
            bytes = bytes
                .checked_add(f(k)?)?
                .checked_add(f(v)?)?
                .checked_add(256)?;
        }
        Some(bytes)
    }
}

/// Backend mechanisms required by the architecture-owned Gemma 4 lifecycle.
///
/// Implementations bind already materialized target and assistant objects to tensor, transfer,
/// cache, and completion operations. Proposal ordering, target-state geometry, replay, and commit
/// policy remain in [`ExternalExecutor`].
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
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
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
    /// Materialized Gemma 4 assistant.
    type Assistant: ?Sized;
    /// Architecture-proven target capture request.
    type Capture;
    /// Prepared ordinary-target input.
    type Input;
    /// Complete ordinary-target cache.
    type Cache;
    /// Exact cache checkpoint.
    type CacheCheckpoint;
    /// Retained native tensor.
    type Tensor: Clone;
    /// Native logits consumed by sampling.
    type Logits: Clone;
    /// Selected target/draft execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact verification completion.
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

    /// Maximum proposal count admitted by the assistant artifact.
    fn max_proposals(assistant: &Self::Assistant) -> usize;

    /// Constructs the stable empty-input failure.
    fn empty_input() -> Self::Error;

    /// Constructs the stable invalid-commit failure.
    fn invalid_commit(verified: usize, available: usize) -> Self::Error;

    /// Constructs the stable target-output geometry failure.
    fn invalid_output(expected: usize, logits: usize, hidden: usize) -> Self::Error;

    /// Observes and optionally replaces one architecture-owned activation.
    fn observe_tensor(
        _assistant: &mut Self::Assistant,
        _path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(value)
    }

    /// Observes and optionally replaces one complete target output and its capture.
    fn observe_output(
        assistant: &mut Self::Assistant,
        _request: &Self::Capture,
        output: &mut ExternalTargetOutput<Self::Tensor>,
        logits_path: &str,
    ) -> Result<(), Self::Error> {
        output.logits = Self::observe_tensor(assistant, logits_path, output.logits.clone())?;
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

    /// Runs ordinary-target prefill and returns the proven Gemma capture.
    fn prefill_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error>;

    /// Runs ordinary-target verification and returns the proven Gemma capture.
    fn verify_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        tokens: &Self::Tensor,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error>;

    /// Returns a fallible exact target-cache checkpoint.
    fn checkpoint(cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error>;

    /// Restores one exact target-cache checkpoint.
    fn restore_checkpoint<'a>(
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>;

    /// Returns the current committed target-cache frontier.
    fn cache_len(cache: &Self::Cache) -> Result<i32, Self::Error>;

    /// Returns the sequence width of a target tensor.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;

    /// Selects one logits row.
    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Selects one hidden row while retaining its sequence dimension.
    fn hidden_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Retains at most the committed prefix of one shared K/V tensor.
    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Moves or orders a target tensor for assistant execution.
    fn target_to_draft<'a>(
        value: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Constructs exact target token ids.
    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Selects a target-token prefix.
    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Applies the target-owned token embedding and makes it available to the assistant.
    fn proposal_embedding<'a>(
        target: &mut Self::Target,
        token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Executes one neutral assistant step.
    fn draft_step<'a>(
        assistant: &mut Self::Assistant,
        embedding: &Self::Tensor,
        state: &mut AssistantState<Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Submits every target output required after verification.
    fn submit_verification<'a>(
        output: &ExternalTargetOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error>;
}

/// Architecture-owned Gemma 4 external-assistant executor.
pub struct ExternalExecutor<'a, M: ExternalMechanisms> {
    target: &'a mut M::Target,
    assistant: &'a mut M::Assistant,
    capture: M::Capture,
    _mechanisms: PhantomData<fn() -> M>,
}

impl<'a, M: ExternalMechanisms> ExternalExecutor<'a, M> {
    /// Binds already materialized target and assistant objects to the neutral lifecycle.
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

    fn state_at<'context>(
        output: &ExternalTargetOutput<M::Tensor>,
        row: usize,
        cache_len: i32,
        context: M::Context<'context>,
    ) -> Result<ExternalTargetState<M::Tensor>, M::Error> {
        let hidden = M::hidden_row(&output.hidden, row, context)?;
        let shared_kv = output
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                Ok((
                    *policy,
                    (
                        M::shared_prefix(keys, cache_len, context)?,
                        M::shared_prefix(values, cache_len, context)?,
                    ),
                ))
            })
            .collect::<Result<SharedAttentionStates<_>, M::Error>>()?;
        Ok(ExternalTargetState {
            hidden,
            shared_kv,
            cache_len,
        })
    }

    fn validate_output(
        output: &ExternalTargetOutput<M::Tensor>,
        expected: usize,
    ) -> Result<(), M::Error> {
        let logits = M::sequence_len(&output.logits)?;
        let hidden = M::sequence_len(&output.hidden)?;
        if logits != expected || hidden != expected {
            return Err(M::invalid_output(expected, logits, hidden));
        }
        Ok(())
    }

    fn state_on_draft<'context>(
        state: &ExternalTargetState<M::Tensor>,
        context: M::Context<'context>,
    ) -> Result<ExternalTargetState<M::Tensor>, M::Error> {
        let hidden = M::target_to_draft(&state.hidden, context)?;
        let shared_kv = state
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                Ok((
                    *policy,
                    (
                        M::target_to_draft(keys, context)?,
                        M::target_to_draft(values, context)?,
                    ),
                ))
            })
            .collect::<Result<SharedAttentionStates<_>, M::Error>>()?;
        Ok(ExternalTargetState {
            hidden,
            shared_kv,
            cache_len: state.cache_len,
        })
    }
}

impl<M: ExternalMechanisms> SpeculativeExecutor for ExternalExecutor<'_, M> {
    fn coordinate_speculative_step<'context>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'context>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'context>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::agree_text_preparation(stage, status, context)
    }

    fn coordinate_decoding_control<'context>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>
    where
        Self: 'context,
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
    type DraftState = AssistantState<M::Tensor>;
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
            .map(|t| M::control_copy_tensor(t, ExternalAssistantTensorPlacement::Target, context))
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
            .map(|t| M::control_copy_tensor(t, ExternalAssistantTensorPlacement::Target, context))
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        M::control_restore(cache, saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(state))
    }

    fn max_proposals(&self) -> usize {
        M::max_proposals(self.assistant)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
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
            let row = sequence - 1;
            let logits = M::logits_row(&output.logits, row, context)?;
            let state = Self::state_at(&output, row, M::cache_len(cache)?, context)?;
            Ok(SpeculativePrefill::new(logits, state, sequence))
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
        _last_token: u32,
        _proposal_capacity: usize,
        context: M::Context<'context>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'context,
    {
        let state = Self::state_on_draft(state, context)?;
        Ok(AssistantState {
            shared_kv: state.shared_kv,
            kv_offset: state.cache_len,
            hidden: state.hidden,
        })
    }

    fn proposal_step<'context>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: M::Context<'context>,
    ) -> Result<eredu_core::speculative::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        let embedding = M::proposal_embedding(self.target, last_token, context)?;
        M::draft_step(self.assistant, &embedding, state, context)
            .map(eredu_core::speculative::SpeculativeProposalStep::CausalLogits)
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
        _draft_state: Self::DraftState,
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
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(M::invalid_commit(verified_inputs, input_len));
        }
        if verified_inputs == input_len {
            return Ok(SpeculativeCommit::new(
                Self::state_at(
                    &verified,
                    verified_inputs - 1,
                    M::cache_len(cache)?,
                    context,
                )?,
                0,
            ));
        }

        M::restore_checkpoint(cache, checkpoint, context)?;
        let retained = M::token_prefix(&inputs, verified_inputs, context)?;
        let mut replayed = M::verify_target(
            self.target,
            self.assistant,
            &self.capture,
            &retained,
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
        Ok(SpeculativeCommit::new(
            Self::state_at(
                &replayed,
                verified_inputs - 1,
                M::cache_len(cache)?,
                context,
            )?,
            verified_inputs,
        ))
    }
}
