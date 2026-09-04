//! Architecture-owned speculative lifecycle for a Muse-Glimmer DFlash assistant.

use std::marker::PhantomData;

use eredu_core::{
    Completion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

use super::DFlashContext;
use crate::{
    composite_execution::ExternalPredictionTargetCapture,
    external_assistant::{
        ExternalAssistantCache, ExternalAssistantCacheCheckpoint,
        ExternalAssistantExecutionMechanisms, ExternalAssistantTensorPlacement,
        ExternalAssistantTransfer, MuseGlimmerAssistantArchitecture,
        EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
        EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,
    },
};

/// Architecture adapter from family-neutral backend mechanisms to DFlash lifecycle mechanisms.
pub struct ArchitectureExternalMechanisms<M>(PhantomData<fn() -> M>);

impl<M> ExternalMechanisms for ArchitectureExternalMechanisms<M>
where
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture> + 'static,
{
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

    fn max_proposals(assistant: &Self::Assistant) -> usize {
        M::config(assistant).block_size.saturating_sub(1).min(15)
    }

    fn mask_token(assistant: &Self::Assistant) -> u32 {
        M::config(assistant).mask_token_id
    }

    fn sliding_window(assistant: &Self::Assistant) -> i32 {
        M::config(assistant).sliding_window
    }

    fn empty_input() -> Self::Error {
        M::error("Muse-Glimmer DFlash input is empty".into())
    }

    fn invalid_capacity(maximum: usize) -> Self::Error {
        M::error(format!(
            "Muse-Glimmer DFlash proposal capacity must be between 1 and {maximum}"
        ))
    }

    fn missing_context() -> Self::Error {
        M::error("DFlash proposal has neither pending nor committed context".into())
    }

    fn block_exhausted() -> Self::Error {
        M::error("Muse-Glimmer DFlash block is exhausted".into())
    }

    fn state_mismatch() -> Self::Error {
        M::error("Muse-Glimmer DFlash verification/checkpoint state mismatch".into())
    }

    fn invalid_output(expected: usize, logits: usize, captures: &[usize]) -> Self::Error {
        M::error(format!(
            "Muse-Glimmer target output lengths disagree: expected={expected}, logits={logits}, captures={captures:?}"
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
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::sequence_row(
            value,
            row,
            false,
            ExternalAssistantTensorPlacement::Draft,
            context,
        )
        .map(M::into_logits)
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

    fn draft_to_target<'a>(
        value: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        M::transfer(value, ExternalAssistantTransfer::DraftToTarget, context)
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
        ids: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let ids = M::target_tokens(ids, context)?;
        let embeddings = M::target_operation(
            target,
            crate::composite_execution::ExternalPredictionTargetOperation::TokenEmbeddings(&ids),
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
        M::module(assistant)
            .proposal_states(
                embeddings,
                committed,
                absolute_end,
                M::neural_context(context, ExternalAssistantTensorPlacement::Draft),
            )
            .map_err(M::neural_error)
    }

    fn target_logits<'a>(
        target: &mut Self::Target,
        states: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let states = M::transfer(states, ExternalAssistantTransfer::DraftToTarget, context)?;
        M::target_operation(
            target,
            crate::composite_execution::ExternalPredictionTargetOperation::ProjectLogits(&states),
            context,
        )
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
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states } = capture else {
        return Err(M::error(
            "Muse-Glimmer target returned a different assistant capture".into(),
        ));
    };
    let paths = cache.capture_paths();
    if paths.len() != target_states.len() {
        return Err(M::error(format!(
            "Muse-Glimmer target capture path count {} differs from state count {}",
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
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture>,
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
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states } = capture else {
        return Err(M::error(
            "Muse-Glimmer target returned a different assistant capture".into(),
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

/// Private fused DFlash proposal block.
#[derive(Clone)]
pub struct ExternalDraftState<T: Clone> {
    logits: T,
    cursor: usize,
    proposal_capacity: usize,
    draft_context: DFlashContext<T>,
    cache_len: i32,
}

/// Retained target verification and exact input tokens.
pub struct ExternalVerification<T> {
    output: ExternalTargetOutput<T>,
    inputs: T,
}

/// Backend mechanisms required by the architecture-owned DFlash lifecycle.
pub trait ExternalMechanisms: 'static {
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
    /// Native sampling logits.
    type Logits;
    /// Selected target/draft assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact target completion.
    type Completion: Completion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Released assistant block capacity excluding the anchor.
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
    /// Selects one proposal-logits row on draft placement.
    fn proposal_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
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
    /// Moves or orders draft output for target execution.
    fn draft_to_target<'a>(
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
    /// Applies the target-owned embedding to the anchor-plus-mask block.
    fn target_embeddings<'a>(
        target: &mut Self::Target,
        ids: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Runs the neutral DFlash block.
    fn proposal_states<'a>(
        assistant: &mut Self::Assistant,
        embeddings: &Self::Tensor,
        committed: &DFlashContext<Self::Tensor>,
        absolute_end: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Applies the target-owned output projection.
    fn target_logits<'a>(
        target: &mut Self::Target,
        states: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Submits logits and every ordered captured target state.
    fn submit_verification<'a>(
        output: &ExternalTargetOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error>;
}

/// Architecture-owned Muse-Glimmer external DFlash executor.
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

fn block_token_ids(anchor: u32, mask: u32, proposal_capacity: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(proposal_capacity + 1);
    ids.push(anchor);
    ids.resize(proposal_capacity + 1, mask);
    ids
}

impl<M: ExternalMechanisms> SpeculativeExecutor for ExternalExecutor<'_, M> {
    type Input = M::Input;
    type Cache = M::Cache;
    type TargetState = ExternalTargetState<M::Tensor>;
    type DraftState = ExternalDraftState<M::Tensor>;
    type CacheCheckpoint = M::CacheCheckpoint;
    type Verification = ExternalVerification<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn max_proposals(&self) -> usize {
        M::max_proposals(self.assistant)
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
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

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: M::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        let maximum = self.max_proposals();
        if proposal_capacity == 0 || proposal_capacity > maximum {
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
        let ids = block_token_ids(last_token, M::mask_token(self.assistant), proposal_capacity);
        let embeddings = M::target_embeddings(self.target, &ids, context)?;
        let embeddings = M::target_to_draft(&embeddings, context)?;
        let states = M::proposal_states(
            self.assistant,
            &embeddings,
            &draft_context,
            state.cache_len,
            context,
        )?;
        let states = M::draft_to_target(&states, context)?;
        let logits = M::target_logits(self.target, &states, context)?;
        let logits = M::target_to_draft(&logits, context)?;
        Ok(ExternalDraftState {
            logits,
            cursor: 0,
            proposal_capacity,
            draft_context,
            cache_len: state.cache_len,
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        _last_token: u32,
        context: M::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        if state.cursor >= state.proposal_capacity {
            return Err(M::block_exhausted());
        }
        let row = state.cursor;
        state.cursor += 1;
        let logits = M::proposal_logits_row(&state.logits, row, context)?;
        M::observe_logits(
            self.assistant,
            EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
            logits,
        )
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        M::checkpoint(cache)
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        M::restore_checkpoint(cache, checkpoint, context)
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: M::Context<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
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
            output: ExternalVerification { output, inputs },
            completion,
        })
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::logits_row(&output.output.logits, index, context)
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: M::Context<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        let input_len = M::sequence_len(&output.inputs)?;
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
            (output.output, 0)
        } else {
            M::restore_checkpoint(cache, checkpoint, context)?;
            let inputs = M::token_prefix(&output.inputs, verified_inputs, context)?;
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use eredu_core::{Completion, SpeculativeExecutor};

    use super::{block_token_ids, ExternalExecutor, ExternalMechanisms, ExternalTargetOutput};
    use crate::muse_glimmer::DFlashContext;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Tensor {
        sequence: usize,
        retained: usize,
        value: u32,
    }

    impl Tensor {
        const fn new(sequence: usize, value: u32) -> Self {
            Self {
                sequence,
                retained: sequence,
                value,
            }
        }
    }

    #[derive(Clone, Default)]
    struct Cache {
        len: i32,
        restores: usize,
    }

    #[derive(Default)]
    struct Target {
        proposal_ids: Vec<u32>,
    }

    struct Assistant;
    struct Ready;

    impl Completion for Ready {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct Mechanisms;

    impl Mechanisms {
        fn output(sequence: usize) -> ExternalTargetOutput<Tensor> {
            ExternalTargetOutput {
                logits: Tensor::new(sequence, 10),
                target_states: vec![Tensor::new(sequence, 1), Tensor::new(sequence, 2)],
            }
        }
    }

    impl ExternalMechanisms for Mechanisms {
        type Target = Target;
        type Assistant = Assistant;
        type Capture = ();
        type Input = usize;
        type Cache = Cache;
        type CacheCheckpoint = Cache;
        type Tensor = Tensor;
        type Logits = u32;
        type Context<'a> = ();
        type Completion = Ready;
        type Telemetry = ();
        type Error = Infallible;

        fn max_proposals(_assistant: &Self::Assistant) -> usize {
            15
        }
        fn mask_token(_assistant: &Self::Assistant) -> u32 {
            99
        }
        fn sliding_window(_assistant: &Self::Assistant) -> i32 {
            3
        }
        fn empty_input() -> Self::Error {
            unreachable!()
        }
        fn invalid_capacity(_maximum: usize) -> Self::Error {
            unreachable!()
        }
        fn missing_context() -> Self::Error {
            unreachable!()
        }
        fn block_exhausted() -> Self::Error {
            unreachable!()
        }
        fn state_mismatch() -> Self::Error {
            unreachable!()
        }

        fn invalid_output(_expected: usize, _logits: usize, _captures: &[usize]) -> Self::Error {
            unreachable!()
        }

        fn bind_prepared_input(
            _input: &Self::Input,
            _cache: &mut Self::Cache,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn prefill_target(
            _target: &mut Self::Target,
            _assistant: &mut Self::Assistant,
            _request: &Self::Capture,
            input: Self::Input,
            cache: &mut Self::Cache,
            _context: Self::Context<'_>,
        ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
            cache.len = i32::try_from(input).unwrap();
            Ok(Self::output(input))
        }

        fn verify_target(
            _target: &mut Self::Target,
            _assistant: &mut Self::Assistant,
            _request: &Self::Capture,
            tokens: &Self::Tensor,
            cache: &mut Self::Cache,
            _context: Self::Context<'_>,
        ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
            cache.len += i32::try_from(tokens.sequence).unwrap();
            Ok(Self::output(tokens.sequence))
        }

        fn checkpoint(cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
            Ok(cache.clone())
        }
        fn restore_checkpoint(
            cache: &mut Self::Cache,
            checkpoint: &Self::CacheCheckpoint,
            _context: Self::Context<'_>,
        ) -> Result<(), Self::Error> {
            let restores = cache.restores + 1;
            cache.clone_from(checkpoint);
            cache.restores = restores;
            Ok(())
        }
        fn cache_len(cache: &Self::Cache) -> Result<i32, Self::Error> {
            Ok(cache.len)
        }
        fn checkpoint_len(checkpoint: &Self::CacheCheckpoint) -> Result<i32, Self::Error> {
            Ok(checkpoint.len)
        }
        fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
            Ok(value.sequence)
        }
        fn logits_row(
            value: &Self::Tensor,
            row: usize,
            _context: Self::Context<'_>,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(value.value + u32::try_from(row).unwrap())
        }
        fn proposal_logits_row(
            value: &Self::Tensor,
            row: usize,
            _context: Self::Context<'_>,
        ) -> Result<Self::Logits, Self::Error> {
            Ok(value.value + u32::try_from(row).unwrap())
        }
        fn sequence_suffix(
            value: &Self::Tensor,
            maximum: i32,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            let mut value = value.clone();
            value.retained = value.retained.min(usize::try_from(maximum).unwrap());
            Ok(value)
        }
        fn token_prefix(
            value: &Self::Tensor,
            end: usize,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(end, value.value))
        }
        fn target_tokens(
            tokens: &[u32],
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(tokens.len(), tokens[0]))
        }
        fn target_to_draft(
            value: &Self::Tensor,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(value.clone())
        }
        fn draft_to_target(
            value: &Self::Tensor,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(value.clone())
        }
        fn assemble_target_states(
            _assistant: &mut Self::Assistant,
            states: &[Self::Tensor],
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            assert_eq!(
                states.iter().map(|state| state.value).collect::<Vec<_>>(),
                [1, 2]
            );
            Ok(Tensor::new(states[0].sequence, 12))
        }
        fn update_context(
            _assistant: &mut Self::Assistant,
            _previous: Option<DFlashContext<Self::Tensor>>,
            pending: &Self::Tensor,
            absolute_end: i32,
            _context: Self::Context<'_>,
        ) -> Result<DFlashContext<Self::Tensor>, Self::Error> {
            Ok(DFlashContext {
                encoded: pending.clone(),
                layers: Vec::new(),
                start: absolute_end - i32::try_from(pending.retained).unwrap(),
                end: absolute_end,
            })
        }
        fn target_embeddings(
            target: &mut Self::Target,
            ids: &[u32],
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            target.proposal_ids = ids.to_vec();
            Ok(Tensor::new(ids.len(), 20))
        }
        fn proposal_states(
            _assistant: &mut Self::Assistant,
            embeddings: &Self::Tensor,
            _committed: &DFlashContext<Self::Tensor>,
            _absolute_end: i32,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(embeddings.sequence - 1, 30))
        }
        fn target_logits(
            _target: &mut Self::Target,
            states: &Self::Tensor,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(states.clone())
        }
        fn submit_verification(
            _output: &ExternalTargetOutput<Self::Tensor>,
            _inputs: &Self::Tensor,
            _context: Self::Context<'_>,
        ) -> Result<Self::Completion, Self::Error> {
            Ok(Ready)
        }
    }

    #[test]
    fn block_contains_one_anchor_and_only_requested_masks() {
        assert_eq!(block_token_ids(7, 99, 1), [7, 99]);
        assert_eq!(block_token_ids(7, 99, 3), [7, 99, 99, 99]);
        assert_eq!(block_token_ids(7, 99, 15).len(), 16);
    }

    #[test]
    fn lifecycle_preserves_ordered_sliding_context_and_zero_commit_rollback() {
        let mut target = Target::default();
        let mut assistant = Assistant;
        let mut cache = Cache::default();
        let mut executor = ExternalExecutor::<Mechanisms>::new(&mut target, &mut assistant, ());
        let (_, state, _) = executor.prefill(5, &mut cache, ()).unwrap().into_parts();
        assert_eq!(state.pending_context.as_ref().unwrap().retained, 3);
        let mut draft = executor.begin_proposal(&state, 7, 2, ()).unwrap();
        assert_eq!(draft.draft_context.start, 2);
        assert_eq!(draft.draft_context.end, 5);
        assert_eq!(executor.proposal_logits(&mut draft, 7, ()).unwrap(), 30);
        assert_eq!(executor.proposal_logits(&mut draft, 8, ()).unwrap(), 31);

        let checkpoint = executor.checkpoint(&cache).unwrap();
        let verification = executor
            .submit_verification(&[7, 8, 9], &mut cache, ())
            .unwrap();
        let (rolled_back, replayed) = executor
            .commit_verification(verification.output, draft, &mut cache, &checkpoint, 0, ())
            .unwrap()
            .into_parts();
        assert_eq!((cache.len, cache.restores, replayed), (5, 1, 0));
        assert!(rolled_back.pending_context.is_none());
        assert_eq!(rolled_back.draft_context.unwrap().end, 5);
        assert_eq!(target.proposal_ids, [7, 99, 99]);
    }
}
