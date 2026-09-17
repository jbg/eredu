//! Architecture-owned speculative lifecycle for a Muse-Glimmer DFlash assistant.

use std::marker::PhantomData;
mod prefill_spans;
mod state;
pub use state::SharedDFlashContext;
use crate::{external_assistant::ExternalOperationResult,speculative_execution::{EmbeddedPredictionTensor,PreparedEmbeddedEvidence}};

use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
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
    fn driver_buffer_bytes<T>(capacity:usize)->Option<usize>{M::state_buffer_bytes::<T>(capacity)}
    fn driver_buffer<T>(capacity:usize,context:Self::Context<'_>)->Result<eredu_core::SpeculativeBuffer<T>,Self::Error>{M::state_buffer(capacity,context)}
    fn driver_host_metadata(bytes:Option<usize>,context:Self::Context<'_>)->Result<eredu_core::HostPreparationAuthority,Self::Error>{M::state_host_metadata(bytes,context)}
    fn request_context<'a>(request:eredu_core::SpeculativeRequestId,context:Self::Context<'a>)->Result<Self::Context<'a>,Self::Error>{M::request_context(request,context)}
    fn coordinate_speculative_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> { M::coordinate_speculative_buffer(local, context) }

    fn driver_identity(context:Self::Context<'_>)->Result<eredu_core::SpeculativeRequestIdentity,Self::Error>{M::driver_identity(context)}
    fn copy_sequence(source:eredu_core::SpeculativeSequenceRef<'_>,context:Self::Context<'_>)->Result<eredu_core::SpeculativeSequence,eredu_core::SpeculativeDriverError<Self::Error>>{M::copy_sequence(source,context)}
    fn sequence_copy_bytes(source:&eredu_core::SpeculativeSequence)->Option<u64>{M::sequence_copy_bytes(source)}
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error>{M::take_retained_failure(error)}

    fn requires_activation_origin() -> bool { M::requires_activation_origin() }
    fn invocation_context<'a>(context: Self::Context<'a>,
        origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) -> Result<Self::Context<'a>, Self::Error> { M::invocation_context(context, origin) }
    fn prepare_control_continuation<'a>(committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus, context: Self::Context<'a>,
    ) -> Result<(), Self::Error> { M::prepare_control_continuation(committed, status, context) }


    fn sequence_len_with_context(value:&Self::Tensor,context:Self::Context<'_>)->Result<usize,Self::Error>{M::state_dimension(value,1,context)}
    fn invalid_output_with_context(_expected:usize,_logits:usize,_captures:&[usize],context:Self::Context<'_>)->Self::Error{M::state_refusal(context)}
    fn control_copy_state(state:&ExternalTargetState<Self::Tensor>,context:Self::Context<'_>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{state::control_copy::<M>(state,context)}
    fn transfer_packet(value:&EmbeddedPredictionTensor<Self::Tensor>,direction:ExternalAssistantTransfer,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{M::transfer_with_source(value,direction,context)}
    fn transfer_value(value:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,direction:ExternalAssistantTransfer,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{M::transfer_tensor_with_source(value,evidence,direction,context)}
    fn assemble_with_source(assistant:&mut Self::Assistant,states:&[Self::Tensor],evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{state::assemble::<M>(assistant,states,evidence,context)}
    fn suffix_with_source(value:&Self::Tensor,maximum:i32,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{
        let length=M::state_dimension(value,1,context)?;let maximum=usize::try_from(maximum).map_err(|_|M::state_refusal(context))?;
        M::tensor_range_with_source(value,1,length.saturating_sub(maximum),length,evidence,ExternalAssistantTensorPlacement::Target,context)
    }
    fn update_with_source(assistant:&mut Self::Assistant,previous:Option<&SharedDFlashContext<Self::Tensor>>,pending:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,absolute_end:i32,context:Self::Context<'_>)->Result<SharedDFlashContext<Self::Tensor>,Self::Error>{state::update::<M>(assistant,previous,pending,evidence,absolute_end,context)}
    fn embeddings_with_source(target:&mut Self::Target,ids:&[u32],context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{state::embeddings::<M>(target,ids,context)}
    fn proposal_with_source(assistant:&mut Self::Assistant,embeddings:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,committed:&SharedDFlashContext<Self::Tensor>,absolute_end:i32,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{state::proposal::<M>(assistant,embeddings,evidence,committed,absolute_end,context)}
    fn target_logits_with_source(target:&mut Self::Target,states:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{state::logits::<M>(target,states,evidence,context)}
    fn proposal_row_with_source(value:&Self::Tensor,row:usize,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<Self::Logits,Self::Error>{M::logits_row_with_source(value,row,evidence,ExternalAssistantTensorPlacement::Draft,context)}
    fn tokens_with_source(ids:&[u32],context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{M::target_tokens_with_source(ids,context)}
    fn prefix_with_source(value:&EmbeddedPredictionTensor<Self::Tensor>,end:usize,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{M::token_prefix_with_source(value,end,context)}
    fn verify_with_tokens(target:&mut Self::Target,assistant:&mut Self::Assistant,request:&Self::Capture,tokens:&EmbeddedPredictionTensor<Self::Tensor>,cache:&mut Self::Cache,context:Self::Context<'_>)->Result<ExternalTargetOutput<Self::Tensor>,Self::Error>{
        let evidence=tokens.evidence();let source=M::source_context(context,evidence.as_slice())?;
        Self::verify_target(target,assistant,request,tokens,cache,source)
    }
    fn complete_with_tokens(output:&ExternalTargetOutput<Self::Tensor>,inputs:&EmbeddedPredictionTensor<Self::Tensor>,context:Self::Context<'_>)->Result<Self::Completion,Self::Error>{
        let mut proofs=M::state_buffer(2,context)?;
        if let Some(proof)=&output.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        if let Some(proof)=inputs.evidence(){proofs.try_push(proof).map_err(|_|M::state_refusal(context))?;}
        M::submit_completion_with_sources(std::iter::once(&output.logits).chain(output.target_states.iter()).chain(std::iter::once(&**inputs)),&proofs,context)
    }
    fn observe_output_in_context(assistant:&mut Self::Assistant,_request:&Self::Capture,output:&mut ExternalTargetOutput<Self::Tensor>,path:&str,context:Self::Context<'_>)->Result<(),Self::Error>{
        output.logits=M::observe_borrowed_tensor(assistant,path,&output.logits,output.evidence.as_ref(),context)?;Ok(())
    }

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

    fn bind_prepared_input_with_context(input:&Self::Input,cache:&mut Self::Cache,context:Self::Context<'_>)->Result<(),Self::Error>{
        M::bind_prepared_input_with_context(input,cache,context)
    }
    fn checkpoint_with_context(cache:&Self::Cache,context:Self::Context<'_>)->Result<Self::CacheCheckpoint,Self::Error>{
        let host=M::state_host_metadata(Some(std::mem::size_of::<(Self::CacheCheckpoint,Result<Self::CacheCheckpoint,Self::Error>)>()),context)?;
        M::checkpoint_native_with_context(cache.native(),context).map(|native|cache.checkpoint_with_host(native,host))
    }
    fn validate_output_capture_with_context(cache:&Self::Cache,
        output:&ExternalTargetOutput<Self::Tensor>,context:Self::Context<'_>)->Result<(),Self::Error>{
        validate_output_capture_in_context::<M>(cache,output,context)
    }

    fn bind_prepared_input(
        input: &Self::Input,
        cache: &mut Self::Cache,
    ) -> Result<(), Self::Error> {
        cache
            .bind_prepared_input_cache_identity(&M::prepared_input_cache_identity(input)?)
            .map_err(M::error)
    }

    fn uses_span_prefill(_input: &Self::Input) -> bool {
        // Every ordinary and controlled external prefill shares demand selection
        // and exact completion. An absent override selects runtime span policy.
        true
    }
    fn prefill_spans<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        input: Self::Input,
        cache: &mut Self::Cache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            SpeculativePrefill<ExternalTargetState<Self::Tensor>, Self::Logits>,
        >,
        Self::Error,
    > {
        prefill_spans::run::<M>(
            target,
            assistant,
            request,
            input,
            cache,
            cancellation,
            context,
        )
    }

    fn prefill_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
        let crate::external_assistant::ExternalTargetResult { logits, capture: mut capture, evidence } =
            M::prefill_target_with_evidence(target, request, input, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant, cache, &mut capture,evidence.as_ref(),context)?;
        target_output::<M>(logits, capture, evidence)
    }

    fn verify_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        tokens: &Self::Tensor,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
        let crate::external_assistant::ExternalTargetResult { logits, capture: mut capture, evidence } =
            M::verify_target_with_evidence(target, request, tokens, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant, cache, &mut capture,evidence.as_ref(),context)?;
        target_output::<M>(logits, capture, evidence)
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
        let host=M::state_host_metadata(Some(std::mem::size_of::<(Self::CacheCheckpoint,Result<Self::CacheCheckpoint,Self::Error>)>()),context)?;
        M::control_checkpoint(cache.native(), context).map(|native| cache.checkpoint_with_host(native,host))
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

    fn logits_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        evidence: Option<&crate::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::logits_row_with_source(value, row, evidence, ExternalAssistantTensorPlacement::Target, context)
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
    evidence:Option<&PreparedEmbeddedEvidence>,context:M::Context<'_>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::MuseGlimmerDFlash { target_states } = capture else {
        return Err(M::error(
            "Muse-Glimmer target returned a different assistant capture".into(),
        ));
    };
    let paths = cache.capture_paths_iter();
    if paths.len() != target_states.len() {
        return Err(M::error(format!(
            "Muse-Glimmer target capture path count {} differs from state count {}",
            paths.len(),
            target_states.len()
        )));
    }
    for (path, state) in paths.into_iter().zip(target_states) {
        *state = M::observe_borrowed_tensor(assistant,path,state,evidence,context)?;
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

fn validate_output_capture_in_context<M>(
    cache: &ExternalAssistantCache<M::NativeCache>,
    output: &ExternalTargetOutput<M::Tensor>,
    context:M::Context<'_>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<MuseGlimmerAssistantArchitecture>,
{
    cache.validate_capture_values::<MuseGlimmerAssistantArchitecture,M,_>(
        ||output.target_states.iter(),context)

}

fn target_output<M>(
    logits: M::Tensor,
    capture: ExternalPredictionTargetCapture<M::Tensor>,
    evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
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
        evidence,
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
    /// Actual completed source retained after this output's native values.
    pub evidence:Option<PreparedEmbeddedEvidence>,
}

/// Committed target and DFlash context state.
#[derive(Clone)]
pub struct ExternalTargetState<T: Clone> {
    pending_context: Option<T>,
    draft_context: Option<SharedDFlashContext<T>>,
    cache_len: i32,
    evidence:Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
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
                }.into())
            })
            .transpose()?;
        Ok(Self {
            pending_context,
            draft_context,
            cache_len: self.cache_len,
            evidence:self.evidence.clone(),
        })
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

/// Private fused DFlash proposal block.
#[derive(Clone)]
pub struct ExternalDraftState<T: Clone> {
    logits: EmbeddedPredictionTensor<T>,
    cursor: usize,
    proposal_capacity: usize,
    draft_context: SharedDFlashContext<T>,
    cache_len: i32,
}

/// Retained target verification and exact input tokens.
pub struct ExternalVerification<T> {
    output: ExternalTargetOutput<T>,
    inputs: EmbeddedPredictionTensor<T>,
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
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: eredu_core::SpeculativeTelemetry;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fixed descriptor query under the current source's metadata destination.
    fn sequence_len_with_context(value:&Self::Tensor,context:Self::Context<'_>)->Result<usize,Self::Error>{let _=context;Self::sequence_len(value)}
    /// Source-retained refusal for a mismatch in this invocation's output.
    fn invalid_output_with_context(expected:usize,logits:usize,captures:&[usize],context:Self::Context<'_>)->Self::Error{let _=context;Self::invalid_output(expected,logits,captures)}
    /// Copies the exact complete seed through the existing admitted tensor copier.
    fn control_copy_state(state:&ExternalTargetState<Self::Tensor>,context:Self::Context<'_>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{state.map(|value,placement|Self::control_copy_tensor(value,placement,context))}
    /// Transfers the exact source packet through the existing placement worker.
    fn transfer_packet(value:&EmbeddedPredictionTensor<Self::Tensor>,direction:ExternalAssistantTransfer,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        match direction{ExternalAssistantTransfer::TargetToDraft=>Self::target_to_draft(value,context),ExternalAssistantTransfer::DraftToTarget=>Self::draft_to_target(value,context)}.map(EmbeddedPredictionTensor::ordinary)
    }
    /// Transfers an immutable value while preserving its completed source.
    fn transfer_value(value:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,direction:ExternalAssistantTransfer,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{
        let _=evidence;match direction{ExternalAssistantTransfer::TargetToDraft=>Self::target_to_draft(value,context),ExternalAssistantTransfer::DraftToTarget=>Self::draft_to_target(value,context)}.map(ExternalOperationResult::ordinary)
    }
    /// Assembles the same target context with its actual receipt.
    fn assemble_with_source(assistant:&mut Self::Assistant,states:&[Self::Tensor],evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{let _=evidence;Self::assemble_target_states(assistant,states,context).map(ExternalOperationResult::ordinary)}
    /// Selects the same rolling suffix with its actual receipt.
    fn suffix_with_source(value:&Self::Tensor,maximum:i32,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{let _=evidence;Self::sequence_suffix(value,maximum,context).map(ExternalOperationResult::ordinary)}
    /// Encodes into a source-preserving immutable committed context.
    fn update_with_source(assistant:&mut Self::Assistant,previous:Option<&SharedDFlashContext<Self::Tensor>>,pending:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,absolute_end:i32,context:Self::Context<'_>)->Result<SharedDFlashContext<Self::Tensor>,Self::Error>{let _=evidence;Self::update_context(assistant,previous.map(|value|(**value).clone()),pending,absolute_end,context).map(Into::into)}
    /// Target embeddings retain their exact input/output evidence.
    fn embeddings_with_source(target:&mut Self::Target,ids:&[u32],context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::target_embeddings(target,ids,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Runs the same fused proposal with both exact source inventories.
    fn proposal_with_source(assistant:&mut Self::Assistant,embeddings:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,committed:&SharedDFlashContext<Self::Tensor>,absolute_end:i32,context:Self::Context<'_>)->Result<ExternalOperationResult<Self::Tensor>,Self::Error>{let _=evidence;Self::proposal_states(assistant,embeddings,committed,absolute_end,context).map(ExternalOperationResult::ordinary)}
    /// Projects target vocabulary scores with the completed state source.
    fn target_logits_with_source(target:&mut Self::Target,states:&Self::Tensor,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{let _=evidence;Self::target_logits(target,states,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Reads one completed fused score row on draft placement.
    fn proposal_row_with_source(value:&Self::Tensor,row:usize,evidence:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'_>)->Result<Self::Logits,Self::Error>{let _=evidence;Self::proposal_logits_row(value,row,context)}
    /// Constructs the existing target token packet.
    fn tokens_with_source(ids:&[u32],context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::target_tokens(ids,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Selects a replay prefix with the existing token source.
    fn prefix_with_source(value:&EmbeddedPredictionTensor<Self::Tensor>,end:usize,context:Self::Context<'_>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::token_prefix(value,end,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Verification consumes the actual token packet source.
    fn verify_with_tokens(target:&mut Self::Target,assistant:&mut Self::Assistant,request:&Self::Capture,tokens:&EmbeddedPredictionTensor<Self::Tensor>,cache:&mut Self::Cache,context:Self::Context<'_>)->Result<ExternalTargetOutput<Self::Tensor>,Self::Error>{Self::verify_target(target,assistant,request,tokens,cache,context)}
    /// Completion keeps the exact target and token receipts.
    fn complete_with_tokens(output:&ExternalTargetOutput<Self::Tensor>,inputs:&EmbeddedPredictionTensor<Self::Tensor>,context:Self::Context<'_>)->Result<Self::Completion,Self::Error>{Self::submit_verification(output,inputs,context)}
    /// Observes the same output through its source-aware borrowed value hook.
    fn observe_output_in_context(assistant:&mut Self::Assistant,request:&Self::Capture,output:&mut ExternalTargetOutput<Self::Tensor>,path:&str,context:Self::Context<'_>)->Result<(),Self::Error>{let _=context;Self::observe_output(assistant,request,output,path)}

    /// Whether the actual mechanism needs the shared scheduler coordinate.
    /// Exact query for the matching actual driver buffer constructor.
    fn driver_buffer_bytes<T>(capacity:usize)->Option<usize>{eredu_core::SpeculativeBuffer::<T>::retained_control_bytes(capacity)}
    /// Actual host buffer for the common driver, paid before construction.
    fn driver_buffer<T>(capacity:usize,context:Self::Context<'_>)->Result<eredu_core::SpeculativeBuffer<T>,Self::Error>{let _=context;Ok(eredu_core::SpeculativeBuffer::with_capacity(capacity))}
    /// Exact concrete payload and erasure metadata source.
    fn driver_host_metadata(bytes:Option<usize>,context:Self::Context<'_>)->Result<eredu_core::HostPreparationAuthority,Self::Error>{let _=(bytes,context);Ok(eredu_core::HostPreparationAuthority::unmanaged())}
    /// Copies only the actual request assignment; no new source is inferred.
    fn request_context<'a>(request:eredu_core::SpeculativeRequestId,context:Self::Context<'a>)->Result<Self::Context<'a>,Self::Error>{let _=request;Ok(context)}
    /// Coordinates the same host rows without discarding their original owner.
    /// Legacy mechanisms accept ordinary buffers; a retained buffer needs an
    /// explicit source-aware implementation.
    fn coordinate_speculative_buffer(
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        let _ = context;
        match local.try_into_ordinary() {
            Ok(local) => Ok(local.into()),
            Err(_) => Err(eredu_core::HostMetadataFundingError::Unavailable.into()),
        }
    }

    /// Retains the shared driver's original request identity.
    fn driver_identity(context:Self::Context<'_>)->Result<eredu_core::SpeculativeRequestIdentity,Self::Error>{let _=context;Ok(eredu_core::SpeculativeRequestIdentity::new())}
    /// Uses the existing canonical sequence-copy mechanism at checkpoint boundaries.
    fn copy_sequence(source:eredu_core::SpeculativeSequenceRef<'_>,context:Self::Context<'_>)->Result<eredu_core::SpeculativeSequence,eredu_core::SpeculativeDriverError<Self::Error>>{let _=context;source.copy_ordinary().map_err(eredu_core::SpeculativeDriverError::Preparation)}
    /// Exact storage query for the matching sequence-copy provider.
    fn sequence_copy_bytes(source:&eredu_core::SpeculativeSequence)->Option<u64>{match source{eredu_core::SpeculativeSequence::Ordinary(_)=>source.snapshot_storage_bytes(),eredu_core::SpeculativeSequence::Retained(_)=>None}}
    /// Transfers an already retained failure without another allocation.
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error>{Err(error)}
    /// Whether the actual native source needs a scheduler coordinate.
    fn requires_activation_origin() -> bool { false }
    /// Lends the actual coordinate to one operation; ordinary contexts are unchanged.
    fn invocation_context<'a>(context: Self::Context<'a>,
        _origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) -> Result<Self::Context<'a>, Self::Error> { Ok(context) }
    /// The request owns monotonic attempts outside all cache snapshots.
    fn prepare_control_continuation<'a>(_committed: usize,
        _status: eredu_core::generation::SpeculativeRequestStatus, _context: Self::Context<'a>,
    ) -> Result<(), Self::Error> { Ok(()) }

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

    /// Uses the same validation with the active source/account loan.
    fn validate_output_capture_with_context(cache:&Self::Cache,
        output:&ExternalTargetOutput<Self::Tensor>,_context:Self::Context<'_>)->Result<(),Self::Error>{
        Self::validate_output_capture(cache,output)
    }

    /// Binds the exact prepared-input identity before target execution.
    /// Source-aware prepared identity; ordinary implementations keep their worker.
    fn bind_prepared_input_with_context(input:&Self::Input,cache:&mut Self::Cache,_context:Self::Context<'_>)->Result<(),Self::Error>{Self::bind_prepared_input(input,cache)}
    /// Source-aware rollback checkpoint under the same shared transaction.
    fn checkpoint_with_context(cache:&Self::Cache,_context:Self::Context<'_>)->Result<Self::CacheCheckpoint,Self::Error>{Self::checkpoint(cache)}
    fn bind_prepared_input(input: &Self::Input, cache: &mut Self::Cache)
        -> Result<(), Self::Error>;

    /// Selects only the already requested shared span mechanism.
    fn uses_span_prefill(_input: &Self::Input) -> bool {
        false
    }
    /// Compatibility mechanisms keep their existing whole-input method.
    fn prefill_spans<'a>(
        _target: &mut Self::Target,
        _assistant: &mut Self::Assistant,
        _request: &Self::Capture,
        _input: Self::Input,
        _cache: &mut Self::Cache,
        _cancellation: &eredu_core::GenerationCancellationToken,
        _context: Self::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            SpeculativePrefill<ExternalTargetState<Self::Tensor>, Self::Logits>,
        >,
        Self::Error,
    > {
        Err(Self::empty_input())
    }

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
    /// Reads a row from this exact completed target output when source evidence is present.
    fn logits_row_with_source<'a>(
        value: &Self::Tensor,
        row: usize,
        evidence: Option<&crate::speculative_execution::PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let _ = evidence;
        Self::logits_row(value, row, context)
    }

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
    capture: crate::external_assistant::CaptureSource<'a,M::Capture>,
    origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    _mechanisms: PhantomData<fn() -> M>,
}

impl<'a, M: ExternalMechanisms> ExternalExecutor<'a, M> {
    /// Lends the exact immutable capture declaration owned by the selected source.
    pub const fn new_borrowed(target:&'a mut M::Target,assistant:&'a mut M::Assistant,capture:&'a M::Capture)->Self{
        Self{target,assistant,capture:crate::external_assistant::CaptureSource::Borrowed(capture),origin:None,_mechanisms:PhantomData}
    }

    /// Binds materialized target and assistant objects to the neutral lifecycle.
    pub const fn new(
        target: &'a mut M::Target,
        assistant: &'a mut M::Assistant,
        capture: M::Capture,
    ) -> Self {
        Self {
            target,
            assistant,
            capture:crate::external_assistant::CaptureSource::Owned(capture),
            origin: None,
            _mechanisms: PhantomData,
        }
    }

    fn target_state<'context>(
        &mut self,
        output: &ExternalTargetOutput<M::Tensor>,
        draft_context: Option<SharedDFlashContext<M::Tensor>>,
        cache_len: i32,
        context: M::Context<'context>,
    ) -> Result<ExternalTargetState<M::Tensor>, M::Error> {
        let assembled = M::assemble_with_source(self.assistant, &output.target_states,output.evidence.as_ref(), context)?;
        let pending = M::suffix_with_source(&assembled.output, M::sliding_window(self.assistant),assembled.evidence.as_ref(), context)?;
        Ok(ExternalTargetState {
            evidence:pending.evidence,
            pending_context: Some(pending.output),
            draft_context,
            cache_len,
        })
    }

    fn validate_output(output:&ExternalTargetOutput<M::Tensor>,expected:usize,context:M::Context<'_>)->Result<(),M::Error>{
        let controls=[std::mem::size_of::<Result<(),M::Error>>(),std::mem::size_of::<eredu_core::SpeculativeBuffer<usize>>(),
            std::mem::size_of::<std::slice::Iter<'_,M::Tensor>>(),std::mem::size_of::<(usize,usize)>()];
        let _host=M::driver_host_metadata(controls.into_iter().try_fold(std::mem::size_of_val(&controls),usize::checked_add),context)?;
        let logits=M::sequence_len_with_context(&output.logits,context)?;
        let mut captures=M::driver_buffer(output.target_states.len(),context)?;
        for value in &output.target_states{captures.try_push(M::sequence_len_with_context(value,context)?).map_err(|_|M::invalid_output_with_context(expected,logits,&captures,context))?;}
        if logits!=expected||captures.iter().any(|count|*count!=expected){return Err(M::invalid_output_with_context(expected,logits,&captures,context));}
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

    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a,
    {
        let _host=M::driver_host_metadata(Some(std::mem::size_of::<(Self::DraftState,Result<Self::DraftState,Self::Error>)>()),context)?;
        Ok(state.clone())
    }
    type CacheCheckpoint = M::CacheCheckpoint;
    type Verification = ExternalVerification<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;
    fn driver_buffer_bytes<T>(&self,capacity:usize)->Option<usize>{M::driver_buffer_bytes::<T>(capacity)}
    fn driver_buffer<T>(&self,capacity:usize,context:Self::Context<'_>)->Result<eredu_core::SpeculativeBuffer<T>,Self::Error>{M::driver_buffer(capacity,context)}
    fn driver_host_metadata(&self,bytes:Option<usize>,context:Self::Context<'_>)->Result<eredu_core::HostPreparationAuthority,Self::Error>{M::driver_host_metadata(bytes,context)}
    fn request_context<'a>(&self,request:eredu_core::SpeculativeRequestId,context:Self::Context<'a>)->Result<Self::Context<'a>,Self::Error> where Self:'a {M::request_context(request,context)}
    fn coordinate_speculative_buffer(
        &mut self, local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> { M::coordinate_speculative_buffer(local, context) }

    fn driver_identity(&self,context:Self::Context<'_>)->Result<eredu_core::SpeculativeRequestIdentity,Self::Error>{M::driver_identity(context)}
    fn copy_sequence(&self,source:eredu_core::SpeculativeSequenceRef<'_>,context:Self::Context<'_>)->Result<eredu_core::SpeculativeSequence,eredu_core::SpeculativeDriverError<Self::Error>>{M::copy_sequence(source,context)}
    fn sequence_copy_bytes(&self,source:&eredu_core::SpeculativeSequence)->Option<u64>{M::sequence_copy_bytes(source)}
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error>{M::take_retained_failure(error)}


    fn requires_activation_origin(&self) -> bool { M::requires_activation_origin() }
    fn set_activation_origin(&mut self, origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>) {
        self.origin = origin;
    }
    fn prepare_control_continuation<'a>(&mut self, committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus, context: Self::Context<'a>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        M::prepare_control_continuation(committed, status, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)
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
    > {
        if self.control_snapshot_estimate(cache, state).is_none() {
            return Ok(None);
        }
        let state = M::control_copy_state(state,context)
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
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError> {
        let state = M::control_copy_state(state,context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        M::control_restore(cache, saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(state))
    }

    fn max_proposals(&self) -> usize {
        M::max_proposals(self.assistant)
    }

    fn prefill_cancellable<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'context>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<SpeculativePrefill<Self::TargetState, Self::Logits>>,
        Self::Error,
    > {
        let context = M::invocation_context(context, self.origin)?;
        let context = M::invocation_context(context, self.origin)?;
        if !M::uses_span_prefill(&input) {
            return self
                .prefill(input, cache, context)
                .map(eredu_core::SpeculativePrefillOutcome::Complete);
        }
        M::bind_prepared_input_with_context(&input, cache, context)?;
        let checkpoint = M::checkpoint_with_context(cache, context)?;
        match M::prefill_spans(
            self.target,
            self.assistant,
            &self.capture,
            input,
            cache,
            cancellation,
            context,
        ) {
            Ok(output) => Ok(output),
            Err(error) => {
                M::restore_checkpoint(cache, &checkpoint, context)?;
                Err(error)
            }
        }
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
        if M::uses_span_prefill(&input) {
            return match self.prefill_cancellable(
                input,
                cache,
                &eredu_core::GenerationCancellationToken::new(),
                context,
            )? {
                eredu_core::SpeculativePrefillOutcome::Complete(output) => Ok(output),
                eredu_core::SpeculativePrefillOutcome::Cancelled { .. } => Err(M::empty_input()),
            };
        }
        M::bind_prepared_input_with_context(&input, cache, context)?;
        let checkpoint = M::checkpoint_with_context(cache, context)?;
        let result = (|| {
            let mut output = M::prefill_target(
                self.target,
                self.assistant,
                &self.capture,
                input,
                cache,
                context,
            )?;
            M::observe_output_in_context(
                self.assistant,
                &self.capture,
                &mut output,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,context,
            )?;
            M::validate_output_capture_with_context(cache, &output, context)?;
            let sequence = M::sequence_len_with_context(&output.logits,context)?;
            if sequence == 0 {
                return Err(M::empty_input());
            }
            Self::validate_output(&output, sequence,context)?;
            Ok(SpeculativePrefill::new(
                M::logits_row_with_source(&output.logits, sequence - 1, output.evidence.as_ref(), context)?,
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
        let context = M::invocation_context(context, self.origin)?;
        let maximum = self.max_proposals();
        if proposal_capacity == 0 || proposal_capacity > maximum {
            return Err(M::invalid_capacity(maximum));
        }
        let pending = state
            .pending_context
            .as_ref()
            .map(|pending| M::transfer_value(pending,state.evidence.as_ref(),ExternalAssistantTransfer::TargetToDraft,context))
            .transpose()?;
        let draft_context = match pending {
            Some(pending) => M::update_with_source(
                self.assistant,
                state.draft_context.as_ref(),
                &pending.output,pending.evidence.as_ref(),
                state.cache_len,
                context,
            )?,
            None => state.draft_context.clone().ok_or_else(M::missing_context)?,
        };
        let count=proposal_capacity.checked_add(1).ok_or_else(M::state_mismatch)?;
        let mut ids=M::driver_buffer(count,context)?;
        ids.try_push(last_token).map_err(|_|M::state_mismatch())?;
        for _ in 0..proposal_capacity{ids.try_push(M::mask_token(self.assistant)).map_err(|_|M::state_mismatch())?;}
        let embeddings = M::embeddings_with_source(self.target, &ids, context)?;
        let embeddings = M::transfer_packet(&embeddings,ExternalAssistantTransfer::TargetToDraft, context)?;
        let states = M::proposal_with_source(
            self.assistant,
            &embeddings,embeddings.evidence(),
            &draft_context,
            state.cache_len,
            context,
        )?;
        let states = M::transfer_value(&states.output,states.evidence.as_ref(),ExternalAssistantTransfer::DraftToTarget,context)?;
        let logits = M::target_logits_with_source(self.target, &states.output,states.evidence.as_ref(), context)?;
        let logits = M::transfer_packet(&logits,ExternalAssistantTransfer::TargetToDraft, context)?;
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
        let context = M::invocation_context(context, self.origin)?;
        if state.cursor >= state.proposal_capacity {
            return Err(M::block_exhausted());
        }
        let row = state.cursor;
        state.cursor += 1;
        let logits = M::proposal_row_with_source(&state.logits, row,state.logits.evidence(), context)?;
        M::observe_logits(
            self.assistant,
            EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
            logits,
        )
    }

    fn checkpoint_with_context<'c>(&self,cache:&Self::Cache,context:Self::Context<'c>)->Result<Self::CacheCheckpoint,Self::Error>{M::checkpoint_with_context(cache,context)}
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
        let context = M::invocation_context(context, self.origin)?;
        let inputs = M::tokens_with_source(input_tokens, context)?;
        let mut output = M::verify_with_tokens(
            self.target,
            self.assistant,
            &self.capture,
            &inputs,
            cache,
            context,
        )?;
        M::observe_output_in_context(
            self.assistant,
            &self.capture,
            &mut output,
            EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,context,
        )?;
        M::validate_output_capture_with_context(cache, &output, context)?;
        Self::validate_output(&output, input_tokens.len(),context)?;
        let completion = M::complete_with_tokens(&output, &inputs, context)?;
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
        M::logits_row_with_source(&output.output.logits, index, output.output.evidence.as_ref(), context)
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
        let context = M::invocation_context(context, self.origin)?;
        let input_len = M::sequence_len_with_context(&output.inputs,context)?;
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
                    evidence:None,
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
            let inputs = M::prefix_with_source(&output.inputs, verified_inputs, context)?;
            let mut replayed = M::verify_with_tokens(
                self.target,
                self.assistant,
                &self.capture,
                &inputs,
                cache,
                context,
            )?;
            M::observe_output_in_context(
                self.assistant,
                &self.capture,
                &mut replayed,
                EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,context,
            )?;
            M::validate_output_capture_with_context(cache, &replayed, context)?;
            Self::validate_output(&replayed, verified_inputs,context)?;
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
    #[test]
    fn snapshot_seed_preserves_target_and_draft_tensor_placement() {
        use crate::external_assistant::ExternalAssistantTensorPlacement as Placement;
        let state = super::ExternalTargetState {
            evidence:None,
            pending_context: Some(1u32),
            draft_context: Some(super::DFlashContext {
                encoded: 2,
                layers: vec![crate::muse_glimmer::DFlashLayerContext { keys: 3, values: 4 }],
                start: 5,
                end: 6,
            }.into()),
            cache_len: 6,
        };
        let copied = state
            .map(|value, placement| {
                Ok::<_, std::convert::Infallible>(
                    value
                        + match placement {
                            Placement::Target => 100,
                            Placement::Draft => 200,
                        },
                )
            })
            .unwrap();
        assert_eq!(copied.pending_context, Some(101));
        let context = copied.draft_context.unwrap();
        assert_eq!(
            (
                context.encoded,
                context.layers[0].keys,
                context.layers[0].values
            ),
            (202, 203, 204)
        );
        assert_eq!((context.start, context.end, copied.cache_len), (5, 6, 6));
        assert_eq!(state.pending_context, Some(1));
    }

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

    impl eredu_core::BoundedCompletion for Ready {
        fn wait_bounded(
            self,
            _policy: eredu_core::BoundedCompletionWait,
        ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
            Ok(eredu_core::BoundedCompletionOutcome::Completed)
        }
    }

    struct Mechanisms;

    impl Mechanisms {
        fn output(sequence: usize) -> ExternalTargetOutput<Tensor> {
            ExternalTargetOutput {
                evidence: None,
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
