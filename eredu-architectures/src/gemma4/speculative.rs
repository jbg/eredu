//! Architecture-owned speculative lifecycle for a Gemma 4 external assistant.
use crate::speculative_execution::EmbeddedPredictionTensor;

use std::marker::PhantomData;
mod prefill_spans;
mod state;
use super::assistant::SharedAssistantStates;

use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

use super::AssistantState;
use crate::{
    composite_execution::ExternalPredictionTargetCapture,
    external_assistant::{
        EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
        EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH, ExternalAssistantCache,
        ExternalAssistantCacheCheckpoint, ExternalAssistantExecutionMechanisms,
        ExternalAssistantTensorPlacement, ExternalAssistantTransfer, Gemma4AssistantArchitecture,
    },
};

/// Architecture adapter from family-neutral backend mechanisms to Gemma lifecycle mechanisms.
pub struct ArchitectureExternalMechanisms<M>(PhantomData<fn() -> M>);

impl<M> ExternalMechanisms for ArchitectureExternalMechanisms<M>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture> + 'static,
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
        let mut result = M::prefill_target_with_evidence(target, request, input, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant,cache,&mut result.capture,result.evidence.as_ref(),context)?;
        target_output::<M>(result.logits,result.capture,result.evidence,context)
    }

    fn verify_target<'a>(
        target: &mut Self::Target,
        assistant: &mut Self::Assistant,
        request: &Self::Capture,
        tokens: &Self::Tensor,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<ExternalTargetOutput<Self::Tensor>, Self::Error> {
        let mut result = M::verify_target_with_evidence(target, request, tokens, cache.native_mut(), context)?;
        cache
            .advance_frontier(M::native_cache_len(cache.native())?)
            .map_err(M::error)?;
        observe_capture::<M>(assistant,cache,&mut result.capture,result.evidence.as_ref(),context)?;
        target_output::<M>(result.logits,result.capture,result.evidence,context)
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
    ) -> Result<EmbeddedPredictionTensor<Self::Tensor>, Self::Error> {
        let ids = M::target_tokens_with_source(&[token], context)?;
        let source = ids.evidence();
        let scoped = M::source_context(context, source.as_slice())?;
        let embedding = M::target_operation_with_source(
            target,
            crate::composite_execution::ExternalPredictionTargetOperation::TokenEmbeddings(&ids),
            scoped,
        )?;
        M::transfer_with_source(
            &embedding,
            ExternalAssistantTransfer::TargetToDraft,
            context,
        )
    }

    fn draft_step<'a>(
        assistant: &mut Self::Assistant,
        embedding: &EmbeddedPredictionTensor<Self::Tensor>,
        state: &mut AssistantState<Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let source = embedding.evidence();
        let context = M::source_context(context, source.as_slice())?;
        let logits = M::assistant_operation_with_evidence::<super::assistant::invocation::DraftStep>(
            assistant,
            super::assistant::invocation::DraftStepArguments { embedding, state },
            context,
        )
        ?;
        let logits = M::into_logits_with_source_at(logits.output, logits.evidence.as_ref(),
            ExternalAssistantTensorPlacement::Draft, context)?;
        M::observe_logits(
            assistant,
            EXTERNAL_ASSISTANT_PROPOSAL_LOGITS_OBSERVATION_PATH,
            logits,
        )
    }

    fn observe_output_with_context<'a>(assistant:&mut Self::Assistant,_request:&Self::Capture,
        output:&mut ExternalTargetOutput<Self::Tensor>,path:&str,context:Self::Context<'a>)->Result<(),Self::Error>{
        output.logits=M::observe_borrowed_tensor(assistant,path,&output.logits,output.evidence.as_ref(),context)?;Ok(())
    }
    fn target_token_packet<'a>(tokens:&[u32],context:Self::Context<'a>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{M::target_tokens_with_source(tokens,context)}
    fn token_packet_prefix<'a>(value:&EmbeddedPredictionTensor<Self::Tensor>,end:usize,context:Self::Context<'a>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{M::token_prefix_with_source(value,end,context)}
    fn verify_target_packet<'a>(target:&mut Self::Target,assistant:&mut Self::Assistant,request:&Self::Capture,
        tokens:&EmbeddedPredictionTensor<Self::Tensor>,cache:&mut Self::Cache,context:Self::Context<'a>)->Result<ExternalTargetOutput<Self::Tensor>,Self::Error>{
        let evidence=tokens.evidence();let context=M::source_context(context,evidence.as_slice())?;
        Self::verify_target(target,assistant,request,tokens,cache,context)
    }
    fn control_state_copy(state:&ExternalTargetState<Self::Tensor>,context:Self::Context<'_>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{state::control_copy::<M>(state,context)}
    fn state_at<'context>(output:&ExternalTargetOutput<Self::Tensor>,row:usize,cache_len:i32,context:Self::Context<'context>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{
        state::select::<M>(&output.hidden,output.shared_kv.iter().map(|(p,(k,v))|(*p,k,v)),output.evidence.as_ref(),row,cache_len,context)
    }
    fn state_on_draft<'context>(state:&ExternalTargetState<Self::Tensor>,context:Self::Context<'context>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{state::on_draft::<M>(state,context)}
    fn copy_draft<'a>(state:&AssistantState<Self::Tensor>,context:Self::Context<'a>)->Result<AssistantState<Self::Tensor>,Self::Error>{state::copy_draft::<M>(state,context)}

    fn submit_verification<'a>(output:&ExternalTargetOutput<Self::Tensor>,inputs:&Self::Tensor,context:Self::Context<'a>)->Result<Self::Completion,Self::Error>{
        if output.evidence.is_none(){return M::submit_completion([&output.logits,&output.hidden,inputs].into_iter().chain(output.shared_kv.values().flat_map(|(k,v)|[k,v])));}
        M::submit_completion_with_sources([&output.logits,&output.hidden].into_iter().chain(output.shared_kv.values().flat_map(|(k,v)|[k,v])),output.evidence.as_ref().as_slice(),context)
    }

}

fn observe_capture<M>(
    assistant: &mut M::Assistant,
    cache: &ExternalAssistantCache<M::NativeCache>,
    capture: &mut ExternalPredictionTargetCapture<M::Tensor>,
    evidence: Option<&crate::speculative_execution::PreparedEmbeddedEvidence>,
    context: M::Context<'_>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = capture else {
        return Err(M::error(
            "Gemma target returned a different assistant capture".into(),
        ));
    };
    let mut paths=cache.capture_paths_iter();
    let controls=std::mem::size_of_val(&paths).checked_add(std::mem::size_of::<Result<(),M::Error>>());
    let _host=M::state_host_metadata(controls,context)?;
    if paths.len()!=shared_kv.len().checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||M::state_refusal(context))?{return Err(M::state_refusal(context));}
    let path=paths.next().ok_or_else(||M::state_refusal(context))?;
    *hidden=M::observe_borrowed_tensor(assistant,path,hidden,evidence,context)?;
    for (_,keys,values) in shared_kv {
        let key_path=paths.next().ok_or_else(||M::state_refusal(context))?;
        let value_path=paths.next().ok_or_else(||M::state_refusal(context))?;
        *keys=M::observe_borrowed_tensor(assistant,key_path,keys,evidence,context)?;
        *values=M::observe_borrowed_tensor(assistant,value_path,values,evidence,context)?;
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

fn validate_output_capture_in_context<M>(
    cache: &ExternalAssistantCache<M::NativeCache>,
    output: &ExternalTargetOutput<M::Tensor>,
    context:M::Context<'_>,
) -> Result<(), M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    cache.validate_capture_values::<Gemma4AssistantArchitecture,M,_>(
        ||std::iter::once(&output.hidden).chain(output.shared_kv.values().flat_map(|(k,v)|[k,v])),context)

}

fn target_output<M>(
    logits: M::Tensor,
    capture: ExternalPredictionTargetCapture<M::Tensor>,
    evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
    context: M::Context<'_>,
) -> Result<ExternalTargetOutput<M::Tensor>, M::Error>
where
    M: ExternalAssistantExecutionMechanisms<Gemma4AssistantArchitecture>,
{
    let input=crate::external_assistant::ExternalTargetResult{logits,capture,evidence};
    let ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv } = input.capture else {
        return Err(M::error(
            "Gemma 4 target returned a different assistant capture".into(),
        ));
    };
    let mut rows=M::state_buffer(shared_kv.len(),context)?;
    for (policy,keys,values) in shared_kv {
        if rows.iter().any(|(p,_)|*p==policy){return Err(M::state_refusal(context));}
        rows.try_push((policy,(keys,values))).map_err(|_|M::state_refusal(context))?;
    }
    Ok(ExternalTargetOutput {logits:input.logits,hidden,shared_kv:SharedAssistantStates::from_prepared(M::freeze_state_values(rows,context)?),evidence:input.evidence})
}

/// Exact ordinary-target output consumed by the Gemma 4 assistant lifecycle.
pub struct ExternalTargetOutput<T> {
    /// Target vocabulary logits.
    pub logits: T,
    /// Target-width hidden capture used to seed the assistant.
    pub hidden: T,
    /// Shared target K/V captures keyed by attention policy.
    pub shared_kv: SharedAssistantStates<T>,
    /// Actual completed source retained until after all output values.
    pub evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
}

/// Committed Gemma 4 target state from which one private assistant round starts.
#[derive(Debug, Clone)]
pub struct ExternalTargetState<T> {
    hidden: T,
    shared_kv: SharedAssistantStates<T>,
    cache_len: i32,
    evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
}

impl<T: Clone> ExternalTargetState<T> {
    fn map<E>(&self, mut f: impl FnMut(&T) -> Result<T, E>) -> Result<Self, E> {
        Ok(Self {
            evidence: self.evidence.clone(),
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

/// Retained target verification output and its exact input tokens.
pub struct ExternalVerification<T> {
    output: ExternalTargetOutput<T>,
    inputs: EmbeddedPredictionTensor<T>,
}

/// Backend mechanisms required by the architecture-owned Gemma 4 lifecycle.
///
/// Implementations bind already materialized target and assistant objects to tensor, transfer,
/// cache, and completion operations. Proposal ordering, target-state geometry, replay, and commit
/// policy remain in [`ExternalExecutor`].
pub trait ExternalMechanisms: 'static {
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
    type Logits;
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
    ) -> Result<EmbeddedPredictionTensor<Self::Tensor>, Self::Error>;

    /// Executes one neutral assistant step.
    fn draft_step<'a>(
        assistant: &mut Self::Assistant,
        embedding: &EmbeddedPredictionTensor<Self::Tensor>,
        state: &mut AssistantState<Self::Tensor>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Ordinary fallback keeps the existing observation callback.
    fn observe_output_with_context<'a>(assistant:&mut Self::Assistant,request:&Self::Capture,
        output:&mut ExternalTargetOutput<Self::Tensor>,path:&str,context:Self::Context<'a>)->Result<(),Self::Error>{let _=context;Self::observe_output(assistant,request,output,path)}
    /// Creates immutable token transport for one verification.
    fn target_token_packet<'a>(tokens:&[u32],context:Self::Context<'a>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::target_tokens(tokens,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Retains source-aware input while selecting the actual accepted prefix.
    fn token_packet_prefix<'a>(value:&EmbeddedPredictionTensor<Self::Tensor>,end:usize,context:Self::Context<'a>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::token_prefix(value,end,context).map(EmbeddedPredictionTensor::ordinary)}
    /// Lends the explicit packet source to the same target verification worker.
    fn verify_target_packet<'a>(target:&mut Self::Target,assistant:&mut Self::Assistant,request:&Self::Capture,
        tokens:&EmbeddedPredictionTensor<Self::Tensor>,cache:&mut Self::Cache,context:Self::Context<'a>)->Result<ExternalTargetOutput<Self::Tensor>,Self::Error>{Self::verify_target(target,assistant,request,tokens,cache,context)}
    /// Builds the committed seed with ordinary or source-aware state views.
    fn state_at<'context>(
        output: &ExternalTargetOutput<Self::Tensor>,
        row: usize,
        cache_len: i32,
        context: Self::Context<'context>,
    ) -> Result<ExternalTargetState<Self::Tensor>, Self::Error> {
        let hidden = Self::hidden_row(&output.hidden, row, context)?;
        let shared_kv = output
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                Ok((
                    *policy,
                    (
                        Self::shared_prefix(keys, cache_len, context)?,
                        Self::shared_prefix(values, cache_len, context)?,
                    ),
                ))
            })
            .collect::<Result<SharedAssistantStates<_>, Self::Error>>()?;
        Ok(ExternalTargetState {
            evidence: output.evidence.clone(),
            hidden,
            shared_kv,
            cache_len,
        })
    }
    /// Places an actual committed seed on the assistant device.
    fn state_on_draft<'context>(
        state: &ExternalTargetState<Self::Tensor>,
        context: Self::Context<'context>,
    ) -> Result<ExternalTargetState<Self::Tensor>, Self::Error> {
        let hidden = Self::target_to_draft(&state.hidden, context)?;
        let shared_kv = state
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                Ok((
                    *policy,
                    (
                        Self::target_to_draft(keys, context)?,
                        Self::target_to_draft(values, context)?,
                    ),
                ))
            })
            .collect::<Result<SharedAssistantStates<_>, Self::Error>>()?;
        Ok(ExternalTargetState {
            evidence: state.evidence.clone(),
            hidden,
            shared_kv,
            cache_len: state.cache_len,
        })
    }    /// Isolated control-state copy; ordinary engines preserve their own tensor worker.
    fn control_state_copy(state:&ExternalTargetState<Self::Tensor>,context:Self::Context<'_>)->Result<ExternalTargetState<Self::Tensor>,Self::Error>{
        state.map(|value|Self::control_copy_tensor(value,ExternalAssistantTensorPlacement::Target,context))
    }
    /// Copies the current private proposal while preserving source custody.
    fn copy_draft<'a>(state:&AssistantState<Self::Tensor>,context:Self::Context<'a>)->Result<AssistantState<Self::Tensor>,Self::Error>{let _=context;Ok(state.clone())}

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
    capture: crate::external_assistant::CaptureSource<'a,M::Capture>,
    origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    _mechanisms: PhantomData<fn() -> M>,
}

impl<'a, M: ExternalMechanisms> ExternalExecutor<'a, M> {
    /// Lends the exact immutable capture declaration owned by the selected source.
    pub const fn new_borrowed(target:&'a mut M::Target,assistant:&'a mut M::Assistant,capture:&'a M::Capture)->Self{
        Self{target,assistant,capture:crate::external_assistant::CaptureSource::Borrowed(capture),origin:None,_mechanisms:PhantomData}
    }

    /// Binds already materialized target and assistant objects to the neutral lifecycle.
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

    fn state_at<'context>(output:&ExternalTargetOutput<M::Tensor>,row:usize,cache_len:i32,context:M::Context<'context>)->Result<ExternalTargetState<M::Tensor>,M::Error>{M::state_at(output,row,cache_len,context)}

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

    fn state_on_draft<'context>(state:&ExternalTargetState<M::Tensor>,context:M::Context<'context>)->Result<ExternalTargetState<M::Tensor>,M::Error>{M::state_on_draft(state,context)}

}

impl<M: ExternalMechanisms> SpeculativeExecutor for ExternalExecutor<'_, M> {
    type Input = M::Input;
    type Cache = M::Cache;
    type TargetState = ExternalTargetState<M::Tensor>;
    type DraftState = AssistantState<M::Tensor>;

    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'a,
    {
        M::copy_draft(state,context)
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
        let state=M::control_state_copy(state,context)
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
        let state=M::control_state_copy(state,context)
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
            M::observe_output_with_context(self.assistant,&self.capture,&mut output,eredu_core::MODEL_LOGITS_OBSERVATION_PATH,context)?;
            M::validate_output_capture_with_context(cache, &output, context)?;
            let sequence = M::sequence_len(&output.logits)?;
            if sequence == 0 {
                return Err(M::empty_input());
            }
            Self::validate_output(&output, sequence)?;
            let row = sequence - 1;
            let logits = M::logits_row_with_source(&output.logits, row, output.evidence.as_ref(), context)?;
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

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        _last_token: u32,
        _proposal_capacity: usize,
        context: M::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        let context = M::invocation_context(context, self.origin)?;
        let state = Self::state_on_draft(state, context)?;
        Ok(AssistantState {
            evidence: state.evidence.clone(),
            shared_kv: state.shared_kv,
            kv_offset: state.cache_len,
            hidden: state.hidden,
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: M::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        let context = M::invocation_context(context, self.origin)?;
        let embedding = M::proposal_embedding(self.target, last_token, context)?;
        M::draft_step(self.assistant, &embedding, state, context)
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
        let inputs = M::target_token_packet(input_tokens, context)?;
        let mut output = M::verify_target_packet(
            self.target,
            self.assistant,
            &self.capture,
            &inputs,
            cache,
            context,
        )?;
        M::observe_output_with_context(self.assistant,&self.capture,&mut output,EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,context)?;
        M::validate_output_capture_with_context(cache, &output, context)?;
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
        M::logits_row_with_source(&output.output.logits, index, output.output.evidence.as_ref(), context)
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: M::Context<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        let context = M::invocation_context(context, self.origin)?;
        let input_len = M::sequence_len(&output.inputs)?;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(M::invalid_commit(verified_inputs, input_len));
        }
        if verified_inputs == input_len {
            return Ok(SpeculativeCommit::new(
                Self::state_at(
                    &output.output,
                    verified_inputs - 1,
                    M::cache_len(cache)?,
                    context,
                )?,
                0,
            ));
        }

        M::restore_checkpoint(cache, checkpoint, context)?;
        let retained = M::token_packet_prefix(&output.inputs, verified_inputs, context)?;
        let mut replayed = M::verify_target_packet(
            self.target,
            self.assistant,
            &self.capture,
            &retained,
            cache,
            context,
        )?;
        M::observe_output_with_context(self.assistant,&self.capture,&mut replayed,EXTERNAL_ASSISTANT_VERIFICATION_LOGITS_OBSERVATION_PATH,context)?;
        M::validate_output_capture_with_context(cache, &replayed, context)?;
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

#[cfg(test)]
mod tests {
    use crate::speculative_execution::EmbeddedPredictionTensor;
    use std::collections::HashMap;

    use eredu_core::{AttentionPolicy, Completion, SpeculativeExecutor};

    use super::{ExternalExecutor, ExternalMechanisms, ExternalTargetOutput};
    use crate::gemma4::AssistantState;

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
    struct Target;

    #[derive(Default)]
    struct Assistant {
        fail_on: Option<&'static str>,
        observations: usize,
    }

    #[derive(Debug)]
    struct TestError(&'static str);

    impl std::fmt::Display for TestError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for TestError {}

    struct Ready;

    impl Completion for Ready {
        type Error = TestError;

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
        fn output(sequence: usize, value: u32) -> ExternalTargetOutput<Tensor> {
            ExternalTargetOutput {
                evidence: None,
                logits: Tensor::new(sequence, value),
                hidden: Tensor::new(sequence, value + 10),
                shared_kv: HashMap::from([(
                    AttentionPolicy::Full,
                    (Tensor::new(sequence + 2, 30), Tensor::new(sequence + 2, 40)),
                )]).into(),
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
        type Error = TestError;

        fn max_proposals(_assistant: &Self::Assistant) -> usize {
            4
        }

        fn empty_input() -> Self::Error {
            unreachable!()
        }

        fn invalid_commit(_verified: usize, _available: usize) -> Self::Error {
            unreachable!()
        }

        fn invalid_output(_expected: usize, _logits: usize, _hidden: usize) -> Self::Error {
            unreachable!()
        }

        fn observe_tensor(
            assistant: &mut Self::Assistant,
            path: &str,
            value: Self::Tensor,
        ) -> Result<Self::Tensor, Self::Error> {
            assistant.observations += 1;
            if assistant.fail_on == Some(path) {
                Err(TestError("injected architecture observer failure"))
            } else {
                Ok(value)
            }
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
            Ok(Self::output(input, 1))
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
            Ok(Self::output(tokens.sequence, 2))
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

        fn hidden_row(
            value: &Self::Tensor,
            _row: usize,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(1, value.value))
        }

        fn shared_prefix(
            value: &Self::Tensor,
            cache_len: i32,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            let mut value = value.clone();
            value.retained = value.retained.min(usize::try_from(cache_len).unwrap());
            Ok(value)
        }

        fn target_to_draft(
            value: &Self::Tensor,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(value.clone())
        }

        fn target_tokens(
            tokens: &[u32],
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(tokens.len(), tokens[0]))
        }

        fn token_prefix(
            value: &Self::Tensor,
            end: usize,
            _context: Self::Context<'_>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor::new(end, value.value))
        }

        fn proposal_embedding(
            _target: &mut Self::Target,
            token: u32,
            _context: Self::Context<'_>,
        ) -> Result<EmbeddedPredictionTensor<Self::Tensor>, Self::Error> {
            Ok(EmbeddedPredictionTensor::ordinary(Tensor::new(1, token)))
        }

        fn draft_step(
            _assistant: &mut Self::Assistant,
            embedding: &EmbeddedPredictionTensor<Self::Tensor>,
            state: &mut AssistantState<Self::Tensor>,
            _context: Self::Context<'_>,
        ) -> Result<Self::Logits, Self::Error> {
            state.kv_offset += 1;
            state.hidden = (**embedding).clone();
            Ok(embedding.value + 100)
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
    fn external_lifecycle_owns_state_geometry_and_partial_replay() {
        let mut target = Target;
        let mut assistant = Assistant::default();
        let mut cache = Cache::default();
        let mut executor = ExternalExecutor::<Mechanisms>::new(&mut target, &mut assistant, ());

        let prefill = executor.prefill(3, &mut cache, ()).unwrap();
        let (logits, target_state, evaluated) = prefill.into_parts();
        assert_eq!((logits, evaluated, cache.len), (3, 3, 3));
        assert_eq!(target_state.shared_kv[&AttentionPolicy::Full].0.retained, 3);

        let mut draft = executor.begin_proposal(&target_state, 7, 2, ()).unwrap();
        assert_eq!(draft.kv_offset, 3);
        assert_eq!(executor.proposal_logits(&mut draft, 7, ()).unwrap(), 107);
        assert_eq!(draft.kv_offset, 4);

        let checkpoint = executor.checkpoint(&cache).unwrap();
        let aborted = executor
            .submit_verification(&[7, 8, 9], &mut cache, ())
            .unwrap();
        assert_eq!(cache.len, 6);
        executor
            .restore_checkpoint(&mut cache, &checkpoint, ())
            .unwrap();
        drop(aborted);
        assert_eq!((cache.len, cache.restores), (3, 1));
        let submission = executor
            .submit_verification(&[7, 8, 9], &mut cache, ())
            .unwrap();
        let commit = executor
            .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
            .unwrap();
        let (committed, replayed) = commit.into_parts();
        assert_eq!((cache.len, cache.restores, replayed), (5, 2, 2));
        assert_eq!(committed.cache_len, 5);
        assert_eq!(committed.hidden.sequence, 1);
        assert_eq!(committed.shared_kv[&AttentionPolicy::Full].0.retained, 4);
    }

    #[test]
    fn observer_failure_restores_prefill_transaction() {
        let mut target = Target;
        let mut assistant = Assistant {
            fail_on: Some(eredu_core::MODEL_LOGITS_OBSERVATION_PATH),
            observations: 0,
        };
        let mut cache = Cache::default();
        let error = {
            let mut executor = ExternalExecutor::<Mechanisms>::new(&mut target, &mut assistant, ());
            match executor.prefill(3, &mut cache, ()) {
                Ok(_) => panic!("injected architecture observer failure unexpectedly succeeded"),
                Err(error) => error,
            }
        };
        assert_eq!(error.to_string(), "injected architecture observer failure");
        assert_eq!((cache.len, cache.restores), (0, 1));
        assert_eq!(assistant.observations, 1);
    }
}
