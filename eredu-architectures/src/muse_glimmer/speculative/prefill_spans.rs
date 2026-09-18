//! DFlash keeps only its ordered raw rolling suffix; first encoding stays lazy.
use super::*;
use crate::external_assistant::{prefill::CaptureIdentity, ExternalPrefillReceiver};
use eredu_core::{Completion, GenerationCancellationToken, SpeculativePrefillOutcome};
use eredu_runtime::{prefill::PrefillChunk, replicated_session::PrefillSourceOutcome};

type Architecture = MuseGlimmerAssistantArchitecture;
struct Receiver<'a, 'c, M: ExternalAssistantExecutionMechanisms<Architecture>> {
    assistant: &'a mut M::Assistant,
    identity: CaptureIdentity<'a>,
    context: M::Context<'c>,
    pending: Option<M::Tensor>,
    pending_source:Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
    score_source:Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
    frontier: Option<i32>,
    next: u64,
    committed: bool,
    whole_prompt: bool,
}
impl<M: ExternalAssistantExecutionMechanisms<Architecture>> Drop for Receiver<'_, '_, M> {
    fn drop(&mut self) {
        if !self.whole_prompt {
            M::finish_prefill(self.assistant, self.committed);
        }
    }
}
impl<M: ExternalAssistantExecutionMechanisms<Architecture>>
    ExternalPrefillReceiver<M::Tensor, M::Error> for Receiver<'_, '_, M>
{
    fn requires_whole_prompt(&self) -> bool {
        self.whole_prompt
    }
    fn output_demand(&self) -> eredu_core::OutputDemand {
        M::prefill_output_demand(self.assistant)
    }
    fn prepare(&mut self, chunk: &PrefillChunk) -> Result<(), M::Error> {
        if chunk.input.start != self.next {
            return Err(M::error("DFlash capture spans are not contiguous".into()));
        }
        if !self.whole_prompt && !M::supports_prefill_observation(self.assistant, false) {
            return Err(M::neural_error(eredu_nn::Error::backend_retained_source(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "observer requires explicit prompt-span support",
                ),
            )));
        }
        Ok(())
    }
    fn consume(
        &mut self,
        chunk: &PrefillChunk,
        frontier: u64,
        scores: &mut Option<M::Tensor>,
        capture: ExternalPredictionTargetCapture<M::Tensor>,
    ) -> Result<(), M::Error> {
        self.consume_with_evidence(chunk,frontier,scores,capture,None)
    }
    fn consume_with_evidence(&mut self,chunk:&PrefillChunk,frontier:u64,scores:&mut Option<M::Tensor>,
        capture:ExternalPredictionTargetCapture<M::Tensor>,evidence:Option<crate::speculative_execution::PreparedEmbeddedEvidence>)->Result<(),M::Error>{
        let controls=[std::mem::size_of::<Self>(),std::mem::size_of::<Result<(),M::Error>>(),
            std::mem::size_of::<crate::external_assistant::ExternalOperationResult<M::Tensor>>(),
            std::mem::size_of::<super::super::assistant::invocation::RawContextArguments<'_,M::Tensor>>()];
        let _host=M::state_host_metadata(controls.into_iter().try_fold(std::mem::size_of_val(&controls),usize::checked_add),self.context)?;
        let ExternalPredictionTargetCapture::MuseGlimmerDFlash { mut target_states } = capture
        else {
            return Err(M::error(
                "DFlash target returned a different assistant capture".into(),
            ));
        };
        let width = usize::try_from(chunk.input.end - chunk.input.start)
            .map_err(|e| M::error(e.to_string()))?;
        if chunk.position.checked_add(width as u64) != Some(frontier) {
            return Err(M::error(
                "DFlash capture differs from installed span frontier".into(),
            ));
        }
        let validate = |states: &[M::Tensor]| {
            for state in states {
                if M::state_dimension(state,1,self.context)? != width {
                    return Err(M::state_refusal(self.context));
                }
            }
            self.identity.validate_values::<Architecture,M,_>(frontier,||states.iter(),self.context)

        };
        validate(&target_states)?;
        let mut paths=self.identity.paths();
        if !self.whole_prompt { M::begin_prefill_chunk(self.assistant,chunk)?; }
        for state in &mut target_states {
            let path=paths.next().ok_or_else(||M::state_refusal(self.context))?;
            *state=M::observe_borrowed_tensor(self.assistant,path,state,evidence.as_ref(),self.context)?;
        }
        if paths.next().is_some(){return Err(M::state_refusal(self.context));}
        validate(&target_states)?;
        if let Some(value) = scores.take() {
            let value = M::observe_tensor(
                self.assistant,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                value,
            )?;
            let expected = if chunk.output == eredu_core::OutputDemand::Sequence {
                width
            } else {
                1
            };
            if M::sequence_len(&value)? != expected {
                return Err(M::error("observer changed selected score width".into()));
            }
            *scores = Some(value);
        }
        let window = M::config(self.assistant).sliding_window;
        if window <= 0 {
            return Err(M::error("DFlash context window must be positive".into()));
        }
        // The actual raw assembly shares its ordinary equation with the quote.
        // Ordered taps and previous suffix remain explicit source loans.
        let mut priors=M::state_buffer(2,self.context)?;
        if let Some(proof)=&evidence{priors.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        if let Some(proof)=&self.pending_source{priors.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        let operation_context=M::source_context(self.context,&priors)?;
        let pending=M::assistant_operation_with_evidence::<super::super::assistant::invocation::RawContext>(self.assistant,
            super::super::assistant::invocation::RawContextArguments{previous:self.pending.as_ref(),states:&target_states},operation_context)?;
        drop(priors);
        let expected = usize::try_from(chunk.input.end)
            .map_err(|e| M::error(e.to_string()))?
            .min(window as usize);
        if M::state_dimension(&pending.output,1,self.context)? != expected {
            return Err(M::error(
                "DFlash raw suffix differs from completed prompt rows".into(),
            ));
        }
        let mut proofs=M::state_buffer(2,self.context)?;
        if let Some(proof)=&evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        if let Some(proof)=&pending.evidence{proofs.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        let completion=M::submit_completion_with_sources(target_states.iter().chain(scores.iter()).chain([&pending.output]),&proofs,self.context)?;
        completion.wait()?;
        drop(proofs);
        self.pending=Some(pending.output);
        self.pending_source=pending.evidence;
        if scores.is_some(){self.score_source=evidence;}
        self.frontier = Some(i32::try_from(frontier).map_err(|e| M::error(e.to_string()))?);
        self.next = chunk.input.end;
        Ok(())
    }
}

pub(super) fn run<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    target: &mut M::Target,
    assistant: &mut M::Assistant,
    request: &crate::composite_execution::ExternalPredictionCaptureRequest,
    input: M::Input,
    cache: &mut ExternalAssistantCache<M::NativeCache>,
    cancellation: &GenerationCancellationToken,
    context: M::Context<'_>,
) -> Result<
    SpeculativePrefillOutcome<SpeculativePrefill<ExternalTargetState<M::Tensor>, M::Logits>>,
    M::Error,
> {
    // Unknown whole-tensor callbacks retain one complete invocation. Their
    // declared readout demand remains independent from full capture/KV storage.
    let whole_prompt = M::prefill_chunk_positions(&input).is_none()
        && !M::supports_prefill_observation(assistant, false);
    let previous_frontier = cache.frontier().map_err(M::error)?;
    let (native, identity) = cache.prefill_parts().map_err(M::error)?;
    let _host=M::state_host_metadata(Some(std::mem::size_of::<Receiver<'_, '_,M>>()
        +std::mem::size_of::<Result<SpeculativePrefillOutcome<SpeculativePrefill<ExternalTargetState<M::Tensor>,M::Logits>>,M::Error>>()),context)?;
    let mut receiver = Receiver::<M> {
        assistant,
        identity,
        context,
        pending: None,
        pending_source:None,score_source:None,
        frontier: None,
        next: 0,
        committed: false,
        whole_prompt,
    };
    let progress = M::prefill_target_spans_native(
        target,
        request,
        input,
        native,
        &mut receiver,
        cancellation,
        context,
    )?;
    let frontier = M::native_cache_len(native)?;
    if frontier < previous_frontier {
        return Err(M::error(
            "DFlash target frontier regressed during prefill".into(),
        ));
    }
    let evaluated =
        usize::try_from(progress.completed_positions).map_err(|e| M::error(e.to_string()))?;
    let result = match progress.outcome {
        PrefillSourceOutcome::Unavailable => Err(M::error(
            "selected external prompt source is unavailable".into(),
        )),
        PrefillSourceOutcome::Cancelled => Ok(SpeculativePrefillOutcome::Cancelled {
            evaluated_tokens: evaluated,
        }),
        PrefillSourceOutcome::Complete(scores) => {
            let scores = scores
                .ok_or_else(|| M::error("selected DFlash prefill omitted final scores".into()))?;
            let pending = receiver
                .pending
                .take()
                .ok_or_else(|| M::error("selected DFlash prefill omitted raw context".into()))?;
            if receiver.frontier != Some(frontier) {
                return Err(M::error(
                    "DFlash raw context frontier differs after target exchange".into(),
                ));
            }
            receiver.committed = true;
            Ok(SpeculativePrefillOutcome::Complete(
                SpeculativePrefill::new(
                    M::into_logits_with_source(scores,receiver.score_source.as_ref(),context)?,
                    ExternalTargetState {
                        pending_context: Some(pending),
                        draft_context: None,
                        cache_len: frontier,
                        evidence:receiver.pending_source.take(),
                    },
                    evaluated,
                ),
            ))
        }
    };
    drop(receiver);
    cache.advance_frontier(frontier).map_err(M::error)?;
    result
}
