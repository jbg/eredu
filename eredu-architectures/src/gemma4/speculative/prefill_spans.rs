//! Gemma retains the final hidden row and policy-keyed complete target context.
use super::*;
use crate::external_assistant::{prefill::CaptureIdentity, ExternalPrefillReceiver};
use eredu_core::{Completion, GenerationCancellationToken, SpeculativePrefillOutcome};
use eredu_runtime::{prefill::PrefillChunk, replicated_session::PrefillSourceOutcome};

type Architecture = Gemma4AssistantArchitecture;
struct Receiver<'a, 'c, M: ExternalAssistantExecutionMechanisms<Architecture>> {
    assistant: &'a mut M::Assistant,
    identity: CaptureIdentity<'a>,
    context: M::Context<'c>,
    state: Option<ExternalTargetState<M::Tensor>>,
    score_source: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
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
            return Err(M::error("external capture spans are not contiguous".into()));
        }
        if !self.whole_prompt && !M::supports_prefill_observation(self.assistant, true) {
            return Err(M::neural_error(eredu_nn::Error::backend_source(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "observer requires explicit prompt-span and complete-context support",
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
        self.consume_with_evidence(chunk, frontier, scores, capture, None)
    }
    fn consume_with_evidence(&mut self, chunk: &PrefillChunk, frontier: u64,
        scores: &mut Option<M::Tensor>, capture: ExternalPredictionTargetCapture<M::Tensor>,
        evidence: Option<crate::speculative_execution::PreparedEmbeddedEvidence>,
    ) -> Result<(), M::Error> {
        let ExternalPredictionTargetCapture::Gemma4 {
            mut hidden,
            mut shared_kv,
        } = capture
        else {
            return Err(M::error(
                "Gemma target returned a different assistant capture".into(),
            ));
        };
        let width = usize::try_from(chunk.input.end - chunk.input.start)
            .map_err(|e| M::error(e.to_string()))?;
        if chunk.position.checked_add(width as u64) != Some(frontier)
            || M::sequence_len(&hidden)? != width
        {
            return Err(M::error(
                "Gemma capture differs from the installed span frontier".into(),
            ));
        }
        let validate =
            |hidden: &M::Tensor, shared: &[(eredu_core::AttentionPolicy, M::Tensor, M::Tensor)]| {
                self.identity.validate_values::<Architecture,M,_>(frontier,
                    ||std::iter::once(hidden).chain(shared.iter().flat_map(|(_,k,v)|[k,v])),self.context)

            };
        validate(&hidden, &shared_kv)?;
        let mut paths = self.identity.paths();
        let hidden_path = paths.next().ok_or_else(||M::state_refusal(self.context))?;
        if !self.whole_prompt {
            M::begin_prefill_chunk(self.assistant, chunk)?;
        }
        hidden = M::observe_borrowed_tensor(self.assistant, hidden_path, &hidden, evidence.as_ref(), self.context)?;
        // Context is never labeled with the prompt span's row window. Its
        // policy-specific axis/extent remains the actual complete target value.
        if !self.whole_prompt {
            M::begin_prefill_context(self.assistant, frontier)?;
        }
        for (_,keys,values) in &mut shared_kv {
            let key_path=paths.next().ok_or_else(||M::state_refusal(self.context))?;
            let value_path=paths.next().ok_or_else(||M::state_refusal(self.context))?;
            *keys=M::observe_borrowed_tensor(self.assistant,key_path,keys,evidence.as_ref(),self.context)?;
            *values=M::observe_borrowed_tensor(self.assistant,value_path,values,evidence.as_ref(),self.context)?;
        }
        if paths.next().is_some(){return Err(M::state_refusal(self.context));}
        validate(&hidden, &shared_kv)?;
        if M::sequence_len(&hidden)? != width {
            return Err(M::error("Gemma observer changed capture span width".into()));
        }
        if let Some(value) = scores.take() {
            if !self.whole_prompt {
                M::begin_prefill_chunk(self.assistant, chunk)?;
            }
            let value = M::observe_borrowed_tensor(
                self.assistant,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                &value,
                evidence.as_ref(),
                self.context,
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
        let cache_len = i32::try_from(frontier).map_err(|_|M::state_refusal(self.context))?;
        let state=state::select::<M>(&hidden,shared_kv.iter().map(|(p,k,v)|(*p,k,v)),
            evidence.as_ref(),width-1,cache_len,self.context)?;
        let mut proofs=M::state_buffer(2,self.context)?;
        if let Some(proof)=evidence.as_ref(){proofs.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        if let Some(proof)=state.evidence.as_ref(){proofs.try_push(proof).map_err(|_|M::state_refusal(self.context))?;}
        let mut completion=M::submit_completion_with_sources(
            [&hidden,&state.hidden].into_iter()
                .chain(shared_kv.iter().flat_map(|(_,k,v)|[k,v]))
                .chain(state.shared_kv.values().flat_map(|(k,v)|[k,v]))
                .chain(scores.iter()),&proofs,self.context)?;
        completion.wait()?;
        drop(proofs);
        if scores.is_some(){self.score_source=evidence;}
        self.state=Some(state);
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
        && !M::supports_prefill_observation(assistant, true);
    let previous_frontier = cache.frontier().map_err(M::error)?;
    let (native, identity) = cache.prefill_parts().map_err(M::error)?;
    let mut receiver = Receiver::<M> {
        assistant,
        identity,
        context,
        state: None,
        score_source: None,
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
            "Gemma target frontier regressed during prefill".into(),
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
                .ok_or_else(|| M::error("selected Gemma prefill omitted final scores".into()))?;
            let state = receiver
                .state
                .take()
                .ok_or_else(|| M::error("selected Gemma prefill omitted seed state".into()))?;
            if state.cache_len != frontier {
                return Err(M::error(
                    "Gemma seed frontier differs after target exchange".into(),
                ));
            }
            receiver.committed = true;
            Ok(SpeculativePrefillOutcome::Complete(
                SpeculativePrefill::new(M::into_logits_with_source(scores, receiver.score_source.as_ref(), context)?, state, evaluated),
            ))
        }
    };
    drop(receiver);
    cache.advance_frontier(frontier).map_err(M::error)?;
    result
}
