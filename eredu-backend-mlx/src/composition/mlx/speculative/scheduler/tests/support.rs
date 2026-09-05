use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::backend::ExecutionContext;
use eredu_core::{SpeculativeCommit, SpeculativePrefill, Submission};
use safemlx::{Device, DeviceType};

use super::*;
use crate::backend::runtime::media::input::InputPart;
use eredu_runtime::{DefaultSampler, GenerationSampler, MirostatV2Sampler};

trait TestInputPart {
    fn text_token_ids(tokens: &Array) -> Self;
}

impl TestInputPart for InputPart {
    fn text_token_ids(tokens: &Array) -> Self {
        crate::backend::runtime::media::input::input_part(
            eredu_core::InputModality::Text,
            crate::backend::runtime::media::input::InputPayload::TokenIds(tokens.clone()),
            [],
            [],
        )
        .unwrap()
    }
}

#[derive(Clone, Default)]
struct CountingSampler {
    process_calls: usize,
    histories: Vec<Vec<u32>>,
    committed: Vec<u32>,
}

impl SpeculativeSampler<MlxSamplingBackend> for CountingSampler {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn process_logits(
        &mut self,
        logits: &MlxTensor,
        _temperature: f32,
        history: &[u32],
        _stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        self.process_calls += 1;
        self.histories.push(history.to_vec());
        Ok(logits.clone())
    }

    fn commit_token(
        &mut self,
        _processed_logits: &MlxTensor,
        token: u32,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        self.committed.push(token);
        Ok(())
    }
}

#[derive(Clone)]
struct GrammarCountingSampler {
    inner: CountingSampler,
    complete_after: usize,
}

impl SpeculativeSampler<MlxSamplingBackend> for GrammarCountingSampler {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn process_logits(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut self.inner,
            logits,
            temperature,
            history,
            stream,
        )
    }

    fn commit_token(
        &mut self,
        processed_logits: &MlxTensor,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        SpeculativeSampler::<MlxSamplingBackend>::commit_token(
            &mut self.inner,
            processed_logits,
            token,
            stream,
        )
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        Ok(self.inner.committed.len() >= self.complete_after)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
        Ok(history.len() >= self.complete_after)
    }
}

#[derive(Clone, Default)]
struct TestSemanticState {
    tokens: Vec<u32>,
    stop: Vec<u32>,
    events: Vec<SemanticEvent>,
}

impl SpeculativeSemanticState for TestSemanticState {
    fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
        let mut fork = self.clone();
        fork.events.clear();
        Ok(Box::new(fork))
    }

    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        self.tokens.push(token);
        self.events
            .push(SemanticEvent::TextDelta(token.to_string()));
        Ok(!self.stop.is_empty() && self.tokens.ends_with(&self.stop))
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        self.events.push(SemanticEvent::Finished { reason });
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        self.events.push(SemanticEvent::Finished {
            reason: FinishReason::Cancelled,
        });
        Ok(())
    }

    fn take_events(&mut self) -> Vec<SemanticEvent> {
        std::mem::take(&mut self.events)
    }
}

#[derive(Clone, Copy, Default)]
struct UniformSampler;

impl SpeculativeSampler<MlxSamplingBackend> for UniformSampler {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn process_logits(
        &mut self,
        logits: &MlxTensor,
        _temperature: f32,
        _history: &[u32],
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        logits
            .as_array()
            .multiply(Array::from_f32(0.0), stream)
            .map(MlxTensor::from_array)
    }
}

struct ScriptedBackend {
    first_token: u32,
    rejection_token: u32,
    reject_first: bool,
    accept_second: bool,
    bonus_token: u32,
    routes: Vec<(&'static str, DeviceType)>,
    draft_storage: Vec<usize>,
    draft_capacities: Vec<usize>,
}

#[derive(Clone)]
struct ScriptedDraftState {
    step: usize,
    storage: Arc<()>,
}

impl ScriptedBackend {
    fn record(&mut self, operation: &'static str, stream: &Stream) -> Result<(), Exception> {
        self.routes
            .push((operation, stream.get_device()?.get_type()?));
        Ok(())
    }
}

impl SpeculativeExecutor for ScriptedBackend {
    type Input = MlxModelInput;
    type Cache = usize;
    type TargetState = ();
    type DraftState = ScriptedDraftState;
    type CacheCheckpoint = usize;
    type Verification = Array;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = SpeculativeComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        2
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn prefill<'context>(
        &mut self,
        _input: MlxModelInput,
        cache: &mut Self::Cache,
        streams: SpeculativeExecutionStreams<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Exception> {
        let stream = streams.target();
        self.record("prefill", stream)?;
        *cache = 1;
        let mut first = [0.0f32; 3];
        first[self.first_token as usize] = 10.0;
        Ok(SpeculativePrefill::new(
            Array::from_slice(&first, &[1, 3]),
            (),
            1,
        ))
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        self.draft_capacities.push(proposal_capacity);
        let _ = (state, last_token);
        self.record("begin_target", streams.target())?;
        self.record("begin_draft", streams.draft())?;
        Ok(ScriptedDraftState {
            step: 0,
            storage: Arc::new(()),
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        _last_token: u32,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let stream = streams.draft();
        self.record("draft", stream)?;
        self.draft_storage
            .push(Arc::as_ptr(&state.storage) as usize);
        let logits = match state.step {
            0 => Array::from_slice(&[0.0f32, 0.0, 10.0], &[1, 1, 3]),
            1 | 2 => Array::from_slice(&[10.0f32, 0.0, 0.0], &[1, 1, 3]),
            _ => Array::from_slice(&[0.0f32, 10.0, 0.0], &[1, 1, 3]),
        };
        state.step += 1;
        Ok(logits)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Ok(*cache)
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        _: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        *cache = *checkpoint;
        Ok(())
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Exception> {
        let input_len = input_tokens.len();
        let stream = streams.target();
        self.record(
            match input_len {
                2 => "verify_input_2",
                3 => "verify_input_3",
                _ => "verify_input_other",
            },
            stream,
        )?;
        self.record("verify", stream)?;
        *cache += input_len;
        let first = if self.reject_first {
            let mut logits = [0.0f32; 3];
            logits[self.rejection_token as usize] = 10.0;
            logits
        } else {
            [0.0f32, 0.0, 10.0]
        };
        let second = if self.accept_second {
            [10.0f32, 0.0, 0.0]
        } else {
            [0.0f32, 10.0, 0.0]
        };
        let mut bonus = [0.0f32; 3];
        bonus[self.bonus_token as usize] = 10.0;
        let output = Array::from_slice(
            &[
                first[0], first[1], first[2], second[0], second[1], second[2], bonus[0], bonus[1],
                bonus[2],
            ],
            &[1, 3, 3],
        );
        let completion = MlxSpeculativeCompletion::submit([&output])?;
        Ok(Submission { output, completion })
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        streams: SpeculativeExecutionStreams<'a>,
    ) -> Result<Array, Exception> {
        output.try_index_device((.., index as i32, ..), streams.target())
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        if verified_inputs != output.dim(1) as usize {
            self.record("cache_truncate", streams.target())?;
        }
        self.record("commit_target", streams.target())?;
        self.record("commit_draft", streams.draft())?;
        *cache = *checkpoint + verified_inputs;
        Ok(SpeculativeCommit::new((), 0))
    }
}

struct CommitFailBackend {
    inner: ScriptedBackend,
}

impl SpeculativeExecutor for CommitFailBackend {
    type Input = MlxModelInput;
    type Cache = usize;
    type TargetState = ();
    type DraftState = ScriptedDraftState;
    type CacheCheckpoint = usize;
    type Verification = Array;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = SpeculativeComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        self.inner.max_proposals()
    }

    fn prefill<'context>(
        &mut self,
        input: MlxModelInput,
        cache: &mut Self::Cache,
        streams: SpeculativeExecutionStreams<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Exception> {
        self.inner.prefill(input, cache, streams)
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        self.inner
            .begin_proposal(state, last_token, proposal_capacity, streams)
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        self.inner.proposal_logits(state, last_token, streams)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        Ok(*cache)
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        _: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        *cache = *checkpoint;
        Ok(())
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Exception> {
        self.inner.submit_verification(input_tokens, cache, streams)
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        streams: SpeculativeExecutionStreams<'a>,
    ) -> Result<Array, Exception> {
        output.try_index_device((.., index as i32, ..), streams.target())
    }

    fn commit_verification(
        &mut self,
        _output: Self::Verification,
        _draft_state: Self::DraftState,
        _cache: &mut Self::Cache,
        _checkpoint: &Self::CacheCheckpoint,
        _verified_inputs: usize,
        _streams: SpeculativeExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        Err(Exception::custom("injected commit failure"))
    }
}

fn scripted_backend() -> ScriptedBackend {
    ScriptedBackend {
        first_token: 1,
        rejection_token: 1,
        reject_first: false,
        accept_second: true,
        bonus_token: 1,
        routes: Vec::new(),
        draft_storage: Vec::new(),
        draft_capacities: Vec::new(),
    }
}
