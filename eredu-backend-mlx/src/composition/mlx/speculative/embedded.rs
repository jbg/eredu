//! MLX executor adapter for checkpoint-embedded prediction heads.

use eredu_core::{SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission};
use eredu_runtime::DraftStateTransaction;
use safemlx::{distributed::Group, error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::generation::sampler::SpeculativeSampler,
    backend::mlx::runtime::media::input::ModelInput,
    composition::mlx::{
        speculative::{
            scheduler::MtpComponentTimings, MlxSpeculativeCompletion, MtpExecutionStreams,
        },
        MlxModelInput,
    },
    MlxTensor,
};

/// Sampler wrapper that keeps PRNG and grammar state identical on every rank
/// while selecting one topology-owned sampling coordinate.
#[derive(Clone)]
pub struct DistributedEmbeddedMtpSampler<'a, S> {
    sampler: S,
    _sampling_rank: usize,
    _group: &'a Group,
}

impl<'a, S> DistributedEmbeddedMtpSampler<'a, S> {
    pub fn new(sampler: S, sampling_rank: usize, group: &'a Group) -> Result<Self, Error> {
        if sampling_rank >= group.size() {
            return Err(Error::Parallel(format!(
                "embedded MTP sampling rank {sampling_rank} is outside world size {}",
                group.size()
            )));
        }
        Ok(Self {
            sampler,
            _sampling_rank: sampling_rank,
            _group: group,
        })
    }

    pub fn into_inner(self) -> S {
        self.sampler
    }
}

impl<S: SpeculativeSampler> SpeculativeSampler for DistributedEmbeddedMtpSampler<'_, S> {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        self.sampler.grammar_is_complete()
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
        self.sampler.prefix_is_complete(history)
    }

    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.sampler
            .process_logits(logits, temperature, history, stream)
    }

    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let sampled = self
            .sampler
            .sample_processed(logits, temperature, prng_state, stream)?;
        // Pipeline MTP publishes identical complete logits to every rank before
        // sampling. The speculative scheduler also advances the same sampler
        // and PRNG state on every rank, so sampling locally is the exact
        // synchronized operation. A world collective here is unsafe after
        // point-to-point pipeline traffic on MLX Ring because it can consume a
        // prior activation payload from the transport queue.
        Ok(sampled)
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.sampler.commit_token(processed_logits, token, stream)
    }
}

pub struct EmbeddedMtpOutput {
    pub logits: MlxTensor,
    pub hidden: MlxTensor,
    pub tokens: MlxTensor,
}

/// Family-owned model math and cache semantics used by one shared scheduler.
pub trait EmbeddedMtpTarget {
    type Cache: Clone;
    type DraftCache: Clone;

    fn prefill_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception>;
    fn verify_target(
        &mut self,
        tokens: &MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception>;
    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache;
    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache);
    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        cache.clone_from(checkpoint);
        Ok(())
    }
    fn draft_logits(
        &mut self,
        hidden: &MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor), Exception>;
    /// Optionally computes a fused block of base proposal logits. Architectures
    /// with token-conditioned block heads can adjust each row in
    /// `adjust_fused_draft_logits` as the scheduler samples it.
    fn fused_draft_logits(
        &mut self,
        _hidden: &MlxTensor,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut Self::DraftCache,
        _stream: &Stream,
    ) -> Result<Option<MlxTensor>, Exception> {
        Ok(None)
    }
    fn adjust_fused_draft_logits(
        &mut self,
        logits: MlxTensor,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        Ok(logits)
    }
    fn advance_draft_cache(
        &mut self,
        hidden: &MlxTensor,
        tokens: &MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn max_draft_tokens(&self) -> usize;
}

pub struct EmbeddedTargetState<C> {
    hidden: MlxTensor,
    draft_cache: C,
}

#[derive(Clone)]
pub struct EmbeddedDraftState<C: Clone> {
    hidden: MlxTensor,
    draft_cache: DraftStateTransaction<C>,
    depth: usize,
    fused_logits: Option<MlxTensor>,
    fused_cursor: usize,
}

pub struct EmbeddedVerification {
    output: EmbeddedMtpOutput,
    inputs: MlxTensor,
}

pub struct EmbeddedMtpExecutor<'a, T> {
    target: &'a mut T,
}

impl<'a, T: EmbeddedMtpTarget> EmbeddedMtpExecutor<'a, T> {
    pub fn new(target: &'a mut T) -> Self {
        Self { target }
    }

    fn state_at(
        output: &EmbeddedMtpOutput,
        row: i32,
        draft_cache: T::DraftCache,
        stream: &Stream,
    ) -> Result<EmbeddedTargetState<T::DraftCache>, Exception> {
        Ok(EmbeddedTargetState {
            hidden: MlxTensor::from_array(
                output
                    .hidden
                    .as_array()
                    .try_index_device((.., row..row + 1, ..), stream)?,
            ),
            draft_cache,
        })
    }
}

impl<T: EmbeddedMtpTarget> SpeculativeExecutor for EmbeddedMtpExecutor<'_, T> {
    type Input = MlxModelInput;
    type Cache = T::Cache;
    type TargetState = EmbeddedTargetState<T::DraftCache>;
    type DraftState = EmbeddedDraftState<T::DraftCache>;
    type CacheCheckpoint = T::Cache;
    type Verification = EmbeddedVerification;
    type Logits = Array;
    type Context<'a>
        = MtpExecutionStreams<'a>
    where
        Self: 'a;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = MtpComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        self.target.max_draft_tokens()
    }

    fn prefill<'context>(
        &mut self,
        input: MlxModelInput,
        cache: &mut Self::Cache,
        streams: MtpExecutionStreams<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Exception>
    where
        Self: 'context,
    {
        let stream = streams.target();
        input.with_borrowed(|input| {
            let output = self.target.prefill_target(input, cache, stream)?;
            let sequence = output.logits.as_array().dim(-2);
            if sequence == 0 {
                return Err(Exception::custom(
                    "embedded MTP input must contain at least one token",
                ));
            }
            let tokens = output.tokens.clone();
            self.target
                .prefill_draft_cache(&output, &tokens, cache, stream)?;
            let logits = output
                .logits
                .as_array()
                .try_index_device((.., sequence - 1, ..), stream)?;
            let state = Self::state_at(
                &output,
                sequence - 1,
                self.target.draft_cache(cache),
                stream,
            )?;
            Ok(SpeculativePrefill {
                logits,
                state,
                evaluated_tokens: sequence as usize,
            })
        })
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        let mut draft_cache = DraftStateTransaction::fork(&state.draft_cache);
        let fused_logits = self.target.fused_draft_logits(
            &state.hidden,
            last_token,
            proposal_capacity,
            draft_cache.draft_mut(),
            streams.draft(),
        )?;
        Ok(EmbeddedDraftState {
            hidden: state.hidden.clone(),
            draft_cache,
            depth: 0,
            fused_logits,
            fused_cursor: 0,
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let stream = streams.draft();
        if let Some(logits) = &state.fused_logits {
            if state.fused_cursor >= logits.as_array().dim(1) as usize {
                return Err(Exception::custom("fused MTP proposal block is exhausted"));
            }
            let row = state.fused_cursor as i32;
            state.fused_cursor += 1;
            let logits =
                MlxTensor::from_array(logits.as_array().try_index_device((.., row, ..), stream)?);
            return self
                .target
                .adjust_fused_draft_logits(logits, last_token, stream)
                .map(MlxTensor::into_array);
        }
        let (logits, hidden) = self.target.draft_logits(
            &state.hidden,
            last_token,
            state.depth,
            state.draft_cache.draft_mut(),
            stream,
        )?;
        state.hidden = hidden;
        state.depth += 1;
        Ok(logits.into_array())
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        cache.clone()
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Exception> {
        let mut inputs = Array::from_slice(input_tokens, &[1, input_tokens.len() as i32]);
        if streams.crosses_devices() {
            inputs = inputs.copy(streams.target())?;
        }
        let inputs = MlxTensor::from_array(inputs);
        let output = EmbeddedVerification {
            output: self
                .target
                .verify_target(&inputs, cache, streams.target())?,
            inputs,
        };
        let completion = MlxSpeculativeCompletion::submit([output.output.logits.as_array()])?;
        Ok(Submission { output, completion })
    }

    fn verification_logits<'a>(
        output: &Self::Verification,
        index: usize,
        streams: MtpExecutionStreams<'a>,
    ) -> Result<Array, Exception>
    where
        Self: 'a,
    {
        output
            .output
            .logits
            .as_array()
            .try_index_device((.., index as i32, ..), streams.target())
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        let stream = streams.target();
        let input_len = output.inputs.as_array().dim(1) as usize;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} embedded-MTP inputs from a block of {input_len}"
            )));
        }
        if verified_inputs > 1 {
            let accepted = verified_inputs as i32 - 1;
            let hidden = MlxTensor::from_array(
                output
                    .output
                    .hidden
                    .as_array()
                    .try_index_device((.., ..accepted, ..), stream)?,
            );
            let tokens = MlxTensor::from_array(
                output
                    .inputs
                    .as_array()
                    .try_index_device((.., 1..verified_inputs as i32), stream)?,
            );
            self.target.advance_draft_cache(
                &hidden,
                &tokens,
                draft_state.draft_cache.draft_mut(),
                stream,
            )?;
        }
        let (committed, replayed_tokens) = if verified_inputs == input_len {
            (output.output, 0)
        } else {
            T::restore_target_checkpoint(cache, &checkpoint, stream)?;
            let retained = output
                .inputs
                .as_array()
                .try_index_device((.., ..verified_inputs as i32), stream)?;
            (
                self.target
                    .verify_target(&MlxTensor::from_array(retained), cache, stream)?,
                verified_inputs,
            )
        };
        self.target
            .commit_draft_cache(cache, draft_state.draft_cache.draft());
        let state = Self::state_at(
            &committed,
            verified_inputs as i32 - 1,
            self.target.draft_cache(cache),
            stream,
        )?;
        Ok(SpeculativeCommit {
            state,
            replayed_tokens,
        })
    }
}
