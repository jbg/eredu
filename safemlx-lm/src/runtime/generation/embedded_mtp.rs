//! Shared speculative backend for checkpoint-embedded prediction heads.

use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters},
    nn,
    ops::indexing::TryIndexOp,
    quantization::MaybeQuantized,
    Array, Stream,
};

use crate::{
    api::input::ModelInput,
    runtime::generation::{
        sampler::SpeculativeSampler,
        speculative::{
            self, MtpBackend, MtpCommit, MtpConfig, MtpExecutionStreams, MtpPrefill,
            MtpSchedulerOptions, MtpSemanticState,
        },
        streaming::{FinishReason, GenerationCancellationToken, SemanticEvent},
    },
};

/// Sampler wrapper that keeps PRNG and grammar state identical on every rank
/// while selecting one topology-owned sampling coordinate.
#[derive(Clone)]
pub(crate) struct DistributedEmbeddedMtpSampler<'a, S> {
    sampler: S,
    sampling_rank: usize,
    group: &'a Group,
}

impl<'a, S> DistributedEmbeddedMtpSampler<'a, S> {
    pub(crate) fn new(sampler: S, sampling_rank: usize, group: &'a Group) -> Result<Self, Error> {
        if sampling_rank >= group.size() {
            return Err(Error::Parallel(format!(
                "embedded MTP sampling rank {sampling_rank} is outside world size {}",
                group.size()
            )));
        }
        Ok(Self {
            sampler,
            sampling_rank,
            group,
        })
    }

    pub(crate) fn into_inner(self) -> S {
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
        let selected = if self.group.rank() == self.sampling_rank {
            sampled
        } else {
            safemlx::ops::zeros_dtype(sampled.shape(), sampled.dtype(), stream)?
        };
        distributed::all_sum(&selected, self.group, stream)
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

use crate::{
    error::Error,
    nn::parallel::VocabParallelLmHead,
    runtime::{
        checkpoint::quantization::WeightQuantization,
        distributed::parallel::{ParallelBuildContext, ParallelExecutionContext},
    },
};

/// One vocabulary projection whose parameter tree stays stable when TP turns
/// the physical storage into rank-local rows.
///
/// Keeping the active representation behind one module boundary means static
/// residency, bounded conversion, and pipeline loading continue to address the
/// same semantic parameter names in replicated and distributed execution.
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedMtpVocabHead {
    ordinary: MaybeQuantized<nn::Linear>,
    parallel: Option<VocabParallelLmHead>,
    input_dims: i32,
    vocabulary: usize,
    quantization: Option<WeightQuantization>,
}

impl EmbeddedMtpVocabHead {
    pub(crate) fn new(
        input_dims: i32,
        vocabulary: usize,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            ordinary: crate::api::common::linear::unloaded_maybe_quantized_linear(
                input_dims,
                i32::try_from(vocabulary)
                    .map_err(|_| Error::Parallel("MTP vocabulary exceeds i32".into()))?,
                false,
                quantization,
                stream,
            )?,
            parallel: None,
            input_dims,
            vocabulary,
            quantization,
        })
    }

    pub(crate) fn configure_parallel(
        &mut self,
        context: ParallelBuildContext,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel = Some(VocabParallelLmHead::unloaded(
            self.input_dims,
            self.vocabulary,
            self.quantization,
            context,
            stream,
        )?);
        Ok(())
    }

    pub(crate) fn register(
        &self,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        prefix: &str,
    ) -> Result<(), Error> {
        planner.register(crate::nn::parallel::vocab_lm_head_parameter_group(
            &self.ordinary,
            prefix,
            self.input_dims,
            self.vocabulary,
            false,
        )?)
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self.parallel.as_mut(), execution) {
            (Some(head), Some(execution)) if execution.is_tensor_parallel() => head
                .forward(hidden, execution)
                .and_then(|output| output.all_gather(execution))
                .map_err(|error| Exception::custom(error.to_string())),
            (None, None) => self.ordinary.forward(hidden, stream),
            (Some(_), None) => Err(Exception::custom(
                "TP-sharded embedded MTP head requires a tensor execution context",
            )),
            (None, Some(execution)) if execution.is_tensor_parallel() => Err(Exception::custom(
                "embedded MTP head was not configured for tensor parallelism",
            )),
            (None, Some(_)) => self.ordinary.forward(hidden, stream),
            (Some(head), Some(execution)) => head
                .forward(hidden, execution)
                .and_then(|output| output.all_gather(execution))
                .map_err(|error| Exception::custom(error.to_string())),
        }
    }

    fn active(&self) -> &dyn ModuleParameters {
        self.parallel
            .as_ref()
            .map_or(&self.ordinary as &dyn ModuleParameters, |head| head)
    }

    fn active_mut(&mut self) -> &mut dyn ModuleParameters {
        self.parallel
            .as_mut()
            .map_or(&mut self.ordinary as &mut dyn ModuleParameters, |head| head)
    }
}

impl ModuleParameters for EmbeddedMtpVocabHead {
    fn num_parameters(&self) -> usize {
        self.active().num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.active().parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.active_mut().parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.active().trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.active_mut().freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.active_mut().unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.active().all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.active().any_frozen()
    }
}

pub(crate) struct EmbeddedMtpOutput {
    pub(crate) logits: Array,
    pub(crate) hidden: Array,
    pub(crate) tokens: Array,
}

/// Family-owned model math and cache semantics used by one shared scheduler.
pub(crate) trait EmbeddedMtpTarget {
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
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception>;
    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn draft_cache(cache: &Self::Cache) -> Self::DraftCache;
    fn commit_draft_cache(cache: &mut Self::Cache, draft: &Self::DraftCache);
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
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception>;
    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn max_draft_tokens(&self) -> usize;
}

pub(crate) struct EmbeddedTargetState<C> {
    hidden: Array,
    draft_cache: C,
}

#[derive(Clone)]
pub(crate) struct EmbeddedDraftState<C> {
    hidden: Array,
    draft_cache: C,
    depth: usize,
}

pub(crate) struct EmbeddedVerification {
    output: EmbeddedMtpOutput,
    inputs: Array,
}

pub(crate) struct EmbeddedMtpBackend<'a, T> {
    target: &'a mut T,
}

impl<'a, T: EmbeddedMtpTarget> EmbeddedMtpBackend<'a, T> {
    pub(crate) fn new(target: &'a mut T) -> Self {
        Self { target }
    }

    fn state_at(
        output: &EmbeddedMtpOutput,
        row: i32,
        draft_cache: T::DraftCache,
        stream: &Stream,
    ) -> Result<EmbeddedTargetState<T::DraftCache>, Exception> {
        Ok(EmbeddedTargetState {
            hidden: output
                .hidden
                .try_index_device((.., row..row + 1, ..), stream)?,
            draft_cache,
        })
    }
}

impl<T: EmbeddedMtpTarget> MtpBackend for EmbeddedMtpBackend<'_, T> {
    type Cache = T::Cache;
    type TargetState = EmbeddedTargetState<T::DraftCache>;
    type DraftState = EmbeddedDraftState<T::DraftCache>;
    type CacheCheckpoint = T::Cache;
    type Verification = EmbeddedVerification;

    fn max_draft_tokens(&self) -> usize {
        self.target.max_draft_tokens()
    }

    fn prefill(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
        let output = self.target.prefill_target(input, cache, stream)?;
        let sequence = output.logits.dim(-2);
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
            .try_index_device((.., sequence - 1, ..), stream)?;
        let state = Self::state_at(&output, sequence - 1, T::draft_cache(cache), stream)?;
        Ok(MtpPrefill {
            logits,
            state,
            evaluated_tokens: sequence as usize,
        })
    }

    fn begin_draft(
        &mut self,
        state: &Self::TargetState,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Self::DraftState, Exception> {
        Ok(EmbeddedDraftState {
            hidden: state.hidden.clone(),
            draft_cache: state.draft_cache.clone(),
            depth: 0,
        })
    }

    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let (logits, hidden) = self.target.draft_logits(
            &state.hidden,
            last_token,
            state.depth,
            &mut state.draft_cache,
            stream,
        )?;
        state.hidden = hidden;
        state.depth += 1;
        Ok(logits)
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        cache.clone()
    }

    fn verify(
        &mut self,
        input_tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<Self::Verification, Exception> {
        Ok(EmbeddedVerification {
            output: self.target.verify_target(input_tokens, cache, stream)?,
            inputs: input_tokens.clone(),
        })
    }

    fn verification_logits(output: &Self::Verification) -> &Array {
        &output.output.logits
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        stream: &Stream,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} embedded-MTP inputs from a block of {input_len}"
            )));
        }
        if verified_inputs > 1 {
            let accepted = verified_inputs as i32 - 1;
            let hidden = output
                .output
                .hidden
                .try_index_device((.., ..accepted, ..), stream)?;
            let tokens = output
                .inputs
                .try_index_device((.., 1..verified_inputs as i32), stream)?;
            self.target.advance_draft_cache(
                &hidden,
                &tokens,
                &mut draft_state.draft_cache,
                stream,
            )?;
        }
        let (committed, replayed_tokens) = if verified_inputs == input_len {
            (output.output, 0)
        } else {
            T::restore_target_checkpoint(cache, &checkpoint, stream)?;
            let retained = output
                .inputs
                .try_index_device((.., ..verified_inputs as i32), stream)?;
            (
                self.target.verify_target(&retained, cache, stream)?,
                verified_inputs,
            )
        };
        T::commit_draft_cache(cache, &draft_state.draft_cache);
        let state = Self::state_at(
            &committed,
            verified_inputs as i32 - 1,
            T::draft_cache(cache),
            stream,
        )?;
        Ok(MtpCommit {
            state,
            replayed_tokens,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_callback<T, S, F>(
    target: &mut T,
    cache: &mut T::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
    on_token: F,
) -> Result<(Vec<u32>, speculative::MtpStats), Exception>
where
    T: EmbeddedMtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception>,
{
    speculative::generate_with_callback(
        &mut EmbeddedMtpBackend::new(target),
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
        on_token,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_semantics_and_options<T, S, F>(
    target: &mut T,
    cache: &mut T::Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    semantic: Box<dyn MtpSemanticState>,
    cancellation: GenerationCancellationToken,
    stream: &Stream,
    options: MtpSchedulerOptions,
    on_event: F,
) -> Result<(Vec<u32>, speculative::MtpStats, FinishReason), Exception>
where
    T: EmbeddedMtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(SemanticEvent),
{
    speculative::generate_with_semantics_and_options(
        &mut EmbeddedMtpBackend::new(target),
        cache,
        input,
        config,
        prng_key,
        sampler,
        semantic,
        cancellation,
        MtpExecutionStreams::single(stream),
        options,
        on_event,
    )
}
