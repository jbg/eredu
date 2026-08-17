//! Qwen3-Next and Qwen3.5/3.6 adapters for embedded MTP layers.

use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};
use safemlx_lm_core::{SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission};

use crate::{
    api::{
        input::{self, ModelInput},
        qwen3_5::{Cache, LayerCache, Model, QwenMtpStepOutput},
    },
    backend::mlx::{
        speculative::{MlxSpeculativeCompletion, MtpExecutionStreams},
        MlxModelInput,
    },
    core::generation::{
        FinishReason, GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent,
    },
    runtime::generation::sampler::SpeculativeSampler,
    runtime::generation::speculative::{self as mtp, MtpComponentTimings, MtpSemanticState},
};

pub(crate) trait QwenMtpTarget {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>;
    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>;
    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception>;
    fn mtp_layer_count(&self) -> usize;
}

impl QwenMtpTarget for Model {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_mtp_head(hidden, tokens, cache, stream)
    }

    fn mtp_layer_count(&self) -> usize {
        self.mtp_len()
    }
}

impl QwenMtpTarget for crate::architectures::qwen::hybrid::layerwise::QwenHybridLayerwiseModel {
    fn prefill_mtp_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_mtp_head(hidden, tokens, cache, stream)
    }

    fn mtp_layer_count(&self) -> usize {
        self.mtp_len()
    }
}

pub(crate) struct QwenTargetState {
    hidden: Array,
    mtp_cache: Vec<LayerCache>,
}

#[derive(Clone)]
pub(crate) struct QwenDraftState {
    hidden: Array,
    mtp_cache: Vec<LayerCache>,
}

pub(crate) struct QwenVerification {
    output: QwenMtpStepOutput,
    inputs: Array,
}

pub(crate) struct QwenMtpBackend<'a, T> {
    target: &'a mut T,
}

impl<'a, T: QwenMtpTarget> QwenMtpBackend<'a, T> {
    pub(crate) fn new(target: &'a mut T) -> Self {
        Self { target }
    }

    fn state_at(
        output: &QwenMtpStepOutput,
        row: i32,
        mtp_cache: &[LayerCache],
        stream: &Stream,
    ) -> Result<QwenTargetState, Exception> {
        Ok(QwenTargetState {
            hidden: output
                .hidden
                .try_index_device((.., row..row + 1, ..), stream)?,
            mtp_cache: mtp_cache.to_vec(),
        })
    }

    fn prefill_draft_cache(
        &mut self,
        output: &QwenMtpStepOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next_tokens = tokens.try_index_device((.., 1..), stream)?;
        let _ = self.target.forward_mtp_drafter(
            &hidden,
            &next_tokens,
            &mut cache.mtp_layers,
            stream,
        )?;
        Ok(())
    }
}

impl<T: QwenMtpTarget> SpeculativeExecutor for QwenMtpBackend<'_, T> {
    type Input = MlxModelInput;
    type Cache = Cache;
    type TargetState = QwenTargetState;
    type DraftState = QwenDraftState;
    type CacheCheckpoint = Cache;
    type Verification = QwenVerification;
    type Logits = Array;
    type Context<'a>
        = MtpExecutionStreams<'a>
    where
        Self: 'a;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = MtpComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        usize::from(self.target.mtp_layer_count() > 0)
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
            let tokens = input::text_token_ids(input, stream)?;
            let output = self.target.prefill_mtp_target(input, cache, stream)?;
            let sequence = output.logits.dim(-2);
            if sequence == 0 {
                return Err(Exception::custom(
                    "Qwen MTP input must contain at least one token",
                ));
            }
            self.prefill_draft_cache(&output, &tokens, cache, stream)?;
            let logits = output
                .logits
                .try_index_device((.., sequence - 1, ..), stream)?;
            let state = Self::state_at(&output, sequence - 1, &cache.mtp_layers, stream)?;
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
        _last_token: u32,
        _proposal_capacity: usize,
        _streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        Ok(QwenDraftState {
            hidden: state.hidden.clone(),
            mtp_cache: state.mtp_cache.clone(),
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let stream = streams.draft();
        let token = Array::from_slice(&[last_token], &[1, 1]);
        self.target
            .forward_mtp_drafter(&state.hidden, &token, &mut state.mtp_cache, stream)
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
        let output = QwenVerification {
            output: self
                .target
                .verify_mtp_target(&inputs, cache, streams.target())?,
            inputs,
        };
        let completion = MlxSpeculativeCompletion::submit([&output.output.logits])?;
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
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} verified Qwen inputs from a block of {input_len}"
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
            let _ = self.target.forward_mtp_drafter(
                &hidden,
                &tokens,
                &mut draft_state.mtp_cache,
                stream,
            )?;
        }

        let (committed, replayed_tokens) = if verified_inputs == input_len {
            (output.output, 0)
        } else {
            *cache = checkpoint;
            let retained = output
                .inputs
                .try_index_device((.., ..verified_inputs as i32), stream)?;
            (
                self.target.verify_mtp_target(&retained, cache, stream)?,
                verified_inputs,
            )
        };
        cache.mtp_layers.clone_from(&draft_state.mtp_cache);
        let state = Self::state_at(
            &committed,
            verified_inputs as i32 - 1,
            &cache.mtp_layers,
            stream,
        )?;
        Ok(SpeculativeCommit {
            state,
            replayed_tokens,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Used directly by device-gated backend parity tests.
pub(crate) fn generate<T: QwenMtpTarget, S: SpeculativeSampler + Clone>(
    target: &mut T,
    cache: &mut Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception> {
    generate_with_callback(
        target,
        cache,
        input,
        config,
        prng_key,
        sampler,
        stream,
        |_| Ok(()),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_callback<T, S, F>(
    target: &mut T,
    cache: &mut Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    stream: &Stream,
    on_token: F,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception>
where
    T: QwenMtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception>,
{
    let mut backend = QwenMtpBackend::new(target);
    mtp::generate_with_callback(
        &mut backend,
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
    cache: &mut Cache,
    input: ModelInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    semantic: Box<dyn MtpSemanticState>,
    cancellation: GenerationCancellationToken,
    stream: &Stream,
    options: MtpSchedulerOptions,
    on_event: F,
) -> Result<(Vec<u32>, mtp::MtpStats, FinishReason), Exception>
where
    T: QwenMtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(SemanticEvent),
{
    let mut backend = QwenMtpBackend::new(target);
    mtp::generate_with_semantics_and_options(
        &mut backend,
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
