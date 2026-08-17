//! Gemma 4 adapter for the architecture-independent MTP engine.

use std::{collections::HashMap, sync::Arc};

use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};
use safemlx_lm_core::{
    SpeculativeCommit, SpeculativeExecutionTopology, SpeculativeExecutor, SpeculativePrefill,
    Submission,
};

use crate::{
    api::{
        gemma4::{Cache, Gemma4Embedding, Gemma4StepOutput, Model as Gemma4Model},
        gemma4_assistant::{Gemma4AssistantDraftModel, Gemma4AssistantDraftState},
        input::ModelInput as RuntimeInput,
    },
    architectures::gemma4::layerwise::Gemma4LayerwiseModel,
    backend::mlx::{
        speculative::{MlxSpeculativeCompletion, MtpExecutionStreams},
        MlxModelInput,
    },
    core::generation::{
        FinishReason, GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent,
    },
    runtime::attention::AttentionPolicy,
    runtime::generation::sampler::SpeculativeSampler,
    runtime::generation::speculative::{self as mtp, MtpComponentTimings, MtpSemanticState},
};

#[derive(Clone)]
pub(crate) struct Gemma4TargetState {
    hidden: Array,
    shared_kv: Arc<HashMap<AttentionPolicy, (Array, Array)>>,
    cache_len: usize,
}

pub(crate) struct Gemma4Verification {
    output: Gemma4StepOutput,
    inputs: Array,
}

pub(crate) trait Gemma4MtpTarget {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception>;
    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception>;
    fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception>;
}

impl<T: Gemma4MtpTarget + ?Sized> Gemma4MtpTarget for Box<T> {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        (**self).prefill_mtp_target(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        (**self).verify_mtp_target(tokens, cache, stream)
    }

    fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        (**self).mtp_embedding_snapshot(stream, copy)
    }
}

impl Gemma4MtpTarget for Gemma4Model {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        self.mtp_embedding_snapshot(stream, copy)
    }
}

impl Gemma4MtpTarget for Gemma4LayerwiseModel {
    fn prefill_mtp_target(
        &mut self,
        input: RuntimeInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.prefill_mtp(input, cache, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Gemma4StepOutput, Exception> {
        self.verify_mtp(tokens, cache, stream)
    }

    fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        self.mtp_embedding_snapshot(stream, copy)
    }
}

/// Gemma 4 target plus an external Gemma assistant.
pub(crate) struct Gemma4MtpBackend<'a, T> {
    target: &'a mut T,
    assistant: &'a mut Gemma4AssistantDraftModel,
    draft_embedding: Option<Gemma4Embedding>,
}

impl<'a, T> Gemma4MtpBackend<'a, T> {
    pub(crate) fn new(target: &'a mut T, assistant: &'a mut Gemma4AssistantDraftModel) -> Self {
        Self {
            target,
            assistant,
            draft_embedding: None,
        }
    }

    fn state_at(
        output: &Gemma4StepOutput,
        row: i32,
        cache_len: usize,
        stream: &Stream,
    ) -> Result<Gemma4TargetState, Exception> {
        let hidden = output
            .hidden
            .try_index_device((.., row..row + 1, ..), stream)?;
        let retained = i32::try_from(cache_len)
            .map_err(|_| Exception::custom("Gemma 4 MTP state length exceeds i32"))?;
        let mut shared_kv = HashMap::with_capacity(output.shared_kv_states.len());
        for (kind, (keys, values)) in &output.shared_kv_states {
            let key_len = keys.dim(-2).min(retained);
            let value_len = values.dim(-2).min(retained);
            shared_kv.insert(
                *kind,
                (
                    keys.try_index_device((.., .., ..key_len, ..), stream)?,
                    values.try_index_device((.., .., ..value_len, ..), stream)?,
                ),
            );
        }
        Ok(Gemma4TargetState {
            hidden,
            shared_kv: Arc::new(shared_kv),
            cache_len,
        })
    }

    fn state_on_draft_stream(
        state: &Gemma4TargetState,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Gemma4TargetState, Exception> {
        if !streams.is_split() {
            return Ok(state.clone());
        }

        if !streams.crosses_devices() {
            let _completion = streams.wait_for_target_outputs(
                std::iter::once(&state.hidden).chain(
                    state
                        .shared_kv
                        .values()
                        .flat_map(|(keys, values)| [keys, values]),
                ),
            )?;
            return Ok(state.clone());
        }

        async_eval_with_event(
            std::iter::once(&state.hidden).chain(
                state
                    .shared_kv
                    .values()
                    .flat_map(|(keys, values)| [keys, values]),
            ),
        )?
        .synchronize()?;

        let hidden = state.hidden.copy(streams.draft())?;
        let shared_kv = state
            .shared_kv
            .iter()
            .map(|(kind, (keys, values))| {
                Ok((
                    *kind,
                    (keys.copy(streams.draft())?, values.copy(streams.draft())?),
                ))
            })
            .collect::<Result<HashMap<_, _>, Exception>>()?;
        async_eval_with_event(
            std::iter::once(&hidden)
                .chain(shared_kv.values().flat_map(|(keys, values)| [keys, values])),
        )?
        .synchronize()?;
        Ok(Gemma4TargetState {
            hidden,
            shared_kv: Arc::new(shared_kv),
            cache_len: state.cache_len,
        })
    }
}

impl<T: Gemma4MtpTarget> SpeculativeExecutor for Gemma4MtpBackend<'_, T> {
    type Input = MlxModelInput;
    type Cache = Cache;
    type TargetState = Gemma4TargetState;
    type DraftState = Gemma4AssistantDraftState;
    type CacheCheckpoint = Cache;
    type Verification = Gemma4Verification;
    type Logits = Array;
    type Context<'a>
        = MtpExecutionStreams<'a>
    where
        Self: 'a;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = MtpComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        self.assistant.block_size().saturating_sub(1)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
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
            let output = self.target.prefill_mtp_target(input, cache, stream)?;
            let sequence = output.logits.dim(-2);
            if sequence == 0 {
                return Err(Exception::custom(
                    "MTP input must contain at least one token",
                ));
            }
            let logits = output
                .logits
                .try_index_device((.., sequence - 1, ..), stream)?;
            let state = Self::state_at(&output, sequence - 1, cache.mtp_len(), stream)?;
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
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        let state = Self::state_on_draft_stream(state, streams)?;
        if self.draft_embedding.is_none() {
            let embedding = self
                .target
                .mtp_embedding_snapshot(streams.draft(), streams.crosses_devices())?;
            if streams.topology() == SpeculativeExecutionTopology::SameDeviceSplit {
                let _completion =
                    streams.wait_for_target_outputs(embedding.materialization_arrays())?;
            }
            self.draft_embedding = Some(embedding);
        }
        let offset = i32::try_from(state.cache_len)
            .map_err(|_| Exception::custom("Gemma 4 MTP cache offset exceeds i32"))?;
        Ok(self
            .assistant
            .begin_round(state.shared_kv, offset, &state.hidden))
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let stream = streams.draft();
        let embedding = self
            .draft_embedding
            .as_mut()
            .ok_or_else(|| Exception::custom("Gemma 4 draft embedding is not initialized"))?
            .forward(&Array::from_slice(&[last_token], &[1, 1]), stream)?
            .multiply(
                Array::from_f32((self.assistant.config.backbone_hidden_size as f32).sqrt()),
                stream,
            )?;
        self.assistant.draft_step(&embedding, state, stream)
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
        let output = Gemma4Verification {
            output: self
                .target
                .verify_mtp_target(&inputs, cache, streams.target())?,
            inputs,
        };
        let completion = MlxSpeculativeCompletion::submit([&output.output.logits])?;
        Ok(Submission { output, completion })
    }

    fn verification_logits(output: &Self::Verification) -> &Array {
        &output.output.logits
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        _draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        let stream = streams.target();
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} verified inputs from a block of {input_len}"
            )));
        }
        if verified_inputs == input_len {
            let state = Self::state_at(
                &output.output,
                verified_inputs.saturating_sub(1) as i32,
                cache.mtp_len(),
                stream,
            )?;
            return Ok(SpeculativeCommit {
                state,
                replayed_tokens: 0,
            });
        }

        let retained_len = checkpoint
            .mtp_len()
            .checked_add(verified_inputs)
            .ok_or_else(|| Exception::custom("Gemma 4 MTP retained cache length overflow"))?;
        let expected_len = checkpoint
            .mtp_len()
            .checked_add(input_len)
            .ok_or_else(|| Exception::custom("Gemma 4 MTP verified cache length overflow"))?;
        if cache.mtp_len() != expected_len {
            return Err(Exception::custom(format!(
                "Gemma 4 MTP verification cache has length {}, expected {expected_len}",
                cache.mtp_len()
            )));
        }
        cache.truncate_mtp(retained_len, stream)?;
        let state = Self::state_at(
            &output.output,
            verified_inputs.saturating_sub(1) as i32,
            cache.mtp_len(),
            stream,
        )?;
        Ok(SpeculativeCommit {
            state,
            replayed_tokens: 0,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_streams_and_callback_and_options<T, S, F>(
    target: &mut T,
    assistant: &mut Gemma4AssistantDraftModel,
    cache: &mut Cache,
    input: RuntimeInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    streams: MtpExecutionStreams<'_>,
    options: MtpSchedulerOptions,
    on_token: F,
) -> Result<(Vec<u32>, mtp::MtpStats), Exception>
where
    T: Gemma4MtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(u32) -> Result<(), Exception>,
{
    let mut backend = Gemma4MtpBackend::new(target, assistant);
    mtp::generate_with_streams_and_callback_and_options(
        &mut backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        streams,
        options,
        on_token,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_semantics_and_options<T, S, F>(
    target: &mut T,
    assistant: &mut Gemma4AssistantDraftModel,
    cache: &mut Cache,
    input: RuntimeInput<'_>,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &mut S,
    semantic: Box<dyn MtpSemanticState>,
    cancellation: GenerationCancellationToken,
    streams: MtpExecutionStreams<'_>,
    options: MtpSchedulerOptions,
    on_event: F,
) -> Result<(Vec<u32>, mtp::MtpStats, FinishReason), Exception>
where
    T: Gemma4MtpTarget,
    S: SpeculativeSampler + Clone,
    F: FnMut(SemanticEvent),
{
    let mut backend = Gemma4MtpBackend::new(target, assistant);
    mtp::generate_with_semantics_and_options(
        &mut backend,
        cache,
        input,
        config,
        prng_key,
        sampler,
        semantic,
        cancellation,
        streams,
        options,
        on_event,
    )
}

#[cfg(test)]
mod tests {
    use safemlx::{Device, DeviceType, ExecutionContext};

    use super::*;

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn target_state_copies_from_gpu_to_cpu_draft_stream() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let hidden = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2])
            .copy(target.stream())
            .unwrap();
        let keys = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 1, 2])
            .copy(target.stream())
            .unwrap();
        let values = Array::from_slice(&[5.0f32, 6.0], &[1, 1, 1, 2])
            .copy(target.stream())
            .unwrap();
        let state = Gemma4TargetState {
            hidden,
            shared_kv: Arc::new(HashMap::from([(AttentionPolicy::Full, (keys, values))])),
            cache_len: 9,
        };

        let copied = Gemma4MtpBackend::<Gemma4Model>::state_on_draft_stream(
            &state,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
        )
        .unwrap();

        assert_eq!(copied.cache_len, 9);
        assert_eq!(
            copied.hidden.evaluated().unwrap().as_slice::<f32>(),
            &[1.0, 2.0]
        );
        let (keys, values) = &copied.shared_kv[&AttentionPolicy::Full];
        assert_eq!(keys.evaluated().unwrap().as_slice::<f32>(), &[3.0, 4.0]);
        assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn target_state_shares_storage_between_streams_on_the_same_gpu() {
        let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let draft = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let hidden = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2])
            .copy(target.stream())
            .unwrap();
        let keys = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 1, 2])
            .copy(target.stream())
            .unwrap();
        let values = Array::from_slice(&[5.0f32, 6.0], &[1, 1, 1, 2])
            .copy(target.stream())
            .unwrap();
        async_eval_with_event([&hidden, &keys, &values])
            .unwrap()
            .synchronize()
            .unwrap();
        let hidden_ptr = hidden.evaluated().unwrap().as_slice::<f32>().as_ptr();
        let keys_ptr = keys.evaluated().unwrap().as_slice::<f32>().as_ptr();
        let values_ptr = values.evaluated().unwrap().as_slice::<f32>().as_ptr();
        let shared_kv = Arc::new(HashMap::from([(AttentionPolicy::Full, (keys, values))]));
        let state = Gemma4TargetState {
            hidden,
            shared_kv: shared_kv.clone(),
            cache_len: 9,
        };

        let shared = Gemma4MtpBackend::<Gemma4Model>::state_on_draft_stream(
            &state,
            MtpExecutionStreams::new(target.stream(), draft.stream()).unwrap(),
        )
        .unwrap();

        assert_eq!(shared.cache_len, 9);
        assert!(Arc::ptr_eq(&shared.shared_kv, &shared_kv));
        assert_eq!(
            shared
                .hidden
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .as_ptr(),
            hidden_ptr
        );
        let (keys, values) = &shared.shared_kv[&AttentionPolicy::Full];
        assert_eq!(
            keys.evaluated().unwrap().as_slice::<f32>().as_ptr(),
            keys_ptr
        );
        assert_eq!(
            values.evaluated().unwrap().as_slice::<f32>().as_ptr(),
            values_ptr
        );
    }
}
