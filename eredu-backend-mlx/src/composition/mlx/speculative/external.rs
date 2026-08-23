//! Neutral external-assistant adapters for the portable speculative engine.

use std::{collections::HashMap, sync::Arc};

use eredu_core::{
    AttentionPolicy, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array,
};

use crate::{
    backend::runtime::{
        cache::state::{MlxHybridState, MlxKeyValueState},
        media::input::ModelInput,
    },
    composition::{
        gemma4::{Gemma4AssistantModel, Gemma4Model, Gemma4MtpOutput},
        mlx::{
            speculative::{
                scheduler::MtpComponentTimings, MlxSpeculativeCompletion, MtpExecutionStreams,
            },
            MlxModelInput,
        },
        muse_glimmer::{MuseGlimmerDFlashModel, MuseGlimmerModel, MuseGlimmerMtpOutput},
    },
    MlxTensor,
};

#[derive(Clone)]
pub struct Gemma4TargetState {
    hidden: Array,
    shared_kv: Arc<HashMap<AttentionPolicy, (Array, Array)>>,
    cache_len: i32,
}

#[derive(Clone)]
pub struct MuseTargetState {
    pending_context: Option<Array>,
    draft_context: Option<eredu_architectures::muse_glimmer::DFlashContext<MlxTensor>>,
    cache_len: i32,
}

#[derive(Clone)]
pub struct MuseDraftState {
    logits: Array,
    cursor: usize,
    proposal_capacity: usize,
    draft_context: eredu_architectures::muse_glimmer::DFlashContext<MlxTensor>,
    cache_len: i32,
}

pub struct MuseVerification {
    output: MuseGlimmerMtpOutput,
    inputs: Array,
}

/// Ordinary neutral Muse-Glimmer target plus its neutral DFlash assistant.
pub struct MuseGlimmerExternalExecutor<'a> {
    target: &'a mut MuseGlimmerModel,
    assistant: &'a mut MuseGlimmerDFlashModel,
}

impl<'a> MuseGlimmerExternalExecutor<'a> {
    pub fn new(
        target: &'a mut MuseGlimmerModel,
        assistant: &'a mut MuseGlimmerDFlashModel,
    ) -> Self {
        Self { target, assistant }
    }

    fn retained_context(
        &self,
        context: Array,
        stream: &safemlx::Stream,
    ) -> Result<Array, Exception> {
        let length = context.dim(1);
        let window = self.assistant.config.sliding_window;
        if length <= window {
            Ok(context)
        } else {
            context.try_index_device((.., length - window.., ..), stream)
        }
    }

    fn assemble_context(
        &self,
        states: &[MlxTensor],
        stream: &safemlx::Stream,
    ) -> Result<Array, Exception> {
        self.assistant
            .assemble_target_states(states, stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn state_on_draft_stream(
        state: &MuseTargetState,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MuseTargetState, Exception> {
        if !streams.is_split() {
            return Ok(state.clone());
        }
        if !streams.crosses_devices() {
            if let Some(pending) = state.pending_context.as_ref() {
                let _completion = streams.wait_for_target_outputs([pending])?;
            }
            return Ok(state.clone());
        }
        let pending_context = state
            .pending_context
            .as_ref()
            .map(|pending| {
                async_eval_with_event([pending])?.synchronize()?;
                let copied = pending.copy(streams.draft())?;
                async_eval_with_event([&copied])?.synchronize()?;
                Ok::<Array, Exception>(copied)
            })
            .transpose()?;
        Ok(MuseTargetState {
            pending_context,
            draft_context: state.draft_context.clone(),
            cache_len: state.cache_len,
        })
    }

    fn target_embeddings_on_draft(
        &mut self,
        ids: &Array,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let hidden = self
            .target
            .embed_dflash_tokens(&MlxTensor::from_array(ids.clone()), streams.target())
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_array();
        if !streams.is_split() {
            return Ok(hidden);
        }
        if !streams.crosses_devices() {
            let _completion = streams.wait_for_target_outputs([&hidden])?;
            return Ok(hidden);
        }
        async_eval_with_event([&hidden])?.synchronize()?;
        let copied = hidden.copy(streams.draft())?;
        async_eval_with_event([&copied])?.synchronize()?;
        Ok(copied)
    }

    fn target_logits_on_draft(
        &mut self,
        states: &Array,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let states = if !streams.is_split() {
            states.clone()
        } else if !streams.crosses_devices() {
            let _completion = streams.wait_for_draft_outputs([states])?;
            states.clone()
        } else {
            async_eval_with_event([states])?.synchronize()?;
            let copied = states.copy(streams.target())?;
            async_eval_with_event([&copied])?.synchronize()?;
            copied
        };
        let logits = self
            .target
            .project_dflash_logits(&MlxTensor::from_array(states.clone()), streams.target())
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_array();
        if !streams.is_split() {
            return Ok(logits);
        }
        if !streams.crosses_devices() {
            let _completion = streams.wait_for_target_outputs([&logits])?;
            return Ok(logits);
        }
        async_eval_with_event([&logits])?.synchronize()?;
        let copied = logits.copy(streams.draft())?;
        async_eval_with_event([&copied])?.synchronize()?;
        Ok(copied)
    }

    fn target_state(
        &self,
        output: &MuseGlimmerMtpOutput,
        draft_context: Option<eredu_architectures::muse_glimmer::DFlashContext<MlxTensor>>,
        cache_len: i32,
        stream: &safemlx::Stream,
    ) -> Result<MuseTargetState, Exception> {
        let pending = self.assemble_context(&output.target_states, stream)?;
        Ok(MuseTargetState {
            pending_context: Some(self.retained_context(pending, stream)?),
            draft_context,
            cache_len,
        })
    }
}

impl SpeculativeExecutor for MuseGlimmerExternalExecutor<'_> {
    type Input = MlxModelInput;
    type Cache = MlxKeyValueState;
    type TargetState = MuseTargetState;
    type DraftState = MuseDraftState;
    type CacheCheckpoint = MlxKeyValueState;
    type Verification = MuseVerification;
    type Logits = Array;
    type Context<'a>
        = MtpExecutionStreams<'a>
    where
        Self: 'a;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = MtpComponentTimings;
    type Error = Exception;

    fn max_proposals(&self) -> usize {
        self.assistant.config.block_size.saturating_sub(1).min(15)
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
        input.with_borrowed(|input: ModelInput<'_>| {
            let layers = self.assistant.target_layer_ids().to_vec();
            let output = self
                .target
                .forward_input_with_taps(input, cache, &layers, streams.target())
                .map_err(|error| Exception::custom(error.to_string()))?;
            let sequence = output.logits.as_array().dim(1);
            if sequence <= 0 {
                return Err(Exception::custom("Muse-Glimmer DFlash input is empty"));
            }
            Ok(SpeculativePrefill {
                logits: output
                    .logits
                    .as_array()
                    .try_index_device((.., sequence - 1, ..), streams.target())?,
                state: self.target_state(&output, None, cache.offset(), streams.target())?,
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
        let maximum = self.max_proposals();
        if proposal_capacity == 0 || proposal_capacity > maximum {
            return Err(Exception::custom(format!(
                "Muse-Glimmer DFlash proposal capacity must be between 1 and {maximum}"
            )));
        }
        let state = Self::state_on_draft_stream(state, streams)?;
        let draft_context = match state.pending_context.as_ref() {
            Some(pending) => self
                .assistant
                .update_context(
                    state.draft_context,
                    &MlxTensor::from_array(pending.clone()),
                    state.cache_len,
                    streams.draft(),
                )
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => state.draft_context.ok_or_else(|| {
                Exception::custom("DFlash proposal has neither pending nor committed context")
            })?,
        };
        let ids = dflash_block_token_ids(
            last_token,
            self.assistant.config.mask_token_id,
            proposal_capacity,
        );
        let ids = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let embeddings = self.target_embeddings_on_draft(&ids, streams)?;
        let states = self
            .assistant
            .proposal_states(
                &MlxTensor::from_array(embeddings),
                &draft_context,
                state.cache_len,
                streams.draft(),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let logits = self.target_logits_on_draft(states.as_array(), streams)?;
        Ok(MuseDraftState {
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
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        if state.cursor >= state.proposal_capacity {
            return Err(Exception::custom("Muse-Glimmer DFlash block is exhausted"));
        }
        let row = state.cursor as i32;
        state.cursor += 1;
        state
            .logits
            .try_index_device((.., row, ..), streams.draft())
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        cache
            .deep_clone_state()
            .expect("validated Muse-Glimmer speculative state must be forkable")
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
        let layers = self.assistant.target_layer_ids().to_vec();
        let output = self
            .target
            .verify_dflash(
                &MlxTensor::from_array(inputs.clone()),
                cache,
                &layers,
                streams.target(),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let completion = MlxSpeculativeCompletion::submit(
            std::iter::once(output.logits.as_array())
                .chain(output.target_states.iter().map(MlxTensor::as_array)),
        )?;
        Ok(Submission {
            output: MuseVerification { output, inputs },
            completion,
        })
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
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs > input_len
            || draft_state.cache_len != checkpoint.offset()
            || draft_state.draft_context.end != checkpoint.offset()
        {
            return Err(Exception::custom(
                "Muse-Glimmer DFlash verification/checkpoint state mismatch",
            ));
        }
        if verified_inputs == 0 {
            cache.restore_checkpoint(&checkpoint, streams.target())?;
            return Ok(SpeculativeCommit {
                state: MuseTargetState {
                    pending_context: None,
                    draft_context: Some(draft_state.draft_context),
                    cache_len: checkpoint.offset(),
                },
                replayed_tokens: 0,
            });
        }
        let (retained, replayed_tokens) = if verified_inputs == input_len {
            (output.output, 0)
        } else {
            cache.restore_checkpoint(&checkpoint, streams.target())?;
            let inputs = output
                .inputs
                .try_index_device((.., ..verified_inputs as i32), streams.target())?;
            let layers = self.assistant.target_layer_ids().to_vec();
            let replayed = self
                .target
                .verify_dflash(
                    &MlxTensor::from_array(inputs),
                    cache,
                    &layers,
                    streams.target(),
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
            (replayed, verified_inputs)
        };
        Ok(SpeculativeCommit {
            state: self.target_state(
                &retained,
                Some(draft_state.draft_context),
                cache.offset(),
                streams.target(),
            )?,
            replayed_tokens,
        })
    }
}

fn dflash_block_token_ids(anchor: u32, mask: u32, proposal_capacity: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(proposal_capacity + 1);
    ids.push(anchor);
    ids.resize(proposal_capacity + 1, mask);
    ids
}

pub struct Gemma4Verification {
    output: Gemma4MtpOutput,
    inputs: Array,
}

/// Ordinary neutral Gemma target plus its neutral external assistant.
pub struct Gemma4ExternalExecutor<'a> {
    target: &'a mut Gemma4Model,
    assistant: &'a mut Gemma4AssistantModel,
}

impl<'a> Gemma4ExternalExecutor<'a> {
    pub fn new(target: &'a mut Gemma4Model, assistant: &'a mut Gemma4AssistantModel) -> Self {
        Self { target, assistant }
    }

    fn state_at(
        output: &Gemma4MtpOutput,
        row: i32,
        cache_len: i32,
        stream: &safemlx::Stream,
    ) -> Result<Gemma4TargetState, Exception> {
        let hidden = output
            .hidden
            .as_array()
            .try_index_device((.., row..row + 1, ..), stream)?;
        let mut shared_kv = HashMap::with_capacity(output.shared_kv.len());
        for (policy, (keys, values)) in &output.shared_kv {
            let key_len = keys.as_array().dim(-2).min(cache_len);
            let value_len = values.as_array().dim(-2).min(cache_len);
            shared_kv.insert(
                *policy,
                (
                    keys.as_array()
                        .try_index_device((.., .., ..key_len, ..), stream)?,
                    values
                        .as_array()
                        .try_index_device((.., .., ..value_len, ..), stream)?,
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
        let values = || {
            std::iter::once(&state.hidden).chain(
                state
                    .shared_kv
                    .values()
                    .flat_map(|(keys, values)| [keys, values]),
            )
        };
        if !streams.crosses_devices() {
            let _completion = streams.wait_for_target_outputs(values())?;
            return Ok(state.clone());
        }
        async_eval_with_event(values())?.synchronize()?;
        let hidden = state.hidden.copy(streams.draft())?;
        let shared_kv = state
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                Ok((
                    *policy,
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

    fn proposal_embedding(
        &mut self,
        token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let embedding = self
            .target
            .embed_mtp_token(token, streams.target())
            .map_err(|error| Exception::custom(error.to_string()))?
            .into_array();
        if !streams.is_split() {
            return Ok(embedding);
        }
        if !streams.crosses_devices() {
            let _completion = streams.wait_for_target_outputs([&embedding])?;
            return Ok(embedding);
        }
        async_eval_with_event([&embedding])?.synchronize()?;
        let embedding = embedding.copy(streams.draft())?;
        async_eval_with_event([&embedding])?.synchronize()?;
        Ok(embedding)
    }
}

impl SpeculativeExecutor for Gemma4ExternalExecutor<'_> {
    type Input = MlxModelInput;
    type Cache = MlxHybridState;
    type TargetState = Gemma4TargetState;
    type DraftState = eredu_architectures::gemma4::AssistantState<MlxTensor>;
    type CacheCheckpoint = MlxHybridState;
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
        self.assistant.max_proposals()
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
        input.with_borrowed(|input: ModelInput<'_>| {
            let output = self
                .target
                .prefill_mtp(input, cache, streams.target())
                .map_err(|error| Exception::custom(error.to_string()))?;
            let sequence = output.logits.as_array().dim(-2);
            if sequence <= 0 {
                return Err(Exception::custom(
                    "Gemma 4 speculative input must contain at least one token",
                ));
            }
            Ok(SpeculativePrefill {
                logits: output
                    .logits
                    .as_array()
                    .try_index_device((.., sequence - 1, ..), streams.target())?,
                state: Self::state_at(&output, sequence - 1, cache.offset(), streams.target())?,
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
        let shared_kv = state
            .shared_kv
            .iter()
            .map(|(policy, (keys, values))| {
                (
                    *policy,
                    (
                        MlxTensor::from_array(keys.clone()),
                        MlxTensor::from_array(values.clone()),
                    ),
                )
            })
            .collect();
        Ok(self.assistant.begin_round(
            shared_kv,
            state.cache_len,
            MlxTensor::from_array(state.hidden),
        ))
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Array, Exception> {
        let embedding = self.proposal_embedding(last_token, streams)?;
        self.assistant
            .draft_step(&MlxTensor::from_array(embedding), state, streams.draft())
            .map(MlxTensor::into_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        cache
            .deep_clone_state()
            .expect("validated Gemma speculative state must be forkable")
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
        let output = self
            .target
            .verify_mtp(
                &MlxTensor::from_array(inputs.clone()),
                cache,
                streams.target(),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let completion = MlxSpeculativeCompletion::submit([output.logits.as_array()])?;
        Ok(Submission {
            output: Gemma4Verification { output, inputs },
            completion,
        })
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
        _draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Exception> {
        let input_len = output.inputs.dim(1) as usize;
        if verified_inputs == 0 || verified_inputs > input_len {
            return Err(Exception::custom(format!(
                "cannot commit {verified_inputs} verified Gemma inputs from a block of {input_len}"
            )));
        }
        if verified_inputs == input_len {
            return Ok(SpeculativeCommit {
                state: Self::state_at(
                    &output.output,
                    verified_inputs as i32 - 1,
                    cache.offset(),
                    streams.target(),
                )?,
                replayed_tokens: 0,
            });
        }

        cache.restore_checkpoint(&checkpoint, streams.target())?;
        let retained = output
            .inputs
            .try_index_device((.., ..verified_inputs as i32), streams.target())?;
        let replayed = self
            .target
            .verify_mtp(&MlxTensor::from_array(retained), cache, streams.target())
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(SpeculativeCommit {
            state: Self::state_at(
                &replayed,
                verified_inputs as i32 - 1,
                cache.offset(),
                streams.target(),
            )?,
            replayed_tokens: verified_inputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::dflash_block_token_ids;

    #[test]
    fn muse_glimmer_dflash_runtime_block_has_only_requested_proposals() {
        assert_eq!(dflash_block_token_ids(7, 99, 1), [7, 99]);
        assert_eq!(dflash_block_token_ids(7, 99, 3), [7, 99, 99, 99]);
        assert_eq!(dflash_block_token_ids(7, 99, 15).len(), 16);
    }
}
