//! Muse-Glimmer target/DFlash adapter for the lossless speculative scheduler.

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp},
    quantization::MaybeQuantized,
    transforms::async_eval_with_event,
    Array, Stream,
};

use crate::{
    api::{input::ModelInput, ModelCache},
    architectures::muse_glimmer::{
        assistant::MuseGlimmerDFlash,
        layerwise::{DFlashTargetOutput, LayerwiseDecoder, MuseGlimmerLayerwiseCache},
        scale_logits,
    },
    runtime::{
        cache::KeyValueCache,
        generation::speculative::{
            MtpBackend, MtpCommit, MtpExecutionStreams, MtpPrefill, MtpStreamTopology,
        },
    },
};

#[derive(Clone)]
pub(crate) struct MuseTargetState {
    context: Array,
    cache_len: usize,
}

#[derive(Clone)]
pub(crate) struct MuseDraftState {
    logits: Array,
    cursor: usize,
    context: Array,
    cache_len: usize,
}

pub(crate) struct MuseVerification {
    output: DFlashTargetOutput,
    input_len: usize,
}

pub(crate) struct MuseGlimmerMtpBackend<'a> {
    target: &'a mut LayerwiseDecoder,
    assistant: &'a mut MuseGlimmerDFlash,
    draft_embedding: Option<MaybeQuantized<nn::Embedding>>,
    draft_head: Option<MaybeQuantized<nn::Linear>>,
}

impl<'a> MuseGlimmerMtpBackend<'a> {
    pub(crate) fn new(
        target: &'a mut LayerwiseDecoder,
        assistant: &'a mut MuseGlimmerDFlash,
    ) -> Self {
        Self {
            target,
            assistant,
            draft_embedding: None,
            draft_head: None,
        }
    }

    fn with_cache<T>(
        cache: &mut ModelCache,
        f: impl FnOnce(&mut MuseGlimmerLayerwiseCache) -> Result<T, Exception>,
    ) -> Result<T, Exception> {
        match cache {
            ModelCache::KeyValue(values) => {
                let mut owned = MuseGlimmerLayerwiseCache::Concat(std::mem::take(values));
                let result = f(&mut owned);
                let MuseGlimmerLayerwiseCache::Concat(owned) = owned else {
                    unreachable!()
                };
                *values = owned;
                result
            }
            ModelCache::PagedKeyValue(values) => {
                let mut owned = MuseGlimmerLayerwiseCache::Paged(std::mem::take(values));
                let result = f(&mut owned);
                let MuseGlimmerLayerwiseCache::Paged(owned) = owned else {
                    unreachable!()
                };
                *values = owned;
                result
            }
            _ => Err(Exception::custom(
                "Muse-Glimmer DFlash target cache has the wrong architecture",
            )),
        }
    }

    fn cache_len(cache: &ModelCache) -> usize {
        match cache {
            ModelCache::KeyValue(values) => values
                .iter()
                .flatten()
                .next()
                .map(|cache| cache.offset() as usize)
                .unwrap_or(0),
            ModelCache::PagedKeyValue(values) => values
                .iter()
                .flatten()
                .next()
                .map(|cache| cache.offset() as usize)
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn truncate(cache: &mut ModelCache, len: usize, stream: &Stream) -> Result<(), Exception> {
        match cache {
            ModelCache::KeyValue(values) => {
                let len = i32::try_from(len)
                    .map_err(|_| Exception::custom("Muse-Glimmer cache length exceeds i32"))?;
                for cache in values.iter_mut().flatten() {
                    cache.truncate(len, stream)?;
                }
            }
            ModelCache::PagedKeyValue(values) => {
                let len = i64::try_from(len)
                    .map_err(|_| Exception::custom("Muse-Glimmer cache length exceeds i64"))?;
                for cache in values.iter_mut().flatten() {
                    cache.truncate(len, stream)?;
                }
            }
            _ => {
                return Err(Exception::custom(
                    "Muse-Glimmer DFlash target cache has the wrong architecture",
                ))
            }
        }
        Ok(())
    }

    fn retain_context(context: Array, window: i32, stream: &Stream) -> Result<Array, Exception> {
        let length = context.dim(1);
        if length <= window {
            Ok(context)
        } else {
            context.try_index_device((.., length - window.., ..), stream)
        }
    }

    fn target_state(
        context: Array,
        cache_len: usize,
        window: i32,
        stream: &Stream,
    ) -> Result<MuseTargetState, Exception> {
        Ok(MuseTargetState {
            context: Self::retain_context(context, window, stream)?,
            cache_len,
        })
    }

    fn state_on_draft_stream(
        state: &MuseTargetState,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MuseTargetState, Exception> {
        if !streams.is_split() {
            return Ok(state.clone());
        }
        if streams.topology() == MtpStreamTopology::SameDeviceSplit {
            let _ = streams.wait_for_target_outputs([&state.context])?;
            return Ok(state.clone());
        }
        async_eval_with_event([&state.context])?.synchronize()?;
        let context = state.context.copy(streams.draft())?;
        async_eval_with_event([&context])?.synchronize()?;
        Ok(MuseTargetState {
            context,
            cache_len: state.cache_len,
        })
    }
}

impl MtpBackend for MuseGlimmerMtpBackend<'_> {
    type Cache = ModelCache;
    type TargetState = MuseTargetState;
    type DraftState = MuseDraftState;
    type CacheCheckpoint = usize;
    type Verification = MuseVerification;

    fn max_draft_tokens(&self) -> usize {
        self.assistant.config.block_size.saturating_sub(1).min(15)
    }

    fn prefill(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<MtpPrefill<Self::TargetState>, Exception> {
        let target_layers = self.assistant.config.target_layer_ids.clone();
        let output = Self::with_cache(cache, |cache| {
            self.target
                .prefill_dflash(input, cache, &target_layers, stream)
        })?;
        let sequence = output.logits.dim(1);
        if sequence == 0 {
            return Err(Exception::custom("Muse-Glimmer DFlash input is empty"));
        }
        let logits = output
            .logits
            .try_index_device((.., sequence - 1, ..), stream)?;
        let cache_len = Self::cache_len(cache);
        let state = Self::target_state(
            output.states,
            cache_len,
            self.assistant.config.sliding_window,
            stream,
        )?;
        Ok(MtpPrefill {
            logits,
            state,
            evaluated_tokens: sequence as usize,
        })
    }

    fn begin_draft(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Self::DraftState, Exception> {
        self.begin_draft_with_streams(state, last_token, MtpExecutionStreams::single(stream))
    }

    fn begin_draft_with_streams(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        let state = Self::state_on_draft_stream(state, streams)?;
        if self.draft_embedding.is_none() || self.draft_head.is_none() {
            let (embedding, head) = self
                .target
                .dflash_weight_snapshot(streams.draft(), streams.crosses_devices())?;
            if streams.topology() == MtpStreamTopology::SameDeviceSplit {
                let embedding_parameters = embedding.parameters().flatten();
                let head_parameters = head.parameters().flatten();
                let _ = streams.wait_for_target_outputs(
                    embedding_parameters
                        .values()
                        .copied()
                        .chain(head_parameters.values().copied()),
                )?;
            }
            self.draft_embedding = Some(embedding);
            self.draft_head = Some(head);
        }
        let mut ids = Vec::with_capacity(self.assistant.config.block_size);
        ids.push(last_token);
        ids.resize(
            self.assistant.config.block_size,
            self.assistant.config.mask_token_id,
        );
        let ids = Array::from_slice(&ids, &[1, self.assistant.config.block_size as i32]);
        let noise = self
            .draft_embedding
            .as_mut()
            .expect("initialized")
            .forward(&ids, streams.draft())?;
        let states = self.assistant.proposal_states(
            &noise,
            &state.context,
            i32::try_from(state.cache_len)
                .map_err(|_| Exception::custom("Muse-Glimmer DFlash offset exceeds i32"))?,
            streams.draft(),
        )?;
        let logits = self
            .draft_head
            .as_mut()
            .expect("initialized")
            .forward(&states, streams.draft())?;
        let logits = scale_logits(
            logits,
            self.target.args().output_multiplier,
            self.target.args().final_logit_softcapping,
            streams.draft(),
        )?;
        Ok(MuseDraftState {
            logits,
            cursor: 0,
            context: state.context,
            cache_len: state.cache_len,
        })
    }

    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        _last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if state.cursor >= self.max_draft_tokens() {
            return Err(Exception::custom(
                "Muse-Glimmer DFlash proposal block is exhausted",
            ));
        }
        let row = i32::try_from(state.cursor)
            .map_err(|_| Exception::custom("DFlash proposal index exceeds i32"))?;
        state.cursor += 1;
        state.logits.try_index_device((.., row, ..), stream)
    }

    fn checkpoint(cache: &Self::Cache) -> Self::CacheCheckpoint {
        Self::cache_len(cache)
    }

    fn verify(
        &mut self,
        input_tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<Self::Verification, Exception> {
        let input_len = input_tokens.dim(1) as usize;
        let target_layers = self.assistant.config.target_layer_ids.clone();
        let output = Self::with_cache(cache, |cache| {
            self.target
                .verify_dflash(input_tokens, cache, &target_layers, stream)
        })?;
        Ok(MuseVerification { output, input_len })
    }

    fn verification_logits(output: &Self::Verification) -> &Array {
        &output.output.logits
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        stream: &Stream,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        if draft_state.cache_len != checkpoint {
            return Err(Exception::custom(format!(
                "Muse-Glimmer DFlash state/cache checkpoint mismatch: state {}, checkpoint {checkpoint}",
                draft_state.cache_len
            )));
        }
        if verified_inputs > output.input_len {
            return Err(Exception::custom(
                "Muse-Glimmer DFlash commit exceeds verification length",
            ));
        }
        let retained = checkpoint
            .checked_add(verified_inputs)
            .ok_or_else(|| Exception::custom("Muse-Glimmer DFlash cache length overflow"))?;
        if verified_inputs != output.input_len {
            Self::truncate(cache, retained, stream)?;
        }
        let context = if verified_inputs == 0 {
            draft_state.context
        } else {
            let count = i32::try_from(verified_inputs)
                .map_err(|_| Exception::custom("DFlash retained input count exceeds i32"))?;
            let accepted = output
                .output
                .states
                .try_index_device((.., ..count, ..), stream)?;
            concatenate_axis(&[draft_state.context, accepted], 1, stream)?
        };
        let state = Self::target_state(
            context,
            retained,
            self.assistant.config.sliding_window,
            stream,
        )?;
        Ok(MtpCommit {
            state,
            replayed_tokens: 0,
        })
    }

    fn commit_verification_with_streams(
        &mut self,
        output: Self::Verification,
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
        match streams.topology() {
            MtpStreamTopology::Single => {}
            MtpStreamTopology::SameDeviceSplit => {
                let _ = streams.wait_for_draft_outputs([&draft_state.context])?;
            }
            MtpStreamTopology::CrossDeviceSplit => {
                async_eval_with_event([&draft_state.context])?.synchronize()?;
                draft_state.context = draft_state.context.copy(streams.target())?;
                async_eval_with_event([&draft_state.context])?.synchronize()?;
            }
        }
        self.commit_verification(
            output,
            draft_state,
            cache,
            checkpoint,
            verified_inputs,
            streams.target(),
        )
    }
}
