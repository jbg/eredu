//! Muse-Glimmer target/DFlash adapter for the lossless speculative scheduler.

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::indexing::TryIndexOp,
    quantization::MaybeQuantized,
    transforms::{async_eval, async_eval_timed, async_eval_with_event},
    Array, Stream,
};

use crate::{
    api::{input::ModelInput, ModelCache},
    architectures::muse_glimmer::{
        assistant::{DFlashContextCache, MuseGlimmerDFlash},
        layerwise::{DFlashTargetOutput, LayerwiseDecoder, MuseGlimmerLayerwiseCache},
        scale_logits,
    },
    runtime::{
        cache::KeyValueCache,
        generation::speculative::{
            MtpBackend, MtpCommit, MtpComponentTimingEvaluations, MtpComponentTimings,
            MtpExecutionStreams, MtpPrefill, MtpStreamTopology,
        },
    },
};

#[derive(Clone)]
pub(crate) struct MuseTargetState {
    pending_context: Option<Array>,
    draft_cache: Option<DFlashContextCache>,
    cache_len: usize,
}

#[derive(Clone)]
pub(crate) struct MuseDraftState {
    logits: Array,
    cursor: usize,
    proposal_capacity: usize,
    draft_cache: DFlashContextCache,
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
    component_timing: bool,
    component_timings: MtpComponentTimingEvaluations,
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
            component_timing: false,
            component_timings: MtpComponentTimingEvaluations::default(),
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

    fn initial_target_state(
        context: Array,
        cache_len: usize,
        window: i32,
        stream: &Stream,
    ) -> Result<MuseTargetState, Exception> {
        Ok(MuseTargetState {
            pending_context: Some(Self::retain_context(context, window, stream)?),
            draft_cache: None,
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
            if let Some(pending) = state.pending_context.as_ref() {
                let _ = streams.wait_for_target_outputs([pending])?;
            }
            return Ok(state.clone());
        }
        let pending_context = if let Some(pending) = state.pending_context.as_ref() {
            async_eval_with_event([pending])?.synchronize()?;
            let pending = pending.copy(streams.draft())?;
            async_eval_with_event([&pending])?.synchronize()?;
            Some(pending)
        } else {
            None
        };
        Ok(MuseTargetState {
            pending_context,
            draft_cache: state.draft_cache.clone(),
            cache_len: state.cache_len,
        })
    }

    fn begin_dflash_block(
        &mut self,
        state: &MuseTargetState,
        last_token: u32,
        proposal_capacity: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MuseDraftState, Exception> {
        let maximum = self.max_draft_tokens();
        if proposal_capacity == 0 || proposal_capacity > maximum {
            return Err(Exception::custom(format!(
                "Muse-Glimmer DFlash proposal capacity must be between 1 and {maximum}"
            )));
        }
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
        // DFlash was trained with a maximum block size, but inference uses the
        // requested anchor-plus-proposal width. Attention within the block is
        // bidirectional, so padding a short request to the maximum changes the
        // proposal distributions rather than merely doing redundant work.
        let ids = dflash_block_token_ids(
            last_token,
            self.assistant.config.mask_token_id,
            proposal_capacity,
        );
        let block_size = ids.len();
        let block_size = i32::try_from(block_size)
            .map_err(|_| Exception::custom("Muse-Glimmer DFlash block size exceeds i32"))?;
        let ids = Array::from_slice(&ids, &[1, block_size]);
        let noise = self
            .draft_embedding
            .as_mut()
            .expect("initialized")
            .forward(&ids, streams.draft())?;
        if self.component_timing {
            // The released target embedding is not part of the assistant body.
            // Submit it before the assistant timestamp boundary so the
            // assistant phase measures only the DFlash layers.
            async_eval([&noise])?;
        }
        let absolute_context_end = i32::try_from(state.cache_len)
            .map_err(|_| Exception::custom("Muse-Glimmer DFlash offset exceeds i32"))?;
        let (draft_cache, context_timing) = self.assistant.update_context_cache(
            state.draft_cache,
            state.pending_context.as_ref(),
            absolute_context_end,
            self.component_timing,
            streams.draft(),
        )?;
        self.component_timings.push_draft_context(context_timing);
        let (states, assistant_timing) = self.assistant.proposal_states(
            &noise,
            &draft_cache,
            absolute_context_end,
            self.component_timing,
            streams.draft(),
        )?;
        self.component_timings
            .push_draft_assistant(assistant_timing);
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
        let head_timing = self
            .component_timing
            .then(|| async_eval_timed([&logits], streams.draft()))
            .transpose()?;
        self.component_timings.push_draft_head(head_timing);
        Ok(MuseDraftState {
            logits,
            cursor: 0,
            proposal_capacity,
            draft_cache,
            cache_len: state.cache_len,
        })
    }
}

fn dflash_block_token_ids(anchor: u32, mask_token: u32, proposal_capacity: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(proposal_capacity + 1);
    ids.push(anchor);
    ids.resize(proposal_capacity + 1, mask_token);
    ids
}

#[cfg(test)]
mod tests {
    use super::dflash_block_token_ids;

    #[test]
    fn dflash_runtime_block_contains_only_requested_proposal_positions() {
        assert_eq!(dflash_block_token_ids(7, 99, 1), [7, 99]);
        assert_eq!(dflash_block_token_ids(7, 99, 3), [7, 99, 99, 99]);
        assert_eq!(dflash_block_token_ids(7, 99, 15).len(), 16);
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

    fn set_component_timing(&mut self, enabled: bool) {
        self.component_timing = enabled;
        self.component_timings = MtpComponentTimingEvaluations::default();
    }

    fn supports_component_timing(&self) -> bool {
        true
    }

    fn take_component_timings(&mut self) -> Result<MtpComponentTimings, Exception> {
        self.component_timings.resolve()
    }

    fn take_verification_component_timings(
        &mut self,
        output: &mut Self::Verification,
    ) -> Result<MtpComponentTimings, Exception> {
        Ok(MtpComponentTimings {
            target_verification: output.output.device_time,
            ..MtpComponentTimings::default()
        })
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
        let state = Self::initial_target_state(
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
        self.begin_dflash_block(
            state,
            last_token,
            self.max_draft_tokens(),
            MtpExecutionStreams::single(stream),
        )
    }

    fn begin_draft_with_streams(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        self.begin_dflash_block(state, last_token, self.max_draft_tokens(), streams)
    }

    fn begin_draft_with_capacity(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<Self::DraftState, Exception> {
        self.begin_dflash_block(state, last_token, proposal_capacity, streams)
    }

    fn draft_logits(
        &mut self,
        state: &mut Self::DraftState,
        _last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if state.cursor >= state.proposal_capacity {
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
            self.target.verify_dflash(
                input_tokens,
                cache,
                &target_layers,
                self.component_timing,
                stream,
            )
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
        if draft_state.cache_len != checkpoint
            || usize::try_from(draft_state.draft_cache.end()).ok() != Some(checkpoint)
        {
            return Err(Exception::custom(format!(
                "Muse-Glimmer DFlash state/cache checkpoint mismatch: state {}, draft {}, checkpoint {checkpoint}",
                draft_state.cache_len,
                draft_state.draft_cache.end()
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
        let pending_context = if verified_inputs == 0 {
            None
        } else {
            let count = i32::try_from(verified_inputs)
                .map_err(|_| Exception::custom("DFlash retained input count exceeds i32"))?;
            Some(
                output
                    .output
                    .states
                    .try_index_device((.., ..count, ..), stream)?,
            )
        };
        let state = MuseTargetState {
            pending_context,
            draft_cache: Some(draft_state.draft_cache),
            cache_len: retained,
        };
        Ok(MtpCommit {
            state,
            replayed_tokens: 0,
        })
    }

    fn commit_verification_with_streams(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: Self::CacheCheckpoint,
        verified_inputs: usize,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MtpCommit<Self::TargetState>, Exception> {
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
