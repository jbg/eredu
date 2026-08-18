//! Qwen3-Next and Qwen3.5/3.6 model math for the shared embedded-MTP executor.

use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    architectures::qwen::hybrid::qwen3_5::{Cache, LayerCache, Model, QwenMtpStepOutput},
    backend::mlx::speculative::embedded::{EmbeddedMtpOutput, EmbeddedMtpTarget},
    runtime::media::input::{self, ModelInput},
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

impl<T: QwenMtpTarget> EmbeddedMtpTarget for T {
    type Cache = Cache;
    type DraftCache = Vec<LayerCache>;

    fn prefill_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        let output = self.prefill_mtp_target(input, cache, stream)?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens,
        })
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let output = self.verify_mtp_target(tokens, cache, stream)?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: tokens.clone(),
        })
    }

    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence > 1 {
            let hidden = output
                .hidden
                .try_index_device((.., ..sequence - 1, ..), stream)?;
            let next_tokens = tokens.try_index_device((.., 1..), stream)?;
            self.forward_mtp_drafter(&hidden, &next_tokens, &mut cache.mtp_layers, stream)?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Self::Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        _draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let logits = self.forward_mtp_drafter(hidden, &token, cache, stream)?;
        Ok((logits, hidden.clone()))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.forward_mtp_drafter(hidden, tokens, cache, stream)?;
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        usize::from(self.mtp_layer_count() > 0)
    }
}
