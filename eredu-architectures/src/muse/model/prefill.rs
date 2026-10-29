//! Retained encoder products with bounded ordinary decoder invocations.
use super::*;
use crate::composite_execution::DecoderPrefillChunk;
use std::ops::Range;

pub(super) struct DecoderPrefill<T> {
    pub range: Range<usize>,
    pub assembled: Option<T>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> MuseKernel<B> {
    pub(super) fn resume_prefill<S>(
        &self,
        chunk: Option<DecoderPrefillChunk<'_, B::Tensor>>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let Some(chunk) = chunk else {
            return Ok(None);
        };
        if chunk.assembled.is_none() && chunk.range.start == 0 {
            return Ok(None);
        }
        self.validate_partition_state(state)?;
        let positions = i32::try_from(chunk.range.len()).map_err(Error::backend)?;
        let hidden = match chunk.assembled {
            Some(assembled) => {
                if assembled.dim(0) != 1 || assembled.dim(2) != self.args.hidden_size {
                    return Err(Error::backend(
                        "Muse retained decoder ingress has invalid geometry",
                    ));
                }
                assembled.index(
                    &[
                        eredu_nn::Index::Full,
                        eredu_nn::Index::Range(chunk.range.start as i32, chunk.range.end as i32),
                        eredu_nn::Index::Full,
                    ],
                    context,
                )?
            }
            None => {
                if !self
                    .selected_state()
                    .is_some_and(|state| !state.owns_global_layer(0))
                {
                    return Err(Error::backend(
                        "Muse decoder ingress lost its retained prefill input",
                    ));
                }
                // An inactive media owner publishes invocation metadata only.
                // Decoder receivers enter through their typed activation boundary.
                // Neither needs to reconstruct embeddings or rerun patch projection.
                B::Tensor::full_f32(0.0, &[1, positions, self.args.hidden_size], context)?
            }
        };
        self.align_media_clocks(false, positions, state)?;
        Ok(Some(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: None,
                parts: Vec::new(),
                vision: None,
                vision_active: false,
                audio: None,
                waveform: None,
                decoder_positions: positions,
                prefill: Some(DecoderPrefill {
                    range: chunk.range,
                    assembled: chunk.assembled.cloned(),
                }),
            },
        }))
    }

    pub(super) fn retain_prefill_ingress(
        &self,
        group: usize,
        hidden: B::Tensor,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 1 {
            return Ok(hidden);
        }
        let Some(prefill) = &mut forward.prefill else {
            return Ok(hidden);
        };
        prefill.assembled = Some(hidden.clone());
        hidden.index(
            &[
                eredu_nn::Index::Full,
                eredu_nn::Index::Range(prefill.range.start as i32, prefill.range.end as i32),
                eredu_nn::Index::Full,
            ],
            context,
        )
    }
}
