//! Shared waveform ingress and logical-clock ownership inside the Muse graph.
use super::*;
use crate::muse::audio::execution::{prepare_audio_forward, AudioInput};

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    pub(super) fn begin_media<S>(
        &mut self,
        vision_input: Option<VisionInput<'_, B::Tensor>>,
        audio_input: Option<AudioInput<'_, B::Tensor>>,
        mask: Option<&B::Tensor>,
        parts: Vec<PreparedPart<B::Tensor>>,
        chunk_positions: Option<i32>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let decoder_positions = parts.iter().try_fold(0i32, |total, part| {
            let tokens = match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens }
                | PreparedPart::Audio { tokens } => tokens,
            };
            if tokens.shape().len() != 2 || tokens.dim(0) <= 0 || tokens.dim(1) <= 0 {
                return Err(Error::backend(
                    "Muse decoder ingress requires nonempty rank-two token spans",
                ));
            }
            total
                .checked_add(tokens.dim(1))
                .ok_or_else(|| Error::backend("Muse decoder position count exceeds i32"))
        })?;
        let decoder_positions = chunk_positions.unwrap_or(decoder_positions);
        let audio = match audio_input {
            Some(input) => Some(prepare_audio_forward::<B>(
                self.args.audio_config.as_ref().ok_or_else(|| {
                    Error::backend("Muse waveform input requires an admitted audio tower")
                })?,
                input,
                self.selected_state()
                    .and_then(|state| {
                        state
                            .selected_global_layers()
                            .iter()
                            .find(|range| range.end > self.args.num_hidden_layers as usize)
                            .map(|range| {
                                range
                                    .start
                                    .saturating_sub(self.args.num_hidden_layers as usize)
                            })
                    })
                    .unwrap_or(0),
                context,
            )?),
            None => None,
        };
        self.align_media_clocks(audio.is_some(), decoder_positions, state)?;
        let vision_active = vision_input.is_some();
        let (vision_hidden, vision) = match vision_input {
            Some(input) => {
                let config = self.args.vision_config.as_ref().ok_or_else(|| {
                    Error::backend("Muse input requires an admitted vision tower")
                })?;
                eredu_nn::sequence_layout::validate_patch_grid(
                    input.grid,
                    config.merge_size,
                    Some(input.pixels.dim(0)),
                )
                .map_err(Error::backend)?;
                if input.pixels.shape().len() != 2
                    || input.pixels.dim(1)
                        != config.temporal_patch_size * 3 * config.patch_size * config.patch_size
                {
                    return Err(Error::backend(
                        "Muse vision input has incompatible patch geometry",
                    ));
                }
                match self
                    .static_modules
                    .vision
                    .as_mut()
                    .filter(|vision| vision.has_parameters())
                {
                    Some(vision) => {
                        let (hidden, state) = vision.begin(input, context)?;
                        (Some(hidden), Some(state))
                    }
                    None => (None, None),
                }
            }
            None => (None, None),
        };
        let (waveform, audio) = audio
            .map(|value| (Some(value.hidden), Some(value.context)))
            .unwrap_or((None, None));
        let hidden = match (vision_hidden, waveform.as_ref()) {
            (Some(hidden), _) => hidden,
            (None, Some(waveform)) => waveform.clone(),
            (None, None) if self.owns_input && !vision_active => {
                self.assemble(&parts, None, None, context)?
            }
            (None, None) => {
                let batch = match parts
                    .first()
                    .ok_or_else(|| Error::backend("Muse request has no input parts"))?
                {
                    PreparedPart::Text { tokens, .. }
                    | PreparedPart::PendingText { tokens }
                    | PreparedPart::Audio { tokens }
                    | PreparedPart::Media { tokens } => tokens.dim(0),
                };
                B::Tensor::full_f32(
                    0.0,
                    &[batch, decoder_positions, self.args.hidden_size],
                    context,
                )?
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: mask.cloned(),
                parts,
                vision,
                vision_active,
                audio,
                waveform,
                decoder_positions,
                prefill: None,
            },
        })
    }
    pub(super) fn align_media_clocks<S>(
        &self,
        audio_active: bool,
        decoder_positions: i32,
        state: &mut S,
    ) -> Result<(), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        // Fixed histories use the decoder's clock for prompt-cache persistence;
        // their internal stage frame counts never become logical positions.
        if let Some(config) = &self.args.audio_config {
            let decoder_frontier = state.layer(0).map_err(Error::backend)?.offset();
            for local in 0..state.layout().len() {
                if state
                    .layout()
                    .layers()
                    .get(local)
                    .is_some_and(|policy| policy.attention().is_some())
                    && state.layer(local).map_err(Error::backend)?.offset() != decoder_frontier
                {
                    return Err(Error::backend(
                        "Muse decoder frontiers describe an incomplete media candidate",
                    ));
                }
            }
            for (index, kind) in config.execution_units()?.iter().enumerate() {
                if matches!(kind, super::super::AudioUnitKind::Convolution { .. })
                    && config.conv_kernel_size > 1
                {
                    let global = self.args.num_hidden_layers as usize + index;
                    let Ok(local) = self.local_state_ordinal(global) else {
                        continue;
                    };
                    if local >= state.layout().len() {
                        continue;
                    }
                    let layer = state.layer(local).map_err(Error::backend)?;
                    if layer.position() != decoder_frontier {
                        return Err(Error::backend(
                            "Muse waveform and decoder frontiers describe an incomplete candidate",
                        ));
                    }
                    if !audio_active {
                        layer
                            .advance_fixed(decoder_positions)
                            .map_err(Error::backend)?;
                    }
                }
            }
        }
        Ok(())
    }
}
