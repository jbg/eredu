//! Fixed-Aria silent-carrier startup and retained TTS prompt inputs.
//!
//! The source encoder runs once on a finite silent carrier. Its output is not
//! interchangeable with repeated codec-silence tokens: boundary history belongs
//! to that encoding. Batch expansion reuses the single prepared carrier.

use eredu_codec::{
    ear::AudioEncoder, ear_encoder_layered::EncoderExecution, prvq_layered::QuantizerExecution,
};
use eredu_nn::{Index, NeuralBackend, Tensor, TensorElementType};

use super::{
    channels::{TEXT_EOS_ID, TEXT_PAD_ID},
    config::VoiceChatConfig,
    tts_conditioning::{CharacterBatch, CharacterMapping},
    tts_ingress::PromptLayout,
    tts_mog::TtsError,
};

/// Fixed-speaker carrier geometry, independent of native allocation.
#[derive(Debug, Clone, Copy)]
pub struct AriaPromptConfig {
    /// Codec PCM rate (released: 22050 Hz).
    pub sample_rate: usize,
    /// Samples per codec frame (released: 1764).
    pub samples_per_frame: usize,
    /// Prompt audio duration before rounding down to complete frames.
    pub audio_prompt_duration: f64,
    /// Number of residual codebooks.
    pub codebooks: i32,
    /// Speech PAD ID, equal to the codebook cardinality.
    pub speech_pad: i32,
}

/// Validated source startup schedule with bounded carrier allocation.
#[derive(Debug, Clone)]
pub struct AriaPromptPlan {
    frames: i32,
    carrier_samples: i32,
    samples_per_frame: i32,
    codebooks: i32,
    speech_pad: i32,
}
impl AriaPromptPlan {
    /// Validates the finite carrier sample budget before native work.
    pub fn new(config: AriaPromptConfig, max_carrier_samples: usize) -> Result<Self, TtsError> {
        if config.sample_rate == 0
            || config.samples_per_frame == 0
            || config.samples_per_frame > i32::MAX as usize
            || !config.audio_prompt_duration.is_finite()
            || config.audio_prompt_duration <= 0.0
            || config.codebooks <= 0
            || config.speech_pad <= 0
        {
            return Err(TtsError::Configuration("invalid Aria carrier geometry"));
        }
        let frames = (config.audio_prompt_duration * config.sample_rate as f64
            / config.samples_per_frame as f64)
            .floor();
        if !frames.is_finite() || frames < 1.0 || frames >= i32::MAX as f64 {
            return Err(TtsError::Configuration(
                "Aria carrier frame count exceeds tensor geometry",
            ));
        }
        let frames = frames as i32;
        let carrier_samples = (frames + 1)
            .checked_mul(config.samples_per_frame as i32)
            .ok_or(TtsError::Configuration(
                "Aria carrier sample count exceeds tensor geometry",
            ))?;
        capacity(
            "carrier samples",
            carrier_samples as usize,
            max_carrier_samples,
        )?;
        Ok(Self {
            frames,
            carrier_samples,
            samples_per_frame: config.samples_per_frame as i32,
            codebooks: config.codebooks,
            speech_pad: config.speech_pad,
        })
    }

    /// Reads the normalized released Aria policy. The TTS tower's system-text
    /// prefix is disabled in this checkpoint; the agent prompt is a separate path.
    pub fn from_config(
        config: &VoiceChatConfig,
        max_carrier_samples: usize,
    ) -> Result<Self, TtsError> {
        if config.speaker_name != "Aria" || config.tts.use_system_prompt {
            return Err(TtsError::Configuration(
                "fixed Aria startup requires the released TTS prompt policy",
            ));
        }
        Self::new(
            AriaPromptConfig {
                sample_rate: config.cadence.target_sample_rate,
                samples_per_frame: config.cadence.target_samples_per_frame,
                audio_prompt_duration: config.tts.audio_prompt_duration,
                codebooks: i32::try_from(config.codec.num_quantizers)
                    .map_err(|_| TtsError::Configuration("prompt depth exceeds I32"))?,
                speech_pad: i32::try_from(config.tokens.speech_pad)
                    .map_err(|_| TtsError::Configuration("speech PAD exceeds I32"))?,
            },
            max_carrier_samples,
        )
    }

    /// Number of retained prefill positions (released: 37).
    pub const fn frames(&self) -> i32 {
        self.frames
    }
    /// Number of silent PCM samples encoded once, including the text-EOS prefix.
    pub const fn carrier_samples(&self) -> i32 {
        self.carrier_samples
    }
    /// Allocates the explicitly bounded single-row silent carrier.
    pub fn carrier<T: Tensor>(&self, context: &T::Context) -> Result<T, TtsError> {
        Ok(T::full_f32(0.0, &[1, 1, self.carrier_samples], context)?)
    }

    /// Encodes the carrier once and prepares repeatable batch-expanded inputs.
    /// Encoder and quantizer use retained parameters without reopening artifacts.
    pub fn encode<B: NeuralBackend, E: EncoderExecution<B>, Q: QuantizerExecution<B>>(
        &self,
        encoder: &mut AudioEncoder<B, E>,
        quantizer: &mut Q,
        batch: i32,
        max_token_positions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedTtsPrompt<B::Tensor>, TtsError> {
        self.positions(batch, max_token_positions)?;
        if encoder.config().samples_per_frame()? != self.samples_per_frame
            || encoder.config().quantizer.depth != self.codebooks as usize
            || encoder.config().quantizer.cardinality != self.speech_pad
            || quantizer.geometry() != encoder.config().quantizer
        {
            return Err(TtsError::Configuration(
                "Aria carrier and codec geometry disagree",
            ));
        }
        let codes = encoder.encode(&self.carrier(context)?, quantizer, context)?;
        self.prepare_encoded(&codes, batch, max_token_positions, context)
    }

    /// Applies the original prompt masks and PAD overrides to finite encoder output.
    /// The final carrier frame is excluded; the first and last retained code
    /// frames become PAD, while the last retained audio-mask position is BOS.
    pub fn prepare_encoded<T: Tensor>(
        &self,
        encoded: &T,
        batch: i32,
        max_token_positions: usize,
        context: &T::Context,
    ) -> Result<PreparedTtsPrompt<T>, TtsError> {
        let positions = self.positions(batch, max_token_positions)?;
        if encoded.shape() != [1, self.frames + 1, self.codebooks]
            || encoded.element_type() != Some(TensorElementType::I32)
        {
            return Err(TtsError::Shape(
                "Aria encoder output must be one complete I32 carrier sequence",
            ));
        }
        let source = encoded.index(
            &[Index::Full, Index::Range(0, self.frames), Index::Full],
            context,
        )?;
        let pad_mask = (0..self.frames)
            .map(|i| i32::from(i == 0 || i == self.frames - 1))
            .collect::<Vec<_>>();
        let mask =
            T::from_i32_slice(&pad_mask, &[1, self.frames, 1], context)?.equal_i32(1, context)?;
        let pad = T::full_i32(self.speech_pad, &[1, 1, self.codebooks], context)?;
        let codes = T::where_condition(&mask, &pad, &source, context)?
            .broadcast_to(&[batch, self.frames, self.codebooks], context)?;
        let audio_mask = (0..positions)
            .map(|i| i % self.frames as usize == (self.frames - 1) as usize)
            .collect::<Vec<_>>();
        let (subword_ids, subword_mask) = self.subwords(positions);
        Ok(PreparedTtsPrompt {
            codes,
            layout: PromptLayout::new(batch, self.frames, &audio_mask)?,
            subword_ids,
            subword_mask,
        })
    }

    /// Admits batch-expanded startup positions before allocating carrier or native state.
    pub fn positions(&self, batch: i32, maximum: usize) -> Result<usize, TtsError> {
        let positions = batch
            .checked_mul(self.frames)
            .filter(|n| batch > 0 && n.checked_mul(self.codebooks).is_some())
            .ok_or(TtsError::Shape("Aria batch extent exceeds tensor geometry"))?
            as usize;
        capacity("prompt positions", positions, maximum)?;
        Ok(positions)
    }

    /// Prepares the exact bounded startup character inputs without neural execution.
    pub fn characters(
        &self,
        mapping: &CharacterMapping,
        batch: i32,
        max_positions: usize,
        max_cells: usize,
    ) -> Result<CharacterBatch, TtsError> {
        let positions = self.positions(batch, max_positions)?;
        let (ids, mask) = self.subwords(positions);
        mapping.prepare(&ids, Some(&mask), batch, self.frames, max_cells)
    }

    fn subwords(&self, positions: usize) -> (Vec<i32>, Vec<bool>) {
        let ids = (0..positions)
            .map(|i| {
                if i % self.frames as usize == (self.frames - 1) as usize {
                    TEXT_EOS_ID
                } else {
                    TEXT_PAD_ID
                }
            })
            .collect();
        let mask = (0..positions)
            .map(|i| i % self.frames as usize >= self.frames.saturating_sub(2) as usize)
            .collect();
        (ids, mask)
    }
}

/// Immutable prompt inputs that can prefill independent session branches.
#[derive(Debug, Clone)]
pub struct PreparedTtsPrompt<T: Tensor> {
    codes: T,
    layout: PromptLayout,
    subword_ids: Vec<i32>,
    subword_mask: Vec<bool>,
}

impl<T: Tensor> eredu_nn::RetainedTensorRoots<T> for PreparedTtsPrompt<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.codes);
    }
}

impl<T: Tensor> PreparedTtsPrompt<T> {
    /// Shift-ready prompt codes in `[batch, frames, codebooks]` order.
    pub fn codes(&self) -> &T {
        &self.codes
    }
    /// Source BOS/pre-BOS mask interpretation.
    pub fn layout(&self) -> &PromptLayout {
        &self.layout
    }
    /// Already shifted live subword IDs in row-major order.
    pub fn subword_ids(&self) -> &[i32] {
        &self.subword_ids
    }
    /// Last two prompt subwords participate in character conditioning.
    pub fn subword_mask(&self) -> &[bool] {
        &self.subword_mask
    }
    /// Packs character inputs through the existing bounded tokenizer mapping.
    pub fn characters(
        &self,
        mapping: &CharacterMapping,
        max_cells: usize,
    ) -> Result<CharacterBatch, TtsError> {
        mapping.prepare(
            &self.subword_ids,
            Some(&self.subword_mask),
            self.layout.batch(),
            self.layout.frames(),
            max_cells,
        )
    }
    /// Unshifted final prompt code seeds the first recurrent invocation without sampling.
    pub fn initial_feedback(&self, context: &T::Context) -> Result<T, TtsError> {
        Ok(self.codes.index(
            &[
                Index::Full,
                Index::Range(self.layout.frames() - 1, self.layout.frames()),
                Index::Full,
            ],
            context,
        )?)
    }
}

fn capacity(resource: &'static str, required: usize, maximum: usize) -> Result<(), TtsError> {
    if required > maximum {
        return Err(TtsError::Capacity {
            resource,
            required,
            maximum,
        });
    }
    Ok(())
}
