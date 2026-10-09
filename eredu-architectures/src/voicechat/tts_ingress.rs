//! Neural EAR-TTS prompt and cached-decode input construction.
//!
//! Prompt layout is explicit host metadata. Text tokenization, prompt carrier
//! codec encoding, EOS-to-silence policy and conversational time remain outside
//! this component. Both guidance lanes use the same projected audio feedback.

use super::{
    tts_mog::{f32_shape, parameter, TtsError},
    tts_rvq_layered::RvqExecution,
};
use eredu_nn::{Index, NeuralBackend, Parameter, ParameterSpec, Parameterized, Tensor};

/// Validated prompt audio-mask layout, including every false-to-true BOS boundary.
#[derive(Debug, Clone)]
pub struct PromptLayout {
    batch: i32,
    frames: i32,
    bos: Vec<i32>,
    before_bos: Vec<i32>,
}
impl PromptLayout {
    /// Builds the pinned BOS and pre-BOS masks without reading a backend tensor.
    /// `audio_mask` is row-major `[batch, frames]`; pre-BOS ends at the first BOS.
    pub fn new(batch: i32, frames: i32, audio_mask: &[bool]) -> Result<Self, TtsError> {
        let cells = batch
            .checked_mul(frames)
            .filter(|_| batch > 0 && frames > 0)
            .ok_or(TtsError::Shape("invalid TTS prompt mask geometry"))?
            as usize;
        if audio_mask.len() != cells {
            return Err(TtsError::Shape(
                "TTS prompt mask length disagrees with geometry",
            ));
        }
        let mut bos = Vec::with_capacity(cells);
        let mut before_bos = Vec::with_capacity(cells);
        for row in audio_mask.chunks_exact(frames as usize) {
            let mut previous = false;
            let mut started = false;
            for &audio in row {
                let start = audio && !previous;
                started |= start;
                bos.push(i32::from(start));
                before_bos.push(i32::from(!started));
                previous = audio;
            }
        }
        Ok(Self {
            batch,
            frames,
            bos,
            before_bos,
        })
    }
    /// Exact host masks retained by coupled invocation agreement.
    pub(super) fn masks(&self) -> (&[i32], &[i32]) {
        (&self.bos, &self.before_bos)
    }
    /// Number of streams represented by this layout.
    pub fn batch(&self) -> i32 {
        self.batch
    }
    /// Number of model positions in each prompt stream.
    pub fn frames(&self) -> i32 {
        self.frames
    }
}

/// Audio prompt substitution before the first BOS position.
#[derive(Debug, Clone, Copy, Default)]
pub enum SpeakerPrompt {
    /// The checkpoint's fixed, already projected Aria latent.
    #[default]
    Aria,
    /// Execute the checkpoint's frozen projection on supplied prompt-code embeddings.
    /// This low-level equation does not encode or acquire a speaker recording.
    ProjectedCodes,
}

/// Audio and text lanes prepared for the shared gated-fusion component.
#[derive(Debug, Clone)]
pub struct ConditioningLanes<T> {
    /// `[batch * lanes, frames, hidden]`; conditional batch precedes unconditional.
    pub audio: T,
    /// Character/context conditioning, with the unconditional lane replaced by null.
    pub text: T,
}

/// Prompt ingress parameters, separate from both TTS transformer stacks.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TtsIngress<B: NeuralBackend> {
    bos: Parameter<B::Tensor>,
    null: Parameter<B::Tensor>,
    aria: Parameter<B::Tensor>,
    projection: Parameter<B::Tensor>,
    #[parameter(skip)]
    hidden: i32,
    #[parameter(skip)]
    speaker_frames: i32,
    #[parameter(skip)]
    local_output: i32,
}
impl<B: NeuralBackend> TtsIngress<B> {
    /// Declares canonical BOS/null, Aria and frozen-projection parameters.
    /// Released geometry is hidden width 1152 and 37 speaker-prompt frames.
    pub fn new(
        hidden: i32,
        speaker_frames: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        Self::new_local(hidden, speaker_frames, hidden, context)
    }

    pub(super) fn new_local(
        hidden: i32,
        speaker_frames: i32,
        local_output: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        if hidden <= 0 || speaker_frames <= 0 || local_output <= 0 || local_output > hidden {
            return Err(TtsError::Configuration("invalid TTS ingress geometry"));
        }
        let mut aria = ParameterSpec::trainable("tts_model.audio_prompt_latents.Aria")
            .map_err(eredu_nn::Error::backend)?;
        aria.trainable = false;
        aria.group = Some("voicechat.tts.speaker.aria".into());
        let mut projection = parameter(
            "audio_prompt_projection_W",
            "voicechat.tts.speaker.projection",
        )?;
        projection.trainable = false;
        Ok(Self {
            bos: Parameter::unloaded(
                parameter("bos_emb", "voicechat.tts.ingress")?,
                &[hidden],
                context,
            )?,
            null: Parameter::unloaded(
                parameter("null_emb", "voicechat.tts.ingress")?,
                &[hidden],
                context,
            )?,
            aria: Parameter::unloaded(aria, &[1, speaker_frames, hidden], context)?,
            projection: Parameter::unloaded(projection, &[hidden, local_output], context)?,
            hidden,
            speaker_frames,
            local_output,
        })
    }

    pub(super) fn geometry(&self) -> (i32, i32, i32) {
        (self.hidden, self.speaker_frames, self.local_output)
    }

    /// Applies the frozen prompt equation `code_embedding @ W`.
    /// The stored matrix is not a normal affine weight: the reference transposes
    /// it before calling its linear operator.
    pub fn project_prompt(
        &self,
        audio: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError> {
        f32_shape(audio, self.hidden)?;
        super::precision::require_float32(&self.projection, audio).map_err(TtsError::Shape)?;
        Ok(B::Tensor::matmul(audio, self.projection.as_ref(), context)?)
    }

    /// Builds guidance lanes from feedback that has already completed projection.
    pub fn decode_embedded(
        &self,
        audio: B::Tensor,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        f32_shape(&audio, self.hidden)?;
        super::precision::require_float32(&self.null, &audio).map_err(TtsError::Shape)?;
        let text = if let Some(conditioning) = conditioning {
            f32_shape(conditioning, self.hidden)?;
            if conditioning.shape() != audio.shape() {
                return Err(TtsError::Shape(
                    "TTS audio and conditioning geometry disagree",
                ));
            }
            conditioning.clone()
        } else {
            B::Tensor::full_f32(0.0, audio.shape(), context)?
        };
        if guidance {
            let null = self.null.as_ref().broadcast_to(audio.shape(), context)?;
            Ok(ConditioningLanes {
                audio: B::Tensor::concatenate(&[audio.clone(), audio], 0, context)?,
                text: B::Tensor::concatenate(&[text, null], 0, context)?,
            })
        } else {
            Ok(ConditioningLanes { audio, text })
        }
    }

    /// Creates cached-decode lanes from the previous frame's codec codes.
    /// `None` conditioning represents the wrapper's complete character-encoder
    /// skip for an all-false subword mask, including its flag embeddings.
    pub fn decode<Z: RvqExecution<B>>(
        &self,
        feedback: &mut Z,
        codes: &B::Tensor,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        validate_decode(codes, self.hidden, conditioning)?;
        self.decode_embedded(
            feedback.embed_codes(codes, context)?,
            conditioning,
            guidance,
            context,
        )
    }

    /// Prepares prompt lanes by shifting codes, substituting the pre-BOS speaker
    /// latent, and adding BOS vectors at every declared transition.
    /// The leading shifted code is zero (an ordinary codebook row), as in the
    /// pinned forward operation; it is not the PAD code.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill<Z: RvqExecution<B>>(
        &self,
        feedback: &mut Z,
        codes: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        validate_prefill(
            codes,
            layout,
            speaker,
            self.speaker_frames,
            self.hidden,
            conditioning,
        )?;
        let shifted = shift_prompt_codes(codes, layout, context)?;
        let embedded = feedback.embed_codes(&shifted, context)?;
        self.prefill_embedded(&embedded, layout, speaker, conditioning, guidance, context)
    }

    /// Applies speaker/BOS substitutions to already projected feedback.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_embedded(
        &self,
        embedded: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError> {
        self.prefill_with_projection(
            embedded,
            layout,
            speaker,
            conditioning,
            guidance,
            context,
            Ok,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_with_projection<F>(
        &self,
        embedded: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        guidance: bool,
        context: &<B::Tensor as Tensor>::Context,
        gather: F,
    ) -> Result<ConditioningLanes<B::Tensor>, TtsError>
    where
        F: FnOnce(B::Tensor) -> Result<B::Tensor, TtsError>,
    {
        validate_embedded(embedded, self.hidden, conditioning)?;
        if embedded.shape()[..2] != [layout.batch, layout.frames]
            || (matches!(speaker, SpeakerPrompt::Aria) && layout.frames != self.speaker_frames)
        {
            return Err(TtsError::Shape(
                "speaker prompt and embedded feedback geometry disagree",
            ));
        }
        super::precision::require_float32(&self.bos, embedded).map_err(TtsError::Shape)?;
        let prompt = match speaker {
            SpeakerPrompt::Aria => {
                super::precision::require_float32(&self.aria, embedded).map_err(TtsError::Shape)?;
                self.aria.as_ref().clone()
            }
            SpeakerPrompt::ProjectedCodes => gather(self.project_prompt(embedded, context)?)?,
        };
        let shape = [layout.batch, layout.frames, 1];
        let before_bos = B::Tensor::from_i32_slice(&layout.before_bos, &shape, context)?
            .equal_i32(1, context)?;
        let bos = B::Tensor::from_i32_slice(&layout.bos, &shape, context)?.equal_i32(1, context)?;
        let zero = B::Tensor::full_f32(0.0, &[], context)?;
        let bos = B::Tensor::where_condition(&bos, self.bos.as_ref(), &zero, context)?;
        let audio = B::Tensor::where_condition(&before_bos, &prompt, embedded, context)?
            .add(&bos, context)?;
        self.decode_embedded(audio, conditioning, guidance, context)
    }
}

/// Validates source codes and optional conditioning before feedback acquisition.
pub fn validate_decode<T: Tensor>(
    codes: &T,
    hidden: i32,
    conditioning: Option<&T>,
) -> Result<(), TtsError> {
    if codes.shape().len() != 3
        || codes.dim(0) <= 0
        || codes.dim(1) <= 0
        || codes.dim(2) != 31
        || codes.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(TtsError::Shape(
            "feedback requires I32 [batch, time, 31] codes",
        ));
    }
    validate_conditioning(conditioning, [codes.dim(0), codes.dim(1), hidden])
}
fn validate_conditioning<T: Tensor>(
    conditioning: Option<&T>,
    shape: [i32; 3],
) -> Result<(), TtsError> {
    if let Some(value) = conditioning {
        f32_shape(value, shape[2])?;
        if value.shape() != shape {
            return Err(TtsError::Shape(
                "TTS audio and conditioning geometry disagree",
            ));
        }
    }
    Ok(())
}
/// Validates projected feedback and conditioning without learned parameters.
pub fn validate_embedded<T: Tensor>(
    audio: &T,
    hidden: i32,
    conditioning: Option<&T>,
) -> Result<(), TtsError> {
    f32_shape(audio, hidden)?;
    validate_conditioning(conditioning, [audio.dim(0), audio.dim(1), hidden])
}
/// Validates prompt layout and fixed-speaker geometry before feedback acquisition.
pub fn validate_prefill<T: Tensor>(
    codes: &T,
    layout: &PromptLayout,
    speaker: SpeakerPrompt,
    speaker_frames: i32,
    hidden: i32,
    conditioning: Option<&T>,
) -> Result<(), TtsError> {
    validate_decode(codes, hidden, conditioning)?;
    if codes.shape() != [layout.batch, layout.frames, 31]
        || codes.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(TtsError::Shape(
            "prompt codes must be I32 [batch, frames, 31] matching the layout",
        ));
    }
    if matches!(speaker, SpeakerPrompt::Aria) && layout.frames != speaker_frames {
        return Err(TtsError::Shape(
            "fixed Aria latent and prompt frame count disagree",
        ));
    }
    Ok(())
}
/// Shifts prompt codes by one ordinary zero-code frame, preserving depth order.
pub fn shift_prompt_codes<T: Tensor>(
    codes: &T,
    layout: &PromptLayout,
    context: &T::Context,
) -> Result<T, TtsError> {
    if codes.shape() != [layout.batch, layout.frames, 31]
        || codes.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(TtsError::Shape("prompt codes do not match layout"));
    }
    let leading = T::full_i32(0, &[layout.batch, 1, 31], context)?;
    let shifted = if layout.frames == 1 {
        leading
    } else {
        let previous = codes.index(&[Index::Full, Index::Range(0, layout.frames - 1)], context)?;
        T::concatenate(&[leading, previous], 1, context)?
    };
    Ok(shifted)
}
