//! Released EAR-TTS control substitutions at the neural component boundary.
//!
//! Input EOS substitution and output feedback substitution are separate policies:
//! the main streaming wrapper preserves generated waveform codes before replacing
//! next-step feedback. Tool-wait paths can explicitly apply their substitution to
//! both. The facade owns idle, acknowledgment and rendering-budget decisions.
//! Current character masks are supplied independently. Live inference uses true
//! masks even for BOS, EOS and PAD; it does not infer a speaking mask. The released
//! model has no delayed context-embedding branch (`context_hidden_size=None`).

use super::tts_mog::TtsError;
use eredu_nn::{Tensor, TensorElementType};

/// Whether an explicit output silence decision also changes rendered audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformSilence {
    /// Main speech path: keep the generated frame for waveform decoding.
    PreserveGenerated,
    /// Tool-wait path: render the same substituted frame used for feedback.
    ApplySilence,
}

/// Explicit output policy for one model position across a batch.
#[derive(Debug, Clone, Copy)]
pub struct OutputPolicy<'a> {
    /// Top-level streaming policy, independent of the TTS input EOS setting.
    pub force_silence_on_eos: bool,
    /// Per-row facade permission to silence PAD, after idle/tail/ack decisions.
    /// Ordinary text and BOS remain unchanged even when this permission is true.
    pub silence_on_pad: &'a [bool],
    /// Selects the released main or tool-wait waveform substitution behavior.
    pub waveform: WaveformSilence,
}

/// Separate waveform and autoregressive feedback payloads for a completed step.
#[derive(Debug, Clone)]
pub struct OutputCodes<T> {
    /// Codec-safe codes with each speech control replaced by its silence code.
    pub waveform: T,
    /// Next model input; unsilenced rows preserve generated speech controls.
    pub next_feedback: T,
}

/// Small retained integer buffers and text controls admitted before execution.
///
/// Construction validates geometry and ranges; exact released source identity is
/// established by checkpoint admission. No method opens an artifact or reads a
/// backend tensor to the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackControls {
    cardinality: i32,
    text_eos: i32,
    text_pad: i32,
    force_input_silence_on_eos: bool,
    speech_controls: Vec<i32>,
    silence: Vec<i32>,
}

impl FeedbackControls {
    /// Retains the admitted control and silence arrays. The released geometry is
    /// cardinality 1024 and 31 silence codes; smaller geometry supports fixtures.
    pub fn new(
        cardinality: i32,
        text_eos: i32,
        text_pad: i32,
        force_input_silence_on_eos: bool,
        speech_controls: &[i32],
        silence: &[i32],
    ) -> Result<Self, TtsError> {
        if cardinality <= 0
            || text_eos < 0
            || text_pad < 0
            || text_eos == text_pad
            || speech_controls.is_empty()
            || speech_controls.len() > 32
            || speech_controls.iter().any(|&v| v < cardinality)
            || speech_controls
                .iter()
                .enumerate()
                .any(|(i, v)| speech_controls[..i].contains(v))
            || silence.is_empty()
            || silence.len() > i32::MAX as usize
            || silence.iter().any(|&v| v < 0 || v >= cardinality)
        {
            return Err(TtsError::Configuration("invalid TTS feedback controls"));
        }
        Ok(Self {
            cardinality,
            text_eos,
            text_pad,
            force_input_silence_on_eos,
            speech_controls: speech_controls.to_vec(),
            silence: silence.to_vec(),
        })
    }

    /// Number of ordinary code values in each admitted codebook.
    pub const fn cardinality(&self) -> i32 {
        self.cardinality
    }

    /// Number of independently substituted residual codebooks.
    pub fn codebooks(&self) -> usize {
        self.silence.len()
    }

    /// Applies `infer_codes_one_step`'s current-EOS substitution to the incoming
    /// previous audio frame. This does not change the parent feedback tensor.
    pub fn prepare_input<T: Tensor>(
        &self,
        previous: &T,
        current_subword: &[i32],
        context: &T::Context,
    ) -> Result<T, TtsError> {
        self.validate_step(previous, current_subword)?;
        let replace = current_subword
            .iter()
            .map(|&id| self.force_input_silence_on_eos && id == self.text_eos)
            .collect::<Vec<_>>();
        self.replace_rows(previous, &replace, context)
    }

    /// Applies explicit output policy after generation, preserving the distinct
    /// waveform and next-feedback ordering in the reference streaming wrapper.
    pub fn finish_output<T: Tensor>(
        &self,
        generated: &T,
        current_subword: &[i32],
        policy: OutputPolicy<'_>,
        context: &T::Context,
    ) -> Result<OutputCodes<T>, TtsError> {
        self.validate_step(generated, current_subword)?;
        if policy.silence_on_pad.len() != current_subword.len() {
            return Err(TtsError::Shape("TTS PAD silence policy batch differs"));
        }
        let replace = current_subword
            .iter()
            .zip(policy.silence_on_pad)
            .map(|(&id, &pad)| {
                (policy.force_silence_on_eos && id == self.text_eos) || (pad && id == self.text_pad)
            })
            .collect::<Vec<_>>();
        let next_feedback = self.replace_rows(generated, &replace, context)?;
        let waveform_source = match policy.waveform {
            WaveformSilence::PreserveGenerated => generated,
            WaveformSilence::ApplySilence => &next_feedback,
        };
        let waveform = self.replace_controls(waveform_source, context)?;
        Ok(OutputCodes {
            waveform,
            next_feedback,
        })
    }

    /// Replaces control entries individually, preserving ordinary codes in the
    /// same frame. The admitted silence table is always explicit, so the source
    /// helper's fallback to the first generated frame is unnecessary.
    pub fn replace_controls<T: Tensor>(
        &self,
        codes: &T,
        context: &T::Context,
    ) -> Result<T, TtsError> {
        self.validate_codes(codes)?;
        let silence = self.silence_tensor::<T>(context)?;
        let mut result = codes.clone();
        for &control in &self.speech_controls {
            result = T::where_condition(
                &codes.equal_i32(control, context)?,
                &silence,
                &result,
                context,
            )?;
        }
        Ok(result)
    }

    fn validate_codes<T: Tensor>(&self, codes: &T) -> Result<(), TtsError> {
        if codes.shape().len() != 3
            || codes.dim(0) <= 0
            || codes.dim(1) <= 0
            || codes.dim(2) as usize != self.silence.len()
            || codes.element_type() != Some(TensorElementType::I32)
        {
            return Err(TtsError::Shape(
                "TTS controls require I32 [batch, frames, codebooks] codes",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_step<T: Tensor>(
        &self,
        codes: &T,
        subword: &[i32],
    ) -> Result<(), TtsError> {
        self.validate_codes(codes)?;
        if codes.dim(1) != 1
            || codes.dim(0) as usize != subword.len()
            || subword.iter().any(|&id| id < 0)
        {
            return Err(TtsError::Shape(
                "TTS feedback requires one nonnegative current subword per stream",
            ));
        }
        Ok(())
    }

    fn silence_tensor<T: Tensor>(&self, context: &T::Context) -> Result<T, TtsError> {
        Ok(T::from_i32_slice(
            &self.silence,
            &[1, 1, self.silence.len() as i32],
            context,
        )?)
    }

    fn replace_rows<T: Tensor>(
        &self,
        codes: &T,
        rows: &[bool],
        context: &T::Context,
    ) -> Result<T, TtsError> {
        if !rows.iter().any(|&v| v) {
            return Ok(codes.clone());
        }
        let mask = T::from_i32_slice(
            &rows.iter().copied().map(i32::from).collect::<Vec<_>>(),
            &[codes.dim(0), 1, 1],
            context,
        )?
        .equal_i32(1, context)?;
        Ok(T::where_condition(
            &mask,
            &self.silence_tensor::<T>(context)?,
            codes,
            context,
        )?)
    }
}
