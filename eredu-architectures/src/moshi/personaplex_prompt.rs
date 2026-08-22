//! Backend-neutral PersonaPlex prompt framing policy.
//!
//! The plan is codec-free: callers provide Mimi/codec and text-token shapes,
//! then a concrete backend materializes the declared frames in its native
//! tensor representation.

/// Number of Mimi codebooks per side in PersonaPlex's dual-stream layout.
pub const AUDIO_TOKENS_PER_STREAM: usize = 8;
/// PersonaPlex uses the tokenizer's existing pad id during prompt forcing.
pub const TEXT_PADDING_TOKEN: i32 = 3;
/// PersonaPlex audio tokens used for an agent-side silence frame.
pub const SILENCE_TOKENS: [i32; AUDIO_TOKENS_PER_STREAM] =
    [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// PersonaPlex audio tokens used as a user-side 440 Hz conditioning frame.
pub const SINE_TOKENS: [i32; AUDIO_TOKENS_PER_STREAM] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

/// Source of the agent-side audio forced for one prompt frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentAudioSource {
    /// A frame from the caller-provided voice prompt.
    VoiceFrame(usize),
    /// The released PersonaPlex silence tokens.
    Silence,
}

/// Source of the text token forced for one prompt frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSource {
    /// The released PersonaPlex text-padding token.
    Padding,
    /// A frame from the caller-provided text prompt.
    PromptFrame(usize),
}

/// Backend-neutral forcing declaration for one PersonaPlex prompt frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptFramePlan {
    /// Source of generated-side agent audio.
    pub agent_audio: AgentAudioSource,
    /// Source of the forced text token.
    pub text: TextSource,
    /// Whether generated audio is forced for this frame.
    pub force_generated_audio: bool,
    /// Optional per-codebook forcing mask; `None` forces every codebook.
    pub forced_generated_audio_codebooks: Option<[bool; AUDIO_TOKENS_PER_STREAM]>,
    /// Whether text is forced for this frame.
    pub force_text: bool,
}

/// Ordered PersonaPlex prompt frames for one batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptBatchPlan {
    /// Batch dimension shared by all supplied prompt tensors.
    pub batch: usize,
    /// Frames in scheduler enqueue order.
    pub frames: Vec<PromptFramePlan>,
}

/// Invalid caller-provided geometry for a PersonaPlex prompt plan.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PromptPlanError {
    /// Voice prompt geometry is not `[batch, 8, frames]`.
    #[error("PersonaPlex voice prompt tokens must have shape [batch, 8, frames], got {shape:?}")]
    InvalidVoiceShape {
        /// Supplied tensor shape.
        shape: Vec<usize>,
    },
    /// Text prompt geometry is not `[batch, frames]`.
    #[error("PersonaPlex text prompt tokens must have shape [batch, frames], got {shape:?}")]
    InvalidTextShape {
        /// Supplied tensor shape.
        shape: Vec<usize>,
    },
    /// Voice and text prompts do not share a batch dimension.
    #[error(
        "PersonaPlex voice/text prompt batches must match, got {voice_batch} and {text_batch}"
    )]
    BatchMismatch {
        /// Voice prompt batch dimension.
        voice_batch: usize,
        /// Text prompt batch dimension.
        text_batch: usize,
    },
}

/// Wraps text prompt content with PersonaPlex system tags if absent.
pub fn wrap_system_prompt(text: &str) -> String {
    let text = text.trim();
    if text.starts_with("<system>") && text.ends_with("<system>") {
        text.to_string()
    } else {
        format!("<system> {text} <system>")
    }
}

/// Declares the forced frames for voice tokens shaped `[batch, 8, frames]`.
pub fn voice_prompt_plan(shape: &[usize]) -> Result<PromptBatchPlan, PromptPlanError> {
    if shape.len() != 3 || shape[1] != AUDIO_TOKENS_PER_STREAM {
        return Err(PromptPlanError::InvalidVoiceShape {
            shape: shape.to_vec(),
        });
    }
    Ok(PromptBatchPlan {
        batch: shape[0],
        frames: (0..shape[2])
            .map(|frame| PromptFramePlan {
                agent_audio: AgentAudioSource::VoiceFrame(frame),
                text: TextSource::Padding,
                force_generated_audio: true,
                forced_generated_audio_codebooks: None,
                force_text: true,
            })
            .collect(),
    })
}

/// Declares the forced frames for text tokens shaped `[batch, frames]`.
pub fn text_prompt_plan(shape: &[usize]) -> Result<PromptBatchPlan, PromptPlanError> {
    if shape.len() != 2 {
        return Err(PromptPlanError::InvalidTextShape {
            shape: shape.to_vec(),
        });
    }
    Ok(PromptBatchPlan {
        batch: shape[0],
        frames: (0..shape[1])
            .map(|frame| PromptFramePlan {
                agent_audio: AgentAudioSource::Silence,
                text: TextSource::PromptFrame(frame),
                force_generated_audio: true,
                forced_generated_audio_codebooks: None,
                force_text: true,
            })
            .collect(),
    })
}

/// Declares PersonaPlex's hybrid voice-then-text system prompt.
pub fn system_prompt_plan(
    voice_shape: Option<&[usize]>,
    text_shape: &[usize],
) -> Result<PromptBatchPlan, PromptPlanError> {
    let mut text = text_prompt_plan(text_shape)?;
    let Some(voice_shape) = voice_shape else {
        return Ok(text);
    };
    let mut voice = voice_prompt_plan(voice_shape)?;
    if voice.batch != text.batch {
        return Err(PromptPlanError::BatchMismatch {
            voice_batch: voice.batch,
            text_batch: text.batch,
        });
    }
    voice.frames.append(&mut text.frames);
    Ok(voice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_wrapping_is_trimmed_and_idempotent() {
        assert_eq!(
            wrap_system_prompt("  be concise  "),
            "<system> be concise <system>"
        );
        assert_eq!(
            wrap_system_prompt("  <system> be concise <system>  "),
            "<system> be concise <system>"
        );
        assert_eq!(wrap_system_prompt(""), "<system>  <system>");
    }

    #[test]
    fn published_prompt_frames_are_exact() {
        assert_eq!(AUDIO_TOKENS_PER_STREAM, 8);
        assert_eq!(TEXT_PADDING_TOKEN, 3);
        assert_eq!(
            SILENCE_TOKENS,
            [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
        );
        assert_eq!(SINE_TOKENS, [430, 1268, 381, 1611, 1095, 1495, 56, 472]);
    }

    #[test]
    fn voice_text_and_mixed_plans_preserve_protocol_enqueue_order() {
        let voice = voice_prompt_plan(&[2, 8, 2]).unwrap();
        assert_eq!(voice.batch, 2);
        assert_eq!(
            voice.frames,
            vec![
                PromptFramePlan {
                    agent_audio: AgentAudioSource::VoiceFrame(0),
                    text: TextSource::Padding,
                    force_generated_audio: true,
                    forced_generated_audio_codebooks: None,
                    force_text: true,
                },
                PromptFramePlan {
                    agent_audio: AgentAudioSource::VoiceFrame(1),
                    text: TextSource::Padding,
                    force_generated_audio: true,
                    forced_generated_audio_codebooks: None,
                    force_text: true,
                },
            ]
        );

        let text = text_prompt_plan(&[2, 3]).unwrap();
        assert_eq!(text.frames.len(), 3);
        assert_eq!(text.frames[0].agent_audio, AgentAudioSource::Silence);
        assert_eq!(text.frames[2].text, TextSource::PromptFrame(2));

        let mixed = system_prompt_plan(Some(&[2, 8, 2]), &[2, 3]).unwrap();
        assert_eq!(mixed.batch, 2);
        assert_eq!(mixed.frames[..2], voice.frames);
        assert_eq!(mixed.frames[2..], text.frames);
        assert_eq!(system_prompt_plan(None, &[2, 3]).unwrap(), text);
        assert!(mixed.frames.iter().all(|frame| {
            frame.force_generated_audio
                && frame.forced_generated_audio_codebooks.is_none()
                && frame.force_text
        }));
    }

    #[test]
    fn malformed_prompt_shapes_are_rejected_by_the_neutral_plan() {
        for shape in [&[1, 8][..], &[1, 7, 2][..], &[1, 8, 2, 1][..]] {
            let error = voice_prompt_plan(shape).unwrap_err().to_string();
            assert!(error.contains("shape [batch, 8, frames]"), "{error}");
        }
        for shape in [&[1][..], &[1, 2, 3][..]] {
            let error = text_prompt_plan(shape).unwrap_err().to_string();
            assert!(error.contains("shape [batch, frames]"), "{error}");
        }
        let error = system_prompt_plan(Some(&[2, 8, 1]), &[3, 1])
            .unwrap_err()
            .to_string();
        assert!(error.contains("batches must match"), "{error}");
    }
}
