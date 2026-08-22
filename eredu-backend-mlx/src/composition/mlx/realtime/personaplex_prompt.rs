//! PersonaPlex prompt protocol for the MLX realtime facade.
//!
//! The protocol is codec-free: callers supply Mimi/codec tokens and tokenize
//! wrapped system text with a PersonaPlex-compatible tokenizer.

use safemlx::{error::Exception, ops::broadcast_to, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::mlx::error::Error;
use eredu_core::realtime::{RealtimeError, RealtimeScheduler};
use eredu_core::scheduler::{RequestId, WorkId};
use eredu_core::RealtimeModel;

use super::{MlxRealtimeBackend, MlxRealtimeInput};

/// Number of Mimi codebooks per side in PersonaPlex's dual-stream layout.
pub const AUDIO_TOKENS_PER_STREAM: i32 = 8;
/// PersonaPlex uses the tokenizer's existing pad id during prompt forcing.
pub const TEXT_PADDING_TOKEN: i32 = 3;
/// PersonaPlex audio tokens used for an agent-side silence frame.
pub const SILENCE_TOKENS: [i32; 8] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// PersonaPlex audio tokens used as a user-side 440 Hz conditioning frame.
pub const SINE_TOKENS: [i32; 8] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentAudioSource {
    VoiceFrame(i32),
    Silence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextSource {
    Padding,
    PromptFrame(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PromptFramePlan {
    agent_audio: AgentAudioSource,
    text: TextSource,
    force_generated_audio: bool,
    forced_generated_audio_codebooks: Option<[bool; AUDIO_TOKENS_PER_STREAM as usize]>,
    force_text: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptBatchPlan {
    batch: i32,
    frames: Vec<PromptFramePlan>,
}

/// One forced system-prompt frame expressed entirely in codec/text tokens.
pub struct PromptFrame<'a> {
    /// Generated-side agent audio tokens shaped `[batch, 8]`.
    pub agent_audio_tokens: &'a Array,
    /// User-side conditioning audio tokens shaped `[batch, 8]`.
    pub user_audio_tokens: &'a Array,
    /// Agent text token shaped `[batch, 1]`.
    pub text_token: &'a Array,
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

/// Creates a repeated silence frame shaped `[batch, 8]`.
pub fn silence_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    repeated_frame(&SILENCE_TOKENS, batch, stream)
}

/// Creates a repeated sine-conditioning frame shaped `[batch, 8]`.
pub fn sine_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    repeated_frame(&SINE_TOKENS, batch, stream)
}

/// Creates a repeated text-padding token shaped `[batch, 1]`.
pub fn text_padding_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    Array::full::<i32>(
        &text_frame_shape(batch),
        Array::from_int(TEXT_PADDING_TOKEN),
        stream,
    )
}

fn repeated_frame(tokens: &[i32; 8], batch: i32, stream: &Stream) -> Result<Array, Exception> {
    broadcast_to(
        Array::from_slice(tokens, &[1, AUDIO_TOKENS_PER_STREAM]),
        &audio_frame_shape(batch),
        stream,
    )
}

const fn audio_frame_shape(batch: i32) -> [i32; 2] {
    [batch, AUDIO_TOKENS_PER_STREAM]
}

const fn text_frame_shape(batch: i32) -> [i32; 2] {
    [batch, 1]
}

/// Enqueues one forced PersonaPlex prompt frame on an existing request.
pub fn enqueue_prompt_frame(
    scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
    model: &RealtimeModel<MlxRealtimeBackend>,
    request: RequestId,
    frame: PromptFrame<'_>,
) -> Result<WorkId, Error> {
    scheduler
        .enqueue(model, request, forced_prompt_input(frame))
        .map_err(realtime_error)
}

fn forced_prompt_input(frame: PromptFrame<'_>) -> MlxRealtimeInput {
    MlxRealtimeInput::encoded_audio(frame.user_audio_tokens)
        .with_forced_generated_audio(frame.agent_audio_tokens)
        .with_forced_text(frame.text_token)
}

/// Enqueues a sequence of forced voice-prompt frames.
///
/// `voice_prompt_tokens` uses codec layout `[batch, 8, frames]`; the user side
/// is filled with PersonaPlex's sine-conditioning token frame and text is
/// forced to the existing text pad id.
pub fn enqueue_voice_prompt(
    scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
    model: &RealtimeModel<MlxRealtimeBackend>,
    request: RequestId,
    voice_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<Vec<WorkId>, Error> {
    scheduler
        .enqueue_batch(
            model,
            request,
            voice_prompt_inputs(voice_prompt_tokens, stream)?,
        )
        .map_err(realtime_error)
}

fn voice_prompt_inputs(
    voice_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<Vec<MlxRealtimeInput>, Error> {
    let plan = voice_prompt_plan(voice_prompt_tokens.shape())?;
    materialize_prompt_plan(&plan, Some(voice_prompt_tokens), None, stream)
}

/// Enqueues a sequence of forced text-prompt tokens.
///
/// `text_prompt_tokens` is shaped `[batch, frames]` and should contain token ids
/// from the caller's PersonaPlex-compatible text tokenizer. The generated audio
/// side is forced to PersonaPlex silence while the user side is filled with the
/// sine-conditioning frame.
pub fn enqueue_text_prompt(
    scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
    model: &RealtimeModel<MlxRealtimeBackend>,
    request: RequestId,
    text_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<Vec<WorkId>, Error> {
    scheduler
        .enqueue_batch(
            model,
            request,
            text_prompt_inputs(text_prompt_tokens, stream)?,
        )
        .map_err(realtime_error)
}

fn text_prompt_inputs(
    text_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<Vec<MlxRealtimeInput>, Error> {
    let plan = text_prompt_plan(text_prompt_tokens.shape())?;
    materialize_prompt_plan(&plan, None, Some(text_prompt_tokens), stream)
}

/// Enqueues PersonaPlex's hybrid system prompt from codec and text tokens.
///
/// `voice_prompt_tokens`, when present, uses codec layout `[batch, 8, frames]`.
/// `text_prompt_tokens` uses text-token layout `[batch, frames]`. Text wrapping
/// and tokenization stay outside this crate; callers can use
/// [`wrap_system_prompt`] before tokenizing with a compatible SentencePiece
/// tokenizer.
pub fn enqueue_system_prompt(
    scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
    model: &RealtimeModel<MlxRealtimeBackend>,
    request: RequestId,
    voice_prompt_tokens: Option<&Array>,
    text_prompt_tokens: &Array,
    stream: &Stream,
) -> Result<Vec<WorkId>, Error> {
    let plan = system_prompt_plan(
        voice_prompt_tokens.map(Array::shape),
        text_prompt_tokens.shape(),
    )?;
    let inputs =
        materialize_prompt_plan(&plan, voice_prompt_tokens, Some(text_prompt_tokens), stream)?;
    scheduler
        .enqueue_batch(model, request, inputs)
        .map_err(realtime_error)
}

fn voice_prompt_plan(shape: &[i32]) -> Result<PromptBatchPlan, Error> {
    if shape.len() != 3 || shape[1] != AUDIO_TOKENS_PER_STREAM {
        return Err(Error::Parallel(format!(
            "PersonaPlex voice prompt tokens must have shape [batch, 8, frames], got {shape:?}"
        )));
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

fn text_prompt_plan(shape: &[i32]) -> Result<PromptBatchPlan, Error> {
    if shape.len() != 2 {
        return Err(Error::Parallel(format!(
            "PersonaPlex text prompt tokens must have shape [batch, frames], got {shape:?}"
        )));
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

fn system_prompt_plan(
    voice_shape: Option<&[i32]>,
    text_shape: &[i32],
) -> Result<PromptBatchPlan, Error> {
    let mut text = text_prompt_plan(text_shape)?;
    let Some(voice_shape) = voice_shape else {
        return Ok(text);
    };
    let mut voice = voice_prompt_plan(voice_shape)?;
    if voice.batch != text.batch {
        return Err(Error::Parallel(format!(
            "PersonaPlex voice/text prompt batches must match, got {} and {}",
            voice.batch, text.batch
        )));
    }
    voice.frames.append(&mut text.frames);
    Ok(voice)
}

fn materialize_prompt_plan(
    plan: &PromptBatchPlan,
    voice_prompt_tokens: Option<&Array>,
    text_prompt_tokens: Option<&Array>,
    stream: &Stream,
) -> Result<Vec<MlxRealtimeInput>, Error> {
    let sine = sine_frame(plan.batch, stream)?;
    let silence = plan
        .frames
        .iter()
        .any(|frame| frame.agent_audio == AgentAudioSource::Silence)
        .then(|| silence_frame(plan.batch, stream))
        .transpose()?;
    let padding = plan
        .frames
        .iter()
        .any(|frame| frame.text == TextSource::Padding)
        .then(|| text_padding_frame(plan.batch, stream))
        .transpose()?;
    let mut inputs = Vec::with_capacity(plan.frames.len());
    for frame in &plan.frames {
        let agent = match frame.agent_audio {
            AgentAudioSource::VoiceFrame(index) => voice_prompt_tokens
                .ok_or_else(|| {
                    Error::Parallel("PersonaPlex voice prompt plan lost its tokens".into())
                })?
                .try_index_device((.., .., index), stream)?,
            AgentAudioSource::Silence => silence
                .as_ref()
                .ok_or_else(|| {
                    Error::Parallel("PersonaPlex silence prompt frame is missing".into())
                })?
                .clone(),
        };
        let text = match frame.text {
            TextSource::Padding => padding
                .as_ref()
                .ok_or_else(|| Error::Parallel("PersonaPlex text padding frame is missing".into()))?
                .clone(),
            TextSource::PromptFrame(index) => text_prompt_tokens
                .ok_or_else(|| {
                    Error::Parallel("PersonaPlex text prompt plan lost its tokens".into())
                })?
                .try_index_device((.., index), stream)?
                .expand_dims(1, stream)?,
        };
        let mut input = MlxRealtimeInput::encoded_audio(&sine);
        if frame.force_generated_audio {
            input = match frame.forced_generated_audio_codebooks {
                Some(mask) => input.with_partially_forced_generated_audio(&agent, mask),
                None => input.with_forced_generated_audio(&agent),
            };
        }
        if frame.force_text {
            input = input.with_forced_text(&text);
        }
        inputs.push(input);
    }
    Ok(inputs)
}

fn realtime_error(error: RealtimeError<Error>) -> Error {
    Error::Parallel(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        audio_frame_shape, system_prompt_plan, text_frame_shape, text_prompt_plan,
        voice_prompt_plan, wrap_system_prompt, AgentAudioSource, PromptFramePlan, TextSource,
        AUDIO_TOKENS_PER_STREAM, SILENCE_TOKENS, SINE_TOKENS, TEXT_PADDING_TOKEN,
    };

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
        assert_eq!(super::wrap_system_prompt(""), "<system>  <system>");
    }

    #[test]
    fn published_prompt_frames_are_exact_without_a_device() {
        assert_eq!(AUDIO_TOKENS_PER_STREAM, 8);
        assert_eq!(TEXT_PADDING_TOKEN, 3);
        assert_eq!(
            SILENCE_TOKENS,
            [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
        );
        assert_eq!(SINE_TOKENS, [430, 1268, 381, 1611, 1095, 1495, 56, 472]);
        assert_eq!(audio_frame_shape(3), [3, 8]);
        assert_eq!(text_frame_shape(3), [3, 1]);
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
        assert_eq!(
            text.frames,
            vec![
                PromptFramePlan {
                    agent_audio: AgentAudioSource::Silence,
                    text: TextSource::PromptFrame(0),
                    force_generated_audio: true,
                    forced_generated_audio_codebooks: None,
                    force_text: true,
                },
                PromptFramePlan {
                    agent_audio: AgentAudioSource::Silence,
                    text: TextSource::PromptFrame(1),
                    force_generated_audio: true,
                    forced_generated_audio_codebooks: None,
                    force_text: true,
                },
                PromptFramePlan {
                    agent_audio: AgentAudioSource::Silence,
                    text: TextSource::PromptFrame(2),
                    force_generated_audio: true,
                    forced_generated_audio_codebooks: None,
                    force_text: true,
                },
            ]
        );

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
    fn malformed_prompt_shapes_fail_before_materialization_or_enqueue() {
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
