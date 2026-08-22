//! MLX materialization of the architecture-owned PersonaPlex prompt protocol.
//!
//! Callers supply Mimi/codec tokens and tokenize wrapped system text with a
//! PersonaPlex-compatible tokenizer. Neutral framing and sequence planning
//! live in [`eredu_architectures::moshi::personaplex_prompt`].

use safemlx::{error::Exception, ops::broadcast_to, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::mlx::error::Error;
use eredu_architectures::moshi::personaplex_prompt::{
    system_prompt_plan, text_prompt_plan, voice_prompt_plan, AgentAudioSource, PromptBatchPlan,
    TextSource,
};
use eredu_core::realtime::{RealtimeError, RealtimeScheduler};
use eredu_core::scheduler::{RequestId, WorkId};
use eredu_core::RealtimeModel;

use super::{MlxRealtimeBackend, MlxRealtimeInput};

pub use eredu_architectures::moshi::personaplex_prompt::{
    wrap_system_prompt, AUDIO_TOKENS_PER_STREAM, SILENCE_TOKENS, SINE_TOKENS, TEXT_PADDING_TOKEN,
};

/// One forced system-prompt frame expressed entirely in codec/text tokens.
pub struct PromptFrame<'a> {
    /// Generated-side agent audio tokens shaped `[batch, 8]`.
    pub agent_audio_tokens: &'a Array,
    /// User-side conditioning audio tokens shaped `[batch, 8]`.
    pub user_audio_tokens: &'a Array,
    /// Agent text token shaped `[batch, 1]`.
    pub text_token: &'a Array,
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

fn repeated_frame(
    tokens: &[i32; AUDIO_TOKENS_PER_STREAM],
    batch: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    broadcast_to(
        Array::from_slice(tokens, &[1, AUDIO_TOKENS_PER_STREAM as i32]),
        &audio_frame_shape(batch),
        stream,
    )
}

const fn audio_frame_shape(batch: i32) -> [i32; 2] {
    [batch, AUDIO_TOKENS_PER_STREAM as i32]
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
    let shape = prompt_shape(voice_prompt_tokens.shape())?;
    let plan = voice_prompt_plan(&shape).map_err(prompt_plan_error)?;
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
    let shape = prompt_shape(text_prompt_tokens.shape())?;
    let plan = text_prompt_plan(&shape).map_err(prompt_plan_error)?;
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
    let voice_shape = voice_prompt_tokens
        .map(Array::shape)
        .map(prompt_shape)
        .transpose()?;
    let text_shape = prompt_shape(text_prompt_tokens.shape())?;
    let plan =
        system_prompt_plan(voice_shape.as_deref(), &text_shape).map_err(prompt_plan_error)?;
    let inputs =
        materialize_prompt_plan(&plan, voice_prompt_tokens, Some(text_prompt_tokens), stream)?;
    scheduler
        .enqueue_batch(model, request, inputs)
        .map_err(realtime_error)
}

fn materialize_prompt_plan(
    plan: &PromptBatchPlan,
    voice_prompt_tokens: Option<&Array>,
    text_prompt_tokens: Option<&Array>,
    stream: &Stream,
) -> Result<Vec<MlxRealtimeInput>, Error> {
    let batch = i32::try_from(plan.batch).map_err(|_| {
        Error::Parallel(format!(
            "PersonaPlex prompt batch {} exceeds MLX dimension range",
            plan.batch
        ))
    })?;
    let sine = sine_frame(batch, stream)?;
    let silence = plan
        .frames
        .iter()
        .any(|frame| frame.agent_audio == AgentAudioSource::Silence)
        .then(|| silence_frame(batch, stream))
        .transpose()?;
    let padding = plan
        .frames
        .iter()
        .any(|frame| frame.text == TextSource::Padding)
        .then(|| text_padding_frame(batch, stream))
        .transpose()?;
    let mut inputs = Vec::with_capacity(plan.frames.len());
    for frame in &plan.frames {
        let agent = match frame.agent_audio {
            AgentAudioSource::VoiceFrame(index) => {
                let index = i32::try_from(index).map_err(|_| {
                    Error::Parallel(format!(
                        "PersonaPlex voice prompt frame {index} exceeds MLX dimension range"
                    ))
                })?;
                voice_prompt_tokens
                    .ok_or_else(|| {
                        Error::Parallel("PersonaPlex voice prompt plan lost its tokens".into())
                    })?
                    .try_index_device((.., .., index), stream)?
            }
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
            TextSource::PromptFrame(index) => {
                let index = i32::try_from(index).map_err(|_| {
                    Error::Parallel(format!(
                        "PersonaPlex text prompt frame {index} exceeds MLX dimension range"
                    ))
                })?;
                text_prompt_tokens
                    .ok_or_else(|| {
                        Error::Parallel("PersonaPlex text prompt plan lost its tokens".into())
                    })?
                    .try_index_device((.., index), stream)?
                    .expand_dims(1, stream)?
            }
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

fn prompt_plan_error(error: impl std::fmt::Display) -> Error {
    Error::Parallel(error.to_string())
}

fn prompt_shape(shape: &[i32]) -> Result<Vec<usize>, Error> {
    shape
        .iter()
        .copied()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                Error::Parallel(format!(
                    "PersonaPlex prompt shape contains a negative dimension: {shape:?}"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{audio_frame_shape, text_frame_shape};

    #[test]
    fn mlx_frame_shapes_follow_the_neutral_protocol() {
        assert_eq!(audio_frame_shape(3), [3, 8]);
        assert_eq!(text_frame_shape(3), [3, 1]);
    }
}
