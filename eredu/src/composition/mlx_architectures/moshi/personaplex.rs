//! PersonaPlex realtime speech-to-speech token model support.
//!
//! PersonaPlex is a Moshi-family full-duplex model with hybrid system prompts:
//! a voice segment forced on the generated audio stream and a text segment
//! forced on the generated text stream. This module intentionally remains
//! codec-free; callers provide Mimi/codec tokens and decode emitted tokens with
//! a codec outside `eredu`.

use eredu_checkpoint::WeightQuantization;

use std::path::Path;

use safemlx::{error::Exception, ops::broadcast_to, ops::indexing::TryIndexOp, Array, Stream};
use serde::Deserialize;

use crate::{
    backend::mlx::error::Error,
    composition::mlx::realtime::{MlxRealtimeBackend, MlxRealtimeInput},
    composition::mlx_architectures::moshi::model as moshi,
    core::realtime::{RealtimeError, RealtimeScheduler},
    RealtimeModel, RequestId, WorkId,
};

/// Hugging Face repository for the released PersonaPlex checkpoint.
pub const DEFAULT_HF_REPO: &str = "nvidia/personaplex-7b-v1";
/// Released PersonaPlex language-model checkpoint filename.
pub const MODEL_SAFETENSORS: &str = "model.safetensors";
/// Released Mimi codec checkpoint filename.
pub const MIMI_SAFETENSORS: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
/// Released text tokenizer filename used by NVIDIA's runtime.
pub const TEXT_TOKENIZER: &str = "tokenizer_spm_32k_3.model";

/// Number of Mimi codebooks per side in PersonaPlex's dual-stream layout.
pub const AUDIO_TOKENS_PER_STREAM: i32 = 8;
/// PersonaPlex uses the tokenizer's existing pad id during prompt forcing.
pub const TEXT_PADDING_TOKEN: i32 = 3;
/// PersonaPlex audio tokens used for an agent-side silence frame.
pub const SILENCE_TOKENS: [i32; 8] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// PersonaPlex audio tokens used as a user-side 440 Hz conditioning frame.
pub const SINE_TOKENS: [i32; 8] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

/// PersonaPlex `config.json` metadata from the released HF repository.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelMetadata {
    /// Expected to be `personaplex`.
    pub model_type: String,
    /// Released model version, for example `7b-v1`.
    #[serde(default)]
    pub version: Option<String>,
    /// Optional MLX affine checkpoint quantization settings.
    #[serde(default)]
    pub quantization: Option<WeightQuantization>,
}

/// Returns the published PersonaPlex 7B v1 language-model defaults.
pub fn model_args_7b_v1() -> moshi::ModelArgs {
    moshi::ModelArgs {
        model_type: Some("personaplex".to_string()),
        dim: 4_096,
        text_card: 32_000,
        existing_text_padding_id: Some(TEXT_PADDING_TOKEN),
        n_q: 16,
        dep_q: 16,
        generated_audio_codebooks: Some(AUDIO_TOKENS_PER_STREAM),
        card: 2_048,
        num_heads: 32,
        num_layers: 32,
        dim_feedforward: Some((4.125 * 4_096.0) as i32),
        causal: true,
        context: 3_000,
        max_period: 10_000.0,
        positional_embedding: "rope".to_string(),
        depformer_dim: 1_024,
        depformer_dim_feedforward: Some((4.125 * 1_024.0) as i32),
        depformer_num_heads: 16,
        depformer_num_layers: 6,
        depformer_context: Some(8),
        depformer_max_period: Some(10_000.0),
        depformer_pos_emb: "none".to_string(),
        delays: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        moshi_name: Some(MODEL_SAFETENSORS.to_string()),
        conditioners: Default::default(),
        cross_attention: false,
        demux_second_stream: false,
        depformer_low_rank_embeddings: None,
        extra_heads_num_heads: 0,
        quantization: None,
    }
}

/// Reads and validates PersonaPlex metadata.
pub fn get_model_metadata(model_dir: impl AsRef<Path>) -> Result<ModelMetadata, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    let metadata: ModelMetadata = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    validate_metadata(&metadata)?;
    Ok(metadata)
}

/// Validates parsed PersonaPlex metadata.
pub fn validate_metadata(metadata: &ModelMetadata) -> Result<(), Error> {
    if let Some(quantization) = metadata.quantization {
        quantization.validate()?;
    }
    if metadata.model_type != "personaplex" {
        return Err(Error::UnsupportedModelType(metadata.model_type.clone()));
    }
    match metadata.version.as_deref() {
        None | Some("7b-v1") => Ok(()),
        Some(version) => Err(Error::UnsupportedArchitecture(format!(
            "unsupported PersonaPlex version {version}; only 7b-v1 defaults are known"
        ))),
    }
}

/// Validates a `personaplex` config value.
pub fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    model_metadata_from_config_value(config).map(|_| ())
}

/// Parses validated PersonaPlex metadata for shared load and inspection planning.
pub(crate) fn model_metadata_from_config_value(
    config: &serde_json::Value,
) -> Result<ModelMetadata, Error> {
    let metadata: ModelMetadata = serde_json::from_value(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid PersonaPlex config: {error}"))
    })?;
    validate_metadata(&metadata)?;
    Ok(metadata)
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
    Array::full::<i32>(&[batch, 1], Array::from_int(TEXT_PADDING_TOKEN), stream)
}

fn repeated_frame(tokens: &[i32; 8], batch: i32, stream: &Stream) -> Result<Array, Exception> {
    broadcast_to(
        Array::from_slice(tokens, &[1, AUDIO_TOKENS_PER_STREAM]),
        &[batch, AUDIO_TOKENS_PER_STREAM],
        stream,
    )
}

/// Enqueues one forced PersonaPlex prompt frame on an existing request.
pub fn enqueue_prompt_frame(
    scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
    model: &RealtimeModel<MlxRealtimeBackend>,
    request: RequestId,
    frame: PromptFrame<'_>,
) -> Result<WorkId, Error> {
    scheduler
        .enqueue(
            model,
            request,
            MlxRealtimeInput::encoded_audio(frame.user_audio_tokens)
                .with_forced_generated_audio(frame.agent_audio_tokens)
                .with_forced_text(frame.text_token),
        )
        .map_err(realtime_error)
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
    if voice_prompt_tokens.shape().len() != 3
        || voice_prompt_tokens.dim(1) != AUDIO_TOKENS_PER_STREAM
    {
        return Err(Error::Parallel(format!(
            "PersonaPlex voice prompt tokens must have shape [batch, 8, frames], got {:?}",
            voice_prompt_tokens.shape()
        )));
    }
    let batch = voice_prompt_tokens.dim(0);
    let sine = sine_frame(batch, stream)?;
    let text = text_padding_frame(batch, stream)?;
    let mut inputs = Vec::with_capacity(voice_prompt_tokens.dim(2) as usize);
    for frame in 0..voice_prompt_tokens.dim(2) {
        let agent = voice_prompt_tokens.try_index_device((.., .., frame), stream)?;
        inputs.push(
            MlxRealtimeInput::encoded_audio(&sine)
                .with_forced_generated_audio(&agent)
                .with_forced_text(&text),
        );
    }
    Ok(inputs)
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
    if text_prompt_tokens.shape().len() != 2 {
        return Err(Error::Parallel(format!(
            "PersonaPlex text prompt tokens must have shape [batch, frames], got {:?}",
            text_prompt_tokens.shape()
        )));
    }
    let batch = text_prompt_tokens.dim(0);
    let silence = silence_frame(batch, stream)?;
    let sine = sine_frame(batch, stream)?;
    let mut inputs = Vec::with_capacity(text_prompt_tokens.dim(1) as usize);
    for frame in 0..text_prompt_tokens.dim(1) {
        let text = text_prompt_tokens
            .try_index_device((.., frame), stream)?
            .expand_dims(1, stream)?;
        inputs.push(
            MlxRealtimeInput::encoded_audio(&sine)
                .with_forced_generated_audio(&silence)
                .with_forced_text(&text),
        );
    }
    Ok(inputs)
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
    let mut inputs = Vec::new();
    if let Some(tokens) = voice_prompt_tokens {
        inputs.extend(voice_prompt_inputs(tokens, stream)?);
    }
    inputs.extend(text_prompt_inputs(text_prompt_tokens, stream)?);
    scheduler
        .enqueue_batch(model, request, inputs)
        .map_err(realtime_error)
}

fn realtime_error(error: RealtimeError<Error>) -> Error {
    Error::Parallel(error.to_string())
}
