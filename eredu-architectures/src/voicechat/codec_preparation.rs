//! Exact retained VoiceChat component sources projected into portable codec preparation.

use super::{
    artifact::RetainedVoiceChatArtifacts, checkpoint::Component, parameters::SpeechParameterError,
};
use eredu_codec::{
    ear_decoder_preparation::{DecoderPreparationError, PreparedDecoderBindings},
    ear_encoder_preparation::{EncoderPreparationError, PreparedEncoderBindings},
    prvq_preparation::{PreparedQuantizerBindings, QuantizerPreparationError},
};

/// Missing component sources, invalid codec geometry or typed codec recipe failure.
#[derive(Debug, thiserror::Error)]
pub enum CodecPreparationError<E> {
    /// Selected sources do not contain the requested component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Normalized codec geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] eredu_codec::Error),
    /// Canonical recipes cannot realize the selected codec component.
    #[error(transparent)]
    Preparation(E),
}

fn component_recipe<'a>(
    artifacts: &'a RetainedVoiceChatArtifacts,
    component: Component,
    name: &str,
) -> Option<&'a eredu_checkpoint::recipe::DerivedWeightRecipe> {
    artifacts
        .plan()
        .parameters()
        .get(name)
        .filter(|parameter| parameter.component == component)
        .map(|parameter| &parameter.recipe)
}

/// Retains only the neural decoder's admitted component source and recipes.
/// Codec-owned preparation validates its symbolic layout and selects bounded
/// F32 execution conversions without reading payloads or creating native modules.
/// Quantizer and spectral reconstruction remain separate construction roles.
pub fn prepare_neural_decoder(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<PreparedDecoderBindings, CodecPreparationError<DecoderPreparationError>> {
    let source = artifacts.component_source(Component::CodecDecoder).ok_or(
        SpeechParameterError::MissingComponent(Component::CodecDecoder),
    )?;
    let config = super::codec::waveform_decoder_config(&artifacts.plan().config().codec)?;
    PreparedDecoderBindings::from_recipes(
        config.neural,
        "tts_model.audio_codec.decoder",
        source.clone(),
        |name| component_recipe(artifacts, Component::CodecDecoder, name),
    )
    .map_err(CodecPreparationError::Preparation)
}

/// Projects retained codec-quantizer authority into independently leased codebooks.
/// Neural encoder/decoder and the TTS feedback table retain separate source roles.
pub fn prepare_quantizer(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<PreparedQuantizerBindings, CodecPreparationError<QuantizerPreparationError>> {
    let source = artifacts
        .component_source(Component::CodecQuantizer)
        .ok_or(SpeechParameterError::MissingComponent(
            Component::CodecQuantizer,
        ))?;
    let config = super::codec::waveform_decoder_config(&artifacts.plan().config().codec)?;
    PreparedQuantizerBindings::from_recipes(
        config.quantizer,
        "tts_model.audio_codec.prvq",
        source.clone(),
        |name| component_recipe(artifacts, Component::CodecQuantizer, name),
    )
    .map_err(CodecPreparationError::Preparation)
}

/// Retains only the neural encoder's admitted component source and recipes.
/// Codec-owned preparation validates its symbolic layout and selects bounded
/// F32 execution conversions without reading payloads or creating native modules.
/// Quantizer and spectral reconstruction remain separate construction roles.
pub fn prepare_neural_encoder(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<PreparedEncoderBindings, CodecPreparationError<EncoderPreparationError>> {
    let source = artifacts.component_source(Component::CodecEncoder).ok_or(
        SpeechParameterError::MissingComponent(Component::CodecEncoder),
    )?;
    let config = super::codec::waveform_encoder_config(&artifacts.plan().config().codec)?;
    PreparedEncoderBindings::from_recipes(
        config.neural,
        "tts_model.audio_codec.encoder",
        source.clone(),
        |name| component_recipe(artifacts, Component::CodecEncoder, name),
    )
    .map_err(CodecPreparationError::Preparation)
}
