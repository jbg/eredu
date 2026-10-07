//! Exact retained VoiceChat component sources projected into portable codec preparation.

use super::{
    artifact::RetainedVoiceChatArtifacts, checkpoint::Component, parameters::SpeechParameterError,
};
use eredu_codec::ear_decoder_preparation::{DecoderPreparationError, PreparedDecoderBindings};

/// Missing decoder sources or invalid exact codec parameter geometry.
#[derive(Debug, thiserror::Error)]
pub enum CodecDecoderPreparationError {
    /// Selected sources do not contain the requested component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Normalized codec geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] eredu_codec::Error),
    /// Canonical recipes cannot realize the declared decoder units.
    #[error(transparent)]
    Preparation(#[from] DecoderPreparationError),
}

/// Retains only the neural decoder's admitted component source and recipes.
/// Codec-owned preparation validates its symbolic layout and selects bounded
/// F32 execution conversions without reading payloads or creating native modules.
/// Quantizer and spectral reconstruction remain separate construction roles.
pub fn prepare_neural_decoder(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<PreparedDecoderBindings, CodecDecoderPreparationError> {
    let source = artifacts.component_source(Component::CodecDecoder).ok_or(
        SpeechParameterError::MissingComponent(Component::CodecDecoder),
    )?;
    let config = super::codec::waveform_decoder_config(&artifacts.plan().config().codec)?;
    let recipes = artifacts
        .plan()
        .parameters()
        .iter()
        .filter(|(_, parameter)| parameter.component == Component::CodecDecoder)
        .map(|(name, parameter)| (name.clone(), parameter.recipe.clone()))
        .collect();
    Ok(PreparedDecoderBindings::from_recipes(
        config.neural,
        "tts_model.audio_codec.decoder",
        source.clone(),
        &recipes,
    )?)
}

/// Missing quantizer sources or invalid exact codebook parameter geometry.
#[derive(Debug, thiserror::Error)]
pub enum CodecQuantizerPreparationError {
    /// Selected sources do not contain the requested component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Normalized codec geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] eredu_codec::Error),
    /// Canonical recipes cannot realize the declared codebooks.
    #[error(transparent)]
    Preparation(#[from] eredu_codec::prvq_preparation::QuantizerPreparationError),
}

/// Projects retained codec-quantizer authority into independently leased codebooks.
/// Neural encoder/decoder and the TTS feedback table retain separate source roles.
pub fn prepare_quantizer(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<eredu_codec::prvq_preparation::PreparedQuantizerBindings, CodecQuantizerPreparationError>
{
    let source = artifacts
        .component_source(Component::CodecQuantizer)
        .ok_or(SpeechParameterError::MissingComponent(
            Component::CodecQuantizer,
        ))?;
    let config = super::codec::waveform_decoder_config(&artifacts.plan().config().codec)?;
    let recipes = artifacts
        .plan()
        .parameters()
        .iter()
        .filter(|(_, parameter)| parameter.component == Component::CodecQuantizer)
        .map(|(name, parameter)| (name.clone(), parameter.recipe.clone()))
        .collect();
    Ok(
        eredu_codec::prvq_preparation::PreparedQuantizerBindings::from_recipes(
            config.quantizer,
            "tts_model.audio_codec.prvq",
            source.clone(),
            &recipes,
        )?,
    )
}

/// Missing encoder sources or invalid exact codec parameter geometry.
#[derive(Debug, thiserror::Error)]
pub enum CodecEncoderPreparationError {
    /// Selected sources do not contain the requested component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Normalized codec geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] eredu_codec::Error),
    /// Canonical recipes cannot realize the declared encoder units.
    #[error(transparent)]
    Preparation(#[from] eredu_codec::ear_encoder_preparation::EncoderPreparationError),
}

/// Retains only the neural encoder's admitted component source and recipes.
/// Codec-owned preparation validates its symbolic layout and selects bounded
/// F32 execution conversions without reading payloads or creating native modules.
/// Quantizer and spectral reconstruction remain separate construction roles.
pub fn prepare_neural_encoder(
    artifacts: &RetainedVoiceChatArtifacts,
) -> Result<
    eredu_codec::ear_encoder_preparation::PreparedEncoderBindings,
    CodecEncoderPreparationError,
> {
    let source = artifacts.component_source(Component::CodecEncoder).ok_or(
        SpeechParameterError::MissingComponent(Component::CodecEncoder),
    )?;
    let config = super::codec::waveform_encoder_config(&artifacts.plan().config().codec)?;
    let recipes = artifacts
        .plan()
        .parameters()
        .iter()
        .filter(|(_, parameter)| parameter.component == Component::CodecEncoder)
        .map(|(name, parameter)| (name.clone(), parameter.recipe.clone()))
        .collect();
    Ok(
        eredu_codec::ear_encoder_preparation::PreparedEncoderBindings::from_recipes(
            config.neural,
            "tts_model.audio_codec.encoder",
            source.clone(),
            &recipes,
        )?,
    )
}
