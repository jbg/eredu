//! Released VoiceChat configuration projection onto the portable EAR codec.

use eredu_codec::{
    ear::{AudioDecoderConfig, AudioEncoderConfig},
    ear_decoder::DecoderConfig,
    ear_encoder::EncoderConfig,
    ear_spectral::SpectralConfig,
    prvq::Geometry,
    Error,
};

use super::config::CodecConfig;

/// Projects the released encoder-order stages without reversing their execution.
pub fn waveform_encoder_config(config: &CodecConfig) -> Result<AudioEncoderConfig, Error> {
    let decoder = waveform_decoder_config(config)?;
    let encoder = AudioEncoderConfig {
        neural: EncoderConfig {
            latent_size: decoder.neural.latent_size,
            base_hidden_size: decoder.neural.base_hidden_size,
            channel_mult: decoder.neural.channel_mult.into_iter().rev().collect(),
            rates: decoder.neural.rates.into_iter().rev().collect(),
            blocks_per_stage: decoder.neural.blocks_per_stage,
            kernel_size: decoder.neural.kernel_size,
            fft_size: decoder.neural.fft_size,
            groups: decoder.neural.groups,
        },
        spectral: decoder.spectral,
        quantizer: decoder.quantizer,
    };
    encoder.validate()?;
    Ok(encoder)
}

/// Normalizes the checkpoint's encoder-ordered stages into executable decoder geometry.
/// This performs no artifact access or native allocation.
pub fn waveform_decoder_config(config: &CodecConfig) -> Result<AudioDecoderConfig, Error> {
    let narrow = |value| {
        i32::try_from(value)
            .map_err(|_| Error::InvalidShape("VoiceChat codec geometry exceeds I32".into()))
    };
    let decoder = AudioDecoderConfig {
        quantizer: Geometry {
            depth: config.num_quantizers,
            cardinality: narrow(config.codebook_size)?,
            channels: narrow(config.latent_size)?,
        },
        neural: DecoderConfig {
            latent_size: narrow(config.latent_size)?,
            base_hidden_size: narrow(config.base_hidden_size)?,
            channel_mult: config
                .channel_mult
                .iter()
                .rev()
                .map(|&v| narrow(v))
                .collect::<Result<_, _>>()?,
            rates: config
                .rates
                .iter()
                .rev()
                .map(|&v| narrow(v))
                .collect::<Result<_, _>>()?,
            blocks_per_stage: config.num_blocks,
            kernel_size: narrow(config.kernel_size)?,
            fft_size: narrow(config.n_fft)?,
            groups: narrow(config.groups)?,
        },
        spectral: SpectralConfig {
            fft_size: narrow(config.n_fft)?,
            hop_length: narrow(config.hop_length)?,
        },
    };
    decoder.validate()?;
    if decoder.samples_per_frame()? != narrow(config.wav_to_token_ratio)? {
        return Err(Error::InvalidShape(
            "VoiceChat codec sample cadence disagrees with its stages".into(),
        ));
    }
    Ok(decoder)
}

/// Source-free neural codec scopes for the validated published VoiceChat profile.
#[derive(Debug)]
pub(crate) struct CodecDeclarations {
    pub encoder: eredu_codec::declarations::EncoderDeclaration,
    pub decoder: eredu_codec::declarations::DecoderDeclaration,
    pub quantizer: eredu_codec::declarations::QuantizerDeclaration,
}
/// Every admitted VoiceChat configuration validates against this exact released profile.
/// These immutable scopes carry no artifact values, invocation state or backend resources.
pub(crate) fn released_declarations() -> Result<&'static CodecDeclarations, Error> {
    static DECLARATIONS: std::sync::OnceLock<Result<CodecDeclarations, String>> =
        std::sync::OnceLock::new();
    DECLARATIONS
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let encoder = waveform_encoder_config(&config.codec).map_err(|e| e.to_string())?;
            let decoder = waveform_decoder_config(&config.codec).map_err(|e| e.to_string())?;
            Ok(CodecDeclarations {
                encoder: eredu_codec::declarations::EncoderDeclaration::new(
                    encoder.neural,
                    "tts_model.audio_codec.encoder",
                )
                .map_err(|e| e.to_string())?,
                decoder: eredu_codec::declarations::DecoderDeclaration::new(
                    decoder.neural,
                    "tts_model.audio_codec.decoder",
                )
                .map_err(|e| e.to_string())?,
                quantizer: eredu_codec::declarations::QuantizerDeclaration::new(
                    decoder.quantizer,
                    "tts_model.audio_codec.prvq",
                )
                .map_err(|e| e.to_string())?,
            })
        })
        .as_ref()
        .map_err(|reason| Error::InvalidShape(reason.clone()))
}
