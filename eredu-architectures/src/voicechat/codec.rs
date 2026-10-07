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
