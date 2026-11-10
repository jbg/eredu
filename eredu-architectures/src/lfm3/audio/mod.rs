//! LFM3 speech-input processor and encoder declarations.

mod artifact;
mod depth;
pub use artifact::{ArtifactError, AudioTokens, BindingSource, PreparedAudio, TextAssets};
mod encoder;
mod prompt;
mod realtime;
mod resident;
mod schedule;
pub use depth::{
    CompletedSpeechFrame, DepthDeclaration, DepthDecoder, DepthFrame, DepthUnit, Feedback,
};
pub use encoder::{Declaration, Downsampler, Encoder, EncoderState, Projector, Unit};
pub use prompt::AudioPrompt;
pub use realtime::{
    AudioAdvance, AudioExecutionKernel, AudioSampling, AudioSession, AudioSessionError,
    AudioSnapshot, AudioSnapshotError,
};
#[cfg(feature = "audio")]
pub use resident::{AudioIngress, AudioIngressError, AudioIngressProgress, AudioIngressRequest};
pub use resident::{
    AudioRequest, AudioResidentParameterBytes, PreparedResidentAudio, ResidentAudio,
    ResidentAudioBinding, ResidentAudioError, ResidentAudioState,
};
pub use schedule::{
    AudioMode, AudioPhase, AudioSchedule, AudioScheduleError, AudioSchedulePolicy, DepthProgress,
};

use eredu_nn::Error;
use serde_json::{json, Value};

/// Released 16-kHz uncentered, affine-normalized Slaney power-mel policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorConfig(());

impl ProcessorConfig {
    /// Admits the released feature extractor settings from preprocessor_config.json.
    pub fn from_feature_extractor(value: &Value) -> Result<Self, Error> {
        for (key, expected) in [
            ("feature_extractor_type", json!("Lfm3AudioFeatureExtractor")),
            ("feature_size", json!(128)),
            ("sampling_rate", json!(16000)),
            ("n_fft", json!(400)),
            ("win_length", json!(400)),
            ("hop_length", json!(160)),
            ("log_floor", json!(-12.0)),
            ("normalization_offset", json!(4.0)),
            ("normalization_divisor", json!(4.0)),
            ("encoder_stride", json!(8)),
            ("padding_side", json!("right")),
            ("padding_value", json!(0.0)),
            ("return_attention_mask", json!(true)),
        ] {
            let actual = &value[key];
            let matches = match (actual.as_f64(), expected.as_f64()) {
                (Some(actual), Some(expected)) => actual == expected,
                _ => actual == &expected,
            };
            if !matches {
                return Err(Error::backend(format!(
                    "unsupported LFM3 audio processor setting {key}={actual}"
                )));
            }
        }
        Ok(Self(()))
    }

    /// Builds the portable host transform for mono PCM at the checkpoint rate.
    #[cfg(feature = "audio")]
    pub fn transform(
        self,
    ) -> Result<eredu_media::audio::power_mel::PowerMel, eredu_media::MediaError> {
        use eredu_media::audio::power_mel::{PowerMel, PowerMelConfig};
        PowerMel::new(PowerMelConfig {
            sample_rate: 16_000,
            fft_length: 400,
            hop_length: 160,
            mel_bins: 128,
            log_floor: -12.0,
            normalization_offset: 4.0,
            normalization_divisor: 4.0,
        })
    }
}
