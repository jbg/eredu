//! LFM3 speech-input processor and encoder declarations.

mod artifact;
mod depth;
pub use artifact::{ArtifactError, AudioTokens, BindingSource, PreparedAudio, TextAssets};
mod encoder;
mod gguf;
pub use gguf::{
    AdmittedAudioGguf, AudioGgufEncoding, AudioGgufError, AudioGgufSpeechBindings,
    AudioGgufSpeechSources,
};
mod language;
pub use language::{
    AudioLanguage, PreparedAudioLanguage, PreparedAudioLanguagePartition,
    TensorParallelAudioLanguage,
};
mod pipeline;
pub use pipeline::AudioPipelineCommunication;
mod placement;
pub use placement::{
    AudioParameterResidency, AudioWeightUnit, PreparedAudioCodec, PreparedAudioWeightUnit,
};
mod prompt;
mod realtime;
mod resident;
mod schedule;
pub use depth::{
    AudioDepth, CompletedSpeechFrame, DepthDeclaration, DepthDecoder, DepthFrame, DepthModule,
    DepthUnit, Feedback, PolicyDepth, PreparedAudioDepth,
};
pub use encoder::{
    AudioEncoder, Declaration, Downsampler, Encoder, EncoderModule, EncoderState, PolicyEncoder,
    PreparedAudioEncoder, Projector, Unit,
};
pub use prompt::AudioPrompt;
pub use realtime::{
    AudioAdvance, AudioExecutionKernel, AudioSampling, AudioSession, AudioSessionError,
    AudioSnapshot, AudioSnapshotError,
};
#[cfg(feature = "audio")]
pub use resident::{
    AudioIngress, AudioIngressError, AudioIngressProgress, AudioIngressRequest,
    AudioIngressSnapshot, AudioPcmTranscription, AudioPcmTranscriptionSnapshot,
};
pub use resident::{
    AudioRequest, AudioResidentParameterBytes, AudioTranscription, AudioTranscriptionBlock,
    AudioTranscriptionError, AudioTranscriptionSnapshot, PreparedResidentAudio, ResidentAudio,
    ResidentAudioBinding, ResidentAudioError, ResidentAudioState,
};
pub use schedule::{
    AudioMode, AudioPhase, AudioSchedule, AudioScheduleError, AudioSchedulePolicy, DepthProgress,
};

pub use eredu_media::audio::LogMelFeatures;
use eredu_nn::Error;
use serde_json::{json, Value};

/// Released 16-kHz uncentered, affine-normalized Slaney power-mel policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorConfig(());

impl ProcessorConfig {
    /// Mono PCM rate for the retained feature extractor.
    pub const fn sample_rate(self) -> u32 {
        16_000
    }

    /// Admits the released feature extractor settings from preprocessor_config.json.
    pub fn from_feature_extractor(value: &Value) -> Result<Self, Error> {
        Self::from_feature_extractor_class(value, "Lfm3AudioFeatureExtractor")
    }

    pub(super) fn from_feature_extractor_class(value: &Value, class: &str) -> Result<Self, Error> {
        for (key, expected) in [
            ("feature_extractor_type", json!(class)),
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
            sample_rate: self.sample_rate(),
            fft_length: 400,
            hop_length: 160,
            mel_bins: 128,
            log_floor: -12.0,
            normalization_offset: 4.0,
            normalization_divisor: 4.0,
        })
    }
}
