//! Compact coupled VoiceChat fixtures shared by neutral and native conformance.
//!
//! Learned values originate in independent NVIDIA component fixtures; compact
//! perception and cadence-preserving codec parameters are synthetic. Factories
//! panic on malformed fixture data or backend construction failure, as test setup.
//! These fixtures do not establish released-checkpoint waveform agreement.

use eredu_architectures::voicechat::{
    config::{GemmaConfig, VoiceChatConfig},
    conformer::ConformerDimensions,
    frontend::{FrontendOptions, PcmPrecision},
    input::InputState,
    perception::PerceptionDimensions,
    perception_rnnt::PerceptionRnnt,
    rnnt::RnntDimensions,
    subsampling::SubsamplingDimensions,
    tts_conditioning::{CharacterFlags, CharacterMapping},
    tts_feedback::FeedbackControls,
    tts_generator::TtsGenerator,
    tts_ingress::PromptLayout,
};
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig},
    ear_decoder::DecoderConfig,
    ear_spectral::SpectralConfig,
    prvq::Geometry,
};
use serde_json::Value;

use eredu_nn::{NeuralBackend, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};

/// Compact NVIDIA TTS fixture, including the actual character encoder.
pub fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/voicechat/tts-character-transition-nvidia.json"
    ))
    .unwrap()
}
/// Pinned released configuration used for clocks and control policy.
pub fn config() -> VoiceChatConfig {
    VoiceChatConfig::from_json(include_str!(
        "../../fixtures/voicechat/official-inference-config.json"
    ))
    .unwrap()
}
/// Empty bounded input state with the coupled fixture preprocessing policy.
pub fn input_state<T: Tensor>() -> InputState<T> {
    InputState::new(
        &config().perception.preprocessor,
        FrontendOptions {
            pcm_precision: PcmPrecision::F32,
            buffer_frames: 3,
            queue_capacity_samples: 8000,
            pad_and_drop_preencoded: false,
        },
    )
    .unwrap()
}
/// Compact nonzero perception and RNNT with a forced blank joint readout.
pub fn perception<B: NeuralBackend>(context: &<B::Tensor as Tensor>::Context) -> PerceptionRnnt<B> {
    perception_with_precision::<B>(Default::default(), context)
}
/// Compact learned perception with selected precision and F32 recurrent transcription.
pub fn perception_with_precision<B: NeuralBackend>(
    precision: eredu_architectures::voicechat::precision::SpeechPrecision,
    context: &<B::Tensor as Tensor>::Context,
) -> PerceptionRnnt<B> {
    struct Bind<'c, T: Tensor>(
        &'c T::Context,
        eredu_architectures::voicechat::precision::SpeechPrecision,
    );
    impl<'a, T: Tensor> ParameterVisitorMut<'a, T> for Bind<'_, T> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut T) {
            let count = target
                .shape()
                .iter()
                .map(|&n| n as usize)
                .product::<usize>();
            let values = match metadata.id.as_str() {
                "stt_model.rnnt_joint.joint_net.2.weight" => vec![0.; count],
                "stt_model.rnnt_joint.joint_net.2.bias" => vec![0., 1.],
                _ => (0..count).map(|i| ((i % 11) as f32 - 4.) * 0.015).collect(),
            };
            *target = T::from_f32_slice(&values, target.shape(), self.0).unwrap();
            if self.1.weights != eredu_architectures::voicechat::precision::SpeechFloat::F32
                && self.1.narrows(metadata.id.as_str(), target.shape().len())
            {
                *target = target.cast_float(self.1.weights.element(), self.0).unwrap();
            }
        }
    }
    let mut model = PerceptionRnnt::<B>::new_with_precision(
        PerceptionDimensions {
            subsampling: SubsamplingDimensions {
                features: 128,
                channels: 2,
                width: 4,
            },
            block: ConformerDimensions {
                width: 4,
                heads: 2,
                ff_hidden: 8,
                conv_kernel: 3,
                left_context: 3,
            },
            layers: 1,
            output_width: 4,
        },
        RnntDimensions {
            vocab_size: 1,
            predictor_width: 1,
            predictor_layers: 1,
            encoder_width: 4,
            joint_width: 1,
        },
        precision,
        context,
    )
    .unwrap();
    model.visit_parameters_mut(&mut Bind::<B::Tensor>(context, precision));
    model
}
/// Causal Gemma geometry for the compact fixture's two guidance cache lanes.
pub fn tts_backbone_config(fixture: &Value) -> GemmaConfig {
    geometry(&fixture["transformer_config"], true)
}

/// Nonzero character/TTS stack with 37 fixed-Aria prompt frames.
pub fn tts<B: NeuralBackend>(
    fixture: &Value,
    context: &<B::Tensor as Tensor>::Context,
) -> TtsGenerator<B> {
    tts_with_precision::<B>(fixture, Default::default(), context)
}
/// Learned TTS projections with selected precision and unchanged conditioning/RVQ.
pub fn tts_with_precision<B: NeuralBackend>(
    fixture: &Value,
    precision: eredu_architectures::voicechat::precision::SpeechPrecision,
    context: &<B::Tensor as Tensor>::Context,
) -> TtsGenerator<B> {
    let mut policy = config().tts;
    policy.backbone = tts_backbone_config(fixture);
    policy.character_encoder = geometry(&fixture["character"]["config"], false);
    policy.latent_size = 4;
    policy.codebook_size = 5;
    policy.mog.intermediate_size = 12;
    policy.mog.low_rank = 2;
    policy.mog.num_predictions = 4;
    let mut generator = TtsGenerator::<B>::new_with_precision(
        &policy,
        fixture["character"]["characters"].as_i64().unwrap() as i32,
        37,
        precision,
        context,
    )
    .unwrap();
    struct Bind<'v, 'c, T: Tensor>(
        &'v Value,
        &'c T::Context,
        eredu_architectures::voicechat::precision::SpeechPrecision,
    );
    impl<'a, T: Tensor> ParameterVisitorMut<'a, T> for Bind<'_, '_, T> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut T) {
            let source = &self.0[metadata.id.as_str()];
            let value = if metadata.id.as_str() == "tts_model.audio_prompt_latents.Aria" {
                // Preserve nonzero admitted conditioning while extending only
                // prompt time to the released model-position origin.
                T::from_f32_slice(&floats(source)[..8].repeat(37), &[1, 37, 8], self.1).unwrap()
            } else {
                tensor::<T>(source, self.1)
            };
            assert_eq!(target.shape(), value.shape(), "{}", metadata.id.as_str());
            *target = value;
            if self.2.weights != eredu_architectures::voicechat::precision::SpeechFloat::F32
                && self.2.narrows(metadata.id.as_str(), target.shape().len())
            {
                *target = target.cast_float(self.2.weights.element(), self.1).unwrap();
            }
        }
    }
    generator.visit_parameters_mut(&mut Bind::<B::Tensor>(
        &fixture["parameters"],
        context,
        precision,
    ));
    generator
}
/// Compact nonzero decoder retaining the released 1764-sample frame cadence.
pub fn codec<B: NeuralBackend>(
    fixture: &Value,
    context: &<B::Tensor as Tensor>::Context,
) -> AudioDecoder<B> {
    let policy = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 31,
            cardinality: 5,
            channels: 4,
        },
        neural: DecoderConfig {
            latent_size: 4,
            base_hidden_size: 2,
            channel_mult: vec![2, 1],
            rates: vec![147, 6],
            blocks_per_stage: 1,
            kernel_size: 3,
            fft_size: 18,
            groups: 1,
        },
        spectral: SpectralConfig {
            fft_size: 18,
            hop_length: 2,
        },
    };
    assert_eq!(policy.spectral.delay_samples().unwrap(), 8);
    let mut decoder = AudioDecoder::<B>::new(policy, "tts_model.audio_codec", context).unwrap();
    struct Bind<'c, T: Tensor>(Vec<f32>, &'c T::Context);
    impl<'a, T: Tensor> ParameterVisitorMut<'a, T> for Bind<'_, T> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut T) {
            let name = metadata.id.as_str();
            let values = if name.ends_with(".variance") {
                vec![1.]
            } else if name.contains(".prvq.") {
                let depth: usize = name
                    .split('.')
                    .find(|part| part.parse::<usize>().is_ok())
                    .unwrap()
                    .parse()
                    .unwrap();
                self.0[depth * 20..(depth + 1) * 20].to_vec()
            } else {
                // Cadence-preserving upsampling changes convolution geometry.
                // This fixture verifies composition/parity, not NVIDIA waveform
                // agreement for these deliberately compact decoder parameters.
                (0..target
                    .shape()
                    .iter()
                    .map(|&n| n as usize)
                    .product::<usize>())
                    .map(|i| ((i % 13) as f32 - 5.) * 0.01)
                    .collect()
            };
            *target = T::from_f32_slice(&values, target.shape(), self.1).unwrap();
        }
    }
    decoder.visit_parameters_mut(&mut Bind::<B::Tensor>(
        floats(&fixture["parameters"]["tts_model.tts_model.rvq_embs"]),
        context,
    ));
    decoder
}
fn integers(value: &Value) -> Vec<i32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}
/// Character mapping admitted from the independent NVIDIA fixture.
pub fn mapping(fixture: &Value) -> CharacterMapping {
    let character = &fixture["character"];
    let vocabulary = character["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, token)| (id as u32, token.as_str().unwrap().to_owned()))
        .collect();
    CharacterMapping::new(
        &vocabulary,
        CharacterFlags {
            continuation: integers(&character["continuation_flags"]),
            continuation_pad: character["continuation_pad"].as_i64().unwrap() as i32,
            special: integers(&character["special_flags"]),
            special_pad: character["special_pad"].as_i64().unwrap() as i32,
        },
    )
    .unwrap()
}
/// Feedback controls for the compact fixture vocabulary and RVQ geometry.
pub fn controls(fixture: &Value) -> FeedbackControls {
    FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &integers(&fixture["silence"])).unwrap()
}
/// Integer RVQ prompt codes and the single released prompt-frame output mask.
pub fn prompt<T: Tensor>(fixture: &Value, context: &T::Context) -> (T, PromptLayout) {
    let codes = integers(&fixture["calls"][0]["input_codes"]);
    let codes = T::from_i32_slice(&codes[..31].repeat(37), &[1, 37, 31], context).unwrap();
    let mut audio_mask = vec![false; 37];
    audio_mask[36] = true;
    (codes, PromptLayout::new(1, 37, &audio_mask).unwrap())
}
fn floats(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn tensor<T: Tensor>(value: &Value, context: &T::Context) -> T {
    let shape: Vec<_> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    T::from_f32_slice(&floats(value), &shape, context).unwrap()
}

fn geometry(value: &Value, causal: bool) -> GemmaConfig {
    let n = |key: &str| value[key].as_u64().unwrap() as usize;
    GemmaConfig {
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        num_hidden_layers: n("num_hidden_layers"),
        num_attention_heads: n("num_attention_heads"),
        num_key_value_heads: n("num_key_value_heads"),
        head_dim: n("head_dim"),
        hidden_activation: value["hidden_activation"].as_str().unwrap().into(),
        max_position_embeddings: n("max_position_embeddings"),
        rms_norm_eps: value["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: value["rope_theta"].as_f64().unwrap(),
        rope_local_base_freq: value["rope_local_base_freq"].as_f64(),
        attention_bias: value["attention_bias"].as_bool().unwrap(),
        query_pre_attn_scalar: n("query_pre_attn_scalar"),
        sliding_window: n("sliding_window"),
        layer_types: serde_json::from_value(value["layer_types"].clone()).unwrap(),
        final_logit_softcapping: value["final_logit_softcapping"].as_f64(),
        attn_logit_softcapping: value["attn_logit_softcapping"].as_f64(),
        rope_scaling: None,
        causal,
    }
}
