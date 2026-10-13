//! Numerical speech composition over independently generated NVIDIA decisions.
use super::voicechat_transformer::CausalCache;
use super::voicechat_tts_generator::{codes, compare, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    frontend::FrontendOptions,
    input::InputState,
    progress,
    speech::SpeechComponents,
    tts_conditioning::{CharacterFlags, CharacterMapping},
    tts_feedback::{FeedbackControls, OutputPolicy, WaveformSilence},
    tts_ingress::{PromptLayout, SpeakerPrompt},
};
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig},
    ear_decoder::DecoderConfig,
    ear_spectral::SpectralConfig,
    prvq::Geometry,
};
use serde_json::Value;

pub(super) fn codec(tts: &Value, context: &NumericContext) -> AudioDecoder<NumericBackend> {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 31,
            cardinality: 5,
            channels: 4,
        },
        neural: DecoderConfig {
            latent_size: 4,
            base_hidden_size: 2,
            channel_mult: vec![2, 1],
            rates: vec![3, 2],
            blocks_per_stage: 2,
            kernel_size: 3,
            fft_size: 8,
            groups: 1,
        },
        spectral: SpectralConfig {
            fft_size: 8,
            hop_length: 2,
        },
    };
    let mut decoder = AudioDecoder::new(config, "tts_model.audio_codec", context).unwrap();
    struct Bind<'a> {
        neural: &'a Value,
        codebooks: NumericTensor,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            if name.ends_with(".variance") {
                *output = NumericTensor::new(vec![], vec![1.]);
            } else if name.contains(".prvq.") {
                // The generation codebook is a derived stack of the same codec
                // means. Bind each depth's independent fixed means.
                let depth: usize = name
                    .split('.')
                    .find(|part| part.parse::<usize>().is_ok())
                    .unwrap()
                    .parse()
                    .unwrap();
                *output = NumericTensor::new(
                    vec![5, 4],
                    self.codebooks.data[depth * 20..(depth + 1) * 20].to_vec(),
                );
            } else {
                *output = tensor(&self.neural[name]);
            }
        }
    }
    decoder.visit_parameters_mut(&mut Bind {
        neural: &fixture["decoders"]["compact"]["parameters"],
        codebooks: tensor(&tts["parameters"]["tts_model.tts_model.rvq_embs"]),
    });
    decoder
}

#[test]
fn voicechat_spoken_text_runs_character_encoder_and_cached_speech_against_nvidia() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_CHARACTER_TRANSITION_REFERENCE_JSON)
            .unwrap();
    let context = NumericContext::default();
    let integers = |value: &Value| {
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    };
    let character = &fixture["character"];
    let vocabulary = character["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, token)| (id as u32, token.as_str().unwrap().to_owned()))
        .collect();
    let mapping = CharacterMapping::new(
        &vocabulary,
        CharacterFlags {
            continuation: integers(&character["continuation_flags"]),
            continuation_pad: character["continuation_pad"].as_i64().unwrap() as i32,
            special: integers(&character["special_flags"]),
            special_pad: character["special_pad"].as_i64().unwrap() as i32,
        },
    )
    .unwrap();
    let mut generator = model(&fixture, &context);
    let mut decoder = codec(&fixture, &context);
    let silence = integers(&fixture["silence"]);
    let controls = FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &silence).unwrap();
    let prompt = &fixture["calls"][0];
    let mask = |value: &Value| {
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<_>>()
    };
    let characters = mapping
        .prepare(
            &integers(&prompt["subword_ids"]),
            Some(&mask(&prompt["subword_mask"])),
            1,
            5,
            1024,
        )
        .unwrap();
    let conditioning = generator
        .condition_text(&characters, &context)
        .unwrap()
        .unwrap();
    compare(&conditioning, &prompt["supplied_subwords"], &fixture);
    let mut speech = SpeechComponents::new(&mut generator, &controls, &mut decoder).unwrap();
    let mut caches = [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ];
    let mut state = speech
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &PromptLayout::new(1, 5, &mask(&prompt["audio_mask"])).unwrap(),
            SpeakerPrompt::Aria,
            Some(&conditioning),
            &mut caches,
            &context,
        )
        .unwrap();
    compare_cache(&caches, &prompt["cache"], &fixture);
    for call in &fixture["calls"].as_array().unwrap()[1..] {
        let spoken = integers(&call["subword_ids"]);
        let policy = OutputPolicy {
            force_silence_on_eos: true,
            silence_on_pad: &[false],
            waveform: WaveformSilence::PreserveGenerated,
        };
        let offsets = caches
            .iter()
            .map(|cache| cache.offset())
            .collect::<Vec<_>>();
        assert!(speech
            .begin_text_frame(&state, &spoken, &mapping, 0, policy, &mut caches, &context)
            .is_err());
        assert_eq!(
            caches
                .iter()
                .map(|cache| cache.offset())
                .collect::<Vec<_>>(),
            offsets,
            "character capacity failure must precede causal execution"
        );
        let frame = speech
            .begin_text_frame(
                &state,
                &spoken,
                &mapping,
                1024,
                policy,
                &mut caches,
                &context,
            )
            .unwrap();
        let hidden = NumericTensor::concatenate(
            &[
                frame.tts().conditional().clone(),
                frame.tts().unconditional().unwrap().clone(),
            ],
            0,
            &context,
        )
        .unwrap();
        compare(&hidden, &call["hidden"], &fixture);
        compare_cache(&caches, &call["cache"], &fixture);
        let mut ordinal = 0;
        let frame = speech
            .generate_frame(
                &frame,
                |slot, logits: &NumericTensor, _| {
                    let expected = &call["refinement"][ordinal];
                    ordinal += 1;
                    assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                    compare(logits, &expected["logits"], &fixture);
                    Ok((
                        codes(&expected["mixture_indices"], &context)
                            .reshape(&[1, 1], &context)
                            .unwrap(),
                        tensor(&expected["gaussian_draw"]),
                    ))
                },
                &context,
            )
            .unwrap();
        let output = speech.render(&frame, &context).unwrap();
        assert_eq!(output.generated.data, codes(&call["codes"], &context).data);
        assert_eq!(
            output.state.feedback().data,
            codes(&call["feedback"], &context).data
        );
        state = output.state;
    }
}

#[test]
fn voicechat_speech_retains_feedback_guidance_and_actual_codec_candidates() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let oracle: Value =
        serde_json::from_str(eredu_evaluation::voicechat::SPEECH_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut generator = model(&fixture, &context);
    let mut decoder = codec(&fixture, &context);
    let silence: Vec<i32> = fixture["silence"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let controls = FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &silence).unwrap();
    let mut speech = SpeechComponents::new(&mut generator, &controls, &mut decoder).unwrap();
    let prompt = &fixture["calls"][0];
    let mask: Vec<bool> = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let mut caches = [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ];
    let mut state = speech
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &PromptLayout::new(1, 5, &mask).unwrap(),
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap();
    assert_eq!(
        state.feedback().data,
        codes(&prompt["feedback"], &context).data
    );
    compare_cache(&caches, &prompt["cache"], &fixture);
    let (empty, drained) = speech.finish(&state, &context).unwrap();
    assert!(empty.is_none());
    assert!(drained.codec().is_finished());
    assert!(!state.codec().is_finished());

    let mut retained_frame = None;
    for (index, call) in fixture["calls"].as_array().unwrap()[1..3]
        .iter()
        .enumerate()
    {
        let parent = state.clone();
        let old_caches = caches.clone();
        let subword = [call["subword_ids"]["values"][0].as_i64().unwrap() as i32];
        let condition = tensor(&call["supplied_subwords"]);
        let policy = OutputPolicy {
            force_silence_on_eos: true,
            silence_on_pad: &[false],
            waveform: WaveformSilence::PreserveGenerated,
        };
        // Malformed rendering policy is rejected before the borrowed cache moves.
        assert!(speech
            .begin_frame(
                &state,
                &subword,
                Some(&condition),
                OutputPolicy {
                    silence_on_pad: &[],
                    ..policy
                },
                &mut caches,
                &context
            )
            .is_err());
        compare_cache(
            &caches,
            if index == 0 {
                &prompt["cache"]
            } else {
                &fixture["calls"][1]["cache"]
            },
            &fixture,
        );
        let initial = speech
            .begin_frame(
                &state,
                &subword,
                Some(&condition),
                policy,
                &mut caches,
                &context,
            )
            .unwrap();
        assert!(speech.render(&initial, &context).is_err());
        let expected_steps = call["refinement"].as_array().unwrap();
        let mut controlled = initial.clone();
        for expected in expected_steps {
            controlled = speech
                .advance_frame(
                    &controlled,
                    |slot, logits: &NumericTensor, _| {
                        assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                        compare(logits, &expected["logits"], &fixture);
                        Ok((
                            codes(&expected["mixture_indices"], &context)
                                .reshape(&[1, 1], &context)
                                .unwrap(),
                            tensor(&expected["gaussian_draw"]),
                        ))
                    },
                    &context,
                )
                .unwrap();
        }
        let mut ordinal = 0;
        let uninterrupted = speech
            .generate_frame(
                &initial,
                |_, _: &NumericTensor, _| {
                    let step = &expected_steps[ordinal];
                    ordinal += 1;
                    Ok((
                        codes(&step["mixture_indices"], &context)
                            .reshape(&[1, 1], &context)
                            .unwrap(),
                        tensor(&step["gaussian_draw"]),
                    ))
                },
                &context,
            )
            .unwrap();
        compare_cache(&caches, &call["cache"], &fixture);
        let output = speech.render(&controlled, &context).unwrap();
        let replay = speech.render(&uninterrupted, &context).unwrap();
        assert_eq!(output.generated.data, codes(&call["codes"], &context).data);
        assert_eq!(output.pcm.data, replay.pcm.data);
        compare(
            &output.pcm,
            &oracle["cases"]["main_eos"]["frames"][index],
            &oracle,
        );
        assert_eq!(output.waveform_codes.data, output.generated.data);
        assert_eq!(output.state.model_position(), 6 + index as i32);
        assert_eq!(output.state.codec().frames_seen(), index as u64 + 1);
        assert_eq!(
            output.state.codec().samples_emitted(),
            12 * (index as u64 + 1)
        );
        assert_eq!(parent.codec().frames_seen(), index as u64);
        assert!(output
            .pcm
            .data
            .iter()
            .all(|v| v.is_finite() && v.abs() <= 1.));
        assert!(output.pcm.data.iter().any(|v| v.abs() > 1e-5));
        if subword[0] == 2 {
            assert_eq!(
                output.state.feedback().data,
                silence.iter().map(|&v| v as f32).collect::<Vec<_>>()
            );
            let mut branch = old_caches.clone();
            let tool = speech
                .begin_frame(
                    &parent,
                    &subword,
                    Some(&condition),
                    OutputPolicy {
                        waveform: WaveformSilence::ApplySilence,
                        ..policy
                    },
                    &mut branch,
                    &context,
                )
                .unwrap();
            let mut ordinal = 0;
            let tool = speech
                .generate_frame(
                    &tool,
                    |_, _: &NumericTensor, _| {
                        let step = &expected_steps[ordinal];
                        ordinal += 1;
                        Ok((
                            codes(&step["mixture_indices"], &context)
                                .reshape(&[1, 1], &context)
                                .unwrap(),
                            tensor(&step["gaussian_draw"]),
                        ))
                    },
                    &context,
                )
                .unwrap();
            let tool = speech.render(&tool, &context).unwrap();
            assert_eq!(tool.generated.data, output.generated.data);
            assert_eq!(tool.waveform_codes.data, tool.state.feedback().data);
            assert_ne!(tool.pcm.data, output.pcm.data);
            compare(
                &tool.pcm,
                &oracle["cases"]["tool_eos"]["frames"][index],
                &oracle,
            );
        }
        retained_frame = Some(controlled);
        state = output.state;
    }
    let (tail, finished) = speech.finish(&state, &context).unwrap();
    let tail = tail.unwrap();
    assert_eq!(tail.shape(), &[1, 1, 4]);
    compare(&tail, &oracle["cases"]["main_eos"]["tail"], &oracle);
    assert_eq!(finished.codec().frames_seen(), 2);
    assert_eq!(finished.codec().samples_emitted(), 28);
    assert_eq!(finished.model_position(), 7);
    // Schedule receipts come from the same completed component candidates,
    // including the spectral tail independently fixed by the NVIDIA fixture.
    let input_config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let input = InputState::<NumericTensor>::new(
        &input_config.perception.preprocessor,
        FrontendOptions::default(),
    )
    .unwrap();
    let backbone_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let mut backbone = super::voicechat_backbone_execution::state();
    let mut backbone_runtime = ResidentRuntime::new(
        super::voicechat_backbone_execution::model(&backbone_fixture, &context),
        &context,
    )
    .unwrap();
    let provider = eredu_architectures::voicechat::backbone_execution::OrdinaryBackbone::new(
        &mut backbone_runtime,
    );
    let progress_generator = model(&fixture, &context);
    let observed = progress::observe::<NumericBackend, _, _, _, _>(
        &input,
        &provider,
        &mut backbone,
        &progress_generator.transformer,
        &caches[..],
        Some(&finished),
    )
    .unwrap();
    assert_eq!(
        (
            observed.tts_conditional_positions,
            observed.tts_unconditional_positions
        ),
        (7, 7)
    );
    assert_eq!((observed.codec_frames, observed.output_samples), (2, 28));
    assert!(observed.codec_finished);
    let mut divergent = caches.clone();
    for cache in &mut divergent {
        cache.0.offset += 1;
    }
    assert!(matches!(
        progress::observe::<NumericBackend, _, _, _, _>(
            &input,
            &provider,
            &mut backbone,
            &progress_generator.transformer,
            &divergent[..],
            Some(&finished)
        ),
        Err(progress::ProgressError::Inconsistent(_))
    ));
    assert!(speech.finish(&finished, &context).unwrap().0.is_none());
    assert!(speech
        .begin_frame(
            &finished,
            &[3],
            None,
            OutputPolicy {
                force_silence_on_eos: false,
                silence_on_pad: &[false],
                waveform: WaveformSilence::PreserveGenerated
            },
            &mut caches,
            &context
        )
        .is_err());
    // Exact silence policy is part of retained state, even when neural geometry
    // and completed refinement agree. Rebinding it cannot reinterpret a branch.
    let mut other_generator = model(&fixture, &context);
    let mut other_silence = silence.clone();
    other_silence.rotate_left(1);
    let other_controls = FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &other_silence).unwrap();
    let mut other =
        SpeechComponents::new(&mut other_generator, &other_controls, &mut decoder).unwrap();
    assert!(other
        .render(retained_frame.as_ref().unwrap(), &context)
        .is_err());
    assert!(other
        .generate_frame(
            retained_frame.as_ref().unwrap(),
            |_, _: &NumericTensor, _| { panic!("policy mismatch must be rejected before a draw") },
            &context
        )
        .is_err());
    assert!(other.finish(&state, &context).is_err());
}
