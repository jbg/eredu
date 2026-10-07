//! Owner-only recorded draws over the ordinary retained speech refinement.
use super::voicechat_rnnt_coordination::coordinator_pair;
use super::voicechat_speech_sampling::{Recorded, Tape, generation};
use super::voicechat_transformer::CausalCache;
use super::voicechat_tts_generator::{codes, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    speech_sampling::SpeechSamplingCoordination,
    tts_generator::{TtsFrame, TtsGenerator, TtsGeneratorParts},
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_mog::{MogDimensions, MogHead, MogPrediction, MogSample, TtsError},
    tts_mog_layered::MogExecution,
    tts_transformer::TtsTransformer,
};
use eredu_core::scheduler::SemanticStateTransaction;
use serde_json::Value;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    None,
    BeforeDraw,
    AfterDraw,
}
struct ObservedMog {
    inner: MogHead<NumericBackend>,
    failure: Failure,
    predictions: usize,
    selections: usize,
    failure_selection: usize,
}
impl MogExecution<NumericBackend> for ObservedMog {
    fn dimensions(&self) -> MogDimensions {
        self.inner.dimensions()
    }
    fn predict(
        &mut self,
        hidden: &NumericTensor,
        guidance: f32,
        context: &NumericContext,
    ) -> Result<MogPrediction<NumericTensor>, TtsError> {
        self.predictions += 1;
        if self.failure == Failure::BeforeDraw {
            return Err(TtsError::Shape("injected before-draw failure"));
        }
        self.inner.predict(hidden, guidance, context)
    }
    fn selected(
        &mut self,
        prediction: &MogPrediction<NumericTensor>,
        mixture: &NumericTensor,
        gaussian: &NumericTensor,
        noise_scale: f32,
        context: &NumericContext,
    ) -> Result<MogSample<NumericTensor>, TtsError> {
        self.selections += 1;
        let sample = self
            .inner
            .selected(prediction, mixture, gaussian, noise_scale, context)?;
        if self.failure == Failure::AfterDraw && self.selections == self.failure_selection {
            return Err(TtsError::Shape("injected after-draw failure"));
        }
        Ok(sample)
    }
}
type Generator = TtsGenerator<NumericBackend, TtsTransformer<NumericBackend>, ObservedMog>;

fn prepared(
    fixture: &Value,
    context: &NumericContext,
) -> (Generator, TtsFrame<NumericTensor>, [CausalCache; 2]) {
    let ordinary = model(fixture, context);
    let config = ordinary.config().clone();
    let parts = ordinary.into_parts();
    let mut generator = TtsGenerator::from_parts(
        &config,
        16,
        5,
        TtsGeneratorParts::new(
            parts.conditioning,
            parts.character_transformer,
            parts.transformer,
            parts.refinement,
            ObservedMog {
                inner: parts.mog,
                failure: Failure::None,
                predictions: 0,
                selections: 0,
                failure_selection: 0,
            },
        ),
    )
    .unwrap();
    let prompt = &fixture["calls"][0];
    let call = &fixture["calls"][1];
    let mask = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_bool().unwrap())
        .collect::<Vec<_>>();
    let mut caches = [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ];
    generator
        .prefill(
            &codes(&prompt["input_codes"], context),
            &PromptLayout::new(1, 5, &mask).unwrap(),
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut caches,
            context,
        )
        .unwrap();
    let frame = generator
        .begin_frame(
            &codes(&call["input_codes"], context),
            Some(&tensor(&call["supplied_subwords"])),
            &mut caches,
            context,
        )
        .unwrap();
    (generator, frame, caches)
}

#[test]
fn voicechat_speech_coordination_owner_draws_failure_rollback_and_replay() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let (coordinators, timeouts) = coordinator_pair();
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let fixture = &fixture;
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let call = &fixture["calls"][1];
                    let (mut generator, frame, caches) = prepared(fixture, &context);
                    let tape = Tape::new(call);
                    let canonical = generation((rank == 0).then(|| tape.clone()));
                    let mut branch = canonical.branch().unwrap();

                    // Admission mismatch must not invoke a learned primitive or draw.
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 10 + rank as u64]);
                    assert!(
                        generator
                            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                &frame,
                                &mut branch,
                                Some(&mut coordination),
                                &context
                            )
                            .is_err()
                    );
                    assert_eq!(generator.mog.predictions, 0);
                    assert_eq!(generator.mog.selections, 0);
                    assert_eq!(tape.issued.get(), 0);

                    // Missing owner state is an agreed admission failure, while
                    // the same absence remains valid for an ordinary follower.
                    let mut missing = generation(None).branch().unwrap();
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 15]);
                    assert!(
                        generator
                            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                &frame,
                                &mut missing,
                                Some(&mut coordination),
                                &context,
                            )
                            .is_err()
                    );
                    assert_eq!(generator.mog.predictions, 0);
                    assert_eq!(tape.issued.get(), 0);

                    // Whole-operation and one-slot advancement cannot share an identity.
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 20]);
                    let result = if rank == 0 {
                        generator.advance_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame,
                            &mut branch,
                            Some(&mut coordination),
                            &context,
                        )
                    } else {
                        generator.generate_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame,
                            &mut branch,
                            Some(&mut coordination),
                            &context,
                        )
                    };
                    assert!(result.is_err());
                    assert_eq!(generator.mog.predictions, 0);
                    assert_eq!(tape.issued.get(), 0);

                    // A retained candidate at a different slot is refused before work.
                    let mut local = generation(Some(Tape::new(call))).branch().unwrap();
                    let next = generator
                        .advance_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame, &mut local, None, &context,
                        )
                        .unwrap();
                    let predictions = generator.mog.predictions;
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 30]);
                    assert!(
                        generator
                            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                if rank == 0 { &frame } else { &next },
                                &mut branch,
                                Some(&mut coordination),
                                &context
                            )
                            .is_err()
                    );
                    assert_eq!(generator.mog.predictions, predictions);
                    assert_eq!(tape.issued.get(), 0);

                    for (invocation, failure) in
                        [(40, Failure::BeforeDraw), (50, Failure::AfterDraw)]
                    {
                        generator.mog.failure = if rank == 1 { failure } else { Failure::None };
                        generator.mog.failure_selection = generator.mog.selections + 3;
                        let draws = tape.issued.get();
                        let mut coordination =
                            SpeechSamplingCoordination::new(&mut coordinator, [9, 2, invocation]);
                        assert!(
                            generator
                                .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                    &frame,
                                    &mut branch,
                                    Some(&mut coordination),
                                    &context
                                )
                                .is_err()
                        );
                        assert_eq!(
                            branch.random_state().map(|random| random.cursor),
                            if rank == 0 { Some(0) } else { None }
                        );
                        assert_eq!(
                            tape.issued.get() - draws,
                            if rank == 0 && failure == Failure::AfterDraw {
                                6
                            } else {
                                0
                            }
                        );
                        assert_eq!(frame.refinement().completed_depth(), 0);
                        assert_eq!(frame.refinement().next_slot(), Some(3));
                        assert_eq!(frame.refinement().codes().data, vec![5.; 31]);
                        compare_cache(&caches, &call["cache"], fixture);
                        generator.mog.failure = Failure::None;
                    }

                    // Owner-side Gaussian failure follows a successful uniform draw.
                    // The follower has no RNG and exits the same agreed phase.
                    let mut failing_tape = Tape::new(call);
                    failing_tape.fail_at = Some(1);
                    let failed = generation((rank == 0).then(|| failing_tape.clone()));
                    let mut failed = failed.branch().unwrap();
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 55]);
                    assert!(
                        generator
                            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                &frame,
                                &mut failed,
                                Some(&mut coordination),
                                &context,
                            )
                            .is_err()
                    );
                    assert_eq!(
                        failed.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(0) } else { None }
                    );
                    assert_eq!(failing_tape.issued.get(), usize::from(rank == 0));
                    assert_eq!(frame.refinement().completed_depth(), 0);

                    // Retrying the unchanged branch consumes only the owner's stream.
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [9, 2, 60]);
                    let complete = generator
                        .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame,
                            &mut branch,
                            Some(&mut coordination),
                            &context,
                        )
                        .unwrap();
                    assert_eq!(
                        complete.codes().unwrap().data,
                        codes(&call["codes"], &context).data
                    );
                    assert_eq!(
                        branch.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(10) } else { None }
                    );
                    let predictions = generator.mog.predictions;
                    // Consumed operation authority never silently authorizes a second invocation.
                    assert!(
                        generator
                            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                                &frame,
                                &mut branch,
                                Some(&mut coordination),
                                &context
                            )
                            .is_err()
                    );
                    assert_eq!(generator.mog.predictions, predictions);

                    let mut controlled = canonical.branch().unwrap();
                    let mut partial = frame.clone();
                    for (index, expected) in
                        call["refinement"].as_array().unwrap().iter().enumerate()
                    {
                        let mut coordination = SpeechSamplingCoordination::new(
                            &mut coordinator,
                            [9, 2, 70 + index as u64],
                        );
                        partial = generator
                            .advance_sampled_frame::<Recorded, _, _, _, _, _>(
                                &partial,
                                &mut controlled,
                                Some(&mut coordination),
                                &context,
                            )
                            .unwrap();
                        assert_eq!(
                            partial.refinement().codes().data,
                            codes(&expected["codes"], &context).data
                        );
                        assert_eq!(
                            controlled.random_state().map(|random| random.cursor),
                            if rank == 0 {
                                Some((index + 1) * 2)
                            } else {
                                None
                            }
                        );
                    }
                    assert_eq!(
                        partial.codes().unwrap().data,
                        complete.codes().unwrap().data
                    );
                    assert_eq!(
                        canonical.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(0) } else { None }
                    );
                    compare_cache(&caches, &call["cache"], fixture);
                    complete.codes().unwrap().data.clone()
                })
            })
            .collect::<Vec<_>>();
        let outputs = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], outputs[1]);
    });
    assert_eq!(
        timeouts.load(Ordering::SeqCst),
        0,
        "asymmetric failures cannot strand the other refinement participant"
    );
}

#[test]
fn voicechat_speech_coordination_render_failure_preserves_rng_and_codec_until_retry() {
    use super::voicechat_speech::codec;
    use eredu_architectures::voicechat::{
        speech::SpeechComponents,
        tts_feedback::{FeedbackControls, OutputPolicy, WaveformSilence},
    };
    use eredu_codec::{
        ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderExecution, AudioDecoderState},
        ear_decoder::DecoderState,
    };
    use std::{cell::Cell, rc::Rc};

    struct ObservedCodec {
        inner: AudioDecoder<NumericBackend>,
        calls: Rc<Cell<usize>>,
    }
    impl AudioDecoderExecution<NumericBackend> for ObservedCodec {
        type NeuralState = DecoderState<NumericTensor>;
        fn config(&self) -> &AudioDecoderConfig {
            self.inner.config()
        }
        fn initial_state(
            &self,
            batch: i32,
        ) -> Result<AudioDecoderState<NumericTensor>, eredu_codec::Error> {
            self.inner.initial_state(batch)
        }
        fn step(
            &mut self,
            codes: &NumericTensor,
            state: &AudioDecoderState<NumericTensor>,
            constrain: bool,
            context: &NumericContext,
        ) -> Result<(NumericTensor, AudioDecoderState<NumericTensor>), eredu_codec::Error> {
            self.calls.set(self.calls.get() + 1);
            self.inner.step(codes, state, constrain, context)
        }
        fn finish(
            &mut self,
            state: &AudioDecoderState<NumericTensor>,
            constrain: bool,
            context: &NumericContext,
        ) -> Result<(Option<NumericTensor>, AudioDecoderState<NumericTensor>), eredu_codec::Error>
        {
            self.inner.finish(state, constrain, context)
        }
    }
    struct MalformedCodebook;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for MalformedCodebook {
        fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
            if metadata.id.as_str().ends_with(".mus_list.0") {
                *output = NumericTensor::new(vec![1, 1], vec![0.]);
            }
        }
    }

    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let (coordinators, timeouts) = coordinator_pair();
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let fixture = &fixture;
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut generator = model(fixture, &context);
                    let codec_calls = Rc::new(Cell::new(0));
                    let mut decoder = ObservedCodec {
                        inner: codec(fixture, &context),
                        calls: codec_calls.clone(),
                    };
                    if rank == 1 {
                        decoder.inner.visit_parameters_mut(&mut MalformedCodebook);
                    }
                    let silence = fixture["silence"]["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_i64().unwrap() as i32)
                        .collect::<Vec<_>>();
                    let controls =
                        FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &silence).unwrap();
                    let mut speech =
                        SpeechComponents::new(&mut generator, &controls, &mut decoder).unwrap();
                    let prompt = &fixture["calls"][0];
                    let call = &fixture["calls"][1];
                    let mask = prompt["audio_mask"]["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_bool().unwrap())
                        .collect::<Vec<_>>();
                    let mut caches = [
                        CausalCache(NumericCache::new(Some(3))),
                        CausalCache(NumericCache::new(None)),
                    ];
                    let state = speech
                        .prefill(
                            &codes(&prompt["input_codes"], &context),
                            &PromptLayout::new(1, 5, &mask).unwrap(),
                            SpeakerPrompt::Aria,
                            Some(&tensor(&prompt["supplied_subwords"])),
                            &mut caches,
                            &context,
                        )
                        .unwrap();
                    let mut policy_caches = caches.clone();
                    let frame = speech
                        .begin_frame(
                            &state,
                            &[3],
                            Some(&tensor(&call["supplied_subwords"])),
                            OutputPolicy {
                                force_silence_on_eos: true,
                                silence_on_pad: &[false],
                                waveform: WaveformSilence::PreserveGenerated,
                            },
                            &mut caches,
                            &context,
                        )
                        .unwrap();
                    let other_policy = speech
                        .begin_frame(
                            &state,
                            &[3],
                            Some(&tensor(&call["supplied_subwords"])),
                            OutputPolicy {
                                force_silence_on_eos: false,
                                silence_on_pad: &[false],
                                waveform: WaveformSilence::PreserveGenerated,
                            },
                            &mut policy_caches,
                            &context,
                        )
                        .unwrap();
                    let tape = Tape::new(call);
                    let canonical = generation((rank == 0).then(|| tape.clone()));
                    let mut branch = canonical.branch().unwrap();

                    // Output policy agreement occurs after refinement, before either
                    // participant may call the codec (including its collectives).
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [12, 3, 1]);
                    assert!(
                        speech
                            .render_sampled_frame::<Recorded, _, _, _, _, _>(
                                if rank == 0 { &frame } else { &other_policy },
                                &mut branch,
                                Some(&mut coordination),
                                &context,
                            )
                            .is_err()
                    );
                    assert_eq!(codec_calls.get(), 0);
                    assert_eq!(tape.issued.get(), if rank == 0 { 10 } else { 0 });
                    assert_eq!(
                        branch.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(0) } else { None }
                    );

                    // The follower's invalid learned codebook fails after all draws;
                    // the owner's successfully rendered candidate remains unpublished.
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [12, 3, 2]);
                    assert!(
                        speech
                            .render_sampled_frame::<Recorded, _, _, _, _, _>(
                                &frame,
                                &mut branch,
                                Some(&mut coordination),
                                &context,
                            )
                            .is_err()
                    );
                    assert_eq!(codec_calls.get(), 1);
                    assert_eq!(tape.issued.get(), if rank == 0 { 20 } else { 0 });
                    assert_eq!(
                        branch.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(0) } else { None }
                    );
                    assert_eq!(state.codec().frames_seen(), 0);
                    assert_eq!(state.codec().samples_emitted(), 0);
                    assert_eq!(frame.tts().refinement().completed_depth(), 0);
                    compare_cache(&caches, &call["cache"], fixture);

                    drop(speech);
                    decoder.inner = codec(fixture, &context);
                    let mut speech =
                        SpeechComponents::new(&mut generator, &controls, &mut decoder).unwrap();
                    let mut coordination =
                        SpeechSamplingCoordination::new(&mut coordinator, [12, 3, 3]);
                    let recovered = speech
                        .render_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame,
                            &mut branch,
                            Some(&mut coordination),
                            &context,
                        )
                        .unwrap();
                    assert_eq!(codec_calls.get(), 2);
                    assert_eq!(
                        branch.random_state().map(|random| random.cursor),
                        if rank == 0 { Some(10) } else { None }
                    );
                    assert_eq!(
                        recovered.generated.data,
                        codes(&call["codes"], &context).data
                    );

                    let mut ordinary_generator = model(fixture, &context);
                    let mut ordinary_decoder = codec(fixture, &context);
                    let mut ordinary = SpeechComponents::new(
                        &mut ordinary_generator,
                        &controls,
                        &mut ordinary_decoder,
                    )
                    .unwrap();
                    let mut ordinary_branch = generation(Some(Tape::new(call))).branch().unwrap();
                    let expected = ordinary
                        .render_sampled_frame::<Recorded, _, _, _, _, _>(
                            &frame,
                            &mut ordinary_branch,
                            None,
                            &context,
                        )
                        .unwrap();
                    assert_eq!(recovered.pcm.data, expected.pcm.data);
                    assert_eq!(recovered.waveform_codes.data, expected.waveform_codes.data);
                    assert_eq!(
                        recovered.state.feedback().data,
                        expected.state.feedback().data
                    );
                    let mut actual_carry = Vec::new();
                    eredu_nn::RetainedTensorRoots::visit_retained_tensors(
                        recovered.state.codec(),
                        &mut |tensor: &NumericTensor| {
                            actual_carry.push((tensor.shape().to_vec(), tensor.data.clone()));
                        },
                    );
                    let mut expected_carry = Vec::new();
                    eredu_nn::RetainedTensorRoots::visit_retained_tensors(
                        expected.state.codec(),
                        &mut |tensor: &NumericTensor| {
                            expected_carry.push((tensor.shape().to_vec(), tensor.data.clone()));
                        },
                    );
                    assert!(!actual_carry.is_empty());
                    assert_eq!(actual_carry, expected_carry);
                    assert_eq!(recovered.state.codec().frames_seen(), 1);
                    assert_eq!(recovered.state.codec().samples_emitted(), 12);
                    assert_eq!(state.codec().frames_seen(), 0);
                    assert_eq!(state.codec().samples_emitted(), 0);
                    compare_cache(&caches, &call["cache"], fixture);
                    recovered.pcm.data
                })
            })
            .collect::<Vec<_>>();
        let outputs = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], outputs[1]);
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
