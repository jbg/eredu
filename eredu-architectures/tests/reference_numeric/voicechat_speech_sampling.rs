//! Recorded NVIDIA random draws drive actual shared speech refinement.
use super::voicechat_speech::codec;
use super::voicechat_transformer::CausalCache;
use super::voicechat_tts_generator::{codes, compare_cache, model, tensor};
use super::*;
use eredu_architectures::voicechat::{
    speech::SpeechComponents,
    speech_sampling::SpeechSamplingError,
    tts_feedback::{FeedbackControls, OutputPolicy, WaveformSilence},
    tts_ingress::{PromptLayout, SpeakerPrompt},
};
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_runtime::{
    DenseSamplingBackend, GumbelTruncation, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeScheduleState,
};
use serde_json::Value;
use std::{cell::Cell, convert::Infallible, rc::Rc};

#[derive(Clone)]
pub(super) struct Tape {
    draws: Arc<Vec<Value>>,
    pub(super) cursor: usize,
    pub(super) fail_at: Option<usize>,
    malformed_at: Option<usize>,
    pub(super) issued: Rc<Cell<usize>>,
}
impl Tape {
    pub(super) fn new(call: &Value) -> Self {
        Self {
            draws: Arc::new(
                call["refinement"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|step| [step["uniform_draw"].clone(), step["gaussian_draw"].clone()])
                    .collect(),
            ),
            cursor: 0,
            fail_at: None,
            malformed_at: None,
            issued: Rc::new(Cell::new(0)),
        }
    }
    fn draw(&mut self, shape: &[i32], gaussian: bool) -> Result<NumericTensor, Error> {
        assert_eq!(
            self.cursor % 2 == 1,
            gaussian,
            "uniform must precede Gaussian"
        );
        if self.fail_at == Some(self.cursor) {
            return Err(Error::backend_source(std::io::Error::other(
                "injected Gaussian failure",
            )));
        }
        let mut output = tensor(&self.draws[self.cursor]);
        assert_eq!(
            output.shape(),
            shape,
            "draw geometry comes from the NVIDIA call"
        );
        if self.malformed_at == Some(self.cursor) {
            output = NumericTensor::new(vec![1, 1, 1], vec![0.]);
        }
        self.cursor += 1;
        self.issued.set(self.issued.get() + 1);
        Ok(output)
    }
}
pub(super) struct Recorded;
impl SamplingBackend for Recorded {
    type Logits = NumericTensor;
    type Token = NumericTensor;
    type RandomState = Tape;
    type Context = NumericContext;
    type Error = Error;
    fn error(s: String) -> Error {
        Error::backend(s)
    }
    fn validate_token(
        t: &NumericTensor,
        d: TokenDomain,
        c: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        NumericBackend::validate_token(t, d, c)
    }
    fn scale_temperature(
        _: &NumericTensor,
        _: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_penalties(
        _: &NumericTensor,
        _: &[u32],
        _: PenaltyConfig,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_top_k(_: NumericTensor, _: i32, _: &NumericContext) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_top_p(_: NumericTensor, _: f32, _: &NumericContext) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_min_p(_: NumericTensor, _: f32, _: &NumericContext) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_token_filter(
        _: &NumericTensor,
        _: &TokenFilter,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_mirostat(
        _: &NumericTensor,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn sample_raw(
        _: &NumericTensor,
        _: f32,
        _: Option<&mut Tape>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn sample_processed(
        _: &NumericTensor,
        _: f32,
        _: Option<&mut Tape>,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn token_id(_: &NumericTensor, _: &NumericContext) -> Result<u32, Error> {
        unreachable!()
    }
    fn token_probability(_: &NumericTensor, _: u32, _: &NumericContext) -> Result<f32, Error> {
        unreachable!()
    }
}
impl DenseSamplingBackend for Recorded {
    fn uniform_f32(
        shape: &[i32],
        random: &mut Tape,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        random.draw(shape, false)
    }
    fn standard_normal_f32(
        shape: &[i32],
        random: &mut Tape,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        random.draw(shape, true)
    }
    fn dense_shape(t: &NumericTensor) -> &[i32] {
        t.shape()
    }
    fn gumbel_max(
        logits: &NumericTensor,
        uniform: &NumericTensor,
        truncation: GumbelTruncation,
        epsilon: f32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        // Independent scalar realization of the declared dense mechanism. Exact
        // mixture/RVQ outputs are checked against NVIDIA, not this calculation.
        let GumbelTruncation::TopP(p) = truncation else {
            panic!("released nucleus policy")
        };
        assert_eq!(p, 0.95);
        assert_eq!(epsilon, 1e-8);
        let width = *logits.shape().last().unwrap() as usize;
        let mut ids = Vec::new();
        for (row, draw) in logits
            .data
            .chunks_exact(width)
            .zip(uniform.data.chunks_exact(width))
        {
            let mut order: Vec<_> = (0..width).collect();
            order.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<_> = row.iter().map(|x| (*x - max).exp()).collect();
            let sum: f32 = weights.iter().sum();
            let mut cumulative = 0.;
            let mut keep = vec![false; width];
            for &id in &order {
                cumulative += weights[id] / sum;
                keep[id] = cumulative > (1. - p) as f32;
            }
            keep[*order.last().unwrap()] = true;
            let sum: f32 = weights
                .iter()
                .enumerate()
                .filter(|(id, _)| keep[*id])
                .map(|(_, v)| v)
                .sum();
            let selected = (0..width)
                .filter(|&id| keep[id])
                .map(|id| {
                    let score =
                        (weights[id] / sum).ln() - (-(draw[id] + epsilon).ln() + epsilon).ln();
                    (id, score)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            ids.push(selected as i32);
        }
        NumericTensor::from_i32_slice(&ids, &logits.shape()[..2], context)
    }
}
#[derive(Clone)]
pub(super) struct Host;
impl SemanticStateTransaction for Host {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, _: Self) -> Result<(), Infallible> {
        Ok(())
    }
}
impl RealtimeScheduleState for Host {
    type Identity = ();
    type Error = Infallible;
    fn identity(&self) -> &() {
        &()
    }
    fn validate_successor(&self, _: &Self) -> Result<(), Infallible> {
        Ok(())
    }
}
pub(super) struct Ready;
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
pub(super) type Generation = RealtimeGenerationState<Host, Host, (), (), Tape, Ready>;
pub(super) fn generation(random: Option<Tape>) -> Generation {
    Generation::new(
        Host,
        RealtimeGenerationContract::new((), 0),
        Host,
        (),
        vec![],
        random,
    )
    .unwrap()
}

#[test]
fn voicechat_speech_sampling_replays_nvidia_draws_and_rolls_back_complete_operations() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut generator = model(&fixture, &context);
    let mut decoder = codec(&fixture, &context);
    let silence: Vec<_> = fixture["silence"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let controls = FeedbackControls::new(5, 2, 12, true, &[5, 6, 7], &silence).unwrap();
    let mut speech = SpeechComponents::new(&mut generator, &controls, &mut decoder).unwrap();
    let prompt = &fixture["calls"][0];
    let call = &fixture["calls"][1];
    let mask = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
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
    let tape = Tape::new(call);
    let mut canonical = generation(Some(tape.clone()));
    let mut ordinary = canonical.branch().unwrap();
    let complete = speech
        .generate_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut ordinary, None, &context)
        .unwrap();
    assert_eq!(
        complete.tts().codes().unwrap().data,
        codes(&call["codes"], &context).data
    );
    assert_eq!(
        ordinary.random_state().unwrap().cursor,
        10,
        "five active slots, two draws each"
    );
    assert_eq!(canonical.random_state().unwrap().cursor, 0);
    let mut controlled = canonical.branch().unwrap();
    let mut partial = frame.clone();
    for expected in call["refinement"].as_array().unwrap() {
        assert_eq!(
            partial.tts().refinement().next_slot(),
            Some(expected["slot"].as_u64().unwrap() as usize)
        );
        partial = speech
            .advance_sampled_frame::<Recorded, _, _, _, _, _>(
                &partial,
                &mut controlled,
                None,
                &context,
            )
            .unwrap();
        assert_eq!(
            partial.tts().refinement().codes().data,
            codes(&expected["codes"], &context).data
        );
    }
    assert_eq!(
        partial.tts().codes().unwrap().data,
        complete.tts().codes().unwrap().data
    );
    assert_eq!(controlled.random_state().unwrap().cursor, 10);
    compare_cache(&caches, &call["cache"], &fixture);
    let left = speech.render(&partial, &context).unwrap();
    let right = speech.render(&complete, &context).unwrap();
    assert_eq!(left.pcm.data, right.pcm.data);
    assert_eq!(left.state.feedback().data, right.state.feedback().data);

    // A bound codec failure occurs after all refinement draws. Its failed
    // candidate must not advance RNG, codec carry or the causal cache. Repairing
    // the binding and retrying the same branch reproduces separate refinement
    // and rendering exactly.
    let mut broken_decoder = codec(&fixture, &context);
    struct MalformedCodebook;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for MalformedCodebook {
        fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
            if metadata.id.as_str().ends_with(".mus_list.0") {
                *output = NumericTensor::new(vec![1, 1], vec![0.]);
            }
        }
    }
    broken_decoder.visit_parameters_mut(&mut MalformedCodebook);
    let mut other_generator = model(&fixture, &context);
    let mut broken =
        SpeechComponents::new(&mut other_generator, &controls, &mut broken_decoder).unwrap();
    let mut rendered = canonical.branch().unwrap();
    let draws_before = tape.issued.get();
    let error = broken
        .render_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut rendered, None, &context)
        .unwrap_err();
    assert!(matches!(
        error,
        SpeechSamplingError::Refinement(eredu_architectures::voicechat::tts_mog::TtsError::Codec(
            eredu_codec::Error::InvalidShape(_)
        ))
    ));
    assert_eq!(tape.issued.get() - draws_before, 10);
    assert_eq!(rendered.random_state().unwrap().cursor, 0);
    assert_eq!(state.codec().frames_seen(), 0);
    assert_eq!(state.codec().samples_emitted(), 0);
    assert_eq!(frame.tts().refinement().completed_depth(), 0);
    let recovered = speech
        .render_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut rendered, None, &context)
        .unwrap();
    assert_eq!(rendered.random_state().unwrap().cursor, 10);
    assert_eq!(recovered.generated.data, right.generated.data);
    assert_eq!(recovered.waveform_codes.data, right.waveform_codes.data);
    assert_eq!(recovered.pcm.data, right.pcm.data);
    assert_eq!(recovered.state.feedback().data, right.state.feedback().data);
    assert_eq!(recovered.state.codec().frames_seen(), 1);
    assert_eq!(recovered.state.codec().samples_emitted(), 12);
    compare_cache(&caches, &call["cache"], &fixture);

    // Error after the first draw and error after successful draws in multiple
    // slots both preserve the stream. A malformed normal reaches neural validation.
    for (fail_at, malformed_at) in [(Some(1), None), (Some(5), None), (None, Some(1))] {
        let mut tape = tape.clone();
        tape.fail_at = fail_at;
        tape.malformed_at = malformed_at;
        let state = generation(Some(tape));
        let mut failed = state.branch().unwrap();
        let error = speech
            .render_sampled_frame::<Recorded, _, _, _, _, _>(&frame, &mut failed, None, &context)
            .unwrap_err();
        if fail_at.is_some() {
            let mut source: &(dyn std::error::Error + 'static) = &error;
            while let Some(next) = source.source() {
                source = next;
            }
            assert!(
                source.is::<std::io::Error>(),
                "native source survives the portable boundary"
            );
        }
        assert_eq!(failed.random_state().unwrap().cursor, 0);
        assert_eq!(frame.tts().refinement().completed_depth(), 0);
    }
    let mut missing = generation(None).branch().unwrap();
    assert!(matches!(
        speech.advance_sampled_frame::<Recorded, _, _, _, _, _>(
            &frame,
            &mut missing,
            None,
            &context
        ),
        Err(SpeechSamplingError::MissingRandomState)
    ));
    assert!(
        speech
            .generate_sampled_frame::<Recorded, _, _, _, _, _>(
                &complete,
                &mut missing,
                None,
                &context
            )
            .is_ok()
    );
    assert!(missing.random_state().is_none());
    let rendered_complete = speech
        .render_sampled_frame::<Recorded, _, _, _, _, _>(&complete, &mut missing, None, &context)
        .unwrap();
    assert_eq!(rendered_complete.pcm.data, right.pcm.data);
    assert!(missing.random_state().is_none());
    assert!(
        speech
            .advance_sampled_frame::<Recorded, _, _, _, _, _>(
                &complete,
                &mut ordinary,
                None,
                &context
            )
            .is_err()
    );
    assert_eq!(ordinary.random_state().unwrap().cursor, 10);
    ordinary.attach_submission_completion(Ready).unwrap();
    assert!(matches!(
        speech.generate_sampled_frame::<Recorded, _, _, _, _, _>(
            &complete,
            &mut ordinary,
            None,
            &context
        ),
        Err(SpeechSamplingError::Submitted)
    ));
    assert!(matches!(
        speech.render_sampled_frame::<Recorded, _, _, _, _, _>(
            &frame,
            &mut ordinary,
            None,
            &context
        ),
        Err(SpeechSamplingError::Submitted)
    ));
    canonical.commit_branch(ordinary).unwrap();
    assert_eq!(canonical.random_state().unwrap().cursor, 10);
}
