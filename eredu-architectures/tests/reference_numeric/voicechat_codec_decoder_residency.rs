//! Pinned EAR neural units and waveform lifecycle through ordinary parameter providers.
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{codes, compare, tensor};
use super::*;
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderExecution, AudioDecoderState},
    ear_decoder::{DecoderConfig, DecoderState, DecoderUnit, NeuralDecoder},
    ear_decoder_layered::{DecoderArchitecture, DecoderExecution, LayerwiseDecoder},
    ear_spectral::SpectralConfig,
    prvq::Geometry,
    streaming::StreamPhase,
};
use serde_json::Value;

const PREFIX: &str = "tts_model.audio_codec.decoder";
type Unit = DecoderUnit<NumericTensor>;
struct BoundFixturePolicy {
    inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::begin(
            &mut self.inner,
            input,
            context,
        )
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit, E>,
    {
        let parameters = &self.parameters;
        self.inner.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
                replace_numeric_parameters(&mut unit, parameters);
                Ok(unit)
            },
            context,
        )
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        retained: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        self.inner
            .complete(ordinal, address, lease, output, state, retained, context)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::finish(
            &mut self.inner,
            output,
            context,
        )
    }
}
fn geometry(value: &Value) -> DecoderConfig {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    DecoderConfig {
        latent_size: i("latent_size"),
        base_hidden_size: i("base_hidden_size"),
        channel_mult: serde_json::from_value(value["channel_mult"].clone()).unwrap(),
        rates: serde_json::from_value(value["rates"].clone()).unwrap(),
        blocks_per_stage: i("num_blocks") as usize,
        kernel_size: i("kernel_size"),
        fft_size: i("n_fft"),
        groups: i("groups"),
    }
}
fn bounded(
    config: DecoderConfig,
    parameters: &Value,
) -> LayerwiseDecoder<NumericBackend, BoundFixturePolicy> {
    LayerwiseDecoder::new(
        DecoderArchitecture::new(config, PREFIX).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: weights(parameters),
        },
    )
}
fn resident(
    config: DecoderConfig,
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseDecoder<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let count = config.unit_specs().unwrap().len();
    let architecture = DecoderArchitecture::new(config, PREFIX).unwrap();
    let parameters = weights(parameters);
    let units = (0..count)
        .map(|index| {
            let mut unit = architecture.build_unit(0, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    LayerwiseDecoder::new(architecture, NumericReplicatedPolicy::resident(units))
}
fn events(count: usize) -> Vec<StreamPolicyEvent> {
    let mut events = vec![StreamPolicyEvent::Begin];
    for index in 0..count {
        events.extend([
            StreamPolicyEvent::Acquire(index, 0, index),
            StreamPolicyEvent::Complete(index, 0, index),
        ]);
    }
    events.push(StreamPolicyEvent::Finish);
    events
}
fn compare_neural(state: &DecoderState<NumericTensor>, cache: &Value, fixture: &Value) {
    assert_eq!(state.phase(), StreamPhase::AcceptingInput);
    let carries = state
        .units()
        .iter()
        .filter_map(|unit| unit.carry())
        .collect::<Vec<_>>();
    // The last independently recorded cache belongs to complex spectral overlap.
    assert_eq!(carries.len(), cache.as_object().unwrap().len() - 1);
    for (index, carry) in carries.into_iter().enumerate() {
        compare(carry, &cache[index.to_string()], fixture);
    }
}

#[test]
fn voicechat_codec_decoder_resident_and_bounded_match_nvidia_carry_and_retry_late_failure() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let case = &fixture["decoders"]["compact"];
    let config = geometry(&case["geometry"]);
    let count = config.unit_specs().unwrap().len();
    let mut ordinary =
        NeuralDecoder::<NumericBackend>::new(config.clone(), PREFIX, &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&case["parameters"]));
    let mut resident = resident(config.clone(), &case["parameters"], &context);
    let mut bounded = bounded(config, &case["parameters"]);
    let full = &case["full"][0];
    for output in [
        ordinary.forward(&tensor(&full["input"]), &context).unwrap(),
        resident.forward(&tensor(&full["input"]), &context).unwrap(),
        bounded.forward(&tensor(&full["input"]), &context).unwrap(),
    ] {
        compare(&output, &full["raw"], &fixture);
    }
    assert_eq!(bounded.runtime().policy().inner.events, events(count));
    let mut ordinary_state = ordinary.initial_state().unwrap();
    let mut resident_state = resident.initial_state().unwrap();
    let mut bounded_state = bounded.initial_state().unwrap();
    let steps = case["streaming"].as_array().unwrap();
    for (index, step) in steps.iter().enumerate() {
        let input = tensor(&step["input"]);
        bounded.runtime_mut().policy_mut().inner.events.clear();
        if index == 1 {
            bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(count - 1);
            let error = bounded.step(&input, &bounded_state, &context).unwrap_err();
            assert!(error
                .to_string()
                .contains("injected dense-stream acquisition failure"));
            compare_neural(&bounded_state, &steps[0]["cache"], &fixture);
            assert_eq!(
                bounded.runtime().policy().inner.events,
                events(count)[..2 * count]
            );
            bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
            bounded.runtime_mut().policy_mut().inner.events.clear();
        }
        let (direct, direct_state) = ordinary.step(&input, &ordinary_state, &context).unwrap();
        let (leased, leased_state) = resident.step(&input, &resident_state, &context).unwrap();
        let (streamed, streamed_state) = bounded.step(&input, &bounded_state, &context).unwrap();
        for output in [&direct, &leased, &streamed] {
            compare(output, &step["raw"], &fixture);
        }
        assert_eq!(bounded.runtime().policy().inner.events, events(count));
        if !step["flush"].as_bool().unwrap() {
            for state in [&direct_state, &leased_state, &streamed_state] {
                compare_neural(state, &step["cache"], &fixture);
            }
        }
        ordinary_state = direct_state;
        resident_state = leased_state;
        bounded_state = streamed_state;
    }
    bounded.runtime_mut().policy_mut().inner.events.clear();
    let finished = bounded.finish(&bounded_state).unwrap();
    assert_eq!(finished.phase(), StreamPhase::Finished);
    assert!(finished.units().iter().all(|unit| unit.carry().is_none()));
    assert!(bounded.runtime().policy().inner.events.is_empty());
    assert!(bounded
        .step(&tensor(&steps[0]["input"]), &finished, &context)
        .is_err());
    assert!(bounded.runtime().policy().inner.events.is_empty());
}

fn compare_audio_state(state: &AudioDecoderState<NumericTensor>, cache: &Value, fixture: &Value) {
    compare_neural(state.neural(), cache, fixture);
    let spectral_index = cache.as_object().unwrap().len() - 1;
    let expected = &cache[spectral_index.to_string()];
    let (real, imag) = state.spectral().history().unwrap();
    compare(real, &expected["real"], fixture);
    compare(imag, &expected["imag"], fixture);
}

#[test]
fn voicechat_codec_decoder_bounded_waveform_matches_nvidia_startup_fork_replay_and_flush() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let case = &fixture["composition"];
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: SpectralConfig {
            fft_size: case["geometry"]["n_fft"].as_i64().unwrap() as i32,
            hop_length: case["geometry"]["hop_length"].as_i64().unwrap() as i32,
        },
    };
    let quantizer = super::voicechat_codec_quantizer_residency::bounded(
        config.quantizer,
        weights(&case["parameters"]),
    );
    let neural = bounded(config.neural.clone(), &case["parameters"]);
    let mut decoder =
        AudioDecoder::<NumericBackend, _, _>::from_parts(config, quantizer, neural, &context)
            .unwrap();
    let all = codes(&case["codes"], &context);
    compare(
        &decoder.forward(&all, true, &context).unwrap(),
        &case["full"][0]["waveform"],
        &fixture,
    );
    decoder
        .neural
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    let initial = decoder.initial_state(1).unwrap();
    decoder
        .quantizer
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    assert!(decoder
        .finish(&initial, true, &context)
        .unwrap()
        .0
        .is_none());
    assert!(decoder.neural.runtime().policy().inner.events.is_empty());
    assert!(decoder.quantizer.runtime().policy().inner.events.is_empty());
    let mut state = initial.clone();
    let mut frame_offset = 0usize;
    let mut emitted = 0;
    for step in case["streaming_unflushed"].as_array().unwrap() {
        let count = usize::try_from(step["input"]["shape"][1].as_u64().unwrap()).unwrap();
        let input = all.axis_slice(1, frame_offset, frame_offset + count);
        let parent = state.clone();
        let (pcm, candidate) = decoder.step(&input, &state, true, &context).unwrap();
        compare(&pcm, &step["waveform"], &fixture);
        let (replayed, fork) = decoder.step(&input, &parent, true, &context).unwrap();
        compare(&replayed, &step["waveform"], &fixture);
        compare_audio_state(&candidate, &step["cache"], &fixture);
        compare_audio_state(&fork, &step["cache"], &fixture);
        assert_eq!(parent.frames_seen(), frame_offset as u64);
        assert_eq!(parent.samples_emitted(), emitted);
        frame_offset += count;
        emitted += step["waveform"]["shape"][2].as_u64().unwrap();
        assert_eq!(candidate.frames_seen(), frame_offset as u64);
        assert_eq!(candidate.samples_emitted(), emitted);
        state = candidate;
    }
    assert_eq!(initial.frames_seen(), 0);
    assert!(initial.spectral().history().is_none());
    decoder
        .neural
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    decoder
        .quantizer
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    let (tail, finished) = decoder.finish(&state, true, &context).unwrap();
    compare(&tail.unwrap(), &case["finish_tail"], &fixture);
    assert_eq!(
        finished.samples_emitted(),
        emitted + case["finish_tail"]["shape"][2].as_u64().unwrap()
    );
    assert_eq!(finished.frames_seen(), frame_offset as u64);
    assert!(finished.is_finished());
    assert!(finished.spectral().history().is_none());
    assert!(finished
        .neural()
        .units()
        .iter()
        .all(|unit| unit.carry().is_none()));
    assert!(decoder
        .finish(&finished, true, &context)
        .unwrap()
        .0
        .is_none());
    assert!(decoder.step(&all, &finished, true, &context).is_err());
    assert!(
        decoder.neural.runtime().policy().inner.events.is_empty(),
        "drain and finished rejection cannot acquire learned units"
    );
    assert!(decoder.quantizer.runtime().policy().inner.events.is_empty());
}

#[path = "voicechat_codec_decoder_residency/parallel.rs"]
mod parallel;
