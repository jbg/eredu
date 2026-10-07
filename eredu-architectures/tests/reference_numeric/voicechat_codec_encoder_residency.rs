//! Finite EAR encoder parity through independently leased learned units.
use super::voicechat_codec_quantizer_residency;
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{codes, compare, tensor};
use super::*;
use eredu_codec::{
    ear::{AudioEncoder, AudioEncoderConfig},
    ear_encoder::{EncoderConfig, EncoderUnit, NeuralEncoder},
    ear_encoder_layered::{EncoderArchitecture, LayerwiseEncoder},
    ear_spectral::SpectralConfig,
    prvq::Geometry,
    prvq_layered::QuantizerExecution,
    spectral::PeriodicHannStft,
};
use serde_json::Value;

const PREFIX: &str = "tts_model.audio_codec.encoder";
type Unit = EncoderUnit<NumericTensor>;
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
fn config(value: &Value) -> AudioEncoderConfig {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    AudioEncoderConfig {
        neural: EncoderConfig {
            latent_size: i("latent_size"),
            base_hidden_size: i("base_hidden_size"),
            channel_mult: serde_json::from_value(value["channel_mult"].clone()).unwrap(),
            rates: serde_json::from_value(value["rates"].clone()).unwrap(),
            blocks_per_stage: i("num_blocks") as usize,
            kernel_size: i("kernel_size"),
            fft_size: i("n_fft"),
            groups: i("groups"),
        },
        spectral: SpectralConfig {
            fft_size: i("n_fft"),
            hop_length: i("hop_length"),
        },
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
    }
}
fn bounded(
    config: EncoderConfig,
    parameters: &Value,
) -> LayerwiseEncoder<NumericBackend, BoundFixturePolicy> {
    LayerwiseEncoder::new(
        EncoderArchitecture::new(config, PREFIX).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: weights(parameters),
        },
    )
}
fn resident(
    config: EncoderConfig,
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseEncoder<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let count = config.unit_specs().unwrap().len();
    let architecture = EncoderArchitecture::new(config, PREFIX).unwrap();
    let parameters = weights(parameters);
    let units = (0..count)
        .map(|index| {
            let mut unit = architecture.build_unit(0, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    LayerwiseEncoder::new(architecture, NumericReplicatedPolicy::resident(units))
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

#[test]
fn voicechat_codec_encoder_numeric_stft_gather_and_forward_dft_validate_inputs() {
    let context = NumericContext::default();
    let source =
        NumericTensor::from_i32_slice(&[16_777_217, 2, 3, 4, 5, 6], &[2, 3], &context).unwrap();
    let indices = NumericTensor::from_i32_slice(&[2, 0, 2], &[3], &context).unwrap();
    let gathered = source.take_axis(&indices, -1, &context).unwrap();
    assert_eq!(
        gathered.element_type(),
        Some(eredu_nn::TensorElementType::I32)
    );
    assert_eq!(
        gathered.to_i32_vec(&context).unwrap(),
        vec![3, 16_777_217, 3, 6, 4, 6]
    );
    for invalid in [-1., 3., 0.5, f32::NAN] {
        assert!(source
            .take_axis(&NumericTensor::new(vec![1], vec![invalid]), 1, &context)
            .is_err());
    }
    let signal = NumericTensor::new(vec![4, 2], vec![1., 0., 0., 1., -1., 0., 0., -1.]);
    let (real, imaginary) = signal.rfft(4, 0, &context).unwrap();
    assert_eq!(real.shape(), &[3, 2]);
    for (actual, expected) in real.data.iter().zip([0., 0., 2., 0., 0., 0.]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in imaginary.data.iter().zip([0., 0., 0., -2., 0., 0.]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert!(signal.rfft(3, 0, &context).is_err());
    assert!(signal.rfft(4, 2, &context).is_err());
    assert!(source.rfft(3, 1, &context).is_err());
}

#[test]
fn voicechat_codec_encoder_resident_and_bounded_match_nvidia_silence_nonzero_latents_and_codes() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let context = NumericContext::default();
    let config = config(&case["geometry"]);
    let count = config.neural.unit_specs().unwrap().len();
    let mut neural =
        NeuralEncoder::<NumericBackend>::new(config.neural.clone(), PREFIX, &context).unwrap();
    replace_numeric_parameters(&mut neural, &weights(&case["parameters"]));
    let mut ordinary =
        AudioEncoder::<NumericBackend>::from_parts(config.clone(), neural, &context).unwrap();
    let mut resident = AudioEncoder::<NumericBackend, _>::from_parts(
        config.clone(),
        resident(config.neural.clone(), &case["parameters"], &context),
        &context,
    )
    .unwrap();
    let mut bounded = AudioEncoder::<NumericBackend, _>::from_parts(
        config.clone(),
        bounded(config.neural.clone(), &case["parameters"]),
        &context,
    )
    .unwrap();
    let mut quantizer = voicechat_codec_quantizer_residency::bounded(
        config.quantizer,
        weights(&case["parameters"]),
    );
    let pcm = tensor(&case["pcm"]);
    assert!(pcm
        .axis_slice(0, 0, 1)
        .data
        .iter()
        .all(|value| *value == 0.));
    assert!(pcm
        .axis_slice(0, 1, 2)
        .data
        .iter()
        .any(|value| *value != 0.));
    let stft = PeriodicHannStft::<NumericTensor>::new(
        config.spectral.fft_size,
        config.spectral.hop_length,
        &context,
    )
    .unwrap();
    let (real, imaginary) = stft.analyze(&pcm, &context).unwrap();
    compare(&real, &case["spectrum"]["real"], &fixture);
    compare(&imaginary, &case["spectrum"]["imag"], &fixture);
    for latent in [
        ordinary.latent(&pcm, &context).unwrap(),
        resident.latent(&pcm, &context).unwrap(),
        bounded.latent(&pcm, &context).unwrap(),
    ] {
        compare(&latent, &case["latent"], &fixture);
    }
    assert_eq!(
        bounded.neural.runtime().policy().inner.events,
        events(count)
    );
    bounded
        .neural
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    for encoded in [
        ordinary.encode(&pcm, &mut quantizer, &context).unwrap(),
        resident.encode(&pcm, &mut quantizer, &context).unwrap(),
        bounded.encode(&pcm, &mut quantizer, &context).unwrap(),
    ] {
        assert_eq!(encoded.data, codes(&case["codes"], &context).data);
        compare(
            &quantizer.decode(&encoded, &context).unwrap(),
            &case["decoded"],
            &fixture,
        );
    }
    assert_eq!(
        bounded.neural.runtime().policy().inner.events,
        events(count)
    );
    assert_eq!(pcm.data, tensor(&case["pcm"]).data);
}

#[test]
fn voicechat_codec_encoder_late_failure_retries_without_carry_or_quantizer_work() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let context = NumericContext::default();
    let config = config(&case["geometry"]);
    let count = config.neural.unit_specs().unwrap().len();
    let mut encoder = AudioEncoder::<NumericBackend, _>::from_parts(
        config.clone(),
        bounded(config.neural.clone(), &case["parameters"]),
        &context,
    )
    .unwrap();
    let mut quantizer = voicechat_codec_quantizer_residency::bounded(
        config.quantizer,
        weights(&case["parameters"]),
    );
    let pcm = tensor(&case["pcm"]);
    let silent = pcm.axis_slice(0, 0, 1);
    let signal = pcm.axis_slice(0, 1, 2);
    let signal_latent = encoder.latent(&signal, &context).unwrap();
    encoder
        .neural
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    encoder.neural.runtime_mut().policy_mut().inner.fail_acquire = Some(count - 1);
    let error = encoder
        .encode(&silent, &mut quantizer, &context)
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("injected dense-stream acquisition failure"));
    assert_eq!(
        encoder.neural.runtime().policy().inner.events,
        events(count)[..2 * count]
    );
    assert!(
        quantizer.runtime().policy().inner.events.is_empty(),
        "failed latent construction cannot start quantization"
    );
    encoder.neural.runtime_mut().policy_mut().inner.fail_acquire = None;
    let silent_latent = encoder.latent(&silent, &context).unwrap();
    compare(
        &NumericTensor::concatenate(&[silent_latent, signal_latent.clone()], 0, &context).unwrap(),
        &case["latent"],
        &fixture,
    );
    let encoded = encoder.encode(&silent, &mut quantizer, &context).unwrap();
    assert_eq!(
        encoded.data,
        codes(&case["codes"], &context).axis_slice(0, 0, 1).data
    );
    assert_eq!(
        encoder.latent(&signal, &context).unwrap().data,
        signal_latent.data,
        "each finite invocation begins with empty convolution carry"
    );
    encoder
        .neural
        .runtime_mut()
        .policy_mut()
        .inner
        .events
        .clear();
    quantizer.runtime_mut().policy_mut().inner.events.clear();
    let partial = signal.axis_slice(2, 0, 47);
    assert!(encoder.encode(&partial, &mut quantizer, &context).is_err());
    assert!(encoder.neural.runtime().policy().inner.events.is_empty());
    assert!(quantizer.runtime().policy().inner.events.is_empty());
    assert_eq!(pcm.data, tensor(&case["pcm"]).data);
}

#[path = "voicechat_codec_encoder_residency/parallel.rs"]
mod parallel;
