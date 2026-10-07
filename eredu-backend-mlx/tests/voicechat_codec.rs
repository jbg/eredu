use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderExecution},
    ear_decoder::{DecoderConfig, NeuralDecoder},
    ear_spectral::{EarSpectralDecoder, SpectralConfig},
    prvq::Geometry,
    spectral::PeriodicHannIstft,
};
use eredu_nn::{Index, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;
const REFERENCE: &str = eredu_evaluation::voicechat::CODEC_REFERENCE_JSON;
fn tensor(v: &Value) -> MlxTensor {
    let shape = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
}
fn codes(v: &Value, ctx: &Stream) -> MlxTensor {
    let shape = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let data = v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    MlxTensor::from_i32_slice(&data, &shape, ctx).unwrap()
}
fn geometry(v: &Value) -> DecoderConfig {
    let i = |key: &str| v[key].as_i64().unwrap() as i32;
    DecoderConfig {
        latent_size: i("latent_size"),
        base_hidden_size: i("base_hidden_size"),
        channel_mult: serde_json::from_value(v["channel_mult"].clone()).unwrap(),
        rates: serde_json::from_value(v["rates"].clone()).unwrap(),
        blocks_per_stage: i("num_blocks") as usize,
        kernel_size: i("kernel_size"),
        fft_size: i("n_fft"),
        groups: i("groups"),
    }
}
fn spectral(v: &Value) -> SpectralConfig {
    SpectralConfig {
        fft_size: v["n_fft"].as_i64().unwrap() as i32,
        hop_length: v["hop_length"].as_i64().unwrap() as i32,
    }
}
struct Bind<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(
            output.shape(),
            replacement.shape(),
            "{}",
            metadata.id.as_str()
        );
        *output = replacement;
    }
}
fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, ctx: &Stream) {
    assert_eq!(actual.shape(), tensor(expected).shape());
    let a = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let r = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (i, (v, e)) in actual
        .to_f32_vec(ctx)
        .unwrap()
        .iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        let e = e.as_f64().unwrap() as f32;
        assert!((v - e).abs() <= a + r * e.abs(), "{i}: {v} vs NVIDIA {e}");
    }
}
#[test]
fn cpu_ear_neural_spectral_streams_match_original_nvidia_cache_and_flush() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let ctx = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for name in ["compact", "released_rates"] {
        let case = &fixture["decoders"][name];
        let g = geometry(&case["geometry"]);
        let mut neural =
            NeuralDecoder::<Backend>::new(g.clone(), "tts_model.audio_codec.decoder", &ctx)
                .unwrap();
        neural.visit_parameters_mut(&mut Bind(&case["parameters"]));
        let spectral =
            EarSpectralDecoder::<Backend>::new(spectral(&case["geometry"]), &ctx).unwrap();
        assert_eq!(
            spectral.delay_samples(),
            case["startup_delay_samples"].as_i64().unwrap() as i32
        );
        let full = &case["full"][0];
        let raw = neural.forward(&tensor(&full["input"]), &ctx).unwrap();
        compare(&raw, &full["raw"], &fixture, &ctx);
        compare(
            &spectral.ordinary(&raw, true, &ctx).unwrap(),
            &full["waveform"],
            &fixture,
            &ctx,
        );
        let mut ns = neural.initial_state().unwrap();
        let mut ss = spectral.initial_state(1).unwrap();
        for step in case["streaming"].as_array().unwrap() {
            let (raw, next) = neural.step(&tensor(&step["input"]), &ns, &ctx).unwrap();
            compare(&raw, &step["raw"], &fixture, &ctx);
            ns = next;
            let flush = step["flush"].as_bool().unwrap();
            let (pcm, next) = spectral.step(&raw, &ss, flush, true, &ctx).unwrap();
            ss = next;
            compare(&pcm, &step["waveform"], &fixture, &ctx);
            if flush {
                ns = neural.finish(&ns).unwrap();
                assert!(ss.is_finished());
                assert!(ss.history().is_none());
            } else {
                for (index, carry) in ns.units().iter().filter_map(|s| s.carry()).enumerate() {
                    compare(carry, &step["cache"][index.to_string()], &fixture, &ctx);
                }
                let history =
                    &step["cache"][(g.channel_mult.len() * g.blocks_per_stage).to_string()];
                let (real, imag) = ss.history().unwrap();
                compare(real, &history["real"], &fixture, &ctx);
                compare(imag, &history["imag"], &fixture, &ctx);
            }
        }
    }
    let case = &fixture["spectral"];
    let istft = PeriodicHannIstft::<MlxTensor>::new(
        case["n_fft"].as_i64().unwrap() as i32,
        case["hop_length"].as_i64().unwrap() as i32,
        &ctx,
    )
    .unwrap();
    for result in case["cases"].as_array().unwrap() {
        let pcm = istft
            .reconstruct(
                &tensor(&case["spectrum"]["real"]),
                &tensor(&case["spectrum"]["imag"]),
                result["constrain"].as_bool().unwrap(),
                &ctx,
            )
            .unwrap();
        compare(
            &pcm.reshape(&[1, -1], &ctx).unwrap(),
            &result["waveform"],
            &fixture,
            &ctx,
        );
    }
}
#[test]
fn cpu_ear_code_to_waveform_forks_retries_invalid_inputs_and_drains_without_new_codes() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let case = &fixture["composition"];
    let ctx = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let mut model = AudioDecoder::<Backend>::new(config, "tts_model.audio_codec", &ctx).unwrap();
    model.visit_parameters_mut(&mut Bind(&case["parameters"]));
    let all = codes(&case["codes"], &ctx);
    compare(
        &model.forward(&all, true, &ctx).unwrap(),
        &case["full"][0]["waveform"],
        &fixture,
        &ctx,
    );
    let initial = model.initial_state(1).unwrap();
    assert!(model.finish(&initial, true, &ctx).unwrap().0.is_none());
    let mut state = initial.clone();
    let mut frames = 0;
    for (ordinal, step) in case["streaming_unflushed"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let count = step["input"]["shape"][1].as_i64().unwrap() as i32;
        let input = all
            .index(
                &[
                    Index::Full,
                    Index::Range(frames, frames + count),
                    Index::Full,
                ],
                &ctx,
            )
            .unwrap();
        let parent = state.clone();
        if ordinal == 1 {
            let invalid = MlxTensor::full_i32(0, &[1, 1, 4], &ctx).unwrap();
            assert!(model.step(&invalid, &state, true, &ctx).is_err());
            assert_eq!(state.frames_seen(), frames as u64);
        }
        let (pcm, next) = model.step(&input, &state, true, &ctx).unwrap();
        compare(&pcm, &step["waveform"], &fixture, &ctx);
        let (replayed, fork) = model.step(&input, &parent, true, &ctx).unwrap();
        compare(&replayed, &step["waveform"], &fixture, &ctx);
        assert_eq!(fork.samples_emitted(), next.samples_emitted());
        assert_eq!(parent.frames_seen(), frames as u64);
        state = next;
        frames += count;
        assert_eq!(state.frames_seen(), frames as u64);
        let (real, imag) = state.spectral().history().unwrap();
        let expected = &step["cache"]["4"];
        compare(real, &expected["real"], &fixture, &ctx);
        compare(imag, &expected["imag"], &fixture, &ctx);
        for (index, carry) in state
            .neural()
            .units()
            .iter()
            .filter_map(|s| s.carry())
            .enumerate()
        {
            compare(carry, &step["cache"][index.to_string()], &fixture, &ctx);
        }
    }
    assert_eq!(initial.frames_seen(), 0);
    assert!(initial.spectral().history().is_none());
    let (tail, finished) = model.finish(&state, true, &ctx).unwrap();
    compare(&tail.unwrap(), &case["finish_tail"], &fixture, &ctx);
    assert!(finished.is_finished());
    assert_eq!(finished.frames_seen(), 6);
    assert_eq!(finished.samples_emitted(), 76);
    assert!(model.finish(&finished, true, &ctx).unwrap().0.is_none());
    assert!(model.step(&all, &finished, true, &ctx).is_err());
}
