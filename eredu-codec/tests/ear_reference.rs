mod support {
    #[allow(dead_code)]
    pub mod numeric_backend;
}
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderExecution},
    ear_decoder::{DecoderConfig, DecoderUnit, NeuralDecoder},
    ear_spectral::{EarSpectralDecoder, SpectralConfig},
    prvq::{Codebook, Geometry, ResidualQuantizer},
    spectral::PeriodicHannIstft,
};
use eredu_nn::{
    Index, ParameterMetadata, ParameterVisitorMut, Parameterized, RetainedTensorRoots, Tensor,
};
use serde_json::Value;
use support::numeric_backend::{activate, Context, FailAt, NumericTensor, ReferenceBackend};
// Evaluation owns this artifact; importing its crate here would create a dev cycle.
const REFERENCE: &str = eredu_evaluation::voicechat::CODEC_REFERENCE_JSON;
fn tensor(v: &Value) -> NumericTensor {
    let shape = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    NumericTensor::f32(
        v["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_f64().unwrap() as f32)
            .collect(),
        &shape,
    )
}
fn codes(v: &Value, ctx: &Context) -> NumericTensor {
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
    NumericTensor::from_i32_slice(&data, &shape, ctx).unwrap()
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
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
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
fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value, ctx: &Context) {
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
fn ear_neural_spectral_streams_match_original_nvidia_cache_and_flush() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let ctx = Context::new(FailAt::None);
    activate(&ctx);
    for name in ["compact", "released_rates"] {
        let case = &fixture["decoders"][name];
        let g = geometry(&case["geometry"]);
        let mut neural = NeuralDecoder::<ReferenceBackend>::new(
            g.clone(),
            "tts_model.audio_codec.decoder",
            &ctx,
        )
        .unwrap();
        neural.visit_parameters_mut(&mut Bind(&case["parameters"]));
        assert_eq!(
            spectral(&case["geometry"]).delay_samples().unwrap(),
            case["startup_delay_samples"].as_i64().unwrap() as i32
        );
        let spectral =
            EarSpectralDecoder::<ReferenceBackend>::new(spectral(&case["geometry"]), &ctx).unwrap();
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
                // Completion must visit all independently retained causal and
                // complex carry, including values not returned as waveform.
                let mut roots = Vec::new();
                ns.visit_retained_tensors(&mut |tensor| roots.push(tensor));
                ss.visit_retained_tensors(&mut |tensor| roots.push(tensor));
                let count = step["cache"].as_object().unwrap().len() - 1;
                assert_eq!(roots.len(), count + 2);
                for (index, carry) in roots[..count].iter().enumerate() {
                    compare(carry, &step["cache"][index.to_string()], &fixture, &ctx);
                }
                let history = &step["cache"][count.to_string()];
                compare(roots[count], &history["real"], &fixture, &ctx);
                compare(roots[count + 1], &history["imag"], &fixture, &ctx);
            }
        }
    }
    let case = &fixture["spectral"];
    let istft = PeriodicHannIstft::<NumericTensor>::new(
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
fn ear_code_to_waveform_forks_retries_failures_and_drains_without_new_codes() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let case = &fixture["composition"];
    let ctx = Context::new(FailAt::None);
    activate(&ctx);
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let units = config
        .neural
        .unit_specs()
        .unwrap()
        .into_iter()
        .map(|spec| {
            let mut unit = DecoderUnit::new(spec, "tts_model.audio_codec.decoder", &ctx).unwrap();
            unit.visit_parameters_mut(&mut Bind(&case["parameters"]));
            unit
        })
        .collect();
    let codebooks = (0..config.quantizer.depth)
        .map(|depth| {
            let mut book =
                Codebook::unloaded(config.quantizer, "tts_model.audio_codec.prvq", depth, &ctx)
                    .unwrap();
            book.visit_parameters_mut(&mut Bind(&case["parameters"]));
            book
        })
        .collect();
    let declared = ctx.counters().unloaded_allocations;
    let mut model = AudioDecoder::<ReferenceBackend>::from_parts(
        config.clone(),
        ResidualQuantizer::<ReferenceBackend>::from_codebooks(config.quantizer, codebooks).unwrap(),
        NeuralDecoder::<ReferenceBackend>::from_units(config.neural.clone(), units).unwrap(),
        &ctx,
    )
    .unwrap();
    assert_eq!(ctx.counters().unloaded_allocations, declared);
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
            ctx.set_failure(FailAt::Execution);
            assert!(model.step(&input, &state, true, &ctx).is_err());
            ctx.set_failure(FailAt::None);
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
    assert_eq!(ctx.counters().execution_failures, 1);
}

#[test]
fn ear_assembly_rejects_incomplete_reordered_and_incompatible_units() {
    use eredu_codec::{
        ear::{AudioEncoder, AudioEncoderConfig},
        ear_encoder::{EncoderConfig, EncoderUnit, NeuralEncoder},
        Error,
    };
    let ctx = Context::new(FailAt::None);
    activate(&ctx);
    let decoder = DecoderConfig {
        latent_size: 2,
        base_hidden_size: 2,
        channel_mult: vec![1],
        rates: vec![2],
        blocks_per_stage: 1,
        kernel_size: 3,
        fft_size: 4,
        groups: 1,
    };
    let encoder = EncoderConfig {
        latent_size: 2,
        base_hidden_size: 2,
        channel_mult: vec![1],
        rates: vec![2],
        blocks_per_stage: 1,
        kernel_size: 3,
        fft_size: 4,
        groups: 1,
    };
    let dec_units = || {
        decoder
            .unit_specs()
            .unwrap()
            .into_iter()
            .map(|spec| DecoderUnit::<NumericTensor>::new(spec, "codec.decoder", &ctx).unwrap())
            .collect::<Vec<_>>()
    };
    let enc_units = || {
        encoder
            .unit_specs()
            .unwrap()
            .into_iter()
            .map(|spec| EncoderUnit::<NumericTensor>::new(spec, "codec.encoder", &ctx).unwrap())
            .collect::<Vec<_>>()
    };
    let mut reordered = dec_units();
    reordered.swap(0, 1);
    assert!(NeuralDecoder::<ReferenceBackend>::from_units(decoder.clone(), reordered).is_err());
    let mut incomplete = dec_units();
    incomplete.pop();
    assert!(NeuralDecoder::<ReferenceBackend>::from_units(decoder.clone(), incomplete).is_err());
    let mut reordered = enc_units();
    reordered.swap(0, 1);
    assert!(NeuralEncoder::<ReferenceBackend>::from_units(encoder.clone(), reordered).is_err());
    let mut incompatible = encoder.clone();
    incompatible.latent_size = 3;
    assert!(NeuralEncoder::<ReferenceBackend>::from_units(incompatible, enc_units()).is_err());

    let geometry = Geometry {
        depth: 2,
        cardinality: 3,
        channels: 2,
    };
    let books = || {
        (0..geometry.depth)
            .map(|depth| {
                Codebook::<NumericTensor>::unloaded(geometry, "codec.prvq", depth, &ctx).unwrap()
            })
            .collect::<Vec<_>>()
    };
    let mut reordered = books();
    reordered.swap(0, 1);
    assert!(ResidualQuantizer::<ReferenceBackend>::from_codebooks(geometry, reordered).is_err());
    let mut incomplete = books();
    incomplete.pop();
    assert!(ResidualQuantizer::<ReferenceBackend>::from_codebooks(geometry, incomplete).is_err());
    let mut malformed = books();
    malformed[0]
        .means
        .replace(NumericTensor::f32(vec![0.; 8], &[4, 2]));
    assert!(ResidualQuantizer::<ReferenceBackend>::from_codebooks(geometry, malformed).is_err());

    let neural_decoder =
        NeuralDecoder::<ReferenceBackend>::from_units(decoder.clone(), dec_units()).unwrap();
    let neural_encoder =
        NeuralEncoder::<ReferenceBackend>::from_units(encoder.clone(), enc_units()).unwrap();
    let quantizer =
        ResidualQuantizer::<ReferenceBackend>::from_codebooks(geometry, books()).unwrap();
    let mut decoder_config = AudioDecoderConfig {
        quantizer: geometry,
        neural: decoder,
        spectral: SpectralConfig {
            fft_size: 4,
            hop_length: 1,
        },
    };
    decoder_config.neural.kernel_size = 5;
    let mut encoder_config = AudioEncoderConfig {
        quantizer: geometry,
        neural: encoder,
        spectral: SpectralConfig {
            fft_size: 4,
            hop_length: 1,
        },
    };
    encoder_config.neural.kernel_size = 5;
    // Geometry failures precede backend capability probing and spectral allocation.
    ctx.set_failure(FailAt::Capability);
    assert!(matches!(
        AudioDecoder::<ReferenceBackend>::from_parts(
            decoder_config,
            quantizer,
            neural_decoder,
            &ctx,
        ),
        Err(Error::InvalidShape(_))
    ));
    assert!(matches!(
        AudioEncoder::<ReferenceBackend>::from_parts(encoder_config, neural_encoder, &ctx,),
        Err(Error::InvalidShape(_))
    ));
}

#[test]
fn ear_decoder_preparation_rejects_incomplete_or_mismatched_recipes_before_payload_reads() {
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{
            CheckpointLease, CheckpointSource, SharedCheckpointSource, StoreError, TensorMetadata,
            TensorReadRequest, TensorSelection, WeightStoreDiagnostics,
        },
        StoredDtype,
    };
    use eredu_codec::ear_decoder_preparation::{DecoderPreparationError, PreparedDecoderBindings};
    use std::{collections::BTreeMap, sync::Arc};

    // Independent fixture headers suffice; cold admission cannot acquire payloads.
    struct Headers(BTreeMap<String, TensorMetadata>);
    impl CheckpointSource for Headers {
        fn source_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("cold decoder preparation must not acquire tensor payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            eredu_checkpoint::store::MemoryWeightStore::from_safetensors([])?.source_diagnostics()
        }
    }
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let case = &fixture["composition"];
    let config = geometry(&case["geometry"]);
    let headers = case["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            (
                name.clone(),
                TensorMetadata {
                    name: name.clone(),
                    logical_shape: shape.clone(),
                    physical_shape: shape,
                    stored_dtype: StoredDtype::F32,
                    encoded_byte_len: value["values"].as_array().unwrap().len() as u64 * 4,
                    backing_shard: None,
                },
            )
        })
        .collect();
    let source: SharedCheckpointSource = Arc::new(Headers(headers));
    let direct = |name: &str| DerivedWeightRecipe::Source {
        key: name.into(),
        selection: TensorSelection::Full,
    };
    let mut recipes: BTreeMap<_, _> = source
        .source_keys()
        .into_iter()
        .map(|name| (name.clone(), direct(&name)))
        .collect();
    let required = "tts_model.audio_codec.decoder.layers.6.weight";
    let neighbor = "tts_model.audio_codec.prvq.mus_list.0";
    let prefix = "tts_model.audio_codec.decoder";
    recipes.remove(required);
    assert!(matches!(
        PreparedDecoderBindings::from_recipes(config.clone(), prefix, source.clone(), &recipes),
        Err(DecoderPreparationError::MissingParameter(name)) if name == required
    ));
    recipes.insert(required.into(), direct(neighbor));
    assert!(matches!(
        PreparedDecoderBindings::from_recipes(config.clone(), prefix, source.clone(), &recipes),
        Err(DecoderPreparationError::Geometry(name)) if name == required
    ));
    recipes.insert(required.into(), direct(required));
    let prepared =
        PreparedDecoderBindings::from_recipes(config, prefix, source.clone(), &recipes).unwrap();
    let (_, _, _, _, restricted, _) = prepared.into_parts();
    assert_eq!(
        restricted.source_metadata(required).unwrap(),
        source.source_metadata(required).unwrap()
    );
    assert!(matches!(restricted.source_metadata(neighbor),
        Err(StoreError::UnauthorizedTensor { key, .. }) if key == neighbor));
}
