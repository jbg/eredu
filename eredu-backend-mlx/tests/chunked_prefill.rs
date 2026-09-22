//! Small native harness for causal prefill boundaries and the ordinary shared driver.
use eredu_backend_mlx::{backend::MlxBackend, native, MlxLoadRequest, MlxTensor};
use eredu_core::{
    Completion, ControlledTextGeneration, GenerationConfigOverrides, ModelRuntime,
    PrefillChunkPolicy, TextGeneration, TextGenerationBackend, TextGenerationConfig, TokenFilter,
    TokenFilterController, TokenOutput,
};
use eredu_runtime::{NormalizedLoadRequest, WeightResidency};
use safemlx::{Array, Device, DeviceType, Stream};
use std::{convert::Infallible, num::NonZeroUsize};

fn fixture(sliding: bool) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let root = tempfile::tempdir().unwrap();
    let mut config = serde_json::json!({
        "model_type": if sliding {"mistral"} else {"llama"},
        "architectures": [if sliding {"MistralForCausalLM"} else {"LlamaForCausalLM"}],
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 64, "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001, "rope_theta": 10000.0, "tie_word_embeddings": false
    });
    if sliding {
        config["sliding_window"] = 4.into();
    }
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_owned(), vec![64, 32]),
        ("model.norm.weight".to_owned(), vec![32]),
        ("lm_head.weight".to_owned(), vec![64, 32]),
    ];
    for layer in 0..2 {
        for (name, shape) in [
            ("input_layernorm.weight", vec![32]),
            ("post_attention_layernorm.weight", vec![32]),
            ("self_attn.q_proj.weight", vec![32, 32]),
            ("self_attn.k_proj.weight", vec![8, 32]),
            ("self_attn.v_proj.weight", vec![8, 32]),
            ("self_attn.o_proj.weight", vec![32, 32]),
            ("mlp.gate_proj.weight", vec![64, 32]),
            ("mlp.up_proj.weight", vec![64, 32]),
            ("mlp.down_proj.weight", vec![32, 64]),
        ] {
            shapes.push((format!("model.layers.{layer}.{name}"), shape));
        }
    }
    let bytes: Vec<Vec<u8>> = shapes
        .iter()
        .enumerate()
        .map(|(tensor, (name, shape))| {
            (0..shape.iter().product::<usize>())
                .flat_map(|index| {
                    let value = if name.contains("norm") {
                        1.0_f32
                    } else {
                        (((index * 17 + tensor * 13) % 97) as f32 - 48.0) / 400.0
                    };
                    value.to_le_bytes()
                })
                .collect()
        })
        .collect();
    let views = shapes.iter().zip(&bytes).map(|((name, shape), bytes)| {
        (
            name.as_str(),
            TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
    root
}

fn values(tensor: &MlxTensor) -> Vec<f32> {
    tensor
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

#[derive(Default)]
struct FullLogits(Option<MlxTensor>);
impl eredu_runtime::ActivationObserver<MlxTensor, eredu_backend_mlx::backend::error::Error>
    for FullLogits
{
    fn observe(
        &mut self,
        path: &str,
        value: &MlxTensor,
    ) -> Result<(), eredu_backend_mlx::backend::error::Error> {
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            self.0 = Some(value.clone());
        }
        Ok(())
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(expected.iter().any(|value| value.abs() > 0.01));
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(error < 2e-4, "maximum absolute logit error {error}");
}

struct Unconstrained;
impl TokenFilterController for Unconstrained {
    type Error = Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Infallible> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Infallible> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Infallible> {
        Ok(false)
    }
}

fn run(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(4),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    for sliding in [false, true] {
        let root = fixture(sliding);
        for case in 0..4 {
            let request = match case {
                0 => NormalizedLoadRequest::default(),
                1 => NormalizedLoadRequest::default()
                    .with_weight_residency(WeightResidency::layerwise_host(Default::default())),
                2 => NormalizedLoadRequest::default()
                    .with_weight_residency(WeightResidency::dense_disk_stream(Default::default())),
                _ => NormalizedLoadRequest::with_quantization(
                    eredu_core::QuantizationRequest::Affine {
                        group_size: 32,
                        bits: 4,
                    },
                ),
            };
            let backend = native::backend(&stream, &weights_stream);
            let model = eredu_core::load_model(
                &backend,
                root.path(),
                MlxLoadRequest::from_normalized(request),
            )
            .unwrap();
            let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            assert!(MlxBackend::text_prefill_chunking_support(&runtime).is_ok());
            for length in [1, 3, 9] {
                let ids: Vec<_> = (0..length).map(|i| (i * 7 + 3) % 64).collect();
                runtime.reset().unwrap();
                let prompt =
                    MlxBackend::prepare_text_prompt(runtime.backend(), ids.clone()).unwrap();
                let mut observed = FullLogits::default();
                let (backend, session) = runtime.parts_mut();
                let submission = session
                    .submit_prefill_with_observer(backend, prompt, &mut observed)
                    .unwrap();
                submission.completion.wait().unwrap();
                let reference = submission
                    .output
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                assert_eq!(
                    observed.0.unwrap().as_array().shape(),
                    &[1, length as i32, 64]
                );
                drop(submission);
                let mut decode_reference = Vec::new();
                for token in [5_u32, 8, 13] {
                    let output = runtime
                        .decode(Array::from_slice(&[token], &[1, 1]))
                        .unwrap()
                        .wait()
                        .unwrap();
                    decode_reference.push(values(output.logits().unwrap()));
                }
                for chunk in [1, 2, 4, 32] {
                    runtime.reset().unwrap();
                    let mut prompt =
                        MlxBackend::prepare_text_prompt(runtime.backend(), ids.clone()).unwrap();
                    let identity = prompt.cache_identity().cloned();
                    let mut state =
                        MlxBackend::start_text_generation(runtime.backend(), config).unwrap();
                    while MlxBackend::prefill_text_prefix(
                        &mut runtime,
                        &mut prompt,
                        NonZeroUsize::new(chunk).unwrap(),
                        &mut state,
                    )
                    .unwrap()
                    {}
                    assert_eq!(prompt.cache_identity(), identity.as_ref());
                    let output = runtime.prefill(prompt).unwrap().wait().unwrap();
                    assert_eq!(output.logits().unwrap().as_array().shape(), &[1, 64]);
                    assert_close(&values(output.logits().unwrap()), &reference);
                    for (token, expected) in [5_u32, 8, 13].into_iter().zip(&decode_reference) {
                        let output = runtime
                            .decode(Array::from_slice(&[token], &[1, 1]))
                            .unwrap()
                            .wait()
                            .unwrap();
                        assert_close(&values(output.logits().unwrap()), expected);
                    }
                }
                runtime.reset().unwrap();
                let ordinary = TextGeneration::new(
                    &mut runtime,
                    ids.clone(),
                    config.with_prefill_chunk_policy(PrefillChunkPolicy::Unchunked),
                )
                .unwrap()
                .map(|token| token.unwrap().token_id().unwrap())
                .collect::<Vec<_>>();
                runtime.reset().unwrap();
                let controlled = ControlledTextGeneration::new(
                    &mut runtime,
                    ids,
                    config.with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(
                        NonZeroUsize::new(2).unwrap(),
                    )),
                    Unconstrained,
                )
                .unwrap()
                .map(|token| token.unwrap().token_id())
                .collect::<Vec<_>>();
                assert_eq!(
                    ordinary, controlled,
                    "sliding={sliding}, residency/quantization={case}, length={length}"
                );
            }
        }
    }
}

#[test]
fn cpu_chunked_prefill_matches_full_observed_logits_and_controlled_generation() {
    run(DeviceType::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn metal_chunked_prefill_matches_full_observed_logits_and_controlled_generation() {
    run(DeviceType::Gpu);
}
