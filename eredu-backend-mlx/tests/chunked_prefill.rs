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
    write_fixture(config, shapes)
}

fn lfm2_fixture(sparse: bool, convolution_only: bool) -> tempfile::TempDir {
    let config = serde_json::json!({
        "model_type": if sparse { "lfm2_moe" } else { "lfm2" },
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 128, "norm_eps": 0.00001,
        "layer_types": ["conv", if convolution_only { "conv" } else { "full_attention" }, "conv"],
        "conv_L_cache": 4, "conv_bias": true,
        "block_auto_adjust_ff_dim": false, "tie_word_embeddings": false,
        "num_dense_layers": if sparse { 1 } else { 0 },
        "moe_intermediate_size": if sparse { 64 } else { 0 },
        "num_experts": if sparse { 2 } else { 0 },
        "num_experts_per_tok": if sparse { 2 } else { 0 },
        "norm_topk_prob": sparse, "use_expert_bias": sparse
    });
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let plan = eredu_architectures::lfm2::safetensors_plan(&args, false).unwrap();
    let shapes = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .flat_map(|group| &group.variants[0].tensors),
        )
        .map(|tensor| (tensor.key.clone(), tensor.shape.clone()))
        .collect();
    write_fixture(config, shapes)
}

fn write_fixture(
    config: serde_json::Value,
    shapes: Vec<(String, Vec<usize>)>,
) -> tempfile::TempDir {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
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

fn lfm2_gguf_fixture(sparse: bool) -> tempfile::TempDir {
    use eredu_gguf::{GgmlType, MetadataArray, MetadataValue as V, TensorInput, Writer};
    let root = lfm2_fixture(sparse, false);
    let config =
        serde_json::from_slice(&std::fs::read(root.path().join("config.json")).unwrap()).unwrap();
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let plan = eredu_architectures::lfm2::gguf_plan(&args).unwrap();
    let arch = if sparse { "lfm2moe" } else { "lfm2" };
    let mut metadata = std::collections::BTreeMap::from([
        ("general.architecture".into(), V::String(arch.into())),
        ("general.file_type".into(), V::Uint32(0)),
    ]);
    for (key, value) in [
        ("block_count", V::Uint32(3)),
        ("embedding_length", V::Uint32(32)),
        ("feed_forward_length", V::Uint32(64)),
        ("context_length", V::Uint32(128)),
        ("attention.head_count", V::Uint32(4)),
        (
            "attention.head_count_kv",
            V::Array(MetadataArray::Uint32(vec![0, 2, 0])),
        ),
        ("attention.layer_norm_rms_epsilon", V::Float32(0.00001)),
        ("shortconv.l_cache", V::Uint32(4)),
        ("rope.freq_base", V::Float32(10000.0)),
        ("vocab_size", V::Uint32(64)),
    ] {
        metadata.insert(format!("{arch}.{key}"), value);
    }
    if sparse {
        for (key, value) in [
            ("expert_feed_forward_length", 64),
            ("leading_dense_block_count", 1),
            ("expert_count", 2),
            ("expert_used_count", 2),
            ("expert_weights_norm", 1),
        ] {
            metadata.insert(format!("{arch}.{key}"), V::Uint32(value));
        }
    }
    let tensors: Vec<_> = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .flat_map(|group| &group.variants[0].tensors),
        )
        .collect();
    let dimensions: Vec<Vec<u64>> = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().rev().map(|n| *n as u64).collect())
        .collect();
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|tensor| {
            (0..tensor.shape.iter().product::<usize>())
                .flat_map(|index| {
                    let value = if tensor.key.contains("norm") {
                        1.0_f32
                    } else {
                        (((index * 17 + tensor.key.len() * 7) % 97) as f32 - 48.0) / 400.0
                    };
                    value.to_le_bytes()
                })
                .collect()
        })
        .collect();
    let inputs: Vec<_> = tensors
        .iter()
        .zip(&dimensions)
        .zip(&bytes)
        .map(|((tensor, dimensions), data)| TensorInput {
            name: &tensor.key,
            dimensions,
            ggml_type: GgmlType::F32,
            data,
        })
        .collect();
    Writer::default()
        .write(
            std::fs::File::create(root.path().join("model.gguf")).unwrap(),
            &metadata,
            &inputs,
        )
        .unwrap();
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
    assert!(actual.iter().all(|value| value.is_finite()));
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
    for (family, root) in [
        ("llama", fixture(false)),
        ("mistral", fixture(true)),
        ("lfm2", lfm2_fixture(false, false)),
        ("lfm2-moe", lfm2_fixture(true, false)),
        ("lfm2-gguf", lfm2_gguf_fixture(false)),
        ("lfm2-moe-gguf", lfm2_gguf_fixture(true)),
        ("lfm2-convolution", lfm2_fixture(false, true)),
    ] {
        let gguf = root.path().join("model.gguf");
        let path = if gguf.exists() {
            gguf.as_path()
        } else {
            root.path()
        };
        // Exercise affine loading through SafeTensors; GGUF cases retain their
        // admitted file encoding across all three residency strategies.
        let cases = if gguf.exists() { 3 } else { 4 };
        for case in 0..cases {
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
            let inspected = native::inspect_model_preparation(
                path,
                native::MlxInspectionOptions::new(MlxLoadRequest::from_normalized(request.clone())),
            )
            .unwrap();
            let cold_chunking = inspected
                .selected()
                .unwrap()
                .preparation()
                .prefill_chunking_support();
            assert_eq!(cold_chunking, Ok(()));
            let backend = native::backend(&stream, &weights_stream);
            let model =
                eredu_core::load_model(&backend, path, MlxLoadRequest::from_normalized(request))
                    .unwrap_or_else(|error| panic!("{family}, case={case}: {error}"));
            let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            assert!(MlxBackend::text_prefill_chunking_support(&runtime).is_ok());
            assert_eq!(
                cold_chunking.is_ok(),
                MlxBackend::text_prefill_chunking_support(&runtime).is_ok()
            );
            use eredu_runtime::memory_estimation::LogitsWorkspace;
            use eredu_runtime::memory_forecast::GenerationForecastBackend;
            let profile = MlxBackend::loaded_memory_profile(&runtime).unwrap();
            assert!(profile.geometry.execution_topology.is_some());
            let ordinary = MlxBackend::forecast_execution_contract(&runtime, None, false);
            assert_eq!(
                ordinary.logits,
                LogitsWorkspace::FinalPosition,
                "{family}, case={case}"
            );
            assert!(ordinary.full_pass_reason.is_none());
            let captured = MlxBackend::forecast_execution_contract(&runtime, None, true);
            assert_eq!(captured.logits, LogitsWorkspace::EveryPosition);
            assert!(captured.full_pass_reason.is_some());
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
                assert!(MlxBackend::loaded_memory_profile(&runtime)
                    .unwrap()
                    .geometry
                    .execution_topology
                    .is_none());
                let mut decode_reference = Vec::new();
                for token in [5_u32, 8, 13] {
                    let output = runtime
                        .decode(Array::from_slice(&[token], &[1, 1]))
                        .unwrap()
                        .wait()
                        .unwrap();
                    decode_reference.push(values(output.logits().unwrap()));
                }
                for chunk in [1, 2, 3, 4, 32] {
                    runtime.reset().unwrap();
                    assert!(MlxBackend::loaded_memory_profile(&runtime)
                        .unwrap()
                        .geometry
                        .execution_topology
                        .is_some());
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
                    "family={family}, residency/quantization={case}, length={length}"
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

// This exercises the GPU path on iOS devices as well as the other supported
// Apple silicon targets. CPU execution would mask an unknown relationship by
// assigning parameters to the host pool in the portable forecast driver.
#[cfg(all(
    feature = "metal",
    target_arch = "aarch64",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
#[test]
fn apple_metal_loaded_request_forecast_uses_one_physical_pool() {
    use eredu_core::{InputTokenCount, Observed, PhysicalMemorySemantics};
    use eredu_runtime::memory_estimation::{estimate_generation_memory, MemoryDomain, MemoryFit};
    use eredu_runtime::memory_forecast::{
        loaded_generation_request, GenerationForecastBackend, GenerationForecastOptions,
    };

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = fixture(false);
    let backend = native::backend(&stream, &weights_stream);
    let model = eredu_core::load_model(
        &backend,
        root.path(),
        MlxLoadRequest::from_normalized(NormalizedLoadRequest::default()),
    )
    .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let mut profile = MlxBackend::loaded_memory_profile(&runtime).unwrap();
    assert!(!profile.host_execution);
    assert_eq!(
        profile.parameters.physical_semantics,
        PhysicalMemorySemantics::Unified
    );
    assert_eq!(
        profile.available.physical_semantics,
        PhysicalMemorySemantics::Unified
    );
    // Exercise missing, exhausted and sufficient app headroom on desktop runners too.
    profile.available.physical_memory_bytes = Observed::unavailable("mobile capacity unavailable");
    let options = GenerationForecastOptions::default();
    for budget in [None, Some(0), Some(1 << 30)] {
        profile.available.available_memory_bytes = budget.map_or_else(
            || Observed::unavailable("app headroom unavailable"),
            |value| Observed::Available {
                value,
                kind: eredu_core::ObservationKind::Estimated,
                source: "app allocation headroom".into(),
            },
        );
        let (request, _) = loaded_generation_request(
            profile.clone(),
            InputTokenCount::text(9),
            4,
            PrefillChunkPolicy::Bounded(NonZeroUsize::new(2).unwrap()),
            MlxBackend::forecast_execution_contract(&runtime, None, false),
            &options,
        )
        .unwrap();
        assert_eq!(request.domains.len(), 1);
        assert_eq!(request.domains[0].domain, MemoryDomain::Unified);
        assert_eq!(request.domains[0].budget.available_bytes, budget);
        assert!(request.domains[0].resident_parameters.lower_bytes > 0);
        let estimate = estimate_generation_memory(&request).unwrap();
        assert_eq!(estimate.requested_positions, 13);
        assert!(estimate.domains[0].generation_peak.upper_bytes.is_some());
        assert_eq!(
            estimate.domains[0].generation_fit,
            match budget {
                None => MemoryFit::InsufficientInformation,
                Some(0) => MemoryFit::LikelyShortfall,
                Some(_) => MemoryFit::LikelyFit,
            }
        );
    }
}

// Run this harness serially: allocator counters are process-wide.
#[cfg(feature = "metal")]
#[test]
fn compact_convolution_history_releases_prompt_backing_storage() {
    use eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend;
    use eredu_nn::{
        CausalDepthwiseConvolution, CausalDepthwiseConvolutionSpec, ConvolutionActivation,
        ParameterSpec, Tensor,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut convolution = CausalDepthwiseConvolution::<MlxNeuralBackend>::new(
        CausalDepthwiseConvolutionSpec {
            channels: 32,
            kernel_size: 4,
            weight: ParameterSpec::trainable("conv.weight").unwrap(),
            bias: None,
            activation: ConvolutionActivation::Identity,
        },
        &stream,
    )
    .unwrap();
    convolution.weight.replace(MlxTensor::from_array(
        Array::ones::<f32>(&[32, 1, 4], &stream).unwrap(),
    ));
    convolution.weight.as_ref().as_array().evaluated().unwrap();
    for length in [1024, 16384] {
        let baseline = safemlx::memory::active_memory().unwrap();
        let history = {
            let input =
                MlxTensor::from_array(Array::ones::<f32>(&[1, length, 32], &stream).unwrap());
            let result = convolution.forward(&input, None, &stream).unwrap();
            result.output.as_array().evaluated().unwrap();
            let history = result.history.unwrap();
            history.as_array().evaluated().unwrap();
            history
        };
        assert_eq!(history.shape(), &[1, 3, 32]);
        assert!(values(&history).iter().all(|v| *v == 1.0));
        stream.synchronize().unwrap();
        let retained = safemlx::memory::active_memory()
            .unwrap()
            .saturating_sub(baseline);
        // Includes allocator alignment and MLX's bounded contiguous-view slack;
        // independent of the 128 KiB / 2 MiB prompt payloads above.
        assert!(
            retained <= 32 * 1024,
            "{length} positions retained {retained} bytes"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires EREDU_PREFILL_REFERENCE from the pinned independent reference exporter"]
fn released_checkpoint_chunked_prefill_and_cached_decode_match_reference() {
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("EREDU_PREFILL_REFERENCE").expect("reference JSON path"))
            .unwrap(),
    )
    .unwrap();
    let tokens = |field: &str| {
        fixture[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let ids = tokens("prompt");
    let decode = tokens("decode_tokens");
    let expected: Vec<Vec<f32>> = fixture["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    assert_eq!(expected.len(), decode.len() + 1);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = native::backend(&stream, &weights);
    let model = eredu_core::load_model(
        &backend,
        std::path::Path::new(fixture["model_path"].as_str().unwrap()),
        MlxLoadRequest::from_normalized(NormalizedLoadRequest::default()),
    )
    .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
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
    for chunk in [1, 2, 4, 128] {
        runtime.reset().unwrap();
        let mut prompt = MlxBackend::prepare_text_prompt(runtime.backend(), ids.clone()).unwrap();
        let mut state = MlxBackend::start_text_generation(runtime.backend(), config).unwrap();
        while MlxBackend::prefill_text_prefix(
            &mut runtime,
            &mut prompt,
            NonZeroUsize::new(chunk).unwrap(),
            &mut state,
        )
        .unwrap()
        {}
        let output = runtime.prefill(prompt).unwrap().wait().unwrap();
        let mut actual = vec![values(output.logits().unwrap())];
        drop(output);
        for token in &decode {
            let output = runtime
                .decode(Array::from_slice(&[*token], &[1, 1]))
                .unwrap()
                .wait()
                .unwrap();
            actual.push(values(output.logits().unwrap()));
        }
        let mut maximum = 0.0_f32;
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                maximum = maximum.max((a - b).abs());
            }
        }
        assert!(
            maximum < 2e-3,
            "chunk={chunk}, independent logit error={maximum}"
        );
        eprintln!("released prefill: chunk={chunk}, prompt={}, cached_decodes={}, max_abs_error={maximum}", ids.len(), decode.len());
    }
    // Measure actual peak growth for caller-selected small and full passes.
    // Allocator policy remains unchanged; active peaks exclude cached free blocks.
    for length in [128, 1024] {
        let mut peaks = Vec::new();
        for policy in [
            PrefillChunkPolicy::Unchunked,
            PrefillChunkPolicy::Bounded(NonZeroUsize::new(32).unwrap()),
        ] {
            runtime.reset().unwrap();
            let baseline = safemlx::memory::active_memory().unwrap();
            safemlx::memory::reset_peak_memory().unwrap();
            let tokens = (0..length).map(|i| ids[i % ids.len()]).collect();
            let generated = TextGeneration::new(
                &mut runtime,
                tokens,
                config.with_prefill_chunk_policy(policy),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
            let peak = safemlx::memory::peak_memory()
                .unwrap()
                .saturating_sub(baseline);
            eprintln!("released memory: positions={length}, policy={policy:?}, additional_peak={peak}, tokens={generated:?}");
            peaks.push((peak, generated));
        }
        assert_eq!(peaks[0].1, peaks[1].1);
        assert!(
            peaks[1].0 < peaks[0].0,
            "chunked peak must improve for the released long-prompt fixture: {peaks:?}"
        );
    }
}
