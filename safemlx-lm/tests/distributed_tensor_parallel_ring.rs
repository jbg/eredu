#![cfg(unix)]

use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    module::{Module, ModuleParameters},
    ops::GgufMetadataValue,
    random::{self, RandomState},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_gguf::{GgmlType, TensorInput, Writer};
use safemlx_lm::{
    architectures::{
        deepseek_v3::{
            layerwise::{load_deepseek_v3_tensor_parallel_model, DeepSeekV3LayerwiseModel},
            model::{self as deepseek_v3, Cache as DeepSeekCache},
        },
        llama::layerwise::{load_llama_tensor_parallel_model, LlamaCache, LlamaModel},
        qwen::dense::{
            self as dense_qwen,
            layerwise::{
                load_tensor_parallel_model as load_qwen_tensor_parallel_model,
                DenseQwenLayerwiseCache, LayerwiseDecoder as DenseQwenLayerwiseDecoder,
            },
        },
    },
    runtime::cache::KeyValueCache,
    runtime::checkpoint::binding::canonical_checkpoint_name,
    runtime::generation::sampler::DefaultSampler,
    sample_and_synchronize, CacheResidencyPolicy, DeviceAssignment, LayerCachePolicy,
    LayerExecutionLoadOptions, LayerwiseLoadOptions, PagedCacheOptions, ParallelBuildContext,
    ParallelModelInfo, ParallelTopology, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheOptions, PromptCacheTopology, ShardingPolicy,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_TENSOR_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_TENSOR_CHECKPOINT";
const PROMPT_CACHE_ROOT: &str = "SAFEMLX_LM_TENSOR_PROMPT_CACHE";
const FIXTURE_FAMILY: &str = "SAFEMLX_LM_TENSOR_FIXTURE_FAMILY";
const ZERO_BIAS_CHECKPOINT: &str = "SAFEMLX_LM_TENSOR_ZERO_BIAS_CHECKPOINT";
const PARAMETER_RESIDENCY: &str = "SAFEMLX_LM_TENSOR_PARAMETER_RESIDENCY";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorParameterResidency {
    LayerwiseHost,
    FullyResident,
}

impl TensorParameterResidency {
    const fn name(self) -> &'static str {
        match self {
            Self::LayerwiseHost => "layerwise-host",
            Self::FullyResident => "fully-resident",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "layerwise-host" => Self::LayerwiseHost,
            "fully-resident" => Self::FullyResident,
            _ => panic!("unknown tensor-parallel parameter residency {value:?}"),
        }
    }

    fn load_options(self) -> LayerExecutionLoadOptions {
        match self {
            Self::LayerwiseHost => {
                LayerExecutionLoadOptions::LayerwiseHost(LayerwiseLoadOptions::default())
            }
            Self::FullyResident => LayerExecutionLoadOptions::FullyResident,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixtureFamily {
    LlamaSafetensors,
    LlamaGguf,
    DeepSeekSafetensors,
    Qwen2Gguf,
    Qwen2Q8Gguf,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::LlamaSafetensors => "llama-safetensors",
            Self::LlamaGguf => "llama-gguf",
            Self::DeepSeekSafetensors => "deepseek-safetensors",
            Self::Qwen2Gguf => "qwen2-gguf",
            Self::Qwen2Q8Gguf => "qwen2-q8_0-gguf",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "llama-safetensors" => Self::LlamaSafetensors,
            "llama-gguf" => Self::LlamaGguf,
            "deepseek-safetensors" => Self::DeepSeekSafetensors,
            "qwen2-gguf" => Self::Qwen2Gguf,
            "qwen2-q8_0-gguf" => Self::Qwen2Q8Gguf,
            _ => panic!("unknown tensor-parallel fixture family {value:?}"),
        }
    }

    const fn layer_count(self) -> usize {
        match self {
            Self::Qwen2Gguf | Self::Qwen2Q8Gguf => 2,
            _ => 1,
        }
    }

    const fn vocab_size(self) -> usize {
        match self {
            Self::DeepSeekSafetensors => 8,
            Self::Qwen2Gguf => 8,
            Self::Qwen2Q8Gguf => 64,
            _ => 5,
        }
    }

    const fn model_type(self) -> &'static str {
        match self {
            Self::LlamaSafetensors | Self::LlamaGguf => "llama",
            Self::DeepSeekSafetensors => "deepseek_v3",
            Self::Qwen2Gguf | Self::Qwen2Q8Gguf => "qwen2",
        }
    }

    const fn persists_paged_prefix(self) -> bool {
        !self.is_qwen2()
    }

    const fn is_qwen2(self) -> bool {
        matches!(self, Self::Qwen2Gguf | Self::Qwen2Q8Gguf)
    }

    const fn qwen2_head_dim(self) -> i32 {
        match self {
            Self::Qwen2Gguf => 2,
            Self::Qwen2Q8Gguf => 16,
            _ => panic!("non-Qwen fixture has no Qwen head dimension"),
        }
    }

    fn qwen2_payload_bytes(self) -> u64 {
        match self {
            Self::Qwen2Gguf => qwen2_gguf_payload_bytes(),
            Self::Qwen2Q8Gguf => qwen2_q8_0_gguf_payload_bytes(),
            _ => panic!("non-Qwen fixture has no Qwen GGUF payload"),
        }
    }
}

enum GeneralizedTensorModel {
    Llama(LlamaModel),
    DeepSeek(DeepSeekV3LayerwiseModel),
    Qwen(DenseQwenLayerwiseDecoder),
}

enum GeneralizedTensorCache {
    Llama(LlamaCache),
    DeepSeek(DeepSeekCache),
    Qwen(DenseQwenLayerwiseCache),
}

impl GeneralizedTensorModel {
    fn load(
        checkpoint: &Path,
        family: FixtureFamily,
        residency: TensorParameterResidency,
        topology: ParallelTopology,
        stream: &Stream,
    ) -> Self {
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let options = residency.load_options();
        match family {
            FixtureFamily::DeepSeekSafetensors => Self::DeepSeek(
                load_deepseek_v3_tensor_parallel_model(checkpoint, options, build, stream, stream)
                    .unwrap(),
            ),
            FixtureFamily::LlamaSafetensors | FixtureFamily::LlamaGguf => Self::Llama(
                load_llama_tensor_parallel_model(checkpoint, options, build, stream, stream)
                    .unwrap(),
            ),
            FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen2Q8Gguf => Self::Qwen(
                load_qwen_tensor_parallel_model(checkpoint, options, build, stream, stream)
                    .unwrap(),
            ),
        }
    }

    fn parallel_info(&self) -> &ParallelModelInfo {
        match self {
            Self::Llama(model) => model.parallel_info().unwrap(),
            Self::DeepSeek(model) => model.parallel_info().unwrap(),
            Self::Qwen(model) => model.parallel_info().unwrap(),
        }
    }

    fn prompt_cache_architecture_fingerprint(&self) -> String {
        match self {
            Self::Llama(model) => model.prompt_cache_architecture_fingerprint(),
            Self::DeepSeek(model) => model.prompt_cache_architecture_fingerprint(),
            Self::Qwen(model) => model.prompt_cache_architecture_fingerprint(),
        }
    }

    fn prompt_cache_layer_layout(&self) -> safemlx_lm::LayerSchedule<LayerCachePolicy> {
        match self {
            Self::Llama(model) => model.prompt_cache_layer_layout().unwrap(),
            Self::DeepSeek(model) => model.prompt_cache_layer_layout().unwrap(),
            Self::Qwen(model) => model.prompt_cache_layer_layout().unwrap(),
        }
    }

    fn new_paged_cache(&self, options: PagedCacheOptions) -> GeneralizedTensorCache {
        match self {
            Self::Llama(model) => GeneralizedTensorCache::Llama(
                model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .unwrap(),
            ),
            Self::DeepSeek(model) => GeneralizedTensorCache::DeepSeek(
                model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .unwrap(),
            ),
            Self::Qwen(model) => {
                GeneralizedTensorCache::Qwen(DenseQwenLayerwiseCache::Concat(model.new_cache()))
            }
        }
    }

    fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut GeneralizedTensorCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Array {
        match (self, cache) {
            (Self::Llama(model), GeneralizedTensorCache::Llama(cache)) => model
                .forward_tensor_parallel(inputs, cache, group, stream)
                .unwrap(),
            (Self::DeepSeek(model), GeneralizedTensorCache::DeepSeek(cache)) => model
                .forward_tensor_parallel(inputs, cache, group, stream)
                .unwrap(),
            (Self::Qwen(model), GeneralizedTensorCache::Qwen(cache)) => model
                .forward_tensor_parallel(inputs, None, cache, group, stream)
                .unwrap(),
            _ => panic!("generalized tensor model/cache architecture mismatch"),
        }
    }

    fn checkpoint_diagnostics(
        &self,
    ) -> safemlx_lm::runtime::checkpoint::store::WeightStoreDiagnostics {
        match self {
            Self::Llama(model) => model.checkpoint_store().unwrap().diagnostics().unwrap(),
            Self::DeepSeek(model) => model.checkpoint_store().diagnostics().unwrap(),
            Self::Qwen(model) => model.checkpoint_store().diagnostics().unwrap(),
        }
    }

    fn save_prompt_cache(
        &self,
        cache: &mut GeneralizedTensorCache,
        root: &Path,
        descriptor: PromptCacheDescriptor,
        stream: &Stream,
    ) -> PromptCacheManifest {
        match (self, cache) {
            (Self::Llama(model), GeneralizedTensorCache::Llama(cache)) => model
                .save_prompt_cache(
                    cache,
                    root,
                    descriptor,
                    &[1, 2],
                    &PromptCacheOptions::default(),
                    stream,
                )
                .unwrap(),
            (Self::DeepSeek(model), GeneralizedTensorCache::DeepSeek(cache)) => model
                .save_prompt_cache(
                    cache,
                    root,
                    descriptor,
                    &[1, 2],
                    &PromptCacheOptions::default(),
                    stream,
                )
                .unwrap(),
            (Self::Qwen(_), GeneralizedTensorCache::Qwen(_)) => {
                panic!("Qwen2 GGUF fixture does not exercise prompt-cache persistence")
            }
            _ => panic!("generalized tensor model/cache architecture mismatch"),
        }
    }

    fn load_prompt_cache(
        &self,
        root: &Path,
        descriptor: &PromptCacheDescriptor,
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> (GeneralizedTensorCache, PromptCacheManifest) {
        match self {
            Self::Llama(model) => {
                let (cache, manifest) = model
                    .load_prompt_cache(root, descriptor, &[1, 2], options, stream)
                    .unwrap();
                (GeneralizedTensorCache::Llama(cache), manifest)
            }
            Self::DeepSeek(model) => {
                let (cache, manifest) = model
                    .load_prompt_cache(root, descriptor, &[1, 2], options, stream)
                    .unwrap();
                (GeneralizedTensorCache::DeepSeek(cache), manifest)
            }
            Self::Qwen(_) => {
                panic!("Qwen2 GGUF fixture does not exercise prompt-cache persistence")
            }
        }
    }
}

impl GeneralizedTensorCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Llama(cache) => cache.offset(),
            Self::DeepSeek(cache) => cache.offset(),
            Self::Qwen(DenseQwenLayerwiseCache::Concat(cache)) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Qwen(DenseQwenLayerwiseCache::Sliding(cache)) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Qwen(DenseQwenLayerwiseCache::Paged(cache)) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
        }
    }

    fn assert_qwen2_local_cache_geometry(&self, full_sequence: i32, head_dim: i32) {
        let Self::Qwen(DenseQwenLayerwiseCache::Concat(layers)) = self else {
            panic!("Qwen2 local cache assertion requires a concat dense-Qwen cache")
        };
        assert_eq!(layers.len(), 2);
        for (index, cache) in layers.iter().enumerate() {
            let cache = cache.as_ref().unwrap();
            let arrays = cache.retained_arrays();
            assert_eq!(arrays.len(), 2);
            let retained_sequence = if index == 1 {
                full_sequence.min(1)
            } else {
                full_sequence
            };
            assert_eq!(arrays[0].shape(), &[1, 1, retained_sequence, head_dim]);
            assert_eq!(arrays[1].shape(), &[1, 1, retained_sequence, head_dim]);
            assert_eq!(cache.max_size(), (index == 1).then_some(2));
        }
    }
}

fn evaluated_values(array: &Array) -> Vec<f32> {
    array.evaluated().unwrap().as_slice::<f32>().to_vec()
}

fn assert_arrays_close(actual: &Array, expected: &Array, tolerance: f32) {
    let actual_values = evaluated_values(actual);
    let expected_values = evaluated_values(expected);
    assert_eq!(actual_values.len(), expected_values.len());
    for (index, (actual, expected)) in actual_values.iter().zip(&expected_values).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "tensor element {index} was {actual}, expected {expected} within {tolerance}; actual={actual_values:?}, expected={expected_values:?}",
        );
    }
}

fn assert_arrays_materially_different(left: &Array, right: &Array, threshold: f32) {
    let left = evaluated_values(left);
    let right = evaluated_values(right);
    assert_eq!(left.len(), right.len());
    let maximum_difference = left
        .iter()
        .zip(&right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maximum_difference > threshold,
        "Q/K/V biases changed logits by only {maximum_difference}; expected more than {threshold}"
    );
}

#[test]
fn tensor_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let family = FixtureFamily::parse(&std::env::var(FIXTURE_FAMILY).unwrap());
    let parameter_residency =
        TensorParameterResidency::parse(&std::env::var(PARAMETER_RESIDENCY).unwrap());
    let layer_count = family.layer_count();
    let vocab_size = family.vocab_size();
    let prompt_cache_root = PathBuf::from(std::env::var_os(PROMPT_CACHE_ROOT).unwrap());
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let mut model =
        GeneralizedTensorModel::load(&checkpoint, family, parameter_residency, topology, &stream);
    let info = model.parallel_info();
    assert_eq!(info.topology(), topology);
    assert_eq!(info.topology().tensor_parallel_rank, expected_rank);
    assert_eq!(info.topology().tensor_parallel_size, 2);
    match parameter_residency {
        TensorParameterResidency::FullyResident => {
            assert_eq!(
                info.pinned_device_parameter_bytes(),
                info.local_parameter_bytes()
            );
            assert_eq!(
                info.maximum_device_parameter_bytes(),
                info.local_parameter_bytes()
            );
        }
        TensorParameterResidency::LayerwiseHost => {
            assert!(info.pinned_device_parameter_bytes() < info.local_parameter_bytes());
            assert!(info.maximum_device_parameter_bytes() <= info.local_parameter_bytes());
        }
    }
    assert!(info.global_parameter_bytes() >= info.local_parameter_bytes());
    if matches!(
        family,
        FixtureFamily::LlamaSafetensors | FixtureFamily::LlamaGguf
    ) {
        assert!(info.local_parameter_bytes() < 656);
    }
    assert!(info
        .owned_tensors()
        .iter()
        .any(|name| canonical_checkpoint_name(name) == "model.embed_tokens.weight"));

    if family.is_qwen2() {
        assert_eq!(info.model_type(), "qwen2");
        assert!(info.local_parameter_bytes() < family.qwen2_payload_bytes());
        for layer in 0..2 {
            for projection in ["q", "k", "v"] {
                assert!(info.owned_tensors().iter().any(|name| {
                    canonical_checkpoint_name(name)
                        == format!("model.layers.{layer}.self_attn.{projection}_proj.bias")
                }));
            }
        }
    }

    let paged = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model.new_paged_cache(paged.clone());
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let logits = model.forward_tensor_parallel(&prompt, &mut cache, &group, &stream);
    assert_eq!(logits.shape(), &[1, 2, vocab_size as i32]);

    if family.is_qwen2() {
        let mut reference = dense_qwen::load_gguf(&checkpoint, &stream, &stream).unwrap();
        let mut reference_cache = reference.new_cache();
        let reference_logits = reference
            .forward(
                dense_qwen::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: &mut reference_cache,
                },
                &stream,
            )
            .unwrap();
        let tolerance = if family == FixtureFamily::Qwen2Q8Gguf {
            2e-4
        } else {
            5e-5
        };
        assert_arrays_close(&logits, &reference_logits, tolerance);
        cache.assert_qwen2_local_cache_geometry(2, family.qwen2_head_dim());

        let diagnostics = model.checkpoint_diagnostics();
        assert!(diagnostics.physical_reads > 0);
        assert!(
            diagnostics.physical_read_bytes < family.qwen2_payload_bytes(),
            "rank {expected_rank} read {} GGUF bytes for a {}-byte global tensor payload",
            diagnostics.physical_read_bytes,
            family.qwen2_payload_bytes()
        );

        let zero_bias_checkpoint = PathBuf::from(std::env::var_os(ZERO_BIAS_CHECKPOINT).unwrap());
        let mut zero_bias = dense_qwen::load_gguf(zero_bias_checkpoint, &stream, &stream).unwrap();
        let mut zero_bias_cache = zero_bias.new_cache();
        let zero_bias_logits = zero_bias
            .forward(
                dense_qwen::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: &mut zero_bias_cache,
                },
                &stream,
            )
            .unwrap();
        assert_arrays_materially_different(&reference_logits, &zero_bias_logits, 1e-5);

        let token = Array::from_slice(&[3u32], &[1, 1]);
        let distributed_decode = model.forward_tensor_parallel(&token, &mut cache, &group, &stream);
        let reference_decode = reference
            .forward(
                dense_qwen::ModelInput {
                    inputs: &token,
                    mask: None,
                    cache: &mut reference_cache,
                },
                &stream,
            )
            .unwrap();
        assert_arrays_close(&distributed_decode, &reference_decode, tolerance);
        cache.assert_qwen2_local_cache_geometry(3, family.qwen2_head_dim());
        assert_eq!(cache.offset(), 3);
        return;
    }

    assert!(family.persists_paged_prefix());
    let descriptor = PromptCacheDescriptor {
        model_family: family.model_type().into(),
        effective_model_type: family.model_type().into(),
        checkpoint_fingerprint: "tensor-ring-fixture".into(),
        prefix_content_fingerprint: "tokens:1,2".into(),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint(),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        batch_size: 1,
        layer_layout: model.prompt_cache_layer_layout(),
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: None,
            tensor_parallel: Some((2, expected_rank)),
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        },
    };
    let saved =
        model.save_prompt_cache(&mut cache, &prompt_cache_root, descriptor.clone(), &stream);
    assert_eq!(saved.topology, descriptor.topology);
    let token = safemlx::Array::from_slice(&[0u32], &[1, 1]);
    let uninterrupted = model.forward_tensor_parallel(&token, &mut cache, &group, &stream);
    let uninterrupted = uninterrupted.evaluated().unwrap();
    let uninterrupted_values = uninterrupted.as_slice::<f32>().to_vec();
    drop(uninterrupted);
    let (mut cache, manifest) =
        model.load_prompt_cache(&prompt_cache_root, &descriptor, paged, &stream);
    assert_eq!(manifest.topology, descriptor.topology);
    let restored = model.forward_tensor_parallel(&token, &mut cache, &group, &stream);
    let restored = restored.evaluated().unwrap();
    assert_eq!(uninterrupted_values, restored.as_slice::<f32>());
    let logits = restored.as_array().clone();
    drop(restored);
    let mut sampler = DefaultSampler;
    let mut prng = (expected_rank == 0).then(|| RandomState::from_key(random::key(7).unwrap()));
    let synchronized = sample_and_synchronize(
        Some(&logits),
        1,
        &mut sampler,
        1.0,
        prng.as_mut(),
        false,
        0,
        &group,
        &stream,
    )
    .unwrap();
    let sampled = synchronized.token.evaluated().unwrap();
    assert!(sampled.as_slice::<u32>()[0] < vocab_size as u32);
    drop(sampled);
    let logits = model.forward_tensor_parallel(&synchronized.token, &mut cache, &group, &stream);
    assert_eq!(logits.shape(), &[1, 1, vocab_size as i32]);
    assert_eq!(cache.offset(), 4);
}

fn write_f32_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let buffers = tensors
        .iter()
        .map(|(_, shape, value)| {
            (0..shape.iter().product::<usize>())
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&buffers)
        .map(|((name, shape, _), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
    serialize_to_file(views, None, path).unwrap();
}

fn write_fixture(directory: &Path) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 5,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false,
            "attention_schedule": [{"sliding": {"window": 2}}]
        }))
        .unwrap(),
    )
    .unwrap();
    let tensors = [
        ("model.embed_tokens.weight", vec![5, 4], 0.01),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.gate_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.up_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.input_layernorm.weight", vec![4], 1.0),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![4],
            1.0,
        ),
        ("model.norm.weight", vec![4], 1.0),
        ("lm_head.weight", vec![5, 4], 0.01),
    ];
    write_f32_shard(&directory.join("model.safetensors"), &tensors);
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| ((*name).to_string(), serde_json::json!("model.safetensors")))
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_llama_gguf_fixture(path: &Path) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("llama.block_count".into(), GgufMetadataValue::Uint32(1)),
        (
            "llama.embedding_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "llama.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "llama.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "llama.attention.key_length".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "llama.feed_forward_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        ("llama.context_length".into(), GgufMetadataValue::Uint32(32)),
        ("llama.vocab_size".into(), GgufMetadataValue::Uint32(5)),
    ]);
    let specs = [
        ("token_embd.weight", vec![4, 5], 0.01f32),
        ("blk.0.attn_q.weight", vec![4, 4], 0.01),
        ("blk.0.attn_k.weight", vec![4, 4], 0.01),
        ("blk.0.attn_v.weight", vec![4, 4], 0.01),
        ("blk.0.attn_output.weight", vec![4, 4], 0.01),
        ("blk.0.ffn_gate.weight", vec![4, 4], 0.01),
        ("blk.0.ffn_up.weight", vec![4, 4], 0.01),
        ("blk.0.ffn_down.weight", vec![4, 4], 0.01),
        ("blk.0.attn_norm.weight", vec![4], 1.0),
        ("blk.0.ffn_norm.weight", vec![4], 1.0),
        ("output_norm.weight", vec![4], 1.0),
        ("output.weight", vec![4, 5], 0.01),
    ];
    let payloads = specs
        .iter()
        .map(|(_, dimensions, value)| {
            (0..dimensions.iter().product::<u64>())
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tensors = specs
        .iter()
        .zip(&payloads)
        .map(|((name, dimensions, _), data)| TensorInput {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn patterned_values(length: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let centered = ((index * 17 + phase * 11) % 29) as f32 - 14.0;
            centered * scale
        })
        .collect()
}

fn qwen2_gguf_specs(with_biases: bool) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let mut specs = vec![(
        "token_embd.weight".into(),
        vec![8, 8],
        patterned_values(64, 0.015, 1),
    )];
    for layer in 0..2 {
        let phase = layer * 10;
        specs.extend([
            (
                format!("blk.{layer}.attn_q.weight"),
                vec![8, 8],
                patterned_values(64, 0.012, phase + 2),
            ),
            (
                format!("blk.{layer}.attn_q.bias"),
                vec![8],
                if with_biases {
                    patterned_values(8, 0.035, phase + 3)
                } else {
                    vec![0.0; 8]
                },
            ),
            (
                format!("blk.{layer}.attn_k.weight"),
                vec![8, 4],
                patterned_values(32, 0.014, phase + 4),
            ),
            (
                format!("blk.{layer}.attn_k.bias"),
                vec![4],
                if with_biases {
                    patterned_values(4, 0.04, phase + 5)
                } else {
                    vec![0.0; 4]
                },
            ),
            (
                format!("blk.{layer}.attn_v.weight"),
                vec![8, 4],
                patterned_values(32, 0.013, phase + 6),
            ),
            (
                format!("blk.{layer}.attn_v.bias"),
                vec![4],
                if with_biases {
                    patterned_values(4, 0.045, phase + 7)
                } else {
                    vec![0.0; 4]
                },
            ),
            (
                format!("blk.{layer}.attn_output.weight"),
                vec![8, 8],
                patterned_values(64, 0.011, phase + 8),
            ),
            (
                format!("blk.{layer}.ffn_gate.weight"),
                vec![8, 16],
                patterned_values(128, 0.009, phase + 9),
            ),
            (
                format!("blk.{layer}.ffn_up.weight"),
                vec![8, 16],
                patterned_values(128, 0.008, phase + 10),
            ),
            (
                format!("blk.{layer}.ffn_down.weight"),
                vec![16, 8],
                patterned_values(128, 0.01, phase + 11),
            ),
            (
                format!("blk.{layer}.attn_norm.weight"),
                vec![8],
                vec![1.0; 8],
            ),
            (
                format!("blk.{layer}.ffn_norm.weight"),
                vec![8],
                vec![1.0; 8],
            ),
        ]);
    }
    specs.extend([
        ("output_norm.weight".into(), vec![8], vec![1.0; 8]),
        (
            "output.weight".into(),
            vec![8, 8],
            patterned_values(64, 0.017, 23),
        ),
    ]);
    specs
}

fn qwen2_gguf_payload_bytes() -> u64 {
    qwen2_gguf_specs(true)
        .iter()
        .map(|(_, _, values)| values.len() as u64 * 4)
        .sum()
}

fn qwen2_gguf_metadata(
    hidden_size: u32,
    head_dim: u32,
    intermediate_size: u32,
    vocab_size: u32,
) -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("qwen2".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("qwen2.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "qwen2.embedding_length".into(),
            GgufMetadataValue::Uint32(hidden_size),
        ),
        (
            "qwen2.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "qwen2.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen2.attention.key_length".into(),
            GgufMetadataValue::Uint32(head_dim),
        ),
        (
            "qwen2.attention.value_length".into(),
            GgufMetadataValue::Uint32(head_dim),
        ),
        (
            "qwen2.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(head_dim),
        ),
        (
            "qwen2.feed_forward_length".into(),
            GgufMetadataValue::Uint32(intermediate_size),
        ),
        (
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.000001),
        ),
        ("qwen2.context_length".into(), GgufMetadataValue::Uint32(32)),
        (
            "qwen2.rope.freq_base".into(),
            GgufMetadataValue::Float32(1_000_000.0),
        ),
        (
            "qwen2.vocab_size".into(),
            GgufMetadataValue::Uint32(vocab_size),
        ),
        (
            "qwen2.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen2.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(vec![false, true])),
        ),
    ])
}

fn write_qwen2_gguf_fixture(path: &Path, with_biases: bool) {
    let metadata = qwen2_gguf_metadata(8, 2, 16, 8);
    let specs = qwen2_gguf_specs(with_biases);
    let payloads = specs
        .iter()
        .map(|(_, _, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tensors = specs
        .iter()
        .zip(&payloads)
        .map(|((name, dimensions, _), data)| TensorInput {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

struct QuantizedGgufTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

fn q8_0_payload(elements: u64, phase: usize) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let blocks = usize::try_from(elements / 32).unwrap();
    let mut data = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        // Exact little-endian f16 encodings for 0.015625, 0.017578125,
        // and 0.01953125 keep the fixture dependency-free and deterministic.
        let scale_bits = [0x2400u16, 0x2480, 0x2500][block % 3];
        data.extend_from_slice(&scale_bits.to_le_bytes());
        data.extend((0..32).map(|index| {
            let quantized = ((index * 13 + block * 7 + phase * 11) % 127) as i16 - 63;
            quantized as i8 as u8
        }));
    }
    data
}

fn q8_0_tensor(name: impl Into<String>, dimensions: Vec<u64>, phase: usize) -> QuantizedGgufTensor {
    let elements = dimensions.iter().product();
    QuantizedGgufTensor {
        name: name.into(),
        dimensions,
        ggml_type: GgmlType::Q8_0,
        data: q8_0_payload(elements, phase),
    }
}

fn f32_gguf_tensor(
    name: impl Into<String>,
    dimensions: Vec<u64>,
    values: Vec<f32>,
) -> QuantizedGgufTensor {
    assert_eq!(dimensions.iter().product::<u64>() as usize, values.len());
    QuantizedGgufTensor {
        name: name.into(),
        dimensions,
        ggml_type: GgmlType::F32,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn qwen2_q8_0_gguf_specs(with_biases: bool) -> Vec<QuantizedGgufTensor> {
    let mut specs = vec![q8_0_tensor("token_embd.weight", vec![64, 64], 1)];
    for layer in 0..2 {
        let phase = layer * 10;
        specs.extend([
            q8_0_tensor(
                format!("blk.{layer}.attn_q.weight"),
                vec![64, 64],
                phase + 2,
            ),
            f32_gguf_tensor(
                format!("blk.{layer}.attn_q.bias"),
                vec![64],
                if with_biases {
                    patterned_values(64, 0.02, phase + 3)
                } else {
                    vec![0.0; 64]
                },
            ),
            q8_0_tensor(
                format!("blk.{layer}.attn_k.weight"),
                vec![64, 32],
                phase + 4,
            ),
            f32_gguf_tensor(
                format!("blk.{layer}.attn_k.bias"),
                vec![32],
                if with_biases {
                    patterned_values(32, 0.025, phase + 5)
                } else {
                    vec![0.0; 32]
                },
            ),
            q8_0_tensor(
                format!("blk.{layer}.attn_v.weight"),
                vec![64, 32],
                phase + 6,
            ),
            f32_gguf_tensor(
                format!("blk.{layer}.attn_v.bias"),
                vec![32],
                if with_biases {
                    patterned_values(32, 0.03, phase + 7)
                } else {
                    vec![0.0; 32]
                },
            ),
            q8_0_tensor(
                format!("blk.{layer}.attn_output.weight"),
                vec![64, 64],
                phase + 8,
            ),
            q8_0_tensor(
                format!("blk.{layer}.ffn_gate.weight"),
                vec![64, 64],
                phase + 9,
            ),
            q8_0_tensor(
                format!("blk.{layer}.ffn_up.weight"),
                vec![64, 64],
                phase + 10,
            ),
            q8_0_tensor(
                format!("blk.{layer}.ffn_down.weight"),
                vec![64, 64],
                phase + 11,
            ),
            f32_gguf_tensor(
                format!("blk.{layer}.attn_norm.weight"),
                vec![64],
                vec![1.0; 64],
            ),
            f32_gguf_tensor(
                format!("blk.{layer}.ffn_norm.weight"),
                vec![64],
                vec![1.0; 64],
            ),
        ]);
    }
    specs.extend([
        f32_gguf_tensor("output_norm.weight", vec![64], vec![1.0; 64]),
        q8_0_tensor("output.weight", vec![64, 64], 23),
    ]);
    specs
}

fn qwen2_q8_0_gguf_payload_bytes() -> u64 {
    qwen2_q8_0_gguf_specs(true)
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn write_qwen2_q8_0_gguf_fixture(path: &Path, with_biases: bool) {
    let mut metadata = qwen2_gguf_metadata(64, 16, 64, 64);
    metadata.insert("general.file_type".into(), GgufMetadataValue::Uint32(7));
    let specs = qwen2_q8_0_gguf_specs(with_biases);
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: tensor.ggml_type,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn write_deepseek_fixture(directory: &Path, layers: i32) {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": layers,
        "moe_layer_freq": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": 0,
        "split_kv_b": false,
        "tie_word_embeddings": false
    });
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = deepseek_v3::model_args_from_config_value(&config).unwrap();
    let mut model = deepseek_v3::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
        };
    }
    let arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

struct ChildGuard(Vec<Child>);

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.0
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
}

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "tensor Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_tensor_parallel_ring ring_two_process_tensor_parallel -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::LlamaSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies bounded GGUF reads through the same two-rank generalized engine.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::LlamaGguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies Qwen2 GQA, learned Q/K/V biases, mixed attention, and bounded GGUF reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Qwen2Gguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies the generalized engine's fully resident policy with Qwen2 TP math.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Qwen2Gguf,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies block-aligned Q8_0 Qwen2 tensor sharding and bounded GGUF reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen2_q8_0_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Qwen2Q8Gguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies DeepSeek MLA paged-prefix persistence across two tensor ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_tensor_parallel_persistence() {
    run_ring_tensor_parallel(
        FixtureFamily::DeepSeekSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

fn run_ring_tensor_parallel(family: FixtureFamily, parameter_residency: TensorParameterResidency) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = match family {
        FixtureFamily::LlamaSafetensors => {
            write_fixture(checkpoint.path());
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::LlamaGguf => {
            let path = checkpoint.path().join("model.gguf");
            write_llama_gguf_fixture(&path);
            path
        }
        FixtureFamily::DeepSeekSafetensors => {
            write_deepseek_fixture(checkpoint.path(), 1);
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::Qwen2Gguf => {
            let path = checkpoint.path().join("model.gguf");
            write_qwen2_gguf_fixture(&path, true);
            path
        }
        FixtureFamily::Qwen2Q8Gguf => {
            let path = checkpoint.path().join("model.gguf");
            write_qwen2_q8_0_gguf_fixture(&path, true);
            path
        }
    };
    let zero_bias_checkpoint = tempfile::tempdir().unwrap();
    let zero_bias_checkpoint_path = zero_bias_checkpoint.path().join("model.gguf");
    match family {
        FixtureFamily::Qwen2Gguf => {
            write_qwen2_gguf_fixture(&zero_bias_checkpoint_path, false);
        }
        FixtureFamily::Qwen2Q8Gguf => {
            write_qwen2_q8_0_gguf_fixture(&zero_bias_checkpoint_path, false);
        }
        _ => {}
    }
    let prompt_cache = tempfile::tempdir().unwrap();
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    let mut reservations = [Some(first_socket), Some(second_socket)];
    for (rank, reservation) in reservations.iter_mut().enumerate() {
        // Release only the address this rank will bind immediately before its
        // process is spawned. Keeping the peer address reserved closes the
        // previous socket-setup race where either port could be stolen between
        // dropping both listeners and launching the workers.
        drop(reservation.take());
        children.0.push(
            Command::new(&executable)
                .args(["--exact", "tensor_ring_worker", "--nocapture"])
                .env(WORKER_RANK, rank.to_string())
                .env(CHECKPOINT_DIR, &checkpoint_path)
                .env(FIXTURE_FAMILY, family.name())
                .env(PARAMETER_RESIDENCY, parameter_residency.name())
                .env(PROMPT_CACHE_ROOT, prompt_cache.path())
                .env(ZERO_BIAS_CHECKPOINT, &zero_bias_checkpoint_path)
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process tensor-parallel Ring test failed:\n{}",
        failures.join("\n\n")
    );
}
