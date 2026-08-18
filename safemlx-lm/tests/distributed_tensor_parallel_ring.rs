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
    ops::{indexing::TryIndexOp, GgufMetadataValue},
    random::{self, RandomState},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_gguf::{GgmlType, TensorInput, Writer};
use safemlx_lm::{
    architectures::{
        deepseek_v3::model as deepseek_v3, gpt_oss::model as gpt_oss_model,
        kimi_linear::model as kimi_model, lfm2::model as lfm2_model, llama::model as llama_model,
        nemotron_h::model as nemotron_model, qwen::dense as dense_qwen,
    },
    core::BackendSession,
    nn::generation::CausalLm,
    runtime::cache::KeyValueCache,
    runtime::checkpoint::binding::canonical_checkpoint_name,
    runtime::generation::sampler::DefaultSampler,
    CacheResidencyPolicy, DenseDiskStreamLoadOptions, DeviceAssignment, LayerCachePolicy,
    LayerWeightResidency, LayerwiseLoadOptions, MlxBackend, MlxParallelContext, ModelLoadOptions,
    PagedCacheOptions, ParallelModelInfo, PromptCacheDescriptor, PromptCacheOptions,
    PromptCacheTopology, WeightResidency,
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
    DenseDiskStream,
}

impl TensorParameterResidency {
    const fn name(self) -> &'static str {
        match self {
            Self::LayerwiseHost => "layerwise-host",
            Self::FullyResident => "fully-resident",
            Self::DenseDiskStream => "dense-disk-stream",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "layerwise-host" => Self::LayerwiseHost,
            "fully-resident" => Self::FullyResident,
            "dense-disk-stream" => Self::DenseDiskStream,
            _ => panic!("unknown tensor-parallel parameter residency {value:?}"),
        }
    }

    fn load_options(self) -> LayerWeightResidency {
        match self {
            Self::LayerwiseHost => {
                LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::default())
            }
            Self::FullyResident => LayerWeightResidency::FullyResident,
            Self::DenseDiskStream => LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixtureFamily {
    LlamaSafetensors,
    LlamaGguf,
    DeepSeekSafetensors,
    DeepSeekGguf,
    Qwen3MoeSafetensors,
    Qwen2Gguf,
    Qwen2Q8Gguf,
    Lfm2Safetensors,
    Lfm2Gguf,
    Lfm2Q8Gguf,
    GptOssSafetensors,
    GptOssGguf,
    KimiLinearSafetensors,
    KimiLinearGguf,
    NemotronSafetensors,
    NemotronGguf,
}

impl FixtureFamily {
    const fn name(self) -> &'static str {
        match self {
            Self::LlamaSafetensors => "llama-safetensors",
            Self::LlamaGguf => "llama-gguf",
            Self::DeepSeekSafetensors => "deepseek-safetensors",
            Self::DeepSeekGguf => "deepseek-gguf",
            Self::Qwen3MoeSafetensors => "qwen3-moe-safetensors",
            Self::Qwen2Gguf => "qwen2-gguf",
            Self::Qwen2Q8Gguf => "qwen2-q8_0-gguf",
            Self::Lfm2Safetensors => "lfm2-safetensors",
            Self::Lfm2Gguf => "lfm2-gguf",
            Self::Lfm2Q8Gguf => "lfm2-q8_0-gguf",
            Self::GptOssSafetensors => "gpt-oss-safetensors",
            Self::GptOssGguf => "gpt-oss-gguf",
            Self::KimiLinearSafetensors => "kimi-linear-safetensors",
            Self::KimiLinearGguf => "kimi-linear-gguf",
            Self::NemotronSafetensors => "nemotron-safetensors",
            Self::NemotronGguf => "nemotron-gguf",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "llama-safetensors" => Self::LlamaSafetensors,
            "llama-gguf" => Self::LlamaGguf,
            "deepseek-safetensors" => Self::DeepSeekSafetensors,
            "deepseek-gguf" => Self::DeepSeekGguf,
            "qwen3-moe-safetensors" => Self::Qwen3MoeSafetensors,
            "qwen2-gguf" => Self::Qwen2Gguf,
            "qwen2-q8_0-gguf" => Self::Qwen2Q8Gguf,
            "lfm2-safetensors" => Self::Lfm2Safetensors,
            "lfm2-gguf" => Self::Lfm2Gguf,
            "lfm2-q8_0-gguf" => Self::Lfm2Q8Gguf,
            "gpt-oss-safetensors" => Self::GptOssSafetensors,
            "gpt-oss-gguf" => Self::GptOssGguf,
            "kimi-linear-safetensors" => Self::KimiLinearSafetensors,
            "kimi-linear-gguf" => Self::KimiLinearGguf,
            "nemotron-safetensors" => Self::NemotronSafetensors,
            "nemotron-gguf" => Self::NemotronGguf,
            _ => panic!("unknown tensor-parallel fixture family {value:?}"),
        }
    }

    const fn layer_count(self) -> usize {
        match self {
            Self::DeepSeekSafetensors
            | Self::DeepSeekGguf
            | Self::Qwen2Gguf
            | Self::Qwen2Q8Gguf
            | Self::Lfm2Safetensors
            | Self::Lfm2Gguf
            | Self::Lfm2Q8Gguf => 2,
            Self::KimiLinearSafetensors | Self::KimiLinearGguf => 2,
            Self::NemotronSafetensors => 4,
            Self::NemotronGguf => 3,
            _ => 1,
        }
    }

    const fn vocab_size(self) -> usize {
        match self {
            Self::DeepSeekSafetensors | Self::DeepSeekGguf => 13,
            Self::Qwen2Gguf => 8,
            Self::Qwen2Q8Gguf => 64,
            Self::Qwen3MoeSafetensors => 16,
            Self::Lfm2Safetensors => 13,
            Self::Lfm2Gguf => 13,
            Self::Lfm2Q8Gguf => 64,
            Self::GptOssSafetensors | Self::GptOssGguf => 64,
            Self::KimiLinearSafetensors | Self::KimiLinearGguf => 13,
            Self::NemotronSafetensors | Self::NemotronGguf => 13,
            _ => 5,
        }
    }

    const fn model_type(self) -> &'static str {
        match self {
            Self::LlamaSafetensors | Self::LlamaGguf => "llama",
            Self::DeepSeekSafetensors | Self::DeepSeekGguf => "deepseek_v3",
            Self::Qwen3MoeSafetensors => "qwen3_moe",
            Self::Qwen2Gguf | Self::Qwen2Q8Gguf => "qwen2",
            Self::Lfm2Safetensors | Self::Lfm2Gguf | Self::Lfm2Q8Gguf => "lfm2",
            Self::GptOssSafetensors | Self::GptOssGguf => "gpt_oss",
            Self::KimiLinearSafetensors | Self::KimiLinearGguf => "kimi_linear",
            Self::NemotronSafetensors | Self::NemotronGguf => "nemotron_h",
        }
    }

    const fn persists_paged_prefix(self) -> bool {
        !self.is_qwen2()
    }

    const fn is_qwen2(self) -> bool {
        matches!(self, Self::Qwen2Gguf | Self::Qwen2Q8Gguf)
    }

    const fn is_deepseek(self) -> bool {
        matches!(self, Self::DeepSeekSafetensors | Self::DeepSeekGguf)
    }

    fn deepseek_payload_bytes(self) -> u64 {
        match self {
            Self::DeepSeekGguf => deepseek_gguf_payload_bytes(),
            _ => panic!("non-GGUF DeepSeek fixture has no GGUF tensor payload"),
        }
    }

    const fn is_lfm2(self) -> bool {
        matches!(
            self,
            Self::Lfm2Safetensors | Self::Lfm2Gguf | Self::Lfm2Q8Gguf
        )
    }

    const fn is_gpt_oss(self) -> bool {
        matches!(self, Self::GptOssSafetensors | Self::GptOssGguf)
    }

    const fn is_kimi_linear(self) -> bool {
        matches!(self, Self::KimiLinearSafetensors | Self::KimiLinearGguf)
    }

    const fn is_nemotron(self) -> bool {
        matches!(self, Self::NemotronSafetensors | Self::NemotronGguf)
    }

    fn nemotron_payload_bytes(self) -> u64 {
        match self {
            Self::NemotronGguf => nemotron_gguf_payload_bytes(),
            _ => panic!("non-GGUF Nemotron fixture has no GGUF tensor payload"),
        }
    }

    fn kimi_linear_payload_bytes(self) -> u64 {
        match self {
            Self::KimiLinearGguf => kimi_linear_gguf_payload_bytes(),
            _ => panic!("non-GGUF Kimi fixture has no GGUF tensor payload"),
        }
    }

    fn gpt_oss_payload_bytes(self) -> u64 {
        match self {
            Self::GptOssGguf => 204_960,
            _ => panic!("non-GGUF GPT-OSS fixture has no GGUF tensor payload"),
        }
    }

    const fn lfm2_head_dim(self) -> i32 {
        match self {
            Self::Lfm2Safetensors | Self::Lfm2Gguf => 2,
            Self::Lfm2Q8Gguf => 16,
            _ => panic!("non-LFM2 fixture has no LFM2 head dimension"),
        }
    }

    const fn lfm2_local_kv_heads(self, rank: usize) -> i32 {
        match self {
            Self::Lfm2Safetensors | Self::Lfm2Gguf => {
                if rank == 0 {
                    2
                } else {
                    1
                }
            }
            Self::Lfm2Q8Gguf => 1,
            _ => panic!("non-LFM2 fixture has no LFM2 KV geometry"),
        }
    }

    const fn lfm2_local_convolution_channels(self) -> i32 {
        match self {
            Self::Lfm2Safetensors | Self::Lfm2Gguf => 6,
            Self::Lfm2Q8Gguf => 32,
            _ => panic!("non-LFM2 fixture has no LFM2 convolution geometry"),
        }
    }

    fn lfm2_payload_bytes(self) -> u64 {
        match self {
            Self::Lfm2Gguf => lfm2_f32_gguf_payload_bytes(),
            Self::Lfm2Q8Gguf => lfm2_q8_0_gguf_payload_bytes(),
            _ => panic!("non-GGUF LFM2 fixture has no GGUF tensor payload"),
        }
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

struct GeneralizedTensorModel(safemlx_lm::core::PreparedModel<safemlx_lm::MlxModel>);

impl GeneralizedTensorModel {
    fn load(
        checkpoint: &Path,
        family: FixtureFamily,
        residency: TensorParameterResidency,
        topology: MlxParallelContext,
        stream: &Stream,
    ) -> Self {
        let options = ModelLoadOptions::default()
            .with_parallel_topology(topology)
            .with_weight_residency(WeightResidency::with_layers(residency.load_options()));
        let backend = MlxBackend::new(stream, stream);
        let loaded = safemlx_lm::load_model(&backend, checkpoint, options).unwrap();
        assert_eq!(loaded.model_type(), family.model_type());
        Self(loaded)
    }

    fn parallel_info(&self) -> &ParallelModelInfo {
        self.complete().parallel_info().unwrap()
    }

    fn prompt_cache_architecture_fingerprint(&self) -> String {
        self.complete()
            .prompt_cache_architecture_fingerprint()
            .unwrap()
    }

    fn prompt_cache_layer_layout(&self) -> safemlx_lm::LayerSchedule<LayerCachePolicy> {
        self.complete().prompt_cache_layer_layout().unwrap()
    }

    fn checkpoint_diagnostics(
        &self,
    ) -> safemlx_lm::runtime::checkpoint::store::WeightStoreDiagnostics {
        checkpoint_diagnostics(self.complete())
    }

    fn complete(&self) -> &safemlx_lm::api::Model {
        match &self.0.inner {
            safemlx_lm::backend::mlx::MlxModelKind::Complete(model) => model,
            _ => panic!("tensor-parallel fixture did not load a complete model"),
        }
    }
}

fn checkpoint_diagnostics(
    model: &safemlx_lm::api::Model,
) -> safemlx_lm::runtime::checkpoint::store::WeightStoreDiagnostics {
    match model {
        safemlx_lm::api::Model::Llama(model) => model.checkpoint_store().diagnostics().unwrap(),
        safemlx_lm::api::Model::DeepSeekV3(model) => {
            model.checkpoint_store().diagnostics().unwrap()
        }
        safemlx_lm::api::Model::DenseQwen(model) => model.checkpoint_store().diagnostics().unwrap(),
        safemlx_lm::api::Model::Lfm2(model) => model.checkpoint_store().diagnostics().unwrap(),
        safemlx_lm::api::Model::GptOss(model) => model.checkpoint_store().diagnostics().unwrap(),
        safemlx_lm::api::Model::KimiLinear(model) => {
            model.checkpoint_store().diagnostics().unwrap()
        }
        safemlx_lm::api::Model::NemotronH(model) => model.checkpoint_store().diagnostics().unwrap(),
        model => panic!(
            "checkpoint diagnostics unavailable for {}",
            model.model_type()
        ),
    }
}

trait GeneralizedTensorCacheExt {
    fn offset(&self) -> i32;
    fn assert_qwen2_local_cache_geometry(
        &self,
        local_kv_heads: i32,
        full_sequence: i32,
        head_dim: i32,
    );
    fn assert_qwen3_moe_local_cache_geometry(&self, full_sequence: i32);
    fn assert_lfm2_local_cache_geometry(
        &self,
        local_kv_heads: i32,
        local_convolution_channels: i32,
        head_dim: i32,
        sequence: i32,
    );
    fn assert_gpt_oss_local_cache_geometry(&self, local_kv_heads: i32, sequence: i32);
    fn assert_kimi_local_cache_geometry(&self, local_heads: i32, sequence: i32);
    fn assert_nemotron_local_cache_geometry(
        &self,
        local_mamba_heads: i32,
        local_mamba_groups: i32,
        local_kv_heads: i32,
        sequence: i32,
    );
}

impl GeneralizedTensorCacheExt for safemlx_lm::api::ModelCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Llama(cache) => cache.offset(),
            Self::DeepSeekV3(cache) => cache.offset(),
            Self::KeyValue(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::PagedKeyValue(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Lfm2(cache) => cache.offset(),
            Self::GptOss(cache) => cache.layers.first().map_or(0, |cache| match cache {
                gpt_oss_model::LayerCache::Full(cache) => cache.offset(),
                gpt_oss_model::LayerCache::Sliding(cache) => cache.offset(),
                gpt_oss_model::LayerCache::Paged(cache) => cache.offset(),
            }),
            Self::KimiLinear(cache) => cache.offset(),
            Self::NemotronH(cache) => cache.offset(),
            _ => panic!("offset unavailable for cache variant"),
        }
    }

    fn assert_qwen2_local_cache_geometry(
        &self,
        local_kv_heads: i32,
        full_sequence: i32,
        head_dim: i32,
    ) {
        let Self::KeyValue(layers) = self else {
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
            assert_eq!(
                arrays[0].shape(),
                &[1, local_kv_heads, retained_sequence, head_dim]
            );
            assert_eq!(
                arrays[1].shape(),
                &[1, local_kv_heads, retained_sequence, head_dim]
            );
            assert_eq!(cache.max_size(), (index == 1).then_some(2));
        }
    }

    fn assert_qwen3_moe_local_cache_geometry(&self, full_sequence: i32) {
        let Self::KeyValue(layers) = self else {
            panic!("Qwen3 MoE local cache assertion requires a concat dense-Qwen cache")
        };
        assert_eq!(layers.len(), 1);
        let cache = layers[0].as_ref().unwrap();
        for array in cache.retained_arrays() {
            assert_eq!(array.shape(), &[1, 1, full_sequence, 4]);
        }
        assert_eq!(cache.max_size(), None);
    }

    fn assert_lfm2_local_cache_geometry(
        &self,
        local_kv_heads: i32,
        local_convolution_channels: i32,
        head_dim: i32,
        sequence: i32,
    ) {
        let Self::Lfm2(cache) = self else {
            panic!("LFM2 local cache assertion requires an LFM2 cache")
        };
        assert_eq!(cache.layers.len(), 2);
        let lfm2_model::LayerCache::Conv(conv) = &cache.layers[0] else {
            panic!("LFM2 layer 0 must retain convolution state")
        };
        assert_eq!(
            conv.state.as_ref().unwrap().shape(),
            &[1, 2, local_convolution_channels]
        );
        let lfm2_model::LayerCache::Attention(attention) = &cache.layers[1] else {
            panic!("LFM2 layer 1 must retain KV state")
        };
        assert!(matches!(
            attention,
            safemlx_lm::runtime::cache::LiveKeyValueCache::Paged(_)
        ));
        assert_eq!(attention.offset(), sequence);
        for array in attention.retained_arrays() {
            assert_eq!(array.shape(), &[1, local_kv_heads, 256, head_dim]);
        }
        let report = cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, sequence as u64);
        assert!(report.current_device_bytes > 0);
    }

    fn assert_gpt_oss_local_cache_geometry(&self, local_kv_heads: i32, sequence: i32) {
        let Self::GptOss(cache) = self else {
            panic!("GPT-OSS local cache assertion requires a GPT-OSS cache")
        };
        assert_eq!(cache.layers.len(), 1);
        let (arrays, offset) = match &cache.layers[0] {
            gpt_oss_model::LayerCache::Full(cache) => (cache.retained_arrays(), cache.offset()),
            gpt_oss_model::LayerCache::Sliding(cache) => (cache.retained_arrays(), cache.offset()),
            gpt_oss_model::LayerCache::Paged(cache) => (cache.retained_arrays(), cache.offset()),
        };
        for array in arrays {
            assert_eq!(array.shape(), &[1, local_kv_heads, 256, 32]);
        }
        assert_eq!(offset, sequence);
    }

    fn assert_kimi_local_cache_geometry(&self, local_heads: i32, sequence: i32) {
        let Self::KimiLinear(cache) = self else {
            panic!("Kimi local cache assertion requires a Kimi cache")
        };
        assert_eq!(cache.layers.len(), 2);
        let kimi_model::LayerCache::Kda(kda) = &cache.layers[0] else {
            panic!("Kimi layer 0 must retain KDA state")
        };
        for convolution in [&kda.q_conv, &kda.k_conv, &kda.v_conv] {
            assert_eq!(convolution.offset, sequence);
            assert_eq!(
                convolution.state.as_ref().unwrap().shape(),
                &[1, 1, local_heads * 4]
            );
        }
        assert_eq!(
            kda.recurrent_state.as_ref().unwrap().shape(),
            &[1, local_heads, 4, 4]
        );
        let kimi_model::LayerCache::Mla(mla) = &cache.layers[1] else {
            panic!("Kimi layer 1 must retain MLA state")
        };
        assert_eq!(mla.offset(), sequence);
        assert!(mla.is_paged());
        assert!(mla.arrays().is_none());
        let report = cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, sequence as u64);
        assert!(report.current_device_bytes > 0);
    }

    fn assert_nemotron_local_cache_geometry(
        &self,
        local_mamba_heads: i32,
        local_mamba_groups: i32,
        local_kv_heads: i32,
        sequence: i32,
    ) {
        let Self::NemotronH(cache) = self else {
            panic!("Nemotron local cache assertion requires a Nemotron cache")
        };
        let nemotron_model::LayerCache::Mamba(mamba) = &cache.layers[0] else {
            panic!("Nemotron layer 0 must retain Mamba state")
        };
        let local_intermediate = local_mamba_heads * 2;
        let local_conv = local_intermediate + 2 * local_mamba_groups * 2;
        assert_eq!(
            mamba.conv_state.as_ref().unwrap().shape(),
            &[1, 2, local_conv]
        );
        assert_eq!(
            mamba.ssm_state.as_ref().unwrap().shape(),
            &[1, local_mamba_heads, 2, 2]
        );
        assert_eq!(mamba.offset, sequence);
        let attention_index = cache.layers.len() - 1;
        let nemotron_model::LayerCache::Attention(attention) = &cache.layers[attention_index]
        else {
            panic!("last Nemotron layer must retain KV state")
        };
        assert_eq!(attention.offset(), sequence);
        assert!(matches!(
            attention,
            nemotron_model::AttentionCache::Paged(_)
        ));
        for array in attention.retained_arrays() {
            assert_eq!(array.dim(1), local_kv_heads);
            assert_eq!(array.dim(-1), 2);
        }
        let report = cache.residency_report().unwrap().unwrap();
        assert_eq!(report.logical_cached_tokens, sequence as u64);
        assert!(report.current_device_bytes > 0);
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
        MlxParallelContext::for_group(&group, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let model =
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
        TensorParameterResidency::LayerwiseHost | TensorParameterResidency::DenseDiskStream => {
            assert!(info.pinned_device_parameter_bytes() < info.local_parameter_bytes());
            assert!(info.maximum_device_parameter_bytes() <= info.local_parameter_bytes());
        }
    }
    assert!(info.global_parameter_bytes() >= info.local_parameter_bytes());
    if matches!(
        family,
        FixtureFamily::LlamaSafetensors | FixtureFamily::LlamaGguf
    ) {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        let expected_kv_heads = if expected_rank == 0 { 2 } else { 1 };
        let layout = model.prompt_cache_layer_layout();
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        } = layout.get(0).unwrap()
        else {
            panic!("Llama fixture must expose ordinary KV cache geometry");
        };
        assert_eq!(num_key_value_heads.get(), expected_kv_heads);
        assert_eq!(head_dim.get(), 2);
    }
    if family.is_lfm2() {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        let layout = model.prompt_cache_layer_layout();
        let LayerCachePolicy::FixedState { tensors } = layout.get(0).unwrap() else {
            panic!("LFM2 convolution layer must expose fixed state")
        };
        assert_eq!(
            tensors[0].shape.last(),
            Some(
                &safemlx_lm::StateTensorDimension::fixed(family.lfm2_local_convolution_channels())
                    .unwrap()
            )
        );
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        } = layout.get(1).unwrap()
        else {
            panic!("LFM2 attention layer must expose ordinary KV geometry")
        };
        assert_eq!(
            num_key_value_heads.get(),
            family.lfm2_local_kv_heads(expected_rank) as u32
        );
        assert_eq!(head_dim.get(), family.lfm2_head_dim() as u32);
        if matches!(family, FixtureFamily::Lfm2Gguf | FixtureFamily::Lfm2Q8Gguf) {
            assert!(info.local_parameter_bytes() < family.lfm2_payload_bytes());
        }
    }
    if family.is_gpt_oss() {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        let expected_kv_heads = if expected_rank == 0 { 2 } else { 1 };
        let layout = model.prompt_cache_layer_layout();
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        } = layout.get(0).unwrap()
        else {
            panic!("GPT-OSS fixture must expose KV cache geometry")
        };
        assert_eq!(num_key_value_heads.get(), expected_kv_heads);
        assert_eq!(head_dim.get(), 32);
        if family == FixtureFamily::GptOssGguf {
            let diagnostics = model.checkpoint_diagnostics();
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < family.gpt_oss_payload_bytes(),
                "rank {expected_rank} read {} GGUF bytes for a {}-byte global GPT-OSS tensor payload",
                diagnostics.physical_read_bytes,
                family.gpt_oss_payload_bytes()
            );
        }
    }
    if family.is_kimi_linear() {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        let local_heads = if expected_rank == 0 { 2 } else { 1 };
        let layout = model.prompt_cache_layer_layout();
        let LayerCachePolicy::FixedState { tensors } = layout.get(0).unwrap() else {
            panic!("Kimi KDA layer must expose fixed state")
        };
        assert_eq!(
            tensors[3].shape[1],
            safemlx_lm::StateTensorDimension::fixed(local_heads).unwrap()
        );
        assert!(matches!(
            layout.get(1).unwrap(),
            LayerCachePolicy::CompressedLatentRotary { .. }
        ));
    }
    if family.is_nemotron() {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        let (local_heads, local_groups, local_kv) = if expected_rank == 0 {
            (4, 2, 2)
        } else {
            (2, 1, 1)
        };
        let layout = model.prompt_cache_layer_layout();
        let LayerCachePolicy::FixedState { tensors } = layout.get(0).unwrap() else {
            panic!("Nemotron Mamba layer must expose fixed state")
        };
        assert_eq!(
            tensors[0].shape[2],
            safemlx_lm::StateTensorDimension::fixed(local_heads * 2 + 2 * local_groups * 2)
                .unwrap()
        );
        assert_eq!(
            tensors[1].shape[1],
            safemlx_lm::StateTensorDimension::fixed(local_heads).unwrap()
        );
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        } = layout.get(layout.len() - 1).unwrap()
        else {
            panic!("last Nemotron layer must expose KV state")
        };
        assert_eq!(num_key_value_heads.get(), local_kv as u32);
        assert_eq!(head_dim.get(), 2);
        if family == FixtureFamily::NemotronGguf {
            assert!(info.local_parameter_bytes() < family.nemotron_payload_bytes());
        }
    }
    if family.is_deepseek() {
        assert!(info.local_parameter_bytes() < info.global_parameter_bytes());
        assert!(info
            .owned_tensors()
            .iter()
            .any(|name| name == "model.layers.1.mlp.experts.gate_proj"));
        if family == FixtureFamily::DeepSeekGguf {
            assert!(info.local_parameter_bytes() < family.deepseek_payload_bytes());
        }
    }
    assert!(info.owned_tensors().iter().any(|name| {
        let canonical = canonical_checkpoint_name(name);
        canonical == "model.embed_tokens.weight"
            || (family.is_nemotron() && canonical == "model.embeddings.weight")
    }));

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

    // Keep a mutable Nemotron tail through the geometry assertions so the
    // rank-local KV-head partition is observed directly, not only via its
    // prompt-cache descriptor. Other fixtures deliberately seal every token.
    let cache_block_size = if family.is_nemotron() { 4 } else { 1 };
    // Physical host-transfer allocations are page-rounded; keep the synthetic
    // cache pool large enough for one complete rank-local block on current MLX.
    let paged = PagedCacheOptions::new(cache_block_size, 64 * 1024, 64 * 1024, 1)
        .unwrap()
        .with_full_attention(true);
    let architecture_fingerprint = model.prompt_cache_architecture_fingerprint();
    let prompt_cache_layer_layout = model.prompt_cache_layer_layout();
    let cache_policy = if family.is_qwen2() || family == FixtureFamily::Qwen3MoeSafetensors {
        CacheResidencyPolicy::Device
    } else {
        CacheResidencyPolicy::Paged(paged.clone())
    };
    let backend = MlxBackend::with_distributed_world(&stream, &stream, &group);
    let mut session = safemlx_lm::core::Backend::create_session(&backend, model.0).unwrap();
    session.configure_cache(cache_policy).unwrap();
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
    let logits = session
        .prefill(
            &backend,
            safemlx_lm::runtime::media::input::ModelInput::new(&parts).into(),
        )
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap();
    assert_eq!(logits.shape(), &[1, vocab_size as i32]);

    if family.is_deepseek() {
        let mut reference = match family {
            FixtureFamily::DeepSeekSafetensors => {
                deepseek_v3::load_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::DeepSeekGguf => {
                deepseek_v3::load_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let reference_logits = reference
            .forward(
                deepseek_v3::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: Some(&mut reference_cache),
                },
                &stream,
            )
            .unwrap();
        let reference_logits = reference_logits
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 8e-5);
        if family == FixtureFamily::DeepSeekGguf {
            let diagnostics = checkpoint_diagnostics(session.test_complete_model());
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < family.deepseek_payload_bytes(),
                "rank {expected_rank} read {} GGUF bytes for a {}-byte global DeepSeek tensor payload",
                diagnostics.physical_read_bytes,
                family.deepseek_payload_bytes()
            );
        }
    }

    if family.is_nemotron() {
        let mut reference = match family {
            FixtureFamily::NemotronSafetensors => {
                nemotron_model::load_nemotron_h_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::NemotronGguf => {
                nemotron_model::load_nemotron_h_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
        let reference_logits = reference
            .prefill_input_logits(
                safemlx_lm::runtime::media::input::ModelInput::new(&parts),
                &mut reference_cache,
                &stream,
            )
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 8e-5);
        let (local_heads, local_groups, local_kv) = if expected_rank == 0 {
            (4, 2, 2)
        } else {
            (2, 1, 1)
        };
        session
            .test_complete_cache()
            .assert_nemotron_local_cache_geometry(local_heads, local_groups, local_kv, 2);
        if family == FixtureFamily::NemotronGguf {
            let diagnostics = checkpoint_diagnostics(session.test_complete_model());
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < family.nemotron_payload_bytes(),
                "rank {expected_rank} read {} GGUF bytes for a {}-byte global Nemotron tensor payload",
                diagnostics.physical_read_bytes,
                family.nemotron_payload_bytes()
            );
        }
    }

    if family == FixtureFamily::Qwen3MoeSafetensors {
        let mut reference = dense_qwen::load_safetensors(&checkpoint, &stream, &stream).unwrap();
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
        let reference_logits = reference_logits
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 5e-5);
        session
            .test_complete_cache()
            .assert_qwen3_moe_local_cache_geometry(2);
        let token = Array::from_slice(&[3u32], &[1, 1]);
        let distributed_decode = session
            .decode(&backend, token.clone())
            .unwrap()
            .wait()
            .unwrap()
            .into_logits()
            .unwrap();
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
        let reference_decode = reference_decode
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&distributed_decode, &reference_decode, 5e-5);
        session
            .test_complete_cache()
            .assert_qwen3_moe_local_cache_geometry(3);
        return;
    }

    if family.is_gpt_oss() {
        let mut reference = match family {
            FixtureFamily::GptOssSafetensors => {
                gpt_oss_model::load_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::GptOssGguf => {
                gpt_oss_model::load_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
        let reference_logits = reference
            .prefill_input_logits(
                safemlx_lm::runtime::media::input::ModelInput::new(&parts),
                &mut reference_cache,
                &stream,
            )
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 2e-4);
        session
            .test_complete_cache()
            .assert_gpt_oss_local_cache_geometry(if expected_rank == 0 { 2 } else { 1 }, 2);
    }

    if family.is_lfm2() {
        let mut reference = match family {
            FixtureFamily::Lfm2Safetensors => {
                lfm2_model::load_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::Lfm2Gguf | FixtureFamily::Lfm2Q8Gguf => {
                lfm2_model::load_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let parts = [safemlx_lm::runtime::media::input::InputPart::text_token_ids(&prompt)];
        let reference_logits = reference
            .prefill_input_logits(
                safemlx_lm::runtime::media::input::ModelInput::new(&parts),
                &mut reference_cache,
                &stream,
            )
            .unwrap();
        let tolerance = if family == FixtureFamily::Lfm2Q8Gguf {
            2e-4
        } else {
            5e-5
        };
        assert_arrays_close(&logits, &reference_logits, tolerance);
        session
            .test_complete_cache()
            .assert_lfm2_local_cache_geometry(
                family.lfm2_local_kv_heads(expected_rank),
                family.lfm2_local_convolution_channels(),
                family.lfm2_head_dim(),
                2,
            );
        if matches!(family, FixtureFamily::Lfm2Gguf | FixtureFamily::Lfm2Q8Gguf) {
            let diagnostics = checkpoint_diagnostics(session.test_complete_model());
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < family.lfm2_payload_bytes(),
                "rank {expected_rank} read {} GGUF bytes for a {}-byte global LFM2 tensor payload",
                diagnostics.physical_read_bytes,
                family.lfm2_payload_bytes()
            );
        }
    }

    if family.is_kimi_linear() {
        let mut reference = match family {
            FixtureFamily::KimiLinearSafetensors => {
                kimi_model::load_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::KimiLinearGguf => {
                kimi_model::load_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let reference_logits = reference
            .forward(
                kimi_model::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: Some(&mut reference_cache),
                },
                &stream,
            )
            .unwrap();
        let reference_logits = reference_logits
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 8e-5);
        session
            .test_complete_cache()
            .assert_kimi_local_cache_geometry(if expected_rank == 0 { 2 } else { 1 }, 2);
        if family == FixtureFamily::KimiLinearGguf {
            let diagnostics = checkpoint_diagnostics(session.test_complete_model());
            assert!(diagnostics.physical_reads > 0);
            assert!(
                diagnostics.physical_read_bytes < family.kimi_linear_payload_bytes(),
                "rank {expected_rank} read {} GGUF bytes for a {}-byte global Kimi tensor payload",
                diagnostics.physical_read_bytes,
                family.kimi_linear_payload_bytes()
            );
        }
    }

    if matches!(
        family,
        FixtureFamily::LlamaSafetensors | FixtureFamily::LlamaGguf
    ) {
        let mut reference = match family {
            FixtureFamily::LlamaSafetensors => {
                llama_model::load_resident_llama_model(&checkpoint, &stream, &stream).unwrap()
            }
            FixtureFamily::LlamaGguf => {
                llama_model::load_llama_gguf(&checkpoint, &stream, &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let mut reference_cache = reference.new_cache();
        let reference_logits = reference
            .forward(
                llama_model::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: &mut reference_cache,
                },
                &stream,
            )
            .unwrap();
        let reference_logits = reference_logits
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&logits, &reference_logits, 5e-5);
    }

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
        let reference_last = reference_logits
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&logits, &reference_last, tolerance);
        let local_kv_heads = if family == FixtureFamily::Qwen2Gguf {
            if expected_rank == 0 {
                2
            } else {
                1
            }
        } else {
            1
        };
        session
            .test_complete_cache()
            .assert_qwen2_local_cache_geometry(local_kv_heads, 2, family.qwen2_head_dim());

        let diagnostics = checkpoint_diagnostics(session.test_complete_model());
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
        let distributed_decode = session
            .decode(&backend, token.clone())
            .unwrap()
            .wait()
            .unwrap()
            .into_logits()
            .unwrap();
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
        let reference_decode = reference_decode
            .try_index_device((.., -1, ..), &stream)
            .unwrap();
        assert_arrays_close(&distributed_decode, &reference_decode, tolerance);
        session
            .test_complete_cache()
            .assert_qwen2_local_cache_geometry(local_kv_heads, 3, family.qwen2_head_dim());
        assert_eq!(session.test_complete_cache().offset(), 3);
        return;
    }

    assert!(family.persists_paged_prefix());
    let descriptor = PromptCacheDescriptor {
        model_family: family.model_type().into(),
        effective_model_type: family.model_type().into(),
        checkpoint_fingerprint: "tensor-ring-fixture".into(),
        prefix_content_fingerprint: "tokens:1,2".into(),
        architecture_fingerprint,
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        batch_size: 1,
        layer_prefix_offsets: vec![0; layer_count],
        layer_layout: prompt_cache_layer_layout,
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: None,
            tensor_parallel: Some((2, expected_rank)),
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        },
    };
    let saved = session
        .save_prompt_cache(
            &backend,
            &prompt_cache_root,
            descriptor.clone(),
            &[1, 2],
            &PromptCacheOptions::default(),
        )
        .unwrap();
    assert_eq!(saved.topology, descriptor.topology);
    let token = safemlx::Array::from_slice(&[0u32], &[1, 1]);
    let uninterrupted = session
        .decode(&backend, token.clone())
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap();
    let uninterrupted = uninterrupted.evaluated().unwrap();
    let uninterrupted_values = uninterrupted.as_slice::<f32>().to_vec();
    drop(uninterrupted);
    let manifest = session
        .load_prompt_cache(&backend, &prompt_cache_root, &descriptor, &[1, 2], paged)
        .unwrap();
    assert_eq!(manifest.topology, descriptor.topology);
    let restored = session
        .decode(&backend, token)
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap();
    let restored = restored.evaluated().unwrap();
    assert_eq!(uninterrupted_values, restored.as_slice::<f32>());
    let logits = restored.as_array().clone();
    drop(restored);
    let mut sampler = DefaultSampler;
    let mut prng = (expected_rank == 0).then(|| RandomState::from_key(random::key(7).unwrap()));
    let synchronized = session
        .sample_and_synchronize(Some(&logits), 1, &mut sampler, 1.0, prng.as_mut(), false)
        .unwrap();
    let sampled = synchronized.token.evaluated().unwrap();
    assert!(sampled.as_slice::<u32>()[0] < vocab_size as u32);
    drop(sampled);
    let logits = session
        .decode(&backend, synchronized.token)
        .unwrap()
        .wait()
        .unwrap()
        .into_logits()
        .unwrap();
    assert_eq!(logits.shape(), &[1, vocab_size as i32]);
    assert_eq!(session.test_complete_cache().offset(), 4);
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
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 17,
            "num_attention_heads": 6,
            "num_key_value_heads": 3,
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
        ("model.embed_tokens.weight", vec![5, 12], 0.01),
        ("model.layers.0.self_attn.q_proj.weight", vec![12, 12], 0.01),
        ("model.layers.0.self_attn.k_proj.weight", vec![6, 12], 0.01),
        ("model.layers.0.self_attn.v_proj.weight", vec![6, 12], 0.01),
        ("model.layers.0.self_attn.o_proj.weight", vec![12, 12], 0.01),
        ("model.layers.0.mlp.gate_proj.weight", vec![17, 12], 0.01),
        ("model.layers.0.mlp.up_proj.weight", vec![17, 12], 0.01),
        ("model.layers.0.mlp.down_proj.weight", vec![12, 17], 0.01),
        ("model.layers.0.input_layernorm.weight", vec![12], 1.0),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![12],
            1.0,
        ),
        ("model.norm.weight", vec![12], 1.0),
        ("lm_head.weight", vec![5, 12], 0.01),
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

fn write_lfm2_fixture(directory: &Path) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "lfm2",
            "architectures": ["Lfm2ForCausalLM"],
            "vocab_size": 13,
            "hidden_size": 12,
            "intermediate_size": 17,
            "num_hidden_layers": 2,
            "num_attention_heads": 6,
            "num_key_value_heads": 3,
            "max_position_embeddings": 32,
            "norm_eps": 0.00001,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": true,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false
        }))
        .unwrap(),
    )
    .unwrap();
    let tensors = [
        ("model.embed_tokens.weight", vec![13, 12], 0.011),
        ("model.embedding_norm.weight", vec![12], 1.0),
        ("model.layers.0.operator_norm.weight", vec![12], 1.0),
        ("model.layers.0.ffn_norm.weight", vec![12], 1.0),
        ("model.layers.0.conv.conv.weight", vec![12, 1, 3], 0.012),
        ("model.layers.0.conv.conv.bias", vec![12], 0.006),
        ("model.layers.0.conv.in_proj.weight", vec![36, 12], 0.013),
        ("model.layers.0.conv.in_proj.bias", vec![36], 0.007),
        ("model.layers.0.conv.out_proj.weight", vec![12, 12], 0.014),
        ("model.layers.0.conv.out_proj.bias", vec![12], 0.008),
        ("model.layers.0.feed_forward.w1.weight", vec![17, 12], 0.015),
        ("model.layers.0.feed_forward.w2.weight", vec![12, 17], 0.016),
        ("model.layers.0.feed_forward.w3.weight", vec![17, 12], 0.017),
        ("model.layers.1.operator_norm.weight", vec![12], 1.0),
        ("model.layers.1.ffn_norm.weight", vec![12], 1.0),
        (
            "model.layers.1.self_attn.q_proj.weight",
            vec![12, 12],
            0.018,
        ),
        ("model.layers.1.self_attn.k_proj.weight", vec![6, 12], 0.019),
        ("model.layers.1.self_attn.v_proj.weight", vec![6, 12], 0.020),
        (
            "model.layers.1.self_attn.out_proj.weight",
            vec![12, 12],
            0.021,
        ),
        ("model.layers.1.self_attn.q_layernorm.weight", vec![2], 1.0),
        ("model.layers.1.self_attn.k_layernorm.weight", vec![2], 1.0),
        ("model.layers.1.feed_forward.w1.weight", vec![17, 12], 0.022),
        ("model.layers.1.feed_forward.w2.weight", vec![12, 17], 0.023),
        ("model.layers.1.feed_forward.w3.weight", vec![17, 12], 0.024),
        ("lm_head.weight", vec![13, 12], 0.025),
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

fn lfm2_gguf_metadata(
    hidden_size: u32,
    query_heads: u32,
    kv_heads: u32,
    intermediate_size: u32,
    vocab_size: u32,
) -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("lfm2".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("lfm2.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "lfm2.embedding_length".into(),
            GgufMetadataValue::Uint32(hidden_size),
        ),
        (
            "lfm2.feed_forward_length".into(),
            GgufMetadataValue::Uint32(intermediate_size),
        ),
        (
            "lfm2.attention.head_count".into(),
            GgufMetadataValue::Uint32(query_heads),
        ),
        (
            "lfm2.attention.head_count_kv".into(),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, kv_heads])),
        ),
        (
            "lfm2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        ("lfm2.context_length".into(), GgufMetadataValue::Uint32(32)),
        (
            "lfm2.shortconv.l_cache".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "lfm2.vocab_size".into(),
            GgufMetadataValue::Uint32(vocab_size),
        ),
    ])
}

fn lfm2_f32_gguf_specs() -> Vec<QuantizedGgufTensor> {
    let tensor = |name: &str, dimensions: Vec<u64>, phase: usize| {
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.004, phase))
    };
    vec![
        tensor("token_embd.weight", vec![12, 13], 1),
        f32_gguf_tensor("token_embd_norm.weight", vec![12], vec![1.0; 12]),
        f32_gguf_tensor("blk.0.attn_norm.weight", vec![12], vec![1.0; 12]),
        f32_gguf_tensor("blk.0.ffn_norm.weight", vec![12], vec![1.0; 12]),
        tensor("blk.0.shortconv.conv.weight", vec![3, 12], 2),
        f32_gguf_tensor(
            "blk.0.shortconv.conv.bias",
            vec![12],
            patterned_values(12, 0.005, 16),
        ),
        tensor("blk.0.shortconv.in_proj.weight", vec![12, 36], 3),
        f32_gguf_tensor(
            "blk.0.shortconv.in_proj.bias",
            vec![36],
            patterned_values(36, 0.005, 17),
        ),
        tensor("blk.0.shortconv.out_proj.weight", vec![12, 12], 4),
        f32_gguf_tensor(
            "blk.0.shortconv.out_proj.bias",
            vec![12],
            patterned_values(12, 0.005, 18),
        ),
        tensor("blk.0.ffn_gate.weight", vec![12, 17], 5),
        tensor("blk.0.ffn_down.weight", vec![17, 12], 6),
        tensor("blk.0.ffn_up.weight", vec![12, 17], 7),
        f32_gguf_tensor("blk.1.attn_norm.weight", vec![12], vec![1.0; 12]),
        f32_gguf_tensor("blk.1.ffn_norm.weight", vec![12], vec![1.0; 12]),
        tensor("blk.1.attn_q.weight", vec![12, 12], 8),
        tensor("blk.1.attn_k.weight", vec![12, 6], 9),
        tensor("blk.1.attn_v.weight", vec![12, 6], 10),
        tensor("blk.1.attn_output.weight", vec![12, 12], 11),
        f32_gguf_tensor("blk.1.attn_q_norm.weight", vec![2], vec![1.0; 2]),
        f32_gguf_tensor("blk.1.attn_k_norm.weight", vec![2], vec![1.0; 2]),
        tensor("blk.1.ffn_gate.weight", vec![12, 17], 12),
        tensor("blk.1.ffn_down.weight", vec![17, 12], 13),
        tensor("blk.1.ffn_up.weight", vec![12, 17], 14),
        tensor("output.weight", vec![12, 13], 15),
    ]
}

fn lfm2_q8_0_gguf_specs() -> Vec<QuantizedGgufTensor> {
    vec![
        q8_0_tensor("token_embd.weight", vec![64, 64], 1),
        f32_gguf_tensor("token_embd_norm.weight", vec![64], vec![1.0; 64]),
        f32_gguf_tensor("blk.0.attn_norm.weight", vec![64], vec![1.0; 64]),
        f32_gguf_tensor("blk.0.ffn_norm.weight", vec![64], vec![1.0; 64]),
        f32_gguf_tensor(
            "blk.0.shortconv.conv.weight",
            vec![3, 64],
            patterned_values(192, 0.004, 2),
        ),
        f32_gguf_tensor(
            "blk.0.shortconv.conv.bias",
            vec![64],
            patterned_values(64, 0.005, 16),
        ),
        q8_0_tensor("blk.0.shortconv.in_proj.weight", vec![64, 192], 3),
        f32_gguf_tensor(
            "blk.0.shortconv.in_proj.bias",
            vec![192],
            patterned_values(192, 0.005, 17),
        ),
        q8_0_tensor("blk.0.shortconv.out_proj.weight", vec![64, 64], 4),
        f32_gguf_tensor(
            "blk.0.shortconv.out_proj.bias",
            vec![64],
            patterned_values(64, 0.005, 18),
        ),
        q8_0_tensor("blk.0.ffn_gate.weight", vec![64, 64], 5),
        q8_0_tensor("blk.0.ffn_down.weight", vec![64, 64], 6),
        q8_0_tensor("blk.0.ffn_up.weight", vec![64, 64], 7),
        f32_gguf_tensor("blk.1.attn_norm.weight", vec![64], vec![1.0; 64]),
        f32_gguf_tensor("blk.1.ffn_norm.weight", vec![64], vec![1.0; 64]),
        q8_0_tensor("blk.1.attn_q.weight", vec![64, 64], 8),
        q8_0_tensor("blk.1.attn_k.weight", vec![64, 32], 9),
        q8_0_tensor("blk.1.attn_v.weight", vec![64, 32], 10),
        q8_0_tensor("blk.1.attn_output.weight", vec![64, 64], 11),
        f32_gguf_tensor("blk.1.attn_q_norm.weight", vec![16], vec![1.0; 16]),
        f32_gguf_tensor("blk.1.attn_k_norm.weight", vec![16], vec![1.0; 16]),
        q8_0_tensor("blk.1.ffn_gate.weight", vec![64, 64], 12),
        q8_0_tensor("blk.1.ffn_down.weight", vec![64, 64], 13),
        q8_0_tensor("blk.1.ffn_up.weight", vec![64, 64], 14),
        q8_0_tensor("output.weight", vec![64, 64], 15),
    ]
}

fn lfm2_f32_gguf_payload_bytes() -> u64 {
    lfm2_f32_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn lfm2_q8_0_gguf_payload_bytes() -> u64 {
    lfm2_q8_0_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn write_lfm2_gguf_fixture(path: &Path, quantized: bool) {
    let (mut metadata, specs) = if quantized {
        (lfm2_gguf_metadata(64, 4, 2, 64, 64), lfm2_q8_0_gguf_specs())
    } else {
        (lfm2_gguf_metadata(12, 6, 3, 17, 13), lfm2_f32_gguf_specs())
    };
    if quantized {
        metadata.insert("general.file_type".into(), GgufMetadataValue::Uint32(7));
    }
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

fn write_qwen3_moe_fixture(directory: &Path) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false,
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "moe_intermediate_size": 9,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true
        }))
        .unwrap(),
    )
    .unwrap();
    let tensors = [
        ("model.embed_tokens.weight", vec![16, 8], 0.011),
        ("model.layers.0.self_attn.q_proj.weight", vec![8, 8], 0.012),
        ("model.layers.0.self_attn.k_proj.weight", vec![8, 8], 0.013),
        ("model.layers.0.self_attn.v_proj.weight", vec![8, 8], 0.014),
        ("model.layers.0.self_attn.o_proj.weight", vec![8, 8], 0.015),
        ("model.layers.0.self_attn.q_norm.weight", vec![4], 1.0),
        ("model.layers.0.self_attn.k_norm.weight", vec![4], 1.0),
        ("model.layers.0.mlp.gate.weight", vec![4, 8], 0.016),
        (
            "model.layers.0.mlp.experts.gate_up_proj",
            vec![4, 18, 8],
            0.017,
        ),
        ("model.layers.0.mlp.experts.down_proj", vec![4, 8, 9], 0.018),
        ("model.layers.0.input_layernorm.weight", vec![8], 1.0),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![8],
            1.0,
        ),
        ("model.norm.weight", vec![8], 1.0),
        ("lm_head.weight", vec![16, 8], 0.019),
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

fn write_gpt_oss_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 1,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 32,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "max_position_embeddings": 64,
        "rope_theta": 150000.0,
        "layer_types": ["full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = gpt_oss_model::model_args_from_config_value(&config).unwrap();
    let mut model = gpt_oss_model::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("_scales") {
            Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
        } else if name.ends_with("_blocks") {
            Array::full::<u8>(&shape, Array::from_slice(&[0x11u8], &[]), stream).unwrap()
        } else if name.ends_with("layernorm.weight") || name.as_ref() == "model.norm.weight" {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 17;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                stream,
            )
            .unwrap()
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

fn write_kimi_linear_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "kimi_linear",
        "vocab_size": 13,
        "hidden_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "intermediate_size": 17,
        "head_dim": 4,
        "model_max_length": 64,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 3,
            "head_dim": 4,
            "short_conv_kernel_size": 2
        },
        "num_experts": 4,
        "moe_intermediate_size": 9,
        "kv_lora_rank": 4,
        "q_lora_rank": null,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": true,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": true,
        "num_expert_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 0
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = kimi_model::model_args_from_config_value(&config).unwrap();
    let mut model = kimi_model::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("layernorm.weight")
            || name.ends_with("o_norm.weight")
            || name.as_ref() == "model.norm.weight"
        {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with("A_log") {
            Array::full::<f32>(&shape, Array::from_f32(-0.2), stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 23;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                stream,
            )
            .unwrap()
        };
    }
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
        if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
            for expert in 0..model.args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"),
                    value
                        .try_index_device((expert, ..model.args.moe_intermediate_size, ..), stream)
                        .unwrap(),
                ));
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"),
                    value
                        .try_index_device((expert, model.args.moe_intermediate_size.., ..), stream)
                        .unwrap(),
                ));
            }
            continue;
        }
        if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
            for expert in 0..model.args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w2.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
            continue;
        }
        let checkpoint_name = if name.starts_with("model.layers.1.mlp.") {
            name.replacen("model.layers.1.mlp.", "model.layers.1.block_sparse_moe.", 1)
        } else {
            name.to_string()
        };
        let value = if checkpoint_name.ends_with("_conv1d.weight") {
            value
                .reshape(
                    &[
                        model.args.kda_config.num_heads * model.args.kda_config.head_dim,
                        model.args.kda_config.short_conv_kernel_size,
                    ],
                    stream,
                )
                .unwrap()
        } else {
            value.clone()
        };
        arrays.push((checkpoint_name, value));
    }
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

fn nemotron_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": 13,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "M-E*",
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 2,
        "max_position_embeddings": 64,
        "sliding_window": 3,
        "layer_norm_epsilon": 0.00001,
        "norm_eps": 0.00001,
        "mamba_num_heads": 6,
        "mamba_head_dim": 2,
        "n_groups": 3,
        "ssm_state_size": 2,
        "conv_kernel": 3,
        "chunk_size": 2,
        "moe_intermediate_size": 5,
        "moe_shared_expert_intermediate_size": 7,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "torch_dtype": "float32"
    })
}

fn nemotron_public_name(runtime: &str, args: &nemotron_model::ModelArgs) -> String {
    if let Some(rest) = runtime.strip_prefix("model.embeddings.") {
        return format!("backbone.embeddings.{rest}");
    }
    if let Some(rest) = runtime.strip_prefix("model.norm_f.") {
        return format!("backbone.norm_f.{rest}");
    }
    for index in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{index}.");
        let Some(rest) = runtime.strip_prefix(&prefix) else {
            continue;
        };
        if rest.starts_with("norm.") {
            return format!("backbone.layers.{index}.{rest}");
        }
        let field = match args.layer_schedule.get(index).unwrap() {
            nemotron_model::LayerPolicy::Mamba => "mamba",
            nemotron_model::LayerPolicy::SelfAttention(_) => "attention",
            nemotron_model::LayerPolicy::DenseMlp => "mlp",
            nemotron_model::LayerPolicy::SparseMoe => "moe",
        };
        let rest = rest.strip_prefix(&format!("{field}.")).unwrap_or(rest);
        return format!("backbone.layers.{index}.mixer.{rest}");
    }
    runtime.to_string()
}

fn write_nemotron_fixture(directory: &Path) {
    let config = nemotron_config();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = nemotron_model::model_args_from_config_value(&config).unwrap();
    let mut model = nemotron_model::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") || name.as_ref() == "model.norm_f.weight" {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with("A_log") {
            Array::full::<f32>(&shape, Array::from_f32(-0.2), stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 29;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                stream,
            )
            .unwrap()
        };
    }
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in model.parameters().flatten() {
        let runtime = canonical_checkpoint_name(&name);
        if runtime.ends_with("moe.experts.up_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".up_proj"), &model.args);
            for expert in 0..model.args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.up_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else if runtime.ends_with("moe.experts.down_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".down_proj"), &model.args);
            for expert in 0..model.args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.down_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else {
            arrays.push((nemotron_public_name(&runtime, &model.args), value.clone()));
        }
    }
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

fn kimi_linear_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("kimi-linear".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            "kimi-linear.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "kimi-linear.attention.head_count".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "kimi-linear.attention.head_count_kv".into(),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 1])),
        ),
        (
            "kimi-linear.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.vocab_size".into(),
            GgufMetadataValue::Uint32(13),
        ),
        (
            "kimi-linear.feed_forward_length".into(),
            GgufMetadataValue::Uint32(17),
        ),
        (
            "kimi-linear.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "kimi-linear.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "kimi-linear.kda.head_dim".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.ssm.conv_kernel".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(9),
        ),
        (
            "kimi-linear.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "kimi-linear.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "kimi-linear.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "kimi-linear.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
    ])
}

fn kimi_linear_gguf_specs() -> Vec<QuantizedGgufTensor> {
    let tensor = |name: &str, mlx_shape: &[u64], phase: usize| {
        let mut dimensions = mlx_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut specs = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        specs.push(norm(&format!("blk.{layer}.attn_norm.weight"), 12));
        specs.push(norm(&format!("blk.{layer}.ffn_norm.weight"), 12));
    }
    specs.extend([
        tensor("blk.0.attn_q.weight", &[12, 12], 3),
        tensor("blk.0.attn_k.weight", &[12, 12], 4),
        tensor("blk.0.attn_v.weight", &[12, 12], 5),
        tensor("blk.0.ssm_conv1d_q.weight", &[12, 2], 6),
        tensor("blk.0.ssm_conv1d_k.weight", &[12, 2], 7),
        tensor("blk.0.ssm_conv1d_v.weight", &[12, 2], 8),
        tensor("blk.0.ssm_f_a.weight", &[4, 12], 9),
        tensor("blk.0.ssm_f_b.weight", &[12, 4], 10),
        tensor("blk.0.ssm_beta.weight", &[3, 12], 11),
        tensor("blk.0.ssm_g_a.weight", &[4, 12], 12),
        tensor("blk.0.ssm_g_b.weight", &[12, 4], 13),
        f32_gguf_tensor("blk.0.ssm_a", vec![3], vec![-0.7, -0.9, -1.1]),
        f32_gguf_tensor(
            "blk.0.ssm_dt.bias",
            vec![12],
            patterned_values(12, 0.002, 14),
        ),
        norm("blk.0.ssm_norm.weight", 4),
        tensor("blk.0.attn_output.weight", &[12, 12], 15),
        tensor("blk.0.ffn_gate.weight", &[17, 12], 16),
        tensor("blk.0.ffn_up.weight", &[17, 12], 17),
        tensor("blk.0.ffn_down.weight", &[12, 17], 18),
        tensor("blk.1.attn_q.weight", &[12, 12], 19),
        tensor("blk.1.attn_kv_a_mqa.weight", &[6, 12], 20),
        norm("blk.1.attn_kv_a_norm.weight", 4),
        tensor("blk.1.attn_kv_b.weight", &[12, 4], 21),
        tensor("blk.1.attn_output.weight", &[12, 6], 22),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 23),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 24),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[9, 12], 25),
        tensor("blk.1.ffn_up_shexp.weight", &[9, 12], 26),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 9], 27),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 9, 12], 28),
        tensor("blk.1.ffn_up_exps.weight", &[4, 9, 12], 29),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 9], 30),
    ]);
    specs
}

fn kimi_linear_gguf_payload_bytes() -> u64 {
    kimi_linear_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn write_kimi_linear_gguf_fixture(path: &Path) {
    let specs = kimi_linear_gguf_specs();
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
        .write(
            std::fs::File::create(path).unwrap(),
            &kimi_linear_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

fn mxfp4_payload(elements: u64, phase: usize) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let mut data = Vec::with_capacity((elements / 32) as usize * 17);
    for block in 0..elements / 32 {
        data.push(127 + ((block as usize + phase) % 3) as u8);
        data.extend((0..16).map(|index| {
            let low = ((index + phase) % 7 + 1) as u8;
            let high = ((index * 3 + phase) % 7 + 1) as u8;
            low | (high << 4)
        }));
    }
    data
}

fn nemotron_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    let architecture = "nemotron_h_moe";
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(3)),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("embedding_length"), GgufMetadataValue::Uint32(12)),
        (key("vocab_size"), GgufMetadataValue::Uint32(13)),
        (
            key("feed_forward_length"),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 17, 0])),
        ),
        (key("attention.head_count"), GgufMetadataValue::Uint32(6)),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Uint32(vec![0, 0, 3])),
        ),
        (key("attention.key_length"), GgufMetadataValue::Uint32(2)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            key("attention.sliding_window"),
            GgufMetadataValue::Uint32(3),
        ),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10_000.0)),
        (key("rope.dimension_count"), GgufMetadataValue::Uint32(2)),
        (key("ssm.inner_size"), GgufMetadataValue::Uint32(12)),
        (key("ssm.time_step_rank"), GgufMetadataValue::Uint32(6)),
        (key("ssm.group_count"), GgufMetadataValue::Uint32(3)),
        (key("ssm.state_size"), GgufMetadataValue::Uint32(2)),
        (key("ssm.conv_kernel"), GgufMetadataValue::Uint32(3)),
        (key("expert_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_shared_count"), GgufMetadataValue::Uint32(1)),
        (
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(5),
        ),
        (
            key("expert_shared_feed_forward_length"),
            GgufMetadataValue::Uint32(7),
        ),
        (key("expert_used_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_group_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_group_used_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_weights_norm"), GgufMetadataValue::Uint32(1)),
        (key("expert_weights_scale"), GgufMetadataValue::Float32(1.0)),
    ])
}

fn nemotron_gguf_specs() -> Vec<QuantizedGgufTensor> {
    let tensor = |name: &str, mlx_shape: &[u64], phase: usize| {
        let mut dimensions = mlx_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.002, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
        norm("blk.0.attn_norm.weight", 12),
        tensor("blk.0.ssm_in.weight", &[42, 12], 3),
        tensor("blk.0.ssm_conv1d.weight", &[24, 3], 4),
        f32_gguf_tensor("blk.0.ssm_dt.bias", vec![6], patterned_values(6, 0.002, 5)),
        f32_gguf_tensor(
            "blk.0.ssm_a",
            vec![6],
            vec![-0.7, -0.8, -0.9, -1.0, -1.1, -1.2],
        ),
        f32_gguf_tensor("blk.0.ssm_d", vec![6], vec![0.5; 6]),
        norm("blk.0.ssm_norm.weight", 12),
        tensor("blk.0.ssm_out.weight", &[12, 12], 6),
        norm("blk.1.attn_norm.weight", 12),
        tensor("blk.1.ffn_gate_inp.weight", &[2, 12], 7),
        f32_gguf_tensor("blk.1.exp_probs_b.bias", vec![2], vec![0.01, -0.01]),
        tensor("blk.1.ffn_up_shexp.weight", &[7, 12], 8),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 7], 9),
        tensor("blk.1.ffn_up_exps.weight", &[2, 5, 12], 10),
        tensor("blk.1.ffn_down_exps.weight", &[2, 12, 5], 11),
        norm("blk.2.attn_norm.weight", 12),
        tensor("blk.2.attn_q.weight", &[12, 12], 12),
        tensor("blk.2.attn_k.weight", &[6, 12], 13),
        tensor("blk.2.attn_v.weight", &[6, 12], 14),
        tensor("blk.2.attn_output.weight", &[12, 12], 15),
    ]
}

fn nemotron_gguf_payload_bytes() -> u64 {
    nemotron_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn write_nemotron_gguf_fixture(path: &Path) {
    let specs = nemotron_gguf_specs();
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
        .write(
            std::fs::File::create(path).unwrap(),
            &nemotron_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

fn write_gpt_oss_gguf_fixture(path: &Path) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("gpt-oss".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(39)),
        (
            "gpt-oss.embedding_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        ("gpt-oss.block_count".into(), GgufMetadataValue::Uint32(1)),
        (
            "gpt-oss.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(96),
        ),
        (
            "gpt-oss.attention.head_count".into(),
            GgufMetadataValue::Uint32(6),
        ),
        (
            "gpt-oss.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "gpt-oss.attention.key_length".into(),
            GgufMetadataValue::Uint32(32),
        ),
        (
            "gpt-oss.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "gpt-oss.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "gpt-oss.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "gpt-oss.rope.freq_base".into(),
            GgufMetadataValue::Float32(150000.0),
        ),
        ("gpt-oss.expert_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gpt-oss.expert_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        ("gpt-oss.vocab_size".into(), GgufMetadataValue::Uint32(64)),
    ]);
    let f32_tensor = |name: &str, dimensions: Vec<u64>, phase: usize| {
        let values = patterned_values(
            usize::try_from(dimensions.iter().product::<u64>()).unwrap(),
            0.003,
            phase,
        );
        QuantizedGgufTensor {
            name: name.into(),
            dimensions,
            ggml_type: GgmlType::F32,
            data: values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        }
    };
    let mxfp4_tensor = |name: &str, dimensions: Vec<u64>, phase: usize| {
        let elements = dimensions.iter().product();
        QuantizedGgufTensor {
            name: name.into(),
            dimensions,
            ggml_type: GgmlType::MxFp4,
            data: mxfp4_payload(elements, phase),
        }
    };
    let tensors = vec![
        f32_tensor("token_embd.weight", vec![64, 64], 0),
        f32_tensor("blk.0.attn_norm.weight", vec![64], 1),
        f32_tensor("blk.0.attn_post_norm.weight", vec![64], 2),
        f32_tensor("blk.0.attn_q.weight", vec![64, 192], 3),
        f32_tensor("blk.0.attn_q.bias", vec![192], 4),
        f32_tensor("blk.0.attn_k.weight", vec![64, 96], 5),
        f32_tensor("blk.0.attn_k.bias", vec![96], 6),
        f32_tensor("blk.0.attn_v.weight", vec![64, 96], 7),
        f32_tensor("blk.0.attn_v.bias", vec![96], 8),
        f32_tensor("blk.0.attn_output.weight", vec![192, 64], 9),
        f32_tensor("blk.0.attn_output.bias", vec![64], 10),
        f32_tensor("blk.0.attn_sinks.weight", vec![6], 11),
        f32_tensor("blk.0.ffn_gate_inp.weight", vec![64, 2], 12),
        f32_tensor("blk.0.ffn_gate_inp.bias", vec![2], 13),
        mxfp4_tensor("blk.0.ffn_gate_exps.weight", vec![64, 96, 2], 14),
        f32_tensor("blk.0.ffn_gate_exps.bias", vec![96, 2], 15),
        mxfp4_tensor("blk.0.ffn_up_exps.weight", vec![64, 96, 2], 16),
        f32_tensor("blk.0.ffn_up_exps.bias", vec![96, 2], 17),
        mxfp4_tensor("blk.0.ffn_down_exps.weight", vec![96, 64, 2], 18),
        f32_tensor("blk.0.ffn_down_exps.bias", vec![64, 2], 19),
        f32_tensor("output_norm.weight", vec![64], 20),
        f32_tensor("output.weight", vec![64, 64], 21),
    ];
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: tensor.ggml_type,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
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
            GgufMetadataValue::Uint32(12),
        ),
        (
            "llama.attention.head_count".into(),
            GgufMetadataValue::Uint32(6),
        ),
        (
            "llama.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "llama.attention.key_length".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "llama.feed_forward_length".into(),
            GgufMetadataValue::Uint32(17),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        ("llama.context_length".into(), GgufMetadataValue::Uint32(32)),
        ("llama.vocab_size".into(), GgufMetadataValue::Uint32(5)),
    ]);
    let specs = [
        ("token_embd.weight", vec![12, 5], 0.01f32),
        ("blk.0.attn_q.weight", vec![12, 12], 0.01),
        ("blk.0.attn_k.weight", vec![12, 6], 0.01),
        ("blk.0.attn_v.weight", vec![12, 6], 0.01),
        ("blk.0.attn_output.weight", vec![12, 12], 0.01),
        ("blk.0.ffn_gate.weight", vec![12, 17], 0.01),
        ("blk.0.ffn_up.weight", vec![12, 17], 0.01),
        ("blk.0.ffn_down.weight", vec![17, 12], 0.01),
        ("blk.0.attn_norm.weight", vec![12], 1.0),
        ("blk.0.ffn_norm.weight", vec![12], 1.0),
        ("output_norm.weight", vec![12], 1.0),
        ("output.weight", vec![12, 5], 0.01),
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
        vec![12, 8],
        patterned_values(96, 0.015, 1),
    )];
    for layer in 0..2 {
        let phase = layer * 10;
        specs.extend([
            (
                format!("blk.{layer}.attn_q.weight"),
                vec![12, 12],
                patterned_values(144, 0.012, phase + 2),
            ),
            (
                format!("blk.{layer}.attn_q.bias"),
                vec![12],
                if with_biases {
                    patterned_values(12, 0.035, phase + 3)
                } else {
                    vec![0.0; 12]
                },
            ),
            (
                format!("blk.{layer}.attn_k.weight"),
                vec![12, 6],
                patterned_values(72, 0.014, phase + 4),
            ),
            (
                format!("blk.{layer}.attn_k.bias"),
                vec![6],
                if with_biases {
                    patterned_values(6, 0.04, phase + 5)
                } else {
                    vec![0.0; 6]
                },
            ),
            (
                format!("blk.{layer}.attn_v.weight"),
                vec![12, 6],
                patterned_values(72, 0.013, phase + 6),
            ),
            (
                format!("blk.{layer}.attn_v.bias"),
                vec![6],
                if with_biases {
                    patterned_values(6, 0.045, phase + 7)
                } else {
                    vec![0.0; 6]
                },
            ),
            (
                format!("blk.{layer}.attn_output.weight"),
                vec![12, 12],
                patterned_values(144, 0.011, phase + 8),
            ),
            (
                format!("blk.{layer}.ffn_gate.weight"),
                vec![12, 17],
                patterned_values(204, 0.009, phase + 9),
            ),
            (
                format!("blk.{layer}.ffn_up.weight"),
                vec![12, 17],
                patterned_values(204, 0.008, phase + 10),
            ),
            (
                format!("blk.{layer}.ffn_down.weight"),
                vec![17, 12],
                patterned_values(204, 0.01, phase + 11),
            ),
            (
                format!("blk.{layer}.attn_norm.weight"),
                vec![12],
                vec![1.0; 12],
            ),
            (
                format!("blk.{layer}.ffn_norm.weight"),
                vec![12],
                vec![1.0; 12],
            ),
        ]);
    }
    specs.extend([
        ("output_norm.weight".into(), vec![12], vec![1.0; 12]),
        (
            "output.weight".into(),
            vec![12, 8],
            patterned_values(96, 0.017, 23),
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
    kv_heads: u32,
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
            GgufMetadataValue::Uint32(hidden_size / head_dim),
        ),
        (
            "qwen2.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(kv_heads),
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
    let metadata = qwen2_gguf_metadata(12, 2, 3, 17, 8);
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
    let mut metadata = qwen2_gguf_metadata(64, 16, 2, 64, 64);
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
        "hidden_size": 12,
        "intermediate_size": 17,
        "moe_intermediate_size": 5,
        "num_hidden_layers": layers,
        "num_attention_heads": 3,
        "vocab_size": 13,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
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
    let mut arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
        .collect::<Vec<_>>();
    for expert in 0..4 {
        for (projection, shape) in [
            ("gate_proj", vec![5, 12]),
            ("up_proj", vec![5, 12]),
            ("down_proj", vec![12, 5]),
        ] {
            arrays.push((
                format!("model.layers.1.mlp.experts.{expert}.{projection}.weight"),
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap(),
            ));
        }
    }
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

fn deepseek_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("deepseek2".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("deepseek2.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "deepseek2.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "deepseek2.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "deepseek2.feed_forward_length".into(),
            GgufMetadataValue::Uint32(17),
        ),
        (
            "deepseek2.attention.head_count".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "deepseek2.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.000001),
        ),
        (
            "deepseek2.rope.freq_base".into(),
            GgufMetadataValue::Float32(10_000.0),
        ),
        (
            "deepseek2.rope.dimension_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.attention.q_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.kv_lora_rank".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.key_length_mla".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.attention.value_length_mla".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.leading_dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "deepseek2.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(5),
        ),
        (
            "deepseek2.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_group_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "deepseek2.expert_gating_func".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "deepseek2.expert_weights_norm".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "deepseek2.expert_weights_scale".into(),
            GgufMetadataValue::Float32(1.0),
        ),
        ("deepseek2.vocab_size".into(), GgufMetadataValue::Uint32(13)),
    ])
}

fn deepseek_gguf_specs() -> Vec<QuantizedGgufTensor> {
    let tensor = |name: &str, mlx_shape: &[u64], phase: usize| {
        let mut dimensions = mlx_shape.to_vec();
        dimensions.reverse();
        let elements = dimensions.iter().product::<u64>() as usize;
        f32_gguf_tensor(name, dimensions, patterned_values(elements, 0.003, phase))
    };
    let norm =
        |name: &str, width: u64| f32_gguf_tensor(name, vec![width], vec![1.0; width as usize]);
    let mut specs = vec![
        tensor("token_embd.weight", &[13, 12], 1),
        norm("output_norm.weight", 12),
        tensor("output.weight", &[13, 12], 2),
    ];
    for layer in 0..2 {
        let phase = 3 + layer * 9;
        specs.extend([
            norm(&format!("blk.{layer}.attn_norm.weight"), 12),
            norm(&format!("blk.{layer}.ffn_norm.weight"), 12),
            tensor(&format!("blk.{layer}.attn_q_a.weight"), &[4, 12], phase),
            norm(&format!("blk.{layer}.attn_q_a_norm.weight"), 4),
            tensor(&format!("blk.{layer}.attn_q_b.weight"), &[12, 4], phase + 1),
            tensor(
                &format!("blk.{layer}.attn_kv_a_mqa.weight"),
                &[6, 12],
                phase + 2,
            ),
            norm(&format!("blk.{layer}.attn_kv_a_norm.weight"), 4),
            tensor(
                &format!("blk.{layer}.attn_k_b.weight"),
                &[3, 4, 2],
                phase + 3,
            ),
            tensor(
                &format!("blk.{layer}.attn_v_b.weight"),
                &[3, 2, 4],
                phase + 4,
            ),
            tensor(
                &format!("blk.{layer}.attn_output.weight"),
                &[12, 6],
                phase + 5,
            ),
        ]);
    }
    specs.extend([
        tensor("blk.0.ffn_gate.weight", &[17, 12], 21),
        tensor("blk.0.ffn_up.weight", &[17, 12], 22),
        tensor("blk.0.ffn_down.weight", &[12, 17], 23),
        tensor("blk.1.ffn_gate_inp.weight", &[4, 12], 24),
        f32_gguf_tensor(
            "blk.1.exp_probs_b.bias",
            vec![4],
            patterned_values(4, 0.001, 25),
        ),
        tensor("blk.1.ffn_gate_shexp.weight", &[5, 12], 26),
        tensor("blk.1.ffn_up_shexp.weight", &[5, 12], 27),
        tensor("blk.1.ffn_down_shexp.weight", &[12, 5], 28),
        tensor("blk.1.ffn_gate_exps.weight", &[4, 5, 12], 29),
        tensor("blk.1.ffn_up_exps.weight", &[4, 5, 12], 30),
        tensor("blk.1.ffn_down_exps.weight", &[4, 12, 5], 31),
    ]);
    specs
}

fn deepseek_gguf_payload_bytes() -> u64 {
    deepseek_gguf_specs()
        .iter()
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

fn write_deepseek_gguf_fixture(path: &Path) {
    let specs = deepseek_gguf_specs();
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
        .write(
            std::fs::File::create(path).unwrap(),
            &deepseek_gguf_metadata(),
            &tensors,
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

/// Verifies uneven Llama geometry through fully resident rank-local materialization.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_llama_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::LlamaSafetensors,
        TensorParameterResidency::FullyResident,
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

/// Verifies uneven Qwen3 MoE expert intermediates against resident execution.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_qwen3_moe_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Qwen3MoeSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies uneven GPT-OSS GQA/sink placement and block-aligned MXFP4 experts.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::GptOssSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies GPT-OSS rank-local weights remain resident across forwards.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::GptOssSafetensors,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies GPT-OSS uses the same uneven plan through dense disk streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_dense_stream_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::GptOssSafetensors,
        TensorParameterResidency::DenseDiskStream,
    );
}

/// Verifies rank-selective native MXFP4 GPT-OSS GGUF loading and execution.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_gpt_oss_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::GptOssGguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies hybrid convolution/KV state and uneven GQA/SwiGLU partitions.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Lfm2Safetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies uneven KDA/MLA heads, dense/shared/routed intermediates, and
/// heterogeneous prompt state across two tensor ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::KimiLinearSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies the Kimi topology with all rank-local weights pinned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::KimiLinearSafetensors,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies rank-local Kimi materialization through dense disk streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_dense_stream_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::KimiLinearSafetensors,
        TensorParameterResidency::DenseDiskStream,
    );
}

/// Verifies Kimi's GGUF name translation, rank-selective reads, mixed KDA/MLA
/// execution, and heterogeneous live/persisted cache state across two ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_kimi_linear_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::KimiLinearGguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies uneven Nemotron Mamba, GQA, dense, and MoE domains together with
/// rank-aware heterogeneous live paging and prompt-cache persistence.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::NemotronSafetensors,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies the Nemotron topology with all rank-local weights pinned
/// while its attention cache remains live-paged.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::NemotronSafetensors,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies rank-local Nemotron materialization through dense disk streaming
/// with the same heterogeneous live-paged cache.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_dense_stream_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::NemotronSafetensors,
        TensorParameterResidency::DenseDiskStream,
    );
}

/// Verifies Nemotron-H-MoE GGUF name translation, bounded rank-local reads,
/// recurrent/live-paged KV cache geometry, persistence, and numerical parity.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_nemotron_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::NemotronGguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies the same hybrid LFM2 topology with all rank-local weights pinned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Lfm2Safetensors,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies rank-local LFM2 reads through the dense disk-stream policy.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_dense_stream_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Lfm2Safetensors,
        TensorParameterResidency::DenseDiskStream,
    );
}

/// Verifies dense LFM2 GGUF parity, local hybrid state, and bounded payload reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Lfm2Gguf,
        TensorParameterResidency::LayerwiseHost,
    );
}

/// Verifies block-aligned Q8_0 LFM2 projection and convolution-channel ranges.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_lfm2_q8_0_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::Lfm2Q8Gguf,
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

/// Verifies the planner-derived uneven DeepSeek layout with rank-local weights pinned.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_fully_resident_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::DeepSeekSafetensors,
        TensorParameterResidency::FullyResident,
    );
}

/// Verifies uneven DeepSeek MLA and expert domains through dense disk streaming.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_dense_stream_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::DeepSeekSafetensors,
        TensorParameterResidency::DenseDiskStream,
    );
}

/// Verifies packed DeepSeek2 GGUF experts and MLA heads use bounded rank-local reads.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_gguf_tensor_parallel() {
    run_ring_tensor_parallel(
        FixtureFamily::DeepSeekGguf,
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
            write_deepseek_fixture(checkpoint.path(), 2);
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::DeepSeekGguf => {
            let path = checkpoint.path().join("model.gguf");
            write_deepseek_gguf_fixture(&path);
            path
        }
        FixtureFamily::Qwen3MoeSafetensors => {
            write_qwen3_moe_fixture(checkpoint.path());
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
        FixtureFamily::Lfm2Safetensors => {
            write_lfm2_fixture(checkpoint.path());
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::Lfm2Gguf | FixtureFamily::Lfm2Q8Gguf => {
            let path = checkpoint.path().join("model.gguf");
            write_lfm2_gguf_fixture(&path, family == FixtureFamily::Lfm2Q8Gguf);
            path
        }
        FixtureFamily::GptOssSafetensors => {
            write_gpt_oss_fixture(checkpoint.path());
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::GptOssGguf => {
            let path = checkpoint.path().join("model.gguf");
            write_gpt_oss_gguf_fixture(&path);
            path
        }
        FixtureFamily::KimiLinearSafetensors => {
            write_kimi_linear_fixture(checkpoint.path());
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::KimiLinearGguf => {
            let path = checkpoint.path().join("model.gguf");
            write_kimi_linear_gguf_fixture(&path);
            path
        }
        FixtureFamily::NemotronSafetensors => {
            write_nemotron_fixture(checkpoint.path());
            checkpoint.path().to_path_buf()
        }
        FixtureFamily::NemotronGguf => {
            let path = checkpoint.path().join("model.gguf");
            write_nemotron_gguf_fixture(&path);
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
                .args([
                    "--exact",
                    "distributed_tensor_parallel_ring::tensor_ring_worker",
                    "--nocapture",
                ])
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
