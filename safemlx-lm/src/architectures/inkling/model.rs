//! Thinking Machines Lab Inkling multimodal model support.
//!
//! The released checkpoint is a multimodal conditional-generation model.  This
//! module owns the decoder, the native dMel audio and hMLP vision towers, the
//! embedded multi-token predictor, and the native safetensors loader.

#![allow(missing_docs)]

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        arange, argpartition_axis, broadcast_to, clip, concatenate_axis,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        matmul, r#where, sigmoid, softmax_axis, sum_axis, GgufCheckpoint, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    api::{
        common::{
            self,
            convolution::{causal_depthwise_conv1d, CausalConv1dCache, DepthwiseConv1d},
            generation::CausalLm,
            layers::SwiGluMlp,
            moe::PackedSwiGluExperts,
        },
        input,
    },
    architectures::qwen::dense::{gguf_i32_catalog, gguf_string},
    core::cache::{
        CacheRankIdentity, LayerCachePolicy, StateTensorDimension, StateTensorDtype,
        StateTensorOwner, StateTensorPolicy, StateTensorRole,
    },
    error::Error,
    runtime::attention::{AttentionPolicy, LayerSchedule},
    runtime::cache::residency::{
        derive_prompt_cache_architecture_fingerprint, open_prompt_cache_snapshot,
        save_prompt_cache_snapshot, CacheBlockArrays, CacheResidencyManager, CacheResidencyReport,
        PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheSnapshotBlock, PromptCacheStateArray,
    },
    runtime::cache::{
        BlockwiseAttentionAccumulator, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    runtime::checkpoint::load::{
        for_each_safetensor_array, gguf_metadata, gguf_quantization_configs, load_array_strict,
        load_named_array_strict, safetensors_files, StrictLoadConfig, StrictLoadReport,
    },
    runtime::checkpoint::quantization::WeightQuantization,
};

fn default_model_type() -> String {
    "inkling_mm_model".into()
}

fn default_true() -> bool {
    true
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_head_dim() -> i32 {
    128
}

fn default_sconv_kernel_size() -> i32 {
    4
}

fn default_rel_extent() -> i32 {
    1024
}

fn default_sliding_window() -> i64 {
    512
}

fn default_route_scale() -> f32 {
    1.0
}

fn default_logit_scale() -> f32 {
    1.0
}

fn default_image_token_id() -> u32 {
    200_054
}

fn default_audio_token_id() -> u32 {
    200_053
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Feed-forward implementation selected for one Inkling decoder layer.
pub enum FeedForwardPolicy {
    /// Dense SwiGLU feed-forward block.
    Dense,
    /// Routed and shared sparse-MoE feed-forward block.
    SparseMoe,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
/// Complete execution policy for one Inkling decoder layer.
pub struct LayerPolicy {
    /// Full or exact-window sliding self-attention.
    pub attention: AttentionPolicy,
    /// Dense or sparse-MoE feed-forward topology.
    pub feed_forward: FeedForwardPolicy,
}

/// Rank-local decoder geometry derived from the canonical parallel plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct ParallelLayerGeometry {
    pub(crate) query_heads: i32,
    pub(crate) kv_heads: i32,
    pub(crate) feed_forward: ParallelFeedForwardGeometry,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ParallelFeedForwardGeometry {
    Dense {
        intermediate: i32,
    },
    SparseMoe {
        routed_intermediate: i32,
        shared_intermediate: i32,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct TextArgsSource {
    /// Dense checkpoint dtype used by the released safetensors model.
    #[serde(default)]
    torch_dtype: Option<String>,
    hidden_size: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    head_dim: i32,
    #[serde(default)]
    swa_num_attention_heads: Option<i32>,
    #[serde(default)]
    swa_num_key_value_heads: Option<i32>,
    #[serde(default)]
    swa_head_dim: Option<i32>,
    #[serde(default = "default_sliding_window", alias = "sliding_window")]
    sliding_window_size: i64,
    #[serde(default)]
    local_layer_ids: Option<Vec<i64>>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    dense_mlp_idx: Option<i64>,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(default = "default_sconv_kernel_size", alias = "conv_kernel_size")]
    sconv_kernel_size: i32,
    #[serde(default = "default_true")]
    use_sconv: bool,
    #[serde(default = "default_rel_extent")]
    rel_extent: i32,
    d_rel: i32,
    #[serde(default)]
    log_scaling_n_floor: Option<i32>,
    #[serde(default)]
    log_scaling_alpha: f32,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_true")]
    use_embed_norm: bool,
    #[serde(default)]
    unpadded_vocab_size: Option<i32>,
    #[serde(default = "default_logit_scale")]
    logits_mup_width_multiplier: f32,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    intermediate_size: i32,
    #[serde(default)]
    dense_intermediate_size: Option<i32>,
    #[serde(default)]
    moe_intermediate_size: Option<i32>,
    #[serde(default, alias = "num_experts")]
    n_routed_experts: i32,
    #[serde(default)]
    num_experts_per_tok: i32,
    #[serde(default)]
    n_shared_experts: i32,
    #[serde(default = "default_route_scale")]
    route_scale: f32,
    #[serde(default = "default_true")]
    shared_expert_sink: bool,
    #[serde(default = "default_true")]
    use_gate_bias: bool,
    #[serde(default = "default_true")]
    norm_after_topk: bool,
    #[serde(default = "default_true")]
    use_global_scale: bool,
    #[serde(default = "default_gate_activation")]
    gate_activation: String,
    #[serde(default = "default_hidden_activation")]
    hidden_act: String,
    #[serde(default)]
    attention_dropout: f32,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
    #[serde(default)]
    model_max_length: Option<i32>,
}

#[derive(Debug, Clone)]
/// Validated Inkling text-decoder geometry normalized from checkpoint metadata.
pub struct TextArgs {
    pub torch_dtype: Option<String>,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub swa_num_attention_heads: Option<i32>,
    pub swa_num_key_value_heads: Option<i32>,
    pub swa_head_dim: Option<i32>,
    /// Authoritative attention and feed-forward policy in decoder-layer order.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    pub sconv_kernel_size: i32,
    pub use_sconv: bool,
    pub rel_extent: i32,
    pub d_rel: i32,
    pub log_scaling_n_floor: Option<i32>,
    pub log_scaling_alpha: f32,
    pub rms_norm_eps: f32,
    pub use_embed_norm: bool,
    pub unpadded_vocab_size: Option<i32>,
    pub logits_mup_width_multiplier: f32,
    pub final_logit_softcapping: Option<f32>,
    pub intermediate_size: i32,
    pub dense_intermediate_size: Option<i32>,
    pub moe_intermediate_size: Option<i32>,
    pub n_routed_experts: i32,
    pub num_experts_per_tok: i32,
    pub n_shared_experts: i32,
    pub route_scale: f32,
    pub shared_expert_sink: bool,
    pub use_gate_bias: bool,
    pub norm_after_topk: bool,
    pub use_global_scale: bool,
    pub gate_activation: String,
    pub hidden_act: String,
    pub attention_dropout: f32,
    pub q_bias: bool,
    pub o_bias: bool,
    pub model_max_length: Option<i32>,
    /// Uniform packed representation requested by bounded load-time conversion.
    pub weight_quantization: Option<WeightQuantization>,
    /// Exact per-weight formats for mixed GGUF checkpoints.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

fn default_gate_activation() -> String {
    "sigmoid".into()
}

fn default_hidden_activation() -> String {
    "silu".into()
}

fn default_audio_mode() -> String {
    "dmel".into()
}

fn default_vision_encoder_type() -> String {
    "hmlp".into()
}

impl TextArgs {
    pub(crate) fn weight_dtype(&self) -> Dtype {
        match self.torch_dtype.as_deref() {
            Some("bfloat16" | "bf16") => Dtype::Bfloat16,
            Some("float16") => Dtype::Float16,
            Some("float32") | None => Dtype::Float32,
            Some(_) => unreachable!("Inkling torch_dtype is validated before model construction"),
        }
    }

    pub(crate) fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
            .or(self.weight_quantization)
    }
    pub(crate) fn dense_intermediate_size(&self) -> i32 {
        self.dense_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub(crate) fn moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Returns one validated layer policy without a fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<&LayerPolicy> {
        self.layer_schedule.get(layer)
    }

    pub(crate) fn q_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_attention_heads
                .unwrap_or(self.num_attention_heads)
        } else {
            self.num_attention_heads
        }
    }

    pub(crate) fn kv_heads(&self, local: bool) -> i32 {
        if local {
            self.swa_num_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    pub(crate) fn attention_head_dim(&self, local: bool) -> i32 {
        if local {
            self.swa_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }
}

pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let geometry = args
        .text_config
        .layer_schedule
        .iter()
        .map(|policy| {
            let local = policy.attention.window().is_some();
            ParallelLayerGeometry {
                query_heads: args.text_config.q_heads(local),
                kv_heads: args.text_config.kv_heads(local),
                feed_forward: match policy.feed_forward {
                    FeedForwardPolicy::Dense => ParallelFeedForwardGeometry::Dense {
                        intermediate: args.text_config.dense_intermediate_size(),
                    },
                    FeedForwardPolicy::SparseMoe => ParallelFeedForwardGeometry::SparseMoe {
                        routed_intermediate: args.text_config.moe_intermediate_size(),
                        shared_intermediate: args.text_config.moe_intermediate_size(),
                    },
                },
            }
        })
        .collect::<Vec<_>>();
    prompt_cache_layer_layout_with_geometry(args, &geometry)
}

pub(crate) fn prompt_cache_layer_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[ParallelLayerGeometry],
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let text = &args.text_config;
    let cache_error = |error: crate::core::cache::CachePolicyError| {
        Error::UnsupportedArchitecture(error.to_string())
    };
    let history = text.sconv_kernel_size.checked_sub(1).ok_or_else(|| {
        Error::UnsupportedArchitecture("invalid Inkling convolution width".into())
    })?;
    let fixed = |value| StateTensorDimension::fixed(value).map_err(cache_error);
    let layers = text.num_hidden_layers as usize;
    if geometry.len() != layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling cache geometry has {} layers, expected {layers}",
            geometry.len()
        )));
    }
    let policies = (0..layers)
        .map(|layer| {
            let policy = *text.layer_policy(layer).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling layer schedule has no decoder layer {layer}"
                ))
            })?;
            let local = geometry[layer];
            let sliding = policy.attention.window().is_some();
            let kv_width = local
                .kv_heads
                .checked_mul(text.attention_head_dim(sliding))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture("Inkling KV width overflow".into())
                })?;
            let tensors = [kv_width, kv_width, text.hidden_size, text.hidden_size]
                .into_iter()
                .enumerate()
                .map(|(slot, width)| {
                    StateTensorPolicy::new(
                        StateTensorRole::Convolution { slot: slot as u32 },
                        vec![StateTensorDimension::Batch, fixed(history)?, fixed(width)?],
                        StateTensorDtype::Floating,
                        crate::MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(cache_error)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            LayerCachePolicy::key_value_with_fixed_state(
                policy.attention,
                local.kv_heads,
                text.attention_head_dim(policy.attention.window().is_some()),
                tensors,
            )
            .map_err(cache_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    #[serde(default = "default_model_type")]
    model_type: String,
    text_config: TextArgsSource,
    #[serde(default)]
    mtp_config: Option<InklingMtpConfig>,
    #[serde(default)]
    audio_config: Option<AudioArgs>,
    #[serde(default)]
    vision_config: Option<VisionArgs>,
    #[serde(default = "default_image_token_id")]
    image_token_id: u32,
    #[serde(default = "default_audio_token_id")]
    audio_token_id: u32,
    #[serde(default)]
    eos_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
/// Released top-level Inkling configuration.
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: TextArgs,
    /// Checkpoint-embedded multi-token prediction geometry.
    pub mtp_config: Option<InklingMtpConfig>,
    pub audio_config: Option<AudioArgs>,
    pub vision_config: Option<VisionArgs>,
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct InklingMtpConfig {
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    #[serde(default)]
    pub chain_hidden_post_norm: bool,
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<i32>,
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub swa_num_attention_heads: Option<i32>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<i32>,
    #[serde(default)]
    pub swa_head_dim: Option<i32>,
    #[serde(default)]
    pub dense_intermediate_size: Option<i32>,
    #[serde(default)]
    pub intermediate_size: Option<i32>,
    #[serde(default)]
    pub sconv_kernel_size: Option<i32>,
    #[serde(default)]
    pub rel_extent: Option<i32>,
    #[serde(default)]
    pub d_rel: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioArgs {
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    #[serde(alias = "n_mel_bins")]
    pub num_codebooks: i32,
    #[serde(alias = "mel_vocab_size")]
    pub codebook_size: i32,
    #[serde(default)]
    pub bias: bool,
    #[serde(default = "default_true")]
    pub use_audio_norm: bool,
    #[serde(default = "default_audio_mode")]
    pub audio_mode: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Exact per-weight formats for a mixed GGUF projector.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

impl AudioArgs {
    fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionArgs {
    #[serde(default = "default_vision_encoder_type")]
    pub vision_encoder_type: String,
    #[serde(alias = "decoder_dmodel")]
    pub text_hidden_size: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    #[serde(alias = "n_channels")]
    pub num_channels: i32,
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_true")]
    pub use_vision_norm: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Exact per-weight formats for a mixed GGUF projector.
    #[serde(skip)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

impl VisionArgs {
    pub(crate) fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(name).copied())
    }

    pub(crate) fn layer_specs(&self) -> [(i32, i32, i32, i32); 4] {
        // Inkling-Small narrows the second hMLP stage while retaining the
        // released fold schedule and the 4,800-wide penultimate stage.
        let second_stage_width = if self.text_hidden_size == 4096 {
            320
        } else {
            512
        };
        [
            (75, 128, 1, 5),
            (512, second_stage_width, 1, 2),
            (second_stage_width * 16, 4800, 1, 4),
            (9600, self.text_hidden_size, 2, 1),
        ]
    }
}

#[derive(Debug, Clone)]
/// Global or bounded KV state selected per decoder layer.
pub enum InklingKvCache {
    Global(ConcatKeyValueCache),
    Sliding(ConcatKeyValueCache),
    Paged(PagedKeyValueCache),
}

impl KeyValueCache for InklingKvCache {
    fn offset(&self) -> i32 {
        match self {
            Self::Global(cache) => cache.offset(),
            Self::Sliding(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Global(cache) => cache.max_size(),
            Self::Sliding(cache) => cache.max_size(),
            Self::Paged(cache) => cache.max_size(),
        }
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Global(cache) => cache.retained_arrays(),
            Self::Sliding(cache) => cache.retained_arrays(),
            Self::Paged(cache) => cache.retained_arrays(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Sliding(cache) => cache.update_and_fetch(keys, values, stream),
            Self::Paged(cache) => cache.update_and_fetch(keys, values, stream),
        }
    }

    fn update_for_attention(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Self::Global(cache) => cache.update_for_attention(keys, values, stream),
            Self::Sliding(cache) => cache.update_for_attention(keys, values, stream),
            Self::Paged(cache) => cache.update_for_attention(keys, values, stream),
        }
    }

    fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
    }
}

impl InklingKvCache {
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        match (self, checkpoint) {
            (Self::Global(cache), Self::Global(previous))
            | (Self::Sliding(cache), Self::Sliding(previous)) => {
                cache.clone_from(previous);
                Ok(())
            }
            (Self::Paged(cache), Self::Paged(previous)) => {
                cache.restore_checkpoint(previous, stream)
            }
            _ => Err(Exception::custom(
                "Inkling cache checkpoint changed attention residency mode",
            )),
        }
    }
}

#[derive(Debug, Clone)]
/// Incremental state for one Inkling decoder layer.
pub struct LayerCache {
    pub kv: InklingKvCache,
    pub convolutions: [CausalConv1dCache; 4],
}

impl LayerCache {
    pub(crate) fn restore_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.kv.restore_checkpoint(&checkpoint.kv, stream)?;
        self.convolutions.clone_from(&checkpoint.convolutions);
        Ok(())
    }

    fn new(policy: AttentionPolicy) -> Self {
        Self {
            kv: match policy.window() {
                Some(window) => {
                    InklingKvCache::Sliding(ConcatKeyValueCache::new_for_sliding_attention(
                        i32::try_from(window.get())
                            .expect("validated Inkling sliding window fits i32"),
                    ))
                }
                None => InklingKvCache::Global(ConcatKeyValueCache::new()),
            },
            convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
        }
    }

    fn new_paged(
        args: &TextArgs,
        layer: usize,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Exception::custom(format!("Inkling has no policy for layer {layer}")))?;
        Ok(Self {
            kv: InklingKvCache::Paged(PagedKeyValueCache::new_with_layout(
                manager,
                layer,
                policy.attention.window().map(|window| window.get() as i32),
                0,
                rank,
            )?),
            convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
        })
    }
}

#[derive(Debug, Clone)]
/// Heterogeneous Inkling generation cache.
pub struct Cache {
    pub layers: Vec<LayerCache>,
    pub(crate) mtp_layers: Vec<LayerCache>,
}

impl Cache {
    pub(crate) fn new(args: &TextArgs) -> Self {
        Self {
            layers: args
                .layer_schedule
                .iter()
                .map(|policy| LayerCache::new(policy.attention))
                .collect(),
            mtp_layers: Vec::new(),
        }
    }

    pub(crate) fn with_mtp_policies(mut self, policies: &[AttentionPolicy]) -> Self {
        self.mtp_layers = policies
            .iter()
            .map(|policy| LayerCache::new(*policy))
            .collect();
        self
    }

    pub(crate) fn with_paged_mtp_policies(
        mut self,
        policies: &[AttentionPolicy],
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        if policies.is_empty() {
            return Ok(self);
        }
        let manager = self
            .layers
            .iter()
            .find_map(|layer| match &layer.kv {
                InklingKvCache::Paged(cache) => Some(cache.manager().clone()),
                _ => None,
            })
            .ok_or_else(|| {
                Exception::custom("Inkling paged MTP requires a shared cache manager")
            })?;
        let start = self.layers.len();
        self.mtp_layers = policies
            .iter()
            .copied()
            .enumerate()
            .map(|(index, attention)| {
                Ok::<LayerCache, Exception>(LayerCache {
                    kv: InklingKvCache::Paged(PagedKeyValueCache::new_with_layout(
                        manager.clone(),
                        start + index,
                        attention.window().map(|window| window.get() as i32),
                        0,
                        rank,
                    )?),
                    convolutions: std::array::from_fn(|_| CausalConv1dCache::default()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    pub(crate) fn new_paged(
        args: &TextArgs,
        options: PagedCacheOptions,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let manager = CacheResidencyManager::new(options)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Self::new_paged_with_manager(args, manager, rank)
    }

    pub(crate) fn new_paged_with_manager(
        args: &TextArgs,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, Exception> {
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Inkling cache layer count"))?;
        Ok(Self {
            layers: (0..layer_count)
                .map(|layer| LayerCache::new_paged(args, layer, manager.clone(), rank))
                .collect::<Result<Vec<_>, _>>()?,
            mtp_layers: Vec::new(),
        })
    }

    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.layers
            .iter()
            .find_map(|layer| match &layer.kv {
                InklingKvCache::Paged(cache) => Some(cache.report()),
                _ => None,
            })
            .transpose()
    }

    pub fn offset(&self) -> i32 {
        self.layers.first().map_or(0, |layer| layer.kv.offset())
    }

    pub(crate) fn reset(&mut self) -> Result<(), Exception> {
        let paged_manager =
            self.layers
                .iter()
                .chain(&self.mtp_layers)
                .find_map(|layer| match &layer.kv {
                    InklingKvCache::Paged(cache) => Some(cache.manager().clone()),
                    _ => None,
                });
        if let Some(manager) = paged_manager {
            manager
                .clear()
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        for layer in self.layers.iter_mut().chain(&mut self.mtp_layers) {
            match &mut layer.kv {
                InklingKvCache::Global(cache) => cache.clear(),
                InklingKvCache::Sliding(cache) => cache.clear(),
                InklingKvCache::Paged(cache) => cache.reset_local_after_manager_clear(),
            }
            layer.convolutions = std::array::from_fn(|_| CausalConv1dCache::default());
        }
        Ok(())
    }

    pub(crate) fn restore_target_checkpoint(
        &mut self,
        checkpoint: &Self,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.layers.len() != checkpoint.layers.len() {
            return Err(Exception::custom(
                "Inkling target cache checkpoint has a different layer count",
            ));
        }
        for (cache, previous) in self.layers.iter_mut().zip(&checkpoint.layers) {
            cache.kv.restore_checkpoint(&previous.kv, stream)?;
            cache.convolutions.clone_from(&previous.convolutions);
        }
        Ok(())
    }

    pub(crate) fn validate(&self, schedule: &LayerSchedule<LayerPolicy>) -> Result<(), Error> {
        if self.layers.len() != schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling cache has {} layers, expected {}",
                self.layers.len(),
                schedule.len()
            )));
        }
        for (layer, (cache, policy)) in self.layers.iter().zip(schedule.iter()).enumerate() {
            let actual = match cache.kv.max_size() {
                Some(window) => AttentionPolicy::sliding(u32::try_from(window).map_err(|_| {
                    Error::UnsupportedArchitecture(format!(
                        "Inkling cache layer {layer} has invalid window {window}"
                    ))
                })?)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                None => AttentionPolicy::Full,
            };
            if actual != policy.attention {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling cache policy mismatch at layer {layer}: expected {:?}, got {actual:?}",
                    policy.attention
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingAttention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    policy: AttentionPolicy,
    log_scaling_n_floor: Option<i32>,
    log_scaling_alpha: f32,
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    r_proj: MaybeQuantized<nn::Linear>,
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rel_proj: Param<Array>,
    #[param]
    k_sconv: DepthwiseConv1d,
    #[param]
    v_sconv: DepthwiseConv1d,
}

impl InklingAttention {
    fn sliding_window(&self) -> Option<i32> {
        self.policy.window().map(|window| window.get() as i32)
    }

    fn new(
        args: &TextArgs,
        layer: i32,
        policy: AttentionPolicy,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let local = policy.window().is_some();
        let n_heads = args.q_heads(local);
        let n_kv_heads = args.kv_heads(local);
        let head_dim = args.attention_head_dim(local);
        let rel_extent = policy
            .window()
            .map(|window| window.get() as i32)
            .unwrap_or(args.rel_extent);
        let prefix = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            d_rel: args.d_rel,
            rel_extent,
            policy,
            log_scaling_n_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            q_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.hidden_size,
                n_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.q_proj.weight")),
                args.weight_dtype(),
                stream,
            )?,
            k_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.k_proj.weight")),
                args.weight_dtype(),
                stream,
            )?,
            v_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.hidden_size,
                n_kv_heads * head_dim,
                false,
                args.weight_quantization_for(&format!("{prefix}.v_proj.weight")),
                args.weight_dtype(),
                stream,
            )?,
            r_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.hidden_size,
                n_heads * args.d_rel,
                false,
                args.weight_quantization_for(&format!("{prefix}.r_proj.weight")),
                args.weight_dtype(),
                stream,
            )?,
            o_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                n_heads * head_dim,
                args.hidden_size,
                false,
                args.weight_quantization_for(&format!("{prefix}.o_proj.weight")),
                args.weight_dtype(),
                stream,
            )?,
            q_norm: nn::RmsNorm::unloaded(
                head_dim,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
            k_norm: nn::RmsNorm::unloaded(
                head_dim,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
            rel_proj: Param::<Array>::unloaded(
                &[args.d_rel, rel_extent],
                args.weight_dtype(),
                stream,
            )?,
            k_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            v_sconv: DepthwiseConv1d::new(
                n_kv_heads * head_dim,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    fn repeat_kv(&self, states: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.n_heads == self.n_kv_heads {
            return Ok(states.clone());
        }
        let shape = states.shape();
        let repeats = self.n_heads / self.n_kv_heads;
        broadcast_to(
            &states.reshape(&[shape[0], self.n_kv_heads, 1, shape[2], shape[3]], stream)?,
            &[shape[0], self.n_kv_heads, repeats, shape[2], shape[3]],
            stream,
        )?
        .reshape(&[shape[0], self.n_heads, shape[2], shape[3]], stream)
    }

    fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let seq_len = hidden.dim(1);
        let q_offset = cache.as_ref().map_or(0, |cache| cache.kv.offset());

        let q = self.q_proj.forward(hidden, stream)?;
        let mut k = self.k_proj.forward(hidden, stream)?;
        let mut v = self.v_proj.forward(hidden, stream)?;
        let relative = self.r_proj.forward(hidden, stream)?;

        if let Some(cache) = cache {
            let LayerCache { kv, convolutions } = cache;
            let (k_cache, rest) = convolutions.split_at_mut(1);
            let (v_cache, _) = rest.split_at_mut(1);
            k = short_convolution(&self.k_sconv, &k, Some(&mut k_cache[0]), stream)?;
            v = short_convolution(&self.v_sconv, &v, Some(&mut v_cache[0]), stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            if kv.is_paged() {
                kv.update_for_attention(k, v, stream)?;
                let InklingKvCache::Paged(cache) = &*kv else {
                    unreachable!("paged Inkling cache variant checked above")
                };
                self.attend_paged(q, &relative, batch, seq_len, q_offset, cache, stream)
            } else {
                let (k, v) = kv.update_and_fetch(k, v, stream)?;
                let key_len = k.dim(2);
                let key_offset = q_offset + seq_len - key_len;
                self.attend_chunked(
                    q, k, v, &relative, batch, seq_len, q_offset, key_offset, stream,
                )
            }
        } else {
            k = short_convolution(&self.k_sconv, &k, None, stream)?;
            v = short_convolution(&self.v_sconv, &v, None, stream)?;
            let q = self
                .q_norm
                .forward(
                    &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let k = self
                .k_norm
                .forward(
                    &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                    stream,
                )?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            let v = v
                .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
                .transpose_axes(&[0, 2, 1, 3], stream)?;
            self.attend_chunked(q, k, v, &relative, batch, seq_len, 0, 0, stream)
        }
    }

    fn forward_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        k_cache: &mut CausalConv1dCache,
        v_cache: &mut CausalConv1dCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let seq_len = hidden.dim(1);
        let q_offset = kv.offset();
        let q = self.q_proj.forward(hidden, stream)?;
        let mut k = short_convolution(
            &self.k_sconv,
            &self.k_proj.forward(hidden, stream)?,
            Some(k_cache),
            stream,
        )?;
        let mut v = short_convolution(
            &self.v_sconv,
            &self.v_proj.forward(hidden, stream)?,
            Some(v_cache),
            stream,
        )?;
        let relative = self.r_proj.forward(hidden, stream)?;
        let q = self
            .q_norm
            .forward(
                &q.reshape(&[batch, seq_len, self.n_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        k = self
            .k_norm
            .forward(
                &k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?,
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        v = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        match kv {
            PipelineInklingKvCache::Standard(cache) => {
                let (k, v) = cache.update_and_fetch(k, v, stream)?;
                let key_len = k.dim(2);
                let key_offset = q_offset + seq_len - key_len;
                self.attend_chunked(
                    q, k, v, &relative, batch, seq_len, q_offset, key_offset, stream,
                )
            }
            PipelineInklingKvCache::Paged(cache) => {
                cache.update_for_attention(k, v, stream)?;
                self.attend_paged(q, &relative, batch, seq_len, q_offset, cache, stream)
            }
        }
    }

    fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let partial = self.forward(hidden, cache, stream)?;
        safemlx::distributed::all_sum(&partial, group, stream)
    }

    fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        k_cache: &mut CausalConv1dCache,
        v_cache: &mut CausalConv1dCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let partial = self.forward_with_operator_cache(hidden, kv, k_cache, v_cache, stream)?;
        safemlx::distributed::all_sum(&partial, group, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_chunked(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        relative: &Array,
        batch: i32,
        query_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Bound the eagerly materialized score/bias tensors. Local layers also
        // discard keys outside the earliest query's window for each chunk.
        const TARGET_SCORE_ELEMENTS: i32 = 16 * 1024 * 1024;
        let total_key_len = k.dim(2).max(1);
        let window = self.sliding_window();
        let chunk_size = if window.is_some() {
            256
        } else {
            (TARGET_SCORE_ELEMENTS / (self.n_heads * total_key_len)).clamp(1, 256)
        };
        let key_limit = key_offset + k.dim(2);
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < query_len {
            let end = (start + chunk_size).min(query_len);
            let query_abs_start = query_offset + start;
            let query_abs_end = query_offset + end;
            let chunk_key_start = if let Some(window) = window {
                (query_abs_start - window + 1).max(key_offset)
            } else {
                key_offset
            };
            let chunk_key_end = query_abs_end.min(key_limit);
            let key_start_index = chunk_key_start - key_offset;
            let key_end_index = chunk_key_end - key_offset;
            let mut q_chunk = q.try_index_device((.., .., start..end, ..), stream)?;
            let k_chunk =
                k.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let v_chunk =
                v.try_index_device((.., .., key_start_index..key_end_index, ..), stream)?;
            let relative_chunk = relative.try_index_device((.., start..end, ..), stream)?;
            let (bias, mask, tau) = self.position_data(
                &relative_chunk,
                batch,
                end - start,
                key_end_index - key_start_index,
                query_abs_start,
                chunk_key_start,
                stream,
            )?;
            if let Some(tau) = tau {
                q_chunk = q_chunk.multiply(tau, stream)?;
            }
            outputs.push(self.attend(
                q_chunk,
                k_chunk,
                v_chunk,
                bias,
                mask,
                batch,
                end - start,
                stream,
            )?);
            start = end;
        }
        concatenate_axis(&outputs, 1, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_paged(
        &mut self,
        q: Array,
        relative: &Array,
        batch: i32,
        query_len: i32,
        query_offset: i32,
        cache: &PagedKeyValueCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        const TARGET_SCORE_ELEMENTS: i32 = 16 * 1024 * 1024;
        let window = self.sliding_window();
        let chunk_size = if window.is_some() {
            256
        } else {
            (TARGET_SCORE_ELEMENTS / (self.n_heads * cache.offset().max(1))).clamp(1, 256)
        };
        let mut outputs = Vec::new();
        let mut scanned_blocks = 0;
        let mut scanned_bytes = 0;
        let mut scratch_bytes = 0;
        let mut start = 0;
        while start < query_len {
            let end = (start + chunk_size).min(query_len);
            let query_abs_start = query_offset + start;
            let mut q_chunk = q.try_index_device((.., .., start..end, ..), stream)?;
            let relative_chunk = relative.try_index_device((.., start..end, ..), stream)?;
            if let Some(tau) = self.global_query_scale(query_abs_start, end - start, stream)? {
                q_chunk = q_chunk.multiply(tau, stream)?;
            }
            let mut accumulator = BlockwiseAttentionAccumulator::new(
                &q_chunk,
                1.0 / self.head_dim as f32,
                None,
                query_abs_start as i64,
                window,
                0,
                None,
                cache.offset() as i64,
                stream,
            )?;
            let visible_start = if let Some(window) = window {
                (query_abs_start - window + 1).max(0) as i64
            } else {
                0
            };
            let (blocks, bytes) = cache.visit_attention_blocks(
                visible_start,
                cache.offset() as i64,
                stream,
                |block| {
                    let (bias, _, _) = self.position_data(
                        &relative_chunk,
                        batch,
                        end - start,
                        i32::try_from(block.end - block.start).map_err(|_| {
                            Exception::custom("Inkling paged cache block length overflow")
                        })?,
                        query_abs_start,
                        i32::try_from(block.start).map_err(|_| {
                            Exception::custom("Inkling paged cache position overflow")
                        })?,
                        stream,
                    )?;
                    accumulator.accumulate_with_bias(block, Some(&bias), stream)
                },
            )?;
            scanned_blocks += blocks;
            scanned_bytes += bytes;
            scratch_bytes = scratch_bytes.max(
                batch as u64
                    * self.n_heads as u64
                    * (end - start) as u64
                    * cache.manager().options().block_size_tokens() as u64
                    * 4,
            );
            outputs.push(accumulator.finish(stream)?);
            start = end;
        }
        cache.record_architecture_attention_scan(
            query_len > 1,
            scanned_blocks,
            scanned_bytes,
            scratch_bytes,
        )?;
        let attended = concatenate_axis(&outputs, 2, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, query_len, self.n_heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }

    fn global_query_scale(
        &self,
        query_offset: i32,
        query_len: i32,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        if self.sliding_window().is_some() {
            return Ok(None);
        }
        let Some(floor) = self.log_scaling_n_floor else {
            return Ok(None);
        };
        let positions =
            arange::<i32, i32>(query_offset + 1, query_offset + query_len + 1, 1, stream)?
                .as_dtype(Dtype::Float32, stream)?;
        let ratio = positions.divide(Array::from_f32(floor as f32), stream)?;
        let ratio = safemlx::ops::maximum(ratio, Array::from_f32(1.0), stream)?;
        Ok(Some(
            ratio
                .log(stream)?
                .multiply(Array::from_f32(self.log_scaling_alpha), stream)?
                .add(Array::from_f32(1.0), stream)?
                .reshape(&[1, 1, query_len, 1], stream)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn position_data(
        &self,
        relative: &Array,
        batch: i32,
        query_len: i32,
        key_len: i32,
        query_offset: i32,
        key_offset: i32,
        stream: &Stream,
    ) -> Result<(Array, Array, Option<Array>), Exception> {
        let relative = relative.reshape(&[batch, query_len, self.n_heads, self.d_rel], stream)?;
        let profiles = matmul(relative, self.rel_proj.as_ref(), stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let q_positions = arange::<i32, i32>(query_offset, query_offset + query_len, 1, stream)?
            .try_index_device((.., NewAxis), stream)?;
        let k_positions = arange::<i32, i32>(key_offset, key_offset + key_len, 1, stream)?
            .try_index_device((NewAxis, ..), stream)?;
        let distances = q_positions.subtract(k_positions, stream)?;
        let mut valid = distances.ge(Array::from_int(0), stream)?;
        if let Some(window) = self.sliding_window() {
            valid = valid.logical_and(&distances.lt(Array::from_int(window), stream)?, stream)?;
        }
        let gather = clip(&distances, (0, self.rel_extent - 1), stream)?
            .as_dtype(Dtype::Int32, stream)?
            .try_index_device((NewAxis, NewAxis, .., ..), stream)?;
        let gather = broadcast_to(&gather, &[batch, self.n_heads, query_len, key_len], stream)?;
        let mut bias = take_along_axis(&profiles, &gather, -1, stream)?;
        let relative_valid = distances.ge(Array::from_int(0), stream)?.logical_and(
            &distances.lt(Array::from_int(self.rel_extent), stream)?,
            stream,
        )?;
        bias = r#where(&relative_valid, bias, Array::from_f32(0.0), stream)?;
        let tau = if self.sliding_window().is_none() {
            if let Some(tau) = self.global_query_scale(query_offset, query_len, stream)? {
                bias = bias.multiply(&tau, stream)?;
                Some(tau)
            } else {
                None
            }
        } else {
            None
        };
        Ok((bias, valid, tau))
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        q: Array,
        k: Array,
        v: Array,
        bias: Array,
        valid: Array,
        batch: i32,
        seq_len: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let k = self.repeat_kv(&k, stream)?;
        let v = self.repeat_kv(&v, stream)?;
        let mut scores = matmul(
            &q.multiply(Array::from_f32(1.0 / self.head_dim as f32), stream)?,
            &k.swap_axes(-1, -2, stream)?,
            stream,
        )?
        .add(bias, stream)?;
        scores = r#where(&valid, scores, Array::from_f32(f32::NEG_INFINITY), stream)?;
        let probabilities = softmax_axis(scores, -1, true, stream)?;
        let attended = matmul(probabilities, v, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }
}

/// Borrowed KV storage for pipeline execution of Inkling relative attention.
pub(crate) enum PipelineInklingKvCache<'a> {
    Standard(&'a mut dyn KeyValueCache),
    Paged(&'a mut PagedKeyValueCache),
}

impl PipelineInklingKvCache<'_> {
    fn offset(&self) -> i32 {
        match self {
            Self::Standard(cache) => cache.offset(),
            Self::Paged(cache) => cache.offset(),
        }
    }
}

fn short_convolution(
    convolution: &DepthwiseConv1d,
    input: &Array,
    cache: Option<&mut CausalConv1dCache>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = input.dtype();
    let input = input.as_dtype(Dtype::Float32, stream)?;
    causal_depthwise_conv1d(convolution, &input, cache, stream)?
        .add(&input, stream)?
        .as_dtype(dtype, stream)
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingRouter {
    num_routed: i32,
    num_shared: i32,
    top_k: i32,
    route_scale: f32,
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Array>,
    #[param]
    global_scale: Param<Array>,
}

impl InklingRouter {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            num_routed: args.n_routed_experts,
            num_shared: args.n_shared_experts,
            top_k: args.num_experts_per_tok,
            route_scale: args.route_scale,
            weight: Param::<Array>::unloaded(
                &[
                    args.n_routed_experts + args.n_shared_experts,
                    args.hidden_size,
                ],
                args.weight_dtype(),
                stream,
            )?,
            bias: Param::<Array>::unloaded(&[args.n_routed_experts], Dtype::Float32, stream)?,
            global_scale: Param::<Array>::unloaded(&[1], Dtype::Float32, stream)?,
        })
    }

    fn forward(&self, hidden: &Array, stream: &Stream) -> Result<(Array, Array, Array), Exception> {
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let logits = matmul(&flat, &self.weight.as_ref().transpose(stream)?, stream)?;
        let routed = logits.try_index_device((.., ..self.num_routed), stream)?;
        let shared = logits.try_index_device((.., self.num_routed..), stream)?;
        let choice = sigmoid(&routed, stream)?.add(self.bias.as_ref(), stream)?;
        let indices = argpartition_axis(choice, -self.top_k, -1, stream)?
            .try_index_device((.., -self.top_k..), stream)?;
        let selected_logits = take_along_axis(&routed, &indices, -1, stream)?;
        let all_logits = concatenate_axis(&[selected_logits, shared], -1, stream)?;
        let weights = softmax_axis(nn::log_sigmoid(all_logits, stream)?, -1, true, stream)?
            .multiply(Array::from_f32(self.route_scale), stream)?
            .multiply(self.global_scale.as_ref(), stream)?;
        let routed_weights = weights.try_index_device((.., ..self.top_k), stream)?;
        let shared_weights = weights.try_index_device((.., self.top_k..), stream)?;
        Ok((indices, routed_weights, shared_weights))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingMoe {
    #[param]
    router: InklingRouter,
    #[param]
    experts: PackedSwiGluExperts,
    #[param]
    shared_experts: PackedSwiGluExperts,
}

impl InklingMoe {
    fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let intermediate = args.moe_intermediate_size();
        let prefix = format!("model.layers.{layer}.moe");
        Ok(Self {
            router: InklingRouter::new(args, stream)?,
            experts: PackedSwiGluExperts::new_with_dtype(
                args.n_routed_experts,
                args.hidden_size,
                intermediate,
                args.weight_quantization_for(&format!("{prefix}.experts.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.experts.down_proj")),
                args.weight_dtype(),
                stream,
            )?,
            shared_experts: PackedSwiGluExperts::new_with_dtype(
                args.n_shared_experts,
                args.hidden_size,
                intermediate,
                args.weight_quantization_for(&format!("{prefix}.shared_experts.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.shared_experts.down_proj")),
                args.weight_dtype(),
                stream,
            )?,
        })
    }

    fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = self.experts.forward(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }

    fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let partial = self.forward(hidden, stream)?;
        safemlx::distributed::all_sum(&partial, group, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }

    fn forward_tensor_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        let routed = execute(&flat, &indices, &weights, stream)?;
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared_partial =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        let shared = safemlx::distributed::all_sum(&shared_partial, group, stream)?;
        routed.add(shared, stream)?.reshape(&shape, stream)
    }

    fn forward_expert_parallel(
        &mut self,
        hidden: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&flat])?;
        let router_started = std::time::Instant::now();
        let (indices, weights, shared_weights) = self.router.forward(hidden, stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&indices, &weights])?;
        statistics.router_time += router_started.elapsed();
        let returned = crate::architectures::distributed::expert::dispatch_replicated(
            &flat,
            &indices,
            &weights,
            assignment,
            &mut self.experts,
            group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        let shared_started = std::time::Instant::now();
        let tokens = flat.dim(0);
        let shared_indices = broadcast_to(
            &arange::<i32, i32>(0, self.router.num_shared, 1, stream)?
                .try_index_device((NewAxis, ..), stream)?,
            &[tokens, self.router.num_shared],
            stream,
        )?;
        let shared =
            self.shared_experts
                .forward(&flat, &shared_indices, &shared_weights, stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&shared])?;
        statistics.shared_expert_time += shared_started.elapsed();
        returned
            .reduced_output
            .add(shared, stream)?
            .reshape(&shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_tensor_expert_parallel(
        &mut self,
        hidden: &Array,
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let partial =
            self.forward_expert_parallel(hidden, assignment, expert_group, statistics, stream)?;
        safemlx::distributed::all_sum(&partial, tensor_group, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct DecoderLayer {
    intermediate: i32,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    self_attn: InklingAttention,
    #[param]
    attn_sconv: DepthwiseConv1d,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    dense: Option<SwiGluMlp>,
    #[param]
    dense_global_scale: Param<Option<Array>>,
    #[param]
    moe: Option<InklingMoe>,
    #[param]
    mlp_sconv: DepthwiseConv1d,
}

impl DecoderLayer {
    pub(crate) fn new(args: &TextArgs, layer: i32, stream: &Stream) -> Result<Self, Exception> {
        let policy = *args
            .layer_policy(layer as usize)
            .ok_or_else(|| Exception::custom(format!("Inkling has no policy for layer {layer}")))?;
        let dense = policy.feed_forward == FeedForwardPolicy::Dense;
        let intermediate = if dense {
            args.dense_intermediate_size()
        } else {
            args.moe_intermediate_size()
        };
        Ok(Self {
            intermediate,
            input_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
            self_attn: InklingAttention::new(args, layer, policy.attention, stream)?,
            attn_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
            dense: if dense {
                let prefix = format!("model.layers.{layer}.dense");
                Some(SwiGluMlp {
                    gate_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                        args.hidden_size,
                        args.dense_intermediate_size(),
                        false,
                        args.weight_quantization_for(&format!("{prefix}.gate_proj.weight")),
                        args.weight_dtype(),
                        stream,
                    )?,
                    down_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                        args.dense_intermediate_size(),
                        args.hidden_size,
                        false,
                        args.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                        args.weight_dtype(),
                        stream,
                    )?,
                    up_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                        args.hidden_size,
                        args.dense_intermediate_size(),
                        false,
                        args.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                        args.weight_dtype(),
                        stream,
                    )?,
                })
            } else {
                None
            },
            dense_global_scale: if dense {
                Param::<Option<Array>>::unloaded_some(&[1], args.weight_dtype(), stream)?
            } else {
                Param::new(None)
            },
            moe: if dense {
                None
            } else {
                Some(InklingMoe::new(args, layer, stream)?)
            },
            mlp_sconv: DepthwiseConv1d::new(
                args.hidden_size,
                args.sconv_kernel_size,
                false,
                stream,
            )?,
        })
    }

    pub(crate) fn new_expert_parallel(
        args: &TextArgs,
        layer: i32,
        local_experts: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut result = Self::new(args, layer, stream)?;
        if let Some(moe) = &mut result.moe {
            let prefix = format!("model.layers.{layer}.moe.experts");
            moe.experts = PackedSwiGluExperts::new_with_dtype(
                local_experts,
                args.hidden_size,
                args.moe_intermediate_size(),
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                args.weight_dtype(),
                stream,
            )?;
        }
        Ok(result)
    }

    pub(crate) fn new_parallel_layerwise(
        args: &TextArgs,
        layer: i32,
        geometry: ParallelLayerGeometry,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let valid_feed_forward = match geometry.feed_forward {
            ParallelFeedForwardGeometry::Dense { intermediate } => intermediate > 0,
            ParallelFeedForwardGeometry::SparseMoe {
                routed_intermediate,
                shared_intermediate,
            } => routed_intermediate > 0 && shared_intermediate > 0,
        };
        if geometry.query_heads <= 0 || geometry.kv_heads <= 0 || !valid_feed_forward {
            return Err(Exception::custom(format!(
                "Inkling layer {layer} has invalid local parallel geometry {geometry:?}"
            )));
        }
        let mut local = args.clone();
        let policy = *args
            .layer_policy(layer as usize)
            .ok_or_else(|| Exception::custom(format!("Inkling has no policy for layer {layer}")))?;
        if policy.attention.window().is_some() {
            local.swa_num_attention_heads = Some(geometry.query_heads);
            local.swa_num_key_value_heads = Some(geometry.kv_heads);
        } else {
            local.num_attention_heads = geometry.query_heads;
            local.num_key_value_heads = geometry.kv_heads;
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let ParallelFeedForwardGeometry::Dense { intermediate } = geometry.feed_forward
                else {
                    return Err(Exception::custom(
                        "Inkling dense layer received sparse parallel geometry",
                    ));
                };
                local.dense_intermediate_size = Some(intermediate);
            }
            FeedForwardPolicy::SparseMoe => {
                let ParallelFeedForwardGeometry::SparseMoe {
                    routed_intermediate,
                    shared_intermediate,
                } = geometry.feed_forward
                else {
                    return Err(Exception::custom(
                        "Inkling sparse layer received dense parallel geometry",
                    ));
                };
                local.moe_intermediate_size = Some(routed_intermediate);
                let mut result = Self::new(&local, layer, stream)?;
                let moe = result.moe.as_mut().expect("validated sparse Inkling layer");
                let prefix = format!("model.layers.{layer}.moe.shared_experts");
                moe.shared_experts = PackedSwiGluExperts::new_with_dtype(
                    local.n_shared_experts,
                    local.hidden_size,
                    shared_intermediate,
                    local.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                    local.weight_quantization_for(&format!("{prefix}.down_proj")),
                    local.weight_dtype(),
                    stream,
                )?;
                return Ok(result);
            }
        }
        Self::new(&local, layer, stream)
    }

    pub(crate) fn new_tensor_expert_parallel(
        args: &TextArgs,
        layer: i32,
        geometry: ParallelLayerGeometry,
        local_experts: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let mut result = Self::new_parallel_layerwise(args, layer, geometry, stream)?;
        if let Some(moe) = &mut result.moe {
            let prefix = format!("model.layers.{layer}.moe.experts");
            moe.experts = PackedSwiGluExperts::new_with_dtype(
                local_experts,
                args.hidden_size,
                result.intermediate,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                args.weight_dtype(),
                stream,
            )?;
        }
        Ok(result)
    }

    pub(crate) fn register_tensor_parallel_parameters(
        &self,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        prefix: &str,
    ) -> Result<(), Error> {
        use crate::runtime::distributed::parallel::{
            aligned_partition_units, array_parameter_member, partitioned_projection_members,
            register_partitioned_projection_group, MemberSharding, ParameterGroupSpec,
            ParameterMemberSpec, ParameterRole, ProjectionSharding,
        };
        let attention_prefix = format!("{prefix}.self_attn");
        let q = format!("{attention_prefix}.q_proj");
        let k = format!("{attention_prefix}.k_proj");
        let v = format!("{attention_prefix}.v_proj");
        let r = format!("{attention_prefix}.r_proj");
        let o = format!("{attention_prefix}.o_proj");
        let (attention_units, mut attention_members) = partitioned_projection_members(
            &[
                (
                    &self.self_attn.q_proj,
                    q.as_str(),
                    ProjectionSharding::Column,
                ),
                (
                    &self.self_attn.k_proj,
                    k.as_str(),
                    ProjectionSharding::Column,
                ),
                (
                    &self.self_attn.v_proj,
                    v.as_str(),
                    ProjectionSharding::Column,
                ),
                (
                    &self.self_attn.r_proj,
                    r.as_str(),
                    ProjectionSharding::Column,
                ),
                (&self.self_attn.o_proj, o.as_str(), ProjectionSharding::Row),
            ],
            usize::try_from(self.self_attn.n_kv_heads)
                .map_err(|_| Error::Parallel("Inkling KV-head count exceeds usize".into()))?,
        )?;
        for (name, convolution) in [
            ("k_sconv", &self.self_attn.k_sconv),
            ("v_sconv", &self.self_attn.v_sconv),
        ] {
            attention_members.push(array_parameter_member(
                format!("{attention_prefix}.{name}.weight"),
                convolution.weight.as_ref(),
                MemberSharding::Partitioned { axis: 0 },
            )?);
            if let Some(bias) = convolution.bias.as_ref().as_ref() {
                attention_members.push(array_parameter_member(
                    format!("{attention_prefix}.{name}.bias"),
                    bias,
                    MemberSharding::Partitioned { axis: 0 },
                )?);
            }
        }
        planner.register(ParameterGroupSpec::partitioned(
            format!("{attention_prefix}.gqa"),
            ParameterRole::AttentionHeads,
            attention_units,
            attention_members,
        )?)?;
        macro_rules! replicated {
            ($name:literal, $module:expr) => {
                crate::nn::parallel::register_replicated_parameter_group(
                    planner,
                    $module,
                    &format!("{prefix}.{}", $name),
                )?;
            };
        }
        replicated!("input_layernorm", &self.input_layernorm);
        replicated!("post_attention_layernorm", &self.post_attention_layernorm);
        replicated!("self_attn.q_norm", &self.self_attn.q_norm);
        replicated!("self_attn.k_norm", &self.self_attn.k_norm);
        replicated!("attn_sconv", &self.attn_sconv);
        replicated!("mlp_sconv", &self.mlp_sconv);
        planner.register(ParameterGroupSpec::new(
            format!("{prefix}.self_attn.rel_proj"),
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                format!("{prefix}.self_attn.rel_proj"),
                self.self_attn
                    .rel_proj
                    .value
                    .shape()
                    .iter()
                    .map(|&value| value as usize)
                    .collect::<Vec<_>>(),
                MemberSharding::Replicated,
            )],
        )?)?;
        if let Some(scale) = self.dense_global_scale.value.as_ref() {
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.dense_global_scale"),
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    format!("{prefix}.dense_global_scale"),
                    scale
                        .shape()
                        .iter()
                        .map(|&value| value as usize)
                        .collect::<Vec<_>>(),
                    MemberSharding::Replicated,
                )],
            )?)?;
        }
        if let Some(dense) = &self.dense {
            let gate = format!("{prefix}.dense.gate_proj");
            let up = format!("{prefix}.dense.up_proj");
            let down = format!("{prefix}.dense.down_proj");
            register_partitioned_projection_group(
                planner,
                &format!("{prefix}.dense.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (&dense.gate_proj, gate.as_str(), ProjectionSharding::Column),
                    (&dense.up_proj, up.as_str(), ProjectionSharding::Column),
                    (&dense.down_proj, down.as_str(), ProjectionSharding::Row),
                ],
                usize::try_from(self.intermediate)
                    .map_err(|_| Error::Parallel("Inkling dense width exceeds usize".into()))?,
            )?;
        }
        if let Some(moe) = &self.moe {
            crate::nn::parallel::register_replicated_parameter_group(
                planner,
                &moe.router,
                &format!("{prefix}.moe.router"),
            )?;
            for (bank_name, bank) in [
                ("experts", &moe.experts),
                ("shared_experts", &moe.shared_experts),
            ] {
                let bank_prefix = format!("{prefix}.moe.{bank_name}");
                let intermediate = usize::try_from(bank.intermediate_dim)
                    .map_err(|_| Error::Parallel("Inkling expert width exceeds usize".into()))?;
                let mut members = vec![ParameterMemberSpec::new(
                    format!("{bank_prefix}.gate_up_proj"),
                    bank.gate_up_proj
                        .value
                        .shape()
                        .iter()
                        .map(|&value| value as usize)
                        .collect::<Vec<_>>(),
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..intermediate, intermediate..2 * intermediate],
                    },
                )];
                for (suffix, value) in [
                    (
                        "gate_up_proj_scales",
                        bank.gate_up_proj_scales.value.as_ref(),
                    ),
                    (
                        "gate_up_proj_biases",
                        bank.gate_up_proj_biases.value.as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        members.push(ParameterMemberSpec::new(
                            format!("{bank_prefix}.{suffix}"),
                            value
                                .shape()
                                .iter()
                                .map(|&value| value as usize)
                                .collect::<Vec<_>>(),
                            MemberSharding::PartitionedSegments {
                                axis: 1,
                                segments: vec![0..intermediate, intermediate..2 * intermediate],
                            },
                        ));
                    }
                }
                let down_shape = bank
                    .down_proj
                    .value
                    .shape()
                    .iter()
                    .map(|&value| value as usize)
                    .collect::<Vec<_>>();
                members.push(ParameterMemberSpec::new(
                    format!("{bank_prefix}.down_proj"),
                    down_shape,
                    MemberSharding::Partitioned { axis: 2 },
                ));
                for (suffix, value) in [
                    ("down_proj_scales", bank.down_proj_scales.value.as_ref()),
                    ("down_proj_biases", bank.down_proj_biases.value.as_ref()),
                ] {
                    if let Some(value) = value {
                        let shape = value
                            .shape()
                            .iter()
                            .map(|&value| value as usize)
                            .collect::<Vec<_>>();
                        members.push(ParameterMemberSpec::new(
                            format!("{bank_prefix}.{suffix}"),
                            shape,
                            MemberSharding::Partitioned { axis: 2 },
                        ));
                    }
                }
                planner.register(ParameterGroupSpec::partitioned(
                    format!("{bank_prefix}.intermediate"),
                    ParameterRole::ExpertIntermediate,
                    aligned_partition_units(
                        &format!("{bank_prefix}.intermediate"),
                        intermediate,
                        1,
                        usize::try_from(
                            bank.down_affine
                                .or(bank.down_iquant)
                                .map_or(1, |quantization| quantization.group_size()),
                        )
                        .map_err(|_| {
                            Error::Parallel("Inkling expert alignment exceeds usize".into())
                        })?,
                    )?,
                    members,
                )?)?;
            }
        }
        Ok(())
    }

    pub(crate) fn forward(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp(&normalized, stream)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    pub(crate) fn forward_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = self.forward_mlp(&normalized, stream)?;
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention, mut mlp_cache) = match cache {
            Some(cache) => {
                let attention = self.self_attn.forward_tensor_parallel(
                    &normalized,
                    Some(cache),
                    group,
                    stream,
                )?;
                (attention, Some(cache))
            }
            None => (
                self.self_attn
                    .forward_tensor_parallel(&normalized, None, group, stream)?,
                None,
            ),
        };
        let attention = short_convolution(
            &self.attn_sconv,
            &attention,
            mlp_cache
                .as_deref_mut()
                .map(|cache| &mut cache.convolutions[2]),
            stream,
        )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let gate =
                crate::nn::layers::silu(dense.gate_proj.forward(&normalized, stream)?, stream)?;
            let up = dense.up_proj.forward(&normalized, stream)?;
            let partial = crate::nn::parallel::forward_row_parallel(
                &mut dense.down_proj,
                &gate.multiply(up, stream)?,
                group,
                stream,
            )?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => partial.multiply(scale, stream)?,
                None => partial,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_tensor_parallel(&normalized, group, stream)?
        };
        let mlp = short_convolution(
            &self.mlp_sconv,
            &mlp,
            mlp_cache.map(|cache| &mut cache.convolutions[3]),
            stream,
        )?;
        hidden.add(mlp, stream)
    }

    pub(crate) fn forward_tensor_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention, mut mlp_cache) = match cache {
            Some(cache) => {
                let attention = self.self_attn.forward_tensor_parallel(
                    &normalized,
                    Some(cache),
                    group,
                    stream,
                )?;
                (attention, Some(cache))
            }
            None => (
                self.self_attn
                    .forward_tensor_parallel(&normalized, None, group, stream)?,
                None,
            ),
        };
        let attention = short_convolution(
            &self.attn_sconv,
            &attention,
            mlp_cache
                .as_deref_mut()
                .map(|cache| &mut cache.convolutions[2]),
            stream,
        )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let gate =
                crate::nn::layers::silu(dense.gate_proj.forward(&normalized, stream)?, stream)?;
            let up = dense.up_proj.forward(&normalized, stream)?;
            let partial = crate::nn::parallel::forward_row_parallel(
                &mut dense.down_proj,
                &gate.multiply(up, stream)?,
                group,
                stream,
            )?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => partial.multiply(scale, stream)?,
                None => partial,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_tensor_with_expert_executor(&normalized, group, stream, execute)?
        };
        let mlp = short_convolution(
            &self.mlp_sconv,
            &mlp,
            mlp_cache.map(|cache| &mut cache.convolutions[3]),
            stream,
        )?;
        hidden.add(mlp, stream)
    }

    pub(crate) fn forward_tensor_parallel_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_tensor_parallel_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            group,
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let gate =
                crate::nn::layers::silu(dense.gate_proj.forward(&normalized, stream)?, stream)?;
            let up = dense.up_proj.forward(&normalized, stream)?;
            let partial = crate::nn::parallel::forward_row_parallel(
                &mut dense.down_proj,
                &gate.multiply(up, stream)?,
                group,
                stream,
            )?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => partial.multiply(scale, stream)?,
                None => partial,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_tensor_parallel(&normalized, group, stream)?
        };
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_with_expert_executor_and_operator_cache<F>(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_tensor_parallel_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            group,
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let gate =
                crate::nn::layers::silu(dense.gate_proj.forward(&normalized, stream)?, stream)?;
            let up = dense.up_proj.forward(&normalized, stream)?;
            let partial = crate::nn::parallel::forward_row_parallel(
                &mut dense.down_proj,
                &gate.multiply(up, stream)?,
                group,
                stream,
            )?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => partial.multiply(scale, stream)?,
                None => partial,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_tensor_with_expert_executor(&normalized, group, stream, execute)?
        };
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_expert_executor_and_operator_cache<F>(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let output = dense.forward(&normalized, stream)?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream)?,
                None => output,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_with_expert_executor(&normalized, stream, execute)?
        };
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        tensor_group: &safemlx::distributed::Group,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_tensor_parallel_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            tensor_group,
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let gate =
                crate::nn::layers::silu(dense.gate_proj.forward(&normalized, stream)?, stream)?;
            let up = dense.up_proj.forward(&normalized, stream)?;
            let partial = crate::nn::parallel::forward_row_parallel(
                &mut dense.down_proj,
                &gate.multiply(up, stream)?,
                tensor_group,
                stream,
            )?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => partial.multiply(scale, stream)?,
                None => partial,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_tensor_expert_parallel(
                    &normalized,
                    tensor_group,
                    assignment,
                    expert_group,
                    statistics,
                    stream,
                )?
        };
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_operator_cache(
        &mut self,
        hidden: &Array,
        kv: &mut PipelineInklingKvCache<'_>,
        convolutions: &mut [CausalConv1dCache; 4],
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let (attention_kv, output) = convolutions.split_at_mut(2);
        let (key_state, value_state) = attention_kv.split_at_mut(1);
        let attention = self.self_attn.forward_with_operator_cache(
            &normalized,
            kv,
            &mut key_state[0],
            &mut value_state[0],
            stream,
        )?;
        let attention =
            short_convolution(&self.attn_sconv, &attention, Some(&mut output[0]), stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = if let Some(dense) = &mut self.dense {
            let output = dense.forward(&normalized, stream)?;
            match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream)?,
                None => output,
            }
        } else {
            self.moe
                .as_mut()
                .expect("validated sparse layer")
                .forward_expert_parallel(&normalized, assignment, group, statistics, stream)?
        };
        let mlp = short_convolution(&self.mlp_sconv, &mlp, Some(&mut output[1]), stream)?;
        hidden.add(mlp, stream)
    }

    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        cache: Option<&mut LayerCache>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        match cache {
            Some(cache) => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, Some(cache), stream)?;
                let attention = short_convolution(
                    &self.attn_sconv,
                    &attention,
                    Some(&mut cache.convolutions[2]),
                    stream,
                )?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(
                    &self.mlp_sconv,
                    &mlp,
                    Some(&mut cache.convolutions[3]),
                    stream,
                )?;
                hidden.add(mlp, stream)
            }
            None => {
                let normalized = self.input_layernorm.forward(hidden, stream)?;
                let attention = self.self_attn.forward(&normalized, None, stream)?;
                let attention = short_convolution(&self.attn_sconv, &attention, None, stream)?;
                let hidden = hidden.add(attention, stream)?;
                let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
                let mlp = self.forward_mlp_with_expert_executor(&normalized, stream, execute)?;
                let mlp = short_convolution(&self.mlp_sconv, &mlp, None, stream)?;
                hidden.add(mlp, stream)
            }
        }
    }

    fn forward_mlp(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward(hidden, stream)
    }

    fn forward_mlp_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        if let Some(dense) = &mut self.dense {
            let output = dense.forward(hidden, stream)?;
            return match self.dense_global_scale.as_ref() {
                Some(scale) => output.multiply(scale, stream),
                None => Ok(output),
            };
        }
        self.moe
            .as_mut()
            .expect("validated sparse layer")
            .forward_with_expert_executor(hidden, stream, execute)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct TextModel {
    #[param]
    pub(crate) embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub(crate) embed_norm: nn::RmsNorm,
    #[param]
    pub(crate) layers: Vec<DecoderLayer>,
    #[param]
    pub(crate) norm: nn::RmsNorm,
}

impl TextModel {
    fn new(args: &TextArgs, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            embed_tokens: common::linear::unloaded_maybe_quantized_embedding_with_dtype(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                args.weight_dtype(),
                stream,
            )?,
            embed_norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|layer| DecoderLayer::new(args, layer, stream))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                args.weight_dtype(),
                stream,
            )?,
        })
    }

    fn embed(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Exception> {
        let embedded = self.embed_tokens.forward(tokens, stream)?;
        self.embed_norm.forward(&embedded, stream)
    }

    fn forward(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut hidden = match inputs_embeds {
            Some(embeddings) => embeddings.clone(),
            None => self.embed(tokens, stream)?,
        };
        if let Some(cache) = cache {
            for (layer, cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
                hidden = layer.forward(&hidden, Some(cache), stream)?;
            }
        } else {
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, None, stream)?;
            }
        }
        self.norm.forward(&hidden, stream)
    }

    fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut hidden = self.embed(tokens, stream)?;
        for (index, (layer, layer_cache)) in self
            .layers
            .iter_mut()
            .zip(cache.layers.iter_mut())
            .enumerate()
        {
            hidden = if layer.moe.is_some() {
                layer.forward_with_expert_executor(
                    &hidden,
                    Some(layer_cache),
                    stream,
                    |flat, ids, weights, stream| execute(index, flat, ids, weights, stream),
                )?
            } else {
                layer.forward(&hidden, Some(layer_cache), stream)?
            };
        }
        self.norm.forward(&hidden, stream)
    }
}

/// Applies Inkling's authoritative post-decoder output policy around a caller-
/// supplied vocabulary projection.
///
/// Resident, bounded, tensor-parallel, and pipeline execution own different
/// projection modules, but token selection, muP scaling, and removal of padded
/// vocabulary rows are architecture semantics and must not diverge by backend.
pub(crate) fn project_text_logits<E>(
    hidden: &Array,
    args: &TextArgs,
    last_token_only: bool,
    stream: &Stream,
    project: impl FnOnce(&Array, &Stream) -> Result<Array, E>,
) -> Result<Array, E>
where
    E: From<Exception>,
{
    let hidden = if last_token_only {
        hidden
            .try_index_device((.., -1, ..), stream)
            .map_err(E::from)?
    } else {
        hidden.clone()
    };
    let hidden = hidden
        .divide(Array::from_f32(args.logits_mup_width_multiplier), stream)
        .map_err(E::from)?;
    let logits = project(&hidden, stream)?;
    let Some(size) = args
        .unpadded_vocab_size
        .filter(|&size| size < logits.dim(-1))
    else {
        return Ok(logits);
    };
    match logits.ndim() {
        2 => logits
            .try_index_device((.., ..size), stream)
            .map_err(E::from),
        3 => logits
            .try_index_device((.., .., ..size), stream)
            .map_err(E::from),
        rank => Err(E::from(Exception::custom(format!(
            "Inkling logits have unsupported rank {rank}"
        )))),
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct AudioModel {
    num_codebooks: i32,
    codebook_size: i32,
    #[param]
    encoder: MaybeQuantized<nn::Embedding>,
    #[param]
    final_norm: nn::RmsNorm,
    parallel_range: Option<Range<usize>>,
    parallel_parts: usize,
}

impl AudioModel {
    pub(crate) fn new(
        args: &AudioArgs,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            num_codebooks: args.num_codebooks,
            codebook_size: args.codebook_size,
            encoder: common::linear::unloaded_maybe_quantized_embedding_with_dtype(
                args.num_codebooks * args.codebook_size,
                args.text_hidden_size,
                args.weight_quantization_for("audio.encoder.weight"),
                dense_dtype,
                stream,
            )?,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                dense_dtype,
                stream,
            )?,
            parallel_range: None,
            parallel_parts: 1,
        })
    }

    pub(crate) fn new_tensor_parallel(
        args: &AudioArgs,
        dense_dtype: Dtype,
        topology: crate::runtime::distributed::topology::ParallelTopology,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let global = (args.num_codebooks * args.codebook_size) as usize;
        let range = crate::runtime::distributed::topology::balanced_contiguous_range(
            global,
            topology.tensor_parallel_size,
            topology.tensor_parallel_rank,
            false,
        )?;
        Ok(Self {
            num_codebooks: args.num_codebooks,
            codebook_size: args.codebook_size,
            encoder: common::linear::unloaded_maybe_quantized_embedding_with_dtype(
                range.len() as i32,
                args.text_hidden_size,
                args.weight_quantization_for("audio.encoder.weight"),
                dense_dtype,
                stream,
            )?,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                dense_dtype,
                stream,
            )?,
            parallel_range: Some(range),
            parallel_parts: topology.tensor_parallel_size,
        })
    }

    pub(crate) fn register_tensor_parallel_parameters(
        &self,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        prefix: &str,
    ) -> Result<(), Error> {
        use crate::runtime::distributed::parallel::{
            MemberSharding, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        };
        let mut members = Vec::new();
        for (name, value) in self.encoder.parameters().flatten() {
            members.push(ParameterMemberSpec::new(
                format!("{prefix}.encoder.{name}"),
                value
                    .shape()
                    .iter()
                    .map(|&dimension| dimension as usize)
                    .collect::<Vec<_>>(),
                MemberSharding::Balanced { axis: 0 },
            ));
        }
        planner.register(ParameterGroupSpec::new(
            format!("{prefix}.encoder"),
            ParameterRole::Vocabulary,
            members,
        )?)?;
        crate::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.final_norm,
            &format!("{prefix}.final_norm"),
        )
    }

    pub(crate) fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let input_ids = match input_ids.ndim() {
            2 => input_ids.try_index_device(NewAxis, stream)?,
            3 if input_ids.dim(0) == 1 => input_ids.clone(),
            _ => {
                return Err(Exception::custom(format!(
                    "Inkling audio IDs must be [frames, {}] or [1, frames, {}], got {:?}",
                    self.num_codebooks,
                    self.num_codebooks,
                    input_ids.shape()
                )))
            }
        };
        if input_ids.dim(-1) != self.num_codebooks {
            return Err(Exception::custom(format!(
                "Inkling audio IDs require {} dMel codebooks, got {:?}",
                self.num_codebooks,
                input_ids.shape()
            )));
        }
        let offsets = arange::<i32, i32>(
            0,
            self.num_codebooks * self.codebook_size,
            self.codebook_size,
            stream,
        )?
        .reshape(&[1, 1, self.num_codebooks], stream)?;
        let indices = input_ids
            .as_dtype(Dtype::Int32, stream)?
            .add(offsets, stream)?;
        let embedded = self.encoder.forward(&indices, stream)?;
        let mut embedded = sum_axis(&embedded, -2, false, stream)?;
        embedded = self.final_norm.forward(&embedded, stream)?;
        if let Some(mask) = mask {
            if mask.ndim() != 2 || mask.dim(0) != 1 || mask.dim(1) != embedded.dim(1) {
                return Err(Exception::custom(format!(
                    "Inkling audio mask must be [1, frames], got {:?}",
                    mask.shape()
                )));
            }
            let valid = mask.sum(None, stream)?.item::<i32>(stream);
            embedded = embedded.try_index_device((.., ..valid, ..), stream)?;
        }
        Ok(embedded)
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if group.size() != self.parallel_parts {
            return Err(Exception::custom("Inkling audio TP group size mismatch"));
        }
        let range = self
            .parallel_range
            .as_ref()
            .expect("TP Inkling audio encoder")
            .clone();
        let input_ids = match input_ids.ndim() {
            2 => input_ids.try_index_device(NewAxis, stream)?,
            3 if input_ids.dim(0) == 1 => input_ids.clone(),
            _ => {
                return Err(Exception::custom(format!(
                    "Inkling audio IDs must be [frames, {}] or [1, frames, {}], got {:?}",
                    self.num_codebooks,
                    self.num_codebooks,
                    input_ids.shape()
                )))
            }
        };
        let offsets = arange::<i32, i32>(
            0,
            self.num_codebooks * self.codebook_size,
            self.codebook_size,
            stream,
        )?
        .reshape(&[1, 1, self.num_codebooks], stream)?;
        let indices = input_ids
            .as_dtype(Dtype::Int32, stream)?
            .add(offsets, stream)?;
        let start = Array::from_int(range.start as i32);
        let end = Array::from_int(range.end as i32);
        let valid = indices
            .ge(&start, stream)?
            .logical_and(indices.lt(&end, stream)?, stream)?;
        let local = indices.subtract(&start, stream)?;
        let safe = r#where(&valid, &local, Array::from_int(0), stream)?;
        let embedded = self.encoder.forward(&safe, stream)?;
        let embedded = r#where(
            &valid.expand_dims(-1, stream)?,
            &embedded,
            safemlx::ops::zeros_like(&embedded, stream)?,
            stream,
        )?;
        let embedded = safemlx::distributed::all_sum(&embedded, group, stream)?;
        let mut embedded = sum_axis(&embedded, -2, false, stream)?;
        embedded = self.final_norm.forward(&embedded, stream)?;
        if let Some(mask) = mask {
            let valid = mask.sum(None, stream)?.item::<i32>(stream);
            embedded = embedded.try_index_device((.., ..valid, ..), stream)?;
        }
        Ok(embedded)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionLayer {
    t_fold: i32,
    hw_fold: i32,
    parallel_input_range: Option<Range<i32>>,
    #[param]
    projection: MaybeQuantized<nn::Linear>,
    #[param]
    layer_norm: Option<nn::RmsNorm>,
}

impl VisionLayer {
    pub(crate) fn new(
        spec: (i32, i32, i32, i32),
        add_norm: bool,
        eps: f32,
        quantization: Option<WeightQuantization>,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (input_dim, output_dim, t_fold, hw_fold) = spec;
        Ok(Self {
            t_fold,
            hw_fold,
            parallel_input_range: None,
            projection: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                input_dim,
                output_dim,
                false,
                quantization,
                dense_dtype,
                stream,
            )?,
            layer_norm: add_norm
                .then(|| nn::RmsNorm::unloaded(output_dim, eps, dense_dtype, stream))
                .transpose()?,
        })
    }

    pub(crate) fn new_parallel_layerwise(
        spec: (i32, i32, i32, i32),
        input_range: Range<i32>,
        add_norm: bool,
        eps: f32,
        quantization: Option<WeightQuantization>,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if input_range.start < 0 || input_range.start >= input_range.end {
            return Err(Exception::custom(format!(
                "Inkling vision layer has invalid local input range {input_range:?}"
            )));
        }
        let mut layer = Self::new(
            (input_range.end - input_range.start, spec.1, spec.2, spec.3),
            add_norm,
            eps,
            quantization,
            dense_dtype,
            stream,
        )?;
        layer.parallel_input_range = Some(input_range);
        Ok(layer)
    }

    pub(crate) fn forward(&mut self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut hidden = self.fold(hidden, stream)?;
        hidden = self.projection.forward(&hidden, stream)?;
        if let Some(norm) = &mut self.layer_norm {
            hidden = norm.forward(&hidden, stream)?;
            hidden = nn::gelu(hidden, stream)?;
        }
        Ok(hidden)
    }

    fn fold(&self, hidden: &Array, stream: &Stream) -> Result<Array, Exception> {
        if self.t_fold > 1 || self.hw_fold > 1 {
            let shape = hidden.shape();
            if shape.len() != 5
                || shape[1] % self.t_fold != 0
                || shape[2] % self.hw_fold != 0
                || shape[3] % self.hw_fold != 0
            {
                return Err(Exception::custom(format!(
                    "Inkling hMLP fold ({}, {}) is incompatible with {:?}",
                    self.t_fold, self.hw_fold, shape
                )));
            }
            let (batch, time, height, width, channels) =
                (shape[0], shape[1], shape[2], shape[3], shape[4]);
            Ok(hidden
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        self.t_fold,
                        height / self.hw_fold,
                        self.hw_fold,
                        width / self.hw_fold,
                        self.hw_fold,
                        channels,
                    ],
                    stream,
                )?
                .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7], stream)?
                .reshape(
                    &[
                        batch,
                        time / self.t_fold,
                        height / self.hw_fold,
                        width / self.hw_fold,
                        self.t_fold * self.hw_fold * self.hw_fold * channels,
                    ],
                    stream,
                )?)
        } else {
            Ok(hidden.clone())
        }
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        hidden: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.fold(hidden, stream)?;
        let range = self.parallel_input_range.as_ref().ok_or_else(|| {
            Exception::custom("Inkling vision TP layer is missing planner-derived input geometry")
        })?;
        if range.end > hidden.dim(-1) {
            return Err(Exception::custom(format!(
                "Inkling vision input range {range:?} exceeds folded width {}",
                hidden.dim(-1)
            )));
        }
        let local = hidden.try_index_device((.., .., .., .., range.clone()), stream)?;
        let partial = self.projection.forward(&local, stream)?;
        let mut hidden = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(norm) = &mut self.layer_norm {
            hidden = norm.forward(&hidden, stream)?;
            hidden = nn::gelu(hidden, stream)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct VisionModel {
    pub(crate) text_hidden_size: i32,
    #[param]
    pub(crate) layers: Vec<VisionLayer>,
    #[param]
    pub(crate) final_norm: nn::RmsNorm,
}

impl VisionModel {
    pub(crate) fn new(
        args: &VisionArgs,
        dense_dtype: Dtype,
        stream: &Stream,
    ) -> Result<Self, Error> {
        if (
            args.temporal_patch_size,
            args.patch_size,
            args.num_hidden_layers,
            args.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling hMLP currently supports the released (temporal_patch_size=2, patch_size=40, n_layers=4, channels=3) tower, got ({}, {}, {}, {})",
                args.temporal_patch_size, args.patch_size, args.num_hidden_layers, args.num_channels
            )));
        }
        // `plan_out_scales` for the released tower selects reduction scales
        // [1, 25, 100, 1600, 3200].
        let specs = args.layer_specs();
        let mut layers = Vec::with_capacity(specs.len());
        for (index, (input_dim, output_dim, t_fold, hw_fold)) in specs.into_iter().enumerate() {
            layers.push(VisionLayer::new(
                (input_dim, output_dim, t_fold, hw_fold),
                index + 1 != specs.len(),
                args.rms_norm_eps,
                args.weight_quantization_for(&format!("visual.layers.{index}.projection.weight")),
                dense_dtype,
                stream,
            )?);
        }
        Ok(Self {
            text_hidden_size: args.text_hidden_size,
            layers,
            final_norm: nn::RmsNorm::unloaded(
                args.text_hidden_size,
                args.rms_norm_eps,
                dense_dtype,
                stream,
            )?,
        })
    }

    fn forward(&mut self, pixels: &Array, stream: &Stream) -> Result<Array, Exception> {
        if pixels.ndim() != 5 || pixels.shape()[1..] != [2, 40, 40, 3] {
            return Err(Exception::custom(format!(
                "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                pixels.shape()
            )));
        }
        let mut hidden = pixels.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, stream)?;
        }
        hidden = self.final_norm.forward(&hidden, stream)?;
        hidden
            .reshape(&[-1, self.text_hidden_size], stream)?
            .try_index_device(NewAxis, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
/// Inkling causal language model.
pub struct Model {
    pub args: ModelArgs,
    #[param]
    pub(crate) model: TextModel,
    #[param]
    audio: Option<AudioModel>,
    #[param]
    visual: Option<VisionModel>,
    #[param]
    pub(crate) lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        validate_args(&args)?;
        Ok(Self {
            model: TextModel::new(&args.text_config, stream)?,
            audio: args
                .audio_config
                .as_ref()
                .map(|config| AudioModel::new(config, args.text_config.weight_dtype(), stream))
                .transpose()?,
            visual: args
                .vision_config
                .as_ref()
                .map(|config| VisionModel::new(config, args.text_config.weight_dtype(), stream))
                .transpose()?,
            lm_head: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.text_config.hidden_size,
                args.text_config.vocab_size,
                false,
                args.text_config.weight_quantization_for("lm_head.weight"),
                args.text_config.weight_dtype(),
                stream,
            )?,
            args,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args.text_config)
    }

    pub fn new_paged_cache(&self, options: PagedCacheOptions) -> Result<Cache, Exception> {
        Cache::new_paged(&self.args.text_config, options, None)
    }

    pub(crate) fn save_prompt_cache(
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        let end = i64::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Inkling prompt length exceeds i64"))?;
        if cache.offset() as i64 != end {
            return Err(Exception::custom(
                "Inkling cache offset does not match the persisted prefix",
            ));
        }
        let paged = cache
            .layers
            .iter()
            .any(|layer| matches!(layer.kv, InklingKvCache::Paged(_)));
        if paged {
            if cache
                .layers
                .iter()
                .any(|layer| !matches!(layer.kv, InklingKvCache::Paged(_)))
            {
                return Err(Exception::custom(
                    "Inkling prompt cache cannot mix paged and resident attention layers",
                ));
            }
            for layer in &mut cache.layers {
                let InklingKvCache::Paged(attention) = &mut layer.kv else {
                    unreachable!("validated paged Inkling cache")
                };
                attention.finalize()?;
            }
        }
        let mut blocks = Vec::with_capacity(cache.layers.len());
        let mut state = Vec::with_capacity(cache.layers.len() * 4);
        for (layer, layer_cache) in cache.layers.iter().enumerate() {
            for (slot, convolution) in layer_cache.convolutions.iter().enumerate() {
                state.push(PromptCacheStateArray {
                    owner: StateTensorOwner::Layer(layer),
                    role: StateTensorRole::Convolution { slot: slot as u32 },
                    array: convolution
                        .state
                        .as_ref()
                        .ok_or_else(|| Exception::custom("Inkling convolution state is missing"))?,
                });
            }
            let (start, keys, values) = match &layer_cache.kv {
                InklingKvCache::Global(cache) => {
                    let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
                        Exception::custom("Inkling full-attention cache state is missing")
                    })?;
                    (0, keys, values)
                }
                InklingKvCache::Sliding(cache) => {
                    let arrays = cache.arrays().cloned().collect::<Vec<_>>();
                    if arrays.len() != 2 {
                        return Err(Exception::custom(
                            "Inkling sliding-attention cache state is missing",
                        ));
                    }
                    let retained = i64::from(arrays[0].dim(-2));
                    (end - retained, arrays[0].clone(), arrays[1].clone())
                }
                InklingKvCache::Paged(_) => continue,
            };
            blocks.push(PromptCacheSnapshotBlock {
                global_layer: layer,
                start,
                end,
                rank: None,
                arrays: CacheBlockArrays::KeyValue { keys, values },
            });
        }
        if paged {
            let manager = cache
                .layers
                .iter()
                .find_map(|layer| match &layer.kv {
                    InklingKvCache::Paged(cache) => Some(cache.manager()),
                    _ => None,
                })
                .expect("validated paged Inkling cache");
            manager
                .save_prompt_cache(destination, descriptor, prefix_token_ids, &state, options)
                .map_err(|error| Exception::custom(error.to_string()))
        } else {
            save_prompt_cache_snapshot(
                destination,
                descriptor,
                prefix_token_ids,
                blocks,
                &state,
                options,
            )
            .map_err(|error| Exception::custom(error.to_string()))
        }
    }

    #[cfg(test)]
    pub(crate) fn load_prompt_cache(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = usize::try_from(args.text_config.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Inkling cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "inkling".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: prompt_cache_layer_layout(args)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        Self::load_prompt_cache_with_identity(
            args,
            directory,
            expected,
            prefix_token_ids,
            &identity,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_identity(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let (blocks, states, manifest) =
            open_prompt_cache_snapshot(directory, expected, identity, prefix_token_ids, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut block_map = BTreeMap::<usize, Vec<_>>::new();
        for block in blocks {
            block_map.entry(block.global_layer).or_default().push(block);
        }
        let mut states = states
            .into_iter()
            .map(|state| ((state.owner, state.role), state.array))
            .collect::<BTreeMap<_, _>>();
        let mut cache = Cache::new(&args.text_config);
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Inkling prompt length exceeds i32"))?;
        for (layer, layer_cache) in cache.layers.iter_mut().enumerate() {
            let mut layer_blocks = block_map.remove(&layer).ok_or_else(|| {
                Exception::custom("Inkling attention prompt-cache block is missing")
            })?;
            layer_blocks.sort_by_key(|block| block.start);
            let expected_end = i64::from(end);
            if layer_blocks
                .last()
                .is_none_or(|block| block.end != expected_end)
            {
                return Err(Exception::custom(
                    "Inkling prompt-cache range ends at the wrong token",
                ));
            }
            let start = layer_blocks[0].start;
            let mut keys = Vec::with_capacity(layer_blocks.len());
            let mut values = Vec::with_capacity(layer_blocks.len());
            for block in layer_blocks {
                let CacheBlockArrays::KeyValue {
                    keys: block_keys,
                    values: block_values,
                } = block.arrays
                else {
                    return Err(Exception::custom("Inkling prompt-cache kind mismatch"));
                };
                keys.push(block_keys);
                values.push(block_values);
            }
            let keys = concatenate_axis(&keys, -2, stream)?;
            let values = concatenate_axis(&values, -2, stream)?;
            if start != expected_end - i64::from(keys.dim(-2)) {
                return Err(Exception::custom(
                    "Inkling prompt-cache retained range is inconsistent",
                ));
            }
            match &mut layer_cache.kv {
                InklingKvCache::Global(cache) => cache.restore_resident(keys, values, end)?,
                InklingKvCache::Sliding(cache) => cache.restore_resident(keys, values, end)?,
                InklingKvCache::Paged(_) => unreachable!("resident cache constructor used"),
            }
            for (slot, convolution) in layer_cache.convolutions.iter_mut().enumerate() {
                convolution.state = Some(
                    states
                        .remove(&(
                            StateTensorOwner::Layer(layer),
                            StateTensorRole::Convolution { slot: slot as u32 },
                        ))
                        .ok_or_else(|| Exception::custom("Inkling convolution state is missing"))?,
                );
                convolution.offset = end;
            }
        }
        if !block_map.is_empty() || !states.is_empty() {
            return Err(Exception::custom(
                "Inkling prompt cache has unexpected state",
            ));
        }
        Ok((cache, manifest))
    }

    pub(crate) fn new_paged_cache_with_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Cache, Exception> {
        Cache::new_paged_with_manager(&self.args.text_config, manager, rank)
    }

    pub(crate) fn forward_logits(
        &mut self,
        tokens: &Array,
        inputs_embeds: Option<&Array>,
        cache: Option<&mut Cache>,
        last_token_only: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if let Some(cache) = cache.as_ref() {
            cache
                .validate(&self.args.text_config.layer_schedule)
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        let hidden = self.model.forward(tokens, inputs_embeds, cache, stream)?;
        project_text_logits(
            &hidden,
            &self.args.text_config,
            last_token_only,
            stream,
            |hidden, stream| self.lm_head.forward(hidden, stream),
        )
    }

    pub(crate) fn forward_cached_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let hidden = self
            .model
            .forward_with_expert_executor(tokens, cache, execute, stream)?;
        project_text_logits(
            &hidden,
            &self.args.text_config,
            false,
            stream,
            |hidden, stream| self.lm_head.forward(hidden, stream),
        )
    }

    fn prepare_typed_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<input::PreparedPrefill, Exception> {
        let modality_tokens = [
            input::ModalityToken {
                modality: input::Modality::Image,
                token_id: self.args.image_token_id,
            },
            input::ModalityToken {
                modality: input::Modality::Audio,
                token_id: self.args.audio_token_id,
            },
        ];
        let embed_tokens = &mut self.model;
        let audio = &mut self.audio;
        let visual = &mut self.visual;
        input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.text_config.hidden_size,
            "Inkling",
            stream,
            |tokens, stream| embed_tokens.embed(tokens, stream),
            |part, stream| match (part.modality, part.payload) {
                (_, input::InputPayload::Embeddings(embeddings)) => Ok(vec![embeddings.clone()]),
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => Ok(vec![visual
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling image input requires vision_config and vision weights",
                        )
                    })?
                    .forward(pixels, stream)?]),
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => Ok(vec![audio
                    .as_mut()
                    .ok_or_else(|| {
                        Exception::custom(
                            "Inkling audio input requires audio_config and audio weights",
                        )
                    })?
                    .forward(ids, part.metadata.audio_mask, stream)?]),
                (modality, input::InputPayload::Tensor(_)) => Err(Exception::custom(format!(
                    "Inkling does not support {} tensor inputs",
                    modality.as_str()
                ))),
                (modality, input::InputPayload::TokenIds(_)) => Err(Exception::custom(format!(
                    "Inkling {} input does not accept token-id payloads",
                    modality.as_str()
                ))),
            },
        )
    }
}

impl CausalLm<Cache> for Model {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self.prepare_typed_prefill(input, stream)? {
            input::PreparedPrefill::Text(tokens) => {
                self.forward_logits(&tokens, None, Some(cache), true, stream)
            }
            input::PreparedPrefill::Embeddings { tokens, embeddings } => {
                self.forward_logits(&tokens, Some(&embeddings), Some(cache), true, stream)
            }
        }
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_logits(input_tokens, None, Some(cache), true, stream)
    }
}

/// Inkling token generation iterator.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

pub(crate) struct LoadedInklingGguf {
    pub(crate) model: Model,
}

pub(crate) struct PreparedInklingGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Optional sibling GGUF containing Inkling's hMLP and dMel towers.
pub(crate) struct InklingMmprojGguf {
    pub(crate) checkpoint: GgufCheckpoint,
    pub(crate) metadata: HashMap<String, GgufMetadataValue>,
}

pub(crate) fn open_sibling_mmproj(gguf_file: &Path) -> Result<Option<InklingMmprojGguf>, Error> {
    let Some(path) = crate::runtime::checkpoint::gguf::find_sibling_mmproj(gguf_file, "inkling")?
    else {
        return Ok(None);
    };
    let checkpoint = GgufCheckpoint::open(path)?;
    let metadata = gguf_metadata(&checkpoint);
    validate_mmproj_metadata(&metadata)?;
    Ok(Some(InklingMmprojGguf {
        checkpoint,
        metadata,
    }))
}

/// Loads an `inkling` GGUF and its optional sibling multimodal projector.
pub fn load_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let gguf_file = gguf_file.as_ref();
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    let mmproj = open_sibling_mmproj(gguf_file)?;
    Ok(load_gguf_checkpoint_with_mmproj(
        &checkpoint,
        metadata,
        mmproj.as_ref(),
        stream,
        weights_stream,
    )?
    .model)
}

/// Loads an Inkling text GGUF with an explicit combined audio/vision mmproj.
pub fn load_gguf_with_mmproj(
    gguf_file: impl AsRef<Path>,
    mmproj_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    let mmproj_checkpoint = GgufCheckpoint::open(mmproj_file)?;
    let mmproj = InklingMmprojGguf {
        metadata: gguf_metadata(&mmproj_checkpoint),
        checkpoint: mmproj_checkpoint,
    };
    validate_mmproj_metadata(&mmproj.metadata)?;
    Ok(load_gguf_checkpoint_with_mmproj(
        &checkpoint,
        metadata,
        Some(&mmproj),
        stream,
        weights_stream,
    )?
    .model)
}

pub(crate) fn load_gguf_checkpoint_with_mmproj(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    mmproj: Option<&InklingMmprojGguf>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedInklingGguf, Error> {
    let prepared = prepare_gguf_checkpoint_with_mmproj(checkpoint, &metadata, mmproj)?;
    let mut model = Model::new(prepared.args, stream)?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    let mut materializer = checkpoint.materializer();
    for tensor in checkpoint.catalog().tensors() {
        let physical = &tensor.descriptor().name;
        if physical.contains("ffn_gate_exps")
            || physical.contains("ffn_up_exps")
            || physical.contains("ffn_gate_shexp")
            || physical.contains("ffn_up_shexp")
        {
            continue;
        }
        for (name, mut value) in materializer.converted_tensor(physical)?.into_arrays() {
            let mut translated = translate_gguf_weight_name(&name);
            if let Some(prefix) = translated.strip_suffix(".global_scale") {
                let layer = prefix
                    .strip_prefix("model.layers.")
                    .and_then(|value| value.parse::<i32>().ok())
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "invalid Inkling GGUF global-scale name {translated:?}"
                        ))
                    })?;
                let policy = model
                    .args
                    .text_config
                    .layer_policy(layer as usize)
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Inkling layer schedule has no layer {layer}"
                        ))
                    })?;
                translated = if policy.feed_forward == FeedForwardPolicy::Dense {
                    format!("model.layers.{layer}.dense_global_scale")
                } else {
                    format!("model.layers.{layer}.moe.router.global_scale")
                };
            }
            if translated.contains("_sconv.weight") && value.ndim() == 2 {
                value = value.reshape(&[value.dim(0), 1, value.dim(1)], weights_stream)?;
            }
            load_named_array_strict(&mut model, translated, value, None, &config, &mut report)?;
        }
    }
    let sparse_layers = model
        .args
        .text_config
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| {
            (policy.feed_forward == FeedForwardPolicy::SparseMoe).then_some(layer)
        })
        .collect::<Vec<_>>();
    for layer in sparse_layers {
        let source = format!("blk.{layer}");
        for (source_gate, source_up, target) in [
            ("ffn_gate_exps", "ffn_up_exps", "moe.experts.gate_up_proj"),
            (
                "ffn_gate_shexp",
                "ffn_up_shexp",
                "moe.shared_experts.gate_up_proj",
            ),
        ] {
            let gate = materializer
                .converted_tensor(&format!("{source}.{source_gate}.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let up = materializer
                .converted_tensor(&format!("{source}.{source_up}.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (source_suffix, target_suffix) in
                [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
            {
                let gate_name = format!("{source}.{source_gate}.{source_suffix}");
                let up_name = format!("{source}.{source_up}.{source_suffix}");
                let Some((gate, up)) = paired_inkling_gguf_component(
                    gate.get(&gate_name),
                    up.get(&up_name),
                    source_suffix == "weight",
                    &source,
                )?
                else {
                    continue;
                };
                load_named_array_strict(
                    &mut model,
                    format!("model.layers.{layer}.{target}{target_suffix}"),
                    concatenate_axis(&[gate.clone(), up.clone()], 1, weights_stream)?,
                    None,
                    &config,
                    &mut report,
                )?;
            }
        }
    }
    if let Some(mmproj) = mmproj {
        let mut materializer = mmproj.checkpoint.materializer();
        for tensor in mmproj.checkpoint.catalog().tensors() {
            let physical = &tensor.descriptor().name;
            for (name, value) in materializer.converted_tensor(physical)?.into_arrays() {
                load_named_array_strict(
                    &mut model,
                    translate_mmproj_weight_name(&name),
                    value,
                    None,
                    &config,
                    &mut report,
                )?;
            }
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(LoadedInklingGguf { model })
}

fn paired_inkling_gguf_component<'a, T>(
    gate: Option<&'a T>,
    up: Option<&'a T>,
    required: bool,
    source: &str,
) -> Result<Option<(&'a T, &'a T)>, Error> {
    match (gate, up) {
        (Some(gate), Some(up)) => Ok(Some((gate, up))),
        (None, None) if !required => Ok(None),
        _ => Err(Error::UnsupportedArchitecture(format!(
            "Inkling GGUF has incomplete gate/up tensors under {source}"
        ))),
    }
}

pub(crate) fn prepare_gguf_checkpoint_with_mmproj(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&InklingMmprojGguf>,
) -> Result<PreparedInklingGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if architecture != "inkling" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports inkling"
        )));
    }
    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::Inkling,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    if let Some(mmproj) = mmproj {
        crate::api::structural::validate_inkling_mmproj_gguf(metadata, mmproj)
            .into_loader_result()?;
    }
    let mut configs = gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    let mut args = args_from_gguf_catalog(metadata)?;
    for layer in args
        .text_config
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| {
            (policy.feed_forward == FeedForwardPolicy::SparseMoe).then_some(layer)
        })
    {
        for prefix in [
            format!("model.layers.{layer}.moe.experts"),
            format!("model.layers.{layer}.moe.shared_experts"),
        ] {
            let gate = format!("{prefix}.gate_proj");
            let up = format!("{prefix}.up_proj");
            match (configs.remove(&gate), configs.remove(&up)) {
                (Some(gate), Some(up)) if gate == up => {
                    configs.insert(format!("{prefix}.gate_up_proj"), gate);
                }
                (None, None) => {}
                (gate, up) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling GGUF gate/up formats differ under {prefix}: {gate:?} vs {up:?}"
                    )))
                }
            }
        }
    }
    args.text_config.quantized_weight_configs = Some(configs);
    if let Some(mmproj) = mmproj {
        apply_mmproj_args(&mut args, metadata, mmproj)?;
    }
    validate_args(&args)?;
    Ok(PreparedInklingGguf {
        args,
        eos_token_ids: crate::api::gguf_eos_token_ids(metadata)?,
    })
}

pub(crate) fn validate_mmproj_metadata(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    let vision_projector = gguf_string(metadata, "clip.vision.projector_type")?;
    let audio_projector = gguf_string(metadata, "clip.audio.projector_type")?;
    for (key, description) in [
        ("clip.has_vision_encoder", "vision encoder"),
        ("clip.has_audio_encoder", "audio encoder"),
    ] {
        match metadata.get(key) {
            Some(GgufMetadataValue::Bool(true)) => {}
            Some(GgufMetadataValue::Bool(false)) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling mmproj does not contain its {description}"
                )))
            }
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling mmproj metadata key {key:?} must be boolean"
                )))
            }
            None => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling mmproj is missing metadata key {key:?}"
                )))
            }
        }
    }
    if architecture != "clip" || vision_projector != "inkling" || audio_projector != "inkling" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected an Inkling audio/vision mmproj, got architecture {architecture:?}, vision projector {vision_projector:?}, and audio projector {audio_projector:?}"
        )));
    }
    Ok(())
}

pub(crate) fn apply_mmproj_args(
    args: &mut ModelArgs,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &InklingMmprojGguf,
) -> Result<(), Error> {
    validate_mmproj_metadata(&mmproj.metadata)?;
    mmproj
        .checkpoint
        .catalog()
        .translated_outputs(translate_mmproj_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let configs = gguf_quantization_configs(&mmproj.checkpoint, translate_mmproj_weight_name)?;
    let audio_configs = configs
        .iter()
        .filter(|(name, _)| name.starts_with("audio."))
        .map(|(name, config)| (name.clone(), *config))
        .collect();
    let vision_configs = configs
        .iter()
        .filter(|(name, _)| name.starts_with("visual."))
        .map(|(name, config)| (name.clone(), *config))
        .collect();
    if configs
        .keys()
        .any(|name| !name.starts_with("audio.") && !name.starts_with("visual."))
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling mmproj contains a quantized tensor outside its audio and vision towers".into(),
        ));
    }

    let vision_hidden = gguf_i32_catalog(&mmproj.metadata, "clip.vision.projection_dim")?;
    let audio_hidden = gguf_i32_catalog(&mmproj.metadata, "clip.audio.projection_dim")?;
    let audio_embedding = gguf_i32_catalog(&mmproj.metadata, "clip.audio.embedding_length")?;
    if vision_hidden != args.text_config.hidden_size
        || audio_hidden != args.text_config.hidden_size
        || audio_embedding != args.text_config.hidden_size
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling mmproj output widths ({vision_hidden}, {audio_hidden}, {audio_embedding}) do not match decoder width {}",
            args.text_config.hidden_size
        )));
    }
    let patch_size = gguf_i32_catalog(&mmproj.metadata, "clip.vision.patch_size")?;
    let image_size = gguf_i32_catalog(&mmproj.metadata, "clip.vision.image_size")?;
    if image_size != patch_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling mmproj image_size {image_size} does not match patch_size {patch_size}"
        )));
    }
    let vision_eps =
        gguf_optional_f32(&mmproj.metadata, "clip.vision.attention.layer_norm_epsilon")?
            .unwrap_or(default_rms_norm_eps());
    let audio_eps = gguf_optional_f32(&mmproj.metadata, "clip.audio.attention.layer_norm_epsilon")?
        .unwrap_or(default_rms_norm_eps());
    args.vision_config = Some(VisionArgs {
        vision_encoder_type: "hmlp".into(),
        text_hidden_size: vision_hidden,
        patch_size,
        temporal_patch_size: 2,
        num_channels: gguf_i32_catalog(&mmproj.metadata, "clip.vision.embedding_length")?,
        num_hidden_layers: gguf_i32_catalog(&mmproj.metadata, "clip.vision.block_count")?,
        use_vision_norm: true,
        rms_norm_eps: vision_eps,
        quantized_weight_configs: Some(vision_configs),
    });
    args.audio_config = Some(AudioArgs {
        text_hidden_size: audio_hidden,
        num_codebooks: gguf_i32_catalog(&mmproj.metadata, "clip.audio.num_mel_bins")?,
        codebook_size: 16,
        bias: false,
        use_audio_norm: true,
        audio_mode: "dmel".into(),
        rms_norm_eps: audio_eps,
        quantized_weight_configs: Some(audio_configs),
    });
    // The released GGUF contract does not carry dedicated placeholder IDs;
    // they are reserved padded-vocabulary slots in the text checkpoint.
    if let Some(id) = gguf_optional_i32(model_metadata, "inkling.audio_token_id")? {
        args.audio_token_id = u32::try_from(id).map_err(|_| {
            Error::UnsupportedArchitecture("Inkling audio placeholder id is negative".into())
        })?;
    }
    if let Some(id) = gguf_optional_i32(model_metadata, "inkling.image_token_id")? {
        args.image_token_id = u32::try_from(id).map_err(|_| {
            Error::UnsupportedArchitecture("Inkling image placeholder id is negative".into())
        })?;
    }
    let vocab_size = args.text_config.vocab_size;
    for (name, id) in [
        ("audio", args.audio_token_id),
        ("image", args.image_token_id),
    ] {
        if id >= vocab_size as u32 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling {name} placeholder id {id} exceeds GGUF vocabulary size {vocab_size}"
            )));
        }
    }
    validate_args(args)?;
    Ok(())
}

pub(crate) fn translate_mmproj_weight_name(name: &str) -> String {
    for (source, target) in [
        ("a.dmel.embedding", "audio.encoder"),
        ("a.dmel.final_norm", "audio.final_norm"),
        ("v.hmlp.final_norm", "visual.final_norm"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    if let Some(rest) = name.strip_prefix("v.hmlp.") {
        if let Some((layer, parameter)) = rest.split_once('.') {
            if layer.parse::<usize>().is_ok() {
                let parameter =
                    parameter
                        .replacen("linear", "projection", 1)
                        .replacen("norm", "layer_norm", 1);
                return format!("visual.layers.{layer}.{parameter}");
            }
        }
    }
    name.to_string()
}

pub(crate) fn args_from_gguf_catalog(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let key = |suffix: &str| format!("inkling.{suffix}");
    let layers = gguf_i32_catalog(metadata, &key("block_count"))?;
    if layers <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling block_count must be positive, got {layers}"
        )));
    }
    let pattern = gguf_bool_pattern(metadata, &key("attention.sliding_window_pattern"), layers)?;
    let kv_values = metadata
        .get(&key("attention.head_count_kv"))
        .and_then(GgufMetadataValue::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Inkling GGUF is missing attention.head_count_kv".into())
        })?;
    let kv_values = if kv_values.len() == 1 {
        vec![kv_values[0]; layers as usize]
    } else {
        kv_values
    };
    if kv_values.len() != layers as usize {
        return Err(Error::UnsupportedArchitecture(
            "Inkling GGUF attention.head_count_kv length does not match block_count".into(),
        ));
    }
    if kv_values.iter().any(|value| *value <= 0) {
        return Err(Error::UnsupportedArchitecture(
            "Inkling GGUF attention.head_count_kv values must be positive".into(),
        ));
    }
    let global_kv = kv_values
        .iter()
        .zip(&pattern)
        .find_map(|(value, local)| (!local).then_some(*value))
        .unwrap_or(kv_values[0]);
    let local_kv = kv_values
        .iter()
        .zip(&pattern)
        .find_map(|(value, local)| local.then_some(*value))
        .unwrap_or(global_kv);
    if kv_values
        .iter()
        .zip(&pattern)
        .any(|(value, local)| *value != if *local { local_kv } else { global_kv })
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling GGUF attention.head_count_kv must be uniform within each attention policy"
                .into(),
        ));
    }
    let hidden_size = gguf_i32_catalog(metadata, &key("embedding_length"))?;
    let heads = gguf_i32_catalog(metadata, &key("attention.head_count"))?;
    if hidden_size <= 0 || heads <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "Inkling GGUF embedding length and attention head count must be positive".into(),
        ));
    }
    let head_dim =
        gguf_optional_i32(metadata, &key("attention.key_length"))?.unwrap_or(hidden_size / heads);
    let vocab_size = gguf_vocab_size(metadata, &key("vocab_size"))?;
    let sliding_window = gguf_i32_catalog(metadata, &key("attention.sliding_window"))?;
    if sliding_window <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling GGUF sliding window must be positive, got {sliding_window}"
        )));
    }
    let sliding = AttentionPolicy::sliding(sliding_window as u32)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let dense_layers = gguf_i32_catalog(metadata, &key("dense_block_count"))?;
    if !(0..=layers).contains(&dense_layers) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling GGUF dense block count {dense_layers} is outside 0..={layers}"
        )));
    }
    let layer_schedule = LayerSchedule::new(
        layers as usize,
        pattern
            .iter()
            .enumerate()
            .map(|(layer, local)| LayerPolicy {
                attention: if *local {
                    sliding
                } else {
                    AttentionPolicy::Full
                },
                feed_forward: if layer < dense_layers as usize {
                    FeedForwardPolicy::Dense
                } else {
                    FeedForwardPolicy::SparseMoe
                },
            })
            .collect(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let args = ModelArgs {
        model_type: "inkling_mm_model".into(),
        mtp_config: None,
        text_config: TextArgs {
            torch_dtype: None,
            hidden_size,
            num_hidden_layers: layers,
            vocab_size,
            num_attention_heads: heads,
            num_key_value_heads: i32::try_from(global_kv).map_err(|_| {
                Error::UnsupportedArchitecture("Inkling global KV heads exceed i32".into())
            })?,
            head_dim,
            swa_num_attention_heads: Some(heads),
            swa_num_key_value_heads: Some(i32::try_from(local_kv).map_err(|_| {
                Error::UnsupportedArchitecture("Inkling local KV heads exceed i32".into())
            })?),
            swa_head_dim: Some(head_dim),
            layer_schedule,
            sconv_kernel_size: gguf_i32_catalog(metadata, &key("shortconv_kernel"))?,
            use_sconv: true,
            rel_extent: gguf_i32_catalog(metadata, &key("rel_extent"))?,
            d_rel: gguf_i32_catalog(metadata, &key("d_rel"))?,
            log_scaling_n_floor: gguf_optional_i32(metadata, &key("log_scaling_n_floor"))?
                .filter(|value| *value > 0),
            log_scaling_alpha: gguf_optional_f32(metadata, &key("log_scaling_alpha"))?
                .unwrap_or(0.0),
            rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
            use_embed_norm: true,
            unpadded_vocab_size: gguf_optional_i32(metadata, &key("unpadded_vocab_size"))?,
            logits_mup_width_multiplier: gguf_optional_f32(metadata, &key("logit_scale_denom"))?
                .unwrap_or(1.0),
            final_logit_softcapping: None,
            intermediate_size: gguf_i32_catalog(metadata, &key("expert_feed_forward_length"))?,
            dense_intermediate_size: Some(gguf_i32_catalog(metadata, &key("feed_forward_length"))?),
            moe_intermediate_size: None,
            n_routed_experts: gguf_i32_catalog(metadata, &key("expert_count"))?,
            num_experts_per_tok: gguf_i32_catalog(metadata, &key("expert_used_count"))?,
            n_shared_experts: gguf_i32_catalog(metadata, &key("expert_shared_count"))?,
            route_scale: gguf_optional_f32(metadata, &key("expert_weights_scale"))?.unwrap_or(1.0),
            shared_expert_sink: true,
            use_gate_bias: true,
            norm_after_topk: true,
            use_global_scale: true,
            gate_activation: "sigmoid".into(),
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            q_bias: false,
            o_bias: false,
            model_max_length: Some(gguf_i32_catalog(metadata, &key("context_length"))?),
            weight_quantization: None,
            quantized_weight_configs: None,
        },
        audio_config: None,
        vision_config: None,
        image_token_id: default_image_token_id(),
        audio_token_id: default_audio_token_id(),
        eos_token_id: crate::api::gguf_eos_token_ids(metadata)?.first().copied(),
    };
    validate_args(&args)?;
    Ok(args)
}

fn gguf_bool_pattern(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    layers: i32,
) -> Result<Vec<bool>, Error> {
    let values = match metadata.get(key) {
        Some(GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(values))) => {
            values.clone()
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling GGUF metadata key {key:?} must be a bool array"
            )))
        }
        None => (0..layers).map(|layer| (layer + 1) % 6 != 0).collect(),
    };
    if values.len() != layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling GGUF {key:?} has {} values for {layers} layers",
            values.len()
        )));
    }
    Ok(values)
}

fn gguf_vocab_size(
    metadata: &HashMap<String, GgufMetadataValue>,
    fallback: &str,
) -> Result<i32, Error> {
    match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        }),
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ))
        }
        None => gguf_i32_catalog(metadata, fallback),
    }
}

fn gguf_optional_i32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i32>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "GGUF metadata key {key:?} must be an i32 scalar"
                    ))
                })
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<f32, Error> {
    gguf_optional_f32(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_f32(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<f32>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value.as_f32().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be a numeric scalar"
                ))
            })
        })
        .transpose()
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embed_norm"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    for (source, target) in [
        ("ffn_gate_exps", "moe.experts.gate_proj"),
        ("ffn_up_exps", "moe.experts.up_proj"),
        ("ffn_down_exps", "moe.experts.down_proj"),
        ("ffn_gate_shexp", "moe.shared_experts.gate_proj"),
        ("ffn_up_shexp", "moe.shared_experts.up_proj"),
        ("ffn_down_shexp", "moe.shared_experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = parameter.strip_prefix(source).unwrap_or_default();
            let suffix = match suffix {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_r", "self_attn.r_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("shortconv_k", "self_attn.k_sconv"),
        ("shortconv_v", "self_attn.v_sconv"),
        ("shortconv_attn", "attn_sconv"),
        ("shortconv_mlp", "mlp_sconv"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "dense.gate_proj"),
        ("ffn_up", "dense.up_proj"),
        ("ffn_down", "dense.down_proj"),
        ("ffn_gate_inp", "moe.router.weight"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = parameter.replacen(source, target, 1);
            if source == "ffn_gate_inp" && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return format!("model.layers.{layer}.{translated}");
        }
    }
    if parameter == "attn_rel_proj.weight" || parameter == "attn_rel_proj" {
        return format!("model.layers.{layer}.self_attn.rel_proj");
    }
    if parameter == "ffn_gscale" || parameter == "ffn_gscale.weight" {
        return format!("model.layers.{layer}.global_scale");
    }
    if matches!(
        parameter,
        "exp_probs_b.bias" | "ffn_exp_probs_b.bias" | "ffn_exp_probs_b"
    ) {
        return format!("model.layers.{layer}.moe.router.bias");
    }
    name.to_string()
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let value: Value =
        serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?;
    model_args_from_config_value(&value)
}

/// Returns a deterministic cache-relevant identity including the complete
/// ordered Inkling layer schedule.
pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let text = &args.text_config;
    let schedule = text
        .layer_schedule
        .iter()
        .map(|policy| {
            let attention = match policy.attention {
                AttentionPolicy::Full => "f".to_string(),
                AttentionPolicy::Sliding { window } => format!("s{}", window.get()),
            };
            let feed_forward = match policy.feed_forward {
                FeedForwardPolicy::Dense => "d",
                FeedForwardPolicy::SparseMoe => "m",
            };
            format!("{attention}{feed_forward}")
        })
        .collect::<Vec<_>>()
        .join(",");
    derive_prompt_cache_architecture_fingerprint(
        "inkling",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", text.hidden_size.to_string()),
            ("num_hidden_layers", text.num_hidden_layers.to_string()),
            ("num_attention_heads", text.num_attention_heads.to_string()),
            ("num_key_value_heads", text.num_key_value_heads.to_string()),
            ("head_dim", text.head_dim.to_string()),
            (
                "swa_num_attention_heads",
                format!("{:?}", text.swa_num_attention_heads),
            ),
            (
                "swa_num_key_value_heads",
                format!("{:?}", text.swa_num_key_value_heads),
            ),
            ("swa_head_dim", format!("{:?}", text.swa_head_dim)),
            ("layer_schedule", schedule),
            ("sconv_kernel_size", text.sconv_kernel_size.to_string()),
            ("d_rel", text.d_rel.to_string()),
            ("rel_extent", text.rel_extent.to_string()),
            ("n_routed_experts", text.n_routed_experts.to_string()),
            ("n_shared_experts", text.n_shared_experts.to_string()),
            ("num_experts_per_tok", text.num_experts_per_tok.to_string()),
            ("image_token_id", args.image_token_id.to_string()),
            ("audio_token_id", args.audio_token_id.to_string()),
        ],
    )
}

fn inkling_layer_schedule(source: &TextArgsSource) -> Result<LayerSchedule<LayerPolicy>, Error> {
    let layers = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Inkling num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Inkling num_hidden_layers must be positive, got 0".into(),
        ));
    }
    if source.sliding_window_size <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling sliding_window_size must be positive, got {}",
            source.sliding_window_size
        )));
    }
    let window = u32::try_from(source.sliding_window_size).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Inkling sliding_window_size exceeds the executable u32 range: {}",
            source.sliding_window_size
        ))
    })?;
    if window > i32::MAX as u32 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling sliding_window_size exceeds the executable i32 range: {window}"
        )));
    }
    let sliding = AttentionPolicy::sliding(window)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;

    let from_ids = source
        .local_layer_ids
        .as_ref()
        .map(|ids| {
            let mut pattern = vec![false; layers];
            for id in ids {
                let layer = usize::try_from(*id).map_err(|_| {
                    Error::UnsupportedArchitecture(format!(
                        "Inkling local_layer_ids contains negative layer {id}"
                    ))
                })?;
                if layer >= layers {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling local_layer_ids contains layer {layer} for {layers} layers"
                    )));
                }
                if std::mem::replace(&mut pattern[layer], true) {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling local_layer_ids contains duplicate layer {layer}"
                    )));
                }
            }
            Ok(pattern)
        })
        .transpose()?;
    let from_types = source
        .layer_types
        .as_ref()
        .map(|types| {
            if types.len() != layers {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling layer_types has {} entries for {layers} layers",
                    types.len()
                )));
            }
            types
                .iter()
                .enumerate()
                .map(|(layer, kind)| match kind.as_str() {
                    "sliding_attention" => Ok(true),
                    "full_attention" => Ok(false),
                    _ => Err(Error::UnsupportedArchitecture(format!(
                        "Inkling layer_types[{layer}] must be sliding_attention or full_attention, got {kind:?}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    if let (Some(ids), Some(types)) = (&from_ids, &from_types) {
        if ids != types {
            return Err(Error::UnsupportedArchitecture(
                "Inkling local_layer_ids conflicts with layer_types".into(),
            ));
        }
    }
    let attention = from_ids
        .or(from_types)
        .unwrap_or_else(|| (0..layers).map(|layer| (layer + 1) % 6 != 0).collect());

    let from_types = source
        .mlp_layer_types
        .as_ref()
        .map(|types| {
            if types.len() != layers {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling mlp_layer_types has {} entries for {layers} layers",
                    types.len()
                )));
            }
            types
                .iter()
                .enumerate()
                .map(|(layer, kind)| match kind.as_str() {
                    "dense" => Ok(FeedForwardPolicy::Dense),
                    "moe" => Ok(FeedForwardPolicy::SparseMoe),
                    _ => Err(Error::UnsupportedArchitecture(format!(
                        "Inkling mlp_layer_types[{layer}] must be dense or moe, got {kind:?}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let from_threshold = source
        .dense_mlp_idx
        .map(|threshold| {
            let threshold = usize::try_from(threshold).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling dense_mlp_idx must be nonnegative, got {threshold}"
                ))
            })?;
            if threshold > layers {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling dense_mlp_idx {threshold} exceeds {layers} layers"
                )));
            }
            Ok((0..layers)
                .map(|layer| {
                    if layer < threshold {
                        FeedForwardPolicy::Dense
                    } else {
                        FeedForwardPolicy::SparseMoe
                    }
                })
                .collect::<Vec<_>>())
        })
        .transpose()?;
    if let (Some(types), Some(threshold)) = (&from_types, &from_threshold) {
        if types != threshold {
            return Err(Error::UnsupportedArchitecture(
                "Inkling dense_mlp_idx conflicts with mlp_layer_types".into(),
            ));
        }
    }
    let feed_forward = from_types
        .or(from_threshold)
        .unwrap_or_else(|| vec![FeedForwardPolicy::SparseMoe; layers]);
    LayerSchedule::new(
        layers,
        attention
            .into_iter()
            .zip(feed_forward)
            .map(|(local, feed_forward)| LayerPolicy {
                attention: if local {
                    sliding
                } else {
                    AttentionPolicy::Full
                },
                feed_forward,
            })
            .collect(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

impl TextArgsSource {
    fn into_args(self, layer_schedule: LayerSchedule<LayerPolicy>) -> TextArgs {
        TextArgs {
            torch_dtype: self.torch_dtype,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            vocab_size: self.vocab_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            swa_num_attention_heads: self.swa_num_attention_heads,
            swa_num_key_value_heads: self.swa_num_key_value_heads,
            swa_head_dim: self.swa_head_dim,
            layer_schedule,
            sconv_kernel_size: self.sconv_kernel_size,
            use_sconv: self.use_sconv,
            rel_extent: self.rel_extent,
            d_rel: self.d_rel,
            log_scaling_n_floor: self.log_scaling_n_floor,
            log_scaling_alpha: self.log_scaling_alpha,
            rms_norm_eps: self.rms_norm_eps,
            use_embed_norm: self.use_embed_norm,
            unpadded_vocab_size: self.unpadded_vocab_size,
            logits_mup_width_multiplier: self.logits_mup_width_multiplier,
            final_logit_softcapping: self.final_logit_softcapping,
            intermediate_size: self.intermediate_size,
            dense_intermediate_size: self.dense_intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            n_routed_experts: self.n_routed_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            n_shared_experts: self.n_shared_experts,
            route_scale: self.route_scale,
            shared_expert_sink: self.shared_expert_sink,
            use_gate_bias: self.use_gate_bias,
            norm_after_topk: self.norm_after_topk,
            use_global_scale: self.use_global_scale,
            gate_activation: self.gate_activation,
            hidden_act: self.hidden_act,
            attention_dropout: self.attention_dropout,
            q_bias: self.q_bias,
            o_bias: self.o_bias,
            model_max_length: self.model_max_length,
            weight_quantization: None,
            quantized_weight_configs: None,
        }
    }
}

/// Parses the same validated configuration used by loading without creating
/// Inkling's text or media module trees.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, Error> {
    let source: ModelArgsSource = serde_json::from_value(value.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Inkling config: {error}"))
    })?;
    let layer_schedule = inkling_layer_schedule(&source.text_config)?;
    let args = ModelArgs {
        model_type: source.model_type,
        text_config: source.text_config.into_args(layer_schedule),
        mtp_config: source.mtp_config,
        audio_config: source.audio_config,
        vision_config: source.vision_config,
        image_token_id: source.image_token_id,
        audio_token_id: source.audio_token_id,
        eos_token_id: source.eos_token_id,
    };
    validate_args(&args)?;
    Ok(args)
}

pub fn validate_model_config_value(value: &Value) -> Result<(), Error> {
    model_args_from_config_value(value).map(|_| ())
}

fn validate_args(args: &ModelArgs) -> Result<(), Error> {
    let text = &args.text_config;
    if let Some(mtp) = &args.mtp_config {
        if mtp.num_nextn_predict_layers < 0 {
            return Err(Error::UnsupportedArchitecture(
                "Inkling MTP layer count cannot be negative".into(),
            ));
        }
        let count = mtp.num_nextn_predict_layers as usize;
        if let Some(layer) = mtp.local_layer_ids.iter().find(|&&layer| layer >= count) {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling MTP local layer id {layer} is outside 0..{count}"
            )));
        }
    }
    if args.model_type != "inkling_mm_model" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected Inkling model_type inkling_mm_model, got {:?}",
            args.model_type
        )));
    }
    if let Some(torch_dtype) = &text.torch_dtype {
        if !matches!(
            torch_dtype.as_str(),
            "bfloat16" | "bf16" | "float16" | "float32"
        ) {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Inkling torch_dtype {torch_dtype:?}"
            )));
        }
    }
    for (name, value) in [
        ("hidden_size", text.hidden_size),
        ("num_hidden_layers", text.num_hidden_layers),
        ("vocab_size", text.vocab_size),
        ("num_attention_heads", text.num_attention_heads),
        ("num_key_value_heads", text.num_key_value_heads),
        ("head_dim", text.head_dim),
        ("d_rel", text.d_rel),
        ("rel_extent", text.rel_extent),
        ("sconv_kernel_size", text.sconv_kernel_size),
        ("n_routed_experts", text.n_routed_experts),
        ("num_experts_per_tok", text.num_experts_per_tok),
        ("n_shared_experts", text.n_shared_experts),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling {name} must be positive, got {value}"
            )));
        }
    }
    if !text.use_sconv
        || !text.use_embed_norm
        || !text.shared_expert_sink
        || !text.use_gate_bias
        || !text.norm_after_topk
        || !text.use_global_scale
        || text.gate_activation != "sigmoid"
        || text.hidden_act != "silu"
        || text.attention_dropout != 0.0
        || text.q_bias
        || text.o_bias
        || text
            .final_logit_softcapping
            .is_some_and(|value| value != 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling config uses an unsupported attention, convolution, routing, or logit variant"
                .into(),
        ));
    }
    for local in [false, true] {
        let query_heads = text.q_heads(local);
        let kv_heads = text.kv_heads(local);
        let head_dim = text.attention_head_dim(local);
        if query_heads <= 0
            || kv_heads <= 0
            || head_dim <= 0
            || query_heads % kv_heads != 0
            || (local && head_dim != text.head_dim)
            || query_heads.checked_mul(head_dim).is_none()
            || kv_heads.checked_mul(head_dim).is_none()
            || query_heads.checked_mul(text.d_rel).is_none()
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling attention head configuration is inconsistent or overflows i32".into(),
            ));
        }
    }
    if text.dense_intermediate_size() <= 0 || text.moe_intermediate_size() <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "Inkling dense and MoE intermediate sizes must be positive".into(),
        ));
    }
    if text
        .n_routed_experts
        .checked_add(text.n_shared_experts)
        .is_none()
        || text.moe_intermediate_size().checked_mul(2).is_none()
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling expert geometry overflows i32".into(),
        ));
    }
    if text.num_experts_per_tok > text.n_routed_experts
        || text.layer_schedule.len() != text.num_hidden_layers as usize
    {
        return Err(Error::UnsupportedArchitecture(
            "Inkling layer schedule or expert top-k configuration is inconsistent".into(),
        ));
    }
    for window in text
        .layer_schedule
        .iter()
        .filter_map(|policy| policy.attention.window())
    {
        if window.get() > i32::MAX as u32 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling sliding attention window exceeds the executable i32 range: {}",
                window.get()
            )));
        }
    }
    if let Some(audio) = &args.audio_config {
        if audio.text_hidden_size != text.hidden_size
            || audio.num_codebooks <= 0
            || audio.codebook_size <= 0
            || audio.bias
            || !audio.use_audio_norm
            || audio.audio_mode != "dmel"
            || audio
                .num_codebooks
                .checked_mul(audio.codebook_size)
                .is_none()
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling audio configuration is inconsistent with the text decoder".into(),
            ));
        }
    }
    if let Some(vision) = &args.vision_config {
        if vision.text_hidden_size != text.hidden_size
            || vision.vision_encoder_type != "hmlp"
            || !vision.use_vision_norm
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision hidden size does not match the text decoder".into(),
            ));
        }
        if (
            vision.temporal_patch_size,
            vision.patch_size,
            vision.num_hidden_layers,
            vision.num_channels,
        ) != (2, 40, 4, 3)
        {
            return Err(Error::UnsupportedArchitecture(
                "Inkling vision configuration is not the released 4-layer 2x40x40 hMLP tower"
                    .into(),
            ));
        }
    }
    Ok(())
}

pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Inkling,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let args = get_model_args(model_dir)?;
    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("model.mtp.");
    let mut report = StrictLoadReport::default();
    let mut params = model.parameters_mut().flatten();
    for file in safetensors_files(model_dir)? {
        for_each_safetensor_array(file, weights_stream, |key, value| {
            for (key, value) in transform_weight(key, value, stream)? {
                load_array_strict(&mut params, key, value, &config, &mut report);
            }
            Ok(())
        })?;
    }
    drop(params);
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

pub(crate) fn transform_weight(
    key: String,
    mut value: Array,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    if key.ends_with("_sconv.weight") {
        value = value.as_dtype(Dtype::Float32, stream)?;
    }
    if let Some(suffix) = key.strip_prefix("model.audio.") {
        return Ok(vec![(format!("audio.{suffix}"), value)]);
    }
    if let Some(suffix) = key.strip_prefix("model.visual.") {
        let mut suffix = suffix.to_string();
        for layer in 0..4 {
            suffix = suffix
                .replace(
                    &format!("layers.linear_{layer}.weight"),
                    &format!("layers.{layer}.projection.weight"),
                )
                .replace(
                    &format!("layers.norm_{layer}.weight"),
                    &format!("layers.{layer}.layer_norm.weight"),
                );
        }
        return Ok(vec![(format!("visual.{suffix}"), value)]);
    }
    if !key.starts_with("model.llm.") {
        return Ok(vec![(key, value)]);
    }
    let mut key = key.replacen("model.llm.", "model.", 1);
    key = key
        .replace("model.embed.weight", "model.embed_tokens.weight")
        .replace("model.unembed.weight", "lm_head.weight")
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
        .replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
        .replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
        .replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
        .replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
        .replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
        .replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
        .replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
        .replace(".attn.rel_logits_proj.proj", ".self_attn.rel_proj")
        .replace(".attn.k_sconv.weight", ".self_attn.k_sconv.weight")
        .replace(".attn.v_sconv.weight", ".self_attn.v_sconv.weight")
        .replace(".mlp.w2_md.weight", ".dense.down_proj.weight")
        .replace(".mlp.global_scale", ".dense_global_scale")
        .replace(".mlp.gate.weight", ".moe.router.weight")
        .replace(".mlp.gate.bias", ".moe.router.bias")
        .replace(".mlp.gate.global_scale", ".moe.router.global_scale")
        .replace(".mlp.experts.w2_weight", ".moe.experts.down_proj")
        .replace(
            ".mlp.shared_experts.shared_w2_weight",
            ".moe.shared_experts.down_proj",
        );

    if key.ends_with(".mlp.w13_dn.weight") {
        let prefix = key.trim_end_matches(".mlp.w13_dn.weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![
            (format!("{prefix}.dense.gate_proj.weight"), gate),
            (format!("{prefix}.dense.up_proj.weight"), up),
        ]);
    }
    if key.ends_with(".mlp.experts.w13_weight") {
        let prefix = key.trim_end_matches(".mlp.experts.w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    if key.ends_with(".mlp.shared_experts.shared_w13_weight") {
        let prefix = key.trim_end_matches(".mlp.shared_experts.shared_w13_weight");
        let (gate, up) = deinterleave_w13(value, stream)?;
        return Ok(vec![(
            format!("{prefix}.moe.shared_experts.gate_up_proj"),
            concatenate_axis(&[gate, up], -2, stream)?,
        )]);
    }
    Ok(vec![(key, value)])
}

fn deinterleave_w13(value: Array, stream: &Stream) -> Result<(Array, Array), Error> {
    let shape = value.shape().to_vec();
    if shape.len() < 2 || shape[shape.len() - 2] % 2 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor has invalid shape {shape:?}"
        )));
    }
    let rows = shape[shape.len() - 2] / 2;
    let hidden = shape[shape.len() - 1];
    let (reshaped, gate_index, up_index) = if shape.len() == 2 {
        (
            value.reshape(&[rows, 2, hidden], stream)?,
            (.., 0, ..),
            (.., 1, ..),
        )
    } else if shape.len() == 3 {
        let experts = shape[0];
        let reshaped = value.reshape(&[experts, rows, 2, hidden], stream)?;
        let gate = reshaped.try_index_device((.., .., 0, ..), stream)?;
        let up = reshaped.try_index_device((.., .., 1, ..), stream)?;
        return Ok((gate, up));
    } else {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor rank {} is unsupported",
            shape.len()
        )));
    };
    Ok((
        reshaped.try_index_device(gate_index, stream)?,
        reshaped.try_index_device(up_index, stream)?,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safemlx::{
        module::ModuleParameters,
        ops::{GgufCheckpoint, GgufMetadataArray, GgufMetadataValue},
        Device, DeviceType, Dtype, ExecutionContext,
    };
    use serde_json::json;

    #[test]
    fn prompt_cache_layout_records_four_convolutions_and_attention_order() {
        use crate::{
            core::cache::{LayerCachePolicy, StateTensorRole},
            runtime::attention::AttentionPolicy,
        };

        let args = super::args_from_gguf_catalog(&tiny_gguf_metadata()).unwrap();
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        assert_eq!(layout.len(), 2);
        for (layer, policy) in layout.iter().enumerate() {
            let LayerCachePolicy::KeyValueWithFixedState {
                attention, tensors, ..
            } = policy
            else {
                panic!("unexpected Inkling policy {policy:?}");
            };
            assert_eq!(tensors.len(), 4);
            assert_eq!(tensors[3].role, StateTensorRole::Convolution { slot: 3 });
            assert_eq!(
                matches!(attention, AttentionPolicy::Sliding { .. }),
                args.text_config
                    .layer_policy(layer)
                    .expect("validated Inkling layer")
                    .attention
                    .window()
                    .is_some()
            );
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn schema_v4_paged_multimodal_convolution_state_save_reload_parity() {
        use crate::runtime::cache::{
            residency::{
                PagedCacheOptions, PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology,
            },
            KeyValueCache,
        };

        let args = super::args_from_gguf_catalog(&tiny_gguf_metadata()).unwrap();
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut cache = super::Cache::new_paged(
            &args.text_config,
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true)
                .with_persistence_retention(true),
            None,
        )
        .unwrap();
        let prefix_ids = [1_u32, 2, 3, 4];
        for (layer, layer_cache) in cache.layers.iter_mut().enumerate() {
            let local = args
                .text_config
                .layer_policy(layer)
                .expect("validated Inkling layer")
                .attention
                .window()
                .is_some();
            let kv_heads = args.text_config.kv_heads(local);
            let head_dim = args.text_config.attention_head_dim(local);
            let keys = safemlx::Array::zeros::<f32>(&[1, kv_heads, 4, head_dim], stream).unwrap();
            layer_cache
                .kv
                .update_and_fetch(keys.clone(), keys, stream)
                .unwrap();
            let widths = [
                kv_heads * head_dim,
                kv_heads * head_dim,
                args.text_config.hidden_size,
                args.text_config.hidden_size,
            ];
            for (convolution, width) in layer_cache.convolutions.iter_mut().zip(widths) {
                convolution.state = Some(
                    safemlx::Array::zeros::<f32>(
                        &[1, args.text_config.sconv_kernel_size - 1, width],
                        stream,
                    )
                    .unwrap(),
                );
                convolution.offset = 4;
            }
        }
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "inkling".into(),
            effective_model_type: args.model_type.clone(),
            checkpoint_fingerprint: "zero-fixture".into(),
            prefix_content_fingerprint: "audio-video:fixture;tokens:1,2,3,4".into(),
            architecture_fingerprint: super::prompt_cache_architecture_fingerprint(&args),
            layer_count: layout.len(),
            global_layer_start: 0,
            global_layer_end: layout.len(),
            batch_size: 1,
            layer_prefix_offsets: vec![0; layout.len()],
            layer_layout: layout,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        super::Model::save_prompt_cache(
            &mut cache,
            &destination,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            stream,
        )
        .unwrap();
        let (restored, _) =
            super::Model::load_prompt_cache(&args, &destination, &descriptor, &prefix_ids, stream)
                .unwrap();
        assert_eq!(restored.offset(), 4);
        for layer in restored.layers {
            assert_eq!(layer.kv.offset(), 4);
            assert!(layer
                .convolutions
                .iter()
                .all(|state| state.offset == 4 && state.state.is_some()));
        }
    }

    #[test]
    fn gguf_names_translate_to_text_parameter_tree() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.attn_rel_proj.weight"),
            "model.layers.4.self_attn.rel_proj"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.ffn_gate_exps.scales"),
            "model.layers.4.moe.experts.gate_proj_scales"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.4.ffn_gscale"),
            "model.layers.4.global_scale"
        );
        for source in [
            "blk.4.exp_probs_b.bias",
            "blk.4.ffn_exp_probs_b.bias",
            "blk.4.ffn_exp_probs_b",
        ] {
            assert_eq!(
                super::translate_gguf_weight_name(source),
                "model.layers.4.moe.router.bias"
            );
        }
    }

    #[test]
    fn packed_gguf_expert_pairs_may_omit_affine_companions() {
        let gate = 1u8;
        let up = 2u8;
        assert!(
            super::paired_inkling_gguf_component(Some(&gate), Some(&up), true, "blk.2")
                .unwrap()
                .is_some()
        );
        assert!(
            super::paired_inkling_gguf_component::<u8>(None, None, false, "blk.2")
                .unwrap()
                .is_none()
        );
        assert!(super::paired_inkling_gguf_component::<u8>(None, None, true, "blk.2").is_err());
        assert!(super::paired_inkling_gguf_component(Some(&gate), None, false, "blk.2").is_err());
    }

    #[test]
    fn mmproj_names_translate_to_native_media_towers() {
        for (source, expected) in [
            ("a.dmel.embedding.weight", "audio.encoder.weight"),
            ("a.dmel.embedding.scales", "audio.encoder.scales"),
            ("a.dmel.final_norm.weight", "audio.final_norm.weight"),
            (
                "v.hmlp.2.linear.weight",
                "visual.layers.2.projection.weight",
            ),
            ("v.hmlp.2.norm.weight", "visual.layers.2.layer_norm.weight"),
            ("v.hmlp.final_norm.weight", "visual.final_norm.weight"),
        ] {
            assert_eq!(super::translate_mmproj_weight_name(source), expected);
        }
    }

    fn tiny_gguf_metadata() -> HashMap<String, GgufMetadataValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("inkling".into()),
            ),
            ("inkling.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "inkling.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.feed_forward_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "inkling.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 1])),
            ),
            (
                "inkling.attention.key_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, false])),
            ),
            (
                "inkling.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "inkling.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            ("inkling.vocab_size".into(), GgufMetadataValue::Uint32(64)),
            ("inkling.d_rel".into(), GgufMetadataValue::Uint32(4)),
            ("inkling.rel_extent".into(), GgufMetadataValue::Uint32(16)),
            (
                "inkling.shortconv_kernel".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.dense_block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            ("inkling.expert_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "inkling.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "inkling.expert_shared_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "inkling.audio_token_id".into(),
                GgufMetadataValue::Uint32(62),
            ),
            (
                "inkling.image_token_id".into(),
                GgufMetadataValue::Uint32(63),
            ),
        ])
    }

    fn tiny_mmproj_metadata() -> HashMap<String, GgufMetadataValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("clip".into()),
            ),
            (
                "clip.has_vision_encoder".into(),
                GgufMetadataValue::Bool(true),
            ),
            (
                "clip.has_audio_encoder".into(),
                GgufMetadataValue::Bool(true),
            ),
            (
                "clip.vision.projector_type".into(),
                GgufMetadataValue::String("inkling".into()),
            ),
            (
                "clip.audio.projector_type".into(),
                GgufMetadataValue::String("inkling".into()),
            ),
            (
                "clip.vision.projection_dim".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "clip.vision.image_size".into(),
                GgufMetadataValue::Uint32(40),
            ),
            (
                "clip.vision.patch_size".into(),
                GgufMetadataValue::Uint32(40),
            ),
            (
                "clip.vision.embedding_length".into(),
                GgufMetadataValue::Uint32(3),
            ),
            (
                "clip.vision.block_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "clip.audio.projection_dim".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "clip.audio.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "clip.audio.num_mel_bins".into(),
                GgufMetadataValue::Uint32(80),
            ),
            (
                "clip.audio.attention.layer_norm_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
        ])
    }

    #[test]
    fn mmproj_metadata_and_quantization_build_native_tower_args() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let main_arrays = HashMap::from([(
            "token_embd.weight".into(),
            safemlx::Array::zeros::<f32>(&[64, 32], stream).unwrap(),
        )]);
        let mmproj_arrays = HashMap::from([
            (
                "a.dmel.embedding.weight".into(),
                safemlx::Array::zeros::<f32>(&[1280, 32], stream).unwrap(),
            ),
            (
                "a.dmel.final_norm.weight".into(),
                safemlx::Array::ones::<f32>(&[32], stream).unwrap(),
            ),
            (
                "v.hmlp.0.linear.weight".into(),
                safemlx::Array::zeros::<f32>(&[32, 32], stream).unwrap(),
            ),
            (
                "v.hmlp.0.norm.weight".into(),
                safemlx::Array::ones::<f32>(&[32], stream).unwrap(),
            ),
        ]);
        let mmproj_fixture = crate::test_utils::SyntheticGguf::with_packed_tensors(
            &mmproj_arrays,
            &tiny_mmproj_metadata(),
            |name, _| {
                matches!(name, "a.dmel.embedding.weight" | "v.hmlp.0.linear.weight")
                    .then_some(safemlx_gguf::GgmlType::Q4_0)
            },
        );
        let main_fixture =
            crate::test_utils::SyntheticGguf::dense(&main_arrays, &tiny_gguf_metadata());
        let checkpoint = GgufCheckpoint::open(main_fixture.path()).unwrap();
        let mmproj_checkpoint = GgufCheckpoint::open(mmproj_fixture.path()).unwrap();
        let mmproj = super::InklingMmprojGguf {
            metadata: tiny_mmproj_metadata(),
            checkpoint: mmproj_checkpoint,
        };
        let mut args = super::args_from_gguf_catalog(&tiny_gguf_metadata()).unwrap();
        super::apply_mmproj_args(&mut args, &tiny_gguf_metadata(), &mmproj).unwrap();
        assert_eq!(args.audio_token_id, 62);
        assert_eq!(args.image_token_id, 63);
        let audio = args.audio_config.unwrap();
        let vision = args.vision_config.unwrap();
        assert_eq!(audio.num_codebooks, 80);
        assert_eq!(audio.codebook_size, 16);
        assert_eq!(vision.patch_size, 40);
        assert_eq!(vision.temporal_patch_size, 2);
        let audio_quantization = audio
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get("audio.encoder.weight"))
            .copied()
            .unwrap();
        let vision_quantization = vision
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get("visual.layers.0.projection.weight"))
            .copied()
            .unwrap();
        let mut audio_model = super::AudioModel::new(&audio, Dtype::Float32, stream).unwrap();
        let mut vision_layer = super::VisionLayer::new(
            (32, 32, 1, 1),
            true,
            1e-6,
            Some(vision_quantization),
            Dtype::Float32,
            stream,
        )
        .unwrap();
        let audio_parameters = audio_model.parameters().flatten();
        let vision_parameters = vision_layer.parameters().flatten();
        assert!(audio_parameters.contains_key("encoder.inner.weight"));
        assert!(audio_parameters.contains_key("encoder.scales"));
        assert!(vision_parameters.contains_key("projection.inner.weight"));
        assert!(vision_parameters.contains_key("projection.scales"));
        assert_eq!(
            audio.weight_quantization_for("audio.encoder.weight"),
            Some(audio_quantization)
        );
        let config = crate::runtime::checkpoint::load::StrictLoadConfig::default();
        let mut audio_report = crate::runtime::checkpoint::load::StrictLoadReport::default();
        let mut vision_report = crate::runtime::checkpoint::load::StrictLoadReport::default();
        let mut materializer = mmproj.checkpoint.materializer();
        for tensor in mmproj.checkpoint.catalog().tensors() {
            let physical = &tensor.descriptor().name;
            for (name, value) in materializer
                .converted_tensor(physical)
                .unwrap()
                .into_arrays()
            {
                let translated = super::translate_mmproj_weight_name(&name);
                if let Some(name) = translated.strip_prefix("audio.") {
                    crate::runtime::checkpoint::load::load_named_array_strict(
                        &mut audio_model,
                        name.into(),
                        value,
                        None,
                        &config,
                        &mut audio_report,
                    )
                    .unwrap();
                } else if let Some(name) = translated.strip_prefix("visual.layers.0.") {
                    crate::runtime::checkpoint::load::load_named_array_strict(
                        &mut vision_layer,
                        name.into(),
                        value,
                        None,
                        &config,
                        &mut vision_report,
                    )
                    .unwrap();
                }
            }
        }
        audio_report.finish(&audio_model, &config).unwrap();
        vision_report.finish(&vision_layer, &config).unwrap();

        let store = crate::architectures::inkling::layerwise::inkling_gguf_store(
            &checkpoint,
            Some(&mmproj),
            2,
        )
        .unwrap();
        let keys = store.keys();
        for expected in [
            "model.embed_tokens.weight",
            "audio.encoder.weight",
            "visual.layers.0.projection.weight",
        ] {
            assert!(keys.iter().any(|key| key == expected), "missing {expected}");
        }
    }

    #[test]
    fn draft_gguf_metadata_builds_text_config() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("inkling".into()),
            ),
            ("inkling.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "inkling.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.feed_forward_length".into(),
                GgufMetadataValue::Uint32(64),
            ),
            (
                "inkling.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "inkling.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 1])),
            ),
            (
                "inkling.attention.key_length".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(8),
            ),
            (
                "inkling.attention.sliding_window_pattern".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, false])),
            ),
            (
                "inkling.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "inkling.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            ("inkling.vocab_size".into(), GgufMetadataValue::Uint32(64)),
            ("inkling.d_rel".into(), GgufMetadataValue::Uint32(4)),
            ("inkling.rel_extent".into(), GgufMetadataValue::Uint32(16)),
            (
                "inkling.shortconv_kernel".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "inkling.dense_block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            ("inkling.expert_count".into(), GgufMetadataValue::Uint32(4)),
            (
                "inkling.expert_used_count".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "inkling.expert_shared_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
        ]);
        let args = super::args_from_gguf_catalog(&metadata).unwrap();
        super::validate_args(&args).unwrap();
        assert_eq!(
            args.text_config
                .layer_schedule
                .iter()
                .map(|policy| policy.attention.window().is_some())
                .collect::<Vec<_>>(),
            vec![true, false]
        );
        assert_eq!(args.text_config.swa_num_key_value_heads, Some(2));
        assert_eq!(args.text_config.num_key_value_heads, 1);
        assert_eq!(
            args.text_config
                .layer_schedule
                .iter()
                .map(|policy| policy.feed_forward)
                .collect::<Vec<_>>(),
            vec![
                super::FeedForwardPolicy::Dense,
                super::FeedForwardPolicy::SparseMoe
            ]
        );
        assert!(args.audio_config.is_none());
        assert!(args.vision_config.is_none());
    }

    #[test]
    fn gguf_layer_schedule_rejects_invalid_pattern_and_geometry_metadata() {
        let mut length = tiny_gguf_metadata();
        length.insert(
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true])),
        );
        assert!(super::args_from_gguf_catalog(&length).is_err());

        let mut encoding = tiny_gguf_metadata();
        encoding.insert(
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![1, 0])),
        );
        assert!(super::args_from_gguf_catalog(&encoding).is_err());

        for window in [0, i32::MAX as u32 + 1] {
            let mut metadata = tiny_gguf_metadata();
            metadata.insert(
                "inkling.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(window),
            );
            assert!(super::args_from_gguf_catalog(&metadata).is_err());
        }

        let mut dense = tiny_gguf_metadata();
        dense.insert(
            "inkling.dense_block_count".into(),
            GgufMetadataValue::Uint32(3),
        );
        assert!(super::args_from_gguf_catalog(&dense).is_err());

        let mut kv_geometry = tiny_gguf_metadata();
        kv_geometry.insert("inkling.block_count".into(), GgufMetadataValue::Uint32(3));
        kv_geometry.insert(
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, true, false])),
        );
        kv_geometry.insert(
            "inkling.attention.head_count_kv".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 3, 1])),
        );
        assert!(super::args_from_gguf_catalog(&kv_geometry).is_err());
    }

    fn schedule_config() -> serde_json::Value {
        json!({
            "model_type":"inkling_mm_model",
            "text_config":{
                "hidden_size":32,"num_hidden_layers":4,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":1,"swa_head_dim":8,
                "sliding_window_size":8,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            }
        })
    }

    #[test]
    fn hf_layer_schedule_normalizes_arbitrary_attention_and_feed_forward_order() {
        let mut config = schedule_config();
        config["text_config"]["layer_types"] = json!([
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ]);
        config["text_config"]["mlp_layer_types"] = json!(["dense", "moe", "dense", "moe"]);
        let args = super::model_args_from_config_value(&config).unwrap();
        assert_eq!(
            args.text_config
                .layer_schedule
                .iter()
                .map(|policy| (policy.attention.window().is_some(), policy.feed_forward))
                .collect::<Vec<_>>(),
            vec![
                (true, super::FeedForwardPolicy::Dense),
                (false, super::FeedForwardPolicy::SparseMoe),
                (true, super::FeedForwardPolicy::Dense),
                (false, super::FeedForwardPolicy::SparseMoe),
            ]
        );
    }

    #[test]
    fn hf_layer_schedule_accepts_all_full_and_all_sliding_attention() {
        for (kind, sliding) in [("full_attention", false), ("sliding_attention", true)] {
            let mut config = schedule_config();
            config["text_config"]["layer_types"] = json!([kind, kind, kind, kind]);
            let args = super::model_args_from_config_value(&config).unwrap();
            assert!(
                args.text_config.layer_schedule.iter().all(|policy| policy
                    .attention
                    .window()
                    .is_some()
                    == sliding),
                "{kind}"
            );
        }
    }

    #[test]
    fn hf_layer_schedule_preserves_published_fallbacks_and_rejects_conflicts() {
        let mut fallback = schedule_config();
        fallback["text_config"]["num_hidden_layers"] = json!(6);
        let args = super::model_args_from_config_value(&fallback).unwrap();
        assert_eq!(
            args.text_config
                .layer_schedule
                .iter()
                .map(|policy| policy.attention.window().is_some())
                .collect::<Vec<_>>(),
            vec![true, true, true, true, true, false]
        );
        assert!(args
            .text_config
            .layer_schedule
            .iter()
            .all(|policy| policy.feed_forward == super::FeedForwardPolicy::SparseMoe));

        let mut attention_conflict = schedule_config();
        attention_conflict["text_config"]["local_layer_ids"] = json!([0]);
        attention_conflict["text_config"]["layer_types"] = json!([
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ]);
        assert!(super::model_args_from_config_value(&attention_conflict).is_err());

        let mut feed_forward_conflict = schedule_config();
        feed_forward_conflict["text_config"]["dense_mlp_idx"] = json!(1);
        feed_forward_conflict["text_config"]["mlp_layer_types"] =
            json!(["moe", "moe", "moe", "moe"]);
        assert!(super::model_args_from_config_value(&feed_forward_conflict).is_err());
    }

    #[test]
    fn hf_layer_schedule_rejects_invalid_lengths_entries_and_windows() {
        let mut length = schedule_config();
        length["text_config"]["layer_types"] = json!(["full_attention"]);
        assert!(super::model_args_from_config_value(&length).is_err());

        let mut entry = schedule_config();
        entry["text_config"]["mlp_layer_types"] = json!(["dense", "unsupported", "moe", "moe"]);
        assert!(super::model_args_from_config_value(&entry).is_err());

        for window in [json!(0), json!(-1), json!(i64::from(i32::MAX) + 1)] {
            let mut config = schedule_config();
            config["text_config"]["sliding_window_size"] = window;
            assert!(super::model_args_from_config_value(&config).is_err());
        }
    }

    #[test]
    fn fingerprint_changes_when_only_the_ordered_inkling_schedule_changes() {
        let mut first = schedule_config();
        first["text_config"]["layer_types"] = json!([
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ]);
        let mut second = schedule_config();
        second["text_config"]["layer_types"] = json!([
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention"
        ]);
        let mut different_feed_forward = first.clone();
        different_feed_forward["text_config"]["mlp_layer_types"] =
            json!(["dense", "moe", "dense", "moe"]);
        let first = super::model_args_from_config_value(&first).unwrap();
        let second = super::model_args_from_config_value(&second).unwrap();
        let different_feed_forward =
            super::model_args_from_config_value(&different_feed_forward).unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&first),
            super::prompt_cache_architecture_fingerprint(&second)
        );
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&first),
            super::prompt_cache_architecture_fingerprint(&different_feed_forward)
        );
    }

    #[test]
    fn released_config_shape_is_accepted() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "eos_token_id":200006,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            }
        });
        super::validate_model_config_value(&config).unwrap();
        let support = crate::api::resolve_model_config(&config);
        let Ok(support) = support else {
            panic!("released Inkling metadata did not dispatch")
        };
        assert_eq!(support.kind, crate::api::ModelKind::Inkling);
        assert_eq!(support.effective_model_type, "inkling_mm_model");
    }

    #[test]
    fn vision_specs_match_released_small_and_large_towers() {
        let mut args: super::VisionArgs = serde_json::from_value(json!({
            "decoder_dmodel": 4096,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4
        }))
        .unwrap();
        assert_eq!(
            args.layer_specs(),
            [
                (75, 128, 1, 5),
                (512, 320, 1, 2),
                (5120, 4800, 1, 4),
                (9600, 4096, 2, 1),
            ]
        );

        args.text_hidden_size = 6144;
        assert_eq!(
            args.layer_specs(),
            [
                (75, 128, 1, 5),
                (512, 512, 1, 2),
                (8192, 4800, 1, 4),
                (9600, 6144, 2, 1),
            ]
        );
    }

    #[test]
    fn cache_schedule_matches_local_and_global_layers() {
        let config = json!({
            "model_type":"inkling_mm_model",
            "image_token_id":200054,
            "audio_token_id":200053,
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            },
            "audio_config":{
                "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
            },
            "vision_config":{
                "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                "n_channels":3,"n_layers":4
            }
        });
        let args = super::model_args_from_config_value(&config).unwrap();
        let cache = super::Cache::new(&args.text_config);
        assert_eq!(cache.layers.len(), 3);
        assert!(matches!(
            cache.layers[0].kv,
            super::InklingKvCache::Sliding(_)
        ));
        assert!(matches!(
            cache.layers[2].kv,
            super::InklingKvCache::Global(_)
        ));
    }
}
